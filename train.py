import os
import json
import torch
import mlflow
import mlflow.pytorch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def load_data():
    dataset = datasets.ImageFolder(DATA_DIR, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader, dataset.classes

def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(512, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0
    correct = 0

    for imgs, labels in loader:
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        correct += (preds.argmax(1) == labels).sum().item()

    acc = correct / total
    return acc

def main():
    loader, classes = load_data()
    num_classes = len(classes)

    model = build_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fast_building_classifier")

    with mlflow.start_run():
        for epoch in range(5):
            acc = train_one_epoch(model, loader, criterion, optimizer)
            mlflow.log_metric("accuracy", acc)
            print("Epoch", epoch, "Accuracy", acc)

        # Save artifacts
        model_path = os.path.join(ARTIFACT_DIR, "model.pth")
        labels_path = os.path.join(ARTIFACT_DIR, "labels.json")

        torch.save(model.state_dict(), model_path)
        json.dump(classes, open(labels_path, "w"))

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(labels_path)

        # Create input example for model signature
        example_input = torch.randn(1, 3, 224, 224)
        
        # Log model with proper signature (using 'name' instead of deprecated 'artifact_path')
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            name="mlflow_model",
            input_example=example_input.numpy()
        )
        print("Training complete. Model registered in MLflow.")

if __name__ == "__main__":
    main()
