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

def get_train_transforms():
    """Training transforms with data augmentation"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms():
    """Validation/inference transforms without augmentation"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_data():
    dataset = datasets.ImageFolder(DATA_DIR, transform=get_train_transforms())
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader, dataset.classes

def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze all layers initially
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze last ResNet block (layer4) for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace final classifier layer
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
    # Optimizer for both fc layer and layer4
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': 0.001},
        {'params': model.layer4.parameters(), 'lr': 0.0001}  # Lower lr for fine-tuning
    ])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fast_building_classifier")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("epochs", 15)
        mlflow.log_param("learning_rate_fc", 0.001)
        mlflow.log_param("learning_rate_layer4", 0.0001)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("architecture", "resnet18_finetuned")
        
        for epoch in range(15):
            acc = train_one_epoch(model, loader, criterion, optimizer)
            mlflow.log_metric("accuracy", acc, step=epoch)
            print(f"Epoch {epoch+1}/15 - Accuracy: {acc:.4f}")

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
