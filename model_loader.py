import os
import json
import torch
import mlflow
import mlflow.pytorch
from torchvision import transforms
from PIL import Image

# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "fast_building_classifier"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model_from_mlflow():
    """Load the latest model from MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Get the latest run from the experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Please train the model first.")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No runs found in the experiment. Please train the model first.")
    
    run_id = runs[0].info.run_id
    
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/mlflow_model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    
    # Load labels from artifacts
    artifact_path = client.download_artifacts(run_id, "labels.json")
    with open(artifact_path, "r") as f:
        classes = json.load(f)
    
    return model, classes

def load_model_from_artifacts():
    """Fallback: Load model from local artifacts directory"""
    from torchvision import models
    
    MODEL_PATH = "artifacts/model.pth"
    LABELS_PATH = "artifacts/labels.json"
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("Model artifacts not found. Please train the model first.")
    
    classes = json.load(open(LABELS_PATH))
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Unfreeze layer4 to match training configuration
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    model.fc = torch.nn.Linear(512, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, classes

# Try to load from MLflow, fallback to artifacts
try:
    model, classes = load_model_from_mlflow()
    print("Model loaded from MLflow")
except Exception as e:
    print(f"Failed to load from MLflow: {e}")
    print("Loading from local artifacts...")
    model, classes = load_model_from_artifacts()

def predict(image_path):
    """Predict the class of an image"""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).float()  # Ensure float32 type
    with torch.no_grad():
        out = model(tensor)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    idx = out.argmax(1).item()
    confidence = probabilities[idx].item()
    return classes[idx], confidence

def predict_top_k(image_path, k=3):
    """Predict top k classes for an image"""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).float()  # Ensure float32 type
    with torch.no_grad():
        out = model(tensor)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class": classes[idx.item()],
            "confidence": prob.item()
        })
    return results
