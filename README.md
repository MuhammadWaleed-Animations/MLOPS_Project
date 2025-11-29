# Building Classifier with MLflow

A modern image classification system for building types, powered by PyTorch and MLflow with a beautiful web interface.

## Features

- **Transfer Learning**: Uses ResNet18 pretrained model
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Beautiful UI**: Modern, responsive web interface for predictions
- **Docker Ready**: Containerized deployment
- **Top-K Predictions**: See confidence scores for multiple predictions

## Project Structure

```
fast_building_classifier/
├── train.py              # Model training script with MLflow logging
├── model_loader.py       # Load models from MLflow or local artifacts
├── app.py               # Flask web application
├── templates/
│   └── index.html       # Web UI for predictions
├── data/                # Training data (ImageFolder format)
│   ├── Cafe/
│   ├── Electrical/
│   ├── Ground/
│   ├── Library/
│   └── NewBuilding/
├── artifacts/           # Saved models and labels
├── mlflow.db           # MLflow tracking database
├── mlruns/             # MLflow runs directory
├── requirements.txt    # Python dependencies
└── dockerfile          # Docker configuration

```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Train a ResNet18 model on your building images
- Log metrics, parameters, and artifacts to MLflow
- Save the model with proper signature and input example
- Create a SQLite database for tracking

### 3. Run the Web Application

```bash
python app.py
```

Then open your browser to `http://localhost:8000`

### 4. Make Predictions

- Drag and drop an image or click to upload
- Click "Predict Building Type"
- See top 3 predictions with confidence scores

## Docker Deployment

Build the Docker image:

```bash
docker build -t building-classifier .
```

Run the container:

```bash
docker run -p 8000:8000 building-classifier
```

## MLflow Tracking

View experiment logs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000` to view:
- Training metrics (accuracy per epoch)
- Model artifacts
- Model signatures
- Run comparisons

## Model Details

- **Architecture**: ResNet18 (pretrained on ImageNet)
- **Training**: Transfer learning with frozen base layers
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Epochs**: 5
- **Input Size**: 224x224 RGB images

## API Endpoints

### `GET /`
Serves the web UI

### `POST /predict`
Predicts building type from uploaded image

**Request**: multipart/form-data with 'file' field

**Response**:
```json
{
  "success": true,
  "predictions": [
    {
      "class": "Library",
      "confidence": 0.95
    },
    {
      "class": "Cafe",
      "confidence": 0.03
    },
    {
      "class": "Ground",
      "confidence": 0.01
    }
  ]
}
```

### `GET /health`
Health check endpoint

## Notes

- The model automatically loads from MLflow if available
- Falls back to local artifacts if MLflow loading fails
- Supports JPG, PNG, GIF, BMP, and WebP image formats
- Maximum file size: 16MB

## Development

### Adding New Building Classes

1. Add new folder to `data/` directory
2. Add images to the folder
3. Run `python train.py` to retrain

### Customizing Training

Edit `train.py` to modify:
- Number of epochs
- Learning rate
- Batch size
- Model architecture

## Requirements

- Python 3.10+
- PyTorch
- TorchVision
- Flask
- MLflow
- Pillow

## Built With

- **PyTorch**: Deep learning framework
- **MLflow**: Experiment tracking and model registry
- **Flask**: Web framework
- **ResNet18**: Pretrained model architecture
- **Beautiful CSS**: Modern, responsive UI

## License

This project is open source and available for educational purposes.

