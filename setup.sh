#!/bin/bash

# Setup script for Building Classifier

echo "Setting up Building Classifier..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p artifacts
mkdir -p uploads
mkdir -p templates

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your training images to the 'data/' folder"
echo "2. Run 'python train.py' to train the model"
echo "3. Run 'python app.py' to start the web application"
echo "4. Open http://localhost:8000 in your browser"

