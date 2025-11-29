#!/bin/bash
# Run MLflow UI with SQLite backend

echo "Starting MLflow UI with SQLite backend..."
echo "Access at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

mlflow ui --backend-store-uri sqlite:///mlflow.db

