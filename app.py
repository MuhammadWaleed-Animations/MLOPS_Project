import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from model_loader import predict, predict_top_k
import mlflow
import mlflow.tracking

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['PREDICTIONS_DB'] = 'predictions.db'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def init_db():
    """Initialize predictions database"""
    conn = sqlite3.connect(app.config['PREDICTIONS_DB'])
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(filename, predicted_class, confidence):
    """Save prediction to database"""
    conn = sqlite3.connect(app.config['PREDICTIONS_DB'])
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (filename, predicted_class, confidence)
        VALUES (?, ?, ?)
    ''', (filename, predicted_class, confidence))
    conn.commit()
    conn.close()

def get_recent_predictions(limit=10):
    """Get recent predictions from database"""
    conn = sqlite3.connect(app.config['PREDICTIONS_DB'])
    c = conn.cursor()
    c.execute('''
        SELECT filename, predicted_class, confidence, timestamp
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    results = c.fetchall()
    conn.close()
    return results

def get_prediction_stats():
    """Get prediction statistics"""
    conn = sqlite3.connect(app.config['PREDICTIONS_DB'])
    c = conn.cursor()
    
    # Total predictions
    c.execute('SELECT COUNT(*) FROM predictions')
    total = c.fetchone()[0]
    
    # Class distribution
    c.execute('''
        SELECT predicted_class, COUNT(*) as count
        FROM predictions
        GROUP BY predicted_class
        ORDER BY count DESC
    ''')
    class_dist = c.fetchall()
    
    # Average confidence
    c.execute('SELECT AVG(confidence) FROM predictions')
    avg_confidence = c.fetchone()[0] or 0
    
    conn.close()
    return {
        'total': total,
        'class_distribution': class_dist,
        'avg_confidence': avg_confidence
    }

def get_mlflow_metrics():
    """Get metrics from MLflow"""
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("fast_building_classifier")
        
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=5
            )
            
            metrics_data = []
            for run in runs:
                metrics_data.append({
                    'run_id': run.info.run_id,
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M'),
                    'metrics': run.data.metrics,
                    'params': run.data.params
                })
            return metrics_data
        return []
    except Exception as e:
        print(f"Error fetching MLflow metrics: {e}")
        return []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Home page"""
    return render_template("index.html")

@app.route("/classify")
def classify():
    """Classification page"""
    return render_template("classify.html")

@app.route("/history")
def history():
    """Prediction history page"""
    predictions = get_recent_predictions(50)
    stats = get_prediction_stats()
    return render_template("history.html", predictions=predictions, stats=stats)

@app.route("/metrics")
def metrics():
    """MLflow metrics page"""
    mlflow_data = get_mlflow_metrics()
    return render_template("metrics.html", runs=mlflow_data)

@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    """API endpoint for predictions"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed types: " + ", ".join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictions
        top_predictions = predict_top_k(filepath, k=3)
        
        # Save to database
        if top_predictions:
            save_prediction(
                filename,
                top_predictions[0]['class'],
                top_predictions[0]['confidence']
            )
        
        return jsonify({
            "success": True,
            "predictions": top_predictions,
            "filename": filename
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route("/api/stats")
def api_stats():
    """API endpoint for statistics"""
    stats = get_prediction_stats()
    return jsonify(stats)

# Initialize database on startup
init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
