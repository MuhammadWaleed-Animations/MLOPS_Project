from flask import Flask, request, jsonify
from model_loader import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    img = request.files["file"]
    img.save("temp.jpg")
    label = predict("temp.jpg")
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

