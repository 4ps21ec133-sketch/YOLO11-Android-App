
import cv2
import base64
import numpy as np

from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Download the ONNX model from Hugging Face
model_path = hf_hub_download(
    repo_id="srushti-5/autonomous-vehicle-model",
    filename="REPLACE_WITH_YOUR_ONNX_FILENAME"
)

# Load the ONNX model
model = YOLO(model_path)

print("AI model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    file = request.files["image"]

    # Convert uploaded image into an OpenCV frame
    image_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Run object detection
    results = model.predict(
        source=frame,
        conf=0.15,
        verbose=False
    )

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Convert output image to JPEG
    success, buffer = cv2.imencode(".jpg", annotated_frame)

    if not success:
        return jsonify({"error": "Could not encode image"}), 500

    # Convert image to Base64 for the browser
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # Collect detected class names
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "name": results[0].names[class_id],
            "confidence": round(confidence * 100, 1)
        })

    return jsonify({
        "image": image_base64,
        "detections": detections
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
