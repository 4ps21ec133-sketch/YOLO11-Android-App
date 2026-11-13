import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

HF_REPO_ID = "srushti-5/autonomous-vehicle-model"
HF_MODEL_FILENAME = os.environ.get(
    "HF_MODEL_FILENAME",
    "REPLACE_WITH_YOUR_EXACT_ONNX_FILENAME.onnx"
)

CONFIDENCE = 0.15
IMAGE_SIZE = 640

if HF_MODEL_FILENAME.startswith("REPLACE_WITH_"):
    raise RuntimeError(
        "Edit HF_MODEL_FILENAME in app.py and enter the exact ONNX filename "
        "from https://huggingface.co/srushti-5/autonomous-vehicle-model"
    )

print("Downloading ONNX model from Hugging Face...")
model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_MODEL_FILENAME,
    repo_type="model"
)

print("Loading ONNX model...")
model = YOLO(model_path, task="detect")
print("ONNX model loaded successfully.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "connected",
        "model_repository": HF_REPO_ID,
        "model_filename": HF_MODEL_FILENAME
    })


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image field was received."}), 400

    image_file = request.files["image"]
    image_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "The uploaded image could not be decoded."}), 400

    height, width = frame.shape[:2]
    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    detections = []
    result = results[0]

    if result.boxes is not None:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for coords, class_id, confidence in zip(
            boxes_xyxy, class_ids, confidences
        ):
            x1, y1, x2, y2 = [float(value) for value in coords]
            x1 = max(0.0, min(float(width), x1))
            x2 = max(0.0, min(float(width), x2))
            y1 = max(0.0, min(float(height), y1))
            y2 = max(0.0, min(float(height), y2))

            class_names = result.names
            if isinstance(class_names, dict):
                class_name = class_names.get(class_id, str(class_id))
            else:
                class_name = (
                    class_names[class_id]
                    if class_id < len(class_names)
                    else str(class_id)
                )

            detections.append({
                "name": str(class_name),
                "conf": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    return jsonify({
        "image_width": width,
        "image_height": height,
        "detections": detections
    })


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "Image is too large. Send a smaller camera frame."
    }), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
