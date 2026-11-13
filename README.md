# Autonomous Vehicle Object Detection

This project uses a Flask Python backend to run an ONNX object-detection model hosted on Hugging Face. The browser sends webcam frames to Flask and displays returned bounding boxes. The original ESP32 BLE interface is retained.

## Project layout

```text
autonomous-vehicle-detection/
├── app.py
├── requirements.txt
└── templates/
    └── index.html
```

## 1. Set the model filename

Open `app.py` and replace:

`REPLACE_WITH_YOUR_EXACT_ONNX_FILENAME.onnx`

with the exact `.onnx` filename shown in:

https://huggingface.co/srushti-5/autonomous-vehicle-model/tree/main

The Hugging Face repository must be public for this unauthenticated download method.

## 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 3. Run locally

```bash
python app.py
```

Open:

http://127.0.0.1:5000

Allow camera access and press **Start Detection**. Chrome camera access works on localhost.

## Notes

- The frontend and backend are served from the same Flask origin, so no CORS configuration is needed for local testing.
- The ONNX model must be compatible with Ultralytics `YOLO(...).predict()` and must retain the correct class vocabulary/metadata.
- If the ONNX model is not an Ultralytics-compatible detection export, this backend will need model-specific preprocessing and postprocessing.
- The original BLE message protocol is preserved from the uploaded HTML. Confirm the ESP32 firmware handles `DETECT:<class>:<confidence>`, `CLEAR`, `STOP`, and `CAMERA_STOPPED` safely before connecting motors.
- This is a development/testing interface, not a safety-certified autonomous-driving controller.
