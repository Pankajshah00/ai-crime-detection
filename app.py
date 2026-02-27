import os
import cv2
import time
import base64
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import numpy as np

# ================================
# Flask App Initialization
# ================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ================================
# Load YOLOv8 Model
# ================================
MODEL_PATH = "models/yolov8n.pt"  # Change if needed
model = YOLO(MODEL_PATH)
weapon_keywords = ['knife', 'gun', 'pistol', 'rifle', 'weapon', 'sword', 'scissors']

# ================================
# Global State Variables
# ================================
video_source = 0  # default webcam
cap = None
lock = threading.Lock()
running = False
prev_frame = None

# ================================
# Helper Functions
# ================================
def detect_objects(frame):
    """Run YOLO detection and return processed frame and detected weapons/people."""
    results = model(frame, conf=0.3, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results.names[cls]
        is_weapon = any(w in label.lower() for w in weapon_keywords)
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "label": label,
            "confidence": conf,
            "is_weapon": is_weapon
        })
    return detections

def motion_intensity(prev_frame, frame):
    """Estimate motion percentage between two frames."""
    if prev_frame is None:
        return 0
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion = np.count_nonzero(thresh)
    total = frame.shape[0] * frame.shape[1]
    return (motion / total) * 100

def draw_boxes(frame, detections):
    """Draw bounding boxes for all detections."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 0, 255) if det["is_weapon"] else (0, 255, 0)
        label = f"{det['label']} ({det['confidence']:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ================================
# Background Video Thread
# ================================
def video_thread():
    """Continuously capture frames, run detection, and send to client."""
    global cap, running, prev_frame
    while running:
        with lock:
            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            detections = detect_objects(frame)
            motion = motion_intensity(prev_frame, frame)
            prev_frame = frame.copy()

            # Alert events
            for det in detections:
                if det["is_weapon"]:
                    socketio.emit("alert", {
                        "type": "Weapon Detected",
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "time": time.strftime("%H:%M:%S")
                    })
            if motion > 25:
                socketio.emit("alert", {
                    "type": "High Motion Detected (Possible Fight)",
                    "motion": f"{motion:.1f}%",
                    "time": time.strftime("%H:%M:%S")
                })

            frame = draw_boxes(frame, detections)
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            socketio.emit("frame", {"image": f"data:image/jpeg;base64,{frame_b64}"})

        time.sleep(0.03)  # limit to ~30 FPS

# ================================
# Flask Routes
# ================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/switch_to_camera", methods=["POST"])
def switch_to_camera():
    global video_source, cap, running, prev_frame
    with lock:
        if cap:
            cap.release()
        video_source = 0
        cap = cv2.VideoCapture(video_source)
        prev_frame = None
        running = True
    threading.Thread(target=video_thread, daemon=True).start()
    return jsonify({"success": True, "message": "Switched to live camera"})

@app.route("/upload_video", methods=["POST"])
def upload_video():
    global video_source, cap, running, prev_frame
    if "video" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    with lock:
        if cap:
            cap.release()
        video_source = filepath
        cap = cv2.VideoCapture(video_source)
        prev_frame = None
        running = True
    threading.Thread(target=video_thread, daemon=True).start()
    return jsonify({"success": True, "message": f"Video '{filename}' uploaded."})

# ================================
# Run the App
# ================================
if __name__ == "__main__":
    print("🚀 AI Suspicious Activity Detection Started")
    print("📹 Visit: http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000)
