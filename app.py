import os
import cv2
import time
import base64
import tempfile
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from detector import Detector
from tracker import Tracker
from crime_detector import CrimeDetector
from loitering import LoiterMonitor
from alert import AlertManager

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'changeme')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

MODEL_PATH        = os.getenv('MODEL_PATH',        'models/yolov8s.pt')
WEAPON_MODEL_PATH = os.getenv('WEAPON_MODEL_PATH', 'models/weapon_yolov8.pt')

detector   = Detector(model_path=MODEL_PATH, weapon_model_path=WEAPON_MODEL_PATH,
                      conf=0.35, weapon_conf=0.40)
tracker    = Tracker()
crime_det  = CrimeDetector()
loiter_mon = LoiterMonitor(stationary_threshold_seconds=30)
alert_mgr  = AlertManager()

# ── SocketIO events from browser ─────────────────────────────────────────────
@socketio.on('test_email')
def handle_test_email():
    test_event = {
        'type':      'weapon_detected',
        'class':     'test',
        'track_id':  0,
        'timestamp': __import__('time').strftime('%H:%M:%S'),
    }
    alert_mgr._last_email.pop('weapon_detected', None)
    alert_mgr.handle_event(test_event)
    return {'status': 'test email queued'}

# ── Stream state ──────────────────────────────────────────────────────────────
cap        = None
cap_lock   = threading.Lock()

_current_stream_id = 0
_stream_id_lock    = threading.Lock()

_last_motion_alert = 0.0
MOTION_COOLDOWN    = 10.0


# ── Worker thread ─────────────────────────────────────────────────────────────
def video_thread(my_id: int):
    global _last_motion_alert

    def still_active():
        with _stream_id_lock:
            return _current_stream_id == my_id

    prev_frame = None

    while still_active():
        with cap_lock:
            local_cap = cap

        if local_cap is None or not local_cap.isOpened():
            time.sleep(0.05)
            continue

        ret, frame = local_cap.read()

        if not still_active():
            break

        if not ret:
            with cap_lock:
                if cap is local_cap:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue

        # ── Detection pipeline ────────────────────────────────────────────
        detections    = detector.detect(frame)
        motion        = detector.compute_motion_intensity(prev_frame, frame)
        prev_frame    = frame.copy()

        if not still_active():
            break

        tracks        = tracker.update(detections)
        crime_events  = crime_det.detect_fighting(tracks, motion)
        loiter_events = loiter_mon.update(tracks, frame)
        all_events    = crime_events + loiter_events

        # ── Emit alerts ───────────────────────────────────────────────────
        now = time.time()
        for event in all_events:
            payload = {**event,
                       'timestamp': time.strftime("%H:%M:%S"),
                       'time':      time.strftime("%H:%M:%S")}
            if 'pair' in payload and payload['pair'] is not None:
                payload['pair'] = list(payload['pair'])
            if 'bbox' in payload and payload['bbox'] is not None:
                payload['bbox'] = [float(v) for v in payload['bbox']]
            socketio.emit("alert", payload)
            alert_mgr.handle_event(event, frame=frame)

        if motion > 25 and not crime_events:
            if now - _last_motion_alert >= MOTION_COOLDOWN:
                _last_motion_alert = now
                socketio.emit("alert", {
                    "type":      "high_motion",
                    "motion":    f"{motion:.1f}%",
                    "timestamp": time.strftime("%H:%M:%S"),
                })

        # ── Stats ─────────────────────────────────────────────────────────
        socketio.emit("stats", {
            "persons": sum(1 for d in detections if d.get('is_person')),
            "weapons": sum(1 for d in detections if d.get('is_weapon')),
            "motion":  round(motion, 1),
        })

        # ── Annotate & stream ─────────────────────────────────────────────
        annotated = tracker.draw_tracks(frame.copy(), tracks)

        h, w = annotated.shape[:2]
        bar_w = int(w * min(motion, 100) / 100)
        cv2.rectangle(annotated, (0, h - 14), (bar_w, h), (0, 165, 255), -1)
        cv2.putText(annotated, f"Motion: {motion:.1f}%", (4, h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if w > 960:
            annotated = cv2.resize(annotated, (960, int(h * 960 / w)),
                                   interpolation=cv2.INTER_LINEAR)

        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
        socketio.emit("frame", {
            "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        })

        for _ in range(3):
            if not still_active():
                break
            time.sleep(0.01)


# ── Stream lifecycle ──────────────────────────────────────────────────────────
def _stop_stream():
    global cap
    with _stream_id_lock:
        globals()['_current_stream_id'] = 0

    with cap_lock:
        if cap:
            cap.release()
            cap = None

    socketio.emit("stopped", {})


def _launch_stream(new_cap):
    global cap, _current_stream_id
    with cap_lock:
        cap = new_cap

    with _stream_id_lock:
        _current_stream_id += 1
        my_id = _current_stream_id

    t = threading.Thread(target=video_thread, args=(my_id,), daemon=True)
    t.start()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/switch_to_camera", methods=["POST"])
def switch_to_camera():
    cam_id = int(request.json.get('camera_id', 0)) if request.is_json else 0
    new_cap = cv2.VideoCapture(cam_id)
    if not new_cap.isOpened():
        new_cap.release()
        return jsonify({"success": False, "message": f"Cannot open camera {cam_id}"}), 400

    _stop_stream()
    _launch_stream(new_cap)
    return jsonify({"success": True, "message": f"Streaming from camera {cam_id}"})


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"success": False, "message": "Empty filename"}), 400

    # Use tempfile — works on Railway and any cloud (filesystem may be read-only)
    suffix = os.path.splitext(secure_filename(file.filename))[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    tmp.close()

    new_cap = cv2.VideoCapture(tmp.name)
    if not new_cap.isOpened():
        new_cap.release()
        os.unlink(tmp.name)
        return jsonify({"success": False, "message": "Cannot open video file"}), 400

    _stop_stream()
    _launch_stream(new_cap)
    return jsonify({"success": True, "message": f"Streaming '{file.filename}'"})


@app.route("/stop_stream", methods=["POST"])
def stop_route():
    _stop_stream()
    return jsonify({"success": True, "message": "Stream stopped"})


@app.route("/status")
def status():
    with cap_lock:
        opened = cap is not None and cap.isOpened()
    with _stream_id_lock:
        active = _current_stream_id > 0
    return jsonify({"streaming": opened and active, "model": MODEL_PATH,
                    "email_configured": alert_mgr._email_ready})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    print(f"🚀  AI Suspicious Activity Detection  →  http://127.0.0.1:{port}")

    # Only auto-start if VIDEO_SOURCE is a file path (not webcam index 0)
    # Railway has no camera — skip webcam auto-start on cloud
    vs = os.getenv('VIDEO_SOURCE', '')
    if vs and vs != '0':
        src = int(vs) if vs.isdigit() else vs
        auto_cap = cv2.VideoCapture(src)
        if auto_cap.isOpened():
            print(f"📹  Auto-started source: {vs}")
            _launch_stream(auto_cap)
        else:
            auto_cap.release()
            print(f"⚠️  Could not open VIDEO_SOURCE={vs}")
    else:
        print("📹  No auto-start — upload a video to begin")

    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)