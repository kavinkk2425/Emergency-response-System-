"""
Hospital Emergency Response & Ambulance Dispatch Backend Server
Provides REST API endpoints and Server-Sent Events (SSE) stream for real-time alert notifications.
"""

import os
import sys
import json
import time
import queue
import base64
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory, send_file

# Absolute path to hospital dashboard directory
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DASHBOARD_DIR = PROJECT_DIR / "hospital-dashboard"
sys.path.insert(0, str(PROJECT_DIR))

from utils.detector import AccidentDetector

app = Flask(__name__, static_folder="../hospital-dashboard")

# Initialize Trained YOLO Accident Model
MODEL_WEIGHTS = PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
if not MODEL_WEIGHTS.exists():
    MODEL_WEIGHTS = PROJECT_DIR / "yolov8s.pt"

detector_config = {
    "model": {
        "weights": str(MODEL_WEIGHTS),
        "confidence_threshold": 0.25,
        "input_size": 640,
        "device": "cpu"
    },
    "advanced": {"detection_buffer": 1}
}
ai_detector = AccidentDetector(detector_config)

# In-Memory Storage
alerts_db = []
sse_subscribers = []

# Ambulance Fleet Database
ambulances_db = [
    {
        "id": "AMB-101",
        "name": "Rapid ICU Unit Alpha",
        "driver": "Driver Rajesh Kumar",
        "phone": "+91 98765-43210",
        "status": "AVAILABLE",
        "type": "Advanced Life Support (ALS)",
        "hospital": "City Central Trauma Center",
        "eta_mins": 6
    },
    {
        "id": "AMB-102",
        "name": "Trauma Care Unit Beta",
        "driver": "Driver Suresh V.",
        "phone": "+91 98765-43211",
        "status": "AVAILABLE",
        "type": "Basic Life Support (BLS)",
        "hospital": "Apex Emergency Care",
        "eta_mins": 9
    },
    {
        "id": "AMB-103",
        "name": "Express Rescue Gamma",
        "driver": "Driver Manoj S.",
        "phone": "+91 98765-43212",
        "status": "ON_MISSION",
        "type": "Mobile ICU",
        "hospital": "St. Mary Medical Hospital",
        "eta_mins": 14
    },
    {
        "id": "AMB-104",
        "name": "City Responder Delta",
        "driver": "Driver Anish P.",
        "phone": "+91 98765-43213",
        "status": "AVAILABLE",
        "type": "Rapid Response Bike/Van",
        "hospital": "Metro Emergency Services",
        "eta_mins": 4
    }
]


def broadcast_event(event_type, data):
    """Broadcast JSON payload to all connected SSE clients"""
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    dead_subscribers = []
    for q in sse_subscribers:
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead_subscribers.append(q)
    for q in dead_subscribers:
        if q in sse_subscribers:
            sse_subscribers.remove(q)


# --------------------------------------------------------------------------
# Dashboard Static Routes
# --------------------------------------------------------------------------
@app.route('/')
def index():
    """Serve main Hospital Emergency Dashboard index.html"""
    index_path = DASHBOARD_DIR / "index.html"
    if index_path.exists():
        return send_file(index_path)
    return "Hospital Dashboard UI not found", 404


@app.route('/camera')
@app.route('/camera.html')
def camera_page():
    """Serve Live Camera & AI Detector Page"""
    camera_path = DASHBOARD_DIR / "camera.html"
    if camera_path.exists():
        return send_file(camera_path)
    return "Live Camera UI not found", 404


# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all registered accident alerts"""
    return jsonify({
        "status": "success",
        "count": len(alerts_db),
        "alerts": alerts_db
    })


@app.route('/api/alerts', methods=['POST'])
def create_alert():
    """Receive accident alert from AI model"""
    data = request.get_json() or {}
    
    alert_id = f"ALT-{int(time.time() * 1000) % 1000000:06d}"
    
    alert = {
        "id": alert_id,
        "camera_id": data.get("camera_id", "CAM-01"),
        "location_name": data.get("location_name", "MG Road Intersection, Zone 3"),
        "latitude": data.get("latitude", 12.9716),
        "longitude": data.get("longitude", 77.5946),
        "confidence": data.get("confidence", 0.85),
        "detection_count": data.get("detection_count", 1),
        "severity": data.get("severity", "HIGH"),
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
        "formatted_time": datetime.now().strftime("%I:%M:%S %p"),
        "image_data": data.get("image_data", ""),
        "status": "PENDING",  # PENDING, ACCEPTED, DISPATCHED, COMPLETED
        "assigned_ambulance": None,
        "dispatched_at": None,
        "eta_mins": None
    }
    
    alerts_db.insert(0, alert)
    
    # Broadcast to web dashboard
    broadcast_event("new_alert", alert)
    
    print(f"[ALERT REGISTERED] #{alert_id} - Severity: {alert['severity']} - Location: {alert['location_name']}")
    
    return jsonify({
        "status": "success",
        "message": "Emergency alert received successfully",
        "alert": alert
    }), 201


@app.route('/api/detect_frame', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/api/detect_frame/', methods=['GET', 'POST', 'OPTIONS'])
def detect_frame():
    """
    Real-time AI Inference endpoint for Web Camera frames.
    Runs inference using trained YOLOv8 model (best.pt).
    Only dispatches alerts for REAL accidents - applies strict validation filters
    to eliminate false positives (faces, random objects, etc.)
    """
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    data = request.get_json() or {}
    image_data = data.get("image_data", "")
    # Enforce minimum confidence of 0.55 regardless of what UI sends
    conf_thresh = max(float(data.get("confidence", 0.55)), 0.55)
    auto_dispatch = data.get("auto_dispatch", True)

    if not image_data:
        return jsonify({"status": "error", "message": "No image data provided"}), 400

    try:
        raw_b64 = image_data.split(",")[1] if "," in image_data else image_data
        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Could not decode image"}), 400

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        # Run inference using trained YOLO model
        ai_detector.model_cfg["confidence_threshold"] = conf_thresh
        raw_detections = ai_detector.detect(frame)

        # ---------------------------------------------------------------
        # STRICT ACCIDENT-ONLY VALIDATION FILTERS
        # Reject any detection that does not match ALL of the following:
        #  1. Class name must contain "accident" (case-insensitive)
        #  2. Confidence must be >= 0.55
        #  3. Bounding box area must be >= 8% of total frame area
        #     (eliminates face-size boxes & small objects)
        #  4. Bounding box must NOT be nearly square and tiny
        #     (typical face bounding boxes are roughly square ~100x100px)
        #  5. Detection box width and height must each be > 15% of frame dims
        # ---------------------------------------------------------------
        MINIMUM_CONFIDENCE     = 0.55
        MINIMUM_BOX_AREA_RATIO = 0.08   # Box must cover at least 8% of frame
        MINIMUM_BOX_W_RATIO    = 0.15   # Box width must be > 15% of frame width
        MINIMUM_BOX_H_RATIO    = 0.12   # Box height must be > 12% of frame height

        valid_detections = []
        for d in raw_detections:
            x1, y1, x2, y2 = d["bbox"]
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            box_area_ratio = box_area / frame_area if frame_area > 0 else 0
            box_w_ratio    = box_w / frame_w if frame_w > 0 else 0
            box_h_ratio    = box_h / frame_h if frame_h > 0 else 0

            class_name = d.get("class_name", "").lower()
            confidence = d["confidence"]

            # Filter 1: Must be accident class only
            if "accident" not in class_name:
                print(f"[FILTER] Rejected: class '{d['class_name']}' is not Accident")
                continue

            # Filter 2: Confidence gate
            if confidence < MINIMUM_CONFIDENCE:
                print(f"[FILTER] Rejected: confidence {confidence:.2%} below threshold {MINIMUM_CONFIDENCE:.0%}")
                continue

            # Filter 3: Bounding box too small (likely a face or small object)
            if box_area_ratio < MINIMUM_BOX_AREA_RATIO:
                print(f"[FILTER] Rejected: box area {box_area_ratio:.1%} too small (min {MINIMUM_BOX_AREA_RATIO:.0%})")
                continue

            # Filter 4: Box too narrow (not wide enough to be a road accident scene)
            if box_w_ratio < MINIMUM_BOX_W_RATIO:
                print(f"[FILTER] Rejected: box width ratio {box_w_ratio:.1%} too narrow")
                continue

            # Filter 5: Box too short
            if box_h_ratio < MINIMUM_BOX_H_RATIO:
                print(f"[FILTER] Rejected: box height ratio {box_h_ratio:.1%} too short")
                continue

            print(f"[VALID] Accident: conf={confidence:.2%}, area={box_area_ratio:.1%}, w={box_w_ratio:.1%}, h={box_h_ratio:.1%}")
            valid_detections.append(d)

        accident_detected = len(valid_detections) > 0
        alert_registered = None

        # Format validated detection bounding boxes for UI canvas
        formatted_detections = []
        for d in valid_detections:
            x1, y1, x2, y2 = d["bbox"]
            formatted_detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(float(d["confidence"]), 3),
                "class_name": d["class_name"]
            })

        # If a VALIDATED REAL accident is detected, auto-dispatch emergency request
        if accident_detected and auto_dispatch:
            max_conf = max(d["confidence"] for d in valid_detections)

            # Cooldown: avoid spamming duplicate alerts within 20 seconds
            current_time = time.time()
            if (current_time - getattr(detect_frame, "last_alert_time", 0)) > 20:
                detect_frame.last_alert_time = current_time

                alert_id = f"ALT-{int(time.time() * 1000) % 1000000:06d}"
                alert_payload = {
                    "id": alert_id,
                    "camera_id": data.get("camera_id", "CAM-WEBCAM (Live Browser Stream)"),
                    "location_name": data.get("location_name", "MG Road Intersection, Zone 3"),
                    "latitude": data.get("latitude", 12.9716),
                    "longitude": data.get("longitude", 77.5946),
                    "confidence": round(float(max_conf), 3),
                    "detection_count": len(valid_detections),
                    "severity": "CRITICAL" if max_conf >= 0.75 else "HIGH",
                    "timestamp": datetime.now().isoformat(),
                    "formatted_time": datetime.now().strftime("%I:%M:%S %p"),
                    "image_data": image_data,
                    "status": "PENDING",
                    "assigned_ambulance": None,
                    "dispatched_at": None,
                    "eta_mins": None
                }

                alerts_db.insert(0, alert_payload)
                broadcast_event("new_alert", alert_payload)
                alert_registered = alert_payload
                print(f"[VALIDATED ACCIDENT ALERT] #{alert_id} | Conf: {max_conf:.2%} | Detections: {len(valid_detections)}")

        return jsonify({
            "status": "success",
            "accident_detected": accident_detected,
            "detections": formatted_detections,
            "alert": alert_registered
        })

    except Exception as e:
        print(f"[ERROR] Live frame detection error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/alerts/<alert_id>/accept', methods=['POST'])
def accept_alert(alert_id):
    """Hospital operator accepts request and dispatches ambulance"""
    data = request.get_json() or {}
    ambulance_id = data.get("ambulance_id", "AMB-101")
    
    # Find alert
    alert = next((a for a in alerts_db if a["id"] == alert_id), None)
    if not alert:
        return jsonify({"status": "error", "message": "Alert not found"}), 404
        
    # Find ambulance
    ambulance = next((amb for amb in ambulances_db if amb["id"] == ambulance_id), ambulances_db[0])
    
    # Update state
    alert["status"] = "DISPATCHED"
    alert["assigned_ambulance"] = ambulance
    alert["dispatched_at"] = datetime.now().strftime("%I:%M:%S %p")
    alert["eta_mins"] = ambulance.get("eta_mins", 6)
    
    # Update ambulance status
    ambulance["status"] = "ON_MISSION"
    
    # Broadcast update to UI
    broadcast_event("alert_updated", alert)
    
    print(f"[AMBULANCE DISPATCHED] #{alert_id} -> {ambulance['name']} ({ambulance['driver']})")
    
    return jsonify({
        "status": "success",
        "message": f"Ambulance {ambulance['id']} dispatched successfully",
        "alert": alert
    })


@app.route('/api/ambulances', methods=['GET'])
def get_ambulances():
    """List available ambulance units"""
    return jsonify({
        "status": "success",
        "ambulances": ambulances_db
    })


@app.route('/api/events', methods=['GET'])
def sse_events():
    """Server-Sent Events (SSE) stream endpoint for real-time dashboard notifications"""
    def event_stream():
        q = queue.Queue(maxsize=20)
        sse_subscribers.append(q)
        
        # Send connection confirmation event
        init_payload = f"event: connected\ndata: {json.dumps({'message': 'Connected to Hospital Emergency Stream', 'total_alerts': len(alerts_db)})}\n\n"
        yield init_payload
        
        try:
            while True:
                msg = q.get()
                yield msg
        except GeneratorExit:
            if q in sse_subscribers:
                sse_subscribers.remove(q)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/api/demo/trigger', methods=['POST'])
def trigger_demo_alert():
    """Trigger a sample simulated accident alert for demonstration"""
    sample_payload = {
        "camera_id": "CAM-04 (High-Speed Flyover)",
        "location_name": "Outer Ring Road Flyover, Junction 7",
        "latitude": 12.9279,
        "longitude": 77.6271,
        "confidence": 0.91,
        "detection_count": 2,
        "severity": "CRITICAL",
        "timestamp": datetime.now().isoformat()
    }
    
    # Mock SVG/Canvas frame if no image passed
    data = request.get_json() or {}
    if "image_data" in data:
        sample_payload["image_data"] = data["image_data"]
        
    return create_alert_internal(sample_payload)


def create_alert_internal(data):
    """Helper method for internal trigger"""
    alert_id = f"ALT-{int(time.time() * 1000) % 1000000:06d}"
    alert = {
        "id": alert_id,
        "camera_id": data.get("camera_id", "CAM-01"),
        "location_name": data.get("location_name", "MG Road Intersection"),
        "latitude": data.get("latitude", 12.9716),
        "longitude": data.get("longitude", 77.5946),
        "confidence": data.get("confidence", 0.88),
        "detection_count": data.get("detection_count", 1),
        "severity": data.get("severity", "CRITICAL"),
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
        "formatted_time": datetime.now().strftime("%I:%M:%S %p"),
        "image_data": data.get("image_data", ""),
        "status": "PENDING",
        "assigned_ambulance": None,
        "dispatched_at": None,
        "eta_mins": None
    }
    alerts_db.insert(0, alert)
    broadcast_event("new_alert", alert)
    return jsonify({"status": "success", "alert": alert}), 201


# --------------------------------------------------------------------------
# Fallback Static Asset Route (Must remain at end of routes)
# --------------------------------------------------------------------------
@app.route('/<path:filename>')
def serve_dashboard_static(filename):
    """Serve CSS, JS, and image assets for dashboard"""
    static_file = DASHBOARD_DIR / filename
    if static_file.exists():
        return send_from_directory(DASHBOARD_DIR, filename)
    return jsonify({"status": "error", "message": "Resource not found"}), 404


if __name__ == '__main__':
    print("=" * 70)
    print("HOSPITAL EMERGENCY RESPONSE SERVER")
    print("  Server URL: http://127.0.0.1:5001")
    print("  Hospital Dashboard UI: http://127.0.0.1:5001")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
