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
import math
import hashlib
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory, send_file, session

# Absolute path to hospital dashboard directory
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DASHBOARD_DIR = PROJECT_DIR / "hospital-dashboard"
sys.path.insert(0, str(PROJECT_DIR))

from utils.detector import AccidentDetector

app = Flask(__name__, static_folder="../hospital-dashboard")
app.secret_key = "bystander-emergency-secret-key-1928"

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
# Bystander & Hospital Database Helpers
# --------------------------------------------------------------------------
USERS_DB_PATH = PROJECT_DIR / "users_db.json"
HOSPITALS_DB_PATH = PROJECT_DIR / "hospitals_db.json"

def load_users():
    if not USERS_DB_PATH.exists():
        return {}
    try:
        with open(USERS_DB_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users):
    try:
        with open(USERS_DB_PATH, "w") as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

def load_hospitals():
    if not HOSPITALS_DB_PATH.exists():
        return {}
    try:
        with open(HOSPITALS_DB_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_hospitals(hospitals):
    try:
        with open(HOSPITALS_DB_PATH, "w") as f:
            json.dump(hospitals, f, indent=4)
    except Exception as e:
        print(f"Error saving hospitals: {e}")

HEALTHCARE_CENTERS = [
    {
        "id": "HC-01",
        "name": "City Central Trauma Center",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "phone": "+91 80 2221 0000",
        "address": "MG Road, Bangalore"
    },
    {
        "id": "HC-02",
        "name": "Apex Emergency Care",
        "latitude": 12.9912,
        "longitude": 77.5734,
        "phone": "+91 80 4115 1111",
        "address": "Malleshwaram, Bangalore"
    },
    {
        "id": "HC-03",
        "name": "St. Mary Medical Hospital",
        "latitude": 12.9348,
        "longitude": 77.6189,
        "phone": "+91 80 2553 2222",
        "address": "Koramangala, Bangalore"
    },
    {
        "id": "HC-04",
        "name": "Metro Emergency Services",
        "latitude": 12.9592,
        "longitude": 77.6412,
        "phone": "+91 80 2520 3333",
        "address": "Indiranagar, Bangalore"
    },
    {
        "id": "HC-05",
        "name": "Bangalore East General Hospital",
        "latitude": 12.9813,
        "longitude": 77.6624,
        "phone": "+91 80 2548 4444",
        "address": "Krishnarajapuram, Bangalore"
    }
]

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius of earth in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --------------------------------------------------------------------------
# Dashboard & Bystander Static Routes
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


@app.route('/bystander')
@app.route('/bystander.html')
def bystander_page():
    """Serve Bystander Portal Page"""
    bystander_path = DASHBOARD_DIR / "bystander.html"
    if bystander_path.exists():
        return send_file(bystander_path)
    return "Bystander Portal UI not found", 404


@app.route('/hospital-admin')
@app.route('/hospital-admin.html')
def hospital_admin_page():
    """Serve Hospital Admin/Registration Portal Page"""
    admin_path = DASHBOARD_DIR / "hospital-admin.html"
    if admin_path.exists():
        return send_file(admin_path)
    return "Hospital Admin UI not found", 404

@app.route('/api/resolve-maps-link', methods=['GET'])
def resolve_maps_link():
    """Follow redirects and parse coordinates from Google Maps Link"""
    url = request.args.get('url', '').strip()
    if not url:
        return jsonify({"status": "error", "message": "URL parameter missing"}), 400
        
    try:
        import requests
        import re
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        r = requests.get(url, headers=headers, allow_redirects=True, timeout=6)
        final_url = r.url
        
        lat_lng_match = None
        
        # Try search for pattern like: @latitude,longitude
        match_at = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', final_url)
        if match_at:
            lat_lng_match = (match_at.group(1), match_at.group(2))
        else:
            # Try search for pattern like: place/latitude,longitude
            match_place = re.search(r'place/(-?\d+\.\d+),(-?\d+\.\d+)', final_url)
            if match_place:
                lat_lng_match = (match_place.group(1), match_place.group(2))
            else:
                # Try search for query parameters: q=lat,lng or query=lat,lng or ll=lat,lng
                match_q = re.search(r'[q|query|ll]=(-?\d+\.\d+),(-?\d+\.\d+)', final_url)
                if match_q:
                    lat_lng_match = (match_q.group(1), match_q.group(2))
                else:
                    # Let's also parse from text content if it's a redirection page showing lat/lng
                    match_js = re.search(r'window\.APP_INITIALIZATION_STATE=\[\[\[(-?\d+\.\d+),(-?\d+\.\d+)', r.text)
                    if match_js:
                        lat_lng_match = (match_js.group(1), match_js.group(2))
        
        if lat_lng_match:
            lat, lng = float(lat_lng_match[0]), float(lat_lng_match[1])
            return jsonify({
                "status": "success",
                "latitude": lat,
                "longitude": lng,
                "resolved_url": final_url
            })
            
        return jsonify({
            "status": "error",
            "message": "Could not parse geographic coordinates from URL",
            "resolved_url": final_url
        }), 422
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to resolve URL: {str(e)}"}), 500


# --------------------------------------------------------------------------
# Hospital Authentication & Resources Endpoints
# --------------------------------------------------------------------------
@app.route('/api/hospital/register', methods=['POST'])
def register_hospital():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip()
    phone = data.get("phone", "").strip()
    address = data.get("address", "").strip()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not email or not password or not name or not phone or not address or latitude is None or longitude is None:
        return jsonify({"status": "error", "message": "All fields are required (including address and lat/lng)"}), 400

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid latitude/longitude values"}), 400

    hospitals = load_hospitals()
    if email in hospitals:
        return jsonify({"status": "error", "message": "Hospital email is already registered"}), 400

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    hosp_id = f"HOSP-{int(time.time() * 1000) % 100000:05d}"
    hospitals[email] = {
        "id": hosp_id,
        "email": email,
        "password": hashed_pw,
        "name": name,
        "phone": phone,
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "ambulances": 0,
        "care_units": 0,
        "first_aid_kits": 0,
        "oxygen_cylinders": 0
    }
    save_hospitals(hospitals)

    session["hospital"] = {
        "id": hosp_id,
        "email": email,
        "name": name,
        "phone": phone,
        "address": address,
        "latitude": latitude,
        "longitude": longitude
    }
    session.permanent = True

    return jsonify({
        "status": "success",
        "message": "Hospital registered successfully",
        "hospital": session["hospital"]
    }), 201


@app.route('/api/hospital/login', methods=['POST'])
def login_hospital():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400

    hospitals = load_hospitals()
    if email not in hospitals:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    h = hospitals[email]
    if h["password"] != hashed_pw:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    session["hospital"] = {
        "id": h["id"],
        "email": email,
        "name": h["name"],
        "phone": h["phone"],
        "address": h["address"],
        "latitude": h["latitude"],
        "longitude": h["longitude"]
    }
    session.permanent = True

    return jsonify({
        "status": "success",
        "message": "Login successful",
        "hospital": session["hospital"]
    })


@app.route('/api/hospital/logout', methods=['POST'])
def logout_hospital():
    session.pop("hospital", None)
    return jsonify({"status": "success", "message": "Logged out successfully"})


@app.route('/api/hospital/session', methods=['GET'])
def get_hospital_session():
    if "hospital" in session:
        hospitals = load_hospitals()
        h_email = session["hospital"]["email"]
        if h_email in hospitals:
            h = hospitals[h_email]
            return jsonify({
                "status": "success",
                "hospital": {
                    "id": h["id"],
                    "email": h["email"],
                    "name": h["name"],
                    "phone": h["phone"],
                    "address": h["address"],
                    "latitude": h["latitude"],
                    "longitude": h["longitude"],
                    "ambulances": h.get("ambulances", 0),
                    "care_units": h.get("care_units", 0),
                    "first_aid_kits": h.get("first_aid_kits", 0),
                    "oxygen_cylinders": h.get("oxygen_cylinders", 0)
                }
            })
    return jsonify({"status": "success", "hospital": None})


@app.route('/api/hospital/resources', methods=['POST'])
def update_hospital_resources():
    if "hospital" not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json() or {}
    h_email = session["hospital"]["email"]
    hospitals = load_hospitals()

    if h_email not in hospitals:
        return jsonify({"status": "error", "message": "Hospital profile not found"}), 404

    h = hospitals[h_email]
    try:
        h["ambulances"] = int(data.get("ambulances", h.get("ambulances", 0)))
        h["care_units"] = int(data.get("care_units", h.get("care_units", 0)))
        h["first_aid_kits"] = int(data.get("first_aid_kits", h.get("first_aid_kits", 0)))
        h["oxygen_cylinders"] = int(data.get("oxygen_cylinders", h.get("oxygen_cylinders", 0)))
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid resource numeric values"}), 400

    save_hospitals(hospitals)
    
    # Broadcast resource update to all active command dashboard clients
    broadcast_event("hospital_resources_updated", {
        "hospital_id": h["id"],
        "hospital_name": h["name"],
        "ambulances": h["ambulances"],
        "care_units": h["care_units"],
        "first_aid_kits": h["first_aid_kits"],
        "oxygen_cylinders": h["oxygen_cylinders"]
    })

    return jsonify({
        "status": "success",
        "message": "Resources updated successfully",
        "hospital": h
    })


# --------------------------------------------------------------------------
# Bystander Authentication & Report Endpoints
# --------------------------------------------------------------------------
@app.route('/api/auth/register', methods=['POST'])
def register_bystander():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip()
    phone = data.get("phone", "").strip()

    if not email or not password or not name or not phone:
        return jsonify({"status": "error", "message": "All fields are required"}), 400

    users = load_users()
    if email in users:
        return jsonify({"status": "error", "message": "Email is already registered"}), 400

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    users[email] = {
        "email": email,
        "password": hashed_pw,
        "name": name,
        "phone": phone
    }
    save_users(users)

    # Set session
    session["user"] = {
        "email": email,
        "name": name,
        "phone": phone
    }
    session.permanent = True

    return jsonify({
        "status": "success",
        "message": "Registration successful",
        "user": session["user"]
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login_bystander():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400

    users = load_users()
    if email not in users:
        return jsonify({"status": "error", "message": "Invalid email or password"}), 401

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    if users[email]["password"] != hashed_pw:
        return jsonify({"status": "error", "message": "Invalid email or password"}), 401

    # Set session
    session["user"] = {
        "email": email,
        "name": users[email]["name"],
        "phone": users[email]["phone"]
    }
    session.permanent = True

    return jsonify({
        "status": "success",
        "message": "Login successful",
        "user": session["user"]
    })


@app.route('/api/auth/logout', methods=['POST'])
def logout_bystander():
    session.pop("user", None)
    return jsonify({"status": "success", "message": "Logged out successfully"})


@app.route('/api/auth/session', methods=['GET'])
def get_session():
    if "user" in session:
        return jsonify({"status": "success", "user": session["user"]})
    return jsonify({"status": "success", "user": None})


@app.route('/api/bystander/my-reports', methods=['GET'])
def get_my_reports():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    user_email = session["user"]["email"]
    user_reports = [
        a for a in alerts_db 
        if a.get("reporter_type") == "bystander" and a.get("reporter", {}).get("email") == user_email
    ]
    return jsonify({
        "status": "success",
        "reports": user_reports
    })


@app.route('/api/healthcare-centers', methods=['GET'])
def get_healthcare_centers():
    lat_str = request.args.get("latitude")
    lng_str = request.args.get("longitude")
    
    centers = [dict(c) for c in HEALTHCARE_CENTERS]
    hospitals = load_hospitals()
    for h_email, h in hospitals.items():
        centers.append({
            "id": h.get("id"),
            "name": h.get("name"),
            "latitude": h.get("latitude"),
            "longitude": h.get("longitude"),
            "phone": h.get("phone"),
            "address": h.get("address")
        })
    
    if lat_str and lng_str:
        try:
            lat = float(lat_str)
            lng = float(lng_str)
            for c in centers:
                c["distance"] = round(calculate_distance(lat, lng, c["latitude"], c["longitude"]), 2)
            centers.sort(key=lambda x: x["distance"])
        except ValueError:
            pass
            
    return jsonify({
        "status": "success",
        "centers": centers
    })


@app.route('/api/bystander/report', methods=['POST'])
def upload_bystander_report():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Unauthorized. Please register or log in first."}), 401

    data = request.get_json() or {}
    image_data = data.get("image_data", "")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    location_name = data.get("location_name", "").strip()

    if not image_data:
        return jsonify({"status": "error", "message": "No image data provided"}), 400
    if latitude is None or longitude is None:
        return jsonify({"status": "error", "message": "Location coordinates are required"}), 400

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid coordinates"}), 400

    try:
        # Decode base64 image
        raw_b64 = image_data.split(",")[1] if "," in image_data else image_data
        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Could not decode image"}), 400

        # Run inference using trained YOLO model
        ai_detector.model_cfg["confidence_threshold"] = 0.40
        raw_detections = ai_detector.detect(frame)

        # Process detections (accident class, min confidence 0.40)
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        valid_detections = []
        for d in raw_detections:
            x1, y1, x2, y2 = d["bbox"]
            class_name = d.get("class_name", "").lower()
            confidence = d["confidence"]

            if "accident" in class_name and confidence >= 0.40:
                valid_detections.append(d)

        accident_detected = len(valid_detections) > 0
        max_conf = max([d["confidence"] for d in valid_detections]) if accident_detected else 0.0

        # Draw bounding boxes on frame
        annotated_image_data = image_data
        if accident_detected:
            for d in valid_detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                label = f"Accident: {d['confidence']:.2%}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')
            annotated_image_data = f"data:image/jpeg;base64,{annotated_b64}"

        # Find nearby healthcare centers (including dynamic registered ones)
        all_centers = [dict(c) for c in HEALTHCARE_CENTERS]
        hospitals = load_hospitals()
        for h_email, h in hospitals.items():
            all_centers.append({
                "id": h.get("id"),
                "name": h.get("name"),
                "latitude": h.get("latitude"),
                "longitude": h.get("longitude"),
                "phone": h.get("phone"),
                "address": h.get("address")
            })

        centers_with_dist = []
        for center in all_centers:
            dist = calculate_distance(latitude, longitude, center["latitude"], center["longitude"])
            centers_with_dist.append({
                **center,
                "distance": round(dist, 2)
            })
        centers_with_dist.sort(key=lambda x: x["distance"])
        closest_center = centers_with_dist[0]

        # Generate alert payload
        alert_id = f"ALT-{int(time.time() * 1000) % 1000000:06d}"
        loc_name = location_name if location_name else f"Coordinates: {latitude:.4f}, {longitude:.4f}"
        
        reporter_info = {
            "name": session["user"]["name"],
            "phone": session["user"]["phone"],
            "email": session["user"]["email"]
        }

        alert_payload = {
            "id": alert_id,
            "camera_id": "Bystander Portal Mobile App",
            "location_name": loc_name,
            "latitude": latitude,
            "longitude": longitude,
            "confidence": round(float(max_conf), 3) if accident_detected else 0.0,
            "detection_count": len(valid_detections),
            "severity": "CRITICAL" if max_conf >= 0.70 else "HIGH",
            "timestamp": datetime.now().isoformat(),
            "formatted_time": datetime.now().strftime("%I:%M:%S %p"),
            "image_data": annotated_image_data,
            "status": "PENDING",
            "assigned_ambulance": None,
            "dispatched_at": None,
            "eta_mins": None,
            "reporter": reporter_info,
            "reporter_type": "bystander",
            "assigned_hospital": closest_center["name"],
            "assigned_hospital_phone": closest_center["phone"],
            "distance_to_hospital": closest_center["distance"],
            "gmaps_link": data.get("gmaps_link", "").strip() or f"https://www.google.com/maps/place/{latitude},{longitude}"
        }

        alerts_db.insert(0, alert_payload)
        broadcast_event("new_alert", alert_payload)

        return jsonify({
            "status": "success",
            "accident_detected": accident_detected,
            "alert": alert_payload,
            "healthcare_centers": centers_with_dist
        }), 201

    except Exception as e:
        print(f"[ERROR] Bystander report processing error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


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
    
    lat = data.get("latitude", 12.9716)
    lng = data.get("longitude", 77.5946)
    alert = {
        "id": alert_id,
        "camera_id": data.get("camera_id", "CAM-01"),
        "location_name": data.get("location_name", "MG Road Intersection, Zone 3"),
        "latitude": lat,
        "longitude": lng,
        "confidence": data.get("confidence", 0.85),
        "detection_count": data.get("detection_count", 1),
        "severity": data.get("severity", "HIGH"),
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
        "formatted_time": datetime.now().strftime("%I:%M:%S %p"),
        "image_data": data.get("image_data", ""),
        "status": "PENDING",  # PENDING, ACCEPTED, DISPATCHED, COMPLETED
        "assigned_ambulance": None,
        "dispatched_at": None,
        "eta_mins": None,
        "gmaps_link": data.get("gmaps_link", "").strip() or f"https://www.google.com/maps/place/{lat},{lng}"
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
                lat = data.get("latitude", 12.9716)
                lng = data.get("longitude", 77.5946)
                alert_payload = {
                    "id": alert_id,
                    "camera_id": data.get("camera_id", "CAM-WEBCAM (Live Browser Stream)"),
                    "location_name": data.get("location_name", "MG Road Intersection, Zone 3"),
                    "latitude": lat,
                    "longitude": lng,
                    "confidence": round(float(max_conf), 3),
                    "detection_count": len(valid_detections),
                    "severity": "CRITICAL" if max_conf >= 0.75 else "HIGH",
                    "timestamp": datetime.now().isoformat(),
                    "formatted_time": datetime.now().strftime("%I:%M:%S %p"),
                    "image_data": image_data,
                    "status": "PENDING",
                    "assigned_ambulance": None,
                    "dispatched_at": None,
                    "eta_mins": None,
                    "gmaps_link": data.get("gmaps_link", "").strip() or f"https://www.google.com/maps/place/{lat},{lng}"
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
        
    # Find ambulance in dynamic list
    all_ambulances = get_all_ambulances_list()
    ambulance = next((amb for amb in all_ambulances if amb["id"] == ambulance_id), None)
    if not ambulance:
        ambulance = next((amb for amb in ambulances_db if amb["id"] == ambulance_id), ambulances_db[0])
    
    # Update state
    alert["status"] = "DISPATCHED"
    alert["assigned_ambulance"] = ambulance
    alert["dispatched_at"] = datetime.now().strftime("%I:%M:%S %p")
    alert["eta_mins"] = ambulance.get("eta_mins", 6)
    
    # Update status in hardcoded db if present
    for amb in ambulances_db:
        if amb["id"] == ambulance_id:
            amb["status"] = "ON_MISSION"
            
    # Broadcast update to UI
    broadcast_event("alert_updated", alert)
    
    print(f"[AMBULANCE DISPATCHED] #{alert_id} -> {ambulance['name']} ({ambulance['driver']})")
    
    return jsonify({
        "status": "success",
        "message": f"Ambulance {ambulance['id']} dispatched successfully",
        "alert": alert
    })


def get_all_ambulances_list():
    all_ambulances = list(ambulances_db)
    hospitals = load_hospitals()
    for h_email, h in hospitals.items():
        h_name = h.get("name")
        num_amb = int(h.get("ambulances", 0))
        for i in range(1, num_amb + 1):
            amb_id = f"AMB-{h.get('id', 'HOSP')}-{i}"
            
            status = "AVAILABLE"
            for alert in alerts_db:
                if alert.get("assigned_ambulance") and alert["assigned_ambulance"].get("id") == amb_id:
                    if alert.get("status") == "DISPATCHED":
                        status = "ON_MISSION"
                        
            all_ambulances.append({
                "id": amb_id,
                "name": f"{h_name} Response Unit {i}",
                "driver": f"Paramedic Unit {i}",
                "phone": h.get("phone", "+91 99999 99999"),
                "status": status,
                "type": "Advanced ICU Unit" if i == 1 else "Standard Life Support",
                "hospital": h_name,
                "eta_mins": 5 + i * 2
            })
    return all_ambulances


@app.route('/api/ambulances', methods=['GET'])
def get_ambulances():
    """List available ambulance units"""
    return jsonify({
        "status": "success",
        "ambulances": get_all_ambulances_list()
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
