"""
Emergency Dispatch Client Module
Sends real-time accident alert payloads with camera snapshots and metadata to the Hospital Dashboard API.
"""

import cv2
import base64
import time
import requests
import threading
from datetime import datetime


class EmergencyClient:
    """Handles emergency dispatch notifications to hospital server"""
    
    def __init__(self, config=None):
        """
        Initialize emergency client
        
        Args:
            config: Dict containing configuration options
        """
        config = config or {}
        dispatch_cfg = config.get("emergency_dispatch", {})
        
        self.enabled = dispatch_cfg.get("enabled", True)
        self.server_url = dispatch_cfg.get("server_url", "http://127.0.0.1:5001")
        self.camera_id = dispatch_cfg.get("camera_id", "CAM-01 (Highway Junction)")
        self.location_name = dispatch_cfg.get("location_name", "MG Road Intersection, Zone 3")
        self.latitude = dispatch_cfg.get("latitude", 12.9716)
        self.longitude = dispatch_cfg.get("longitude", 77.5946)
        self.cooldown_seconds = dispatch_cfg.get("cooldown_seconds", 20)
        
        self.last_alert_time = 0
        self.total_alerts_sent = 0
        
        print(f"[EMERGENCY CLIENT] Initialized. Server: {self.server_url}")

    def send_alert(self, frame, detections, confidence=0.0):
        """
        Send accident alert to hospital server in background thread (non-blocking)
        
        Args:
            frame: BGR numpy image frame of the accident
            detections: List of detection tuples/boxes
            confidence: Max confidence score of detection
        """
        if not self.enabled:
            return
            
        current_time = time.time()
        # Cooldown check to avoid sending 30 alerts per second for same accident
        if (current_time - self.last_alert_time) < self.cooldown_seconds:
            return
            
        self.last_alert_time = current_time
        
        # Start async worker to not block live video processing
        thread = threading.Thread(
            target=self._send_payload_worker,
            args=(frame.copy() if frame is not None else None, detections, confidence),
            daemon=True
        )
        thread.start()

    def _send_payload_worker(self, frame, detections, confidence):
        """Background thread worker to encode image and post HTTP request"""
        try:
            image_base64 = ""
            if frame is not None:
                # Resize snapshot for optimal network payload
                h, w = frame.shape[:2]
                max_dim = 640
                if max(h, w) > max_dim:
                    scale = max_dim / float(max(h, w))
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                # Encode frame to JPEG base64 string
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            
            # Determine severity
            if confidence >= 0.75 or len(detections) > 1:
                severity = "CRITICAL"
            elif confidence >= 0.40:
                severity = "HIGH"
            else:
                severity = "MEDIUM"

            payload = {
                "camera_id": self.camera_id,
                "location_name": self.location_name,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "confidence": round(float(confidence), 3),
                "detection_count": len(detections),
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "image_data": image_base64
            }
            
            url = f"{self.server_url.rstrip('/')}/api/alerts"
            response = requests.post(url, json=payload, timeout=3.0)
            
            if response.status_code in (200, 201):
                self.total_alerts_sent += 1
                res_data = response.json()
                print(f"[EMERGENCY DISPATCH SENT] Hospital Alert ID: #{res_data.get('alert', {}).get('id', 'N/A')} | Severity: {severity}")
            else:
                print(f"[WARNING] Emergency server returned status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"[WARNING] Could not reach Hospital Server at {self.server_url}")
        except Exception as e:
            print(f"[ERROR] Sending emergency dispatch payload: {e}")
