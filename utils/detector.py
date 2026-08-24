"""
Detector Module - YOLO model inference and detection processing
"""

from ultralytics import YOLO
from collections import deque
import numpy as np


class AccidentDetector:
    """Handle YOLO model inference and detection"""
    
    def __init__(self, config, visualizer=None):
        """
        Initialize accident detector
        
        Args:
            config: Configuration dictionary
            visualizer: Visualizer instance for alerts
        """
        self.config = config
        self.visualizer = visualizer
        self.model_cfg = config.get("model", {})
        self.advanced_cfg = config.get("advanced", {})
        
        # Load model
        self.model = self._load_model()
        
        # Detection buffering for smoothing
        self.detection_buffer = deque(
            maxlen=self.advanced_cfg.get("detection_buffer", 1)
        )
        
        # Statistics
        self.detection_count = 0
        self.frame_count = 0
        self.total_detections = 0
    
    def _load_model(self):
        """
        Load YOLO model
        
        Returns:
            Loaded YOLO model
        """
        model_path = self.model_cfg.get("weights", "yolov8s.pt")
        device = self.model_cfg.get("device", "cpu")
        
        print(f"🤖 Loading YOLO model: {model_path}")
        
        try:
            model = YOLO(model_path)
            # Set device
            if device == "cuda" or device == "0":
                model.to("cuda")
                print("✓ Model loaded on GPU")
            else:
                model.to("cpu")
                print("✓ Model loaded on CPU")
            
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    def detect(self, frame):
        """
        Run accident detection on frame
        
        Args:
            frame: Input frame
            
        Returns:
            list: Detections as [(bbox, confidence, class_name), ...]
        """
        if self.model is None:
            return []
        
        self.frame_count += 1
        
        # Run inference
        conf_threshold = self.model_cfg.get("confidence_threshold", 0.25)
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False,
            imgsz=self.model_cfg.get("input_size", 640)
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # Extract detections
            for i, box in enumerate(result.boxes):
                # Get coordinates
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                
                # Get confidence
                confidence = box.conf[0].item()
                
                # Get class name
                class_id = int(box.cls[0].item())
                class_name = result.names.get(class_id, f"Class {class_id}")
                
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "class_name": class_name,
                    "class_id": class_id
                })
        
        # Add to buffer for smoothing
        self.detection_buffer.append(detections)
        
        # Update statistics
        if len(detections) > 0:
            self.detection_count += 1
            self.total_detections += len(detections)
            
            # Alert if configured
            if self.visualizer:
                for detection in detections:
                    self.visualizer.alert_console(
                        f"Accident detected with {detection['confidence']:.1%} confidence",
                        "ALERT"
                    )
                    self.visualizer.alert_sound()
        
        return detections
    
    def get_smoothed_detections(self):
        """
        Get smoothed detections using detection buffer
        
        Returns:
            list: Smoothed detections
        """
        if len(self.detection_buffer) == 0:
            return []
        
        if self.advanced_cfg.get("detection_buffer", 1) <= 1:
            return self.detection_buffer[-1] if self.detection_buffer else []
        
        # Combine detections from buffer
        all_detections = []
        for detections in self.detection_buffer:
            all_detections.extend(detections)
        
        # Filter overlapping detections (keep highest confidence)
        if all_detections:
            return self._filter_overlapping(all_detections)
        
        return []
    
    def _filter_overlapping(self, detections, iou_threshold=0.5):
        """
        Filter overlapping detections by IoU
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for filtering
            
        Returns:
            Filtered detections
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        filtered = []
        for det in sorted_dets:
            # Check overlap with already selected detections
            overlap = False
            for selected in filtered:
                iou = self._calculate_iou(det["bbox"], selected["bbox"])
                if iou > iou_threshold:
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(det)
        
        return filtered
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU)
        
        Args:
            box1: Bounding box 1 (x1, y1, x2, y2)
            box2: Bounding box 2 (x1, y1, x2, y2)
            
        Returns:
            float: IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_stats(self):
        """
        Get detection statistics
        
        Returns:
            dict: Statistics dictionary
        """
        stats = {
            "Total Frames": self.frame_count,
            "Accident Frames": self.detection_count,
            "Total Detections": self.total_detections
        }
        
        if self.frame_count > 0:
            detection_rate = (self.detection_count / self.frame_count) * 100
            stats["Detection Rate"] = f"{detection_rate:.1f}%"
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.detection_count = 0
        self.frame_count = 0
        self.total_detections = 0
        print("📊 Statistics reset")
