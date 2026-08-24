"""
Visualizer Module - Drawing and alert functionality for live detection
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path


class Visualizer:
    """Handle all visualization and alert tasks"""
    
    def __init__(self, config):
        """
        Initialize visualizer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.display_cfg = config.get("display", {})
        self.alert_cfg = config.get("alerts", {})
        self.recording_cfg = config.get("recording", {})
        
        # Create output directories
        self._create_output_dirs()
        
        # Video writer
        self.video_writer = None
        self.output_video_path = None
        
    def _create_output_dirs(self):
        """Create necessary output directories"""
        if self.alert_cfg.get("save_frames"):
            frames_dir = Path(self.alert_cfg.get("frames_dir", "output/accident_frames"))
            frames_dir.mkdir(parents=True, exist_ok=True)
        
        if self.recording_cfg.get("save_video"):
            video_dir = Path(self.recording_cfg.get("output_dir", "output/detected_videos"))
            video_dir.mkdir(parents=True, exist_ok=True)
        
        log_dir = Path(self.config.get("logging", {}).get("log_dir", "output/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize_video_writer(self, frame_width, frame_height, fps):
        """
        Initialize video writer for saving output
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            fps: Frames per second
        """
        if not self.recording_cfg.get("save_video"):
            return
        
        output_dir = Path(self.recording_cfg.get("output_dir", "output/detected_videos"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_video_path = output_dir / f"accident_detection_{timestamp}.mp4"
        
        # Get codec
        codec = self.recording_cfg.get("codec", "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Adjust size for quality setting
        quality = self.recording_cfg.get("quality", 1.0)
        out_width = int(frame_width * quality)
        out_height = int(frame_height * quality)
        
        output_fps = self.recording_cfg.get("fps", fps) or fps
        
        self.video_writer = cv2.VideoWriter(
            str(self.output_video_path),
            fourcc,
            output_fps,
            (out_width, out_height)
        )
        
        print(f"📹 Video recording started: {self.output_video_path}")
    
    def draw_frame(self, frame, detections, fps, stats=None):
        """
        Draw detections and info on frame
        
        Args:
            frame: Input frame
            detections: List of (bbox, confidence, class_name) tuples
            fps: Current FPS value
            stats: Statistics dictionary
            
        Returns:
            Annotated frame
        """
        frame_copy = frame.copy()
        height, width = frame.shape[:2]
        
        # Alert color and normal color
        alert_color = tuple(self.display_cfg.get("alert_color", [0, 0, 255]))
        text_color = tuple(self.display_cfg.get("text_color", [0, 255, 0]))
        thickness = self.display_cfg.get("box_thickness", 2)
        
        has_accident = len(detections) > 0
        
        # Draw detections
        for bbox, confidence, class_name in detections:
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            color = alert_color if has_accident else text_color
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {confidence:.2%}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(
                frame_copy,
                (x1, y1 - label_size[1] - baseline),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Draw FPS counter
        if self.display_cfg.get("show_fps"):
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame_copy,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2
            )
        
        # Draw statistics
        if self.display_cfg.get("show_stats") and stats:
            y_offset = 70
            for key, value in stats.items():
                stat_text = f"{key}: {value}"
                cv2.putText(
                    frame_copy,
                    stat_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    1
                )
                y_offset += 30
        
        # Draw alert overlay if accident detected
        if has_accident and self.alert_cfg.get("enable_visual"):
            # Red border
            cv2.rectangle(frame_copy, (0, 0), (width - 1, height - 1), alert_color, 5)
            
            # Alert text
            alert_text = "🔴 ACCIDENT DETECTED!"
            font_scale = 1.5
            text_size, baseline = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
            text_x = (width - text_size[0]) // 2
            text_y = height - 20
            
            # Background for text
            cv2.rectangle(
                frame_copy,
                (text_x - 10, text_y - text_size[1] - 10),
                (text_x + text_size[0] + 10, text_y + 10),
                alert_color,
                -1
            )
            cv2.putText(
                frame_copy,
                alert_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                3
            )
        
        return frame_copy
    
    def save_frame(self, frame, detection_info):
        """
        Save a frame with accident detection
        
        Args:
            frame: Input frame
            detection_info: Detection details dictionary
        """
        if not self.alert_cfg.get("save_frames"):
            return
        
        frames_dir = Path(self.alert_cfg.get("frames_dir", "output/accident_frames"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Include confidence in filename
        confidence = detection_info.get("confidence", 0)
        filename = f"accident_{timestamp}_{confidence:.0%}.jpg"
        
        filepath = frames_dir / filename
        cv2.imwrite(str(filepath), frame)
        
        print(f"💾 Accident frame saved: {filename}")
    
    def save_video_frame(self, frame):
        """
        Write frame to output video
        
        Args:
            frame: Frame to save
        """
        if self.video_writer and self.video_writer.isOpened():
            # Resize if needed based on quality setting
            quality = self.recording_cfg.get("quality", 1.0)
            if quality < 1.0:
                height, width = frame.shape[:2]
                new_width = int(width * quality)
                new_height = int(height * quality)
                frame = cv2.resize(frame, (new_width, new_height))
            
            self.video_writer.write(frame)
    
    def release_video_writer(self):
        """Release video writer and finalize video"""
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            print(f"✓ Video saved: {self.output_video_path}")
    
    def display_frame(self, frame):
        """
        Display frame in window
        
        Args:
            frame: Frame to display
        """
        if self.display_cfg.get("show_window"):
            window_title = self.display_cfg.get("window_title", "Live Detection")
            cv2.imshow(window_title, frame)
    
    def close_window(self):
        """Close display window"""
        cv2.destroyAllWindows()
    
    def alert_console(self, message, alert_type="INFO"):
        """
        Print alert to console
        
        Args:
            message: Message to print
            alert_type: Type of alert (INFO, WARNING, ALERT)
        """
        if not self.alert_cfg.get("enable_console"):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if alert_type == "ALERT":
            print(f"🔴 [{timestamp}] ACCIDENT ALERT: {message}")
        elif alert_type == "WARNING":
            print(f"⚠️  [{timestamp}] WARNING: {message}")
        else:
            print(f"ℹ️  [{timestamp}] INFO: {message}")
    
    def alert_sound(self):
        """Play sound alert"""
        if not self.alert_cfg.get("enable_sound"):
            return
        
        try:
            import winsound
            frequency = self.alert_cfg.get("sound_frequency", 1000)
            duration = self.alert_cfg.get("sound_duration", 500)
            winsound.Beep(frequency, duration)
        except ImportError:
            # Fallback for non-Windows systems
            print("\a")  # Bell character
