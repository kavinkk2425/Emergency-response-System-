"""
Video Processor Module - Handle video input from various sources
"""

import cv2
import numpy as np
from pathlib import Path


class VideoProcessor:
    """Handle video input from webcam, files, or IP cameras"""
    
    def __init__(self, config):
        """
        Initialize video processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.video_cfg = config.get("video", {})
        
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30
        self.width = None
        self.height = None
        self.frame_skip = self.video_cfg.get("frame_skip", 1)
        self.frame_count = 0
    
    def open_source(self, source=None):
        """
        Open video source
        
        Args:
            source: Video source (0=webcam, 'file.mp4', 'rtsp://...')
                   If None, uses config value
                    
        Returns:
            bool: Success status
        """
        if source is None:
            source = self.video_cfg.get("source", 0)

        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        print(f"📹 Opening video source: {source}")
        
        # Determine source type
        if isinstance(source, int):
            self.cap = self._open_webcam(source)
            source_type = f"Webcam (Device {source})"
        elif isinstance(source, str):
            self.cap = self._open_video_file(source)
            source_type = f"Video File: {Path(source).name}"
        else:
            print("❌ Invalid source type")
            return False
        
        if not self.cap or not self.cap.isOpened():
            print(f"❌ Failed to open source: {source}")
            self._print_troubleshooting(source)
            return False

        if isinstance(source, int) or (isinstance(source, str) and source.startswith("rtsp://")):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"✓ Connected to {source_type}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        if self.total_frames > 0:
            print(f"  Total Frames: {self.total_frames}")
        
        return True
    
    def _open_webcam(self, source):
        """Open webcam with fallback backends"""
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
        return cap
    
    def _open_video_file(self, source):
        """Open video file with existence check and backend fallbacks"""
        source_path = Path(source)
        
        if not source_path.exists():
            print(f"❌ File not found: {source_path.resolve()}")
            return None
        
        if not source_path.is_file():
            print(f"❌ Path is not a file: {source_path.resolve()}")
            return None
        
        backends = [
            ("FFMPEG", cv2.CAP_FFMPEG),
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("Default", None),
        ]
        
        for name, backend in backends:
            try:
                if backend is not None:
                    cap = cv2.VideoCapture(str(source_path), backend)
                else:
                    cap = cv2.VideoCapture(str(source_path))
                
                if cap.isOpened():
                    print(f"✓ Opened with backend: {name}")
                    return cap
                cap.release()
            except Exception as e:
                print(f"⚠️  Backend {name} failed: {e}")
        
        return None
    
    def _print_troubleshooting(self, source):
        """Print helpful troubleshooting tips for common capture failures"""
        print("\n🔧 Troubleshooting tips:")
        if isinstance(source, str):
            source_path = Path(source)
            if not source_path.exists():
                print(f"  • Verify the file exists: {source_path.resolve()}")
                print("  • Use an absolute path if the file is outside the project folder.")
            else:
                print("  • Install/reinstall FFmpeg so OpenCV can decode the video.")
                print("  • Ensure the file is not locked by another process.")
                print("  • Try a smaller/copy-edited MP4 with H.264 encoding.")
                print("  • Check OneDrive sync status if the file is in OneDrive.")
        else:
            print("  • Make sure the webcam is connected and not in use by another app.")
            print("  • Try device index 0 or 1 in the config.")
        print()
    
    def read_frame(self):
        """
        Read next frame from video
        
        Returns:
            tuple: (frame, success) where frame is numpy array or None
        """
        if self.cap is None or not self.cap.isOpened():
            return None, False
        
        # Handle frame skipping
        if self.frame_skip > 1:
            for _ in range(self.frame_skip - 1):
                self.cap.read()
            self.frame_count += self.frame_skip
        else:
            self.frame_count += 1
        
        ret, frame = self.cap.read()
        
        if not ret:
            return None, False
        
        # Apply frame resizing if configured
        frame = self._resize_frame(frame)
        
        self.current_frame_idx += 1
        
        return frame, True
    
    def _resize_frame(self, frame):
        """
        Resize frame based on configuration
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        resize_width = self.video_cfg.get("resize_width", 0)
        resize_height = self.video_cfg.get("resize_height", 0)
        
        if resize_width > 0 and resize_height > 0:
            frame = cv2.resize(frame, (resize_width, resize_height))
            self.width = resize_width
            self.height = resize_height
        
        return frame
    
    def get_properties(self):
        """
        Get video properties
        
        Returns:
            dict: Video properties
        """
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame_idx
        }
    
    def get_frame_progress(self):
        """
        Get progress of video playback
        
        Returns:
            float: Progress percentage (0-100)
        """
        if self.total_frames <= 0:
            return -1  # Unknown (live stream)
        
        return (self.current_frame_idx / self.total_frames) * 100
    
    def close(self):
        """Close video source"""
        if self.cap is not None:
            self.cap.release()
            print("✓ Video source closed")
    
    def extract_roi(self, frame, roi_config):
        """
        Extract Region of Interest from frame
        
        Args:
            frame: Input frame
            roi_config: ROI configuration with x_start, y_start, x_end, y_end
            
        Returns:
            Cropped frame and adjusted ROI coordinates
        """
        if not roi_config or not roi_config.get("enable"):
            return frame, None
        
        height, width = frame.shape[:2]
        
        # Convert proportions to pixels
        x_start = int(roi_config.get("x_start", 0) * width)
        y_start = int(roi_config.get("y_start", 0) * height)
        x_end = int(roi_config.get("x_end", 1.0) * width)
        y_end = int(roi_config.get("y_end", 1.0) * height)
        
        # Clamp values
        x_start = max(0, min(x_start, width - 1))
        y_start = max(0, min(y_start, height - 1))
        x_end = max(x_start + 1, min(x_end, width))
        y_end = max(y_start + 1, min(y_end, height))
        
        roi_frame = frame[y_start:y_end, x_start:x_end]
        roi_coords = {
            "x_start": x_start,
            "y_start": y_start,
            "x_end": x_end,
            "y_end": y_end
        }
        
        return roi_frame, roi_coords
    
    def adjust_bbox_to_roi(self, bbox, roi_coords):
        """
        Adjust bounding box coordinates from ROI to full frame
        
        Args:
            bbox: Bounding box from ROI (x1, y1, x2, y2)
            roi_coords: ROI coordinates
            
        Returns:
            Adjusted bounding box
        """
        if roi_coords is None:
            return bbox
        
        x1, y1, x2, y2 = bbox
        x1 += roi_coords["x_start"]
        y1 += roi_coords["y_start"]
        x2 += roi_coords["x_start"]
        y2 += roi_coords["y_start"]
        
        return (x1, y1, x2, y2)
