#!/usr/bin/env python3
"""
Live Video Accident Detection System
Main entry point for real-time accident detection from video sources

Usage:
    # Webcam
    python live_detector.py --source 0
    
    # Video file
    python live_detector.py --source video.mp4
    
    # IP Camera
    python live_detector.py --source "rtsp://camera_ip:554/stream"
"""

import cv2
import yaml
import argparse
import time
from pathlib import Path
from utils import VideoProcessor, AccidentDetector, Visualizer, EmergencyClient


PROJECT_DIR = Path(__file__).resolve().parent


class LiveDetectionSystem:
    """Main live detection system"""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize live detection system
        
        Args:
            config_path: Path to config.yaml
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.visualizer = Visualizer(self.config)
        self.detector = AccidentDetector(self.config, self.visualizer)
        self.processor = VideoProcessor(self.config)
        self.emergency_client = EmergencyClient(self.config)
        
        # Control flags
        self.running = True
        self.paused = False
        self.frame_times = []
        self.max_fps_samples = 30
    
    def _load_config(self, config_path):
        """
        Load YAML configuration
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.is_absolute() and not config_file.exists():
            config_file = PROJECT_DIR / config_file
        
        if not config_file.exists():
            print(f"⚠️  Config file not found: {config_path}")
            print("Creating default configuration...")
            return self._get_default_config()
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            config = self._resolve_paths(config)
            print(f"✓ Loaded configuration from: {config_file}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()

    def _resolve_paths(self, config):
        """Resolve project-relative paths independently of the current folder."""
        path_settings = (
            ("model", "weights"),
            ("alerts", "frames_dir"),
            ("recording", "output_dir"),
            ("logging", "log_dir"),
        )

        for section, key in path_settings:
            value = config.get(section, {}).get(key)
            if value and not Path(value).is_absolute():
                config[section][key] = str(PROJECT_DIR / value)

        return config
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            "model": {
                "weights": "runs/detect/train/weights/best.pt",
                "confidence_threshold": 0.25,
                "input_size": 640,
                "device": "cpu"
            },
            "video": {
                "source": 0,
                "frame_skip": 1,
                "resize_width": 640,
                "resize_height": 480,
                "max_fps": 30
            },
            "display": {
                "show_window": True,
                "window_title": "🚨 Live Accident Detection System 🚨",
                "draw_boxes": True,
                "show_fps": True,
                "show_stats": True
            },
            "alerts": {
                "enable_visual": True,
                "enable_console": True,
                "enable_sound": False,
                "save_frames": True,
                "frames_dir": "output/accident_frames"
            },
            "recording": {
                "save_video": True,
                "output_dir": "output/detected_videos",
                "codec": "mp4v",
                "fps": 30,
                "quality": 1.0
            },
            "logging": {
                "enable_logging": True,
                "log_dir": "output/logs"
            },
            "advanced": {
                "detection_buffer": 3,
                "roi": {"enable": False}
            }
        }
    
    def run(self, source=None):
        """
        Run live detection system
        
        Args:
            source: Video source (overrides config)
        """
        print("\n" + "=" * 70)
        print("🚨 LIVE ACCIDENT DETECTION SYSTEM 🚨")
        print("=" * 70)
        
        # Open video source
        if not self.processor.open_source(source):
            print("❌ Failed to open video source")
            return
        
        print("\n📋 Controls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  's' - Save current frame")
        print("  'r' - Reset statistics")
        print()
        
        # Initialize video writer
        props = self.processor.get_properties()
        self.visualizer.initialize_video_writer(
            props["width"],
            props["height"],
            props["fps"]
        )
        
        # Main loop
        self._process_video()
        
        # Cleanup
        self._cleanup()
    
    def _process_video(self):
        """Main video processing loop"""
        print("▶️  Starting detection...\n")
        
        start_time = time.time()
        
        while self.running:
            # Read frame
            if not self.paused:
                frame, success = self.processor.read_frame()
                
                if not success:
                    print("\n⏹️  End of video reached")
                    break
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Get smoothed detections
                smoothed_detections = self.detector.get_smoothed_detections()
                
                # Calculate FPS
                current_time = time.time()
                self.frame_times.append(current_time)
                if len(self.frame_times) > self.max_fps_samples:
                    self.frame_times.pop(0)
                
                if len(self.frame_times) > 1:
                    fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
                else:
                    fps = 0
                
                # Prepare detection data for visualization
                detection_data = []
                for det in smoothed_detections:
                    detection_data.append((
                        det["bbox"],
                        det["confidence"],
                        det["class_name"]
                    ))
                
                # Get statistics
                stats = self.detector.get_stats()
                
                # Draw frame
                drawn_frame = self.visualizer.draw_frame(
                    frame,
                    detection_data,
                    fps,
                    stats
                )
                
                # Save to video
                self.visualizer.save_video_frame(drawn_frame)
                
                # Display frame
                self.visualizer.display_frame(drawn_frame)
                # Handle accidents
                if len(smoothed_detections) > 0:
                    for detection in smoothed_detections:
                        self.visualizer.save_frame(frame, detection)
                    
                    # Send real-time emergency alert message to Hospital Dashboard
                    max_conf = max(d["confidence"] for d in smoothed_detections)
                    self.emergency_client.send_alert(frame, smoothed_detections, confidence=max_conf)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)
            
            # Check FPS limit
            if self.config.get("video", {}).get("max_fps", 0) > 0:
                elapsed = time.time() - start_time
                expected_time = self.detector.frame_count / self.config["video"]["max_fps"]
                if elapsed < expected_time:
                    time.sleep(expected_time - elapsed)
    
    def _handle_key(self, key):
        """
        Handle keyboard input
        
        Args:
            key: Key code from cv2.waitKey()
        """
        if key == ord('q'):
            print("\n⏹️  Quitting...")
            self.running = False
        
        elif key == ord('p'):
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RESUMED"
            self.visualizer.alert_console(status, "INFO")
        
        elif key == ord('s'):
            frame, _ = self.processor.read_frame()
            if frame is not None:
                self.visualizer.save_frame(frame, {"confidence": 0})
        
        elif key == ord('r'):
            self.detector.reset_stats()
    
    def _cleanup(self):
        """Cleanup resources"""
        print("\n" + "=" * 70)
        print("🧹 Cleaning up...")
        print("=" * 70)
        
        # Release video writer
        self.visualizer.release_video_writer()
        
        # Close video source
        self.processor.close()
        
        # Close display window
        self.visualizer.close_window()
        
        # Print final statistics
        stats = self.detector.get_stats()
        print("\n📊 Final Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ System shutdown complete")
        print("=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Live Video Accident Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam detection
  python live_detector.py --source 0
  
  # Video file detection
  python live_detector.py --source video.mp4
  
  # IP Camera detection
  python live_detector.py --source "rtsp://camera_ip:554/stream"
  
  # With confidence threshold
  python live_detector.py --source 0 --conf 0.35
  
  # With custom config
  python live_detector.py --config my_config.yaml --source 0
        """
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source (0=webcam, file path, or RTSP URL)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (0-1)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output video with detections"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window"
    )
    
    args = parser.parse_args()
    
    # Create and run system
    system = LiveDetectionSystem(args.config)
    
    # Override config with command line arguments
    if args.conf is not None:
        system.config["model"]["confidence_threshold"] = args.conf
    
    if args.no_display:
        system.config["display"]["show_window"] = False
    
    if args.save_video is False:
        system.config["recording"]["save_video"] = False
    
    # Run detection
    system.run(args.source)


if __name__ == "__main__":
    main()
