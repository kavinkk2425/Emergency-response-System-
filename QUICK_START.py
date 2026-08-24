#!/usr/bin/env python3
"""
Quick Start Guide - Copy this file to understand the system quickly
"""

QUICK_START = """
╔════════════════════════════════════════════════════════════════╗
║   🚨 LIVE VIDEO ACCIDENT DETECTION - QUICK START GUIDE 🚨     ║
╚════════════════════════════════════════════════════════════════╝

📦 WHAT YOU HAVE:
  ✓ Trained YOLO model (25 epochs)
  ✓ Real-time accident detection
  ✓ 3 input sources (webcam, video, IP camera)
  ✓ Video output with detections marked
  ✓ Accident frame capture
  ✓ Fully configurable system

═══════════════════════════════════════════════════════════════

🚀 BASIC USAGE (Choose one):

1️⃣  WEBCAM (Live detection)
    python live_detector.py --source 0
    
    Controls: q=quit, p=pause, s=save, r=reset

2️⃣  VIDEO FILE (Analyze recorded video)
    python live_detector.py --source "video.mp4"

3️⃣  IP CAMERA (Stream monitoring)
    python live_detector.py --source "rtsp://192.168.1.100:554/stream"

4️⃣  TEST MODE (Demo without camera)
    python test_live_detector.py

═══════════════════════════════════════════════════════════════

⚙️  CUSTOMIZE DETECTION:

Lower Sensitivity (more detections):
  python live_detector.py --source 0 --conf 0.15

Higher Sensitivity (fewer false positives):
  python live_detector.py --source 0 --conf 0.45

Faster Processing:
  Edit config.yaml:
    video:
      resize_width: 320
      resize_height: 240

═══════════════════════════════════════════════════════════════

📁 OUTPUT LOCATIONS:

Videos:
  output/detected_videos/accident_detection_*.mp4

Accident Frames:
  output/accident_frames/accident_*.jpg

Logs:
  output/logs/

═══════════════════════════════════════════════════════════════

⌨️  KEYBOARD SHORTCUTS (while running):

  Q - Quit application
  P - Pause/Resume
  S - Save current frame
  R - Reset statistics

═══════════════════════════════════════════════════════════════

🎯 COMMON SCENARIOS:

Scenario 1: Monitor highway camera
  python live_detector.py --source "rtsp://highway_camera:554" \\
    --conf 0.20 --save-video

Scenario 2: Analyze accident video
  python live_detector.py --source "accident.mp4" --conf 0.25

Scenario 3: Live webcam with alerts
  python live_detector.py --source 0 --save-video
  # Check output/ folder for results

Scenario 4: High accuracy mode
  python live_detector.py --source 0 --conf 0.50

═══════════════════════════════════════════════════════════════

🔧 CONFIGURATION:

All settings in config.yaml:
  - Model confidence threshold
  - Video resolution and FPS
  - Alert types (visual/sound)
  - Video recording options
  - Performance settings

═══════════════════════════════════════════════════════════════

📊 MODEL INFO:

  Architecture: YOLOv8 Small
  Training: 3,200+ accident images
  Classes: 1 (Accident)
  Accuracy: 45-50% on test set
  Speed: 20-25 FPS (CPU)

═══════════════════════════════════════════════════════════════

🔍 UNDERSTANDING RESULTS:

Confidence Score Interpretation:
  90-100% : Very high confidence - Likely accident
  70-90%  : High confidence - Probable accident
  50-70%  : Medium confidence - Possible accident
  30-50%  : Low confidence - May be false positive
  <30%    : Very low - Likely not an accident

═══════════════════════════════════════════════════════════════

⚡ PERFORMANCE TIPS:

  For SPEED:
    - Lower resolution (320x240)
    - Increase frame skip
    - Higher confidence threshold (0.40+)

  For ACCURACY:
    - Full resolution (640x480)
    - Lower confidence threshold (0.15-0.25)
    - Use GPU if available

═══════════════════════════════════════════════════════════════

🐛 TROUBLESHOOTING:

Problem: "No webcam detected"
  Solution: python live_detector.py --source 1  # Try device 1, 2, etc

Problem: "Model not found"
  Solution: Ensure runs/detect/train/weights/best.pt exists
  Fix: python run_analysis.py

Problem: "Too slow"
  Solution: Edit config.yaml - reduce resolution, increase frame_skip

Problem: "Too many false positives"
  Solution: python live_detector.py --conf 0.45  # Increase confidence

═══════════════════════════════════════════════════════════════

📚 DOCUMENTATION:

Full Guide:
  LIVE_DETECTOR_README.md

Implementation Details:
  IMPLEMENTATION_SUMMARY.md

Source Code:
  live_detector.py - Main script
  utils/detector.py - Detection engine
  utils/video_processor.py - Video handling
  utils/visualizer.py - Output & alerts

═══════════════════════════════════════════════════════════════

✅ NEXT STEPS:

1. Try webcam: python live_detector.py --source 0
2. Check output: open output/detected_videos/
3. Read LIVE_DETECTOR_README.md for advanced features
4. Edit config.yaml to customize
5. Deploy to production!

═══════════════════════════════════════════════════════════════

🌟 KEY FEATURES:

✓ Real-time video analysis
✓ Multiple input sources (webcam/file/stream)
✓ Trained accident detection model
✓ Save videos with detection boxes
✓ Capture accident frames
✓ Console and visual alerts
✓ Statistics and logging
✓ Fully configurable

═══════════════════════════════════════════════════════════════

💡 TIPS:

• Start with test: python test_live_detector.py
• Use --conf 0.25 as default starting point
• Lower --conf (0.15) = more detections, more false positives
• Higher --conf (0.45+) = fewer detections, more accurate
• Check output/ folder after each run
• Edit config.yaml for permanent changes

═══════════════════════════════════════════════════════════════

🚀 START NOW:

  python live_detector.py --source 0

That's it! Enjoy accident detection! 🚨

═══════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(QUICK_START)
    
    # Save to file
    with open("QUICK_START.txt", "w") as f:
        f.write(QUICK_START)
    
    print("\n✓ Quick start guide saved to QUICK_START.txt")
