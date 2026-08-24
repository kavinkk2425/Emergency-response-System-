#!/usr/bin/env python3
"""
Test script for Live Accident Detection System
Demonstrates the system with static images
"""

import cv2
from pathlib import Path
from utils import VideoProcessor, AccidentDetector, Visualizer
import yaml


def test_with_images():
    """Test detection on static images"""
    
    print("=" * 70)
    print("🧪 LIVE DETECTOR SYSTEM TEST")
    print("=" * 70)
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    visualizer = Visualizer(config)
    detector = AccidentDetector(config, visualizer)
    
    # Get test images
    test_dir = Path("data/test/images")
    test_images = sorted(list(test_dir.glob("*.jpg")))[:5]  # First 5 images
    
    print(f"\n📊 Testing with {len(test_images)} images\n")
    
    # Create dummy video properties
    first_frame = cv2.imread(str(test_images[0]))
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    visualizer.initialize_video_writer(width, height, 30)
    
    print("=" * 70)
    print("Processing Images:")
    print("=" * 70)
    
    total_accidents = 0
    total_frames = 0
    
    for idx, img_path in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] {img_path.name}")
        
        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("  ❌ Failed to read image")
            continue
        
        total_frames += 1
        
        # Run detection
        detections = detector.detect(frame)
        
        # Get smoothed detections
        smoothed_detections = detector.get_smoothed_detections()
        
        # Prepare detection data
        detection_data = []
        for det in smoothed_detections:
            detection_data.append((
                det["bbox"],
                det["confidence"],
                det["class_name"]
            ))
        
        # Get statistics
        stats = detector.get_stats()
        
        # Draw frame
        drawn_frame = visualizer.draw_frame(
            frame,
            detection_data,
            30,
            stats
        )
        
        # Save to video
        visualizer.save_video_frame(drawn_frame)
        
        # Print results
        if len(smoothed_detections) > 0:
            total_accidents += 1
            print(f"  ✓ Accident Detected!")
            for i, det in enumerate(smoothed_detections, 1):
                print(f"    - Detection {i}: {det['class_name']} ({det['confidence']:.1%})")
                visualizer.save_frame(frame, det)
        else:
            print(f"  - No accident detected")
    
    # Release video writer
    visualizer.release_video_writer()
    visualizer.close_window()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Total Frames Processed: {total_frames}")
    print(f"Accident Detections: {total_accidents}")
    print(f"Detection Rate: {(total_accidents/total_frames*100):.1f}%")
    
    stats = detector.get_stats()
    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Test completed!")
    print("\n📁 Output saved to:")
    print("  • output/detected_videos/ - Video with all detections")
    print("  • output/accident_frames/ - Individual accident frames")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("""
1. Review the output video: output/detected_videos/
2. Check accident frames: output/accident_frames/
3. Run live detection:
   
   # Webcam
   python live_detector.py --source 0
   
   # Video file
   python live_detector.py --source "video.mp4"
   
   # Custom confidence
   python live_detector.py --source 0 --conf 0.35
""")
    print("=" * 70)


if __name__ == "__main__":
    test_with_images()
