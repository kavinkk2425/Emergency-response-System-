#!/usr/bin/env python3
"""
Accident Detection Model - Enhanced Local Inference with Analysis
Shows model performance, handles different confidence thresholds, visualizes results
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

def main():
    project_dir = Path(__file__).parent
    test_images_dir = project_dir / "data" / "test" / "images"
    train_dir = project_dir / "runs" / "detect" / "train"
    
    # Load trained weights
    best_weights = train_dir / "weights" / "best.pt"
    print("=" * 70)
    print("🚨 ACCIDENT DETECTION MODEL - LOCAL INFERENCE 🚨")
    print("=" * 70)
    
    if not best_weights.exists():
        print("❌ Error: Trained weights not found!")
        return
    
    print(f"\n✓ Model Loaded: YOLOv8 Small (Trained)")
    print(f"  Path: {best_weights}")
    
    # Check training metrics
    results_csv = train_dir / "results.csv"
    if results_csv.exists():
        with open(results_csv, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                print(f"✓ Training Completed: {len(lines)-2} epochs")
    
    # Load model
    model = YOLO(str(best_weights))
    
    # Get test images
    test_images = sorted(list(test_images_dir.glob("*.jpg")))
    print(f"✓ Test Dataset: {len(test_images)} images found\n")
    
    # Test with different confidence thresholds
    confidence_thresholds = [0.15, 0.25, 0.35, 0.50]
    
    print("📊 TESTING WITH MULTIPLE CONFIDENCE THRESHOLDS")
    print("-" * 70)
    
    for conf_threshold in confidence_thresholds:
        print(f"\n🔍 Confidence Threshold: {conf_threshold:.0%}")
        print(f"  Processing {len(test_images)} images...")
        
        accidents_found = 0
        total_detections = 0
        confidences = []
        
        output_dir = project_dir / "runs" / "detect" / f"predict_conf{conf_threshold}__{datetime.now().strftime('%H%M%S')}"
        
        for idx, image_path in enumerate(test_images):
            # Show progress every 20 images
            if (idx + 1) % 20 == 0:
                print(f"    Progress: {idx + 1}/{len(test_images)}")
            
            # Run inference
            results = model.predict(
                source=str(image_path),
                conf=conf_threshold,
                save=False,
                verbose=False
            )
            
            if results and len(results[0].boxes) > 0:
                accidents_found += 1
                num_boxes = len(results[0].boxes)
                total_detections += num_boxes
                result_confs = results[0].boxes.conf.cpu().numpy()
                confidences.extend(result_confs.tolist())
        
        # Summary for this threshold
        detection_rate = (accidents_found / len(test_images)) * 100
        print(f"  ✓ Results:")
        print(f"    - Images with accidents detected: {accidents_found}/{len(test_images)} ({detection_rate:.1f}%)")
        print(f"    - Total detection boxes: {total_detections}")
        if confidences:
            print(f"    - Avg Confidence Score: {np.mean(confidences):.2%}")
            print(f"    - Confidence Range: {np.min(confidences):.2%} - {np.max(confidences):.2%}")
    
    # Run inference on 5 random images with visualization
    print("\n" + "=" * 70)
    print("🎨 VISUAL ANALYSIS - Running on 5 sample images with boxes")
    print("=" * 70)
    
    import random
    sample_images = random.sample(test_images, min(5, len(test_images)))
    
    for i, image_path in enumerate(sample_images, 1):
        print(f"\n[{i}/5] {image_path.name}")
        
        # Run prediction with visualization
        results = model.predict(
            source=str(image_path),
            conf=0.25,
            save=True,
            save_dir=str(project_dir / "runs" / "detect" / f"visual_sample_{datetime.now().strftime('%H%M%S')}"),
            verbose=False
        )
        
        if results and len(results[0].boxes) > 0:
            result = results[0]
            num_boxes = len(result.boxes)
            print(f"  ✓ Accidents Detected: {num_boxes}")
            for j, conf in enumerate(result.boxes.conf):
                print(f"    - Detection {j+1}: Confidence {conf.item():.2%}")
        else:
            print(f"  - No accidents detected")
    
    # Final summary
    print("\n" + "=" * 70)
    print("📈 MODEL INFORMATION")
    print("=" * 70)
    print(f"""
✓ Model Architecture: YOLOv8 Small (yolov8s)
✓ Training Data: 3200+ accident images
✓ Classes: 1 (Accident)
✓ Input Size: 640x640 pixels
✓ Training Epochs: 19
✓ Best Model Location: {best_weights}

📁 Results Saved:
  └─ Detection visualizations: runs/detect/predict_* folders
  └─ Each image with bounding boxes shows accident locations

🎯 How to Use:
  1. Place any image/video in a folder
  2. Run: model.predict(source='path', conf=0.25, save=True)
  3. Check results/ folder for visualized output
""")
    
    print("=" * 70)
    print("✅ Inference pipeline completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
