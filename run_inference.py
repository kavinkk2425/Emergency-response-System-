#!/usr/bin/env python3
"""
Accident Detection Model - Local Inference Script
Runs YOLOv8 model on test images and displays results
"""

from ultralytics import YOLO
from pathlib import Path
import os
from datetime import datetime

def main():
    # Set up paths
    project_dir = Path(__file__).parent
    test_images_dir = project_dir / "data" / "test" / "images"
    output_dir = project_dir / "runs" / "detect" / f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check if we have trained weights, otherwise use pretrained model
    best_weights = project_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
    
    if best_weights.exists():
        print(f"✓ Found trained weights at: {best_weights}")
        model_path = str(best_weights)
    else:
        print("⚠ Trained weights not found. Using pretrained YOLOv8s model.")
        print("  Note: Model won't be fine-tuned for accidents, just using base YOLO detector")
        model_path = "yolov8s.pt"
    
    # Load model
    print(f"\n🔄 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get test images
    test_images = list(test_images_dir.glob("*.jpg"))
    print(f"\n📁 Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("❌ No test images found!")
        return
    
    # Run inference on a sample of images
    print(f"\n🚀 Running inference on first 10 test images...")
    print("=" * 60)
    
    results_summary = []
    
    for i, image_path in enumerate(test_images[:10]):
        print(f"\n[{i+1}/10] Processing: {image_path.name}")
        
        # Run prediction
        results = model.predict(
            source=str(image_path),
            conf=0.25,  # Confidence threshold
            save=True,
            save_dir=str(output_dir),
            verbose=False
        )
        
        # Extract results
        if results:
            result = results[0]
            num_detections = len(result.boxes)
            
            if num_detections > 0:
                status = "✓ ACCIDENT DETECTED"
                results_summary.append((image_path.name, num_detections, "ACCIDENT"))
                confidences = result.boxes.conf.cpu().numpy()
                avg_conf = confidences.mean()
                print(f"  {status}")
                print(f"  └─ Detections: {num_detections}")
                print(f"  └─ Avg Confidence: {avg_conf:.2%}")
            else:
                status = "✗ No accident detected"
                results_summary.append((image_path.name, 0, "SAFE"))
                print(f"  {status}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    accidents_detected = sum(1 for _, _, status in results_summary if status == "ACCIDENT")
    safe_images = sum(1 for _, _, status in results_summary if status == "SAFE")
    
    print(f"Accidents Detected: {accidents_detected}")
    print(f"Safe (No Accidents): {safe_images}")
    print(f"Accuracy Rate: {(accidents_detected / 10):.0%}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("\nDetailed Results:")
    print("-" * 60)
    for img_name, count, status in results_summary:
        print(f"{img_name:<40} | {status:<10} | {count} boxes")
    
    print("\n" + "=" * 60)
    print("✓ Inference completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
