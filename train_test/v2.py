#!/usr/bin/env python3
"""
Quick verification script - just run this to check your annotations
"""

import os
from pathlib import Path
import random


def quick_verify_annotations(dataset_path):
    """Quick check of YOLO annotations"""

    base_path = Path(dataset_path)
    output_dir = base_path / "output"

    print("🔍 Quick YOLO Annotation Check")
    print("=" * 50)

    # Check if output directory exists
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return False

    # Get annotation files
    annotation_files = list(output_dir.glob("*.txt"))
    classes_file = output_dir / "classes.txt"

    # Remove classes.txt from annotation files
    annotation_files = [f for f in annotation_files if f.name != "classes.txt"]

    print(f"📊 Statistics:")
    print(f"   Annotation files: {len(annotation_files)}")

    if not annotation_files:
        print("❌ No annotation files found!")
        return False

    # Check classes file
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"   Classes: {len(classes)}")
        for i, cls in enumerate(classes):
            print(f"     {i}: {cls}")
    else:
        print("❌ classes.txt not found!")
        return False

    # Check sample annotation files
    print(f"\n📄 Sample Annotation Files:")
    sample_files = random.sample(annotation_files, min(5, len(annotation_files)))

    total_boxes = 0
    for i, ann_file in enumerate(sample_files):
        try:
            with open(ann_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            print(f"   {i + 1}. {ann_file.name}: {len(lines)} objects")
            total_boxes += len(lines)

            # Show first annotation
            if lines:
                parts = lines[0].split()
                if len(parts) == 5:
                    class_id, x, y, w, h = parts
                    class_name = classes[int(class_id)] if int(class_id) < len(classes) else "unknown"
                    print(f"      First object: {class_name} (class {class_id})")
                    print(f"      Bbox: center=({x}, {y}), size=({w}×{h})")
                else:
                    print(f"      ⚠️  Invalid format: {lines[0]}")
            else:
                print(f"      ⚠️  Empty file")

        except Exception as e:
            print(f"   {i + 1}. {ann_file.name}: Error - {e}")

    print(f"\n📦 Total bounding boxes in sample: {total_boxes}")

    # Basic validation
    print(f"\n✅ Quick Validation:")

    # Check a few files for proper format
    valid_count = 0
    for ann_file in sample_files:
        try:
            with open(ann_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Wrong number of values: {len(parts)}")

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Check ranges
                if not (0 <= x_center <= 1):
                    raise ValueError(f"x_center out of range: {x_center}")
                if not (0 <= y_center <= 1):
                    raise ValueError(f"y_center out of range: {y_center}")
                if not (0 < width <= 1):
                    raise ValueError(f"width out of range: {width}")
                if not (0 < height <= 1):
                    raise ValueError(f"height out of range: {height}")
                if class_id >= len(classes):
                    raise ValueError(f"class_id out of range: {class_id}")

            valid_count += 1

        except Exception as e:
            print(f"   ❌ {ann_file.name}: {e}")

    print(f"   ✅ Valid files checked: {valid_count}/{len(sample_files)}")

    # Summary
    if valid_count == len(sample_files):
        print(f"\n🎉 Annotations look good!")
        print(f"   📁 Location: {output_dir}")
        print(f"   📊 {len(annotation_files)} annotation files")
        print(f"   🏷️ {len(classes)} classes")

        print(f"\n💡 To visualize annotations:")
        print(f"   1. Use the annotation_verification.py script")
        print(f"   2. Or manually check images with bounding boxes")

        return True
    else:
        print(f"\n⚠️  Some annotation files have issues!")
        return False


if __name__ == "__main__":
    # Your dataset path
    DATASET_PATH = "/home/karan-sankla/Testbench/objectfiles/dataset"

    quick_verify_annotations(DATASET_PATH)