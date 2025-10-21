#!/usr/bin/env python3
"""
Updated script for your specific directory structure:
/home/karan-sankla/Testbench/objectfiles/dataset/
├── rgb/
├── poses/
├── models/     # Your .ply files are here
└── depth/
"""

from ycb_yolo_generator import YCBToYOLO
import os
from pathlib import Path


def main():
    # Your actual directory structure
    BASE_PATH = "/home/karan-sankla/Testbench/objectfiles/dataset"
    MODELS_DIR = f"{BASE_PATH}/models"  # Your .ply files location
    OUTPUT_DIR = f"{BASE_PATH}/output"  # Where to save YOLO annotations

    print(f"🚀 Processing YCB dataset from: {BASE_PATH}")
    print("=" * 60)

    # Check if directories exist
    rgb_dir = f"{BASE_PATH}/rgb"
    poses_dir = f"{BASE_PATH}/poses"

    if not os.path.exists(rgb_dir):
        print(f"❌ RGB directory not found: {rgb_dir}")
        return

    if not os.path.exists(poses_dir):
        print(f"❌ Poses directory not found: {poses_dir}")
        return

    if not os.path.exists(MODELS_DIR):
        print(f"❌ Models directory not found: {MODELS_DIR}")
        return

    # Show what we found
    rgb_files = list(Path(rgb_dir).glob("*.png")) + list(Path(rgb_dir).glob("*.jpg"))
    pose_files = list(Path(poses_dir).glob("*.txt"))
    model_files = list(Path(MODELS_DIR).glob("*.ply"))

    print(f"✅ Found {len(rgb_files)} RGB images")
    print(f"✅ Found {len(pose_files)} pose files")
    print(f"✅ Found {len(model_files)} model files")

    if rgb_files:
        print(f"   RGB example: {rgb_files[0].name}")
    if pose_files:
        print(f"   Pose example: {pose_files[0].name}")
    if model_files:
        print(f"   Model examples:")
        for model in model_files[:3]:
            print(f"     - {model.name}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize converter with your models directory
    print(f"\n📁 Loading models from: {MODELS_DIR}")
    converter = YCBToYOLO(MODELS_DIR)

    # Your camera intrinsics
    fx = fy = 570.34222412109381
    cx = 319.5
    cy = 239.5

    print(f"\n🔄 Processing images...")
    # Process all your data
    converter.batch_process_directory(
        BASE_PATH,  # Root directory
        fx, fy, cx, cy,  # Camera intrinsics
        OUTPUT_DIR  # Output directory
    )

    # Save class names
    classes_file = f"{OUTPUT_DIR}/classes.txt"
    converter.save_class_names(classes_file)

    # Create YOLO dataset config
    create_dataset_yaml(converter.class_names, OUTPUT_DIR)

    print(f"\n🎉 Complete! Generated files:")
    print(f"   📄 Annotations: {OUTPUT_DIR}/*.txt")
    print(f"   📄 Classes: {classes_file}")
    print(f"   📄 YOLO config: {OUTPUT_DIR}/dataset.yaml")

    print(f"\n📊 Summary:")
    print(f"   Classes: {len(converter.class_names)}")
    for i, name in enumerate(converter.class_names):
        print(f"     {i}: {name}")


def create_dataset_yaml(class_names, output_dir):
    """Create dataset.yaml for YOLO training"""
    yaml_content = f"""# YCB Dataset for YOLO Training
# Generated from: /home/karan-sankla/Testbench/objectfiles/dataset

# Paths (update these for your train/val/test split)
train: {output_dir}/images/train
val: {output_dir}/images/val  
test: {output_dir}/images/test

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""

    yaml_path = f"{output_dir}/dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✅ Created {yaml_path}")


if __name__ == "__main__":
    main()