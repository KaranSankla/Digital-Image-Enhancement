import os
import shutil
import json
import numpy as np
from pathlib import Path
import re


def reorganize_existing_ycb_dataset(source_dir, target_dir):
    """
    Reorganize YCB dataset from existing folder structure
    Source structure: /depth, /models, /poses, /rgb folders
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory structure
    dirs_to_create = ['rgb', 'depth', 'poses', 'models']
    for dir_name in dirs_to_create:
        (target_path / dir_name).mkdir(parents=True, exist_ok=True)

    print(f"Created directory structure at {target_dir}")

    # Track objects found
    objects_found = set()
    file_counts = {'rgb': 0, 'depth': 0, 'poses': 0, 'models': 0}

    # Process RGB files
    rgb_source = source_path / "rgb"
    if rgb_source.exists():
        print(f"Processing RGB files from {rgb_source}")
        for file_path in rgb_source.iterdir():
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Parse filename to extract object and number
                obj_name, img_num = parse_ycb_filename(file_path.name)
                if obj_name:
                    objects_found.add(obj_name)
                    new_name = f"{obj_name}_{img_num:06d}_color.png"
                    shutil.copy2(file_path, target_path / "rgb" / new_name)
                    file_counts['rgb'] += 1

    # Process Depth files
    depth_source = source_path / "depth"
    if depth_source.exists():
        print(f"Processing depth files from {depth_source}")
        for file_path in depth_source.iterdir():
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                obj_name, img_num = parse_ycb_filename(file_path.name)
                if obj_name:
                    new_name = f"{obj_name}_{img_num:06d}_depth.png"
                    shutil.copy2(file_path, target_path / "depth" / new_name)
                    file_counts['depth'] += 1

    # Process Pose files
    poses_source = source_path / "poses"
    if poses_source.exists():
        print(f"Processing pose files from {poses_source}")
        for file_path in poses_source.iterdir():
            if file_path.suffix.lower() == '.txt':
                obj_name, img_num = parse_ycb_filename(file_path.name)
                if obj_name:
                    new_name = f"{obj_name}_{img_num:06d}_pose.txt"
                    shutil.copy2(file_path, target_path / "poses" / new_name)
                    file_counts['poses'] += 1

    # Process Model files
    models_source = source_path / "models"
    if models_source.exists():
        print(f"Processing model files from {models_source}")
        for file_path in models_source.iterdir():
            if file_path.suffix.lower() == '.ply':
                shutil.copy2(file_path, target_path / "models" / file_path.name)
                file_counts['models'] += 1
                # Extract object name from model filename
                obj_name = file_path.name.replace("_nontextured.ply", "").replace(".ply", "")
                objects_found.add(obj_name)

    print(f"\n📊 Files processed:")
    for file_type, count in file_counts.items():
        print(f"   {file_type}: {count} files")

    # Create additional required files
    create_camera_intrinsics(target_path)
    create_object_list(target_path, objects_found)
    create_metadata_json(target_path, objects_found)

    if file_counts['rgb'] > 0:
        create_train_test_split(target_path, objects_found)

    print(f"\n✅ Dataset setup complete! Found objects: {sorted(objects_found)}")
    return sorted(objects_found)


def parse_ycb_filename(filename):
    """
    Parse YCB filenames like:
    - apple0_color.png -> ('apple', 0)
    - banana3_depth.png -> ('banana', 3)
    - apple0.txt -> ('apple', 0)
    """
    # Remove common suffixes
    base_name = filename
    suffixes = ['_color.png', '_depth.png', '.png', '.jpg', '.jpeg', '.txt']
    for suffix in suffixes:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break

    # Use regex to separate object name and number
    match = re.match(r'^([a-zA-Z_]+)(\d+)$', base_name)
    if match:
        obj_name = match.group(1)
        img_num = int(match.group(2))
        return obj_name, img_num

    # Fallback: assume no number at end
    return base_name, 0


def create_camera_intrinsics(target_path):
    """Create camera intrinsics file"""
    intrinsics_content = "570.34222412109381 570.3422241210938 319.5 239.5"

    with open(target_path / "camera_intrinsics.txt", 'w') as f:
        f.write(intrinsics_content)

    print("✅ Created camera_intrinsics.txt")


def create_object_list(target_path, objects):
    """Create object list file"""
    with open(target_path / "object_list.txt", 'w') as f:
        for obj in sorted(objects):
            f.write(f"{obj}\n")

    print(f"✅ Created object_list.txt with {len(objects)} objects")


def create_metadata_json(target_path, objects):
    """Create metadata JSON file"""
    metadata = {
        "dataset_name": "ycb_custom",
        "num_objects": len(objects),
        "image_width": 640,
        "image_height": 480,
        "camera_intrinsics": {
            "fx": 570.34222412109381,
            "fy": 570.3422241210938,
            "cx": 319.5,
            "cy": 239.5
        },
        "pose_format": "4x4_transformation_matrix",
        "coordinate_system": "camera_coordinates",
        "depth_scale": 1000,  # Assuming depth in mm
        "objects": []
    }

    for i, obj_name in enumerate(sorted(objects), 1):
        metadata["objects"].append({
            "id": i,
            "name": obj_name,
            "model_file": f"{obj_name}_nontextured.ply"
        })

    with open(target_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✅ Created metadata.json")


def create_train_test_split(target_path, objects, train_ratio=0.8):
    """Create train/test split files"""

    train_files = []
    test_files = []

    # Get all RGB files and split per object
    rgb_dir = target_path / "rgb"

    for obj_name in objects:
        obj_files = sorted([f for f in rgb_dir.iterdir()
                            if f.name.startswith(f"{obj_name}_") and f.name.endswith("_color.png")])

        if obj_files:
            # Split files (80% train, 20% test)
            n_train = max(1, int(len(obj_files) * train_ratio))
            train_files.extend([f.name for f in obj_files[:n_train]])
            test_files.extend([f.name for f in obj_files[n_train:]])

    # Write train list
    with open(target_path / "train_list.txt", 'w') as f:
        for filename in sorted(train_files):
            f.write(f"{filename}\n")

    # Write test list
    with open(target_path / "test_list.txt", 'w') as f:
        for filename in sorted(test_files):
            f.write(f"{filename}\n")

    print(f"✅ Created train_list.txt ({len(train_files)} files)")
    print(f"✅ Created test_list.txt ({len(test_files)} files)")


def validate_dataset(dataset_path):
    """Validate that dataset is properly structured"""
    path = Path(dataset_path)

    # Check required directories
    required_dirs = ['rgb', 'depth', 'poses', 'models']
    for dir_name in required_dirs:
        if not (path / dir_name).exists():
            print(f"❌ Missing directory: {dir_name}")
            return False

    # Count files
    rgb_files = list((path / "rgb").glob("*_color.png"))
    depth_files = list((path / "depth").glob("*_depth.png"))
    pose_files = list((path / "poses").glob("*_pose.txt"))
    model_files = list((path / "models").glob("*.ply"))

    print(f"\n📊 Dataset Statistics:")
    print(f"   RGB images: {len(rgb_files)}")
    print(f"   Depth images: {len(depth_files)}")
    print(f"   Pose files: {len(pose_files)}")
    print(f"   Model files: {len(model_files)}")

    # Check if counts match
    if len(rgb_files) == len(depth_files) == len(pose_files):
        print("✅ RGB, depth, and pose file counts match!")

        # Test loading a sample pose
        if pose_files:
            test_pose_loading(dataset_path, pose_files[0].name)

        return True
    else:
        print("❌ File counts don't match!")
        print("   This might indicate missing files")
        return False


def test_pose_loading(dataset_path, pose_filename):
    """Test loading a specific pose file"""
    pose_path = Path(dataset_path) / "poses" / pose_filename

    try:
        pose_matrix = np.loadtxt(pose_path)
        print(f"\n✅ Pose file test ({pose_filename}):")
        print(f"   Shape: {pose_matrix.shape}")
        if pose_matrix.shape == (4, 4):
            det = np.linalg.det(pose_matrix[:3, :3])
            print(f"   Rotation determinant: {det:.3f} (should be ≈ ±1)")
            print(f"   Translation: [{pose_matrix[0, 3]:.3f}, {pose_matrix[1, 3]:.3f}, {pose_matrix[2, 3]:.3f}]")
        return True
    except Exception as e:
        print(f"❌ Error loading pose: {e}")
        return False


# Usage:
if __name__ == "__main__":
    # Update these paths to match your setup
    source_directory = "/home/karan-sankla/Testbench/objectfiles/dataset"
    target_directory = "/home/karan-sankla/Testbench/objectfiles/dataset/ycb_dataset_organized"

    print("🚀 Starting YCB dataset reorganization...")
    objects = reorganize_existing_ycb_dataset(source_directory, target_directory)

    print("\n🔍 Validating dataset...")
    validate_dataset(target_directory)