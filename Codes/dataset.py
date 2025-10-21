import os
import json
import shutil
from pathlib import Path
from collections import defaultdict


def analyze_actual_files(dataset_path):
    """
    Analyze what files actually exist after renaming
    """
    path = Path(dataset_path)

    print("🔍 Analyzing actual files in dataset...\n")

    # Check each directory
    dirs = ['rgb', 'depth', 'poses', 'models']
    file_info = {}

    for dir_name in dirs:
        dir_path = path / dir_name
        if not dir_path.exists():
            print(f"❌ Directory {dir_name}/ not found")
            continue

        files = list(dir_path.iterdir())
        file_info[dir_name] = files

        print(f"📁 {dir_name}/ - {len(files)} files")

        # Show some examples
        valid_files = [f for f in files if not f.name.startswith('.')]
        for i, file_path in enumerate(valid_files[:3]):
            print(f"   📄 {file_path.name}")

        if len(valid_files) > 3:
            print(f"   📄 ... and {len(valid_files) - 3} more")
        print()

    return file_info


def extract_clean_objects(file_info):
    """
    Extract clean object names from actual files
    """
    objects = set()

    # Get object names from RGB files (most reliable)
    if 'rgb' in file_info:
        for file_path in file_info['rgb']:
            filename = file_path.name

            # Expected pattern: {object}_{sequence}_color.png
            if '_color.png' in filename:
                # Split by underscore and take everything before the sequence number
                parts = filename.replace('_color.png', '').split('_')

                # Find where the sequence number starts (all digits)
                for i in range(len(parts) - 1, 0, -1):
                    if parts[i].isdigit():
                        object_name = '_'.join(parts[:i])
                        if object_name and not 'model' in object_name.lower():
                            objects.add(object_name)
                        break

    return sorted(objects)


def count_files_per_object(file_info, objects):
    """
    Count how many files each object actually has
    """
    object_counts = {obj: {'rgb': 0, 'depth': 0, 'poses': 0} for obj in objects}

    # Count RGB files
    if 'rgb' in file_info:
        for file_path in file_info['rgb']:
            for obj in objects:
                if file_path.name.startswith(f"{obj}_") and '_color.png' in file_path.name:
                    object_counts[obj]['rgb'] += 1
                    break

    # Count depth files
    if 'depth' in file_info:
        for file_path in file_info['depth']:
            for obj in objects:
                if file_path.name.startswith(f"{obj}_") and '_depth.png' in file_path.name:
                    object_counts[obj]['depth'] += 1
                    break

    # Count pose files
    if 'poses' in file_info:
        for file_path in file_info['poses']:
            for obj in objects:
                if file_path.name.startswith(f"{obj}_") and '_pose.txt' in file_path.name:
                    object_counts[obj]['poses'] += 1
                    break

    return object_counts


def clean_model_files(dataset_path, objects):
    """
    Clean up model files to have proper names
    """
    models_dir = Path(dataset_path) / "models"
    if not models_dir.exists():
        return

    print("🧹 Cleaning model files...")

    # Get current model files
    model_files = list(models_dir.glob("*.ply"))

    # Create mapping of messy names to clean names
    rename_mapping = {}

    for model_file in model_files:
        filename = model_file.name

        # Find which object this belongs to
        for obj in objects:
            if obj.lower() in filename.lower() and obj != '':
                clean_name = f"{obj}_model.ply"
                if filename != clean_name:
                    rename_mapping[model_file] = models_dir / clean_name
                break

    # Rename model files
    for old_path, new_path in rename_mapping.items():
        print(f"   🔄 {old_path.name} → {new_path.name}")
        old_path.rename(new_path)


def create_clean_file_lists(dataset_path, objects, object_counts):
    """
    Create clean train/test/val lists
    """
    path = Path(dataset_path)

    print("📝 Creating clean file lists...")

    # Collect all RGB files per object
    rgb_files_per_object = defaultdict(list)

    rgb_dir = path / "rgb"
    if rgb_dir.exists():
        for file_path in rgb_dir.glob("*_color.png"):
            for obj in objects:
                if file_path.name.startswith(f"{obj}_"):
                    rgb_files_per_object[obj].append(file_path.name)
                    break

    # Sort files for each object
    for obj in rgb_files_per_object:
        rgb_files_per_object[obj].sort()

    # Create train/test split
    train_files = []
    test_files = []

    for obj in objects:
        obj_files = rgb_files_per_object[obj]
        if obj_files:
            split_idx = max(1, int(len(obj_files) * 0.8))
            train_files.extend(obj_files[:split_idx])
            test_files.extend(obj_files[split_idx:])

    # Write lists
    with open(path / "train_list.txt", 'w') as f:
        for filename in sorted(train_files):
            f.write(f"{filename}\n")

    with open(path / "test_list.txt", 'w') as f:
        for filename in sorted(test_files):
            f.write(f"{filename}\n")

    val_files = test_files[:len(test_files) // 2] if len(test_files) > 2 else test_files[:1]
    with open(path / "val_list.txt", 'w') as f:
        for filename in sorted(val_files):
            f.write(f"{filename}\n")

    return len(train_files), len(test_files), len(val_files)


def create_clean_metadata(dataset_path, objects, object_counts, train_count, test_count, val_count):
    """
    Create clean metadata files
    """
    path = Path(dataset_path)

    # Object list
    with open(path / "object_list.txt", 'w') as f:
        for obj in objects:
            f.write(f"{obj}\n")

    # Class mapping
    with open(path / "class_mapping.txt", 'w') as f:
        for i, obj in enumerate(objects, 1):
            f.write(f"{i} {obj}\n")

    # Camera intrinsics
    with open(path / "camera_intrinsics.txt", 'w') as f:
        f.write("570.34222412109381 570.3422241210938 319.5 239.5\n")

    # Complete metadata
    total_sequences = sum(counts['rgb'] for counts in object_counts.values())

    metadata = {
        "dataset_name": "ycb_pose_dataset",
        "version": "1.1_cleaned",
        "created_by": "ycb_cleanup_tool",
        "num_objects": len(objects),
        "total_sequences": total_sequences,
        "train_samples": train_count,
        "test_samples": test_count,
        "val_samples": val_count,
        "image_format": "png",
        "depth_format": "png",
        "pose_format": "4x4_transformation_matrix",
        "coordinate_system": "camera_coordinates",
        "objects": [
            {
                "id": i + 1,
                "name": obj,
                "sequences": object_counts[obj]['rgb'],
                "rgb_files": object_counts[obj]['rgb'],
                "depth_files": object_counts[obj]['depth'],
                "pose_files": object_counts[obj]['poses'],
                "model_file": f"{obj}_model.ply",
                "complete_triplets": min(object_counts[obj]['rgb'],
                                         object_counts[obj]['depth'],
                                         object_counts[obj]['poses'])
            }
            for i, obj in enumerate(objects)
        ],
        "camera_intrinsics": {
            "fx": 570.34222412109381,
            "fy": 570.3422241210938,
            "cx": 319.5,
            "cy": 239.5,
            "image_width": 640,
            "image_height": 480
        },
        "file_naming_convention": {
            "rgb": "{object}_{sequence:06d}_color.png",
            "depth": "{object}_{sequence:06d}_depth.png",
            "pose": "{object}_{sequence:06d}_pose.txt",
            "model": "{object}_model.ply"
        }
    }

    with open(path / "dataset_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def validate_dataset(dataset_path, objects):
    """
    Final validation of the cleaned dataset
    """
    path = Path(dataset_path)

    print("\n✅ Final Dataset Validation:")
    print("=" * 50)

    issues = []

    # Check required files exist
    required_files = [
        "object_list.txt",
        "class_mapping.txt",
        "camera_intrinsics.txt",
        "dataset_info.json",
        "train_list.txt",
        "test_list.txt",
        "val_list.txt"
    ]

    for filename in required_files:
        if not (path / filename).exists():
            issues.append(f"Missing {filename}")
        else:
            print(f"✅ {filename}")

    # Check model files
    models_dir = path / "models"
    for obj in objects:
        model_file = models_dir / f"{obj}_model.ply"
        if not model_file.exists():
            issues.append(f"Missing model file for {obj}")
        else:
            print(f"✅ {obj}_model.ply")

    # Check sample files
    rgb_dir = path / "rgb"
    depth_dir = path / "depth"
    poses_dir = path / "poses"

    sample_files = list(rgb_dir.glob("*_000000_color.png"))[:1]
    if sample_files:
        sample_base = sample_files[0].name.replace("_color.png", "")
        depth_file = depth_dir / f"{sample_base}_depth.png"
        pose_file = poses_dir / f"{sample_base}_pose.txt"

        if depth_file.exists() and pose_file.exists():
            print(f"✅ Sample triplet: {sample_base}")
        else:
            issues.append(f"Incomplete sample triplet: {sample_base}")

    if issues:
        print(f"\n❌ Issues found:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print(f"\n🎉 Dataset is clean and ready for training!")
        return True


# Main execution
if __name__ == "__main__":
    dataset_path = "/home/karan-sankla/Testbench/objectfiles/dataset"

    print("🧹 Dataset Cleanup and Validation Tool")
    print("=" * 50)

    # Step 1: Analyze actual files
    file_info = analyze_actual_files(dataset_path)

    # Step 2: Extract clean object names
    objects = extract_clean_objects(file_info)
    print(f"📦 Found objects: {objects}")

    if not objects:
        print("❌ No valid objects found!")
        exit(1)

    # Step 3: Count files per object
    object_counts = count_files_per_object(file_info, objects)

    print(f"\n📊 File counts per object:")
    for obj, counts in object_counts.items():
        complete = min(counts['rgb'], counts['depth'], counts['poses'])
        print(f"   {obj}: RGB={counts['rgb']}, Depth={counts['depth']}, Poses={counts['poses']} → {complete} complete")

    # Step 4: Clean model files
    clean_model_files(dataset_path, objects)

    # Step 5: Create clean file lists
    train_count, test_count, val_count = create_clean_file_lists(dataset_path, objects, object_counts)

    # Step 6: Create clean metadata
    metadata = create_clean_metadata(dataset_path, objects, object_counts, train_count, test_count, val_count)

    print(f"\n📄 Created clean metadata files:")
    print(f"   📊 Objects: {len(objects)}")
    print(f"   📊 Total sequences: {metadata['total_sequences']}")
    print(f"   📊 Train: {train_count}, Test: {test_count}, Val: {val_count}")

    # Step 7: Final validation
    validate_dataset(dataset_path, objects)

    print(f"\n🚀 Your dataset is now ready for training!")
    print(f"   Objects: {', '.join(objects)}")
    print(f"   Ready for: DenseFusion, PVNet, DOPE, etc.")