import os
import shutil
import re
from pathlib import Path


def safe_rename_and_split(images_dir, labels_dir, output_base_dir='dataset', train_dir='train', test_dir='test',
                          split_ratio=0.8):
    """
    Safely rename files and split into train/test WITHOUT modifying original dataset.
    Creates renamed copies in a new directory structure.

    Args:
        images_dir: Directory containing original PNG files (e.g., apple0_color.png)
        labels_dir: Directory containing TXT files (e.g., apple____000000_color.txt)
        output_base_dir: Base directory for all output
        train_dir: Train subdirectory name
        test_dir: Test subdirectory name
        split_ratio: Ratio for train split (default 0.8 for 80%)
    """
    import random

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_base_dir)

    # Create output directory structure
    renamed_images_path = output_path / 'renamed_images'
    renamed_labels_path = output_path / 'renamed_labels'
    train_images_path = output_path / train_dir / 'images'
    train_labels_path = output_path / train_dir / 'labels'
    test_images_path = output_path / test_dir / 'images'
    test_labels_path = output_path / test_dir / 'labels'

    renamed_images_path.mkdir(parents=True, exist_ok=True)
    renamed_labels_path.mkdir(parents=True, exist_ok=True)
    train_images_path.mkdir(parents=True, exist_ok=True)
    train_labels_path.mkdir(parents=True, exist_ok=True)
    test_images_path.mkdir(parents=True, exist_ok=True)
    test_labels_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAFE RENAME AND SPLIT - ORIGINAL FILES WILL NOT BE MODIFIED")
    print("=" * 70)

    # Step 1: Build mapping from TXT files
    print("\nStep 1: Analyzing label files...")
    txt_files = list(labels_path.glob('*.txt'))

    # Create mapping: (prefix, number) -> txt_basename
    txt_mapping = {}
    for txt_file in txt_files:
        basename = txt_file.stem  # e.g., apple____000000_color

        # Parse pattern: prefix____number_color
        match = re.match(r'^(.+?)____(\d+)_color$', basename)
        if match:
            prefix = match.group(1)  # e.g., 'apple'
            number = int(match.group(2))  # e.g., 0
            txt_mapping[(prefix, number)] = basename

    print(f"Found {len(txt_mapping)} label files")

    # Step 2: Match PNG files and create renamed copies
    print("\nStep 2: Matching and renaming image files...")
    png_files = list(images_path.glob('*.png'))
    matched_pairs = []
    unmatched_pngs = []
    unmatched_txts = set(txt_mapping.keys())

    for png_file in png_files:
        basename = png_file.stem  # e.g., apple0_color

        # Parse pattern: prefixNUMBER_color
        match = re.match(r'^(.+?)(\d+)_color$', basename)
        if match:
            prefix = match.group(1)  # e.g., 'apple'
            number = int(match.group(2))  # e.g., 0

            key = (prefix, number)
            if key in txt_mapping:
                # Found a match!
                txt_basename = txt_mapping[key]
                txt_file = labels_path / f'{txt_basename}.txt'

                # Copy and rename PNG
                new_png_name = f'{txt_basename}.png'
                new_png_path = renamed_images_path / new_png_name
                shutil.copy2(png_file, new_png_path)

                # Copy TXT (already has correct name)
                new_txt_path = renamed_labels_path / f'{txt_basename}.txt'
                shutil.copy2(txt_file, new_txt_path)

                matched_pairs.append((new_txt_path, new_png_path))
                unmatched_txts.discard(key)

                print(f"✓ Matched: {png_file.name} -> {new_png_name}")
            else:
                unmatched_pngs.append(png_file.name)
        else:
            unmatched_pngs.append(png_file.name)

    print(f"\n✓ Successfully matched and renamed {len(matched_pairs)} pairs")

    if unmatched_pngs:
        print(f"\n⚠ Warning: {len(unmatched_pngs)} PNG files had no matching TXT:")
        for name in unmatched_pngs[:5]:
            print(f"  - {name}")
        if len(unmatched_pngs) > 5:
            print(f"  ... and {len(unmatched_pngs) - 5} more")

    if unmatched_txts:
        print(f"\n⚠ Warning: {len(unmatched_txts)} TXT files had no matching PNG:")
        for key in list(unmatched_txts)[:5]:
            print(f"  - {key[0]}{key[1]}_color.png (expected)")
        if len(unmatched_txts) > 5:
            print(f"  ... and {len(unmatched_txts) - 5} more")

    # Step 3: Split into train/test
    print(f"\nStep 3: Splitting into train ({int(split_ratio * 100)}%) and test ({int((1 - split_ratio) * 100)}%)...")
    random.seed(42)
    random.shuffle(matched_pairs)

    split_idx = int(len(matched_pairs) * split_ratio)
    train_pairs = matched_pairs[:split_idx]
    test_pairs = matched_pairs[split_idx:]

    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Test set: {len(test_pairs)} pairs")

    # Step 4: Copy to train/test directories
    print("\nStep 4: Creating train/test datasets...")

    for txt_file, png_file in train_pairs:
        shutil.copy2(txt_file, train_labels_path / txt_file.name)
        shutil.copy2(png_file, train_images_path / png_file.name)

    for txt_file, png_file in test_pairs:
        shutil.copy2(txt_file, test_labels_path / txt_file.name)
        shutil.copy2(png_file, test_images_path / png_file.name)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOriginal files: UNTOUCHED")
    print(f"  - {images_dir}/")
    print(f"  - {labels_dir}/")
    print(f"\nRenamed files:")
    print(f"  - {output_base_dir}/renamed_images/")
    print(f"  - {output_base_dir}/renamed_labels/")
    print(f"\nTrain/Test split:")
    print(f"  - {output_base_dir}/{train_dir}/images/ ({len(train_pairs)} images)")
    print(f"  - {output_base_dir}/{train_dir}/labels/ ({len(train_pairs)} labels)")
    print(f"  - {output_base_dir}/{test_dir}/images/ ({len(test_pairs)} images)")
    print(f"  - {output_base_dir}/{test_dir}/labels/ ({len(test_pairs)} labels)")
    print("=" * 70)


if __name__ == "__main__":
    # Configuration
    IMAGES_DIR = "/home/karan-sankla/Testbench/objectfiles/Images"  # Original PNG files (apple0_color.png)
    LABELS_DIR = "/home/karan-sankla/Testbench/objectfiles/dataset/labels/Original labels "  # Original TXT files (apple____000000_color.txt)
    OUTPUT_BASE_DIR = "/home/karan-sankla/Testbench/objectfiles/dataset_1"  # All output goes here
    TRAIN_DIR = "train"
    TEST_DIR = "test"
    SPLIT_RATIO = 0.8  # 80-20 split

    # Run the script
    safe_rename_and_split(IMAGES_DIR, LABELS_DIR, OUTPUT_BASE_DIR, TRAIN_DIR, TEST_DIR, SPLIT_RATIO)