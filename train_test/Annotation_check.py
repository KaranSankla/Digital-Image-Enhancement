#!/usr/bin/env python3
"""
Comprehensive YOLO Annotation Verification Tool
Visualizes bounding boxes on images to verify correctness
"""

import cv2
import numpy as np
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YOLOVerifier:
    def __init__(self, dataset_path, output_path):
        """
        Initialize YOLO annotation verifier

        Args:
            dataset_path: Path to your dataset root
            output_path: Path to annotation files
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.rgb_dir = self.dataset_path / "rgb"
        self.annotations_dir = self.output_path

        # Load class names
        self.classes = self.load_classes()

        # Define colors for each class (BGR format for OpenCV)
        self.colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

    def load_classes(self):
        """Load class names from classes.txt"""
        classes_file = self.annotations_dir / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            print(f"⚠️  Classes file not found: {classes_file}")
            return [f"class_{i}" for i in range(10)]  # Default class names

    def yolo_to_bbox(self, x_center, y_center, width, height, img_width, img_height):
        """
        Convert YOLO format to bounding box coordinates

        Args:
            x_center, y_center, width, height: YOLO format (normalized 0-1)
            img_width, img_height: Image dimensions

        Returns:
            (x_min, y_min, x_max, y_max) in pixel coordinates
        """
        # Convert normalized coordinates to pixel coordinates
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height

        # Calculate corner coordinates
        x_min = int(x_center_px - width_px / 2)
        y_min = int(y_center_px - height_px / 2)
        x_max = int(x_center_px + width_px / 2)
        y_max = int(y_center_px + height_px / 2)

        return (x_min, y_min, x_max, y_max)

    def load_annotations(self, annotation_file):
        """Load YOLO annotations from file"""
        annotations = []

        if not annotation_file.exists():
            return annotations

        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            annotations.append({
                                'class_id': class_id,
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height
                            })
        except Exception as e:
            print(f"Error loading annotations from {annotation_file}: {e}")

        return annotations

    def visualize_single_image(self, image_name, save_visualization=True, show_image=False):
        """
        Visualize annotations on a single image

        Args:
            image_name: Name of the image file (without extension)
            save_visualization: Save the visualization image
            show_image: Display image using matplotlib
        """
        # Find RGB image
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = self.rgb_dir / f"{image_name}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break

        if not image_path:
            print(f"❌ Image not found: {image_name}")
            return None

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None

        img_height, img_width = image.shape[:2]

        # Load annotations
        annotation_file = self.annotations_dir / f"{image_name}.txt"
        annotations = self.load_annotations(annotation_file)

        if not annotations:
            print(f"⚠️  No annotations found for {image_name}")
            return image

        # Draw bounding boxes
        annotated_image = image.copy()

        for i, ann in enumerate(annotations):
            class_id = ann['class_id']

            # Convert YOLO to pixel coordinates
            x_min, y_min, x_max, y_max = self.yolo_to_bbox(
                ann['x_center'], ann['y_center'], ann['width'], ann['height'],
                img_width, img_height
            )

            # Get color for this class
            color = self.colors[class_id % len(self.colors)]

            # Draw bounding box
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color, 2)

            # Draw class label
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
            label = f"{class_name} ({class_id})"

            # Calculate label position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = max(y_min - 5, label_size[1])

            # Draw label background
            cv2.rectangle(annotated_image,
                          (x_min, label_y - label_size[1] - 5),
                          (x_min + label_size[0], label_y + 5),
                          color, -1)

            # Draw label text
            cv2.putText(annotated_image, label, (x_min, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Print annotation info
            print(f"  📦 {label}: center=({ann['x_center']:.3f}, {ann['y_center']:.3f}), "
                  f"size=({ann['width']:.3f}×{ann['height']:.3f})")

        # Save visualization
        if save_visualization:
            vis_dir = self.output_path / "visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = vis_dir / f"{image_name}_annotated.png"
            cv2.imwrite(str(vis_path), annotated_image)
            print(f"✅ Saved visualization: {vis_path}")

        # Show image using matplotlib (better for Jupyter)
        if show_image:
            plt.figure(figsize=(12, 8))
            # Convert BGR to RGB for matplotlib
            rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_image)
            plt.title(f"YOLO Annotations: {image_name}")
            plt.axis('off')
            plt.show()

        return annotated_image

    def verify_random_samples(self, num_samples=5):
        """Verify annotations on random sample of images"""

        print(f"🔍 Verifying {num_samples} random samples...")
        print("=" * 60)

        # Get all annotation files
        annotation_files = list(self.annotations_dir.glob("*.txt"))
        annotation_files = [f for f in annotation_files if f.name != "classes.txt"]

        if not annotation_files:
            print("❌ No annotation files found!")
            return

        # Sample random files
        sample_files = random.sample(annotation_files, min(num_samples, len(annotation_files)))

        for i, ann_file in enumerate(sample_files):
            image_name = ann_file.stem
            print(f"\n📷 Sample {i + 1}: {image_name}")
            print("-" * 40)

            self.visualize_single_image(image_name, save_visualization=True)

    def check_annotation_statistics(self):
        """Generate statistics about annotations"""

        print("📊 Annotation Statistics")
        print("=" * 60)

        annotation_files = list(self.annotations_dir.glob("*.txt"))
        annotation_files = [f for f in annotation_files if f.name != "classes.txt"]

        total_annotations = 0
        class_counts = {}
        empty_files = 0
        bbox_sizes = []

        for ann_file in annotation_files:
            annotations = self.load_annotations(ann_file)

            if not annotations:
                empty_files += 1
                continue

            total_annotations += len(annotations)

            for ann in annotations:
                class_id = ann['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

                # Calculate bbox area (normalized)
                area = ann['width'] * ann['height']
                bbox_sizes.append(area)

        print(f"📄 Total annotation files: {len(annotation_files)}")
        print(f"📦 Total bounding boxes: {total_annotations}")
        print(f"🚫 Empty annotation files: {empty_files}")

        print(f"\n📊 Class Distribution:")
        for class_id, count in sorted(class_counts.items()):
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
            percentage = (count / total_annotations) * 100
            print(f"   {class_id}: {class_name} - {count} boxes ({percentage:.1f}%)")

        if bbox_sizes:
            print(f"\n📏 Bounding Box Size Statistics (normalized area):")
            print(f"   Average: {np.mean(bbox_sizes):.4f}")
            print(f"   Median:  {np.median(bbox_sizes):.4f}")
            print(f"   Min:     {np.min(bbox_sizes):.4f}")
            print(f"   Max:     {np.max(bbox_sizes):.4f}")

    def validate_annotation_format(self):
        """Validate annotation file formats"""

        print("🔍 Validating Annotation Format")
        print("=" * 60)

        annotation_files = list(self.annotations_dir.glob("*.txt"))
        annotation_files = [f for f in annotation_files if f.name != "classes.txt"]

        valid_files = 0
        invalid_files = []

        for ann_file in annotation_files:
            try:
                annotations = self.load_annotations(ann_file)

                for ann in annotations:
                    # Check value ranges
                    if not (0 <= ann['x_center'] <= 1):
                        raise ValueError(f"x_center out of range: {ann['x_center']}")
                    if not (0 <= ann['y_center'] <= 1):
                        raise ValueError(f"y_center out of range: {ann['y_center']}")
                    if not (0 < ann['width'] <= 1):
                        raise ValueError(f"width out of range: {ann['width']}")
                    if not (0 < ann['height'] <= 1):
                        raise ValueError(f"height out of range: {ann['height']}")
                    if ann['class_id'] < 0:
                        raise ValueError(f"negative class_id: {ann['class_id']}")

                valid_files += 1

            except Exception as e:
                invalid_files.append((ann_file.name, str(e)))

        print(f"✅ Valid files: {valid_files}")
        print(f"❌ Invalid files: {len(invalid_files)}")

        if invalid_files:
            print("\n🚨 Invalid Files:")
            for filename, error in invalid_files[:10]:  # Show first 10
                print(f"   {filename}: {error}")


def main():
    """Main verification function"""

    # Your dataset paths
    DATASET_PATH = "/home/karan-sankla/Testbench/objectfiles/dataset"
    OUTPUT_PATH = "/home/karan-sankla/Testbench/objectfiles/dataset/output"

    # Initialize verifier
    verifier = YOLOVerifier(DATASET_PATH, OUTPUT_PATH)

    print("🚀 YOLO Annotation Verification")
    print("=" * 60)

    # 1. Check statistics
    verifier.check_annotation_statistics()

    # 2. Validate format
    verifier.validate_annotation_format()

    # 3. Verify random samples with visualization
    verifier.verify_random_samples(num_samples=5)

    # 4. Verify specific image (if you want to check a particular one)
    # verifier.visualize_single_image("cleanser____000485_color", show_image=True)

    print(f"\n✅ Verification complete!")
    print(f"📁 Check visualizations in: {OUTPUT_PATH}/visualizations/")


if __name__ == "__main__":
    main()