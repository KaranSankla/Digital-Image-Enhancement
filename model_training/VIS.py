import cv2
import numpy as np
from pathlib import Path
import random


def read_yolo_label(label_path):
    """
    Read YOLO format label file.
    Format: class_id center_x center_y width height (normalized 0-1)

    Returns: List of (class_id, center_x, center_y, width, height)
    """
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append((class_id, center_x, center_y, width, height))
    return boxes


def yolo_to_corners(center_x, center_y, width, height, img_width, img_height):
    """
    Convert YOLO format (normalized) to corner coordinates (pixels).

    Args:
        center_x, center_y, width, height: Normalized values (0-1)
        img_width, img_height: Image dimensions in pixels

    Returns: (x1, y1, x2, y2) in pixels
    """
    # Convert from normalized to pixel coordinates
    x_center = center_x * img_width
    y_center = center_y * img_height
    w = width * img_width
    h = height * img_height

    # Calculate corners
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    return x1, y1, x2, y2


def generate_colors(num_classes):
    """Generate distinct colors for each class."""
    random.seed(42)
    colors = {}
    for i in range(num_classes):
        colors[i] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return colors


def visualize_bboxes(images_dir, labels_dir, output_dir='visualizations', class_names=None, max_images=None):
    """
    Visualize bounding boxes on images.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        output_dir: Directory to save visualizations
        class_names: Dictionary mapping class_id to class name (optional)
        max_images: Maximum number of images to process (None for all)
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("BOUNDING BOX VISUALIZATION")
    print("=" * 70)

    # Get all image files
    image_files = list(images_path.glob('*.png')) + list(images_path.glob('*.jpg'))

    if max_images:
        image_files = image_files[:max_images]

    print(f"\nFound {len(image_files)} images to process")

    # Determine number of classes for color generation
    all_class_ids = set()
    for label_file in labels_path.glob('*.txt'):
        boxes = read_yolo_label(label_file)
        for box in boxes:
            all_class_ids.add(box[0])

    num_classes = max(all_class_ids) + 1 if all_class_ids else 10
    colors = generate_colors(num_classes)

    print(f"Detected {len(all_class_ids)} unique classes: {sorted(all_class_ids)}")

    processed = 0
    skipped = 0

    for img_file in image_files:
        # Find corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"

        if not label_file.exists():
            print(f"⚠ No label file for {img_file.name}")
            skipped += 1
            continue

        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"⚠ Could not read {img_file.name}")
            skipped += 1
            continue

        img_height, img_width = img.shape[:2]

        # Read bounding boxes
        boxes = read_yolo_label(label_file)

        if not boxes:
            print(f"⚠ No boxes in {label_file.name}")
            skipped += 1
            continue

        # Draw bounding boxes
        for class_id, center_x, center_y, width, height in boxes:
            # Convert to corner coordinates
            x1, y1, x2, y2 = yolo_to_corners(center_x, center_y, width, height, img_width, img_height)

            # Get color for this class
            color = colors.get(class_id, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            if class_names and class_id in class_names:
                label_text = f"{class_names[class_id]} ({class_id})"
            else:
                label_text = f"Class {class_id}"

            # Draw label background
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save visualization
        output_file = output_path / f"viz_{img_file.name}"
        cv2.imwrite(str(output_file), img)

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(image_files)} images...")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"✓ Successfully processed: {processed} images")
    if skipped > 0:
        print(f"⚠ Skipped: {skipped} images")
    print(f"\nOutput saved to: {output_dir}/")
    print("=" * 70)


def visualize_single_image(image_path, label_path, class_names=None, show=True, save_path=None):
    """
    Visualize bounding boxes on a single image.

    Args:
        image_path: Path to image file
        label_path: Path to label file
        class_names: Dictionary mapping class_id to class name (optional)
        show: Whether to display the image (requires GUI)
        save_path: Path to save the visualization (optional)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    img_height, img_width = img.shape[:2]

    # Read bounding boxes
    boxes = read_yolo_label(label_path)

    if not boxes:
        print(f"Warning: No boxes found in {label_path}")

    # Generate colors
    all_class_ids = [box[0] for box in boxes]
    num_classes = max(all_class_ids) + 1 if all_class_ids else 10
    colors = generate_colors(num_classes)

    # Draw bounding boxes
    for class_id, center_x, center_y, width, height in boxes:
        x1, y1, x2, y2 = yolo_to_corners(center_x, center_y, width, height, img_width, img_height)

        color = colors.get(class_id, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if class_names and class_id in class_names:
            label_text = f"{class_names[class_id]} ({class_id})"
        else:
            label_text = f"Class {class_id}"

        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save if requested
    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Saved visualization to {save_path}")

    # Show if requested
    if show:
        cv2.imshow('Bounding Box Visualization', img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configuration
    IMAGES_DIR = "/home/karan-sankla/Testbench/objectfiles/dataset_1/renamed_images"  # Directory with images
    LABELS_DIR = "/home/karan-sankla/Testbench/objectfiles/dataset_1/renamed_labels"  # Directory with label files
    OUTPUT_DIR = "/home/karan-sankla/Testbench/objectfiles/dataset_1/vis"  # Output directory
    MAX_IMAGES = 50  # Set to None to process all images

    # Optional: Define class names (if you know them)
    CLASS_NAMES = {
        0: "woodblock",
        1: "banana",
        2: "cleanser",
        3: "mediumclamp",
        4: "apple"
        # Add more classes as needed
    }

    # Visualize all images
    visualize_bboxes(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR, CLASS_NAMES, MAX_IMAGES)

    # Or visualize a single image:
    # visualize_single_image(
    #     "dataset/train/images/apple____000000_color.png",
    #     "dataset/train/labels/apple____000000_color.txt",
    #     CLASS_NAMES,
    #     show=False,
    #     save_path="single_viz.png"
    # )