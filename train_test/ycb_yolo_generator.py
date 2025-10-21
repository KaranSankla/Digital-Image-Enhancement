import numpy as np
import cv2
import trimesh
import os
from pathlib import Path
import json
import glob


class YCBToYOLO:
    def __init__(self, ycb_objects_path):
        """
        Initialize the YCB to YOLO converter

        Args:
            ycb_objects_path: Path to directory containing YCB .ply files
        """
        self.ycb_objects_path = ycb_objects_path
        self.object_meshes = {}
        self.class_names = []
        self.load_ycb_meshes()

    def load_ycb_meshes(self):
        """Load all .ply files and extract bounding box information"""
        ply_files = list(Path(self.ycb_objects_path).glob("*.ply"))

        for i, ply_file in enumerate(ply_files):
            object_name = ply_file.stem
            self.class_names.append(object_name)

            try:
                mesh = trimesh.load(str(ply_file))
                self.object_meshes[i] = {
                    'name': object_name,
                    'mesh': mesh,
                    'bbox': mesh.bounding_box.bounds  # min and max coordinates
                }
                print(f"Loaded {object_name}: {mesh.vertices.shape[0]} vertices")
            except Exception as e:
                print(f"Error loading {ply_file}: {e}")

    def load_pose_from_txt(self, pose_txt_path):
        """
        Load 4x4 transformation matrix from .txt file

        Args:
            pose_txt_path: Path to .txt file containing 4x4 transformation matrix

        Returns:
            Dictionary with rotation matrix and translation vector
        """
        try:
            # Load the 4x4 transformation matrix
            transform_matrix = np.loadtxt(pose_txt_path)

            if transform_matrix.shape != (4, 4):
                raise ValueError(f"Expected 4x4 matrix, got {transform_matrix.shape}")

            # Extract rotation (3x3) and translation (3x1)
            rotation_matrix = transform_matrix[:3, :3]
            translation_vector = transform_matrix[:3, 3]

            return {
                'rotation': rotation_matrix,
                'translation': translation_vector
            }

        except Exception as e:
            print(f"Error loading pose from {pose_txt_path}: {e}")
            return None

    def create_intrinsic_matrix(self, fx, fy, cx, cy):
        """
        Create 3x3 intrinsic matrix from camera parameters

        Args:
            fx, fy: focal lengths
            cx, cy: principal point coordinates
        """
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    def project_3d_bbox_to_2d(self, bbox_3d, rotation_matrix, translation_vector, intrinsic_matrix):
        """
        Project 3D bounding box corners to 2D image coordinates

        Args:
            bbox_3d: 3D bounding box bounds [min_xyz, max_xyz]
            rotation_matrix: 3x3 rotation matrix
            translation_vector: 3x1 translation vector
            intrinsic_matrix: 3x3 camera intrinsic matrix

        Returns:
            2D bounding box corners
        """
        # Create 8 corners of 3D bounding box
        min_pt, max_pt = bbox_3d
        corners_3d = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],  # min corner
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],  # max corner
            [min_pt[0], max_pt[1], max_pt[2]]
        ]).T  # 3x8

        # Apply rotation and translation
        transformed_corners = rotation_matrix @ corners_3d + translation_vector.reshape(3, 1)

        # Project to 2D using camera intrinsics
        # Add homogeneous coordinate
        corners_2d_homo = intrinsic_matrix @ transformed_corners

        # Convert from homogeneous to 2D coordinates
        corners_2d = corners_2d_homo[:2] / corners_2d_homo[2:3]

        return corners_2d.T  # Return as Nx2

    def get_2d_bbox_from_corners(self, corners_2d, image_width, image_height):
        """
        Get axis-aligned 2D bounding box from projected corners

        Args:
            corners_2d: Nx2 array of 2D corner points
            image_width: Image width
            image_height: Image height

        Returns:
            YOLO format bounding box [x_center, y_center, width, height] (normalized)
        """
        # Get min/max coordinates
        x_min = np.min(corners_2d[:, 0])
        x_max = np.max(corners_2d[:, 0])
        y_min = np.min(corners_2d[:, 1])
        y_max = np.max(corners_2d[:, 1])

        # Clamp to image boundaries
        x_min = max(0, x_min)
        x_max = min(image_width, x_max)
        y_min = max(0, y_min)
        y_max = min(image_height, y_max)

        # Convert to YOLO format (normalized)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        x_center = x_min + bbox_width / 2
        y_center = y_min + bbox_height / 2

        # Normalize to [0, 1]
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = bbox_width / image_width
        height_norm = bbox_height / image_height

        return [x_center_norm, y_center_norm, width_norm, height_norm]

    def extract_object_info_from_filename(self, filename):
        """
        Extract object name and image ID from filename

        Examples:
        - cleanser____000485_color.png -> object: cleanser, image_id: 000485
        - mediumclamp____000197_pose.txt -> object: mediumclamp, image_id: 000197
        """
        try:
            # Remove file extension
            name_without_ext = Path(filename).stem

            # Split on '____' to separate object name and rest
            parts = name_without_ext.split('____')
            if len(parts) >= 2:
                object_name = parts[0]
                # Extract image ID (assuming it's the numeric part)
                rest = parts[1]
                # Remove '_color', '_pose', etc. and extract numbers
                image_id = ''.join(filter(str.isdigit, rest.split('_')[0]))
                return object_name, image_id
            else:
                return None, None
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            return None, None

    def find_object_id_by_name(self, object_name):
        """Find object ID by matching object name with loaded models"""
        for obj_id, obj_info in self.object_meshes.items():
            model_name = obj_info['name']
            # Check if object_name is contained in model_name
            if object_name.lower() in model_name.lower():
                return obj_id
        return None

    def generate_yolo_annotation_from_files(self, rgb_image_path, pose_txt_path, fx, fy, cx, cy, object_id, output_dir):
        """
        Generate YOLO annotation from individual files (your data format)

        Args:
            rgb_image_path: Path to RGB image
            pose_txt_path: Path to .txt file with 4x4 transformation matrix
            fx, fy, cx, cy: Camera intrinsic parameters
            object_id: ID of the object (0-4 for your 5 YCB objects)
            output_dir: Directory to save annotations
        """
        # Load RGB image to get dimensions
        rgb_image = cv2.imread(rgb_image_path)
        if rgb_image is None:
            print(f"Could not load RGB image: {rgb_image_path}")
            return

        image_height, image_width = rgb_image.shape[:2]

        # Load pose data
        pose_data = self.load_pose_from_txt(pose_txt_path)
        if pose_data is None:
            return

        # Create intrinsic matrix
        intrinsic_matrix = self.create_intrinsic_matrix(fx, fy, cx, cy)

        # Prepare annotation
        annotations = []

        if object_id not in self.object_meshes:
            print(f"Object ID {object_id} not found in loaded meshes")
            return

        # Get 3D bounding box
        bbox_3d = self.object_meshes[object_id]['bbox']

        # Project 3D bbox to 2D
        corners_2d = self.project_3d_bbox_to_2d(
            bbox_3d,
            pose_data['rotation'],
            pose_data['translation'],
            intrinsic_matrix
        )

        # Get YOLO format bbox
        yolo_bbox = self.get_2d_bbox_from_corners(corners_2d, image_width, image_height)

        # Check if bbox is valid (not too small or outside image)
        if yolo_bbox[2] > 0.01 and yolo_bbox[3] > 0.01:  # width and height > 1% of image
            annotations.append(f"{object_id} {' '.join(map(str, yolo_bbox))}")

        # Save annotation file
        image_name = Path(rgb_image_path).stem
        annotation_path = Path(output_dir) / f"{image_name}.txt"

        os.makedirs(output_dir, exist_ok=True)

        # If file exists, append to it (for multiple objects in same image)
        mode = 'a' if annotation_path.exists() else 'w'
        with open(annotation_path, mode) as f:
            if annotations:
                if mode == 'a':
                    f.write('\n')
                f.write('\n'.join(annotations))

        print(f"Generated annotation: {annotation_path}")
        return annotation_path

    def batch_process_directory(self, data_directory, fx, fy, cx, cy, output_dir):
        """
        Process a directory with your specific naming convention

        Your format:
        - RGB: cleanser____000485_color.png
        - Pose: mediumclamp____000197_pose.txt

        Args:
            data_directory: Root directory containing rgb/ and poses/ folders
            fx, fy, cx, cy: Camera intrinsic parameters
            output_dir: Directory to save YOLO annotations
        """
        data_dir = Path(data_directory)
        rgb_dir = data_dir / "rgb"
        poses_dir = data_dir / "poses"

        if not rgb_dir.exists():
            print(f"RGB directory not found: {rgb_dir}")
            return

        if not poses_dir.exists():
            print(f"Poses directory not found: {poses_dir}")
            return

        # Clear output directory
        output_path = Path(output_dir)
        if output_path.exists():
            for f in output_path.glob("*.txt"):
                f.unlink()
        os.makedirs(output_dir, exist_ok=True)

        # Get all pose files
        pose_files = list(poses_dir.glob("*_pose.txt"))
        print(f"Found {len(pose_files)} pose files")

        processed_count = 0
        for pose_file in pose_files:
            # Extract object name and image ID from pose filename
            object_name, image_id = self.extract_object_info_from_filename(pose_file.name)

            if not object_name or not image_id:
                print(f"Could not parse pose filename: {pose_file.name}")
                continue

            # Find corresponding RGB image
            rgb_pattern = f"{object_name}____{image_id}_color.png"
            rgb_path = rgb_dir / rgb_pattern

            if not rgb_path.exists():
                print(f"RGB image not found: {rgb_pattern}")
                continue

            # Find object ID by name
            object_id = self.find_object_id_by_name(object_name)
            if object_id is None:
                print(f"Object '{object_name}' not found in loaded models")
                continue

            # Generate annotation
            try:
                self.generate_yolo_annotation_from_files(
                    str(rgb_path),
                    str(pose_file),
                    fx, fy, cx, cy,
                    object_id,
                    output_dir
                )
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")

            except Exception as e:
                print(f"Error processing {pose_file.name}: {e}")
                continue

        print(f"Successfully processed {processed_count} pose files")

    def save_class_names(self, output_path):
        """Save class names for YOLO training"""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        print(f"Saved class names to: {output_path}")


# Example usage for your specific data format
if __name__ == "__main__":
    # Initialize converter with your models
    BASE_PATH = "/home/karan-sankla/Testbench/objectfiles/dataset"
    MODELS_DIR = f"{BASE_PATH}/models"
    OUTPUT_DIR = f"{BASE_PATH}/output"

    converter = YCBToYOLO(MODELS_DIR)

    # Your camera intrinsics
    fx = fy = 570.34222412109381
    cx = 319.5
    cy = 239.5

    # Process all your data
    converter.batch_process_directory(
        BASE_PATH,
        fx, fy, cx, cy,
        OUTPUT_DIR
    )

    # Save class names
    converter.save_class_names(f"{OUTPUT_DIR}/classes.txt")

    print("✅ YOLO annotation generation complete!")
    print(f"Generated annotations for {len(converter.class_names)} classes:")
    for i, name in enumerate(converter.class_names):
        print(f"  {i}: {name}")