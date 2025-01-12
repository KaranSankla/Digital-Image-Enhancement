import os
import numpy as np
import cv2
import apriltag

# Input and output paths
input_path = r"D:\Masters\Project\datas\SampleData\daten\tomatocan"
output_path = r"D:\Masters\Project\datas\ProcessedImages"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Process .npy files
for filename in os.listdir(input_path):
    if filename.endswith(".npy"):
        # Construct full file path
        npy_file_path = os.path.join(input_path, filename)

        # Load the .npy array
        try:
            img_array = np.load(npy_file_path)
        except Exception as e:
            print(f"Failed to load {npy_file_path}: {e}")
            continue

        # Check the shape of the array
        if img_array.ndim != 3 or img_array.shape[2] != 3:
            print(f"Skipping {filename}: Not a valid RGB image array.")
            continue

        # Convert array to uint8 for saving as an image
        img_array = img_array.astype(np.uint8)

        # Save as an RGB image
        output_file_path = os.path.join(output_path, filename.replace(".npy", ".png"))
        cv2.imwrite(output_file_path, img_array)
        print(f"Processed and saved: {output_file_path}")

        # Read the saved image
        processed_img = cv2.imread(output_file_path)

        # Display the image


image = cv2.imread(output_file_path,)
