import cv2
import numpy as np
import pupil_apriltags as apriltag
import os


IMAGE_PATH = "/home/karan-sankla/Testbench/objectfiles/apple/RGB_IMAGE"
OUTPUT_PATH = "/home/karan-sankla/Testbench/objectfiles/apple/Masked_images"

def process_apriltag_detection(image_path):
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    detector = apriltag.Detector()
    detections = detector.detect(gray_image)
    corners_list = []
    for detection in detections:
        corners = detection.corners
        corners_list.append(corners)
    return detections, corners_list, input_image.shape[:2]

def create_mask(image_shape, corners_list):
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)  # Black mask

    for corners in corners_list:
        # Convert corners to int32 and reshape for fillPoly
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)  # Fill tag area with white (255)
    return mask
for filename in os.listdir(IMAGE_PATH):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(IMAGE_PATH, filename)
        try:
            detections, corners_list, image_shape = process_apriltag_detection(image_path)
            print(f"Detected {len(detections)} AprilTags in {filename}.")

            if len(corners_list) > 0:
                mask = create_mask(image_shape, corners_list)
                mask_filename = os.path.splitext(filename)[0] + "_mask.png"
                mask_path = os.path.join(OUTPUT_PATH, mask_filename)
                cv2.imwrite(mask_path, mask)
                print(f"Mask saved: {mask_path}")
            else:
                print(f"No AprilTags detected in {filename}. Skipping mask.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


git config --global user.email "you@example.com"
  git config --global user.name "Your Name"
