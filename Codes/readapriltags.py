import cv2
import numpy as np
import pupil_apriltags as apriltag

# Constants
IMAGE_PATH = "tomatocan0_color.png"



def create_mask_from_polygons(image_shape, corners_list):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Initialize mask (same size as image)
    for corners in corners_list:
        polygon = np.array(corners , dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)  # Fill the polygon
    return mask
def process_apriltag_detection(image_path):
    """
    Detects AprilTags in the given image and returns the detections, an annotated image, and expanded corner coordinates.
    """
    # Read the input image
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Initialize the AprilTag detector
    detector = apriltag.Detector()

    # Perform detection
    detections = detector.detect(gray_image)

    # List to store expanded corner coordinates
    expanded_corners_list = []

    # Expand margin around the tags
    expansion_margin = 10  # Change this value to control how much to expand

    # Draw detections on the original image
    for detection in detections:
        # Extract the corners of the detection
        corners = np.array(detection.corners, dtype=np.int32)

        # Calculate the center of the tag
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])

        # Expand each corner outward
        expanded_corners = []
        for corner in corners:
            x, y = corner
            dx = x - center_x
            dy = y - center_y

            # Push corner outward by the expansion margin
            expanded_x = int(x + dx / np.linalg.norm([dx, dy]) * expansion_margin)
            expanded_y = int(y + dy / np.linalg.norm([dx, dy]) * expansion_margin)
            expanded_corners.append([expanded_x, expanded_y])

        # Append expanded corners as a polygon
        expanded_corners = np.array(expanded_corners, dtype=np.int32)
        expanded_corners_list.append(expanded_corners)



    return detections, input_image, expanded_corners_list

# Call the function
try:
    detections, detection_image, expanded_corners = process_apriltag_detection(IMAGE_PATH)  # Unpack all three values
    print(f"Detected {len(detections)} AprilTags.")

    # Print the corner coordinates for each detected tag
    for i, corners in enumerate(expanded_corners):
        print(f"AprilTag {i + 1} corners: {expanded_corners}")
    mask = create_mask_from_polygons(detection_image.shape,expanded_corners)
    inpainted_image = cv2.inpaint(detection_image, mask, inpaintRadius=60, flags=cv2.INPAINT_TELEA)
    cv2.imwrite('image_without_apriltags.jpg', inpainted_image)

    # Display the detection results
    cv2.imshow("Detection Results", inpainted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except FileNotFoundError as e:
    print(e)
