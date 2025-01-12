import cv2
import numpy as np
import pupil_apriltags as apriltag

# Constants
IMAGE_PATH = "tomatocan0_color.png"

def process_apriltag_detection(image_path):
    # Read the input image
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Initialize the detector
    detector = apriltag.Detector()

    # Perform detection
    detections = detector.detect(gray_image)

    # Draw detections on the original image
    for detection in detections:
        # Extract the corners of the detection
        corners = np.array(detection.corners, dtype=np.int32)
        for i in range(4):
            cv2.line(
                input_image,
                tuple(corners[i]),
                tuple(corners[(i + 1) % 4]),
                (0, 255, 0),
                2
            )

    return detections, input_image

# Call the function
try:
    detections, detection_image = process_apriltag_detection(IMAGE_PATH)
    print(f"Detected {len(detections)} AprilTags.")
    cv2.imshow("Detection Results", detection_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except FileNotFoundError as e:
    print(e)
