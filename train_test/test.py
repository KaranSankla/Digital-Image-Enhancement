# Save the complete script as 'ycb_yolo_complete.py'
from ycb_yolo_generator import YCBToYOLO

# Your paths
BASE_PATH = "/home/karan-sankla/Testbench/objectfiles/dataset"
converter = YCBToYOLO(f"{BASE_PATH}/models")

# Camera intrinsics
fx = fy = 570.34222412109381
cx = 319.5
cy = 239.5

# Process everything
converter.batch_process_directory(
    BASE_PATH, fx, fy, cx, cy, f"{BASE_PATH}/output"
)

# Save class names
converter.save_class_names(f"{BASE_PATH}/output/classes.txt")