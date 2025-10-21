import os
import numpy as np
import cv2

# ==== CONFIG ====
input_folder = "/media/karan-sankla/KINGSTON/rgbddata_clenser"
output_color_folder = "/home/karan-sankla/Testbench/objectfiles/Images"


# Create output directories if they don't exist
os.makedirs(output_color_folder, exist_ok=True)


# ==== PROCESS FILES ====
for filename in os.listdir(input_folder):
    if not filename.endswith(".npy"):
        continue

    npy_path = os.path.join(input_folder, filename)
    data = np.load(npy_path)

    if "_color" in filename:
        out_path = os.path.join(output_color_folder, filename.replace(".npy", ".png"))
        cv2.imwrite(out_path, data)  # No conversion needed
        print(f"Saved COLOR image: {out_path}")
