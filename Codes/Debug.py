import numpy as np
import cv2

npy_path = "/home/karan-sankla/Testbench/objectfiles/woodblock/woodblock0_color.npy"  # put one file here
data = np.load(npy_path)
print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Min:", data.min(), "Max:", data.max())

# Show without any conversion
cv2.imshow("Raw data (as BGR)", data)
cv2.waitKey(0)

# Show with RGB→BGR conversion
bgr_img = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
cv2.imshow("RGB to BGR", bgr_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
