from process_image import CirclesDetection
import cv2
import numpy as np


# Load the pre-trained ONNX model for circle detection
model_path = "coin.onnx"  
detection = CirclesDetection(model_path)

# Specify the path to the image to be processed
img_path = "test.png"

# Use the CirclesDetection object to detect circles in the image and obtain a mask
mask, circles = detection.detect_circles(image_path=img_path)

# Load the original image for display purposes
img = cv2.imread(img_path)

# If circles were detected, draw them on the original image with green color and thickness of 3 pixels
if circles is not None:
    for circle in circles:
        x, y, r = map(int, circle)
        cv2.circle(img, (x, y), r, (0, 255, 0), 3)

# If a mask was generated, convert it to BGR color space. Otherwise, create a black mask with the same dimensions as the original image
if mask is not None:
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
else:
    mask = np.zeros_like(img)

# Combine the original image and the mask side by side for display purposes
combined = np.hstack((img, mask))

# Display the results
cv2.imshow("Image + Mask", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
