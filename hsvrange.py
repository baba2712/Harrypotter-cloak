import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Allow the webcam to warm up
time.sleep(2)

# Capture the initial background
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)

# Define the HSV range for the fabric
lower_color = np.array([25, 40, 100])  # Lower bound of the HSV range
upper_color = np.array([32, 255, 200])  # Upper bound of the HSV range

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally to simulate a mirror image
    frame = np.flip(frame, axis=1)

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to detect the chosen cloak color
    mask1 = cv2.inRange(hsv, lower_color, upper_color)

    # Refine the mask using morphological operations to reduce noise
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Create an inverse mask to segment out the cloak from the frame
    mask2 = cv2.bitwise_not(mask1)

    # Segment the cloak area from the background
    cloak_area = cv2.bitwise_and(background, background, mask=mask1)

    # Segment the non-cloak area from the current frame
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask2)

    # Combine both cloak and non-cloak areas
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    # Display the final output
    cv2.imshow("Invisibility Cloak", final_output)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
