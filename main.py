import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Reduce resolution to improve performance (e.g., 640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Allow the webcam to warm up
time.sleep(2)

# Capture the initial background (this will change dynamically)
ret, background = cap.read()
background = cv2.resize(background, (640, 480))  # Reduce background size for faster processing
background = np.flip(background, axis=1)  # Flip for consistency

# Function to capture and resize dynamic background
def update_background():
    ret, bg = cap.read()
    if ret:
        bg = cv2.resize(bg, (640, 480))  # Reduce size for faster processing
        bg = np.flip(bg, axis=1)
    return bg

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame for consistency and reduce its resolution
    frame = np.flip(frame, axis=1)
    frame = cv2.resize(frame, (640, 480))

    # Update the background dynamically
    background = update_background()

    # Convert the frame from BGR to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV range for detecting blue color (adjust for lighter blue)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask to detect the blue color in the cloak
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    # Refine the mask with limited morphological operations for speed
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Invert the mask to get everything except the blue cloak
    mask2 = cv2.bitwise_not(mask1)

    # Segment out the blue cloak part and replace it with the background
    cloak_part = cv2.bitwise_and(background, background, mask=mask1)

    # Segment out everything else except the cloak
    non_cloak_part = cv2.bitwise_and(frame, frame, mask=mask2)

    # Combine both parts to get the final output
    final_output = cv2.addWeighted(cloak_part, 1, non_cloak_part, 1, 0)

    # Display the final result
    cv2.imshow("Invisibility Cloak", final_output)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
