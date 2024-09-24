import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)

ret, background = cap.read()
background = cv2.resize(background, (640, 480))  # Reduce background size for faster processing
background = np.flip(background, axis=1)  # Flip for consistency

def update_background():
    ret, bg = cap.read()
    if ret:
        bg = cv2.resize(bg, (640, 480))  
        bg = np.flip(bg, axis=1)
    return bg

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = np.flip(frame, axis=1)
    frame = cv2.resize(frame, (640, 480))

    background = update_background()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    cloak_part = cv2.bitwise_and(background, background, mask=mask1)

    non_cloak_part = cv2.bitwise_and(frame, frame, mask=mask2)

    final_output = cv2.addWeighted(cloak_part, 1, non_cloak_part, 1, 0)

    cv2.imshow("Invisibility Cloak", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
