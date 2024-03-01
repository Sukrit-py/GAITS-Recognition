import cv2
import torch
import numpy as np



def preprocess_frame(frame):

    resized_frame = cv2.resize(frame, (720, 360))
    return resized_frame


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read frame.")
        break
    preprocessed_frame = preprocess_frame(frame)
    cv2.imshow('Video', preprocessed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
