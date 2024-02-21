import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw the keypoints on the frame
        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Get the coordinates of the hip joints
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y

        # Calculate the center of gravity
        center_of_gravity = (left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2

        # Mark the center of gravity
        cv2.circle(frame, (int(center_of_gravity[0] * frame.shape[1]), int(center_of_gravity[1] * frame.shape[0])), 5, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('MediaPipe Pose', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
