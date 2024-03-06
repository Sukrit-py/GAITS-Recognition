import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define weight percentages for different body parts
weight_percentages = {
    "head": 7.1,
    "torso": 42.7,
    #"right_arm": 6.3,
    #"left_arm": 6.1,
    "right_upper_arm": 3.1,
    "left_upper_arm": 3.6,
    #"forearm_hand_right": 3.2,
    #"forearm_hand_left": 3.0,
    "forearm_right": 2.3,
    "forearm_left": 2.2,
    "hand_right": 0.9,
    "hand_left": 0.8,
    #"right_leg": 18.2,
    #"left_leg": 19.1,
    "thigh_right": 11.0,
    "thigh_left": 12.1,
    #"calf_foot_right": 7.1,
    #"calf_foot_left": 7.0,
    "calf_right": 5.3,
    "calf_left": 5.2,
    "foot_right": 1.8,
    "foot_left": 1.8
}

# Function to preprocess frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (960, 720))
    return resized_frame

# Function to detect specific body parts and calculate their center of mass
def detect_body_parts(frame, results):
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        image_height, image_width, _ = frame.shape

        # Define dictionary to store body part landmarks and center of mass
        body_parts = {
            "head": [0, 1, 2, 3, 4],
            "torso": [11, 12, 23, 24],
            "right_arm": [11, 13, 15],
            "left_arm": [12, 14, 16],
            "right_upper_arm": [11, 13],
            "left_upper_arm": [12, 14],
            "forearm_hand_right": [13, 15, 17, 19, 21],
            "forearm_hand_left": [14, 16, 18, 20, 22],
            "forearm_right": [13, 15, 17],
            "forearm_left": [14, 16, 18],
            "hand_right": [15, 17, 19, 21],
            "hand_left": [16, 18, 20, 22],
            "right_leg": [23, 25, 27, 29, 31],
            "left_leg": [24, 26, 28, 30, 32],
            "thigh_right": [23, 25, 27],
            "thigh_left": [24, 26, 28],
            "calf_foot_right": [25, 27, 29, 31],
            "calf_foot_left": [26, 28, 30, 32],
            "calf_right": [25, 27, 29],
            "calf_left": [26, 28, 30],
            "foot_right": [29, 31],
            "foot_left": [30, 32]
        }

        total_com_x = 0
        total_com_y = 0

        # Calculate center of mass for each body part
        for part, landmarks_idx in body_parts.items():
            part_com_x = 0
            part_com_y = 0
            for idx in landmarks_idx:
                part_com_x += landmarks[idx].x * image_width
                part_com_y += landmarks[idx].y * image_height
            part_com_x /= len(landmarks_idx)
            part_com_y /= len(landmarks_idx)

            # Weight the center of mass by the assigned weight percentage
            weight_percentage = weight_percentages.get(part, 0)
            total_com_x += part_com_x * weight_percentage
            total_com_y += part_com_y * weight_percentage

        # Normalize the total center of mass coordinates
        total_com_x /= sum(weight_percentages.values())
        total_com_y /= sum(weight_percentages.values())

        # Draw a point at the total center of mass
        cv2.circle(frame, (int(total_com_x), int(total_com_y)), 5, (0, 0, 255), -1)

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Main loop
while True:
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Preprocess frame
    preprocessed_frame = preprocess_frame(frame)

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2RGB)

    # Detect pose
    results = pose.process(frame_rgb)

    # Detect specific body parts and calculate their center of mass
    detect_body_parts(preprocessed_frame, results)

    # Display annotated frame
    cv2.imshow('Video', preprocessed_frame)

    # Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
