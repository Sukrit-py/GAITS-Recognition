import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define weight percentages for different body parts (example percentages)
weight_percentages = {
    "head_neck_trunk": 0.47,
    "right_upper_arm": 0.04,
    "left_upper_arm": 0.04,
    "forearm_right": 0.02,
    "forearm_left": 0.02,
    "hand_right": 0.005,
    "hand_left": 0.005,
    "thigh_right": 0.09,
    "thigh_left": 0.09,
    "calf_right": 0.055,
    "calf_left": 0.055,
    "foot_right": 0.01,
    "foot_left": 0.01
}

# Function to preprocess frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (960, 720))
    return resized_frame

# Function to detect specific body parts and calculate their center of gravity
def detect_body_parts(frame, results):
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        image_height, image_width, _ = frame.shape

        # Define dictionary to store body part landmarks
        body_parts = {
            "head_neck_trunk": [0, 1, 2, 3, 4],
            "right_upper_arm": [11, 13, 15],
            "left_upper_arm": [12, 14, 16],
            "forearm_right": [13, 15, 17],
            "forearm_left": [14, 16, 18],
            "hand_right": [15, 17, 19, 21],
            "hand_left": [16, 18, 20, 22],
            "thigh_right": [23, 25, 27],
            "thigh_left": [24, 26, 28],
            "calf_right": [25, 27, 29],
            "calf_left": [26, 28, 30],
            "foot_right": [29, 31],
            "foot_left": [30, 32]
        }

        total_cog_x = 0
        total_cog_y = 0
        total_weight = 0

        # Calculate center of gravity for each body part
        for part, landmarks_idx in body_parts.items():
            part_cog_x = 0
            part_cog_y = 0
            part_weight = weight_percentages.get(part, 0)
            for idx in landmarks_idx:
                part_cog_x += landmarks[idx].x * image_width
                part_cog_y += landmarks[idx].y * image_height
            part_cog_x /= len(landmarks_idx)
            part_cog_y /= len(landmarks_idx)

            # Weight the center of gravity by the assigned weight
            total_cog_x += part_cog_x * part_weight
            total_cog_y += part_cog_y * part_weight
            total_weight += part_weight

        # Normalize the total center of gravity coordinates
        total_cog_x /= total_weight
        total_cog_y /= total_weight

        # Draw a point at the total center of gravity
        cv2.circle(frame, (int(total_cog_x), int(total_cog_y)), 5, (0, 0, 255), -1)

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

    # Detect specific body parts and calculate their center of gravity
    detect_body_parts(preprocessed_frame, results)

    # Display annotated frame
    cv2.imshow('Video', preprocessed_frame)

    # Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
