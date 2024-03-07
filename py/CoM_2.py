import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to preprocess frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (960, 720))
    return resized_frame

def calculate_body_weights(body_weight):
# Define weight percentages for different body parts
    body_weights = {
        "head_neck_trunk": f"0.47 * {body_weight} + 12.0 ± 6.4",
        #"torso": 42.7,
        #"right_arm": 6.3,
        #"left_arm": 6.1,
        "right_upper_arm": f"0.04 * {body_weight} - 1.45 ± 0.5",
        "left_upper_arm": f"0.04 * {body_weight} - 1.45 ± 0.5",
        #"forearm_hand_right": 3.2,
        #"forearm_hand_left": 3.0,
        "forearm_right": f"0.02 * {body_weight} - 0.25 ± 0.5",
        "forearm_left": f"0.02 * {body_weight} - 0.25 ± 0.5",
        "hand_right": f"0.005 * {body_weight} + 0.35 ± 0.2",
        "hand_left": f"0.005 * {body_weight} + 0.35 ± 0.2",
        #"right_leg": 18.2,
        #"left_leg": 19.1,
        "thigh_right": f"0.09 * {body_weight} + 1.6 ± 1.8",
        "thigh_left": f"0.09 * {body_weight} + 1.6 ± 1.8",
        #"calf_foot_right": 7.1,
        #"calf_foot_left": 7.0,
        "calf_right": f"0.055 * {body_weight} - 0.95 ± 0.8",
        "calf_left": f"0.055 * {body_weight} - 0.95 ± 0.8",
        "foot_right": f"0.01 * {body_weight} + 0.75 ± 0.3",
        "foot_left": f"0.01 * {body_weight} + 0.75 ± 0.3"
    }

    return body_weights

# Function to detect specific body parts and calculate their center of mass
def detect_body_parts(frame, results, body_weights):
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        image_height, image_width, _ = frame.shape

        # Define dictionary to store body part landmarks and center of mass
        body_parts = {
            "head_neck_trunk": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24],
            #"torso": [11, 12, 23, 24],
            #"right_arm": [11, 13, 15],
            #"left_arm": [12, 14, 16],
            "right_upper_arm": [11, 13],
            "left_upper_arm": [12, 14],
            #"forearm_hand_right": [13, 15, 17, 19, 21],
            #"forearm_hand_left": [14, 16, 18, 20, 22],
            "forearm_right": [13, 15, 17],
            "forearm_left": [14, 16, 18],
            "hand_right": [15, 17, 19, 21],
            "hand_left": [16, 18, 20, 22],
            #"right_leg": [23, 25, 27, 29, 31],
            #"left_leg": [24, 26, 28, 30, 32],
            "thigh_right": [23, 25, 27],
            "thigh_left": [24, 26, 28],
            #"calf_foot_right": [25, 27, 29, 31],
            #"calf_foot_left": [26, 28, 30, 32],
            "calf_right": [25, 27, 29],
            "calf_left": [26, 28, 30],
            "foot_right": [29, 31],
            "foot_left": [30, 32]
        }

        total_com_x = 0
        total_com_y = 0
        total_weight = 0

    
        # Calculate center of mass for each body part
        # Calculate center of mass for each body part
        for part, landmarks_idx in body_parts.items():
            part_com_x = 0
            part_com_y = 0
            for idx in landmarks_idx:
                part_com_x += landmarks[idx].x * image_width
                part_com_y += landmarks[idx].y * image_height
            part_com_x /= len(landmarks_idx)
            part_com_y /= len(landmarks_idx)

            # Calculate weight for the body part
            if part == "head_neck_trunk":
                weight = 0.47 * body_weight + 12.0
            elif part in ["right_upper_arm", "left_upper_arm"]:
                weight = 0.04 * body_weight - 1.45
            elif part in ["forearm_right", "forearm_left"]:
                weight = 0.02 * body_weight - 0.25
            elif part in ["hand_right", "hand_left"]:
                weight = 0.005 * body_weight + 0.35
            elif part in ["thigh_right", "thigh_left"]:
                weight = 0.09 * body_weight + 1.6
            elif part in ["calf_right", "calf_left"]:
                weight = 0.055 * body_weight - 0.95
            elif part in ["foot_right", "foot_left"]:
                weight = 0.01 * body_weight + 0.75
            else:
                weight = 0  # Default weight for other body parts

            # Weight the center of mass by the assigned weight
            total_com_x += part_com_x * weight
            total_com_y += part_com_y * weight
            total_weight += weight

        # Normalize the total center of mass coordinates
        total_com_x /= total_weight
        total_com_y /= total_weight

        # Draw a point at the total center of mass
        cv2.circle(frame, (int(total_com_x), int(total_com_y)), 5, (0, 0, 255), -1)

# Get user input for body weight
body_weight = float(input("Enter your body weight (in kg): "))

# Calculate body segment weights based on user input
body_weights = calculate_body_weights(body_weight)

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
    detect_body_parts(preprocessed_frame, results, body_weights)

    # Display annotated frame
    cv2.imshow('Video', preprocessed_frame)

    # Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
