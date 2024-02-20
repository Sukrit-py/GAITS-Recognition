import math
import cv2
import mediapipe as mp

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

        # Calculate the midpoint between left and right hip keypoints
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        midpoint = (left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2

        # Calculate the angle between the line connecting the left and right hip keypoints and the horizontal axis
        angle = math.degrees(math.atan2(right_hip.y - left_hip.y, right_hip.x - left_hip.x))

        # Calculate the length of the line
        line_length = 200

        # Calculate the endpoints of the line
        line_end_x = midpoint[0] + line_length * math.cos(math.radians(angle))
        line_end_y = midpoint[1] + line_length * math.sin(math.radians(angle))

        # Draw the line as a series of line segments
        line_points = [(int(midpoint[0]), int(midpoint[1])), (int(line_end_x), int(line_end_y))]
        for i in range(0, len(line_points), 2):
            if i % 2 == 0:
                cv2.line(frame, line_points[i], line_points[i + 1], (0, 255, 0), 2)

        # Draw a circle at the midpoint
        cv2.circle(frame, (int(midpoint[0]), int(midpoint[1])), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    # Check if the user has pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit_loop = True

cap.release()
cv2.destroyAllWindows()
