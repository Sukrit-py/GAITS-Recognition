import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw the keypoints on the frame
        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the weighted center of mass for upper body
            upper_body_keypoints = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]

            # Assign weights (arbitrary for demonstration purposes)
            weights = [1, 1, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125]

            # Calculate the center of mass
            center_of_mass_x = sum(weight * results.pose_landmarks.landmark[keypoint.value].x for keypoint, weight in zip(upper_body_keypoints, weights))
            center_of_mass_y = sum(weight * results.pose_landmarks.landmark[keypoint.value].y for keypoint, weight in zip(upper_body_keypoints, weights))

            # Calculate the center of gravity for the entire body
            all_body_keypoints = [landmark for landmark in mp_pose.PoseLandmark]
            cog_weights = [1] * len(all_body_keypoints)  # Equal weights for all body parts
            center_of_gravity_x = sum(weight * results.pose_landmarks.landmark[keypoint.value].x for keypoint, weight in zip(all_body_keypoints, cog_weights))
            center_of_gravity_y = sum(weight * results.pose_landmarks.landmark[keypoint.value].y for keypoint, weight in zip(all_body_keypoints, cog_weights))

            # Display the center of mass
            center_of_mass = (int(center_of_mass_x * frame.shape[1]), int(center_of_mass_y * frame.shape[0]))
            cv2.circle(frame, center_of_mass, 5, (255, 0, 0), -1)

            # Display the center of gravity
            center_of_gravity = (int(center_of_gravity_x * frame.shape[1]), int(center_of_gravity_y * frame.shape[0]))
            cv2.circle(frame, center_of_gravity, 5, (0, 255, 0), -1)

            # Display the center of mass and center of gravity coordinates
            cv2.putText(frame, f"Center of Mass: {center_of_mass}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Center of Gravity: {center_of_gravity}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Center of Mass and Gravity', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
