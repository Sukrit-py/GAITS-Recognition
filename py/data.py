import cv2
import mediapipe as mp
import os

def resize_frame(frame, target_width, target_height):
    """Resize the frame to the target width and height while maintaining the aspect ratio."""
    height, width, _ = frame.shape
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame

def compute_keypoint_stability(initial_keypoints, current_keypoints):
    """Compute the stability of key points based on their initial and current positions."""
    stability = {}
    for i in range(len(initial_keypoints.landmark)):
        initial_x = initial_keypoints.landmark[i].x
        initial_y = initial_keypoints.landmark[i].y
        current_x = current_keypoints.landmark[i].x
        current_y = current_keypoints.landmark[i].y
        distance = ((initial_x - current_x) ** 2 + (initial_y - current_y) ** 2) ** 0.5
        stability[i] = distance
    return stability

def annotate_skeleton(video_path, output_video_path, target_width=640, target_height=480):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (target_width, target_height))

    initial_keypoints = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to landscape format
        resized_frame = resize_frame(frame, target_width, target_height)

        # Convert the resized image to RGB
        image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe
        results = pose.process(image_rgb)

        # Draw the pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(resized_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Compute stability if initial keypoints are available
            if initial_keypoints is None:
                initial_keypoints = results.pose_landmarks
            else:
                current_stability = compute_keypoint_stability(initial_keypoints, results.pose_landmarks)
                print("Stability:", current_stability)

        # Write the annotated frame to the output video
        out.write(resized_frame)

        # Display the annotated frame
        cv2.imshow('Annotated Video', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Close MediaPipe Pose
    pose.close()

if __name__ == "__main__":
    # Input video file path
    video_path = "Video for Data/VID20240319153404.mp4"

    # Output video file path (saved in the parent directory)
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    output_video_path = os.path.join(parent_directory, "landscape_annotated_video.mp4")

    # Annotate skeleton and save the annotated video in landscape format
    annotate_skeleton(video_path, output_video_path)
