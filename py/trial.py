import cv2
import numpy as np
import pyrealsense2 as rs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def setup_depth_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def extract_gait_features(frames):
    # Example: Extracting mean pixel intensity as a feature
    features = []
    for frame in frames:
        mean_intensity = np.mean(frame)
        features.append(mean_intensity)
    return features

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Main function
def main():
    # Video Acquisition Setup
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    # Feature Extraction
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Perform any preprocessing or feature extraction here
        # For example, you can resize the frame to a fixed size
        frame = cv2.resize(frame, (64, 64))
        
        # Example: Extracting mean pixel intensity as a feature
        mean_intensity = np.mean(frame)
        gait_features = [mean_intensity]

        # Display the frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
