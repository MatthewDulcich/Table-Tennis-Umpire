from read_webcam_stream import list_available_cameras, launch_webcam
from sorted_pose_detection import DeepSORT, PoseModel, run_pose_tracking
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV3Small
import tensorflow as tf
import cv2
import numpy as np
from typing import Any

# Initialize required models globally once
pose_model = PoseModel()
detector = YOLO("models/yolov5nu.pt")
tracker = DeepSORT(max_age=30)

base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet',
    include_preprocessing=False
)
x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)
feature_model = tf.keras.Model(inputs=base_model.input, outputs=x)

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Processes a single frame by performing pose estimation and tracking.

    Parameters:
        frame (numpy.ndarray): The input frame from the webcam.

    Returns:
        numpy.ndarray: The processed frame with results drawn.
    """
    return run_pose_tracking(frame, detector, tracker, pose_model, feature_model)

def main() -> None:
    """
    Main function to list available cameras and launch the selected webcam.
    """
    # List all available cameras
    cameras = list_available_cameras()

    if not cameras:
        print("No cameras found.")
        return

    print("Available cameras:")
    for i, cam in enumerate(cameras):
        print(f"{i}: Camera Index {cam}")

    # Allow the user to select a camera
    try:
        selected_index = int(input("Select a camera index from the list above: "))
        if selected_index < 0 or selected_index >= len(cameras):
            print("Invalid selection.")
        else:
            # Launch the webcam with the processing function
            launch_webcam(camera_index=cameras[selected_index], frame_callback=process_frame)
    except ValueError:
        print("Invalid input. Exiting.")

if __name__ == "__main__":
    main()
