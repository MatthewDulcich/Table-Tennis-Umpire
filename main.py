from read_webcam_stream import list_available_cameras, launch_webcam
from sorted_pose_detection import DeepSORT, PoseModel, run_pose_tracking
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV3Small
import tensorflow as tf
import cv2
import numpy as np
from typing import Any
import os

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

OUTPUT_FPS = 15  # Desired FPS for the output video

def process_frame(frame: np.ndarray, video_writer: cv2.VideoWriter) -> np.ndarray:
    """
    Processes a single frame by performing pose estimation and tracking,
    displays it, and writes it to the video file.

    Parameters:
        frame (numpy.ndarray): The input frame from the webcam.
        video_writer (cv2.VideoWriter): The video writer object to save the frame.

    Returns:
        numpy.ndarray: The processed frame with results drawn.
    """
    # Process the frame
    processed_frame = run_pose_tracking(frame, detector, tracker, pose_model, feature_model)

    # Display the processed frame
    cv2.imshow("Processed Frame", processed_frame)

    # Write the frame to the video file
    video_writer.write(processed_frame)

    return processed_frame

def main() -> None:
    """
    Main function to list available cameras, launch the selected webcam,
    and save the processed video to a file.
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
            return

        # Open the selected camera
        cap = cv2.VideoCapture(cameras[selected_index])
        if not cap.isOpened():
            print("Failed to open selected camera.")
            return

        # Get the frame width, height, and FPS from the camera
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # Detect the camera's FPS
        print(f"[INFO] Camera FPS: {fps}")

        # Ensure the output directory exists
        output_dir = 'output_videos'
        os.makedirs(output_dir, exist_ok=True)

        # Generate a unique filename by incrementing a counter
        counter = 1
        while True:
            output_file = os.path.join(output_dir, f'output_{counter}.mp4')
            if not os.path.exists(output_file):
                break
            counter += 1

        # Initialize the video writer with the desired output FPS
        video_writer = cv2.VideoWriter(
            output_file,  # Unique output file name
            cv2.VideoWriter_fourcc(*'mp4v'),  # Codec (e.g., 'mp4v' for MP4)
            OUTPUT_FPS,  # Desired frame rate for the output video
            (frame_width, frame_height)  # Frame size
        )

        print("[INFO] Press 'q' to quit.")

        try:
            while True:
                # Read a frame from the camera
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera.")
                    break

                # Process the frame and save it to the video
                process_frame(frame, video_writer)

                # Handle quitting
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Release resources
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()

    except ValueError:
        print("Invalid input. Exiting.")

if __name__ == "__main__":
    main()
