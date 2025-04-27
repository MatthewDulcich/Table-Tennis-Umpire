import argparse
import os
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from read_webcam_stream import list_available_cameras
from sorted_pose_detection import DeepSORT, PoseModel, run_pose_tracking
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV3Small
import tensorflow as tf

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


def process_frame(frame: np.ndarray, video_writer: cv2.VideoWriter, show_live: bool = True) -> None:
    """
    Processes a single frame by performing pose estimation and tracking,
    optionally displays it, and writes it to the video file.

    Parameters:
        frame (numpy.ndarray): The input frame from the webcam or video.
        video_writer (cv2.VideoWriter): The video writer object to save the frame.
        show_live (bool): Whether to display the frame live.
    """
    # Process the frame
    processed_frame = run_pose_tracking(frame, detector, tracker, pose_model, feature_model)

    # Optionally display the processed frame
    if show_live:
        cv2.imshow("Processed Frame", processed_frame)

    # Write the frame to the video file
    video_writer.write(processed_frame)


def main():
    """
    Main function to process either a webcam stream or a video file.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process webcam or video input.")
    parser.add_argument("--video", action="store_true", help="Process a video file instead of the webcam.")
    args = parser.parse_args()

    if args.video:
        # Use tkinter to open a file dialog for video selection
        Tk().withdraw()  # Hide the root tkinter window
        video_path = askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])

        if not video_path:
            print("No video file selected. Exiting.")
            return

        # Open the selected video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video file.")
            return

        print(f"[INFO] Processing video: {video_path}")

    else:
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

        except ValueError:
            print("Invalid input. Exiting.")
            return

    # Get the frame width, height, and FPS from the input source
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if args.video else OUTPUT_FPS
    print(f"[INFO] Input FPS: {fps}")

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

    # Initialize the video writer
    video_writer = cv2.VideoWriter(
        output_file,  # Unique output file name
        cv2.VideoWriter_fourcc(*'mp4v'),  # Codec (e.g., 'mp4v' for MP4)
        fps,  # Frame rate
        (frame_width, frame_height)  # Frame size
    )

    print("[INFO] Press 'q' to quit (only for webcam mode).")

    try:
        while True:
            # Read a frame from the input source
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video or failed to read frame.")
                break

            # Process the frame and save it to the video
            process_frame(frame, video_writer, show_live=not args.video)

            # Handle quitting (only for webcam mode)
            if not args.video and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        print(f"[INFO] Processed video saved to: {output_file}")


if __name__ == "__main__":
    main()
