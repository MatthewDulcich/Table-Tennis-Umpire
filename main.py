import argparse
import os
import cv2
import numpy as np
import multiprocessing as mp
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

OUTPUT_FPS = 30  # Desired FPS for the output video


def process_frame_chunk(chunk):
    """
    Processes a chunk of frames by performing pose estimation and tracking.

    Parameters:
        chunk (list): A list of frames to process.

    Returns:
        list: A list of processed frames.
    """
    processed_frames = []
    for frame in chunk:
        processed_frame = run_pose_tracking(frame, detector, tracker, pose_model, feature_model)
        processed_frames.append(processed_frame)
    return processed_frames


def split_video_into_chunks(video_path, num_chunks):
    """
    Splits a video into chunks of frames.

    Parameters:
        video_path (str): Path to the video file.
        num_chunks (int): Number of chunks to split the video into.

    Returns:
        list: A list of frame chunks.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chunk_size = total_frames // num_chunks

    chunks = []
    for i in range(num_chunks):
        chunk = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            chunk.append(frame)
        chunks.append(chunk)

    # Add remaining frames to the last chunk
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        chunks[-1].append(frame)

    cap.release()
    return chunks


def combine_chunks_to_video(chunks, output_file, frame_width, frame_height, fps):
    """
    Combines processed frame chunks into a single video.

    Parameters:
        chunks (list): A list of processed frame chunks.
        output_file (str): Path to the output video file.
        frame_width (int): Width of the video frames.
        frame_height (int): Height of the video frames.
        fps (float): Frame rate of the output video.
    """
    video_writer = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    for chunk in chunks:
        for frame in chunk:
            video_writer.write(frame)

    video_writer.release()


def get_unique_filename(output_dir, base_name, extension):
    """
    Generates a unique file name by incrementing a counter if the file already exists.

    Parameters:
        output_dir (str): The directory where the file will be saved.
        base_name (str): The base name of the file (without extension).
        extension (str): The file extension (e.g., '.mp4').

    Returns:
        str: A unique file name with the given base name and extension.
    """
    counter = 1
    while True:
        file_name = f"{base_name}_{counter}{extension}"
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1


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

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        print(f"[INFO] Video properties: {frame_width}x{frame_height} at {fps} FPS")

        # Split the video into chunks
        max_processes = min(4, mp.cpu_count())  # Limit to 4 processes or the number of CPU cores
        print(f"[INFO] Using up to {max_processes} processes for multiprocessing...")
        chunks = split_video_into_chunks(video_path, max_processes)

        # Process chunks in parallel
        print("[INFO] Processing chunks in parallel...")
        with mp.Pool(processes=max_processes) as pool:
            processed_chunks = pool.map(process_frame_chunk, chunks)

        # Combine processed chunks into a single video
        output_dir = 'output_videos'
        os.makedirs(output_dir, exist_ok=True)
        output_file = get_unique_filename(output_dir, "processed_output", ".mp4")
        print(f"[INFO] Combining processed chunks into {output_file}...")
        combine_chunks_to_video(processed_chunks, output_file, frame_width, frame_height, fps)

        print(f"[INFO] Processed video saved to: {output_file}")

    else:
        # Webcam mode (unchanged)
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
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or OUTPUT_FPS  # Fallback to OUTPUT_FPS if FPS is unavailable

            # Initialize the video writer
            output_dir = 'output_videos'
            os.makedirs(output_dir, exist_ok=True)
            output_file = get_unique_filename(output_dir, "webcam_output", ".mp4")
            video_writer = cv2.VideoWriter(
                output_file,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )

            print("[INFO] Press 'q' to quit.")

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("[ERROR] Failed to read frame from camera.")
                        break

                    # Process the frame (e.g., pose estimation)
                    processed_frame = run_pose_tracking(frame, detector, tracker, pose_model, feature_model)

                    # Display the processed frame
                    cv2.imshow("Processed Frame", processed_frame)

                    # Write the processed frame to the video file
                    video_writer.write(processed_frame)

                    # Handle quitting
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                # Release resources
                cap.release()
                video_writer.release()
                cv2.destroyAllWindows()

                print(f"[INFO] Processed video saved to: {output_file}")

        except ValueError:
            print("Invalid input. Exiting.")


if __name__ == "__main__":
    main()
