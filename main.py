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
from queue import Queue
from threading import Thread
import atexit
from tqdm import tqdm
import logging
from table_detection import load_video, select_corners, process_video  # Import table detection functions

logging.getLogger("ultralytics").setLevel(logging.WARNING)

cv2.setNumThreads(0)

OUTPUT_FPS = 15  # Desired FPS for the output video

# Global references for safe cleanup
cap = None
writer = None
interrupted = False

def cleanup():
    global cap, writer
    if cap is not None and cap.isOpened():
        cap.release()
        print("[INFO] Camera released.")
    if writer is not None:
        writer.release()
        print("[INFO] Video writer released.")

atexit.register(cleanup)

class VideoStreamReader:
    def __init__(self, video_path, queue_size=32):
        self.cap = cv2.VideoCapture(video_path)
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    break
                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def more(self):
        return not self.queue.empty()

    def stop(self):
        self.stopped = True
        self.cap.release()


def stream_process_and_write(video_path, output_path, detector, tracker, pose_model, feature_model, batch_size=8, quad_pts=None):
    global cap, writer
    print(f"[DEBUG] Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("Key Capture Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Key Capture Window", np.zeros((100, 400, 3), dtype=np.uint8))
    print("[INFO] Press Ctrl+G in the OpenCV window to stop processing gracefully.")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

    print(f"[DEBUG] Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}, Total Frames: {total_frames}")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),  # Use H.264 codec
        fps,
        (frame_width, frame_height)
    )

    if not writer.isOpened():
        print("[ERROR] Failed to open video writer. Check codec or output path.")
        cap.release()
        return

    print("[INFO] Streaming frame-by-frame...")
    frame_count = 0
    batch = []

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

    try:
        while not interrupted:
            ret, frame = cap.read()
            if not ret:
                print("[DEBUG] End of video or failed to read frame.")
                break

            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                print(f"[ERROR] Frame dimensions do not match: expected ({frame_height}, {frame_width}), got ({frame.shape[0]}, {frame.shape[1]})")
                break

            # If table corners are provided, draw the quadrilateral
            if quad_pts is not None:
                quad_int = quad_pts.astype(int)
                cv2.polylines(frame, [quad_int.reshape(-1, 2)], isClosed=True, color=(255, 0, 0), thickness=2)

            cv2.imshow("Key Capture Window", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 7:  # ASCII code for Ctrl+G
                print("[INFO] Ctrl+G detected. Stopping gracefully...")
                break

            batch.append(frame)

            if len(batch) == batch_size:
                processed_batch = [run_pose_tracking(f, detector, tracker, pose_model, feature_model) for f in batch]
                for processed in processed_batch:
                    if processed.shape[:2] != (frame_height, frame_width):
                        processed = cv2.resize(processed, (frame_width, frame_height))
                    writer.write(processed)
                    frame_count += 1
                    progress_bar.update(1)  # Update progress bar
                batch = []

        # Process remaining frames in the batch
        if batch:
            processed_batch = [run_pose_tracking(f, detector, tracker, pose_model, feature_model) for f in batch]
            for processed in processed_batch:
                if processed.shape[:2] != (frame_height, frame_width):
                    processed = cv2.resize(processed, (frame_width, frame_height))
                writer.write(processed)
                frame_count += 1
                progress_bar.update(1)  # Update progress bar

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Processing any remaining frames before exit.")

    finally:
        progress_bar.close()  # Close the progress bar
        print("[DEBUG] Releasing resources.")
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        if frame_count == 0:
            print("[WARNING] No frames were written to the video. Output file may be empty.")
        else:
            print(f"[INFO] Processed video saved to: {output_path}")


def get_unique_filename(output_dir, base_name, extension):
    counter = 1
    while True:
        file_name = f"{base_name}_{counter}{extension}"
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1


def main():
    parser = argparse.ArgumentParser(description="Process webcam or video input.")
    parser.add_argument("--video", action="store_true", help="Process a video file instead of the webcam.")
    args = parser.parse_args()

    if args.video:
        Tk().withdraw()  # Hide the root tkinter window
        video_path = askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])

        if not video_path:
            print("No video file selected. Exiting.")
            return

        # Load video and select table corners
        cap, first_frame = load_video(video_path)
        if cap is None or first_frame is None:
            return
        quad_pts = select_corners(first_frame)
        if quad_pts is None:
            return

        detector = YOLO("models/yolov5nu.pt", verbose=False)
        pose_model = PoseModel()
        tracker = DeepSORT(max_age=5760)

        base_model = MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling='avg',
            weights='imagenet',
            include_preprocessing=False
        )
        x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)
        feature_model = tf.keras.Model(inputs=base_model.input, outputs=x)

        output_dir = 'output_videos'
        os.makedirs(output_dir, exist_ok=True)
        output_file = get_unique_filename(output_dir, "processed_output", ".mp4")
        stream_process_and_write(video_path, output_file, detector, tracker, pose_model, feature_model, quad_pts=quad_pts)
        print(f"[INFO] Processed video saved to: {output_file}")

    else:
        # Webcam mode
        cameras = list_available_cameras()

        if not cameras:
            print("No cameras found.")
            return

        print("Available cameras:")
        for i, cam in enumerate(cameras):
            print(f"{i}: Camera Index {cam}")

        try:
            selected_index = int(input("Select a camera index from the list above: "))
            if selected_index < 0 or selected_index >= len(cameras):
                print("Invalid selection.")
                return

            cap = cv2.VideoCapture(cameras[selected_index])
            if not cap.isOpened():
                print("Failed to open selected camera.")
                return

            # Select table corners for webcam
            ret, first_frame = cap.read()
            if not ret:
                print("Failed to read from webcam.")
                return
            quad_pts = select_corners(first_frame)
            if quad_pts is None:
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or OUTPUT_FPS

            detector = YOLO("models/yolov5nu.pt")
            pose_model = PoseModel()
            tracker = DeepSORT(max_age=5760)
            base_model = MobileNetV3Small(
                input_shape=(224, 224, 3),
                include_top=False,
                pooling='avg',
                weights='imagenet',
                include_preprocessing=False
            )
            x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)
            feature_model = tf.keras.Model(inputs=base_model.input, outputs=x)

            output_dir = 'output_videos'
            os.makedirs(output_dir, exist_ok=True)
            output_file = get_unique_filename(output_dir, "webcam_output", ".mp4")
            video_writer = cv2.VideoWriter(
                output_file,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )

            print("[INFO] Press Ctrl+G in the webcam preview window to stop.")

            cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)
            interrupted = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera.")
                    break

                # Draw table quadrilateral
                if quad_pts is not None:
                    quad_int = quad_pts.astype(int)
                    cv2.polylines(frame, [quad_int.reshape(-1, 2)], isClosed=True, color=(255, 0, 0), thickness=2)

                processed_frame = run_pose_tracking(frame, detector, tracker, pose_model, feature_model)
                cv2.imshow("Processed Frame", processed_frame)
                video_writer.write(processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 7:  # Ctrl+G
                    print("[INFO] Ctrl+G detected. Stopping webcam recording...")
                    break

            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Processed video saved to: {output_file}")

        except ValueError:
            print("Invalid input. Exiting.")


if __name__ == "__main__":
    main()
