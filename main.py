import argparse
import os
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from read_webcam_stream import list_available_cameras
from sorted_pose_detection import DeepSORT, PoseModel, run_pose_tracking
from train_models.ball_event_train import crop_centered_with_padding, extract_specific_frames
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


def stream_process_and_write(video_path, output_path, detector, tracker, pose_model, feature_model, batch_size=8, quad_pts=None, flow_state=None):
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

    # Load models once before the loop
    ball_track_model = tf.keras.models.load_model("models/tracknet_pre_model.keras")
    ball_event_model = tf.keras.models.load_model("models/ball_event_model.keras")
    target_size = (320, 220)
    scale_x = target_size[0] / frame_width
    scale_y = target_size[1] / frame_height

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    prev_gray = None
    if flow_state and flow_state.get("use_optical_flow", False):
        ret, first_frame = cap.read()
        if ret:
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    try:
        while not interrupted:
            ret, frame = cap.read()
            if not ret:
                print("[DEBUG] End of video or failed to read frame.")
                break

            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                print(f"[ERROR] Frame dimensions do not match: expected ({frame_height}, {frame_width}), got ({frame.shape[0]}, {frame.shape[1]})")
                break

            # --- Optical Flow Update ---
            if quad_pts is not None:
                if flow_state and flow_state.get("use_optical_flow", False) and prev_gray is not None:
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    new_quad, quad_status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, quad_pts, None, **lk_params)
                    if new_quad is not None and quad_status.sum() == 4:
                        quad_pts = new_quad
                    prev_gray = curr_gray

            # --- Pose Estimation ---
            processed_frame = run_pose_tracking(frame.copy(), detector, tracker, pose_model, feature_model)

            # --- Ball Tracking ---
            resized_frame = cv2.resize(frame, target_size)
            pred_pos = ball_track_model.predict(np.expand_dims(resized_frame, axis=0) / 255.0)
            orig_x = int(float(pred_pos[0][0]) / scale_x)
            orig_y = int(float(pred_pos[0][1]) / scale_y)

            # --- Ball Event Detection ---
            cropped_frame = crop_centered_with_padding(frame, (orig_x, orig_y), target_size)
            cropped_frame = cv2.resize(cropped_frame, target_size)
            pred_event = ball_event_model.predict(np.expand_dims(cropped_frame, axis=0))
            event = np.argmax(pred_event[0])
            event_label = {0: "bounce", 1: "net", 2: "empty_event"}[event]

            # --- Annotate Frame ---
            cv2.circle(processed_frame, (orig_x, orig_y), 5, (0, 255, 0), -1)
            processed_frame = place_event_in_frame(processed_frame, event_label, position=(10, 20))

            # --- Draw Quad ---
            if quad_pts is not None:
                quad_int = quad_pts.astype(int)
                cv2.polylines(processed_frame, [quad_int.reshape(-1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)

            # --- Output ---
            if processed_frame.shape[:2] != (frame_height, frame_width):
                processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

            writer.write(processed_frame)
            progress_bar.update(1)
            cv2.imshow("Processed Frame", processed_frame)

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == 7:  # Ctrl+G
                print("[INFO] Ctrl+G detected. Stopping gracefully...")
                break
            elif key == ord('s') and flow_state is not None:
                flow_state["use_optical_flow"] = not flow_state["use_optical_flow"]
                print(f"[INFO] Toggled mode: {'Optical Flow' if flow_state['use_optical_flow'] else 'Static Only'}")
                if flow_state["use_optical_flow"]:
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Processing any remaining frames before exit.")

    finally:
        progress_bar.close()
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

def downsample_frames(frames, frame_dims, target_size=(320, 220)):
    global scale_x, scale_y
    # Calculate scaling factors
    scale_x = target_size[0] / frame_dims[0]
    scale_y = target_size[1] / frame_dims[1]
    print(scale_x, scale_y)
    down_frames = []
    orig_frames = []
    for frame in frames:
        # Resize video frames
        resized = cv2.resize(frame, target_size)
        orig_frames.append(frame)
        down_frames.append(resized)
    return down_frames, orig_frames

scale_x, scale_y = 1, 1
def downsample_video(cap, target_size=(320, 220)):
    global scale_x, scale_y
    # Create video writer
    scale_x = target_size[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    scale_y = target_size[1] / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(scale_x, scale_y)
    ret = True
    down_frames = []
    orig_frames =[]
    count = 0
    while ret and count < 1000:
        # Extract video frames
        ret, frame = cap.read()
        if not ret:
            break
        # Resize video frames and write to output video
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame, target_size)
        orig_frames.append(frame)
        down_frames.append(resized)
        #print(resized.shape)
        count += 1
    return down_frames, orig_frames

def place_event_in_frame(img, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=1, thickness=2, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Define rectangle coordinates
    x, y = position
    top_left = (x - 10, y - text_height - 10)
    bottom_right = (x + text_width + 10, y + 10)

    # Draw background rectangle
    cv2.rectangle(img, top_left, bottom_right, bg_color, -1)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 2)  # frame border

    # Put the text on top of the rectangle
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

    return img

def main():
    parser = argparse.ArgumentParser(description="Process webcam or video input.")
    parser.add_argument("--video", action="store_true", help="Process a video file instead of the webcam.")
    parser.add_argument("--opticalflow", action="store_true", help="Enable optical flow for video processing.")
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

        flow_state = {"use_optical_flow": args.opticalflow}
        stream_process_and_write(
            video_path,
            output_file,
            detector,
            tracker,
            pose_model,
            feature_model,
            quad_pts=quad_pts,
            flow_state=flow_state
        )
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

            print("[INFO] Waiting for a valid frame from the webcam...")

            # Wait until a valid frame (not black) is captured
            while True:
                ret, first_frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read from webcam. Retrying...")
                    continue

                # Check if the frame is not black (average pixel intensity > threshold)
                if np.mean(first_frame) > 10:  # Threshold for non-black frame
                    print("[INFO] Valid frame detected.")
                    break

            # Select table corners for webcam
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

            if not video_writer.isOpened():
                print("[ERROR] Failed to initialize video writer. Exiting.")
                return

            # Initialize optical flow state
            flow_state = {"use_optical_flow": False}  # Default to static mode

            print("[INFO] Press Ctrl+G in the webcam preview window to stop.")
            print("[INFO] Press 's' to toggle between Static and Optical Flow modes.")

            cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)

            # Parameters for Lucas-Kanade Optical Flow
            lk_params = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

            # Detect good features to track for optical flow points
            features_to_track = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=7
            )
            ball_track_model = tf.keras.models.load_model("tracknet_pre_model.keras")
            ball_event_model = tf.keras.models.load_model("ball_event_model_2.keras")
            target_size = (320, 220)
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera.")
                    break

                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # --- Optical Flow ---
                if flow_state["use_optical_flow"]:
                    new_quad, quad_status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, quad_pts, None, **lk_params)
                    if new_quad is not None and quad_status.sum() == 4:
                        quad_pts = new_quad

                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features_to_track, None, **lk_params)
                    if next_pts is not None and status is not None:
                        valid_pts = next_pts[status.flatten() == 1]
                        features_to_track = valid_pts.reshape(-1, 1, 2)
                        for pt in valid_pts:
                            x, y = pt.ravel()
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

                # --- Pose Tracking ---
                processed_frame = run_pose_tracking(frame.copy(), detector, tracker, pose_model, feature_model)

                # --- Ball Tracking ---
                resized_frame = cv2.resize(frame, target_size)
                pred_pos = ball_track_model.predict(np.expand_dims(resized_frame, axis=0) / 255.0)
                orig_x = int(float(pred_pos[0][0]) / scale_x)
                orig_y = int(float(pred_pos[0][1]) / scale_y)

                # --- Ball Event Detection ---
                cropped_frame = crop_centered_with_padding(frame, (orig_x, orig_y), target_size)
                cropped_frame = cv2.resize(cropped_frame, target_size)
                pred_event = ball_event_model.predict(np.expand_dims(cropped_frame, axis=0))
                event = np.argmax(pred_event[0])
                event_label = {0: "bounce", 1: "net", 2: "empty_event"}[event]

                # --- Annotate ---
                cv2.circle(processed_frame, (orig_x, orig_y), 5, (0, 255, 0), -1)
                processed_frame = place_event_in_frame(processed_frame, event_label, position=(10, 20))
                quad_int = quad_pts.astype(int)
                cv2.polylines(processed_frame, [quad_int.reshape(-1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)

                if processed_frame.shape[:2] != (frame_height, frame_width):
                    processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

                video_writer.write(processed_frame)
                cv2.imshow("Processed Frame", processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 7:  # Ctrl+G
                    print("[INFO] Ctrl+G detected. Stopping webcam recording...")
                    break
                elif key == ord('s'):
                    flow_state["use_optical_flow"] = not flow_state["use_optical_flow"]
                    print(f"[INFO] Toggled mode: {'Optical Flow' if flow_state['use_optical_flow'] else 'Static'}")
                    if flow_state["use_optical_flow"]:
                        features_to_track = cv2.goodFeaturesToTrack(
                            curr_gray, maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=7
                        )

                prev_gray = curr_gray

            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Processed video saved to: {output_file}")

        except ValueError:
            print("Invalid input. Exiting.")


if __name__ == "__main__":
    main()
