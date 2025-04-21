import cv2
import mediapipe as mp
import time

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Path to your lightweight model
MODEL_PATH = "pose_landmarker_heavy.task"

# Smoothing parameters
alpha = 0.7
previous_landmarks = []

# Base and options setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=5
)

# Initialize the pose landmarker
landmarker = vision.PoseLandmarker.create_from_options(options)

# Open the webcam
cap = cv2.VideoCapture(0)

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap in MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Generate timestamp
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Draw results
    current_landmarks = []
    for pose_idx, pose_landmarks in enumerate(result.pose_landmarks):
        smoothed_landmarks = []

        for lm_idx, lm in enumerate(pose_landmarks):
            x = lm.x * frame.shape[1]
            y = lm.y * frame.shape[0]

            # If we have previous landmarks, smooth with EMA
            if len(previous_landmarks) > pose_idx and lm_idx < len(previous_landmarks[pose_idx]):
                prev_x, prev_y = previous_landmarks[pose_idx][lm_idx]
                x = alpha * x + (1 - alpha) * prev_x
                y = alpha * y + (1 - alpha) * prev_y

            smoothed_landmarks.append((x, y))
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Draw connections
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(smoothed_landmarks) and end_idx < len(smoothed_landmarks):
                start_coords = smoothed_landmarks[start_idx]
                end_coords = smoothed_landmarks[end_idx]
                cv2.line(frame, (int(start_coords[0]), int(start_coords[1])),
                         (int(end_coords[0]), int(end_coords[1])), (0, 255, 255), 2)

        current_landmarks.append(smoothed_landmarks)

    previous_landmarks = current_landmarks

    # Show frame
    cv2.imshow("Multi-Person Pose with Smoothed Skeleton", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
