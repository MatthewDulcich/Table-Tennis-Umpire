import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from sorted_pose_detection.pose_model import PoseModel
from sorted_pose_detection.deep_sort import DeepSORT

# --- TensorFlow Configuration ---
def configure_tensorflow():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU for TensorFlow.")
        except RuntimeError as e:
            print(f"Error configuring TensorFlow GPU: {e}")
    else:
        print("No GPU detected. Using CPU for TensorFlow.")

# --- Feature Extractor ---
def build_model(input_shape=(224, 224, 3)):
    from tensorflow.keras.applications import MobileNetV3Small
    base = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        pooling='avg',
        weights='imagenet',
        include_preprocessing=False
    )
    x = tf.keras.layers.Dense(128, activation='relu')(base.output)
    return tf.keras.Model(inputs=base.input, outputs=x)

# --- Crop and Feature Extraction ---
def extract_crops(frame, bboxes):
    crops = []
    for x, y, w, h in bboxes:
        x, y, w, h = map(int, [x, y, w, h])
        crop = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
        if crop.size == 0:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (224, 224))
        crops.append(crop)
    return np.array(crops)

def extract_features(images, model):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images = images / 127.5 - 1.0
    return model(images, training=False).numpy()

# --- Pose Tracking Pipeline ---
def run_pose_tracking(frame, detector, tracker, pose_model, feature_model):
    original_height, original_width = frame.shape[:2]
    downscale_width, downscale_height = 640, 360
    scale_x = original_width / downscale_width
    scale_y = original_height / downscale_height
    resized_frame = cv2.resize(frame, (downscale_width, downscale_height))

    results = detector(resized_frame)[0]
    bboxes = []
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) != 0:
            continue
        w, h = x2 - x1, y2 - y1
        bboxes.append([x1 * scale_x, y1 * scale_y, w * scale_x, h * scale_y])

    if bboxes:
        crops = extract_crops(frame, bboxes)
        features = extract_features(crops, feature_model)
        tracks = tracker.update(bboxes, features)

        for track, crop in zip(tracks, crops):
            if track.time_since_update > 0:
                continue

            poses = pose_model.predict(crop)
            x, y, w, h = map(int, track.get_bbox())

            for pose in poses:
                keypoints = pose['keypoints']
                connections = pose['connections']

                for px, py, confidence in keypoints:
                    if confidence > 0.5:
                        gx = int(px / 224 * w + x)
                        gy = int(py / 224 * h + y)
                        cv2.circle(frame, (gx, gy), 5, (0, 255, 0), -1)

                for start_idx, end_idx in connections:
                    if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                        x1 = int(keypoints[start_idx][0] / 224 * w + x)
                        y1 = int(keypoints[start_idx][1] / 224 * h + y)
                        x2 = int(keypoints[end_idx][0] / 224 * w + x)
                        y2 = int(keypoints[end_idx][1] / 224 * h + y)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track.id}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame
