import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from sorted_pose_detection.pose_model import PoseModel
from sorted_pose_detection.deep_sort import DeepSORT


def configure_tensorflow() -> None:
    """
    Configure TensorFlow to use GPU with memory growth if available.
    """
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


def build_model(input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """
    Build and return a MobileNetV3Small-based feature extractor model.

    Args:
        input_shape (tuple): The shape of the input image.

    Returns:
        tf.keras.Model: A Keras model that outputs 128-dimensional feature vectors.
    """
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


def extract_crops(frame: np.ndarray, bboxes: list) -> np.ndarray:
    """
    Crop and resize image patches based on bounding boxes.

    Args:
        frame (np.ndarray): The full image frame.
        bboxes (list): List of bounding boxes [x, y, w, h].

    Returns:
        np.ndarray: Array of cropped and resized image patches.
    """
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


def extract_features(images: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    """
    Extract features for each image crop using the given model.

    Args:
        images (np.ndarray): Batch of images.
        model (tf.keras.Model): Feature extractor model.

    Returns:
        np.ndarray: Extracted feature vectors.
    """
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images = images / 127.5 - 1.0
    return model(images, training=False).numpy()


def non_maximum_suppression_with_occlusion(boxes, scores, iou_threshold=0.5, occlusion_threshold=0.7):
    """
    Applies Non-Maximum Suppression (NMS) while accounting for occlusion.

    Parameters:
        boxes (numpy.ndarray): Array of bounding boxes (x1, y1, x2, y2).
        scores (numpy.ndarray): Array of confidence scores for each box.
        iou_threshold (float): IoU threshold for standard NMS.
        occlusion_threshold (float): IoU threshold to retain overlapping boxes for occlusion.

    Returns:
        list: Indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    # Convert boxes to float for calculations
    boxes = boxes.astype(float)

    # Compute the area of each box
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # Sort boxes by confidence scores in descending order
    sorted_indices = np.argsort(scores)[::-1]

    keep = []
    while len(sorted_indices) > 0:
        # Pick the box with the highest score
        current = sorted_indices[0]
        keep.append(current)

        # Compute IoU of the current box with the rest
        x1 = np.maximum(boxes[current, 0], boxes[sorted_indices[1:], 0])
        y1 = np.maximum(boxes[current, 1], boxes[sorted_indices[1:], 1])
        x2 = np.minimum(boxes[current, 2], boxes[sorted_indices[1:], 2])
        y2 = np.minimum(boxes[current, 3], boxes[sorted_indices[1:], 3])

        inter_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
        iou = inter_area / (areas[current] + areas[sorted_indices[1:]] - inter_area)

        # Suppress boxes with IoU > iou_threshold
        suppressed_indices = np.where(iou > iou_threshold)[0]

        # Retain overlapping boxes for occlusion if IoU > occlusion_threshold
        for idx in suppressed_indices:
            if iou[idx] > occlusion_threshold:
                keep.append(sorted_indices[1:][idx])

        # Remove processed indices
        sorted_indices = np.delete(sorted_indices, np.concatenate(([0], suppressed_indices)))

    return list(set(keep))  # Ensure unique indices


def run_pose_tracking(
    frame: np.ndarray,
    detector: YOLO,
    tracker: DeepSORT,
    pose_model: PoseModel,
    feature_model: tf.keras.Model
) -> np.ndarray:
    """
    Perform pose estimation and tracking on a single frame.

    Args:
        frame (np.ndarray): The input video frame.
        detector (YOLO): YOLO object detector.
        tracker (DeepSORT): DeepSORT tracker instance.
        pose_model (PoseModel): MediaPipe pose estimator.
        feature_model (tf.keras.Model): Feature extractor model.

    Returns:
        np.ndarray: Annotated output frame.
    """
    original_height, original_width = frame.shape[:2]
    downscale_width, downscale_height = 640, 360
    scale_x = original_width / downscale_width
    scale_y = original_height / downscale_height
    resized_frame = cv2.resize(frame, (downscale_width, downscale_height))

    # Detect objects in the frame
    results = detector(resized_frame)[0]
    bboxes = []
    scores = []
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) != 0:  # Only process person class (class ID 0)
            continue
        # Scale bounding box coordinates back to the original frame size
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y
        bboxes.append([x1, y1, x2, y2])
        scores.append(conf)

    bboxes = np.array(bboxes)
    scores = np.array(scores)

    # Apply NMS with occlusion handling
    if len(bboxes) > 0:
        keep_indices = non_maximum_suppression_with_occlusion(bboxes, scores)
        refined_bboxes = bboxes[keep_indices]
        refined_scores = scores[keep_indices]

        # Extract crops and features for refined bounding boxes
        crops = extract_crops(frame, refined_bboxes)
        features = extract_features(crops, feature_model)

        # Update the tracker with refined detections
        tracks = tracker.update(refined_bboxes, features)

        for track, crop in zip(tracks, crops):
            if track.time_since_update > 0:
                continue

            poses = pose_model.predict(crop)
            x1, y1, x2, y2 = map(int, track.get_bbox())
            w, h = x2 - x1, y2 - y1

            # Draw pose keypoints and connections
            for pose in poses:
                keypoints = pose['keypoints']
                connections = pose['connections']

                for px, py, confidence in keypoints:
                    if confidence > 0.5:
                        # Scale keypoints to the original frame size
                        gx = int(px / 224 * w + x1)
                        gy = int(py / 224 * h + y1)
                        cv2.circle(frame, (gx, gy), 5, (0, 255, 0), -1)

                for start_idx, end_idx in connections:
                    if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                        x1_conn = int(keypoints[start_idx][0] / 224 * w + x1)
                        y1_conn = int(keypoints[start_idx][1] / 224 * h + y1)
                        x2_conn = int(keypoints[end_idx][0] / 224 * w + x1)
                        y2_conn = int(keypoints[end_idx][1] / 224 * h + y1)
                        cv2.line(frame, (x1_conn, y1_conn), (x2_conn, y2_conn), (255, 0, 0), 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track.id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame
