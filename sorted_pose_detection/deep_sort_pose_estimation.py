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


def extract_crops(frame: np.ndarray, bboxes: list, target_size: int = 224) -> np.ndarray:
    """
    Crop, resize, and pad image patches based on bounding boxes.

    Args:
        frame (np.ndarray): The full image frame.
        bboxes (list): List of bounding boxes [x1, y1, x2, y2].
        target_size (int): The target size for the output crops (e.g., 224x224).

    Returns:
        np.ndarray: Array of cropped, resized, and padded image patches.
    """
    crops = []
    for x1, y1, x2, y2 in bboxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

        if crop.size == 0:
            # If the crop is empty, create a blank image
            padded_crop = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        else:
            # Get the original dimensions of the crop
            h, w = crop.shape[:2]

            # Calculate the scaling factor to fit within the target size
            scale = min(target_size / w, target_size / h)
            new_w, new_h = int(w * scale), int(h * scale)

            # Resize the crop while maintaining aspect ratio
            resized_crop = cv2.resize(crop, (new_w, new_h))

            # Create a blank canvas of the target size
            padded_crop = np.zeros((target_size, target_size, 3), dtype=np.uint8)

            # Calculate padding to center the resized crop
            pad_x = (target_size - new_w) // 2
            pad_y = (target_size - new_h) // 2

            # Place the resized crop onto the blank canvas
            padded_crop[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_crop

        crops.append(padded_crop)

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

            # Preprocess crop for pose model
            crop = (crop * 255).astype(np.uint8)  # Convert back to uint8 after normalization
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # Convert to RGB

            poses = pose_model.predict(crop)
            x1, y1, x2, y2 = map(int, track.get_bbox())
            w, h = x2 - x1, y2 - y1
            crop_width, crop_height = 224, 224

            for pose in poses:
                keypoints = pose['keypoints']  # List of [x, y, confidence]
                connections = pose['connections']  # List of (start_idx, end_idx)

                # Validate keypoints and draw them
                for px, py, confidence in keypoints:
                    if confidence > 0.5:  # Adjusted confidence threshold
                        # Check if keypoints are normalized (assume [0, 1] range)
                        if px <= 1.0 and py <= 1.0:
                            # Keypoints are normalized, scale to crop dimensions
                            px *= crop_width
                            py *= crop_height

                        # Calculate relative position within the bounding box
                        relative_x = px / crop_width  # Relative x-coordinate (0 to 1)
                        relative_y = py / crop_height  # Relative y-coordinate (0 to 1)

                        # Rescale keypoints to the original frame using relative position
                        gx = int(relative_x * w + x1)  # Scale x-coordinate
                        gy = int(relative_y * h + y1)  # Scale y-coordinate

                        # Debugging information
                        # print(f"[DEBUG] Keypoint (px, py): ({px}, {py}), Relative (rel_x, rel_y): ({relative_x}, {relative_y}), Scaled (gx, gy): ({gx}, {gy}), Confidence: {confidence}")
                        # print(f"[DEBUG] Bounding box (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2}), Width: {w}, Height: {h}")

                        # Ensure keypoints are within frame bounds
                        if 0 <= gx < frame.shape[1] and 0 <= gy < frame.shape[0]:
                            cv2.circle(frame, (gx, gy), 5, (0, 255, 0), -1)

                # Validate connections and draw them
                for start_idx, end_idx in connections:
                    if (
                        keypoints[start_idx][2] > 0.5 and  # Confidence for start keypoint
                        keypoints[end_idx][2] > 0.5       # Confidence for end keypoint
                    ):
                        # Calculate relative positions for connections
                        rel_x1 = keypoints[start_idx][0] / crop_width
                        rel_y1 = keypoints[start_idx][1] / crop_height
                        rel_x2 = keypoints[end_idx][0] / crop_width
                        rel_y2 = keypoints[end_idx][1] / crop_height

                        # Rescale connections to the original frame
                        x1_conn = int(rel_x1 * w + x1)
                        y1_conn = int(rel_y1 * h + y1)
                        x2_conn = int(rel_x2 * w + x1)
                        y2_conn = int(rel_y2 * h + y1)

                        # Ensure connections are within frame bounds
                        if (
                            0 <= x1_conn < frame.shape[1] and 0 <= y1_conn < frame.shape[0] and
                            0 <= x2_conn < frame.shape[1] and 0 <= y2_conn < frame.shape[0]
                        ):
                            cv2.line(frame, (x1_conn, y1_conn), (x2_conn, y2_conn), (255, 0, 0), 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track.id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame
