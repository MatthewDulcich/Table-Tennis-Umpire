import cv2
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV2
# TODO: Revert back to normal tf
# --- TensorFlow Cross-Platform Configuration ---
def configure_tensorflow():
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU for TensorFlow.")
        except RuntimeError as e:
            print(f"Error configuring TensorFlow GPU: {e}")
    else:
        print("No GPU detected. Using CPU for TensorFlow.")

# Configure TensorFlow
configure_tensorflow()

# --- Kalman Filter ---
class KalmanFilter:
    def __init__(self):
        self.dt = 1.0
        self.A = np.eye(7)
        for i, j in zip([0, 1, 2], [4, 5, 6]):
            self.A[i, j] = self.dt
        self.H = np.eye(4, 7)
        self.P = np.eye(7) * 10
        self.Q = np.eye(7) * 1e-2
        self.R = np.eye(4) * 1e-1

    def predict(self, mean, cov):
        mean = self.A @ mean
        cov = self.A @ cov @ self.A.T + self.Q
        return mean, cov

    def update(self, mean, cov, measurement):
        S = self.H @ cov @ self.H.T + self.R
        K = cov @ self.H.T @ np.linalg.inv(S)
        y = measurement - (self.H @ mean)
        mean = mean + K @ y
        cov = cov - K @ self.H @ cov
        return mean, cov

# --- Appearance Feature Extractor ---
def build_model(input_shape=(224, 224, 3)):
    base = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        pooling='avg',
        weights='imagenet',
        include_preprocessing=False  # We'll handle preprocessing manually
    )
    x = tf.keras.layers.Dense(128, activation='relu')(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model

model = build_model()

# --- Tracker ---
class Track:
    def __init__(self, id, bbox, feature, kf, max_age=30):
        self.id = id
        self.kf = kf
        self.mean = self._xywh_to_state(bbox)
        self.cov = np.eye(7)
        self.feature = feature
        self.age = 0
        self.time_since_update = 0
        self.max_age = max_age  # Maximum allowed frames without updates

    def _xywh_to_state(self, bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2, w * h, w / h, 0, 0, 0], dtype=np.float32)

    def get_bbox(self):
        cx, cy, s, r = self.mean[:4]
        val = s * r
        if val <= 0:
            # Assign default width and height
            w, h = 1.0, 1.0
        else:
            w = np.sqrt(val)
            h = s / w
        return [cx - w / 2, cy - h / 2, w, h]

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, feature):
        measurement = self._xywh_to_state(bbox)[:4]
        self.mean, self.cov = self.kf.update(self.mean, self.cov, measurement)
        self.feature = feature
        self.time_since_update = 0

class DeepSORT:
    def __init__(self, max_age=30):
        self.kf = KalmanFilter()
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age  # Maximum allowed frames without updates

    def update(self, bboxes, features):
        for track in self.tracks:
            track.predict()

        if len(bboxes) == 0:
            # No detections; only predict existing tracks
            # Remove tracks that have exceeded max_age
            self.tracks = [t for t in self.tracks if t.time_since_update <= t.max_age]
            return self.tracks

        cost = np.zeros((len(self.tracks), len(bboxes)))
        for i, t in enumerate(self.tracks):
            for j, f in enumerate(features):
                cost[i, j] = 1 - np.dot(t.feature, f) / (np.linalg.norm(t.feature) * np.linalg.norm(f) + 1e-6)

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_detections = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 0.7:
                self.tracks[r].update(bboxes[c], features[c])
                assigned_tracks.add(r)
                assigned_detections.add(c)

        # Create new tracks for unmatched detections
        for i in range(len(bboxes)):
            if i not in assigned_detections:
                self.tracks.append(Track(self.next_id, bboxes[i], features[i], self.kf, max_age=self.max_age))
                self.next_id += 1

        # Remove tracks that have exceeded max_age
        self.tracks = [t for t in self.tracks if t.time_since_update <= t.max_age]

        return self.tracks

# --- Helper functions ---
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

def extract_features(images):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images = images / 127.5 - 1.0  # Scale pixel values to [-1, 1]
    return model(images, training=False).numpy()


# --- Main Execution ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = YOLO("models/yolov5nu.pt")  # Lightweight YOLOv5 model
    tracker = DeepSORT(max_age=30)  # Adjust max_age as needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale the frame for faster processing
        original_height, original_width = frame.shape[:2]
        downscale_width, downscale_height = 640, 360  # Adjust as needed
        scale_x = original_width / downscale_width
        scale_y = original_height / downscale_height
        resized_frame = cv2.resize(frame, (downscale_width, downscale_height))

        # Run YOLO detection on the downscaled frame
        results = detector(resized_frame)[0]
        bboxes = []

        for det in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) != 0:  # Only track person class
                continue
            w, h = x2 - x1, y2 - y1
            # Scale bounding boxes back to the original resolution
            bboxes.append([x1 * scale_x, y1 * scale_y, w * scale_x, h * scale_y])

        if len(bboxes) == 0:
            cv2.imshow("DeepSORT", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # Extract crops and features for tracking
        crops = extract_crops(frame, bboxes)
        features = extract_features(crops)
        tracks = tracker.update(bboxes, features)

        # Draw tracked objects
        for track in tracks:
            if track.time_since_update > 0:
                continue
            x, y, w, h = map(int, track.get_bbox())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track.id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("DeepSORT", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()