import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV3Small
import mediapipe as mp

# --- TensorFlow Cross-Platform Configuration ---
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
        include_preprocessing=False
    )
    x = tf.keras.layers.Dense(128, activation='relu')(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model

model = build_model()

# --- Deep SORT Tracker ---
class Track:
    def __init__(self, id, bbox, feature, kf, max_age=30):
        self.id = id
        self.kf = kf
        self.mean = self._xywh_to_state(bbox)
        self.cov = np.eye(7)
        self.feature = feature
        self.age = 0
        self.time_since_update = 0
        self.max_age = max_age

    def _xywh_to_state(self, bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2, w * h, w / h, 0, 0, 0], dtype=np.float32)

    def get_bbox(self):
        cx, cy, s, r = self.mean[:4]
        val = s * r
        if val <= 0:
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
        self.max_age = max_age

    def update(self, bboxes, features):
        for track in self.tracks:
            track.predict()

        if len(bboxes) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= t.max_age]
            return self.tracks

        cost = np.zeros((len(self.tracks), len(bboxes)))
        for i, t in enumerate(self.tracks):
            for j, f in enumerate(features):
                cost[i, j] = 1 - np.dot(t.feature, f) / (np.linalg.norm(t.feature) * np.linalg.norm(f) + 1e-6)

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_detections = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 0.7:
                self.tracks[r].update(bboxes[c], features[c])
                assigned_tracks.add(r)
                assigned_detections.add(c)

        for i in range(len(bboxes)):
            if i not in assigned_detections:
                self.tracks.append(Track(self.next_id, bboxes[i], features[i], self.kf, max_age=self.max_age))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= t.max_age]
        return self.tracks

# --- Pose Model ---
class PoseModel:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS

    def predict(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        poses = []
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.visibility])
            poses.append({'keypoints': keypoints, 'connections': self.pose_connections})
        return poses

# --- Helper Functions ---
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
    images = images / 127.5 - 1.0
    return model(images, training=False).numpy()

# --- Main Execution ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = YOLO("models/yolov5nu.pt")
    tracker = DeepSORT(max_age=30)
    pose_model = PoseModel()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        if len(bboxes) == 0:
            cv2.imshow("Multi-Person Pose Tracking", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        crops = extract_crops(frame, bboxes)
        features = extract_features(crops)
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
                        global_x = int(px / 224 * w + x)
                        global_y = int(py / 224 * h + y)
                        cv2.circle(frame, (global_x, global_y), 5, (0, 255, 0), -1)

                for start_idx, end_idx in connections:
                    if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                        x1 = int(keypoints[start_idx][0] / 224 * w + x)
                        y1 = int(keypoints[start_idx][1] / 224 * h + y)
                        x2 = int(keypoints[end_idx][0] / 224 * w + x)
                        y2 = int(keypoints[end_idx][1] / 224 * h + y)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track.id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Multi-Person Pose Tracking", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
