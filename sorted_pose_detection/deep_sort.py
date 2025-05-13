import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


class KalmanFilter:
    """
    A simple Kalman Filter for tracking objects in video.
    It predicts the next state and updates the state estimate given a new measurement.
    """

    def __init__(self):
        self.dt = 1.0
        self.A = np.eye(7)
        for i, j in zip([0, 1, 2], [4, 5, 6]):
            self.A[i, j] = self.dt
        self.H = np.eye(4, 7)
        self.P = np.eye(7) * 10
        self.Q = np.eye(7) * 1e-2
        self.R = np.eye(4) * 1e-1

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state and covariance.

        Parameters:
            mean (np.ndarray): The current state estimate.
            cov (np.ndarray): The current covariance estimate.

        Returns:
            tuple: Predicted state and covariance.
        """
        mean = self.A @ mean
        cov = self.A @ cov @ self.A.T + self.Q
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the state estimate with a new measurement.

        Parameters:
            mean (np.ndarray): The predicted state estimate.
            cov (np.ndarray): The predicted covariance estimate.
            measurement (np.ndarray): The new observation.

        Returns:
            tuple: Updated state and covariance.
        """
        S = self.H @ cov @ self.H.T + self.R
        K = cov @ self.H.T @ np.linalg.inv(S)
        y = measurement - (self.H @ mean)
        mean = mean + K @ y
        cov = cov - K @ self.H @ cov
        return mean, cov


class Track:
    """
    Represents a single tracked object with state and Kalman Filter.
    """

    def __init__(self, id: int, bbox: List[float], feature: np.ndarray, kf: KalmanFilter, max_age: int = 30):
        self.id = id
        self.kf = kf
        self.mean = self._xywh_to_state(bbox)
        self.cov = np.eye(7)
        self.feature = feature
        self.smoothed_feature = feature  # Initialize smoothed feature
        self.age = 0
        self.time_since_update = 0
        self.max_age = max_age

    def _xywh_to_state(self, bbox: List[float]) -> np.ndarray:
        """
        Convert a bounding box (x, y, w, h) to a Kalman filter state vector.

        Parameters:
            bbox (list): Bounding box in (x, y, w, h) format.

        Returns:
            np.ndarray: State vector.
        """
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2, w * h, w / h, 0, 0, 0], dtype=np.float32)

    def get_bbox(self) -> List[float]:
        """
        Convert current state back to bounding box format (x, y, w, h).

        Returns:
            list: Bounding box as [x, y, w, h].
        """
        cx, cy, s, r = self.mean[:4]
        val = s * r
        if val <= 0:
            w, h = 1.0, 1.0
        else:
            w = np.sqrt(val)
            h = s / w
        return [cx - w / 2, cy - h / 2, w, h]

    def predict(self) -> None:
        """
        Predict the next state of the track using the Kalman Filter.
        """
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: List[float], feature: np.ndarray) -> None:
        """
        Update the track state with a new bounding box and feature vector.

        Parameters:
            bbox (list): Detected bounding box.
            feature (np.ndarray): Associated appearance feature.
        """
        measurement = self._xywh_to_state(bbox)[:4]
        self.mean, self.cov = self.kf.update(self.mean, self.cov, measurement)
        self.feature = feature
        self.smoothed_feature = 0.9 * self.smoothed_feature + 0.1 * feature  # Exponential smoothing
        self.time_since_update = 0


class DeepSORT:
    """
    A multi-object tracker that uses Kalman filtering and appearance feature matching.
    """

    def __init__(self, max_age: int = 5760):
        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self.deleted_tracks: List[Track] = []  # Buffer for deleted tracks
        self.next_id = 0
        self.max_age = max_age

    def delete_track(self, track: Track):
        """
        Move a track to the deleted tracks buffer.
        """
        self.deleted_tracks.append(track)
        if len(self.deleted_tracks) > 50:  # Limit the size of the buffer
            self.deleted_tracks.pop(0)

    def reidentify(self, bbox: List[float], feature: np.ndarray) -> int:
        """
        Attempt to reassign an ID to a detection by comparing it with deleted tracks.

        Parameters:
            bbox (list): Detected bounding box.
            feature (np.ndarray): Appearance feature of the detection.

        Returns:
            int: Reassigned track ID, or None if no match is found.
        """
        for track in self.deleted_tracks:
        # Compare appearance features
            similarity = np.dot(track.feature, feature) / (
                np.linalg.norm(track.feature) * np.linalg.norm(feature) + 1e-6
            )
            # Apply a time decay factor
            decay = max(0.5, 1 - 0.01 * track.time_since_update)  # Decay factor (0.5 minimum)
            similarity *= decay
            if similarity > 0.75:  # Adjusted threshold
                return track.id
        return None

    def update(self, bboxes: List[List[float]], features: List[np.ndarray]) -> List[Track]:
        """
        Update tracked objects based on new detections.

        Parameters:
            bboxes (list): List of bounding boxes (x, y, w, h).
            features (list): List of feature vectors for each detection.

        Returns:
            list: Updated list of active Track objects.
        """
        for track in self.tracks:
            track.predict()

        if len(bboxes) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= t.max_age]
            return self.tracks

        cost = np.zeros((len(self.tracks), len(bboxes)))
        for i, t in enumerate(self.tracks):
            for j, f in enumerate(features):
                # Appearance cost (cosine similarity)
                appearance_cost = 1 - np.dot(t.feature, f) / (np.linalg.norm(t.feature) * np.linalg.norm(f) + 1e-6)
                
                # Spatial cost (Euclidean distance)
                predicted_bbox = t.get_bbox()
                detected_bbox = bboxes[j]
                spatial_cost = np.linalg.norm(np.array(predicted_bbox[:2]) - np.array(detected_bbox[:2]))  # Distance between centers
                
                # Combine costs (weighted sum)
                cost[i, j] = appearance_cost + 0.1 * spatial_cost  # Adjust weight as needed

                # IoU cost
                iou_cost = 1 - iou(t.get_bbox(), bboxes[j])  # IoU-based cost
                cost[i, j] += 0.5 * iou_cost  # Adjust weight as needed

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_detections = set()
        for r, c in zip(row_ind, col_ind):
            predicted_bbox = self.tracks[r].get_bbox()
            detected_bbox = bboxes[c]
            velocity = np.linalg.norm(np.array(predicted_bbox[:2]) - np.array(detected_bbox[:2]))
            
            if velocity < 50:  # Threshold for maximum allowed velocity
                self.tracks[r].update(bboxes[c], features[c])
                assigned_tracks.add(r)
                assigned_detections.add(c)

        for i in range(len(bboxes)):
            if i not in assigned_detections:
                # Attempt to reassign an ID
                reassigned_id = self.reidentify(bboxes[i], features[i])
                if reassigned_id is not None:
                    # Reactivate the track
                    track = next(t for t in self.deleted_tracks if t.id == reassigned_id)
                    track.update(bboxes[i], features[i])
                    self.tracks.append(track)
                    self.deleted_tracks.remove(track)
                else:
                    # Create a new track if no reassignment is possible
                    self.tracks.append(Track(self.next_id, bboxes[i], features[i], self.kf, max_age=self.max_age))
                    self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= t.max_age]
        return self.tracks


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0
