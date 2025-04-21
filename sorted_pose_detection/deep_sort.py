import numpy as np
from scipy.optimize import linear_sum_assignment

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

