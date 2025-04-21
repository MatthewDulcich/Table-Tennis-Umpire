import cv2
import mediapipe as mp

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
                keypoints.append([
                    landmark.x * frame.shape[1],
                    landmark.y * frame.shape[0],
                    landmark.visibility
                ])
            poses.append({
                'keypoints': keypoints,
                'connections': self.pose_connections
            })
        return poses
