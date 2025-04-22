import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any


class PoseModel:
    """
    A simple wrapper around MediaPipe Pose to estimate human body landmarks
    from a given image frame.
    """

    def __init__(self):
        """
        Initializes the MediaPipe pose estimation model.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False
        )
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Estimate pose landmarks from a single image frame.

        Parameters:
            frame (numpy.ndarray): A BGR image frame (as read by OpenCV).

        Returns:
            list: A list of dictionaries with keypoints and connections.
        """
        # Convert frame from BGR to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        poses = []

        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([
                    landmark.x * frame.shape[1],  # X-coordinate in pixels
                    landmark.y * frame.shape[0],  # Y-coordinate in pixels
                    landmark.visibility           # Visibility confidence
                ])
            poses.append({
                'keypoints': keypoints,
                'connections': self.pose_connections
            })

        return poses
