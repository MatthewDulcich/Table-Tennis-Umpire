import cv2
import numpy as np
from deepsort.deep_sort import DeepSort  # Assuming DeepSORT is implemented in a `deepsort` module
from models.pose_model import PoseModel  # Assuming a pose estimation model is implemented in `models`

class PoseEstimationWithTracking:
    def __init__(self, pose_model_path, deepsort_model_path):
        """
        Initializes the pose estimation and DeepSORT tracking system.

        Parameters:
            pose_model_path (str): Path to the pre-trained pose estimation model.
            deepsort_model_path (str): Path to the DeepSORT model weights.
        """
        # Initialize pose estimation model
        self.pose_model = PoseModel(pose_model_path)

        # Initialize DeepSORT tracker
        self.tracker = DeepSort(deepsort_model_path)

    def estimate_poses(self, frame):
        """
        Performs pose estimation on a single frame.

        Parameters:
            frame (numpy.ndarray): The input frame from the video or webcam.

        Returns:
            list: List of pose estimation results.
        """
        return self.pose_model.predict(frame)

    def track_objects(self, bboxes, frame):
        """
        Tracks objects using DeepSORT.

        Parameters:
            bboxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
            frame (numpy.ndarray): The input frame.

        Returns:
            list: List of tracked objects with IDs.
        """
        return self.tracker.update(bboxes, frame)

    def extract_bounding_boxes(self, poses):
        """
        Extracts bounding boxes from pose estimation results.

        Parameters:
            poses (list): List of pose estimation results.

        Returns:
            list: List of bounding boxes in the format [x1, y1, x2, y2].
        """
        bboxes = []
        for pose in poses:
            keypoints = pose['keypoints']
            x_coords = [kp[0] for kp in keypoints if kp[2] > 0.5]  # Confidence threshold
            y_coords = [kp[1] for kp in keypoints if kp[2] > 0.5]
            if x_coords and y_coords:
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                bboxes.append([x1, y1, x2, y2])
        return bboxes