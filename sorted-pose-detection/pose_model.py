import mediapipe as mp
import numpy as np
import cv2

class PoseModel:
    def __init__(self, model_path=None):
        """
        Initializes the pose estimation model.

        Parameters:
            model_path (str): Path to the pre-trained pose estimation model (not used for MediaPipe).
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS  # Connections for drawing the skeleton

    def predict(self, frame):
        """
        Performs pose estimation on a single frame.

        Parameters:
            frame (numpy.ndarray): The input frame from the video or webcam.

        Returns:
            dict: Pose estimation results containing keypoints and connections.
        """
        # Convert the frame to RGB (MediaPipe expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = self.pose.process(rgb_frame)

        # Extract keypoints and connections
        poses = []
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x * frame.shape[1],  # Convert normalized x to pixel x
                                  landmark.y * frame.shape[0],  # Convert normalized y to pixel y
                                  landmark.visibility])         # Confidence score
            poses.append({'keypoints': keypoints, 'connections': self.pose_connections})

        return poses


# Main section for testing the PoseModel class
if __name__ == "__main__":
    # Initialize the PoseModel
    pose_model = PoseModel()

    # Open a webcam feed
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Press 'q' to quit.")

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform pose estimation
        poses = pose_model.predict(frame)

        # Draw keypoints and connections (skeleton) on the frame
        for pose in poses:
            keypoints = pose['keypoints']
            connections = pose['connections']

            # Draw keypoints
            for x, y, confidence in keypoints:
                if confidence > 0.5:  # Confidence threshold
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Draw connections (skeleton)
            for start_idx, end_idx in connections:
                if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:  # Confidence threshold
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Pose Estimation with Skeleton", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()