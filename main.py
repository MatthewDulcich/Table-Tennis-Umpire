from read_webcam_stream import list_available_cameras, launch_webcam
from pose_estimation import PoseEstimationWithTracking
import cv2

def process_frame(frame):
    """
    Processes a single frame by performing pose estimation and tracking.

    Parameters:
        frame (numpy.ndarray): The input frame from the webcam.

    Returns:
        numpy.ndarray: The processed frame with results drawn.
    """
    # Initialize pose estimation and tracking
    pose_tracker = PoseEstimationWithTracking("path/to/pose_model", "path/to/deepsort_model")

    # Step 1: Perform pose estimation
    poses = pose_tracker.estimate_poses(frame)

    # Step 2: Extract bounding boxes from poses
    bboxes = pose_tracker.extract_bounding_boxes(poses)

    # Step 3: Perform tracking
    tracked_objects = pose_tracker.track_objects(bboxes, frame)

    # Step 4: Draw results (you can implement a helper function for this)
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

def main():
    """
    Main function to list available cameras and launch the selected webcam.
    """
    # List all available cameras
    cameras = list_available_cameras()

    if not cameras:
        print("No cameras found.")
        return

    print("Available cameras:")
    for i, cam in enumerate(cameras):
        print(f"{i}: Camera Index {cam}")

    # Allow the user to select a camera
    try:
        selected_index = int(input("Select a camera index from the list above: "))
        if selected_index < 0 or selected_index >= len(cameras):
            print("Invalid selection.")
        else:
            # Launch the webcam with the processing function
            launch_webcam(camera_index=cameras[selected_index], frame_callback=process_frame)
    except ValueError:
        print("Invalid input. Exiting.")

if __name__ == "__main__":
    main()