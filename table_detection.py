import cv2
import numpy as np

# Global list to store selected points
selected_points = []

def select_point(event, x, y, flags, param):
    """Callback function to select points on the frame."""
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append([x, y])
        print(f"Selected Point {len(selected_points)}: ({x}, {y})")

def load_video(video_path):
    """Load the video and return the video capture object and the first frame."""
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        cap.release()
        return None, None
    return cap, first_frame

def select_corners(first_frame):
    """Allow the user to select 4 corners on the first frame."""
    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", select_point)

    while True:
        temp = first_frame.copy()
        for pt in selected_points:
            cv2.circle(temp, tuple(pt), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 Corners", temp)
        if len(selected_points) == 4:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return None
    cv2.destroyWindow("Select 4 Corners")
    return np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)

def process_video(cap, first_frame, quad_pts, use_optical_flow=False):
    """Process the video to track points and quadrilateral."""
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Detect good features to track
    features_to_track = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=7
    )
    if features_to_track is None:
        print("Could not detect features. Exiting.")
        return

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    motion_threshold = 5.0  # Threshold for filtering moving points

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features_to_track, None, **lk_params)

        if next_pts is None or status is None or np.count_nonzero(status) < 10:
            print("Too few points or tracking failed. Re-detecting features...")
            features_to_track = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=7
            )
            if features_to_track is None:
                print("Re-detection failed. Exiting.")
                break
            prev_gray = curr_gray
            continue

        valid_pts = next_pts[status.flatten() == 1].reshape(-1, 2)
        prev_pts = features_to_track[status.flatten() == 1].reshape(-1, 2)
        motion = np.linalg.norm(valid_pts - prev_pts, axis=1)
        motion_mask = motion < motion_threshold
        static_pts = valid_pts[motion_mask]

        # Determine which points to show
        points_to_draw = valid_pts if use_optical_flow else static_pts

        # Draw optical flow points
        for pt in points_to_draw:
            x, y = pt.ravel()
            color = (0, 255, 0) if not use_optical_flow else (255, 255, 0)
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)

        if static_pts.shape[0] > 0:
            features_to_track = static_pts.reshape(-1, 1, 2)
        else:
            print("No static points left. Re-detecting features...")
            features_to_track = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=7
            )
            if features_to_track is None:
                print("Re-detection failed. Exiting.")
                break

        # Always draw the quadrilateral
        quad_int = quad_pts.astype(int)
        cv2.polylines(frame, [quad_int.reshape(-1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Update quadrilateral points if using optical flow
        if use_optical_flow:
            new_quad, quad_status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, quad_pts, None, **lk_params)
            if new_quad is not None and quad_status.sum() == 4:
                quad_pts = new_quad

        prev_gray = curr_gray

        # Display mode label
        label = "Mode: Optical Flow" if use_optical_flow else "Mode: Static Only"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("Tracked Quadrilateral and Points", frame)

        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == 7:  # ASCII code for Ctrl+G
            print("Ctrl+G detected. Exiting...")
            break
        elif key == ord('s'):
            use_optical_flow = not use_optical_flow
            print(f"Switched mode: {'Optical Flow' if use_optical_flow else 'Static Only'}")

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = 'data/train/game_1.mp4'
    cap, first_frame = load_video(video_path)
    if cap is None or first_frame is None:
        return

    quad_pts = select_corners(first_frame)
    if quad_pts is None:
        return

    process_video(cap, first_frame, quad_pts)

if __name__ == "__main__":
    main()