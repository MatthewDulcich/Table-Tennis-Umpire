import cv2
import numpy as np

# Global list to store selected points
selected_points = []

def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append([x, y])
        print(f"Selected Point {len(selected_points)}: ({x}, {y})")

# Load video
video_path = 'output_videos/IMG_5152.MOV'
cap = cv2.VideoCapture(video_path)

ret, first_frame = cap.read()
if not ret:
    print("Failed to read video.")
    cap.release()
    exit()

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
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Select 4 Corners")

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Detect good features to track
features_to_track = cv2.goodFeaturesToTrack(
    prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=7
)

if features_to_track is None:
    print("Could not detect features. Exiting.")
    cap.release()
    exit()

# Save original corners
quad_pts = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Threshold for filtering moving points
motion_threshold = 5.0

# Process video
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

    # Draw static points
    for pt in static_pts:
        x, y = pt.ravel()
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Update tracked points or re-detect
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

    # Track the 4 manually selected corners using optical flow
    new_quad, quad_status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, quad_pts, None, **lk_params)
    if new_quad is not None and quad_status.sum() == 4:
        quad_pts = new_quad
        quad_int = quad_pts.astype(int)
        cv2.polylines(frame, [quad_int.reshape(-1, 2)], isClosed=True, color=(255, 0, 0), thickness=2)

    prev_gray = curr_gray

    cv2.imshow("Tracked Quadrilateral and Static Points", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()