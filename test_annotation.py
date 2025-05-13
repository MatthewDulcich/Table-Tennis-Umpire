import cv2
import json

# File paths
video_path = "data/train/game_1_down.mp4"
json_path = "data/train/game_1/ball_markup_down.json"
output_video_path = "output_videos/annotated_game_1_down.mp4"

# Load the JSON file
with open(json_path, "r") as f:
    ball_data = json.load(f)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create a VideoWriter object to save the annotated video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check if the current frame has ball coordinates in the JSON
    if str(frame_index) in ball_data:
        coords = ball_data[str(frame_index)]
        x, y = coords["x"], coords["y"]

        # Draw a circle at the ball's position
        cv2.circle(frame, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

        # Optionally, add text to display the coordinates
        cv2.putText(frame, f"({int(x)}, {int(y)})", (int(x) + 15, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Write the annotated frame to the output video
    out.write(frame)
    frame_index += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved to {output_video_path}")