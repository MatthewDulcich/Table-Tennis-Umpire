import cv2
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from ball_event_train import crop_centered_with_padding, extract_specific_frames, data_preprocess

file_name = input("Enter the path to the video file: ")
video_path = f"./data/test/{file_name}.mp4"

target_size = (320, 220)
while not os.path.exists(video_path):
    file_name = input("Enter the path to the video file: ")
    video_path = f"./data/test/{file_name}.mp4"

# #retrieve the models
ball_event_model = tf.keras.models.load_model("ball_event_model.keras")
# ball_json = f"./data/test/{file_name}/ball_markup.json"
# event_json = f"./data/test/{file_name}/events_markup.json"
# cropped_frames, ball_data, events_data =data_preprocess(video_path, ball_json, event_json)

with open(f"./data/test/{file_name}/ball_markup.json", 'r') as f:
    ball_data = json.load(f)

frame_nums = list(map(int, ball_data.keys()))
#print(frame_nums)
num_of_frames = 10
fps_out = 10

frames = extract_specific_frames(video_path, frame_nums[0:num_of_frames])
print("Frames extracted")
# cv2.imshow("frame", frames[frame_nums[0]])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cropped_frames = []
for frame in frames:
    frame_num = str(frame)
    center = (int(ball_data[frame_num]['x']), int(ball_data[frame_num]['y']))
    if frame & 20 == 0:
        print(frame_num, center)
    cropped_frame = crop_centered_with_padding(frames[frame], center, (320, 220))
    cropped_frames.append(cropped_frame)

events = []
event_labels = {0:"bounce", 1:"net", 2:"empty_event"}
for i, frame in enumerate(cropped_frames):
    pred_event = ball_event_model.predict(np.expand_dims(frame, axis=0))
    print(pred_event)
    event = np.argmax(pred_event[0])
    print(event)
    events.append(event_labels[event])
    print(events[-1])


def place_list_in_frame(img, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=1, thickness=2, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Define rectangle coordinates
    x, y = position
    top_left = (x - 10, y - text_height - 10)
    bottom_right = (x + text_width + 10, y + 10)

    # Draw background rectangle
    cv2.rectangle(img, top_left, bottom_right, bg_color, -1)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 2)  # frame border

    # Put the text on top of the rectangle
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

    return img

dest_video_path = f"./data/test/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(dest_video_path, fourcc, 10, target_size)
for i, frame in enumerate(frames):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # resized = cv2.resize(frame, target_size)
    # orig_frames.append(frame)
    # down_frames.append(resized)
    # print(resized.shape)
    # count += 1
    frame = place_list_in_frame(frame, events[i], position=(10, 20))
    out.write(frame)
print("Video path: ", dest_video_path)
out.release()