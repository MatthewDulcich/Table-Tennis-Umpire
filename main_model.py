import cv2
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from ball_event_train import crop_centered_with_padding, extract_specific_frames

scale_x, scale_y = 1, 1
def downsample_video(video_path, target_size=(320, 220)):
    global scale_x, scale_y
    # Create video writer
    cap = cv2.VideoCapture(video_path)
    scale_x = target_size[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    scale_y = target_size[1] / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(scale_x, scale_y)
    ret = True
    down_frames = []
    orig_frames =[]
    count = 0
    while ret and count < 1000:
        # Extract video frames
        ret, frame = cap.read()
        if not ret:
            break
        # Resize video frames and write to output video
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame, target_size)
        orig_frames.append(frame)
        down_frames.append(resized)
        #print(resized.shape)
        count += 1
    return down_frames, orig_frames

def place_event_in_frame(img, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX,
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
if __name__ == "__main__":

    file_name = input("Enter the path to the video file: ")
    video_path = f"./data/test/{file_name}.mp4"
    #retrieve the models
    ball_track_model = tf.keras.models.load_model("tracknet_pre_model.keras")
    ball_event_model = tf.keras.models.load_model("ball_tracker_model.keras")
    target_size = (320, 220)
    while not os.path.exists(video_path):
        file_name = input("Enter the path to the video file: ")
        video_path = f"./data/test/{file_name}.mp4"
    #extract the downsampled frames
    down_frames, frames = downsample_video(video_path, target_size)
    ball_position = []
    for i, frame in enumerate(down_frames):
        pred_pos = ball_track_model.predict(np.expand_dims(frame, axis=0)/ 255.0)
        print("Predicted position:")
        print(pred_pos[0][0], pred_pos[0][1])
        orig_x = int(float(pred_pos[0][0]) / float(scale_x))
        orig_y = int(float(pred_pos[0][1]) / float(scale_y))
        print(orig_x, orig_y)
        ball_position.append((orig_x, orig_y))
        print(ball_position[-1])
    
    # orig_frames = extract_specific_frames(video_path,)
    # with open(f"./data/test/{file_name}/ball_markup.json", "r") as f:
    #     ball_position = json.load(f)
    
    cropped_frames = []
    events = []
    event_labels = {0:"bounce", 1:"net", 2:"empty_event"}
    for i, frame in enumerate(frames):
        x, y = ball_position[i]
        cropped_frame = crop_centered_with_padding(frame, (x, y), (320, 220))
        cropped_frames.append(cropped_frame)
        pred_event = ball_event_model.predict(np.expand_dims(cropped_frame, axis=0))
        #print(pred_event)
        event = np.argmax(pred_event[0])
        events.append(event_labels[event])
        #print(events[-1])
    # writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, target_size)
    # for i, frame in enumerate(orig_frames):
    #     writer.write(frame)
    
    # TODO: Replace with writing out to a video
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dest_video_path = f"./data/test/{file_name}/output_ball.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dest_video_path, fourcc, fps, (width, height))
    for i, frame in enumerate(frames):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resized = cv2.resize(frame, target_size)
        # orig_frames.append(frame)
        # down_frames.append(resized)
        # print(resized.shape)
        # count += 1
        # text= f"({int(ball_position[i][0])}, {int(ball_position[i][1])})"
        frame = place_event_in_frame(frame, events[i], position=(10, 20))
        cv2.circle(frame, (int(ball_position[i][0]), int(ball_position[i][1])), 5, (0, 255, 0), -1)

        out.write(frame)
    print("Video path: ", dest_video_path)
    out.release()
    #Annotate original frames given some sample
    # 100-200 frames of active gameplay
    #evaluate the results
    #Test video - on slides as backup
    #OOF events
    #Generalize to live footage - input as image from camera automatically update
    #identify throughput and acceptable latency 
