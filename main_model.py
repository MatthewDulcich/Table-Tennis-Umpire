import cv2
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from ball_event_train import crop_centered_with_padding
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


if __name__ == "__main__":

    file_name = input("Enter the path to the video file: ")
    video_path = f"./{file_name}.mp4"
    #retrieve the models
    ball_track_model = tf.keras.models.load_model("tracknet_pre_model.keras")
    ball_event_model = tf.keras.models.load_model("ball_event_model.keras")
    target_size = (320, 220)
    while not os.path.exists(video_path):
        file_name = input("Enter the path to the video file: ")
        video_path = f"./{file_name}.mp4"
    #extract the downsampled frames
    down_frames, orig_frames = downsample_video(video_path, target_size)
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
    
    cropped_frames = []
    for i, frame in enumerate(orig_frames):
        x, y = ball_position[i]
        cropped_frame = crop_centered_with_padding(frame, (x, y), (320, 220))
        cropped_frames.append(cropped_frame)
    
    # events = []
    # for i, frame in enumerate(cropped_frames):
    #     pred_event = ball_event_model.predict(np.expand_dims(frame, axis=0))
    #     print(pred_event)
    #     event = np.argmax(pred_event[0])
    #     events.append(event)
    #     print(events[-1])
    
    for i, frame in enumerate(orig_frames):
        if i >100:
            cv2.circle(frame, (int(ball_position[i][0]), int(ball_position[i][1])), 3, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # TODO: Replace with writing out to a video
    

    #Annotate original frames given some sample
    # 100-200 frames of active gameplay
    #evaluate the results
    #Test video - on slides as backup
    #OOF events
    #Generalize to live footage - input as image from camera automatically update
    #identify throughput and acceptable latency 
