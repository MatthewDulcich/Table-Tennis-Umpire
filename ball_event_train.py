import cv2
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np

def get_frames_set(ball_json, event_json):
    #open ball positions
    with open(ball_json, 'r') as f:
        ball_data = json.load(f)
    #open event positions and transform classes to model output value
    with open(event_json, 'r') as f:
        events = {"bounce": np.array([1,0,0]), "net": np.array([0,1,0]), "empty_event": np.array([0,0,1])}
        event_data = json.load(f)
        event_data = {frame: events[event] for frame, event in event_data.items()}
    #find intersection b/w keys of ball and event data 
    #commonality acts as the training set
    train_frames = set(event_data.keys()) & set(ball_data.keys())
    ball_data = {frame: ball_data[frame] for frame in train_frames}
    event_data = {frame: event_data[frame] for frame in train_frames}
    print(f"Train frames: {len(train_frames)}")
    print(f"Ball data: {len(ball_data)}")
    print(f"Event data: {len(event_data)}")
    return train_frames, ball_data, event_data

def crop_centered_with_padding(image, center, crop_size):
    img_h, img_w = image.shape[:2]
    crop_w, crop_h = crop_size
    cx, cy = center

    # Calculate crop box coordinates
    x1 = int(cx - crop_w // 2)
    y1 = int(cy - crop_h // 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Determine padding if crop box is outside image boundaries
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    # Adjust crop box to fit inside image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Crop the valid region
    cropped = image[y1:y2, x1:x2]

    # Pad to maintain the same size
    cropped_padded = cv2.copyMakeBorder(
        cropped,
        pad_top, pad_bottom,
        pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # black padding
    )

    return cropped_padded

def extract_specific_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    extracted_frames = {}

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return extracted_frames

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = cap.read()
        if success:
            extracted_frames[index] = frame
        else:
            print(f"Warning: Could not read frame {index}")

    cap.release()
    return extracted_frames

def data_preprocess(video_path, ball_json, event_json):
    # Step 1: Get frames set
    train_frames, ball_data, event_data = get_frames_set(ball_json, event_json)
    
    train_frames = list(map(int, event_data.keys()))
    print("HERE ARE THE TRAIN FRAMES")
    #print(train_frames)
    frames = extract_specific_frames(video_path, train_frames)
    # Step 3: Center crop frames
    cropped_frames = []
    for frame in frames:
        frame_num = str(frame)
        center = (int(ball_data[frame_num]['x']), int(ball_data[frame_num]['y']))
        cropped_frame = crop_centered_with_padding(frames[frame], center, (320, 220))
        cropped_frames.append(cropped_frame)
    print("Frames all cropped around ball coordinate")
    return cropped_frames, ball_data, event_data

# Step 3: Model
def create_event_predictor_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="softmax"),
        layers.Dense(3)  # bounce, net, empty_event
    ])
    lr_scheduler = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100,
        decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Step 4: Training
def train_event_predictor(frames, event_data, model_save_path="ball_event_model.h5", batch_size=16, epochs=10):
    #dataset = BallTrackingDataset(video_path, label_json_path, batch_size=batch_size)
    # print("TRAIN EVENT PREDICTOR")
    # print(len(frames), len(event_data))
    # print(f"Frames shape: {frames[0].shape}")

    #convert to usable format
    frames = np.array(frames, dtype=np.float32) / 255.0
    event_data = np.array(list(event_data.values()), dtype=np.float32)
    
    if os.path.exists(model_save_path):
        print("LOADING EXISTING MODEL")
        model = tf.keras.models.load_model(model_save_path)
        print(f"Model loaded from {model_save_path}")
    else:
        print("CREATING NEW MODEL")
        model = create_event_predictor_model(input_shape=(320, 220, 3))
    model.fit(frames, event_data, batch_size=batch_size, epochs=epochs)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

# Full Pipeline
def full_pipeline(video_path, ball_json_path, event_json_path):
    frames, ball_data, event_data = data_preprocess(video_path, ball_json_path, event_json_path)
    train_event_predictor(frames, event_data)

folder = "./data/train/game_"
for i in range(1,6):
    if i == 2:
        continue
    full_pipeline(
        video_path = folder + f"{i}.mp4",
        ball_json_path = folder + f"{i}/ball_markup.json",
        event_json_path = folder + f"{i}/events_markup.json"
    )

