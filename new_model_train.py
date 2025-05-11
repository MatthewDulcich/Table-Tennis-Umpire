import cv2
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np

'''
Script to take a video and convert it to downsample size and save it
The labels in the test files are also downsized
'''
def process_video_and_labels(video_path, input_json_path, output_video_path, output_json_path, target_size=(640, 360)):
    #check if downsampled video exists
    if not os.path.exists(output_video_path):
        cap = cv2.VideoCapture(video_path)
        #check if video opened successfully
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        #extract video resolution and fps
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)
        # Calculate scaling factors
        scale_x = target_size[0] / original_width
        scale_y = target_size[1] / original_height

        while True:
            #extract video frames
            ret, frame = cap.read()
            if not ret:
                break
            #resize video frames and write to output video
            resized = cv2.resize(frame, target_size)
            out.write(resized)

        cap.release()
        out.release()
        print(f"Saved resized video to: {output_video_path}")
    else:
        scale_x = scale_y = 1  # Assume no resize if video already exists
    
    #Check if dowmsampled labels exist
    if not os.path.exists(output_json_path):
        with open(input_json_path, 'r') as f:
            data = json.load(f)

        scaled_data = {
            frame: None if coords is None else {
                "x": coords["x"] * scale_x,
                "y": coords["y"] * scale_y
            }
            for frame, coords in data.items()
        }

        with open(output_json_path, 'w') as f:
            json.dump(scaled_data, f, indent=4)
        print(f"Saved scaled labels to: {output_json_path}")

# Step 2: Define Dataset Class
class BallTrackingDataset(tf.keras.utils.Sequence):
    def __init__(self, video_path, label_json, batch_size=16, target_size=(224, 224)):
        self.cap = cv2.VideoCapture(video_path)
        self.labels = json.load(open(label_json))
        self.batch_size = batch_size
        self.target_size = target_size

        self.frames = []
        self.targets = []
        frame_num = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, target_size)
            key = str(frame_num)
            if key in self.labels and self.labels[key] is not None:
                self.frames.append(resized)
                self.targets.append([self.labels[key]['x'], self.labels[key]['y']])
            frame_num += 1

        self.cap.release()
        self.frames = np.array(self.frames, dtype=np.float32) / 255.0
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return int(np.ceil(len(self.frames) / self.batch_size))
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.frames))
        return self.frames[start:end], self.targets[start:end]

# Step 3: Model
def create_ball_predictor_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)  # x and y
    ])
    lr_scheduler = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100,
        decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Step 4: Training
def train_ball_predictor(video_path, label_json_path, model_save_path="ball_tracker_model.h5", batch_size=16, epochs=10):
    print("Creating dataset...")
    dataset = BallTrackingDataset(video_path, label_json_path, batch_size=batch_size)
    if os.path.exists(model_save_path):
        model = tf.keras.models.load_model(model_save_path)
        print(f"Model loaded from {model_save_path}")
    else:
        model = create_ball_predictor_model(input_shape=(640, 360, 3))
    model.fit(dataset, epochs=epochs)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

# Full Pipeline
def full_pipeline(video_path, input_json_path, output_video_path, output_json_path):
    process_video_and_labels(video_path, input_json_path, output_video_path, output_json_path)
    train_ball_predictor(output_video_path, output_json_path)

# Example Usage
folder = "./data/train/game_"
for i in range(2,6):
    full_pipeline(
        video_path = folder + f"{i}.mp4",
        input_json_path = folder + f"{i}/ball_markup.json",
        output_video_path = folder + f"{i}_down.mp4",
        output_json_path = folder + f"{i}/ball_markup_down.json"
    )
