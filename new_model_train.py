import cv2
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
print("Eager execution enabled: ", tf.executing_eagerly())
scale_x, scale_y = 1, 1
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    #check if video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    #extract video resolution and fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, width, height, fps
'''
Script to take a video and convert it to downsample size and save it
The labels in the test files are also downsized
'''
def process_video_and_labels(video_path, input_json_path, output_video_path, output_json_path, target_size=(320, 220)):
    #check if downsampled video exists
    global scale_x, scale_y
    #get original video properties
    cap, original_width, original_height, fps = extract_video_features(video_path)
    # get scaling from original to target size
    scale_x = target_size[0] / original_width
    scale_y = target_size[1] / original_height
    if not os.path.exists(output_video_path):
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)
        while True:
            #extract video frames
            ret, frame = cap.read()
            if not ret:
                break
            #resize video frames and write to output video
            resized = cv2.resize(frame, target_size)
            out.write(resized)

        out.release()
        print(f"Saved resized video to: {output_video_path}")
    cap.release()
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

# # Step 2: Define Dataset Class
# class BallTrackingDataset(tf.keras.utils.Sequence):
#     def __init__(self, video_path, label_json, batch_size=16, target_size=(320, 220)):
#         self.cap = cv2.VideoCapture(video_path)
#         self.labels = json.load(open(label_json))
#         self.batch_size = batch_size
#         self.target_size = target_size

#         self.frames = []
#         self.targets = []
#         frame_num = 0

#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#             resized = cv2.resize(frame, target_size)
#             key = str(frame_num)
#             if key in self.labels and self.labels[key] is not None:
#                 self.frames.append(resized)
#                 self.targets.append([self.labels[key]['x'], self.labels[key]['y']])
#             frame_num += 1

#         self.cap.release()
#         self.frames = np.array(self.frames, dtype=np.float32) / 255.0
#         self.targets = np.array(self.targets, dtype=np.float32)

#     def __len__(self):
#         return int(np.ceil(len(self.frames) / self.batch_size))
    
#     def __getitem__(self, idx):
#         start = idx * self.batch_size
#         end = min((idx + 1) * self.batch_size, len(self.frames))
#         return self.frames[start:end], self.targets[start:end]

# Step 3: Model
def create_ball_predictor_model(input_shape=(224, 224, 3)):
    # model = models.Sequential([
    #     layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    #     layers.GlobalAveragePooling2D(),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(2)  # x and y
    # ])
    inputs = tf.keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D()(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D()(conv2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.GlobalAveragePooling2D()(conv3)
    dense1 = layers.Dense(64, activation='relu')(pool3)
    outputs = layers.Dense(2)(dense1)
    
    lr_scheduler = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100,
        decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    # model.compile(optimizer=optimizer, loss='mean_squared_error')
    model = models.Model(inputs=inputs, outputs=outputs, name = "BallPredictor")
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


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

# Step 4: Training
def train_ball_predictor(video_path, label_json_path, model_save_path="ball_tracker_model.keras", batch_size=16, epochs=10):
    print("Crgeating dataset...")
    #dataset = BallTrackingDataset(video_path, label_json_path, batch_size=batch_size, target_size=(320, 220))
    with open(label_json_path, 'r') as f:
        data = json.load(f)
    #print(data.values())
    frames = list(map(int, data.keys()))
    extracted_frames = extract_specific_frames(video_path, frames)
    output = [list(coord.values()) for coord in data.values()]
    if os.path.exists(model_save_path):
        print("LOADING EXISTING MODEL")
        model = tf.keras.models.load_model(model_save_path)
        print(f"Model loaded from {model_save_path}")
    else:
        print("CREATING NEW MODEL")
        model = create_ball_predictor_model(input_shape=(220, 320, 3))
    print("Training model...")
    # model.fit(dataset, epochs=epochs)
    frames = np.array(list(extracted_frames.values()), dtype=np.float32) / 255.0
    frames =tf.convert_to_tensor(frames)
    output = np.array(output, dtype=np.float32)
    output = tf.convert_to_tensor(output)
    print(frames.shape)
    model.fit(frames, output, epochs=epochs, batch_size=batch_size)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

# Full Pipeline
def full_pipeline(video_path, input_json_path, output_video_path, output_json_path):
    process_video_and_labels(video_path, input_json_path, output_video_path, output_json_path)
    train_ball_predictor(output_video_path, output_json_path)

# Example Usage
folder = "./data/train/game_"
for i in range(1,6):
    if i == 2:
        continue
    full_pipeline(
        video_path = folder + f"{i}.mp4",
        input_json_path = folder + f"{i}/ball_markup.json",
        output_video_path = folder + f"{i}_down.mp4",
        output_json_path = folder + f"{i}/ball_markup_down.json"
    )
