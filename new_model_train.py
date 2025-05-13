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
            resized = cv2.resize(frame, target_size)  # target_size is (width, height)
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

def create_ball_predictor_TrackNet(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    conv_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    bn = layers.BatchNormalization()(conv_1)
    conv_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(bn)
    bn_2 = layers.BatchNormalization()(conv_2)

    maxpool_1 = layers.MaxPooling2D()(bn_2)
    conv_3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(maxpool_1)
    bn_3 = layers.BatchNormalization()(conv_3)
    conv_4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn_3)
    bn_4 = layers.BatchNormalization()(conv_4)

    maxpool_2 = layers.MaxPooling2D()(bn_4)
    conv_5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(maxpool_2)
    bn_5 = layers.BatchNormalization()(conv_5)
    conv_6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn_5)
    bn_6 = layers.BatchNormalization()(conv_6)
    conv_7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn_6)
    bn_7 = layers.BatchNormalization()(conv_7)

    maxpool_3 = layers.MaxPooling2D()(bn_7)
    conv_8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(maxpool_3)
    bn_8 = layers.BatchNormalization()(conv_8)
    conv_9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn_8)
    bn_9 = layers.BatchNormalization()(conv_9)
    conv_10 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn_9)
    bn_10 = layers.BatchNormalization()(conv_10)
    upsample_1 = layers.UpSampling2D()(bn_10)
    upsample_1 = layers.ZeroPadding2D(((1, 0), (0, 0)))(upsample_1)  # Pads 1 row at top

    concatenate_1 = layers.Concatenate()([upsample_1, bn_7])
    conv_11 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concatenate_1)
    bn_11 = layers.BatchNormalization()(conv_11)
    conv_12 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn_11)
    bn_12 = layers.BatchNormalization()(conv_12) 
    conv_13 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn_12)
    bn_13 = layers.BatchNormalization()(conv_13)
    upsample_2 = layers.UpSampling2D()(bn_13)
    
    concatenate_2 = layers.Concatenate()([upsample_2, bn_4])
    conv_14 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concatenate_2)
    bn_14 = layers.BatchNormalization()(conv_14)
    conv_15 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn_14)
    bn_15 = layers.BatchNormalization()(conv_15)
    upsample_3 = layers.UpSampling2D()(bn_15)

    concatenate_3 = layers.Concatenate()([upsample_3, bn_2])
    conv_16 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concatenate_3)
    bn_16 = layers.BatchNormalization()(conv_16)
    conv_17 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn_16)
    bn_17 = layers.BatchNormalization()(conv_17)
    sigconv=layers.Conv2D(64, (1, 1), activation='sigmoid', padding='same')(bn_17)

    #method for extracting coordinates
    conv18 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(sigconv)
    conv19 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv18)
    conv20 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv19)
    pool3 = layers.GlobalAveragePooling2D()(conv20)
    dense1 = layers.Dense(64, activation='relu')(pool3)
    dense2 = layers.Dense(32, activation='relu')(dense1)
    dense3 = layers.Dense(16, activation='relu')(dense2)
    outputs = layers.Dense(2)(dense3)
    lr_scheduler = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100,
        decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    # model.compile(optimizer=optimizer, loss='mean_squared_error')
    model = models.Model(inputs=inputs, outputs=outputs, name = "BallPredictor")
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print(model.summary())
    return model

# def create_ball_predictor_TrackNet(input_shape=(220, 320, 3)):
#     inputs = tf.keras.Input(shape=input_shape)
#     x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(32, activation='relu')(x)
#     outputs = layers.Dense(2)(x)  # Output (x, y) coordinates

#     model = models.Model(inputs=inputs, outputs=outputs, name="BallPredictorTest")
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# Step 3: Model
def create_ball_predictor_model(input_shape=(224, 224, 3)):
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

def create_model_predictor(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)

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
def train_ball_predictor(video_path, label_json_path, model_save_path="ball_tracker_model.keras", trackNet_save_path = "tracknet_pre_model.keras", batch_size=16, epochs=3):
    print("Creating dataset...")
    #dataset = BallTrackingDataset(video_path, label_json_path, batch_size=batch_size, target_size=(320, 220))
    with open(label_json_path, 'r') as f:
        data = json.load(f)
    #print(data.values())
    frames = list(map(int, data.keys()))
    extracted_frames = extract_specific_frames(video_path, frames)
    sample_frame = next(iter(extracted_frames.values()))
    print("Frame shape example:", sample_frame.shape)  # Should be (220, 320, 3)
    output = [list(coord.values()) for coord in data.values()]
    # if os.path.exists(model_save_path):
    #     print("LOADING EXISTING MODEL")
    #     model = tf.keras.models.load_model(model_save_path)
    #     print(f"Model loaded from {model_save_path}")
    # else:
    #     print("CREATING NEW MODEL")
    #     model = create_ball_predictor_model(input_shape=(220, 320, 3))

    if os.path.exists(trackNet_save_path):
        print("LOADING EXISTING MODEL")
        model_tracknet = tf.keras.models.load_model(trackNet_save_path)
        print(f"Model loaded from {trackNet_save_path}")
    else:
        print("CREATING NEW MODEL")
        model_tracknet = create_ball_predictor_TrackNet(input_shape=(220, 320, 3))
    print("Training model...")
    # model.fit(dataset, epochs=epochs)
    frames = np.array(list(extracted_frames.values()), dtype=np.float32) / 255.0
    frames =tf.convert_to_tensor(frames)
    
    output = np.array(output, dtype=np.float32)
    output = tf.convert_to_tensor(output)
    print(frames.shape)
    model_tracknet.fit(frames, output, epochs=epochs, batch_size=batch_size)
    model_tracknet.save(trackNet_save_path)
    print(f"Model saved to {trackNet_save_path}")

# Full Pipeline
def full_pipeline(video_path, input_json_path, output_video_path, output_json_path):
    process_video_and_labels(video_path, input_json_path, output_video_path, output_json_path)
    train_ball_predictor(output_video_path, output_json_path)

# Example Usage
if __name__ == "__main__":
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
