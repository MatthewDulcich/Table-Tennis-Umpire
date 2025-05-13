# Table Tennis Umpire

## Overview
The **Table Tennis Umpire** project leverages computer vision and deep learning to assist in umpiring table tennis matches. It uses models like YOLO for object detection and custom TensorFlow models for ball tracking and event detection. The system provides real-time insights and decisions, supporting both live webcam feeds and pre-recorded videos.

---

## Features
- **Real-time Ball and Player Tracking**:
  - Detects and tracks the ball and players during matches.
- **Event Detection**:
  - Identifies key events such as ball bounces, net hits, and empty frames.
- **Optical Flow Support**:
  - Enhances tracking accuracy using Lucas-Kanade optical flow.
- **Webcam and Video File Support**:
  - Works with live webcam feeds or pre-recorded video files.
- **Customizable Annotations**:
  - Annotates frames with ball positions, event labels, and quadrilateral boundaries.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/Table-Tennis-Umpire.git
cd Table-Tennis-Umpire
```

### 2. Download Model Weights
- Download the following model weights:
  - [YOLOv5nu.pt](https://example.com/yolo5nu.pt) for player and ball detection.
  - `ball_tracker_model.keras` for ball position prediction.
  - `ball_event_model.keras` for event detection.
- Place the weights in the `models/` directory.

### 3. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

Alternatively, create and activate a conda environment:
```bash
conda env create -f environment.yaml -n table-tennis-umpire
conda activate table-tennis-umpire
```

---

## Usage

### Webcam Mode
Run the application using your webcam:
```bash
python main.py
```

### Video File Mode
Process a video file:
```bash
python main.py --video
```

### Optical Flow Mode
Enable optical flow for enhanced tracking:
```bash
python main.py --opticalflow
```

---

## Output Directory
- Processed videos and logs are saved in the `output/` directory.
- Ensure this directory exists or is created during runtime.

---

## Dependencies
The project requires the following Python libraries:
- OpenCV
- TensorFlow
- NumPy
- Matplotlib
- tqdm

Refer to `requirements.txt` or `environment.yaml` for the full list of dependencies.

---

## Known Issues
- **FPS Discrepancies**:
  - May occur on lower-end hardware.
- **Frame Dropping**:
  - Might happen with high-resolution videos.
- **Stuttering in Webcam Mode**:
  - Ensure consistent frame dimensions and scaling factors to avoid stuttering.

---

## Future Enhancements
- Add support for doubles matches.
- Integrate audio feedback for decisions.
- Improve event detection accuracy with additional training data.
- Add support for multi-camera setups.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
