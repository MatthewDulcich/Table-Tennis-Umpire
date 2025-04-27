# Table Tennis Umpire

## Overview
The **Table Tennis Umpire** project leverages computer vision to assist in umpiring table tennis matches. It uses the YOLO (You Only Look Once) object detection model to track the ball and players, providing insights and decisions in real-time.

---

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/Table-Tennis-Umpire.git
    cd Table-Tennis-Umpire
    ```
2. Download the YOLO model weights:
    - [Download YOLOv5nu.pt](https://example.com/yolo5nu.pt) and place it in the `models/` directory.
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
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

---

## Features
- Real-time ball and player tracking.
- Decision-making based on game rules.
- Support for both live webcam feeds and pre-recorded videos.

---

## Output Directory
Processed videos and logs are saved in the `output/` directory. Ensure this directory exists or is created during runtime.

---

## Dependencies
The project requires the following Python libraries (Find the rest in the `environment.yaml`):
- OpenCV
- TensorFlow
- NumPy
- Matplotlib

Create and activate a conda environment, then install the dependencies using:
```bash
conda env create -f environment.yaml -n table-tennis-umpire
conda activate table-tennis-umpire
```

---

## Known Issues
- FPS discrepancies may occur on lower-end hardware.
- Frame dropping might happen with high-resolution videos.

---

## Future Enhancements
- Add support for doubles matches.
- Integrate audio feedback for decisions.
