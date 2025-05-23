import cv2
import time
from typing import Callable, Optional, Union, List

def list_available_cameras(max_cameras: int = 20) -> List[int]:
    """
    Lists all available cameras by attempting to open them.

    Parameters:
        max_cameras (int): The maximum number of camera indices to check.

    Returns:
        list: A list of available camera indices.
    """
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

def launch_webcam(
    camera_index: Union[int, str] = 0,
    frame_callback: Optional[Callable[[any], any]] = None
) -> None:
    """
    Launches the webcam and processes frames using an optional callback function.
    Press 'q' to quit the video stream.

    Parameters:
        camera_index (int or str): The index or device path of the webcam.
        frame_callback (function): A function to process each frame.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open webcam with index/path {camera_index}.")
        return

    print("Press 'esc' to quit the video stream.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Warning: Failed to capture frame. Retrying...")
            time.sleep(0.1)  # Wait briefly before retrying
            continue

        # Process the frame using the callback function, if provided
        if frame_callback:
            try:
                frame = frame_callback(frame)
            except Exception as e:
                print(f"Error during frame processing: {e}")
                continue

        # Display the frame
        cv2.imshow("Webcam Stream", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for 'esc'
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # List all available cameras
    cameras = list_available_cameras()

    if not cameras:
        print("No cameras found.")
    else:
        print("Available cameras:")
        for i, cam in enumerate(cameras):
            print(f"{i}: Camera Index {cam}")

        # Allow the user to select a camera
        try:
            selected_index = int(input("Select a camera index from the list above: "))
            if selected_index < 0 or selected_index >= len(cameras):
                print("Invalid selection.")
            else:
                launch_webcam(camera_index=cameras[selected_index])
        except ValueError:
            print("Invalid input. Exiting.")
