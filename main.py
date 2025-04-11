from read_webcam_stream import list_available_cameras, launch_webcam
from table_detection import detect_table_top
import cv2

def process_frame(frame):
    """
    Processes a single frame by detecting the table top.

    Parameters:
        frame (numpy.ndarray): The input frame from the webcam.

    Returns:
        numpy.ndarray: The processed frame with the table top detected.
    """
    return detect_table_top(frame)

def main():
    """
    Main function to list available cameras and launch the selected webcam.
    """
    # List all available cameras
    cameras = list_available_cameras()

    if not cameras:
        print("No cameras found.")
        return

    print("Available cameras:")
    for i, cam in enumerate(cameras):
        print(f"{i}: Camera Index {cam}")

    # Allow the user to select a camera
    try:
        selected_index = int(input("Select a camera index from the list above: "))
        if selected_index < 0 or selected_index >= len(cameras):
            print("Invalid selection.")
        else:
            # Launch the webcam with the processing function
            launch_webcam(camera_index=cameras[selected_index], frame_callback=process_frame)
    except ValueError:
        print("Invalid input. Exiting.")

if __name__ == "__main__":
    main()