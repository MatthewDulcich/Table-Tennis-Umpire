import cv2
import numpy as np

def detect_table_top(frame):
    """
    Detects the table top in a given frame and draws a rectangle around it.

    Parameters:
        frame (numpy.ndarray): The input frame from the webcam.

    Returns:
        numpy.ndarray: The frame with the rectangle drawn around the detected table top.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find the table top
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 sides (rectangle)
        if len(approx) == 4:
            # Draw the contour on the original frame
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

            # Draw a bounding box around the table
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            break  # Assuming there's only one table top to detect

    return frame

def segment_table_top(frame, lower_hsv, upper_hsv):
    """
    Segments the table top in a given frame using color-based segmentation.

    Parameters:
        frame (numpy.ndarray): The input frame from the webcam.
        lower_hsv (tuple): The lower HSV bounds for segmentation.
        upper_hsv (tuple): The upper HSV bounds for segmentation.

    Returns:
        numpy.ndarray: A binary mask where the table top is segmented.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask based on the HSV range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Optional: Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask