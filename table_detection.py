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