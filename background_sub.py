import cv2
import numpy as np

def background_subtraction():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    back_sub = cv2.createBackgroundSubtractorMOG2()

    # Let the background model warm up
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            return
        back_sub.apply(frame)

    print("Press 'q' to quit. Click on the ball in the window to print HSV value.")

    def print_hsv(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
            print(f"HSV at ({x}, {y}): {hsv[y, x]}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.setMouseCallback("Frame", print_hsv, frame)

        # Background subtraction
        fg_mask = back_sub.apply(blurred_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # 🎯 Adjusted HSV range for light-colored balls (tweak if needed)
        lower_ball = np.array([0, 0, 170])
        upper_ball = np.array([180, 90, 255])
        color_mask = cv2.inRange(hsv_frame, lower_ball, upper_ball)

        # ✅ Combine foreground and color masks only (removes noisy adaptive threshold)
        combined_mask = cv2.bitwise_or(fg_mask, color_mask)

        # Optional: refine the mask to reduce noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30 or area > 8000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if 0.5 <= aspect_ratio <= 1.5:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
                if circularity < 0.5:
                    continue

                # Draw rectangle and center
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center = (x + w // 2, y + h // 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                print(f"Detected ball at: {center}, area={area:.1f}, circ={circularity:.2f}")

        cv2.imshow("Frame", frame)
        cv2.imshow("Combined Mask", combined_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Running ping pong ball detection...")
    background_subtraction()

if __name__ == "__main__":
    main()