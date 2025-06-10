import cv2
import csv
import time
import keyboard
import pyautogui
import numpy as np

# === Config ===
LABEL = 'one'  # Change as needed
FPS = 120         # Fixed FPS for coordinate capture
CSV_PATH = 'raw_sequences.csv'

sequence = []
last_capture_time = time.time()
recording = False

print(f"Hold SPACE to start recording '{LABEL}' gesture")
print("Release SPACE to save sequence")
print("Press 'Esc' to quit")

# Create a blank window to draw mouse path
window_size = (1280, 720)
canvas = 255 * np.ones((window_size[1], window_size[0], 3), dtype=np.uint8)

cv2.namedWindow("Mouse Tracking")
cv2.moveWindow("Mouse Tracking", 0, 0)  # Move window to top-left corner of the screen

while True:
    current_time = time.time()
    space_pressed = keyboard.is_pressed('space')

    # Start recording on space pressed
    if space_pressed:
        if not recording:
            print("Recording started...")
            recording = True
            sequence = []
            canvas[:] = 255  # Clear canvas

        if (current_time - last_capture_time) >= 1.0 / FPS:
            last_capture_time = current_time
            x, y = pyautogui.position()

            # Since window is at (0,0), relative coordinates equal screen coordinates
            rel_x = x
            rel_y = y

            # Clamp coordinates inside canvas window
            rel_x = min(max(rel_x, 0), window_size[0] - 1)
            rel_y = min(max(rel_y, 0), window_size[1] - 1)
            sequence.append((rel_x, rel_y))

    # Draw recorded path on canvas
    if sequence:
        for i in range(1, len(sequence)):
            cv2.line(canvas, sequence[i - 1], sequence[i], (0, 255, 0), 2)
        cv2.circle(canvas, sequence[-1], 5, (0, 0, 255), -1)

    # Stop recording on space released
    if not space_pressed and recording:
        recording = False
        if sequence:
            row = [LABEL]
            for pt in sequence:
                row.extend(pt)
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Saved sequence with {len(sequence)} points.")
        else:
            print("No data captured.")
        sequence = []

    cv2.imshow("Mouse Tracking", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
