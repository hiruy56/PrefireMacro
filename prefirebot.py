import time
import bettercam
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import Canvas
import threading
import queue
from screeninfo import get_monitors
import os
import numpy as np
import keyboard  # For detecting key presses
import pyautogui  # For simulating mouse clicks

class Overlay:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = None
        self.square_id = None

    def run(self, width, height, x, y):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'black')

        self.canvas = Canvas(self.root, bg='black', highlightthickness=0, cursor="none")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.bind("<Button-1>", lambda e: "break")

        self.square_id = self.canvas.create_rectangle(0, 0, width, height, outline='red', width=2)

        self.process_queue()
        self.root.mainloop()

    def show(self, width, height, x, y):
        self.clear()
        self.thread = threading.Thread(target=self.run, args=(width, height, x, y), daemon=True)
        self.thread.start()

    def process_queue(self):
        while not self.queue.empty():
            command, args = self.queue.get()
            command(*args)
        self.root.after(2, self.process_queue)

    def clear_canvas(self):
        for item in self.canvas.find_all():
            if item != self.square_id:
                self.canvas.delete(item)

    def draw_square(self, x1, y1, x2, y2, color='white', size=1):
        self.queue.put((self._draw_square, (x1, y1, x2, y2, color, size)))

    def _draw_square(self, x1, y1, x2, y2, color='white', size=1):
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=size)
        print(f"DEBUG: Drawing square on overlay at ({x1}, {y1}, {x2}, {y2})")

    def clear(self):
        if self.thread is not None and self.thread.is_alive():
            self.stop_overlay()
            self.thread.join()

    def stop_overlay(self):
        print("DEBUG: Stopping the overlay.")
        if self.root:
            self.root.quit()

def get_primary_display_resolution():
    monitors = get_monitors()
    for m in monitors:
        if m.is_primary:
            return m.width, m.height
    return 1920, 1080

# Function to save screenshot
def save_screenshot(image, count):
    screenshot_dir = 'screenshots'
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    file_path = os.path.join(screenshot_dir, f'screenshot_{count}.png')
    cv2.imwrite(file_path, image)
    print(f"DEBUG: Screenshot saved to {file_path}")

# Calculate Mean Squared Error (MSE) between two images
def mse(imageA, imageB):
    return np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) / float(imageA.shape[0] * imageA.shape[1])

# Load the YOLOv8 model
model = YOLO('wall.pt')

# Capture region dimensions
region_width = 420
region_height = 420

# Define the region for capturing (centered 420x420 pixels)
screen_width, screen_height = get_primary_display_resolution()
LEFT = (screen_width - region_width) // 2
TOP = (screen_height - region_height) // 2
RIGHT = LEFT + region_width
BOTTOM = TOP + region_height
region = (LEFT, TOP, RIGHT, BOTTOM)

print(f"DEBUG: Capture region: {region}")

# Initialize the overlay
overlay = Overlay()
overlay.show(region_width, region_height, LEFT, TOP)

# Start the camera
camera = bettercam.create(output_idx=0, output_color="BGRA")
camera.start(target_fps=60, video_mode=True, region=region)

screenshot_count = 0
last_screenshot = None
last_screenshot_time = 0
wall_detected = False  # Track wall detection state

# Threshold to determine whether two images are sufficiently different
mse_threshold = 200  # Lower value to make the system more sensitive to changes

while True:
    image = camera.get_latest_frame()

    # Convert the image from BGRA to BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Get the actual dimensions of the captured frame
    frame_height, frame_width = image.shape[:2]

    # Perform detection
    results = model(image)
    detections = results[0].boxes

    wall_detected_current = False

    for box in detections:
        if len(box.xyxy) > 0:
            x1, y1, x2, y2 = box.xyxy[0][:4]
            conf = box.conf[0]
            cls = box.cls[0]
            label = f'{model.names[int(cls)]} {conf:.2f}'

            # Check if confidence is >= 80% to show the overlay
            if model.names[int(cls)] == "detect walls" and conf >= 0.50:
                wall_detected_current = True
                wall_detected = True  # Set the wall detected flag

                overlay.clear_canvas()

                x1_screen = int((x1 / frame_width) * region_width)
                y1_screen = int((y1 / frame_height) * region_height)
                x2_screen = int((x2 / frame_width) * region_width)
                y2_screen = int((y2 / frame_height) * region_height)

                print(f"DEBUG: Detected wall with {conf:.2f} confidence - Bounding box (OpenCV): {x1}, {y1}, {x2}, {y2}")
                print(f"DEBUG: Scaled bounding box (Overlay): {x1_screen}, {y1_screen}, {x2_screen}, {y2_screen}")

                overlay.draw_square(x1_screen, y1_screen, x2_screen, y2_screen, color='red', size=2)

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # If the wall was detected but not anymore, simulate a left click if Alt is held
    if wall_detected and not wall_detected_current and keyboard.is_pressed('alt'):
        pyautogui.click()  # Simulate left click
        print("DEBUG: Simulated left click since wall disappeared while Alt is held.")

    if not wall_detected_current:
        overlay.clear_canvas()

    wall_detected = wall_detected_current  # Update the wall detected status
    cv2.imshow("Frame", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
