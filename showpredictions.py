# Description: This script reads images and YOLO format annotations from the "images" and "labels" folders, respectively,
# and displays the images with red bounding boxes around the detected objects. The YOLO format annotations are read
# and parsed to draw the bounding boxes on the images. The script allows for pausing, navigating through images, and
# exiting the slideshow. The script uses OpenCV for image processing and display.
# This script assumes the images and labels are in their base folders, ie. it works on the folder structure downloaded from exportlabelstudio.py
# Usage: Run the script to display images with YOLO annotations.
#   SHORTCUTS:
#   - Space: Pause/Resume the slideshow
#   - Left arrow key: Navigate to the previous image (when paused)
#   - Right arrow key: Navigate to the next image (when paused)
#   - ESC: Exit the slideshow

import os
import cv2
import time

# Paths
base_folder = "/data/P-Prosjekter2/22660210_droner_sjofugl/yolo/ringreading2024/tobepublished/datasets/rf/"
images_folder = os.path.join(base_folder, "images")
labels_folder = os.path.join(base_folder, "labels")

# Function to draw red boxes on an image based on YOLO format annotations
def draw_boxes(image_path, label_path, target_width=1200):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    # Rescale image to target_width while maintaining aspect ratio
    original_height, original_width, _ = image.shape
    scale = target_width / original_width
    new_width = target_width
    new_height = int(original_height * scale)
    image = cv2.resize(image, (new_width, new_height))

    # Read YOLO annotations if available
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Parse each line in YOLO format: <class_id> <x_center> <y_center> <width> <height> <precision>
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"Invalid line: {line}")
                continue

            _, x_center, y_center, box_width, box_height, precision = map(float, parts)

            # Convert YOLO relative coordinates to pixel coordinates in the resized image
            x1 = int((x_center - box_width / 2) * new_width)
            y1 = int((y_center - box_height / 2) * new_height)
            x2 = int((x_center + box_width / 2) * new_width)
            y2 = int((y_center + box_height / 2) * new_height)

            # Draw red rectangle on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
            print(f"Detected object at ({x1}, {y1}) to ({x2}, {y2})")

    return image

# Main loop to process images with slideshow, pause, and navigation
def process_images():
    # Get a list of image files
    image_files = [f for f in sorted(os.listdir(images_folder)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print("No image files found in the images folder.")
        return

    idx = 0
    lastidx = 0
    paused = False

    while True:
        if not paused or idx != lastidx:
            # Get current image and label paths
            image_path = os.path.join(images_folder, image_files[idx])
            label_path = os.path.join(labels_folder, image_files[idx].rsplit(".", 1)[0] + ".txt")

            # Load and annotate the image
            annotated_image = draw_boxes(image_path, label_path)
            if annotated_image is None:
                idx = (idx + 1) % len(image_files)  # Skip to the next image if the current one cannot be loaded
                continue

            # Display the image
            lastidx = idx
            cv2.imshow("YOLO Detections", annotated_image)
            idx = (idx + 1) % len(image_files)

        # Wait for 100 ms or a key press
        key = cv2.waitKey(100 if not paused else 0)

        if key == 27:  # ESC key to exit
            print("Exiting...")
            break
        elif key == 32:  # Space key to pause/resume
            paused = not paused
        elif key == 81:  # Left arrow key
            if paused:
                idx = (idx - 1)
                print(idx)
        elif key == 83:  # Right arrow key
            if paused:
                idx = (idx + 1)
                print(idx)

# Run the script
process_images()

# Cleanup
cv2.destroyAllWindows()
