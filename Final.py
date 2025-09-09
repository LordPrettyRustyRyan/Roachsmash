import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pygame
import random
import os

# Initialize the HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize pygame for audio playback
pygame.mixer.init()

# Function to generate a new image with random initial position and speed
def generate_new_image():
    # Random initial x-coordinate for the new image within the frame width
    initial_x = np.random.randint(0, frame_width - 100)
    
    # Randomly select if the image will start from the top or bottom of the frame
    if np.random.rand() < 0.5:
        # Start from the top of the frame
        initial_y = 0
        # Random speed for the new image within the range of 10 to 20
        speed = np.random.randint(10, 21)
    else:
        # Start from the bottom of the frame
        initial_y = frame_height - 100
        # Random speed for the new image within the range of -20 to -10 (moving upwards)
        speed = -np.random.randint(10, 21)
    
    return [initial_x, initial_y, speed]

# Function to update the positions of the images
def update_image_positions():
    global images
    
    # Update positions of existing images
    for i in range(len(images)):
        images[i][1] += images[i][2]  # Increment y-coordinate by the speed
        
        # Check if the image is still within the frame
        if images[i][1] + image_rgb.shape[0] > frame_height or images[i][1] < 0:
            # Remove the image if it goes beyond the frame
            images.pop(i)
            break

# Function to display the images on the frame
def display_images(frame):
    for i in range(len(images)):
        new_x, new_y, _ = images[i]
        
        # Overlay the image onto the frame using alpha channel as mask
        for y in range(image_rgb.shape[0]):
            for x in range(image_rgb.shape[1]):
                if alpha_channel[y, x] > 0:  # Check if the pixel is not transparent
                    frame[y + new_y, x + new_x] = image_rgb[y, x]

# Set the dimensions of the camera frame
frame_width = 1600
frame_height = 1000

# Initialize the list to store the images
images = []

# Open the default camera (usually the first camera connected to your system)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load the image with alpha channel
image_with_alpha = cv2.imread(r'./attackroach.png', cv2.IMREAD_UNCHANGED)

# Check if the image was loaded successfully
if image_with_alpha is None:
    print("Error: Could not load image.")
    exit()

# Extract RGB channels from the image and resize it to 100x100
image_rgb = cv2.resize(image_with_alpha[:, :, :3], (100, 100))
alpha_channel = cv2.resize(image_with_alpha[:, :, 3], (100, 100))

# Initialize score counter
score = 0

# Font settings for displaying score
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (0, 255, 0)  # Green color

# Directory containing MP3 sound files
sound_directory = r'./auds'

# List of MP3 sound files
mp3_sounds = [os.path.join(sound_directory, filename) for filename in os.listdir(sound_directory) if filename.endswith('.mp3')]

# Loop to continuously capture frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally to create a mirror effect
    mirrored_frame = cv2.flip(frame, 1)

    # Resize the camera frame to 1000x1000
    resized_frame = cv2.resize(mirrored_frame, (frame_width, frame_height))

    # Generate new images randomly
    if len(images) < 8 and np.random.rand() < 0.1:
        images.append(generate_new_image())

    # Update positions of existing images
    update_image_positions()

    # Display the images on the frame
    display_images(resized_frame)
    
    # Detect hand landmarks
    frame = detector.findHands(resized_frame)
    
    # If hands are detected, check for collision with images and remove if clicked
    if frame:
        for hand in frame:
            # Check if hand has landmarks
            if isinstance(hand, list) and len(hand) > 0:
                # Get finger positions
                finger_positions = hand[0]["lmList"]

                # Iterate over all fingers and check if any is within the bounding box of any image
                for finger_pos in finger_positions:
                    for i in range(len(images)):
                        img_x, img_y, _ = images[i]

                        # Check if finger is within the bounding box of the image
                        if (img_x <= finger_pos[0] <= img_x + image_rgb.shape[1] and
                                img_y <= finger_pos[1] <= img_y + image_rgb.shape[0]):
                            # Remove the image
                            images.pop(i)
                            # Increment score by 5 points
                            score += 5
                            # Play a random MP3 sound
                            sound_file = random.choice(mp3_sounds)
                            pygame.mixer.music.load(sound_file)
                            pygame.mixer.music.play()
                            break  # Break the loop to avoid removing multiple images with the same finger
        
    # Display score counter at top left corner
    cv2.putText(resized_frame, f'Score: {score}', (20, 40), font, font_scale, font_color, font_thickness)


    # Display the mirrored frame with the images overlay
    cv2.imshow('Camera', resized_frame)

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
