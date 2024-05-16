import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import tkinter as tk
from tkinter import messagebox

pygame.init()

# Initialize the first camera (change the index if needed)
cap1 = cv2.VideoCapture(0)

# Initialize the second camera (change the index if needed)
cap2 = cv2.VideoCapture(1)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status marking for different states
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Function to compute distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to detect blinking
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Blink check
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

# Function to play alert sound using pygame
def play_alert_sound():
    pygame.mixer.music.load("Voicy_Airhorn.mp3")  # Change the file path to your alert sound file
    pygame.mixer.music.play()

# Function to display dialog box using tkinter
def display_dialog_box():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("SleepGuard", "You are sleeping!")

#Loop to capture and process frames from both cameras
while True:
    # Capture frame from the first camera
    ret1, frame1 = cap1.read()
    if not ret1:
        print("Error: Failed to capture frame from camera 1")
        break

    # Capture frame from the second camera
    ret2, frame2 = cap2.read()
    if not ret2:
        print("Error: Failed to capture frame from camera 2")
        break

    # Convert frames to grayscale for face detection
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Process frame1 (Camera 1) for drowsiness detection
    faces1 = detector(gray1)
    for face in faces1:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(gray1, face)
        landmarks = face_utils.shape_to_np(landmarks)
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Update status based on blink detection
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "YOU ARE SLEEPING!!!"
                color = (255, 0, 0)
                play_alert_sound()  # Play alert sound
                display_dialog_box()  # Display dialog box
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "YOU ARE DROWSY WAKE UP!"
                color = (0, 0, 255)
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "YOU ARE ACTIVE :D"
                color = (0, 255, 0)
            
        cv2.putText(frame1, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Display the first camera frame (Camera 1) with status overlay
    cv2.imshow("Camera 1 - SleepGuard v1.0", frame1)

    # Process frame2 (Camera 2) for additional features (e.g., face detection, landmarks, info overlay)
    faces2 = detector(gray2)
    for face in faces2:
        # Draw rectangle around detected face
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Predict facial landmarks
        landmarks = predictor(gray2, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Draw facial landmarks on the frame2 (Camera 2)
        for (x, y) in landmarks:
            cv2.circle(frame2, (x, y), 1, (0, 255, 0), -1)  # Draw green circles at each landmark point

    # Display the second camera frame (Camera 2) with additional features
    cv2.imshow("Camera 2 - Face Detection & Landmarks", frame2)

    # Check for key press to exit
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Release both camera instances and close all OpenCV windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
