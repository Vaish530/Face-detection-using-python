from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime
from win32com.client import Dispatch
import csv

def speak(text):
    speaker = Dispatch(("SAPI.SpVoice"))
    speaker.Speak(text)

# Create attendance directory if it doesn't exist
attendance_dir = "attendance"
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

# Initialize webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load face detection model
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load labels and faces
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Ensure FACES and LABELS have the same length
if len(FACES) != len(LABELS):
    print(f"Error: FACES and LABELS have mismatched lengths. FACES: {len(FACES)}, LABELS: {len(LABELS)}")
    min_len = min(len(FACES), len(LABELS))
    FACES = FACES[:min_len]
    LABELS = LABELS[:min_len]
    print(f"Data truncated to minimum length: {min_len}")

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w, :]
        resized_image = cv2.resize(crop_image, (25, 15)).flatten().reshape(1, -1)
        output = knn.predict(resized_image)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist = os.path.isfile(f"{attendance_dir}/attendance_{date}.csv")

        # Draw rectangles and text on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-48), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        attendance = [str(output[0]), str(timestamp)]

    cv2.imshow("Webcam", frame)

    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance taken...")
        time.sleep(5)
        
        with open(f"{attendance_dir}/attendance_{date}.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)  # Write header only if file is new
            writer.writerow(attendance)
    
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()