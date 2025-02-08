from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import  os
import time
from datetime import datetime
from win32com.client import Dispatch
import csv

def speak(str1):
   speak=Dispatch(("SAPI.SpVoice"))
   speak.Speak(str1)

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
with open('data/names.pkl','rb') as f:
    LABELS=pickle.load(f)
with open('data/faces.pkl','rb') as f:
    FACES=pickle.load(f)
if len(FACES) != len(LABELS):
    print(f"Error: FACES and LABELS have mismatched lengths. FACES: {len(FACES)}, LABELS: {len(LABELS)}")
    min_len = min(len(FACES), len(LABELS))
    FACES = FACES[:min_len]
    LABELS = LABELS[:min_len]
    print(f"Data truncated to minimum length: {min_len}")

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES=['NAME','TIME',]


while True:
  ret,frame=video.read()
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces=facedetect.detectMultiScale(gray, 1.3 ,5)
  for (x,y,w,h) in faces:
    crop_image=frame[y:y+h, x:x+w, :]
    resized_image=cv2.resize(crop_image, (25,15)).flatten().reshape(1,-1)
    output=knn.predict(resized_image)
    ts=time.time()
    date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
    exist=os.path.isfile("attendance/attendance_"+date +".csv")
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 2)

    cv2.rectangle(frame, (x,y-48), (x+w,y), (50, 50, 255),-1)
    cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
    cv2.rectangle(frame,(x,y), (x+w, y+h), (50,50,255),1)
    attendance=[str(output[0]),str(timestamp)]
  k=cv2.waitKey(1)
  if k==ord('o'):
    speak("Attendance taken...")
    time.sleep(5)
    if exist:
        with open ("attendance/attendance_"+date +".csv","+a") as csvfile:
          writer=csv.writer(csvfile)
          writer.writerow(COL_NAMES)
          writer.writerow(attendance)
        csvfile.close()
    else:
        with open ("attendance/attendance_"+date +".csv","+a") as csvfile:
          writer=csv.writer(csvfile)
          writer.writerow(COL_NAMES)
          writer.writerow(attendance)
        csvfile.close()
 
video.release()
cv2.destroyAllWindows()


