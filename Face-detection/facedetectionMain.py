# pip install opencv-python
#haarcascade_frontalface_default.xml

import cv2
cascade_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#main part of code as it loads face


cap=cv2.VideoCapture(0)#give access to camera or web cam capture
if not cap.isOpened:
   print("error")
   exit()
while True:
   ret,frame=cap.read() #detects the face from b i.e., videocapture and stores in 2 variables c,d for recording and image and starts reading the face or capture frame-by-frame
   #operations on the frame come here
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
   for (x,y,w,h) in detections:
      
      #covering of x_axis and y_axis and height anfd width

 

      #caputuring or saving of the image of particular one
     
      frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0,),2)
      name='Vaishnavi'
      font=cv2.FONT_HERSHEY_SIMPLEX
      font_scale=0.7
      font_color=(0,255,0)
      thickness=2
      text_size=cv2.getTextSize(name,font,font_scale,thickness)[0]
      text_x=x+(w-text_size[0])//2
      text_y=y-10
      cv2.putText(frame,name,(text_x,text_y),font,font_scale,font_color,thickness)

     #display the rsulting frame
   cv2.imshow('frame',frame)#img is ifor caputuring image d_image is the image of detection


   k = cv2.waitKey(1)  # Wait for 1 ms
   if k == ord('q'):  # Press 'q' to quit
        break

     # h=cv2.waitKey(40) & 0xff#waiting for particular amnt of time or specified amnt of time inthat particular of time if it doesn't recognize or face then exit out from entire code


    

cap.release() #exit out of camera
cv2.destroyAllWindows()# exit out of all windows













