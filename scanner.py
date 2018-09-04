import cv2

import numpy as np


class Scanner:
    def __init__(self):
        self.timer = 100

    #capture face
    def startScan(self):

        #face detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        cap = cv2.VideoCapture(0)

        while True:
            self.timer -= 1
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)


            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('img', frame)

            if(self.timer <= 0):
                cv2.imwrite("./network/data/face.jpg", gray)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.timer = 3
        cap.release()
        cv2.destroyAllWindows()
