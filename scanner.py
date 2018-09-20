import cv2
import os

import numpy as np
import matplotlib.pyplot as plt

class Scanner:
    def __init__(self):
        self.timer = 100

    #capture face
    def startScan(self):
        im = cv2.imread("./network/data/happy/smile.jpg")
        im = cv2.resize(im, (28,28), interpolation=cv2.INTER_CUBIC)

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


    def buildDatabase(self):
        ########  sections  #############
        expressions = ["happy", "sad", "confused"]


        # face detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        counter = 0
        imCounter = 0

        cap = cv2.VideoCapture(0)
        timer = self.timer
        while True:
            timer -= 1

            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "hello", (10, 500), font, 1, (255, 255, 255))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('img', frame)



            if (timer <= 0):
                imCounter += 1
                cv2.imwrite("./network/data/" + expressions[counter] + "/face" + str(imCounter) + ".jpg", gray)
                if(counter == 2):
                    counter = 0
                else:
                    counter += 1
                timer = self.timer

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
