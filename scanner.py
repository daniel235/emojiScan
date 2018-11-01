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

        cap = cv2.VideoCapture(0)

        while True:
            self.timer -= 1
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('img', frame)

            if(self.timer <= 0):
                cv2.imwrite("./network/data/face.jpg", gray)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.timer = 3
        cap.release()
        cv2.destroyAllWindows()


