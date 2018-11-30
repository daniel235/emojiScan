import pygame
import os
import sys

#add file

import car as c
import numpy as np
from skimage import color
from PIL import Image

class track:
    def __init__(self):
        self.id = None
        self.screen = None
        self.endPos = (650, 10)

    def drawTrack(self, car=None):
        pygame.init()
        self.screen = pygame.display.set_mode((880, 880))
        if car == None:
            self.car = c.Car(0)

        self.car.createCar()

    def update_input(self):
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return True

            #input
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                self.car.control(0)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.car.control(2)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.car.control(1)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.car.control(3)

        return False


    def update_screen(self):
        self.screen.fill((0, 100, 30))
        #x y width height
        pygame.draw.rect(self.screen, (255, 255, 255), [400, 300, 130, 500])
        pygame.draw.rect(self.screen, (255, 255, 255), [400, 300, 350, 100])
        pygame.draw.rect(self.screen, (255, 255, 255), [620, 0, 130, 400])
        pygame.draw.circle(self.screen, (100, 0, 0), (650, 10), 20)
        self.drawCar()
        pygame.display.flip()


    def drawCar(self):
        self.screen.blit(self.car.car, (self.car.x, self.car.y))

    def save_image(self):
        self.count = 0
        pygame.image.save(self.screen, "../game/images/track" + str(self.count) + ".jpg")
        pic = Image.open("../game/images/track" + str(self.count) + ".jpg")
        pic = np.array(pic.getdata())
        return pic







