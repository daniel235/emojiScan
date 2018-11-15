import pygame
import os
import sys
import game.car as c


class track:
    def __init__(self):
        self.id = None
        self.screen = None

    def drawTrack(self, car=None):
        pygame.init()
        self.screen = pygame.display.set_mode((860, 860))
        self.car = c.Car(1)
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
        self.drawCar()
        pygame.display.flip()


    def drawCar(self):
        self.screen.blit(self.car.car, (self.car.x, self.car.y))

    def save_image(self):
        pass




