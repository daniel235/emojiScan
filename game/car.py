import pygame

class Car:
    def __init__(self, id):
        self.id = id
        self.car = None
        self.x = 835 / 2
        self.y = 720
        self.destination = None
        self.boundary = None
        self.pos = None

    def createCar(self):
        self.car = pygame.image.load("../game/images/car.png")
        self.pos = (self.x, self.y)

        #resize image
        self.car = pygame.transform.scale(self.car, (80, 130))


    def control(self, direction):
        #up
        print("in control ")
        if direction == 0:
            self.y -= 10

        #right
        elif direction == 1:
            self.x += 10

        #down
        elif direction == 2:
            self.y += 10

        #left
        elif direction == 3:
            self.x -= 10

    #network input
    def getNetInput(self):
        pass
