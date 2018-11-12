import scanner as sc
import tensorflow as tf
from network.convolution import faceConvolution
import matplotlib as plt
import numpy as np
import game.track as t

display = t.track()
display.drawTrack()

while True:
    display.update_screen()
    display.update_input()



