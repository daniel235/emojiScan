import scanner as sc
import tensorflow as tf
from network.convolution import faceConvolution
import matplotlib as plt
import numpy as np


s = sc.Scanner()
s.startScan()

f = faceConvolution()
f.prepare_data()



k_size = [5,5]

with tf.Session() as sess:
    stride = (2, 2)
    tf.global_variables_initializer()
