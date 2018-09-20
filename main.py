import scanner as sc
import tensorflow as tf
import matplotlib as plt
import numpy as np


s = sc.Scanner()
s.startScan()

k_size = [5,5]

with tf.Session() as sess:
    stride = (2, 2)
    tf.global_variables_initializer()
