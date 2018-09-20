import scanner as sc
import tensorflow as tf
import matplotlib as plt
import numpy as np


s = sc.Scanner()
s.startScan()

k_size = [5,5]

with tf.Session() as sess:
    k_size = 5
    stride = (2, 2)
    tf.global_variables_initializer()

'''x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")
H = x * w + b
cost = tf.reduce_mean(tf.square(H - y))

optimization = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimization.minimize(cost)

x_data = [3.9, 2.1, 4.7, 8.5, 1.9, 6.3, 8.9]
y_data = [9.1, 4.8, 10.7, 18.0, 4.9, 13.2, 18.4]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100):
    _, _cost, _w, _b = sess.run([train, cost, w, b], feed_dict={x: x_data, y: y_data})

print("w ", _w, "b ", _b)
'''