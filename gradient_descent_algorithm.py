import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X = [1, 2, 3]
Y = [3, 4, 5]

W = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
h = X * W + b

cost = tf.reduce_mean(tf.square(h - Y))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []
b_val = []
for i in range(-30, 50):
    for j in range(0, 10):
        curr_cost, curr_W, curr_b = sess.run([cost, W, b], feed_dict={W: i * 0.1, b: j})
        W_val.append(curr_W)
        b_val.append(curr_b)
        cost_val.append(curr_cost)
        print(curr_W, curr_b, curr_cost)

plt.plot(W_val, cost_val)
plt.show()