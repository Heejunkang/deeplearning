import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(2.0)

h = X * W
gradient = tf.reduce_mean((W * X - Y) * X) * 2

cost = tf.reduce_mean(tf.square(h - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)


gvs = optimizer.compute_gradients(cost)

# custom

apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

