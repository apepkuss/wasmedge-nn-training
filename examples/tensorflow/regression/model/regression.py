import os

# import tensorflow as tf

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

x = tf.placeholder(tf.float32, name="x")
y = tf.placeholder(tf.float32, name="y")

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="w")
b = tf.Variable(tf.zeros([1]), name="b")
y_hat = w * x + b

loss = tf.reduce_mean(tf.square(y_hat - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss, name="train")

init = tf.variables_initializer(tf.global_variables(), name="init")

definition = tf.Session().graph_def
directory = "."
tf.train.write_graph(definition, directory, "model.pb", as_text=False)
