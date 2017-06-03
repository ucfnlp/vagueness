#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from seq2seq import basic_rnn_seq2seq
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  cell = tf.contrib.rnn.BasicRNNCell(2)
#   y, _ = basic_rnn_seq2seq(features['x'], labels, cell)
  y, states = tf.contrib.rnn.static_rnn(cell, [features['x']], dtype=tf.float32)
  y = y[-1]
  
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x = np.array([[1.,1.], [2.,2.], [3.,3.], [4.,4.]])
y = np.array([[0.,0.], [-1.,-1.], [-2.,-2.], [-3.,-3.]])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))