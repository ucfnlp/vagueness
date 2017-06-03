import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq

EPOCHS = 10000
PRINT_STEP = 1000

data = np.array([[1, 2, 3, 4, 5], [ 2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
target = np.array([[.6,.7,.8,.9,1.0], [.7,.8,.9,1.0,1.0], [.8,.9,1.0,1.0,1.0]])

x_ = tf.placeholder(tf.float32, [None, data.shape[1]])
y_ = tf.placeholder(tf.float32, [None, 5])

cell = tf.contrib.rnn.BasicRNNCell(num_units=data.shape[1])

# outputs, states = tf.contrib.rnn.static_rnn(cell, [x_], dtype=tf.float32)
outputs, states = basic_rnn_seq2seq([x_], [y_], cell)
outputs = outputs[-1]

# W = tf.Variable(tf.random_normal([data.shape[1], 1]))     
# b = tf.Variable(tf.random_normal([1]))
#   
# y = tf.matmul(outputs, W) + b

y = outputs

cost = tf.reduce_mean(tf.square(y - y_))
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(EPOCHS):
        sess.run(train_op, feed_dict={x_:data, y_:target})
#         response = sess.run(y, feed_dict={x_:data})
#         print(response)
        if i % PRINT_STEP == 0:
            c = sess.run(cost, feed_dict={x_:data, y_:target})
            print('training cost:', c)

    response = sess.run(y, feed_dict={x_:data, y_:target})
    print(response)