import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq
from tensorflow.contrib.rnn import BasicRNNCell

prediction_file = 'predictions.txt'
y_file = 'y.txt'

EPOCHS = 10000
PRINT_STEP = 1000
NUM_TRAIN = 1000
NUM_TEST = 1000
SEQUENCE_LEN = 20
VOCAB_SIZE = 1000
np.random.seed(123)

# Each input instance is a sequence of ints. 
train_x = np.random.randint(VOCAB_SIZE, size=(NUM_TRAIN, SEQUENCE_LEN))
# Each target is the sequence in reverse.
train_y = np.fliplr(train_x)

test_x = np.random.randint(VOCAB_SIZE, size=(NUM_TEST, SEQUENCE_LEN))
test_y = np.fliplr(test_x)

data = train_x
target = train_y

x_ = tf.placeholder(tf.float32, [None, data.shape[1]])
y_ = tf.placeholder(tf.float32, [None, target.shape[1]])

cell = BasicRNNCell(num_units=data.shape[1], activation=tf.nn.relu)

# outputs, states = tf.contrib.rnn.static_rnn(cell, [x_], dtype=tf.float32)
y, states = basic_rnn_seq2seq([x_], [y_], cell)
y = y[-1]

cost = tf.reduce_mean(tf.square(y - y_))
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(EPOCHS):
        # train for 1 epoch
        sess.run(train_op, feed_dict={x_:data, y_:target})
        
        # print loss
        if i % PRINT_STEP == 0:
            c = sess.run(cost, feed_dict={x_:data, y_:target})
            print('training cost:', c)

    # test on test set
    response = sess.run(y, feed_dict={x_:test_x, y_:test_y})
    response = np.round(response)
    
    #print to files
    with open(prediction_file, 'w') as f:
        np.savetxt(f, response, fmt='%d\t')
    with open(y_file, 'w') as f:
        np.savetxt(f, test_y, fmt='%d\t')
    
    
    
    
    
    