import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq, rnn_decoder
from tensorflow.contrib.rnn import BasicRNNCell

prediction_file = 'predictions.txt'
y_file = 'y.txt'

EPOCHS = 500000
PRINT_STEP = 1000
NUM_TRAIN = 1000
NUM_TEST = 1000
LATENT_SIZE = 10
SEQUENCE_LEN = 20
VOCAB_SIZE = 1000
PATIENCE = 2000
np.random.seed(123)

# Each input is a latent vector of size LATENT_SIZE
x = np.random.randint(VOCAB_SIZE, size=(NUM_TRAIN+NUM_TEST, LATENT_SIZE))
# Each target is a sequence based on the latent vector
y = np.zeros((NUM_TRAIN+NUM_TEST, SEQUENCE_LEN))
# sum of each row
row_sum = np.sum(x, axis=1)
# To calculate the y at time t, take the sum of the latent vector,
# then divide by x[t], then add y[t-1]
for r in range(len(y)):
    for c in range(len(y[r])):
        temp = row_sum[r] / x[r,c%LATENT_SIZE]
        if c == 0:
            previous = 0
        else:
            previous = y[r,c-1]
        y[r,c] = temp + previous
    
train_x = x[:NUM_TRAIN]
train_y = y[:NUM_TRAIN]

test_x = x[NUM_TRAIN:]
test_y = y[NUM_TRAIN:]

data = train_x
target = train_y

x_ = tf.placeholder(tf.float32, [None, data.shape[1]])
y_ = [tf.placeholder(tf.float32, [None, 1]) for _ in xrange(target.shape[1])]

cell = BasicRNNCell(num_units=data.shape[1])

# outputs, states = tf.contrib.rnn.static_rnn(cell, [x_], dtype=tf.float32)
outputs, states = rnn_decoder(y_, x_, cell)
W = tf.Variable(tf.random_normal([int(outputs[0].shape[1]), 1]))     
b = tf.Variable(tf.random_normal([1]))

y = []
for output_t in outputs:
    y.append(tf.matmul(output_t, W) + b)
    
cost = 0
for i in range(len(y)):
    cost += tf.reduce_mean(tf.square(y[i] - y_[i]))
cost /= len(y)

train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    train_dict = {i: d for i, d in zip(y_, np.transpose(target).reshape(SEQUENCE_LEN,-1,1))}
    train_dict[x_] = data
    test_dict = {i: d for i, d in zip(y_, np.transpose(test_y).reshape(SEQUENCE_LEN,-1,1))}
    test_dict[x_] = test_x
    min_test_cost = np.inf
    num_mistakes = 0
    for i in range(EPOCHS):
        # train for 1 epoch
        sess.run(train_op, feed_dict=train_dict)
        
        
#         response = sess.run(y, feed_dict=train_dict)
#         response = np.round(response)
#         response = np.transpose(response)
#         print(response)
        
        # print loss
        if i % PRINT_STEP == 0:
            train_cost = sess.run(cost, feed_dict=train_dict)
            test_cost = sess.run(cost, feed_dict=test_dict)
            print('training cost:', train_cost, 'test cost:', test_cost)
        if test_cost < min_test_cost:
            num_mistakes = 0
            min_test_cost = test_cost
        else:
            num_mistakes += 1
        if num_mistakes >= PATIENCE:
            break
    # test on test set
    response = sess.run(y, feed_dict=test_dict)
    response = np.round(response)
    response = response.reshape((SEQUENCE_LEN,-1))
    response = np.transpose(response)
    
    #print to files
    with open(prediction_file, 'w') as f:
        np.savetxt(f, response, fmt='%d\t')
    with open(y_file, 'w') as f:
        np.savetxt(f, test_y, fmt='%d\t')
print('done')
    
    
    
    
    