import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq, rnn_decoder, embedding_rnn_decoder, sequence_loss
from tensorflow.contrib.rnn import BasicRNNCell

prediction_file = 'predictions.txt'
y_file = 'y.txt'
summary_file = '/home/logan/tmp'

EPOCHS = 500
PRINT_STEP = 1
NUM_TRAIN = 1500
NUM_TEST = 1500
LATENT_SIZE = 10
SEQUENCE_LEN = 5
VOCAB_SIZE = 1000
EMBEDDING_SIZE = 100
PATIENCE = 10
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
        y[r,c] = x[r, c % x.shape[1]]
# for r in range(len(y)):
#     for c in range(len(y[r])):
#         temp = row_sum[r] / x[r,c%LATENT_SIZE]
#         if c == 0:
#             previous = 0
#         else:
#             previous = y[r,c-1]
#         y[r,c] = temp + previous
    
train_x = x[:NUM_TRAIN]
train_y = y[:NUM_TRAIN]

test_x = x[NUM_TRAIN:]
test_y = y[NUM_TRAIN:]

data = train_x
target = train_y

x_ = tf.placeholder(tf.float32, [None, data.shape[1]], name='x_')
y_ = [tf.placeholder(tf.int64, [None,], name='y_' + str(i)) for i in xrange(target.shape[1])]
weights = [tf.fill(tf.shape(y_[0]), 1.0) for i in xrange(target.shape[1])]
# feed_previous = tf.placeholder(tf.int64, name='feed_previous')
# _feed_previous = (feed_previous == 1)

cell = BasicRNNCell(num_units=data.shape[1], activation=tf.nn.sigmoid)

W = tf.Variable(tf.random_normal([LATENT_SIZE, VOCAB_SIZE]), name='W')     
b = tf.Variable(tf.random_normal([VOCAB_SIZE]), name='b')

with tf.variable_scope("foo"):
    outputs, states = embedding_rnn_decoder(y_,
                              x_,
                              cell,
                              VOCAB_SIZE,
                              EMBEDDING_SIZE,
                              output_projection=(W,b),
                              feed_previous=False,
                              update_embedding_for_previous=True)
    tf.get_variable_scope().reuse_variables()
    outputs_fed_truth, states_fed_truth = embedding_rnn_decoder(y_,
                              x_,
                              cell,
                              VOCAB_SIZE,
                              EMBEDDING_SIZE,
                              output_projection=(W,b),
                              feed_previous=True,
                              update_embedding_for_previous=True)

def calcCost(y_, outputs):

    z = []
    y = []
    for t in range(len(outputs)):
        output_t = outputs[t]
        z.append(tf.matmul(output_t, W) + b)
        y.append(tf.arg_max(z[t], -1))
    cost = sequence_loss(z,
                  y_,
                  weights)

    return cost, y

cost, y = calcCost(y_, outputs)
cost_fed_truth, y_fed_truth = calcCost(y_, outputs_fed_truth)
# test_cost = calcCost(y, test_outputs)
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
    tf.initialize_all_variables().run()
    train_dict = {i: d for i, d in zip(y_, np.transpose(target))}
    train_dict[x_] = data
    test_dict = {i: d for i, d in zip(y_, np.transpose(test_y))}
    test_dict[x_] = test_x
    min_test_cost = np.inf
    num_mistakes = 0
    for i in range(EPOCHS):
        # train for 1 epoch
        sess.run(train_op, feed_dict=train_dict)
        
        # print loss
        if i % PRINT_STEP == 0:
            train_cost = sess.run(cost, feed_dict=train_dict)
            test_cost = sess.run(cost_fed_truth, feed_dict=test_dict)
            print('training cost:', train_cost, 'test cost:', test_cost)
        if test_cost < min_test_cost:
            num_mistakes = 0
            min_test_cost = test_cost
        else:
            num_mistakes += 1
        if num_mistakes >= PATIENCE:
            break
    # test on test set
    response = sess.run(y_fed_truth, feed_dict=test_dict)
    response = np.array(response)
    response = np.transpose(response)
    
    #print to files
    with open(prediction_file, 'w') as f:
        np.savetxt(f, response, fmt='%d\t')
    with open(y_file, 'w') as f:
        np.savetxt(f, test_y, fmt='%d\t')
print('done')
    
    
    
    
    