import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq, rnn_decoder, embedding_rnn_decoder, sequence_loss
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
from metrics import performance

prediction_file = 'predictions.txt'
y_file = 'y.txt'
summary_file = '/home/logan/tmp'

EPOCHS = 5000
PRINT_STEP = 1
NUM_TRAIN = 10000
NUM_TEST = 1000
LATENT_SIZE = 5
SEQUENCE_LEN = 5
VOCAB_SIZE = 10
EMBEDDING_SIZE = 50
PATIENCE = 20
cell_type = 1
np.random.seed(123)

if cell_type == 2:
    len_x = LATENT_SIZE/2
else:
    len_x = LATENT_SIZE
# Each input is a latent vector of size LATENT_SIZE
x = np.random.randint(1, VOCAB_SIZE, size=(NUM_TRAIN+NUM_TEST, len_x))
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

y_shifted = np.roll(y, 1)
y_shifted[:,0] = 0
    
train_x = x[:NUM_TRAIN]
train_y = y[:NUM_TRAIN]
train_y_shifted = y_shifted[:NUM_TRAIN]

test_x = x[NUM_TRAIN:]
test_y = y[NUM_TRAIN:]
test_y_shifted = y_shifted[NUM_TRAIN:]

data = train_x
target = train_y

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

x_ = tf.placeholder(tf.float32, [None, data.shape[1]], name='x_')
c_init = tf.fill(tf.shape(x_), 0.0)
y_ = [tf.placeholder(tf.int64, [None,], name='y_' + str(i)) for i in xrange(target.shape[1])]
inputs = [tf.placeholder(tf.int64, [None,], name='input_' + str(i)) for i in xrange(target.shape[1])]
weights = [tf.fill(tf.shape(y_[0]), 1.0) for i in xrange(target.shape[1])]

if cell_type == 2:
    cell = BasicLSTMCell(num_units=data.shape[1], activation=tf.nn.tanh, state_is_tuple=False)
elif cell_type == 0:
    cell = BasicRNNCell(num_units=data.shape[1], activation=tf.nn.tanh)
elif cell_type == 1:
    cell = GRUCell(num_units=data.shape[1], activation=tf.nn.tanh)

W = tf.Variable(tf.random_normal([LATENT_SIZE, VOCAB_SIZE]), name='W')    
b = tf.Variable(tf.random_normal([VOCAB_SIZE]), name='b')    
variable_summaries(W) 
variable_summaries(b)

with tf.variable_scope("decoder"):
    outputs, states = embedding_rnn_decoder(inputs,
                              x_,
                              cell,
                              VOCAB_SIZE,
                              EMBEDDING_SIZE,
                              output_projection=(W,b),
                              feed_previous=False,
                              update_embedding_for_previous=True)
    tf.get_variable_scope().reuse_variables()
    outputs_fed_previous, states_fed_previous = embedding_rnn_decoder(inputs,
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
cost_fed_previous, y_fed_previous = calcCost(y_, outputs_fed_previous)
# test_cost = calcCost(y, test_outputs)
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
    tf.global_variables_initializer().run()
    train_dict = {i: d for i, d in zip(y_, np.transpose(target))}
    temp = {i: d for i, d in zip(inputs, np.transpose(train_y_shifted))}
    train_dict.update(temp)
    train_dict[x_] = data
    test_dict = {i: d for i, d in zip(y_, np.transpose(test_y))}
    temp = {i: d for i, d in zip(inputs, np.transpose(test_y_shifted))}
    test_dict.update(temp)
    test_dict[x_] = test_x
    min_test_cost = np.inf
    num_mistakes = 0
    for i in range(EPOCHS):
        # train for 1 epoch
        sess.run(train_op, feed_dict=train_dict)
        
        # print loss
        if i % PRINT_STEP == 0:
            train_cost, summary = sess.run([cost, merged], feed_dict=train_dict)
            train_writer.add_summary(summary, i)
            test_cost = sess.run(cost_fed_previous, feed_dict=test_dict)
            print('training cost:', train_cost, 'test cost:', test_cost)
        if test_cost < min_test_cost:
            num_mistakes = 0
            min_test_cost = test_cost
        else:
            num_mistakes += 1
        if num_mistakes >= PATIENCE:
            break
        
    train_writer.close()
        
    # test on train set
    response = sess.run(y_fed_previous, feed_dict=train_dict)
    response = np.array(response)
    response = np.transpose(response)
    acc,_,_,_ = performance(response, train_y)
    print('Train accuracy: ', acc)
    
    # test on test set
    response = sess.run(y_fed_previous, feed_dict=test_dict)
    response = np.array(response)
    response = np.transpose(response)
    acc,_,_,_ = performance(response, test_y)
    print('Test accuracy: ', acc)
    #print to files
    with open(prediction_file, 'w') as f:
        np.savetxt(f, response, fmt='%d\t')
    with open(y_file, 'w') as f:
        np.savetxt(f, test_y, fmt='%d\t')
print('done')
    
    
    
    
    