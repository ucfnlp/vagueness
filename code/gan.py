import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq, rnn_decoder, embedding_rnn_decoder, sequence_loss
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from metrics import performance
import time

prediction_file = 'predictions.txt'
y_file = 'y.txt'
summary_file = '/home/logan/tmp'

EPOCHS = 2000
PRINT_STEP = 10
NUM_TRAIN = 10000
NUM_TEST = 1000
LATENT_SIZE = 200
SEQUENCE_LEN = 20
VOCAB_SIZE = 100
EMBEDDING_SIZE = 50
PATIENCE = 200
BATCH_SIZE = 128
cell_type = 1
np.random.seed(123)

start_time = time.time()

if cell_type == 2:
    len_x = LATENT_SIZE/2
else:
    len_x = LATENT_SIZE
# Each input is a latent vector of size LATENT_SIZE
x = np.random.randint(1, VOCAB_SIZE, size=(NUM_TRAIN+NUM_TEST, len_x))
for i in range(len(x)):
    for j in range(1,len(x[i])):
        x[i][j] = (x[i][j-1] + (VOCAB_SIZE/10+1)) % VOCAB_SIZE
# Each target is a sequence based on the latent vector
y = np.zeros((NUM_TRAIN+NUM_TEST, SEQUENCE_LEN))
for r in range(len(y)):
    for c in range(len(y[r])):
        y[r,c] = x[r, c % x.shape[1]]
y_one_hot = np.transpose(y)
y_one_hot = np.eye(VOCAB_SIZE)[y_one_hot.astype(int)]

y_shifted = np.roll(y, 1)
y_shifted[:,0] = 0
    
train_x = x[:NUM_TRAIN]
train_y = y[:NUM_TRAIN]
train_y_shifted = y_shifted[:NUM_TRAIN]
train_y_one_hot = y_one_hot[:NUM_TRAIN]

test_x = x[NUM_TRAIN:]
test_y = y[NUM_TRAIN:]
test_y_shifted = y_shifted[NUM_TRAIN:]
test_y_one_hot = y_one_hot[NUM_TRAIN:]

def batch_generator(x,y,y_shifted):
    for i in range(len(x)):
        yield x[i:min(i+BATCH_SIZE,len(x))], y[i:min(i+BATCH_SIZE,len(x))], y_shifted[i:min(i+BATCH_SIZE,len(x))]
    yield None, None, None

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

X = [tf.placeholder(tf.float32, shape=[None, VOCAB_SIZE]) for i in xrange(SEQUENCE_LEN)]
z = tf.placeholder(tf.float32, [None, data.shape[1]], name='z')
c_init = tf.fill(tf.shape(z), 0.0)
y_ = [tf.placeholder(tf.int64, [None,], name='y_' + str(i)) for i in xrange(target.shape[1])]
# inputs = [tf.placeholder(tf.int64, [None,], name='input_' + str(i)) for i in xrange(target.shape[1])]
inputs = [tf.fill(tf.shape(y_[0]), 0) for i in xrange(SEQUENCE_LEN)]
weights = [tf.fill(tf.shape(y_[0]), 1.0) for i in xrange(target.shape[1])]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    with tf.variable_scope("G_"):
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
        
        outputs, states = embedding_rnn_decoder(inputs,
                                  z,
                                  cell,
                                  VOCAB_SIZE,
                                  EMBEDDING_SIZE,
                                  output_projection=(W,b),
                                  feed_previous=True,
                                  update_embedding_for_previous=True)
        logits = [tf.matmul(output, W) + b for output in outputs]
        x = [tf.nn.softmax(logit) for logit in logits] # is this softmaxing over the right dimension? this turns into 3D
        return x


def discriminator(x):
    with tf.variable_scope("D_"):
        embeddings = tf.get_variable('embeddings_W', [VOCAB_SIZE, EMBEDDING_SIZE], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer())
#         embeddings = tf.Variable(
#             tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0), name='embeddings_W')
        b1 = tf.get_variable('embeddings_b', [EMBEDDING_SIZE], dtype=tf.float32, 
                             initializer=tf.zeros_initializer())
        
        embed = [tf.nn.tanh(tf.matmul(_x, embeddings) + b1) for _x in x]
        
#         initial_state = cell.zero_state(batch_size, data_type())
        if cell_type == 2:
            cell = BasicLSTMCell(num_units=data.shape[1], activation=tf.nn.tanh, state_is_tuple=False)
        elif cell_type == 0:
            cell = BasicRNNCell(num_units=data.shape[1], activation=tf.nn.tanh)
        elif cell_type == 1:
            cell = GRUCell(num_units=data.shape[1], activation=tf.nn.tanh)
        outputs, state = tf.contrib.rnn.static_rnn(
            cell, embed, dtype=tf.float32)
        
        logit = tf.layers.dense(outputs[-1], 1)
#         W2 = tf.get_variable('W', [LATENT_SIZE, 1], initializer=tf.random_normal_initializer()) 
#         b2 = tf.get_variable('b', [1], initializer=tf.zeros_initializer())
#         logit = tf.matmul(outputs[-1], W2) + b2
        prob = tf.nn.sigmoid(logit)
    
        return prob, logit

    
# train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)
    
with tf.variable_scope(tf.get_variable_scope()) as scope:
    G_sample = generator(z)
    D_real, D_logit_real = discriminator(X)
    tf.get_variable_scope().reuse_variables()
    D_fake, D_logit_fake = discriminator(G_sample)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

theta_D = [var for var in tf.trainable_variables() if 'D_' in var.name]
theta_G = [var for var in tf.trainable_variables() if 'G_' in var.name]
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
for var in theta_D:
    print var
for var in theta_G:
    print var
    
    
merged = tf.summary.merge_all()

if not os.path.exists('out/'):
    os.makedirs('out/')

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
    tf.global_variables_initializer().run()
    train_dict = {i: d for i, d in zip(X, train_y_one_hot)}
    temp = {i: d for i, d in zip(y_, np.transpose(train_y))}
    train_dict.update(temp)
    train_dict[z] = sample_Z(NUM_TRAIN, LATENT_SIZE)
    test_dict = {i: d for i, d in zip(X, test_y_one_hot)}
    temp = {i: d for i, d in zip(y_, np.transpose(test_y))}
    test_dict.update(temp)
    test_dict[z] = sample_Z(NUM_TRAIN, LATENT_SIZE)
    min_test_cost = np.inf
    num_mistakes = 0
    
    xaxis = 0

    for i in range(EPOCHS):
        # train for 1 epoch
        
        
    
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict=train_dict)
        for j in range(3):
            sample, _, G_loss_curr = sess.run([G_sample, G_solver, G_loss], feed_dict=train_dict)
    
        if i % 1 == 0:
            generated_sequences = np.transpose(np.argmax(sample, axis=-1))
            print('Iter: {}'.format(i))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print('Samples', generated_sequences)
            print()
        
    train_writer.close()
        
    with open(prediction_file, 'w') as f:
        np.savetxt(f, generated_sequences, fmt='%d\t')
        
#     # test on train set
#     response = sess.run(y_fed_previous, feed_dict=train_dict)
#     response = np.array(response)
#     response = np.transpose(response)
#     acc,_,_,_ = performance(response, train_y)
#     print('Train accuracy: ', acc)
#     
#     # test on test set
#     response = sess.run(y_fed_previous, feed_dict=test_dict)
#     response = np.array(response)
#     response = np.transpose(response)
#     acc,_,_,_ = performance(response, test_y)
#     print('Test accuracy: ', acc)
#     #print to files
#     with open(prediction_file, 'w') as f:
#         np.savetxt(f, response, fmt='%d\t')
#     with open(y_file, 'w') as f:
#         np.savetxt(f, test_y, fmt='%d\t')
print('done')
print('Execution time: ', time.time() - start_time)
    
    
    
    
    