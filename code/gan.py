import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq, rnn_decoder, embedding_rnn_decoder, sequence_loss
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import matplotlib.pyplot as plt
import os
from metrics import performance
import time

prediction_file = 'predictions.txt'
y_file = 'y.txt'
summary_file = '/home/logan/tmp'

EPOCHS = 5000
PRINT_STEP = 10
NUM_TRAIN = 10000
NUM_TEST = 1000
LATENT_SIZE = 200
SEQUENCE_LEN = 4
VOCAB_SIZE = 10
EMBEDDING_SIZE = 50
PATIENCE = 200
BATCH_SIZE = 128
cell_type = 1
# np.random.seed(123)

start_time = time.time()


'''
--------------------------------

CREATE DATA

--------------------------------
Each input is an incrementing sequence.

Examples:
1 2 3 4
5 6 7 8
3 4 5 6
8 9 0 1
'''
x = np.zeros((NUM_TRAIN+NUM_TEST, SEQUENCE_LEN))
for i in range(len(x)):
    x[i][0] = np.random.randint(0, VOCAB_SIZE)
    for j in range(1,len(x[i])):
        x[i][j] = (x[i][j-1] + 1) % VOCAB_SIZE
x_one_hot = np.transpose(x)
x_one_hot = np.eye(VOCAB_SIZE)[x_one_hot.astype(int)]
    
train_x = x[:NUM_TRAIN]
train_x_one_hot = x_one_hot[:,:NUM_TRAIN]

test_x = x[NUM_TRAIN:]
test_x_one_hot = x_one_hot[:,NUM_TRAIN:]

print train_x

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def batch_generator(x_one_hot):
    data_len = x_one_hot.shape[1]
    for i in range(0, data_len, BATCH_SIZE):
        yield x_one_hot[:,i:min(i+BATCH_SIZE,data_len)], i, data_len
    yield None, None, None

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean ' + var.name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev ' + var.name, stddev)
    tf.summary.scalar('max ' + var.name, tf.reduce_max(var))
    tf.summary.scalar('min ' + var.name, tf.reduce_min(var))
    tf.summary.histogram('histogram ' + var.name, var)


'''
--------------------------------

MODEL

--------------------------------
'''
X = [tf.placeholder(tf.float32, shape=[None, VOCAB_SIZE]) for i in xrange(SEQUENCE_LEN)]
z = tf.placeholder(tf.float32, [None, LATENT_SIZE], name='z')
dims = tf.stack([tf.shape(X[0])[0],])
zero_inputs = [tf.fill(dims, 0) for i in xrange(SEQUENCE_LEN)]


def sample_Z(m, n):
    return np.random.normal(size=[m, n])
#     return np.random.uniform(low=-1, high=1, size=[m, n])


def generator(z):
    with tf.variable_scope("G_"):
        if cell_type == 2:
            cell = BasicLSTMCell(num_units=LATENT_SIZE, activation=tf.nn.tanh, state_is_tuple=False)
        elif cell_type == 0:
            cell = BasicRNNCell(num_units=LATENT_SIZE, activation=tf.nn.tanh)
        elif cell_type == 1:
            cell = GRUCell(num_units=LATENT_SIZE, activation=tf.nn.tanh)
        
        W = tf.Variable(tf.random_normal([LATENT_SIZE, VOCAB_SIZE]), name='W')    
        b = tf.Variable(tf.random_normal([VOCAB_SIZE]), name='b')    
        
        outputs, states = embedding_rnn_decoder(zero_inputs,   # is this ok? I'm not sure what giving 0 inputs does (although it should be completely ignoring inputs)
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
        b1 = tf.get_variable('embeddings_b', [EMBEDDING_SIZE], dtype=tf.float32, 
                             initializer=tf.zeros_initializer())
        
        embed = [tf.nn.tanh(tf.matmul(_x, embeddings) + b1) for _x in x]
        if cell_type == 2:
            cell = BasicLSTMCell(num_units=LATENT_SIZE, activation=tf.nn.tanh, state_is_tuple=False)
        elif cell_type == 0:
            cell = BasicRNNCell(num_units=LATENT_SIZE, activation=tf.nn.tanh)
        elif cell_type == 1:
            cell = GRUCell(num_units=LATENT_SIZE, activation=tf.nn.tanh)
        outputs, state = tf.contrib.rnn.static_rnn(
            cell, embed, dtype=tf.float32)
        
        logit = tf.layers.dense(outputs[-1], 1)
        prob = tf.nn.sigmoid(logit)
    
        return prob, logit

    
with tf.variable_scope(tf.get_variable_scope()) as scope:
    G_sample = generator(z)
    D_real, D_logit_real = discriminator(X)
    tf.get_variable_scope().reuse_variables()
    D_fake, D_logit_fake = discriminator(G_sample)
    D_real_acc = tf.cast(tf_count(tf.round(D_real), 1), tf.float32) / tf.cast(tf.shape(D_real)[0], tf.float32)
    D_fake_acc = tf.cast(tf_count(tf.round(D_fake), 0), tf.float32) / tf.cast(tf.shape(D_fake)[0], tf.float32)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

theta_D = [var for var in tf.trainable_variables() if 'D_' in var.name]
theta_G = [var for var in tf.trainable_variables() if 'G_' in var.name]
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(-G_loss, var_list=theta_G)
for var in theta_D:
    variable_summaries(var) 
    print var
for var in theta_G:
    variable_summaries(var) 
    print var
    
merged = tf.summary.merge_all()


'''
--------------------------------

MAIN

--------------------------------
'''
if not os.path.exists('out/'):
    os.makedirs('out/')

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
    tf.global_variables_initializer().run()
    min_test_cost = np.inf
    num_mistakes = 0
    
    xaxis = 0
    step = 0
    for i in range(EPOCHS):
        for batch, cur, data_len in batch_generator(train_x_one_hot):
            if batch == None:
                break
            train_dict = {i: d for i, d in zip(X, batch)}
            train_dict[z] = sample_Z(batch.shape[1], LATENT_SIZE)
        
    
            _, D_loss_curr, real_acc, fake_acc, summary = sess.run([D_solver, D_loss, D_real_acc, D_fake_acc, merged], feed_dict=train_dict)
            for j in range(1):
                sample, _, G_loss_curr = sess.run([G_sample, G_solver, G_loss], feed_dict=train_dict)
    
            if i % 1 == 0:
                train_writer.add_summary(summary, step)
                step += 1
                generated_sequences = np.transpose(np.argmax(sample, axis=-1))
                print('Iter: {}'.format(i))
                print('Instance ', cur, ' out of ', data_len)
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print('D real acc: ', real_acc, ' D fake acc: ', fake_acc)
                print('Samples', generated_sequences)
                print()
                
            if i % 100 == 0:
                with open(prediction_file, 'w') as f:
                    np.savetxt(f, generated_sequences, fmt='%d\t')
        
    train_writer.close()
        
    with open(prediction_file, 'w') as f:
        np.savetxt(f, generated_sequences, fmt='%d\t')
        
print('done')
print('Execution time: ', time.time() - start_time)
    
    
    
    
    