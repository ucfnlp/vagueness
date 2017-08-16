import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq, rnn_decoder, embedding_rnn_decoder, sequence_loss
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import matplotlib.pyplot as plt
import os
from metrics import performance
import time
import h5py
from docutils.nodes import generated
import generator
from discriminator import discriminator

prediction_file = 'predictions.txt'
y_file = 'y.txt'
prediction_words_file = 'predictions_words.txt'
summary_file = '/home/logan/tmp'
model_file = '../models/gan_model'
dataset_file = '../data/annotated_dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
dictionary_file = '../data/words.dict'

# np.random.seed(123)

start_time = time.time()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('EPOCHS', 5000,
                            'Num epochs.')
tf.app.flags.DEFINE_integer('LATENT_SIZE', 200,
                            'Size of both the hidden state of RNN and random vector z.')
tf.app.flags.DEFINE_integer('SEQUENCE_LEN', 50,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('EMBEDDING_SIZE', 300,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('PATIENCE', 200,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('BATCH_SIZE', 128,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_string('CELL_TYPE', 'GRU',
                            'Which RNN cell for the RNNs.')


'''
--------------------------------

LOAD DATA

--------------------------------
'''

print('loading embedding weights')
with h5py.File(embedding_weights_file, 'r') as hf:
    embedding_weights = hf['embedding_weights'][:]
    
print('loading training and test data')
with h5py.File(dataset_file, 'r') as data_file:
    train_x = data_file['train_X'][:]
    
# train_x = train_x[:1]
    

print('loading dictionary')
d = {}
vocab_size = 0
with open(dictionary_file) as f:
    for line in f:
       (val, key) = line.split()
       d[int(key)] = val
       vocab_size += 1
       
tf.app.flags.DEFINE_integer('VOCAB_SIZE', vocab_size,
                            'Number of words in the vocabulary.')

train_x_one_hot = np.transpose(train_x)
train_x_one_hot = np.eye(FLAGS.VOCAB_SIZE)[train_x_one_hot.astype(int)]

print train_x
for i in range(min(5, len(train_x))):
    for j in range(len(train_x[i])):
        if train_x[i][j] == 0:
            continue
        word = d[train_x[i][j]]
        print word + ' ',
    print '\n'

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def batch_generator(x_one_hot):
    data_len = x_one_hot.shape[1]
    for i in range(0, data_len, FLAGS.BATCH_SIZE):
        yield x_one_hot[:,i:min(i+FLAGS.BATCH_SIZE,data_len)], i, data_len
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
X = [tf.placeholder(tf.float32, shape=[None, FLAGS.VOCAB_SIZE]) for i in xrange(FLAGS.SEQUENCE_LEN)]
z = tf.placeholder(tf.float32, [None, FLAGS.LATENT_SIZE], name='z')
dims = tf.stack([tf.shape(X[0])[0],])
zero_inputs = [tf.fill(dims, 0) for i in xrange(FLAGS.SEQUENCE_LEN)]

def sample_Z(m, n):
    return np.random.normal(size=[m, n])
#     return np.random.uniform(low=-1, high=1, size=[m, n])

    
with tf.variable_scope(tf.get_variable_scope()) as scope:
    G_sample = generator.generator(z, zero_inputs)
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

def save_samples_to_file(generated_sequences):
    with open(prediction_file, 'w') as f:
        np.savetxt(f, generated_sequences, fmt='%d\t')
    with open(prediction_words_file, 'w') as f:
        for i in range(len(generated_sequences)):
            for j in range(len(generated_sequences[i])):
                if generated_sequences[i][j] == 0:
                    f.write('<UNK> ')
                else:
                    word = d[generated_sequences[i][j]]
                    f.write(word + ' ')
            f.write('\n')

if not os.path.exists('out/'):
    os.makedirs('out/')

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
    tf.global_variables_initializer().run()
    min_test_cost = np.inf
    num_mistakes = 0
    
    xaxis = 0
    step = 0
    for i in range(FLAGS.EPOCHS):
        for batch, cur, data_len in batch_generator(train_x_one_hot):
            if batch == None:
                break
            train_dict = {i: d for i, d in zip(X, batch)}
            train_dict[z] = sample_Z(batch.shape[1], FLAGS.LATENT_SIZE)
        
    
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
                for i in range(min(3, len(generated_sequences))):
                    for j in range(len(generated_sequences[i])):
                        if generated_sequences[i][j] == 0:
                            print '<UNK> ',
                        else:
                            word = d[generated_sequences[i][j]]
                            print word + ' ',
                    print '\n'
                
            if i % 100 == 0:
                save_samples_to_file(generated_sequences)
        
    train_writer.close()
        
    save_samples_to_file(generated_sequences)
        
print('Execution time: ', time.time() - start_time)
    
    
    
    
    