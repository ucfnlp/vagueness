#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
from seq2seq import embedding_rnn_decoder
import utils
import param_names
import load
import argparse
from tqdm import tqdm

train_variables_file = '../models/lm_ckpts_l2/tf_lm_variables.npz'
dataset_file = '../data/dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
dictionary_file = '../data/words.dict'
output_dataset_file = '../data/generated_dataset.h5'
output_sentences_file = '../data/generated_dataset_words.txt'
annotated_dataset_file = '../data/annotated_dataset.h5'
start_symbol_index = 2
validation_ratio = 0.1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('EPOCHS', 20,
                            'Num epochs.')
tf.app.flags.DEFINE_integer('VOCAB_SIZE', 10000,
                            'Number of words in the vocabulary.')
tf.app.flags.DEFINE_integer('LATENT_SIZE', 512,
                            'Size of both the hidden state of RNN and random vector z.')
tf.app.flags.DEFINE_integer('SEQUENCE_LEN', 50,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('EMBEDDING_SIZE', 300,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('PATIENCE', 200,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('BATCH_SIZE', 1000,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('NUM_CLASSES', 4,
                            'Number of classes for sentence classification.')
tf.app.flags.DEFINE_integer('NUM_OUTPUT_SENTENCES', 100000,
                            'How many instances should be generated.')
tf.app.flags.DEFINE_string('CELL_TYPE', 'LSTM',
                            'Which RNN cell for the RNNs.')
tf.app.flags.DEFINE_boolean('SAMPLE', False,
                            'Whether to sample from the generator distribution to get fake samples.')
# tf.app.flags.DEFINE_float('HIDDEN_NOISE_STD_DEV', 0, #0.05
#                             'Standard deviation for the gaussian noise added to each time '
#                             + 'step\'s hidden state. To turn off, set = 0')
tf.app.flags.DEFINE_float('VOCAB_NOISE_STD_DEV', 0,
                            'Standard deviation for the gaussian noise added to each time '
                            + 'step\'s output vocab distr. To turn off, set = 0')
tf.app.flags.DEFINE_integer('RANDOM_SEED', 123,
                            'Random seed used for numpy and tensorflow (dropout, sampling)')
tf.app.flags.DEFINE_boolean('GUMBEL', True,
                            'Whether to use Gumbel-Softmax Relaxation')
tf.set_random_seed(FLAGS.RANDOM_SEED)
np.random.seed(FLAGS.RANDOM_SEED)

    
print('loading model parameters')
params = np.load(train_variables_file)
# print (params.keys())
embedding_weights = load.load_embedding_weights()
    
d, word_to_id = load.load_dictionary()

vague_terms = load.load_vague_terms_vector(word_to_id, FLAGS.VOCAB_SIZE)
    
print('loading training and test data')
with h5py.File(dataset_file, 'r') as data_file:
    train_X = data_file['train_X'][:]
    train_Y = data_file['train_Y'][:]
    test_X = data_file['test_X'][:]
    test_Y = data_file['test_Y'][:]
        
# build model
print('building model')

c = tf.placeholder(tf.int32, [None,], 'class')
dims = tf.stack([tf.shape(c)[0],])
vague_terms_tensor = tf.constant(vague_terms, dtype=tf.float32)
def create_vague_weights(vague_terms, fake_c):
    a = tf.tile(vague_terms, dims)
    b = tf.reshape(a,[-1,FLAGS.VOCAB_SIZE])
    vague_weights = tf.multiply(b,tf.cast(tf.reshape(fake_c - 1, [-1,1]),tf.float32))
    return vague_weights
# embedding_matrix = tf.get_variable(shape=[FLAGS.VOCAB_SIZE, FLAGS.EMBEDDING_SIZE],
#                             initializer=tf.contrib.layers.xavier_initializer(), name='embedding_matrix')
vague_weights = create_vague_weights(vague_terms_tensor, c)
z = tf.placeholder(tf.float32, [None, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE], name='z')
ones = tf.ones(shape=[FLAGS.BATCH_SIZE, FLAGS.SEQUENCE_LEN], dtype=tf.int32)
start_symbol_input = [tf.fill(dims, start_symbol_index) for i in range(FLAGS.SEQUENCE_LEN)]
gumbel_noise = z if FLAGS.GUMBEL else None
gumbel_mu = tf.constant(0.)
gumbel_sigma = tf.constant(1.)
cell = utils.create_cell(1.)

def sample_Z(size):
    return np.random.gumbel(size=size)
#     return np.random.normal(size=[m, n], scale=FLAGS.HIDDEN_NOISE_STD_DEV)
def sample_C(m):
    return np.random.randint(low=0, high=FLAGS.NUM_CLASSES, size=m)

W = tf.Variable(tf.random_normal([FLAGS.LATENT_SIZE, FLAGS.VOCAB_SIZE]), name='W')    
b = tf.Variable(tf.random_normal([FLAGS.VOCAB_SIZE]), name='b')    
initial_state = tf.contrib.rnn.LSTMStateTuple(tf.zeros([dims[0], FLAGS.LATENT_SIZE]), tf.zeros([dims[0], FLAGS.LATENT_SIZE]))
outputs, states, samples, probs, logits, pure_logits = embedding_rnn_decoder(start_symbol_input,   # is this ok? I'm not sure what giving 0 inputs does (although it should be completely ignoring inputs)
                                  initial_state,
                                  cell,
                                  FLAGS.VOCAB_SIZE,
                                  FLAGS.EMBEDDING_SIZE,
                                  output_projection=(W,b),
                                  feed_previous=True,
                                  update_embedding_for_previous=True,
                                  sample_from_distribution=FLAGS.SAMPLE,
                                  vague_weights=vague_weights,
#                                   embedding_matrix=embedding_matrix,
                                  hidden_noise_std_dev=None,
                                  vocab_noise_std_dev=None,
                                  gumbel=gumbel_noise,
                                  gumbel_mu=gumbel_mu,
                                  gumbel_sigma=gumbel_sigma)
#                                   class_embedding=c_embedding)
samples = tf.cast(tf.stack(samples, axis=1), tf.int32)
probs = tf.stack(probs, axis=1)
logits = tf.stack(logits, axis=1)

# logits = tf.matmul(output, W) + b
# logits = tf.reshape(logits, [-1, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
tvars = tf.trainable_variables()
tvar_names = [var.name for var in tvars]
# print(tvar_names)
assign_ops = []
for pair in param_names.LSTM_TEST_PARAMS.VARIABLE_PAIRS:
    assign_ops.append(utils.assign_variable_op(params, pair[0], pair[1]))
predictions = tf.cast(tf.argmax(logits, axis=2, name='predictions'), tf.int32)

def remove_trailing_words(samples):
    batch_size = tf.stack([tf.shape(c)[0],])
    eos = tf.fill(batch_size, 3)
    eos = tf.reshape(eos, [-1, 1])
    d=tf.concat([samples,eos],1)
    B=tf.cast(tf.argmax(tf.cast(tf.equal(d, 3), tf.int32), axis=1), tf.int32)
    m=tf.sequence_mask(B, tf.shape(samples)[1], dtype=tf.int32)
    samples=tf.multiply(samples,m)
    return samples
samples = remove_trailing_words(samples)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, targets), "float"))


def batch_generator(x, y):
    data_len = x.shape[0]
    for i in range(0, data_len, FLAGS.BATCH_SIZE):
        batch_x = x[i:min(i+FLAGS.BATCH_SIZE,data_len)]
        batch_y = y[i:min(i+FLAGS.BATCH_SIZE,data_len)]
        yield batch_x, batch_y, i, data_len
        
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    res = np.zeros(x.shape)
#     for i in range(len(x)):
#         res[i] = np.exp(x[i]) / np.sum(np.exp(x[i]), axis=0)
    res = np.exp(x) / np.sum(np.exp(x), axis=0)
    return res


x = np.zeros((FLAGS.NUM_OUTPUT_SENTENCES, FLAGS.SEQUENCE_LEN))
y = np.zeros((FLAGS.NUM_OUTPUT_SENTENCES))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(assign_ops)
#     utils.Progress_Bar.startProgress('generating sentences')	
    for output_idx in tqdm(range(0, FLAGS.NUM_OUTPUT_SENTENCES, FLAGS.BATCH_SIZE), desc='batch'):
    	batch_c = sample_C(FLAGS.BATCH_SIZE)
        batch_z = sample_Z([FLAGS.BATCH_SIZE, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
        batch_samples = sess.run(samples, feed_dict={c: batch_c, z: batch_z})
        x[output_idx:output_idx+FLAGS.BATCH_SIZE] = batch_samples
        y[output_idx:output_idx+FLAGS.BATCH_SIZE] = batch_c
#         utils.Progress_Bar.progress(float(output_idx)/float(FLAGS.NUM_OUTPUT_SENTENCES)*100)
        
        output = ''
        for i in range(min(2,len(batch_samples))):
            for j in range(len(batch_samples[i])):
                if batch_samples[i][j] == 0:
                    if not np.any(batch_samples[i,j:]):
                        break
                    else:
                        output += '_ '
                else:
                    word = d[batch_samples[i][j]]
                    output += word + ' '
            output += '(' + str(int(batch_c[i])) + ')\n'
        print (output)
        
#     utils.Progress_Bar.endProgress()

output = ''
for i in range(len(x)):
    for j in range(len(x[i])):
        if x[i][j] == 0:
            if not np.any(x[i,j:]):
                break
            else:
                output += '_ '
        else:
            word = d[x[i][j]]
            output += word + ' '
    output += '(' + str(int(y[i])) + ')\n'
with open(output_sentences_file, 'w') as f:
    f.write(output)
    
num_val = int(validation_ratio * len(x))
val_x = x[:num_val]
val_y = y[:num_val]
train_x = x[num_val:]
train_y = y[num_val:]
        
outfile = h5py.File(output_dataset_file, 'w')
outfile.create_dataset('train_X', data=train_x)
outfile.create_dataset('train_Y', data=train_y)
outfile.create_dataset('val_X', data=val_x)
outfile.create_dataset('val_Y', data=val_y)
outfile.flush()
outfile.close()

print('done')

















































