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

train_variables_file = '../models/tf_lm_variables.npz'
dataset_file = '../data/dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
dictionary_file = '../data/words.dict'
output_dataset_file = '../data/generated_dataset.h5'
output_sentences_file = '../data/generated_dataset_words.txt'
annotated_dataset_file = '../data/annotated_dataset.h5'

validation_ratio = 0.1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('EPOCHS', 20,
                            'Num epochs.')
tf.app.flags.DEFINE_integer('VOCAB_SIZE', 5000,
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
tf.app.flags.DEFINE_string('CELL_TYPE', 'GRU',
                            'Which RNN cell for the RNNs.')
tf.app.flags.DEFINE_boolean('SAMPLE', False,
                            'Whether to sample from the generator distribution to get fake samples.')
tf.app.flags.DEFINE_boolean('OUTPUT_EMBEDDING', False,
                            'If true, output layer is the embedding size, otherwise the vocab size.')
tf.app.flags.DEFINE_string('EMBEDDING_TRAINABLE', 'mixed',
                            'trainable, fixed, or mixed')
tf.app.flags.DEFINE_float('HIDDEN_NOISE_STD_DEV', 0, #0.05
                            'Standard deviation for the gaussian noise added to each time '
                            + 'step\'s hidden state. To turn off, set = 0')
tf.app.flags.DEFINE_float('VOCAB_NOISE_STD_DEV', 1,
                            'Standard deviation for the gaussian noise added to each time '
                            + 'step\'s output vocab distr. To turn off, set = 0')
tf.app.flags.DEFINE_integer('RANDOM_SEED', 123,
                            'Random seed used for numpy and tensorflow (dropout, sampling)')
tf.set_random_seed(FLAGS.RANDOM_SEED)
np.random.seed(FLAGS.RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--fast", help="run in fast mode for testing",
                    action="store_true")
parser.add_argument("--output_embedding", help="If true, output layer is the embedding size, otherwise the vocab size.",
                    action="store_true")
args = parser.parse_args()

if args.output_embedding:
    FLAGS.OUTPUT_EMBEDDING = True
    ckpt_dir = '../models/lm_output_embedding_' + FLAGS.EMBEDDING_TRAINABLE + '_ckpts'
    train_variables_file = ckpt_dir + '/variables.npz'
    output_dataset_file = '../data/generated_output_embedding_dataset.h5'
    output_sentences_file = '../data/generated_output_embedding_dataset_words.txt'
 
if args.fast:
    FLAGS.EPOCHS = 2
    
print('loading model parameters')
params = np.load(train_variables_file)
    
embedding_weights = load.load_embedding_weights()
    
d, word_to_id = load.load_dictionary()

vague_terms = load.load_vague_terms_vector(word_to_id, FLAGS.VOCAB_SIZE)
    
print('loading training and test data')
with h5py.File(dataset_file, 'r') as data_file:
    train_X = data_file['train_X'][:]
    train_Y = data_file['train_Y'][:]
    test_X = data_file['test_X'][:]
    test_Y = data_file['test_Y'][:]

if args.fast:
    howmany = 259
    train_X = train_X[:howmany]
    train_Y = train_Y[:howmany]
        
# build model
print('building model')

# inputs = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='inputs')
# targets = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='targets')
c = tf.placeholder(tf.int32, [None,], 'class')
dims = tf.stack([tf.shape(c)[0],])
vague_terms_tensor = tf.constant(vague_terms, dtype=tf.float32)
def create_vague_weights(vague_terms, fake_c):
    a = tf.tile(vague_terms, dims)
    b = tf.reshape(a,[-1,FLAGS.VOCAB_SIZE])
    vague_weights = tf.multiply(b,tf.cast(tf.reshape(fake_c*2 - 2, [-1,1]),tf.float32))
    return vague_weights
vague_weights = create_vague_weights(vague_terms_tensor, c)
# z = tf.zeros(dtype=tf.float32, shape=[FLAGS.BATCH_SIZE, FLAGS.LATENT_SIZE], name='z')
z = tf.placeholder(tf.float32, [FLAGS.BATCH_SIZE, FLAGS.LATENT_SIZE], name='z')
# zero_inputs = tf.unstack(tf.zeros_like(inputs, dtype=tf.int32), axis=1)
ones = tf.ones(shape=[FLAGS.BATCH_SIZE, FLAGS.SEQUENCE_LEN], dtype=tf.int32)
start_symbol_input = tf.unstack(ones + ones, axis=1)
embedding_tensor_fixed = tf.Variable(initial_value=embedding_weights, name='embedding_matrix', trainable=False)
# embedding_tensor = tf.Variable(initial_value=embedding_weights, name='embedding_matrix')
# embeddings = tf.nn.embedding_lookup(embedding_tensor, inputs)
cell = utils.create_cell(1.)
# embeddings_time_steps = tf.unstack(embeddings, axis=1)
# outputs, state = tf.contrib.rnn.static_rnn(
#             cell, embeddings_time_steps, dtype=tf.float32)

def sample_Z(m, n):
    return np.zeros((m, n))
#     return np.random.normal(size=[m, n], scale=FLAGS.HIDDEN_NOISE_STD_DEV)
def sample_C(m):
    return np.random.randint(low=0, high=FLAGS.NUM_CLASSES, size=m)

if FLAGS.OUTPUT_EMBEDDING:
    W = tf.Variable(tf.random_normal([FLAGS.LATENT_SIZE, FLAGS.EMBEDDING_SIZE]), name='W')    
    b = tf.Variable(tf.random_normal([FLAGS.EMBEDDING_SIZE]), name='b')    
    outputs, states, samples = embedding_rnn_decoder(start_symbol_input,   # is this ok? I'm not sure what giving 0 inputs does (although it should be completely ignoring inputs)
                                      z,
                                      cell,
                                      FLAGS.VOCAB_SIZE,
                                      FLAGS.EMBEDDING_SIZE,
                                      output_projection=(W,b),
                                      feed_previous=True,
                                      update_embedding_for_previous=True,
                                      sample_from_distribution=False,
                                      fixed_embedding=embedding_tensor_fixed,
                                      hidden_noise_std_dev=FLAGS.HIDDEN_NOISE_STD_DEV,
                                      vocab_noise_std_dev=FLAGS.VOCAB_NOISE_STD_DEV)
    #                                   class_embedding=c_embedding)
    samples = tf.cast(tf.stack(samples, axis=1), tf.int32)
else:
    W = tf.Variable(tf.random_normal([FLAGS.LATENT_SIZE, FLAGS.VOCAB_SIZE]), name='W')    
    b = tf.Variable(tf.random_normal([FLAGS.VOCAB_SIZE]), name='b')    
    outputs, states, samples, probs = embedding_rnn_decoder(start_symbol_input,   # is this ok? I'm not sure what giving 0 inputs does (although it should be completely ignoring inputs)
                                      z,
                                      cell,
                                      FLAGS.VOCAB_SIZE,
                                      FLAGS.EMBEDDING_SIZE,
                                      output_projection=(W,b),
                                      feed_previous=True,
                                      update_embedding_for_previous=True,
                                      sample_from_distribution=FLAGS.SAMPLE,
                                      vague_weights=vague_weights,
                                      hidden_noise_std_dev=FLAGS.HIDDEN_NOISE_STD_DEV,
                                      vocab_noise_std_dev=FLAGS.VOCAB_NOISE_STD_DEV)
    #                                   class_embedding=c_embedding)
    samples = tf.cast(tf.stack(samples, axis=1), tf.int32)
    probs = tf.stack(probs, axis=1)
    



# is this right?
output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, FLAGS.LATENT_SIZE])
# output = tf.nn.dropout(output, 0.5)
# logits = tf.layers.dense(output, FLAGS.VOCAB_SIZE)
logits = tf.matmul(output, W) + b
if FLAGS.OUTPUT_EMBEDDING:
    logits = tf.layers.dense(output, FLAGS.EMBEDDING_SIZE)
    logits = tf.reshape(logits, [-1, FLAGS.SEQUENCE_LEN, FLAGS.EMBEDDING_SIZE])
else:
    logits = tf.reshape(logits, [-1, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
tvars = tf.trainable_variables()
tvar_names = [var.name for var in tvars]
    
assign_ops = []
for pair in param_names.GRU_TEST_PARAMS.VARIABLE_PAIRS:
    assign_ops.append(utils.assign_variable_op(params, tvars, pair[0], pair[1]))
# TODO: change to rms optimizer
# optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=tvars)
predictions = tf.cast(tf.argmax(logits, axis=2, name='predictions'), tf.int32)
# samples = tf.stack(samples, axis=1)
def remove_trailing_words(samples):
    batch_size = tf.stack([tf.shape(c)[0],])
    eos = tf.fill(batch_size, 3)
    eos = tf.reshape(eos, [-1, 1])
    d=tf.concat([samples,eos],1)
    B=tf.cast(tf.argmax(tf.cast(tf.equal(d, 3), tf.int32), axis=1), tf.int32)
    m=tf.sequence_mask(B, tf.shape(samples)[1], dtype=tf.int32)
    samples=tf.multiply(samples,m)
    return samples
if not FLAGS.OUTPUT_EMBEDDING:
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
    utils.Progress_Bar.startProgress('generating sentences')	
    for output_idx in range(0, FLAGS.NUM_OUTPUT_SENTENCES, FLAGS.BATCH_SIZE):
    	batch_c = sample_C(FLAGS.BATCH_SIZE)
        batch_z = sample_Z(FLAGS.BATCH_SIZE, FLAGS.LATENT_SIZE)
        batch_samples = sess.run(samples, feed_dict={c: batch_c, z: batch_z})
        x[output_idx:output_idx+FLAGS.BATCH_SIZE] = batch_samples
        y[output_idx:output_idx+FLAGS.BATCH_SIZE] = batch_c
        utils.Progress_Bar.progress(float(output_idx)/float(FLAGS.NUM_OUTPUT_SENTENCES)*100)
        
        output = ''
        for i in range(min(2,len(batch_samples))):
            for j in range(len(batch_samples[i])):
                if batch_samples[i][j] == 0:
                    if not np.any(batch_samples[i,j:]):
                        break
                    else:
                        output += '<UNK> '
                else:
                    word = d[batch_samples[i][j]]
                    output += word + ' '
            output += '(' + str(int(batch_c[i])) + ')\n'
        print (output)
        
    utils.Progress_Bar.endProgress()

output = ''
for i in range(len(x)):
    for j in range(len(x[i])):
        if x[i][j] == 0:
            if not np.any(x[i,j:]):
                break
            else:
                output += '<UNK> '
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

















































