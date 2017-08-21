#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
from seq2seq import  embedding_rnn_decoder
import utils
import param_names
import argparse

np.random.seed(123)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, merge, Input, TimeDistributed, Bidirectional
from keras.layers import Embedding, LSTM, GRU, Flatten
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras import metrics
from keras.callbacks import EarlyStopping

train_model_file = '../models/tf_lm_model'
# train_variables_file = '../models/tf_lm_variables (copy).npz'
train_variables_file = '../models/tf_enc_dec_variables.npz'
test_model_file = '../models/tf_lm_test_model'
test_variables_file = '../models/tf_lm_test_variables.npz'
dataset_file = '../data/dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
dictionary_file = '../data/words.dict'

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
tf.app.flags.DEFINE_integer('BATCH_SIZE', 128,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_string('CELL_TYPE', 'GRU',
                            'Which RNN cell for the RNNs.')

parser = argparse.ArgumentParser()
parser.add_argument("--fast", help="run in fast mode for testing",
                    action="store_true")
args = parser.parse_args()
 
if args.fast:
    FLAGS.EPOCHS = 2
    
print('loading model parameters')
params = np.load(train_variables_file)
    
print('loading embedding weights')
with h5py.File(embedding_weights_file, 'r') as hf:
    embedding_weights = hf['embedding_weights'][:]
    
print('loading dictionary')
d = {}
with open(dictionary_file) as f:
    for line in f:
       (val, key) = line.split()
       d[int(key)] = val
    
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
# my_input = Input(shape=(maxlen,), dtype='int32')
# 
# embedded = Embedding(vocab_size, 
#               embedding_dim, 
#               input_length=maxlen, 
#               weights=[embedding_weights], 
#               dropout=0.2,
#               trainable=False)(my_input)
#               
# forwards = Bidirectional(GRU(hidden_dim,
#                return_sequences=True,
#                dropout_W=0.2,
#                dropout_U=0.2))(embedded)
# 
# output = Dropout(0.5)(forwards)
# 
# # output_vague = TimeDistributed(Dense(1, activation='sigmoid'), name='loss_vague')(output)
# output_lm = TimeDistributed(Dense(vocab_size, activation='softmax'), name='loss_lm')(output)



# # Load the VGG-16 model in the default graph
# saver = tf.train.import_meta_graph('gru-model.meta')
# # saver = tf.train.import_meta_graph(
# #     meta_graph_def='gru-model.meta', 
# #     input_map={"lstm-cell-weights-name": weights})
# # Access the graph
# gru_graph = tf.get_default_graph()
# 
# # Retrieve VGG inputs
# embedding_tensor = gru_graph.get_tensor_by_name('embedding_matrix:0')
# tf.get_variable_scope().reuse_variables()

inputs = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='inputs')
# targets = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='targets')
z = tf.placeholder(tf.float32, [None, FLAGS.LATENT_SIZE], name='z')
# zero_inputs = tf.unstack(tf.zeros_like(inputs, dtype=tf.int32), axis=1)
ones = tf.ones_like(inputs, dtype=tf.int32)
start_symbol_input = tf.unstack(ones + ones, axis=1)
# embedding_tensor = tf.Variable(initial_value=embedding_weights, name='embedding_matrix')
# embeddings = tf.nn.embedding_lookup(embedding_tensor, inputs)
cell = utils.create_cell()
# embeddings_time_steps = tf.unstack(embeddings, axis=1)
# outputs, state = tf.contrib.rnn.static_rnn(
#             cell, embeddings_time_steps, dtype=tf.float32)

def sample_Z(m, n):
    return np.zeros((m, n))
#     return np.random.normal(size=[m, n])

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
                          sample_from_distribution=True)

# is this right?
output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, FLAGS.LATENT_SIZE])
# output = tf.nn.dropout(output, 0.5)
# logits = tf.layers.dense(output, FLAGS.VOCAB_SIZE)
logits = tf.matmul(output, W) + b
logits = tf.reshape(logits, [-1, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
# loss = tf.contrib.seq2seq.sequence_loss(
#         logits,
#         targets,
#         tf.ones_like(inputs, dtype=tf.float32),
#         average_across_timesteps=False,
#         average_across_batch=True
#     )
# cost = tf.reduce_sum(loss)
tvars = tf.trainable_variables()
tvar_names = [var.name for var in tvars]
    
assign_ops = []
for pair in param_names.ENC_DEC_PARAMS.VARIABLE_PAIRS:
    assign_ops.append(utils.assign_variable_op(params, tvars, pair[0], pair[1]))
# TODO: change to rms optimizer
# optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=tvars)
predictions = tf.cast(tf.argmax(logits, axis=2, name='predictions'), tf.int32)
samples = tf.stack(samples, axis=1)
probs = tf.stack(probs, axis=1)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, targets), "float"))


def idx_to_categorical(y, num_categories):
    categorical_y = np.array(np_utils.to_categorical(y.flatten(), num_categories))
    categorical_y = categorical_y.reshape(-1, y.shape[1], num_categories)
    return categorical_y

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

with tf.Session() as sess:
#     train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
    tf.global_variables_initializer().run()
    sess.run(assign_ops)
    for cur_epoch in range(FLAGS.EPOCHS):
        for batch_x, batch_y, cur, data_len in batch_generator(train_X, train_Y):
            batch_z = sample_Z(batch_x.shape[0], FLAGS.LATENT_SIZE)
            batch_logits, batch_predictions, batch_samples, batch_probs = sess.run(
                        [logits, predictions, samples, probs], 
                        feed_dict={inputs:batch_x, z:batch_z})
            
#             preds = batch_predictions
            preds = batch_samples
            a = batch_logits[0]
            b=softmax(a)
            c=np.argmax(b)
            e=np.max(b)
            m = c/5000
            n = c%5000
            
            for i in range(min(2, len(preds))):
                for j in range(len(preds[i])):
#                     if test_batch_y[i][j] == 0:
#                         print '<>',
#                     else:
#                         word = d[test_batch_y[i][j]]
#                         print word,
#                     print '\t\t',
                    if preds[i][j] == 0:
                        print '<>',
                    else:
                        word = d[preds[i][j]]
                        print word,
                print '\n'
            print(preds)
#             sorted_logits = np.sort(batch_logits[0])[::-1]
#             for j in range(3):
#                 for k in range(20):
#                     print sorted_logits[j][k]
#                 print '\n'
            print('Iter: {}'.format(cur_epoch))
            print('Instance ', cur, ' out of ', data_len)
#             print('Loss ', batch_cost)
#             print('Accuracy ', batch_accuracy)
            
    
        

print('done')

















































