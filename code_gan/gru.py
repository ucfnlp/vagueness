#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function
import numpy
import tensorflow as tf
import h5py
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils
import argparse

numpy.random.seed(123)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, merge, Input, TimeDistributed, Bidirectional
from keras.layers import Embedding, LSTM, GRU, Flatten
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras import metrics
from keras.callbacks import EarlyStopping

model_file = '../models/tf_lm_model'
variables_file = '../models/tf_lm_variables.npz'
dataset_file = '../data/dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
dictionary_file = '../data/words.dict'
 
vocab_size = 5000
embedding_dim = 300
maxlen = 50
hidden_dim = 512
batch_size = 128
nb_epoch = 20
samples_per_epoch = None
fast = False

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('EPOCHS', 5000,
                            'Num epochs.')
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
 
if args.fast or fast:
    nb_epoch = 1
    
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

if args.fast or fast:
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




inputs = tf.placeholder(tf.int32, shape=(None, maxlen), name='inputs')
targets = tf.placeholder(tf.int32, shape=(None, maxlen), name='targets')
embedding_tensor = tf.Variable(initial_value=embedding_weights, name='embedding_matrix')
embeddings = tf.nn.embedding_lookup(embedding_tensor, inputs)
cell = utils.create_cell()
embeddings_time_steps = tf.unstack(embeddings, axis=1)
outputs, state = tf.contrib.rnn.static_rnn(
            cell, embeddings_time_steps, dtype=tf.float32)

# is this right?
output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, hidden_dim])
# output = tf.nn.dropout(output, 0.5)
logits = tf.layers.dense(output, vocab_size)
logits = tf.reshape(logits, [-1, maxlen, vocab_size])
loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones_like(inputs, dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True
    )
cost = tf.reduce_sum(loss)
tvars = tf.trainable_variables()
tvar_names = [var.name for var in tvars]
# TODO: change to rms optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=tvars)
predictions = tf.cast(tf.argmax(logits, axis=2, name='predictions'), tf.int32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, targets), "float"))


def idx_to_categorical(y, num_categories):
    categorical_y = numpy.array(np_utils.to_categorical(y.flatten(), num_categories))
    categorical_y = categorical_y.reshape(-1, y.shape[1], num_categories)
    return categorical_y

def batch_generator(x, y):
    data_len = x.shape[0]
    for i in range(0, data_len, batch_size):
        batch_x = x[i:min(i+batch_size,data_len)]
        batch_y = y[i:min(i+batch_size,data_len)]
        yield batch_x, batch_y, i, data_len

with tf.Session() as sess:
    # Create a saver.
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    tf.add_to_collection('inputs', inputs)
    tf.add_to_collection('predictions', predictions)
    saver2 = tf.train.Saver(
        [embedding_tensor])
#     train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
    tf.global_variables_initializer().run()
    for cur_epoch in range(nb_epoch):
        for batch_x, batch_y, cur, data_len in batch_generator(train_X, train_Y):
            batch_cost, batch_accuracy, batch_logits, _ = sess.run([cost, accuracy, logits, optimizer], 
                                                     feed_dict={inputs:batch_x, targets:batch_y})
            
            test_batch_x = test_X[:batch_size]
            test_batch_y = test_Y[:batch_size]
            preds = sess.run([predictions], 
                             feed_dict={inputs:test_batch_x[:batch_size], targets:test_batch_y})
            preds = preds[0]
            for i in range(min(2, len(preds))):
                for j in range(len(preds[i])):
                    if test_batch_y[i][j] == 0:
                        print '<>',
                    else:
                        word = d[test_batch_y[i][j]]
                        print word,
                    print '\t\t',
                    if preds[i][j] == 0:
                        print '<>',
                    else:
                        word = d[preds[i][j]]
                        print word,
                    print '\n'
                print '\n'
            print(preds)
            print('Iter: {}'.format(cur_epoch))
            print('Instance ', cur, ' out of ', data_len)
            print('Loss ', batch_cost)
            print('Accuracy ', batch_accuracy)
            
    print 'saving model to file:'
    saver.save(sess, model_file)
    vars = sess.run(tvars)
    variables = dict(zip(tvar_names, vars))
    numpy.savez(variables_file, **variables)
        

print('done')

















































