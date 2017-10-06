#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils
import load
import argparse
import os

model_file = '../models/tf_lm_model'
variables_file = '../models/tf_lm_variables.npz'
dataset_file = '../data/dataset.h5'
ckpt_dir = '../models/lm_ckpts'
fast = False

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
tf.set_random_seed(123)
np.random.seed(123)


parser = argparse.ArgumentParser()
parser.add_argument("--fast", help="run in fast mode for testing",
                    action="store_true")
parser.add_argument("--resume", help="resume from last saved epoch",
                    action="store_true")
args = parser.parse_args()
 
if args.fast or fast:
    FLAGS.EPOCHS = 2

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    
embedding_weights = load.load_embedding_weights()
d, word_to_id = load.load_dictionary()
train_X, train_Y, test_X, test_Y = load.load_unannotated_dataset()

if args.fast or fast:
    howmany = 259
    train_X = train_X[:howmany]
    train_Y = train_Y[:howmany]
        
print('building model')
inputs = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='inputs')
targets = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='targets')
embedding_tensor = tf.Variable(initial_value=embedding_weights, name='embedding_matrix')
embeddings = tf.nn.embedding_lookup(embedding_tensor, inputs)
cell = utils.create_cell()
embeddings_time_steps = tf.unstack(embeddings, axis=1)
outputs, state = tf.contrib.rnn.static_rnn(
            cell, embeddings_time_steps, dtype=tf.float32)

# is this right?
output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, FLAGS.LATENT_SIZE])
# output = tf.nn.dropout(output, 0.5)
logits = tf.layers.dense(output, FLAGS.VOCAB_SIZE)
logits = tf.reshape(logits, [-1, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
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

global_step = tf.Variable(-1, name='global_step', trainable=False)
saver = tf.train.Saver()


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

with tf.Session() as sess:
    # Create a saver.
#     saver = tf.train.Saver(var_list=tf.trainable_variables())
    tf.add_to_collection('inputs', inputs)
    tf.add_to_collection('predictions', predictions)
#     train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
    tf.global_variables_initializer().run()
    if args.resume:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

    start = global_step.eval() + 1 # get last global_step and start the next one
    print "Start from:", start
        
    for cur_epoch in range(start, FLAGS.EPOCHS):
        for batch_x, batch_y, cur, data_len in batch_generator(train_X, train_Y):
            batch_cost, batch_accuracy, batch_logits, _ = sess.run([cost, accuracy, logits, optimizer], 
                                                     feed_dict={inputs:batch_x, targets:batch_y})
            
            test_batch_x = test_X[:FLAGS.BATCH_SIZE]
            test_batch_y = test_Y[:FLAGS.BATCH_SIZE]
            preds = sess.run([predictions], 
                             feed_dict={inputs:test_batch_x[:FLAGS.BATCH_SIZE], targets:test_batch_y})
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
                    print ''
                print '\n'
            print(preds)
            print('Iter: {}'.format(cur_epoch))
            print('Instance ', cur, ' out of ', data_len)
            print('Loss ', batch_cost)
            print('Accuracy ', batch_accuracy)
            
        print 'saving model to file:'
        global_step.assign(cur_epoch).eval() # set and update(eval) global_step with index, i
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
        vars = sess.run(tvars)
        variables = dict(zip(tvar_names, vars))
        np.savez(variables_file, **variables)
        

print('done')

















































