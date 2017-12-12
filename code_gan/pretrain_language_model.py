#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils
import load
import param_names
import argparse
import os
from tqdm import tqdm

models_folder = '../models'
default_model_name = 'lm_ckpts_l2'
dataset_file = '../data/dataset.h5'
summary_file = os.path.join('/home','logan','tensorboard')
fast = False

FLAGS = tf.app.flags.FLAGS
parser = argparse.ArgumentParser()
parser.add_argument('--EPOCHS', default=20, type=int,
                            help='Num epochs.')
parser.add_argument('--VOCAB_SIZE', default=10000, type=int,
                            help='Number of words in the vocabulary.')
parser.add_argument('--LATENT_SIZE', default=512, type=int,
                            help='Size of both the hidden state of RNN and random vector z.')
parser.add_argument('--SEQUENCE_LEN', default=50, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--EMBEDDING_SIZE', default=300, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--PATIENCE', default=200, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--BATCH_SIZE', default=64, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--NUM_CLASSES', default=4, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--CELL_TYPE', default='LSTM', type=str,
                            help='Which RNN cell for the RNNs.')
parser.add_argument('--RANDOM_SEED', default=123, type=int,
                            help='Random seed used for numpy and tensorflow (dropout, sampling)')
parser.add_argument('--L2_LAMBDA', default=1e-4, type=float,
                            help='L2 regularization lambda parameter')
parser.add_argument('--KEEP_PROB', default=1, type=float,
                            help='Dropout probability of keeping a node')
parser.add_argument("--fast", help="run in fast mode for testing",
                    action="store_true")
parser.add_argument("--resume", help="resume from last saved epoch",
                    action="store_true")
parser.add_argument('--name', default=default_model_name, help='Optional name of model, and affects where to save model') 
args = parser.parse_args()

for arg_name, arg_value in vars(args).iteritems():
    if type(arg_value) == bool:
        tf.app.flags.DEFINE_boolean(arg_name, arg_value, docstring='')
    elif type(arg_value) == int:
        tf.app.flags.DEFINE_integer(arg_name, arg_value, docstring='')
    elif type(arg_value) == float:
        tf.app.flags.DEFINE_float(arg_name, arg_value, docstring='')
    elif type(arg_value) == str:
        tf.app.flags.DEFINE_string(arg_name, arg_value, docstring='')

tf.set_random_seed(FLAGS.RANDOM_SEED)
np.random.seed(FLAGS.RANDOM_SEED)

 
if args.fast or fast:
    FLAGS.EPOCHS = 2

ckpt_dir = os.path.join(models_folder, args.name)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    
variables_file = ckpt_dir + '/tf_lm_variables.npz'
    
embedding_weights = load.load_embedding_weights()
d, word_to_id = load.load_dictionary()
train_X, train_Y, train_weights, test_X, test_Y, test_weights = load.load_unannotated_dataset()

if args.fast or fast:
    howmany = 259
    train_X = train_X[:howmany]
    train_Y = train_Y[:howmany]
        
print('building model')
inputs = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='inputs')
targets = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='targets')
weights = tf.placeholder(tf.float32, shape=(None, FLAGS.SEQUENCE_LEN), name='weights')
embedding_tensor = tf.Variable(initial_value=embedding_weights, name='embedding_matrix')
embeddings = tf.nn.embedding_lookup(embedding_tensor, inputs)
cell = utils.create_cell(1.)
embeddings_time_steps = tf.unstack(embeddings, axis=1)
outputs, state = tf.contrib.rnn.static_rnn(
            cell, embeddings_time_steps, dtype=tf.float32)

# is this right?
output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, FLAGS.LATENT_SIZE])
output = tf.nn.dropout(output, FLAGS.KEEP_PROB)

logits = tf.layers.dense(output, FLAGS.VOCAB_SIZE)
logits = tf.reshape(logits, [-1, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        weights,
        average_across_timesteps=False,
        average_across_batch=True
    )
tvars = tf.trainable_variables()
tvar_names = [var.name for var in tvars]
l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name ]) * FLAGS.L2_LAMBDA
cost = tf.reduce_mean(loss) + l2_loss
# TODO: change to rms optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=tvars)
predictions = tf.cast(tf.argmax(logits, axis=2, name='predictions'), tf.int32)
total = tf.reduce_sum(weights)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, targets), "float"))
correct_predictions = tf.logical_and(tf.equal(predictions, targets), tf.cast(weights, tf.bool))
accuracy = tf.reduce_sum(tf.cast(correct_predictions, "float"))/total

utils.variable_summaries(tvars) 
merged = tf.summary.merge_all()

global_step = tf.Variable(-1, name='global_step', trainable=False)
saver = tf.train.Saver()


def idx_to_categorical(y, num_categories):
    categorical_y = np.array(np_utils.to_categorical(y.flatten(), num_categories))
    categorical_y = categorical_y.reshape(-1, y.shape[1], num_categories)
    return categorical_y

def batch_generator(x, y, weights):
    data_len = x.shape[0]
    for i in range(0, data_len, FLAGS.BATCH_SIZE):
        batch_x = x[i:min(i+FLAGS.BATCH_SIZE,data_len)]
        batch_y = y[i:min(i+FLAGS.BATCH_SIZE,data_len)]
        batch_weights = weights[i:min(i+FLAGS.BATCH_SIZE,data_len)]
        yield batch_x, batch_y, batch_weights, i, data_len

with tf.Session() as sess:
    # Create a saver.
#     saver = tf.train.Saver(var_list=tf.trainable_variables())
    tf.add_to_collection('inputs', inputs)
    tf.add_to_collection('predictions', predictions)
    train_writer = tf.summary.FileWriter(os.path.join(summary_file, FLAGS.name), sess.graph)
    tf.global_variables_initializer().run()
    if args.resume:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

    start = global_step.eval() + 1 # get last global_step and start the next one
    print "Start from:", start
        
    step = 0
    for cur_epoch in tqdm(range(start, FLAGS.EPOCHS), desc='Epoch', total=FLAGS.EPOCHS-start):
        for batch_x, batch_y, batch_weights, cur, data_len in tqdm(batch_generator(train_X, train_Y, train_weights), desc='Batch', total=len(train_Y)/FLAGS.BATCH_SIZE):
            batch_cost, batch_accuracy, batch_logits, _, summary = sess.run([cost, accuracy, logits, optimizer, merged], 
                                         feed_dict={inputs:batch_x, targets:batch_y, weights:batch_weights})
            
            test_batch_x = test_X[:FLAGS.BATCH_SIZE]
            test_batch_y = test_Y[:FLAGS.BATCH_SIZE]
            preds = sess.run(predictions, 
                         feed_dict={inputs:test_batch_x[:FLAGS.BATCH_SIZE], targets:test_batch_y})
            for i in range(min(5, len(preds))):
                print d[test_batch_x[i][0]]
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
            
            projection_w = utils.eval_variable(param_names.TRAIN_OUTPUT_WEIGHTS)
            projection_b = utils.eval_variable(param_names.TRAIN_OUTPUT_BIASES)
            emb = utils.eval_variable(param_names.TRAIN_EMBEDDING)
            out = 'Proj W: {:.6}\t'.format(np.mean(np.absolute(projection_w)))
            out += 'Proj B: {:.6}\t'.format(np.mean(np.absolute(projection_b)))
            out += 'Emb: {:.6}\t'.format(np.mean(np.absolute(emb)))
            tqdm.write(out)
            
        print 'saving model to file:'
        global_step.assign(cur_epoch).eval() # set and update(eval) global_step with index, i
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
        vars = sess.run(tvars)
        variables = dict(zip(tvar_names, vars))
        np.savez(variables_file, **variables)
        
        train_writer.add_summary(summary, step)
        step += 1
        

print('done')
    













































