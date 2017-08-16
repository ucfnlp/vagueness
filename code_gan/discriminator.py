import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils

FLAGS = tf.app.flags.FLAGS

def discriminator(x):
    with tf.variable_scope("D_"):
        embeddings = tf.get_variable('embeddings_W', [FLAGS.VOCAB_SIZE, FLAGS.EMBEDDING_SIZE], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('embeddings_b', [FLAGS.EMBEDDING_SIZE], dtype=tf.float32, 
                             initializer=tf.zeros_initializer())
        
        x_stacked = tf.reshape(tf.stack(axis=1, values=x), [-1, FLAGS.VOCAB_SIZE])
        embeddings = tf.layers.dense(x_stacked,FLAGS.EMBEDDING_SIZE)
        embeddings_unstacked = tf.unstack(
            tf.reshape(embeddings, [-1, FLAGS.SEQUENCE_LEN, FLAGS.EMBEDDING_SIZE]), axis=1)
        
#         embed = [tf.nn.tanh(tf.matmul(_x, embeddings) + b1) for _x in x]
        
        cell = utils.create_cell()
#         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        outputs, state = tf.contrib.rnn.static_rnn(
            cell, embeddings_unstacked, dtype=tf.float32)
        
#         outputs = tf.nn.dropout(outputs, keep_prob=0.5)
        
        logit = tf.layers.dense(outputs[-1], 1)
        prob = tf.nn.sigmoid(logit)
    
        return prob, logit