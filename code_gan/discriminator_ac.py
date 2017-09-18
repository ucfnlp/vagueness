import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils

FLAGS = tf.app.flags.FLAGS

def discriminator(x, embedding_matrix):
    with tf.variable_scope("D_"):
        if not FLAGS.SAMPLE:
            x_stacked = tf.reshape(x, [-1, FLAGS.VOCAB_SIZE])
        
#         embeddings = tf.layers.dense(x_stacked,FLAGS.EMBEDDING_SIZE, use_bias=False, name='embedding')

        if FLAGS.SAMPLE:
            embeddings = tf.nn.embedding_lookup(embedding_matrix, x)
        else:
            embeddings = tf.matmul(x_stacked, embedding_matrix)
        embeddings_unstacked = tf.unstack(
            tf.reshape(embeddings, [-1, FLAGS.SEQUENCE_LEN, FLAGS.EMBEDDING_SIZE]), axis=1)
        
        cell = utils.create_cell()
#         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        outputs, state = tf.contrib.rnn.static_rnn(
            cell, embeddings_unstacked, dtype=tf.float32)
        
#         outputs = tf.nn.dropout(outputs, keep_prob=0.5)
        
        logit = tf.layers.dense(outputs[-1], 1)
        prob = tf.nn.sigmoid(logit)
        
        class_logits = tf.layers.dense(outputs[-1], FLAGS.NUM_CLASSES)
    
        return prob, logit, class_logits