import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils
from cnn import cnn

FLAGS = tf.app.flags.FLAGS

def discriminator(x, embedding_matrix, keep_prob):
    with tf.variable_scope("D_"):
        
#         embeddings = tf.layers.dense(x_stacked,FLAGS.EMBEDDING_SIZE, use_bias=False, name='embedding')

        if FLAGS.SAMPLE:
            embeddings = tf.nn.embedding_lookup(embedding_matrix, x)
        else:
            x_stacked = tf.reshape(x, [-1, FLAGS.VOCAB_SIZE])
            embeddings = tf.matmul(x_stacked, embedding_matrix)
            embeddings = tf.reshape(embeddings, [-1, FLAGS.SEQUENCE_LEN, FLAGS.EMBEDDING_SIZE])
#         dropped_embeddings = tf.nn.dropout(embeddings, keep_prob, seed=FLAGS.RANDOM_SEED, name='dropped_embeddings')
        
        if FLAGS.USE_CNN:
            output = cnn(embeddings, keep_prob)
        else:
            embeddings_unstacked = tf.unstack(
                tf.reshape(embeddings, [-1, FLAGS.SEQUENCE_LEN, FLAGS.EMBEDDING_SIZE]), axis=1)
            
            cell = utils.create_cell(keep_prob)
            outputs, state = tf.contrib.rnn.static_rnn(
                cell, embeddings_unstacked, dtype=tf.float32)
            output = outputs[-1]
            
        
        logit = tf.layers.dense(output, 1)
        prob = tf.nn.sigmoid(logit)
        
        class_logits = tf.layers.dense(output, FLAGS.NUM_CLASSES)
    
        return prob, logit, class_logits