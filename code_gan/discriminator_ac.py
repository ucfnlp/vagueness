import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils
from cnn import cnn

FLAGS = tf.app.flags.FLAGS

def discriminator(x_emb, embedding_matrix, keep_prob, EOS_idx):
    with tf.variable_scope("D_"):
        if FLAGS.USE_CNN:
            output = cnn(x_emb, keep_prob, EOS_idx)
        else:
            embeddings_unstacked = tf.unstack(
                tf.reshape(embeddings, [-1, FLAGS.SEQUENCE_LEN, FLAGS.EMBEDDING_SIZE]), axis=1)
            
            cell = utils.create_cell(keep_prob)
            outputs, state = tf.contrib.rnn.static_rnn(
                cell, embeddings_unstacked, dtype=tf.float32)
            output = outputs[-1]
            
#         softmax instaed of sigmoid
#         logit = tf.layers.dense(output, 2)
#         prob = tf.slice(tf.nn.softmax(logit), [0, 1], [-1, 1])
        logit = tf.layers.dense(output, 1, 
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='dense_source')
        prob = tf.nn.sigmoid(logit)
        
        class_logits = tf.layers.dense(output, FLAGS.NUM_CLASSES, 
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='dense_class')
    
        return prob, logit, class_logits