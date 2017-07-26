import numpy as np
import tensorflow as tf
from seq2seq import  embedding_rnn_decoder
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils

FLAGS = tf.app.flags.FLAGS

def generator(z, zero_inputs):
    with tf.variable_scope("G_"):
        cell = utils.create_cell()
#         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        
        W = tf.Variable(tf.random_normal([FLAGS.LATENT_SIZE, FLAGS.VOCAB_SIZE]), name='W')    
        b = tf.Variable(tf.random_normal([FLAGS.VOCAB_SIZE]), name='b')    
        
        outputs, states = embedding_rnn_decoder(zero_inputs,   # is this ok? I'm not sure what giving 0 inputs does (although it should be completely ignoring inputs)
                                  z,
                                  cell,
                                  FLAGS.VOCAB_SIZE,
                                  FLAGS.EMBEDDING_SIZE,
                                  output_projection=(W,b),
                                  feed_previous=True,
                                  update_embedding_for_previous=True)

        logits = [tf.matmul(output, W) + b for output in outputs]
        x = [tf.nn.softmax(logit) for logit in logits] # is this softmaxing over the right dimension? this turns into 3D
        return x