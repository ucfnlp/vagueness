import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell

FLAGS = tf.app.flags.FLAGS

def create_cell(reuse=False):
    if FLAGS.CELL_TYPE == 'LSTM':
        cell = BasicLSTMCell(num_units=FLAGS.LATENT_SIZE, activation=tf.nn.tanh,
                              state_is_tuple=False, reuse=reuse)
    elif FLAGS.CELL_TYPE == 'BASIC_RNN':
        cell = BasicRNNCell(num_units=FLAGS.LATENT_SIZE, activation=tf.nn.tanh, reuse=reuse)
    elif FLAGS.CELL_TYPE == 'GRU':
        cell = GRUCell(num_units=FLAGS.LATENT_SIZE, activation=tf.nn.tanh, reuse=reuse)
    return cell

def get_variable_by_name(tvars, name):
    list = [v for v in tvars if v.name == name]
    if len(list) < 0:
        raise 'No variable found by name: ' + name
    if len(list) > 1:
        raise 'Multiple variables found by name: ' + name
    return list[0]

def assign_variable_op(params, tvars, pretrained_name, cur_name, append=False): #TODO change becuase not using class embedding here
    var = get_variable_by_name(tvars, cur_name)
    pretrained_value = params[pretrained_name]
    if append:
        extra_weights = np.random.normal(size=(FLAGS.CLASS_EMBEDDING_SIZE, pretrained_value.shape[1]))
        pretrained_value = np.concatenate((pretrained_value, extra_weights), axis=0)
    return var.assign(pretrained_value)