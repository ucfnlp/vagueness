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

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean ' + var.name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev ' + var.name, stddev)
    tf.summary.scalar('max ' + var.name, tf.reduce_max(var))
    tf.summary.scalar('min ' + var.name, tf.reduce_min(var))
    tf.summary.histogram('histogram ' + var.name, var)