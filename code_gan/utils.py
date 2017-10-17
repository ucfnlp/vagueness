from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import sys
from sklearn import metrics
import os

FLAGS = tf.app.flags.FLAGS

def create_cell(keep_prob, reuse=False):
    if FLAGS.CELL_TYPE == 'LSTM':
        cell = BasicLSTMCell(num_units=FLAGS.LATENT_SIZE, activation=tf.nn.tanh,
                              state_is_tuple=False, reuse=reuse)
    elif FLAGS.CELL_TYPE == 'BASIC_RNN':
        cell = BasicRNNCell(num_units=FLAGS.LATENT_SIZE, activation=tf.nn.tanh, reuse=reuse)
    elif FLAGS.CELL_TYPE == 'GRU':
        cell = GRUCell(num_units=FLAGS.LATENT_SIZE, activation=tf.nn.tanh, reuse=reuse)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob, seed=FLAGS.RANDOM_SEED)
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
    
class Metrics:
    def __init__(self):
        self.metrics_collections = []
        
    def print_and_save_metrics(self, y_true, y_pred):
        self.print_metrics(y_true, y_pred)
        self.save_metrics_for_fold(y_true, y_pred)
        
    def save_metrics_for_fold(self, y_true, y_pred):
        self.metrics_collections.append( [metrics.accuracy_score(y_true, y_pred),
                metrics.precision_score(y_true, y_pred, average='weighted'),
                metrics.recall_score(y_true, y_pred, average='weighted'),
                metrics.f1_score(y_true, y_pred, average='weighted')] )
        
    def print_metrics(self, y_true, y_pred):
        print ('Performance Metrics\n-------------------\n')
        print ('Accuracy', metrics.accuracy_score(y_true, y_pred))
        print ('')
        report = metrics.classification_report(y_true,y_pred)
        print (report + '\n')
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        print ('Confusion Matrix\n-------------------\n')
        print ('\t\t',end='')
        for i in range(len(confusion_matrix)):
            print (str(i) + '\t',end='')
        print ('\n')
        for i in range(len(confusion_matrix)):
            print (str(i) + '\t\t',end='')
            for j in range(len(confusion_matrix[i])):
                print (str(confusion_matrix[i,j]) + '\t',end='')
            print ('')
    
    def print_metrics_for_all_folds(self):
        if len(self.metrics_collections) == 0:
            raise Exception('No metrics have been saved')
        accuracy, precision, recall, f1 = tuple(np.mean(np.array(self.metrics_collections), axis=0))
        print ('Average Performance on All Folds\n-------------------\n')
        print ('Accuracy', accuracy)
        print ('Precision', precision)
        print ('Recall', recall)
        print ('F1 Score', f1)
        print ('')
        
class Progress_Bar:
    @staticmethod
    def startProgress(title):
        global progress_x
        sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
        sys.stdout.flush()
        progress_x = 0
    @staticmethod
    def progress(x):
        global progress_x
        x = int(x * 40 // 100)
        sys.stdout.write("#" * (x - progress_x))
        sys.stdout.flush()
        progress_x = x
    @staticmethod
    def endProgress():
        sys.stdout.write("#" * (40 - progress_x) + "]\n")
        sys.stdout.flush()
        
def create_leaky_one_hot_table():
    epsilon = 0.0001
    I = np.eye(FLAGS.VOCAB_SIZE)
    I_excluding_first_row = I[1:,:]     # exclude first row, which is padding
                                        # we don't want to leak padding because the generator creates
                                        # perfect one-hot padding at end of sentence
    
    # add a small probability for each possible word in training set
    I_excluding_first_row[I_excluding_first_row == 0] = epsilon
    I_excluding_first_row[I_excluding_first_row == 0] = 1 - (epsilon * FLAGS.VOCAB_SIZE)
    return I
        
def batch_generator(x, y, batch_size=64, one_hot=False):
    one_hot_table = create_leaky_one_hot_table()
    data_len = x.shape[0]
    for i in range(0, data_len, batch_size):
        x_batch = x[i:min(i+batch_size,data_len)]
        # If giving the discriminator the vocab distribution, then we need to use a 1-hot representation
        if one_hot:
            x_batch = x_batch.astype(float)
            x_batch_transpose = np.transpose(x_batch)
            x_batch_one_hot = one_hot_table[x_batch_transpose.astype(int)]
            x_batch_one_hot_reshaped = x_batch_one_hot.reshape([-1,FLAGS.SEQUENCE_LEN,FLAGS.VOCAB_SIZE])
            
        y_batch = y[i:min(i+batch_size,data_len)]
        if one_hot:
            yield x_batch_one_hot_reshaped, y_batch, i, data_len
        else:
            yield x_batch, y_batch, i, data_len
        
def create_dirs(dir, num_folds):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for fold_num in range(num_folds):
        fold_dir = dir + '/' + str(fold_num)
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        