#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
import utils
import load
import argparse
import os
from sklearn import metrics
import time
from tqdm import tqdm
import nltk
from nltk.tag import pos_tag, map_tag
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--train_only", help="run in train mode only",
                    action="store_true")
parser.add_argument("--test_only", help="run in test mode only",
                    action="store_true")
parser.add_argument("--resume", help="Resume training from last checkpoint",
                    action="store_true")
parser.add_argument("--one_fold", help="perform only on one fold instead of five-fold cross validation",
                    action="store_true")
args = parser.parse_args()

prediction_words_file = 'word-level context-aware predictions.html'
prediction_folder = '../predictions'
summary_file = '/home/logan/tmp'
ckpt_dir = '../models/word_level_lstm'
variables_file = ckpt_dir + '/variables.npz'
fast = False
num_folds = 5
start_time = time.time()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('EPOCHS', 20,
                            'Num epochs.')
tf.app.flags.DEFINE_integer('VOCAB_SIZE', 10000,
                            'Number of words in the vocabulary.')
tf.app.flags.DEFINE_integer('LATENT_SIZE', 256,
                            'Size of both the hidden state of RNN and random vector z.')
tf.app.flags.DEFINE_integer('SEQUENCE_LEN', 50,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('EMBEDDING_SIZE', 300,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('PATIENCE', 3,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('BATCH_SIZE', 64,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_string('CELL_TYPE', 'LSTM',
                            'Which RNN cell for the RNNs.')
tf.app.flags.DEFINE_integer('RANDOM_SEED', 123,
                            'Random seed used for numpy and tensorflow (dropout, sampling)')
tf.set_random_seed(FLAGS.RANDOM_SEED)
np.random.seed(FLAGS.RANDOM_SEED)

'''
--------------------------------

LOAD DATA

--------------------------------
'''
    
# Make directories for model files and prediction files
utils.create_dirs(ckpt_dir, num_folds)
utils.create_dirs(prediction_folder, num_folds)
        
embedding_weights = load.load_embedding_weights()
d, word_to_id = load.load_dictionary()
d[0] = '_'
Metrics = utils.Metrics(is_binary=True)
        
'''
--------------------------------

MODEL

--------------------------------
'''
        
print('building model')
inputs = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='inputs')
targets = tf.placeholder(tf.int32, shape=(None, FLAGS.SEQUENCE_LEN), name='targets')
weights = tf.placeholder(tf.float32, shape=(None, FLAGS.SEQUENCE_LEN), name='weights')
embedding_tensor = tf.Variable(initial_value=embedding_weights, name='embedding_matrix')
embeddings = tf.nn.embedding_lookup(embedding_tensor, inputs)
with tf.variable_scope('fw'):
    cell_fw = utils.create_cell(1.)
with tf.variable_scope('bw'):
    cell_bw = utils.create_cell(1.)
embeddings_time_steps = tf.unstack(embeddings, axis=1)
outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
            cell_fw, cell_bw, embeddings_time_steps, dtype=tf.float32)

# is this right?
output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, FLAGS.LATENT_SIZE*2])
# output = tf.nn.dropout(output, 0.5)

logits = tf.layers.dense(output, 2)
logits = tf.reshape(logits, [-1, FLAGS.SEQUENCE_LEN, 2])
loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        weights,
        average_across_timesteps=True,
        average_across_batch=True
    )
# cost = tf.reduce_sum(loss)
cost = loss
tvars = tf.trainable_variables()
tvar_names = [var.name for var in tvars]
# TODO: change to rms optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=tvars)
predictions = tf.cast(tf.argmax(logits, axis=2, name='predictions'), tf.int32)
total = tf.reduce_sum(weights)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, targets), "float"))
correct_predictions = tf.logical_and(tf.equal(predictions, targets), tf.cast(weights, tf.bool))
accuracy = tf.reduce_sum(tf.cast(correct_predictions, "float"))/total


global_step = tf.Variable(-1, name='global_step', trainable=False)
saver = tf.train.Saver()

utils.variable_summaries(tvars) 
merged = tf.summary.merge_all()


def batch_generator(x, y, weights, batch_size):
    data_len = x.shape[0]
    for i in range(0, data_len, batch_size):
        batch_x = x[i:min(i+batch_size,data_len)]
        batch_y = y[i:min(i+batch_size,data_len)]
        batch_weights = weights[i:min(i+batch_size,data_len)]
        yield batch_x, batch_y, batch_weights, i, data_len
        
def save_predictions_to_file(x, y, weights, predict, fold_num):
    file_name = os.path.join(prediction_folder, str(fold_num), prediction_words_file)
    out = "<font color='green'>green</font>=true positive | <font color='blue'>blue</font>=false negative | "
    out += "<font color='red'>red</font>=false positive | black=true negative.<br><br>"
    for i in range(x.shape[0]):
        line = ''
        for j in range(x.shape[1]):
            idx = x[i][j]
            if idx == 0:
                continue
            word = d[idx]
            if y[i][j] == 1 and predict[i][j] == 1:
                start_tag = "<font color='green'>"
                end_tag = "</font>"
            elif y[i][j] == 1 and predict[i][j] == 0:
                start_tag = "<font color='blue'>"
                end_tag = "</font>"
            elif y[i][j] == 0 and predict[i][j] == 1:
                start_tag = "<font color='red'>"
                end_tag = "</font>"
            else:
                start_tag = ""
                end_tag = ""
            line += start_tag + word + end_tag + ' '
        line += '<br>'
        out += line
    with open(file_name, 'w') as f:
        f.write(out)

            
def validate(sess, val_x, val_y, val_weights):
    def run_val_step(sess, batch_x, batch_y, batch_weights):
        to_return = cost
        return sess.run(to_return,
                    feed_dict={inputs: batch_x,
                               targets: batch_y,
                               weights: batch_weights})
    sum_cost = 0
    for batch_x, batch_y, batch_weights, cur, data_len in batch_generator(val_x, val_y, val_weights, FLAGS.BATCH_SIZE):
        batch_cost = run_val_step(sess, batch_x, batch_y, batch_weights)
        sum_cost += batch_cost * len(batch_y)       # Add up cost based on how many instances in batch
    val_cost = sum_cost / len(val_y)
    return val_cost
    
    
def train(train_x, train_y, train_weights, val_x, val_y, val_weights, fold_num):
    
    def run_train_step(sess, batch_x, batch_y, batch_weights):
        to_return = [optimizer, cost, accuracy, merged]
        return sess.run(to_return,
                    feed_dict={inputs: batch_x,
                               targets: batch_y,
                               weights: batch_weights})
        
    
    print 'training'
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
        saver = tf.train.Saver(max_to_keep=5)
        tf.global_variables_initializer().run()
        min_val_cost = np.inf
        num_mistakes = 0
        
        fold_ckpt_dir = ckpt_dir + '/' + str(fold_num)
        variables_file = fold_ckpt_dir + '/variables_'
        if args.resume:
            ckpt = tf.train.get_checkpoint_state(fold_ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print ckpt.model_checkpoint_path
                model.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
    
        start = global_step.eval() + 1 # get last global_step and start the next one
        print "Start from:", start
        
        step = 0
        for cur_epoch in tqdm(range(start, FLAGS.EPOCHS), desc='Fold ' + str(fold_num) + ': Epoch'):
            for batch_x, batch_y, batch_weights, _, _ in tqdm(batch_generator(train_x, train_y, train_weights, FLAGS.BATCH_SIZE), desc='Batch', total=len(train_y)/FLAGS.BATCH_SIZE):
                _, batch_cost, batch_accuracy, summary = run_train_step(sess, batch_x, batch_y, batch_weights)
                train_writer.add_summary(summary, step)
                
#                 tqdm.write('Fold', fold_num, 'Epoch: ', cur_epoch,)
                tqdm.write('Loss: {:.4} \tAcc: {:.4}'. format(batch_cost, batch_accuracy))
            
            tqdm.write('saving model to file:')
            global_step.assign(cur_epoch).eval()  # set and update(eval) global_step with index, cur_epoch
            saver.save(sess, fold_ckpt_dir + "/model.ckpt", global_step=cur_epoch)
            vars = sess.run(tvars)
            tvar_names = [var.name for var in tf.trainable_variables()]
            variables = dict(zip(tvar_names, vars))
            np.savez(variables_file, **variables)
            
            val_cost = validate(sess, val_x, val_y, val_weights)
            tqdm.write('Val Loss: ' + str(val_cost))
            if val_cost < min_val_cost:
                min_val_cost = val_cost
            else:
                num_mistakes += 1
            if num_mistakes >= FLAGS.PATIENCE:
                tqdm.write('Stopping early at epoch: ' + str(cur_epoch))
                break
            
        
    
def test(test_x, test_y, test_weights, fold_num):
    
    def run_test_step(sess, batch_x):
        to_return = predictions
        return sess.run(to_return,
                        feed_dict={inputs: batch_x})
        return predictions_indices, f1_score
    def predict(sess, x, y, weights):
        total_predictions = []
        for batch_x, batch_y, batch_weights, cur, data_len in tqdm(batch_generator(x, y, weights, batch_size=1), desc='Fold ' + str(fold_num) + ':Testing', total=len(y)):
            batch_predictions = run_test_step(sess, batch_x)
            total_predictions.append(batch_predictions)
        total_predictions = np.squeeze(np.array(total_predictions))
        f1_score = metrics.f1_score(y.flatten(), total_predictions.flatten(), average='binary', sample_weight=test_weights.flatten())
        return total_predictions, f1_score
    
    print 'testing'
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        fold_ckpt_dir = ckpt_dir + '/' + str(fold_num)
        ckpt = tf.train.get_checkpoint_state(fold_ckpt_dir)
        if not ckpt:
            raise Exception('Could not find saved model in: ' + fold_ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        total_predictions, _ = predict(sess, test_x, test_y, test_weights)
        Metrics.print_and_save_metrics(test_y.flatten(), total_predictions.flatten(), weights=test_weights.flatten())
        save_predictions_to_file(test_x, test_y, test_weights, total_predictions, fold_num)
        return total_predictions
        
        

def count_POS(y_pred, test_x, test_y, test_weights):
    FP = np.logical_and(y_pred == 1, test_y == 0)
    FP = np.logical_and(FP, test_weights)
    FN = np.logical_and(y_pred == 0, test_y == 1)
    FN = np.logical_and(FN, test_weights)
    
    FP_counts = Counter()
    FN_counts = Counter()
    
    for sent_idx, cur_sent in enumerate(test_x):
        cur_predict = y_pred[sent_idx]
        words = np.array([ d[id] for id in cur_sent])
#             print(text)
        posTagged = pos_tag(words)
        tags = [item[1] for item in posTagged]
        simplified_tags = [map_tag('en-ptb', 'universal', tag) for tag in tags]
        for word_idx, tag in enumerate(tags):
            if test_weights[sent_idx, word_idx] != 0:
                if y_pred[sent_idx, word_idx] == 1 and test_y[sent_idx, word_idx] == 0:
                    if tag == 'NNP':
                        FP_counts[tag] += 1
                    else:
                        simplified_tag = simplified_tags[word_idx]
                        FP_counts[simplified_tag] += 1
                elif y_pred[sent_idx, word_idx] == 0 and test_y[sent_idx, word_idx] == 1:
                    if tag == 'NNP':
                        FN_counts[tag] += 1
                    else:
                        simplified_tag = simplified_tags[word_idx]
                        FN_counts[simplified_tag] += 1
    
    FP_percent = [(i, 1.0*FP_counts[i] / sum(FP_counts.values()) * 100.0) for i in FP_counts]
    FN_percent = [(i, 1.0*FN_counts[i] / sum(FN_counts.values()) * 100.0) for i in FN_counts]
    
    return FP_counts, FN_counts, FP_percent, FN_percent
    
    
        
def run_in_mode(mode, one_fold):
    if one_fold:
        folds = [0]
    else:
        folds = range(num_folds)
        y_pred_combined = np.array([]).reshape([0,FLAGS.SEQUENCE_LEN])
        x_test_combined = np.array([]).reshape([0,FLAGS.SEQUENCE_LEN])
        y_test_combined = np.array([]).reshape([0,FLAGS.SEQUENCE_LEN])
        y_weights_combined = np.array([]).reshape([0,FLAGS.SEQUENCE_LEN])
    for fold_num in folds:
        train_x, train_y, _, train_weights, val_x, val_y, _, val_weights, test_x, test_y, _, test_weights = load.load_annotated_data(fold_num)
        if mode == 'train':
            train(train_x, train_y, train_weights, val_x, val_y, val_weights, fold_num)
        else:
            y_pred = test(test_x, test_y, test_weights, fold_num)
            y_pred_combined = np.concatenate([y_pred_combined,y_pred])
            x_test_combined = np.concatenate([x_test_combined,test_x])
            y_test_combined = np.concatenate([y_test_combined,test_y])
            y_weights_combined = np.concatenate([y_weights_combined,test_weights])
    if mode == 'test':
        Metrics.print_metrics_for_all_folds()
        FP_counts, FN_counts, FP_percent, FN_percent = count_POS(y_pred_combined, x_test_combined, y_test_combined, y_weights_combined)
        print ('POS tags for False Positives: ')
        for key, value in FP_counts.items():
            print(key, value)
        print ('POS tags for False Negatives: ')
        for key, value in FN_counts.items():
            print(key, value)
        print ('POS tags percentages for False Positives: ')
        for key, value in FP_percent:
            print(key, value)
        print ('POS tags percentages for False Negatives: ')
        for key, value in FN_percent:
            print(key, value)
    
def main(unused_argv):
    if args.train_only and args.test_only: raise Exception('provide only one mode')
    train = args.train_only or not args.test_only
    test = not args.train_only or args.test_only
    if train:
        run_in_mode('train', args.one_fold)
    if test:
        run_in_mode('test', args.one_fold)
        

    localtime = time.asctime( time.localtime(time.time()) )
    print ("Finished at: ", localtime     )
    print('Execution time: ', (time.time() - start_time)/3600., ' hours')
    
if __name__ == '__main__':
  tf.app.run()














































