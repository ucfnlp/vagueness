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

prediction_words_file = 'predictions_logit.html'
prediction_folder = '../predictions'
summary_file = '/home/logan/tmp'
ckpt_dir = '../models/word_level_logit'
variables_file = ckpt_dir + '/variables.npz'
vague_phrases_file = '../data/vague_phrases.txt'
val_ratio = 0.2
fast = False
num_folds = 5
start_time = time.time()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('EPOCHS', 20,
                            'Num epochs.')
tf.app.flags.DEFINE_integer('VOCAB_SIZE', 10000,
                            'Number of words in the vocabulary.')
tf.app.flags.DEFINE_integer('SEQUENCE_LEN', 50,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('EMBEDDING_SIZE', 300,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('PATIENCE', 1,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('BATCH_SIZE', 64,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_string('CELL_TYPE', 'LSTM',
                            'Which RNN cell for the RNNs.')
tf.app.flags.DEFINE_integer('RANDOM_SEED', 123,
                            'Random seed used for numpy and tensorflow (dropout, sampling)')
tf.set_random_seed(FLAGS.RANDOM_SEED)
np.random.seed(FLAGS.RANDOM_SEED)

# load data for fold
# get all unique unigrams, bigrams, and trigrams in (train + val)
# pick out words that are vague >= 2 (in vague phrases list)
# pick out equal number of words that are not vague
# make train, val only based on words
# train logit
# run model on ALL words to get vague labels for each unqiue word
# assign labels to test set sentences
# metrics

def get_unique_unigrams(x):
    return np.unique(x)

# def get_vague_word_ids(word_to_id):
#     all_ids = []
#     with open(vague_phrases_file) as f:
#         for line in f:
#             phrase, _ = line.split(':')
#             words = phrase.split()
#             in_dictionary = True
#             ids = [word_to_id.get(word) for word in words]
#             if None in ids:
#                print('Phrase: ', words, 'excluded because a word was not found in dictionary' )
#             else:
#                 all_ids.append(ids)
#     return all_ids

def get_vague_word_ids(x, y):
    condition = y==1
    vague_word_ids = np.extract(condition, x)
    vague_word_ids = np.unique(vague_word_ids)
    return vague_word_ids

def get_clear_word_ids(x, y):
    condition = y==0
    clear_word_ids = np.extract(condition, x)
    clear_word_ids = np.unique(clear_word_ids)
    return clear_word_ids

def filter_vague(unique_x, vague_word_ids):
    return np.array([x for x in unique_x if x in vague_word_ids])

def filter_not_in_list(my_array, to_exclude):
    mask = np.in1d(my_array, to_exclude)
    return my_array[~mask]

def choose_n_clear_words(x, vague_word_ids, n):
    unique_x = get_unique_unigrams(x)
    clear_words = np.array([x for x in unique_x if not x in vague_word_ids])
    np.random.shuffle(clear_words)
#     return clear_words[:n]
    return clear_words

def create_x_y(vague_words, clear_words):
    x = np.concatenate([vague_words, clear_words])
    y = np.concatenate([np.ones_like(vague_words), np.zeros_like(clear_words)])
    return x, y
    
def shuffle(*args):
    if len(args) == 0:
        raise Exception('No lists to shuffle')
    permutation = np.random.permutation(len(args[0]))
    return [arg[permutation] for arg in args]

def split_train_val(val_ratio, *args):
    val_len = int(len(args[0]) * val_ratio)
    val_args = [arg[:val_len] for arg in args]
    train_args = [arg[val_len:] for arg in args]
    return train_args + val_args

def create_dataset(train_x, train_y, val_x, val_y, test_x, test_y):
    x = np.concatenate([train_x, val_x, test_x])
    y = np.concatenate([train_y, val_y, test_y])
    train_val_x = np.concatenate([train_x, val_x])
    train_val_y = np.concatenate([train_y, val_y])
    all_words = get_unique_unigrams(x)
    all_vague_words = get_vague_word_ids(x, y)
#     all_vague_words = get_vague_word_ids(word_to_id)
#     with open('../data/vague_WORDS.txt', 'w') as f:
#         one_word_vague_words = [word[0] for word in all_vague_words if len(word) == 1]
#         for w in one_word_vague_words:
#             if w == 0: continue
#             f.write(d[w] + '\n')
    all_clear_words = filter_not_in_list(all_words, all_vague_words)
    train_val_unique_words = get_unique_unigrams(train_val_x)
    train_val_vague = filter_vague(train_val_unique_words, all_vague_words)
    train_val_clear = choose_n_clear_words(train_val_unique_words, all_vague_words, len(train_val_vague))
    new_x, new_y = create_x_y(train_val_vague, train_val_clear)
    new_x, new_y = shuffle(new_x, new_y)
    train_x, train_y, val_x, val_y = split_train_val(val_ratio, new_x, new_y)
    
    test_vague = filter_not_in_list(all_vague_words, train_val_vague)
    test_clear = filter_not_in_list(all_clear_words, train_val_clear)
    test_x, test_y = create_x_y(test_vague, test_clear)
    test_x, test_y = shuffle(test_x, test_y)
    return train_x, train_y, val_x, val_y, test_x, test_y

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

Metrics = utils.Metrics(is_binary=True)
        
'''
--------------------------------

MODEL

--------------------------------
'''
        
print('building model')
inputs = tf.placeholder(tf.int32, shape=(None,), name='inputs')
targets = tf.placeholder(tf.float32, shape=(None,), name='targets')
embedding_tensor = tf.Variable(initial_value=embedding_weights, name='embedding_matrix')
embeddings = tf.nn.embedding_lookup(embedding_tensor, inputs)
# hidden = tf.layers.dense(embeddings, 100)
# logits = tf.layers.dense(hidden, 1)
logits = tf.layers.dense(embeddings, 1)
logits = tf.squeeze(logits)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
cost = tf.reduce_sum(loss)
tvars = tf.trainable_variables()
tvar_names = [var.name for var in tvars]
# TODO: change to rms optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=tvars)
predictions = tf.round(tf.sigmoid(logits))
correct_predictions = tf.equal(predictions, targets)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))


global_step = tf.Variable(-1, name='global_step', trainable=False)
saver = tf.train.Saver()

utils.variable_summaries(tvars) 
merged = tf.summary.merge_all()


def batch_generator(x, y, batch_size):
    data_len = x.shape[0]
    for i in range(0, data_len, batch_size):
        batch_x = x[i:min(i+batch_size,data_len)]
        batch_y = y[i:min(i+batch_size,data_len)]
        yield batch_x, batch_y
        
def save_predictions_to_file(x, y, weights, predict, fold_num):
    file_name = os.path.join(prediction_folder, str(fold_num), prediction_words_file)
    out = ''
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
        

def validate(sess, val_x, val_y):
    def run_val_step(sess, batch_x, batch_y):
        to_return = cost
        return sess.run(to_return,
                    feed_dict={inputs: batch_x,
                               targets: batch_y})
    sum_cost = 0
    for batch_x, batch_y, cur, data_len in utils.batch_generator(val_x, val_y):
        batch_cost = run_val_step(sess, batch_x, batch_y)
        sum_cost += batch_cost * len(batch_y)  # Add up cost based on how many instances in batch
    val_cost = sum_cost / len(val_y)
    return val_cost
    
    
def train(train_x, train_y, val_x, val_y, fold_num):
    
    def run_train_step(sess, batch_x, batch_y):
        to_return = [optimizer, cost, accuracy, merged]
        return sess.run(to_return,
                    feed_dict={inputs: batch_x,
                               targets: batch_y})
        
    
    print 'training'
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
        saver = tf.train.Saver(max_to_keep=5)
        tf.global_variables_initializer().run()
        min_val_cost = np.inf
        num_mistakes = 0
        
        fold_ckpt_dir = ckpt_dir + '/' + str(fold_num)
        variables_file = fold_ckpt_dir + '/tf_acgan_variables_'
        if args.resume:
            ckpt = tf.train.get_checkpoint_state(fold_ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print ckpt.model_checkpoint_path
                model.saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables
    
        start = global_step.eval() + 1  # get last global_step and start the next one
        print "Start from:", start
        
        step = 0
        for cur_epoch in tqdm(range(start, FLAGS.EPOCHS), desc='Fold ' + str(fold_num) + ': Epoch'):
            for batch_x, batch_y, _, _ in tqdm(utils.batch_generator(train_x, train_y), desc='Batch', total=len(train_y)/FLAGS.BATCH_SIZE):
                _, batch_cost, batch_accuracy, summary = run_train_step(sess, batch_x, batch_y)
                train_writer.add_summary(summary, step)
                step += 1
#                 tqdm.write('Fold', fold_num, 'Epoch: ', cur_epoch,)
                tqdm.write('Loss: {:.4} \tAcc: {:.4}'. format(batch_cost, batch_accuracy))
            
            tqdm.write('saving model to file:')
            global_step.assign(cur_epoch).eval()  # set and update(eval) global_step with index, cur_epoch
            saver.save(sess, fold_ckpt_dir + "/model.ckpt", global_step=cur_epoch)
            vars = sess.run(tvars)
            tvar_names = [var.name for var in tf.trainable_variables()]
            variables = dict(zip(tvar_names, vars))
            np.savez(variables_file, **variables)
            
            val_cost = validate(sess, val_x, val_y)
            tqdm.write('Val Loss: ' + str(val_cost))
            if val_cost < min_val_cost:
                min_val_cost = val_cost
            else:
                num_mistakes += 1
            if num_mistakes >= FLAGS.PATIENCE:
                tqdm.write('Stopping early at epoch: ' + str(cur_epoch))
                break
            
        train_writer.close()
            
        
    
def test(test_x, test_y, fold_num):
    
    def run_test_step(sess, batch_x):
        to_return = predictions
        return sess.run(to_return,
                        feed_dict={inputs: batch_x})
    def predict(sess, x, y):
        total_predictions = []
        for batch_x, batch_y, _, _ in tqdm(utils.batch_generator(x, y, batch_size=1), desc='Fold ' + str(fold_num) + ':Testing', total=len(y)):
            batch_predictions = run_test_step(sess, batch_x)
            total_predictions.append(batch_predictions)
        total_predictions = np.array(total_predictions)
        f1_score = metrics.f1_score(y, total_predictions, average='macro')
        return total_predictions, f1_score
    
    print 'testing'
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)
        tf.global_variables_initializer().run()
        fold_ckpt_dir = ckpt_dir + '/' + str(fold_num)
        ckpt = tf.train.get_checkpoint_state(fold_ckpt_dir)
        if not ckpt:
            raise Exception('Could not find saved model in: ' + fold_ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables
        total_predictions, _ = predict(sess, test_x, test_y)
        Metrics.print_and_save_metrics(test_y, total_predictions)
        
        

def run_on_fold(mode, fold_num):
    train_x, train_y, _, _, val_x, val_y, _, _, test_x, test_y, _, _ = load.load_annotated_data(fold_num)
    train_x, train_y, val_x, val_y, test_x, test_y = create_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
    if mode == 'train':
        train(train_x, train_y, val_x, val_y, fold_num)
    else:
        test(test_x, test_y, fold_num)
        
def run_in_mode(mode, one_fold):
    if one_fold:
        run_on_fold(mode, 0)
    else:
        for fold_num in range(num_folds):
            run_on_fold(mode, fold_num)
    if mode == 'test':
        Metrics.print_metrics_for_all_folds()
    
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














































