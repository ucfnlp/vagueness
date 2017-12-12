from __future__ import print_function

import h5py
import numpy as np
from scipy.ndimage.interpolation import shift
import param_names
import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

dictionary_file = os.path.join('..','data','words.dict')
# train_variables_file = os.path.join('..','models', 'lm_ckpts', 'tf_lm_variables.npz')
# train_variables_file = os.path.join('..','models', 'lm_ckpts_l2', 'tf_lm_variables.npz')
vague_terms_file = os.path.join('..','data','vague_terms')
    
def load_annotated_data(fold_num=0):
    annotated_dataset_file = os.path.join('..','data','annotated_dataset_' + str(FLAGS.VOCAB_SIZE) + '.h5')
    print('loading training and test data')
    with h5py.File(annotated_dataset_file, 'r') as data_file:
        fold = data_file['fold'+str(fold_num)]
        train_x = fold['train_X'][:]
        train_y_word = fold['train_Y_word'][:]
        train_y_sentence = fold['train_Y_sentence'][:]
        train_weights = fold['train_weights'][:]
        val_x = fold['val_X'][:]
        val_y_word = fold['val_Y_word'][:]
        val_y_sentence = fold['val_Y_sentence'][:]
        val_weights = fold['val_weights'][:]
        test_x = fold['test_X'][:]
        test_y_word = fold['test_Y_word'][:]
        test_y_sentence = fold['test_Y_sentence'][:]
        test_weights = fold['test_weights'][:]
    print ('Number of training instances: ' + str(train_y_sentence.shape[0]))
#     # Remove </s> symbols
#     train_y_word[train_x == 3] = 0
#     val_y_word[val_x == 3] = 0
#     test_y_word[test_x == 3] = 0
#     train_weights[train_x == 3] = 0
#     val_weights[val_x == 3] = 0
#     test_weights[test_x == 3] = 0
#     train_x[train_x == 3] = 0
#     val_x[val_x == 3] = 0
#     test_x[test_x == 3] = 0
    # Shift over to remove <s> symbols
    train_x = shift(train_x, [0,-1], cval=0)
    val_x = shift(val_x, [0,-1], cval=0)
    test_x = shift(test_x, [0,-1], cval=0)
    train_y_word = shift(train_y_word, [0,-1], cval=0)
    val_y_word = shift(val_y_word, [0,-1], cval=0)
    test_y_word = shift(test_y_word, [0,-1], cval=0)
    train_weights = shift(train_weights, [0,-1], cval=0)
    val_weights = shift(val_weights, [0,-1], cval=0)
    test_weights = shift(test_weights, [0,-1], cval=0)
    
            
#     print train_x
#     for i in range(min(5, len(train_x))):
#         for j in range(len(train_x[i])):
#             if train_x[i][j] == 0:
#                 continue
#             word = d[train_x[i][j]]
#             print word + ' ',
#         print '(' + str(train_y[i]) + ')\n'
    return train_x, train_y_word, train_y_sentence, train_weights, val_x, val_y_word, val_y_sentence, val_weights, test_x, test_y_word, test_y_sentence, test_weights

def load_unannotated_dataset():
    unannotated_dataset_file = os.path.join('..','data','dataset_' + str(FLAGS.VOCAB_SIZE) + '.h5')
    print('loading training and test data')
    with h5py.File(unannotated_dataset_file, 'r') as data_file:
        train_X = data_file['train_X'][:]
        train_Y = data_file['train_Y'][:]
        train_weights = data_file['train_weights'][:]
        test_X = data_file['test_X'][:]
        test_Y = data_file['test_Y'][:]
        test_weights = data_file['test_weights'][:]
    return train_X, train_Y, train_weights, test_X, test_Y, test_weights

def load_generated_data():
    annotated_dataset_file = os.path.join('..','data','annotated_dataset_' + str(FLAGS.VOCAB_SIZE) + '.h5')
    generated_dataset_file = os.path.join('..','data','generated_dataset_' + str(FLAGS.VOCAB_SIZE) + '.h5')
    print('loading training and test data')
    with h5py.File(generated_dataset_file, 'r') as data_file:
        train_X = data_file['train_X'][:]
        train_Y = data_file['train_Y'][:]
        val_X = data_file['val_X'][:]
        val_Y = data_file['val_Y'][:]
    with h5py.File(annotated_dataset_file, 'r') as data_file:
        test_X = data_file['X'][:]
        test_Y = data_file['Y'][:]
        # Remove </s> symbols
        test_X[test_X == 3] = 0
        # Shift over to remove <s> symbols
        test_X = shift(test_X, [0,-1], cval=0)
#     print (train_X)
#     print (train_Y)
#     for i in range(min(5, len(train_X))):
#         for j in range(len(train_X[i])):
#             if train_X[i][j] == 0:
#                 continue
#             word = d[train_X[i][j]]
#             print (word + ' ',)
#         print ('(' + str(train_Y[i]) + ')\n')
#     print (test_X)
#     print (test_Y)
#     for i in range(min(5, len(test_X))):
#         for j in range(len(test_X[i])):
#             if test_X[i][j] == 0:
#                 continue
#             word = d[test_X[i][j]]
#             print (word + ' ',)
#         print ('(' + str(test_Y[i]) + ')\n')
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def load_embedding_weights():
    embedding_weights_file = os.path.join('..','data','embedding_weights_' + str(FLAGS.VOCAB_SIZE) + '.h5')
    print('loading embedding weights')
    with h5py.File(embedding_weights_file, 'r') as hf:
        embedding_weights = hf['embedding_weights'][:]
    return embedding_weights

def load_dictionary():
    print('loading dictionary')
    d = {}
    word_to_id = {}
    with open(dictionary_file) as f:
        for line in f:
           (word, id) = line.split()
           d[int(id)] = word
           word_to_id[word] = int(id)
    return d, word_to_id

def load_pretrained_params(lm_name):
    train_variables_file = os.path.join('..','models', lm_name, 'tf_lm_variables.npz')
    print('loading model parameters')
    params = np.load(train_variables_file)
    params_dict = {}
    for key in params.keys():
        params_dict[key] = params[key]
    params.close()
    params = params_dict
    # Make padding symbol's embedding = 0
    pretrained_embedding_matrix = params[param_names.GAN_PARAMS.EMBEDDING[0]]
    pretrained_embedding_matrix[0] = np.zeros(pretrained_embedding_matrix[0].shape)
    params[param_names.GAN_PARAMS.EMBEDDING[0]] = pretrained_embedding_matrix
    return params

def load_vague_terms_vector(word_to_id, vocab_size):
    print('loading vague terms vector')
    vague_terms = np.zeros((vocab_size))
    with open(vague_terms_file) as f:
        for line in f:
            words = line.split()
            if not len(words) == 1:
                print('excluded', words, 'because it is not 1 word:')
                continue
            word = words[0]
            if not word in word_to_id:
                print(word, 'is not in dictionary')
                continue
            id = word_to_id[word]
            if id >= vague_terms.shape[0]:
                print(word, 'is out of vocabulary')
                continue
            vague_terms[id] = 1
    return vague_terms



















