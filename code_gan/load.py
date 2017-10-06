from __future__ import print_function

import h5py
import numpy as np
from scipy.ndimage.interpolation import shift
import param_names
import os

annotated_dataset_file = os.path.join('..','data','annotated_dataset.h5')
unannotated_dataset_file = os.path.join('..','data','dataset.h5')
generated_dataset_file = os.path.join('..','data','generated_dataset.h5')
embedding_weights_file = os.path.join('..','data','embedding_weights.h5')
dictionary_file = os.path.join('..','data','words.dict')
train_variables_file = os.path.join('..','models','tf_lm_variables.npz')
vague_terms_file = os.path.join('..','data','vague_terms')
    
def load_annotated_data(fold_num=0):
    print('loading training and test data')
    with h5py.File(annotated_dataset_file, 'r') as data_file:
        fold = data_file['fold'+str(fold_num)]
        train_x = fold['train_X'][:]
        train_y = fold['train_Y_sentence'][:]
        val_x = fold['val_X'][:]
        val_y = fold['val_Y_sentence'][:]
        test_x = fold['test_X'][:]
        test_y = fold['test_Y_sentence'][:]
    print ('Number of training instances: ' + str(train_y.shape[0]))
    # Remove </s> symbols
    train_x[train_x == 3] = 0
    val_x[val_x == 3] = 0
    test_x[test_x == 3] = 0
    # Shift over to remove <s> symbols
    train_x = shift(train_x, [0,-1], cval=0)
    val_x = shift(val_x, [0,-1], cval=0)
    test_x = shift(test_x, [0,-1], cval=0)
            
#     print train_x
#     for i in range(min(5, len(train_x))):
#         for j in range(len(train_x[i])):
#             if train_x[i][j] == 0:
#                 continue
#             word = d[train_x[i][j]]
#             print word + ' ',
#         print '(' + str(train_y[i]) + ')\n'
    return train_x, train_y, val_x, val_y, test_x, test_y

def load_unannotated_dataset():
    print('loading training and test data')
    with h5py.File(unannotated_dataset_file, 'r') as data_file:
        train_X = data_file['train_X'][:]
        train_Y = data_file['train_Y'][:]
        test_X = data_file['test_X'][:]
        test_Y = data_file['test_Y'][:]
    return train_X, train_Y, test_X, test_Y

def load_embedding_weights():
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

def load_pretrained_params():
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



















