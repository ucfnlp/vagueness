#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import theano
import codecs
import operator
import h5py
import gensim
from gensim.models.word2vec import Word2Vec
import cPickle

numpy.random.seed(123)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

data_folder = '../data/'

train_file = data_folder + 'Privacy_Sentences.txt'
word_id_file = data_folder + 'train.h5'
dict_file = data_folder + 'words.dict'
embedding_file = data_folder + 'GoogleNews-vectors-negative300.bin'
vague_file = data_folder + 'vague_terms'
dataset_file = data_folder + 'dataset.h5'
embedding_weights_file = data_folder + 'embedding_weights.h5'
 
vocab_size = 5000
embedding_dim = 300
maxlen = 50
batch_size = 128
val_samples = batch_size * 10
train_ratio = 0.8

# load file, one sentence per line
sentences = []
end_tag = ['</s>']
with codecs.open(train_file) as infile:
    for line in infile:
        words = line.strip().split() + end_tag
        sentences.append(' '.join(words))
print('total number of sentences in train file: %d' % len(sentences))
    
# tokenize, create vocabulary
tokenizer = Tokenizer(nb_words=vocab_size, filters=' ')
tokenizer.fit_on_texts(sentences)
word_id_seqs = tokenizer.texts_to_sequences(sentences)
print('finished creating the dictionary')
    
# output dictionary
with codecs.open(dict_file, 'w') as outfile:
    for word, idx in sorted(tokenizer.word_index.items(), key=operator.itemgetter(1)):
        outfile.write('%s %d\n' % (word, idx))
    
# output list of word ids
total_word_ids = 0
my_word_ids = []
for word_id_seq in word_id_seqs:
    for word_id in word_id_seq[-maxlen-1:]: #TODO
        total_word_ids += 1
        my_word_ids.append(word_id)
    
outfile = h5py.File(word_id_file, 'w')
states = outfile.create_dataset('words', data=numpy.array(my_word_ids))
outfile.flush()
outfile.close()
    
    
# load file containing vague terms
vague_terms = []
with codecs.open(vague_file) as infile:
    for line in infile:
        words = line.strip().split()
        word_ids = []
        for w in words: 
            word_ids.append(tokenizer.word_index[w] if w in tokenizer.word_index else 0)
        vague_terms.append(word_ids)
    
# prepare data
X_raw = []
Y_raw = []
Y_vague = []
    
# calculate statistics
total_vague_terms = 0
total_terms = 0
total_vague_sents = 0
    
for word_id_seq in word_id_seqs:
    X_raw.append(word_id_seq[:-1])
    Y_raw.append(word_id_seq[1:])
        
    X_curr = word_id_seq[:-1]
    X_curr_len = len(X_curr)
    Y_curr = [0] * X_curr_len
    for idx in xrange(X_curr_len):
        for gap in xrange(1,6):
            if idx + gap > X_curr_len: break
            if X_curr[idx:idx+gap] in vague_terms:
                Y_curr[idx:idx+gap] = [1] * gap
    Y_vague.append(Y_curr)
        
    vague_flag = 0
    for vv in Y_curr:
        if vv == 1: 
            total_vague_terms += 1
            vague_flag = 1
    total_terms += X_curr_len
    if vague_flag == 1: total_vague_sents += 1
        
        
print('total vague terms: %d' % (total_vague_terms))
print('total vague sentences: %d' % (total_vague_sents))
print('total terms: %d' % (total_terms))
           
# prepare embedding weights
word2vec_model = Word2Vec.load_word2vec_format(embedding_file, binary=True)
embedding_weights = numpy.zeros((vocab_size, embedding_dim), dtype=theano.config.floatX)
   
n_words_in_word2vec = 0
n_words_not_in_word2vec = 0
   
for word, idx in tokenizer.word_index.items():
    if idx < vocab_size:
        try: 
            embedding_weights[idx,:] = word2vec_model[word]
            n_words_in_word2vec += 1
        except:
            embedding_weights[idx,:] = 0.01 * numpy.random.randn(1, embedding_dim).astype(theano.config.floatX)
            n_words_not_in_word2vec += 1
print('%d words found in word2vec, %d are not' % (n_words_in_word2vec, n_words_not_in_word2vec))
outfile = h5py.File(embedding_weights_file, 'w')
outfile.create_dataset('embedding_weights', data=embedding_weights)
outfile.flush()
outfile.close()
        
X_padded = pad_sequences(X_raw, maxlen=maxlen, padding='post')
Y_padded = pad_sequences(Y_raw, maxlen=maxlen, padding='post')
Y_padded_vague = pad_sequences(Y_vague, maxlen=maxlen, padding='post')
Y_padded_vague = Y_padded_vague.reshape(Y_padded_vague.shape[0], Y_padded_vague.shape[1], 1)

# split train and test
permutation = numpy.random.permutation(X_padded.shape[0])
X_padded = X_padded[permutation]
Y_padded = Y_padded[permutation]
Y_padded_vague = Y_padded_vague[permutation]
train_len = int(len(X_padded) * train_ratio)
train_X_padded = X_padded[:train_len]
train_Y_padded = Y_padded[:train_len]
train_Y_padded_vague = Y_padded_vague[:train_len]
test_X_padded = X_padded[train_len:]
test_Y_padded = Y_padded[train_len:]
test_Y_padded_vague = Y_padded_vague[train_len:]
    
# # truncate because of keras's predict generator bug
# len_test = test_X_padded.shape[0] - (test_X_padded.shape[0] % val_samples)
len_test = test_X_padded.shape[0]
test_X_padded = test_X_padded[:len_test]
test_Y_padded = test_Y_padded[:len_test]
test_Y_padded_vague = test_Y_padded_vague[:len_test]

outfile = h5py.File(dataset_file, 'w')
outfile.create_dataset('train_X', data=train_X_padded)
outfile.create_dataset('train_Y', data=train_Y_padded)
outfile.create_dataset('train_Y_vague', data=train_Y_padded_vague)
outfile.create_dataset('test_X', data=test_X_padded)
outfile.create_dataset('test_Y', data=test_Y_padded)
outfile.create_dataset('test_Y_vague', data=test_Y_padded_vague)
outfile.flush()
outfile.close()

print('done')




































