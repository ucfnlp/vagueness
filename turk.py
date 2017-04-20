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

train_file = 'Privacy_Sentences.txt'
word_id_file = 'train.h5'
dict_file = 'words.dict'
vague_file = 'vague_terms'
vague_lack_definition_file = 'vague_terms_lack_definition'
vague_sents_csv = 'vague_sentences_groups_of_5.csv'

 
vocab_size = 5000
total_documents = 1010
num_desired_documents = 100
maxlen = 50

# load file, one sentence per line
sentences = []
which_document = []
document_indices = numpy.random.permutation(list(xrange(total_documents)))[:num_desired_documents]
doc_ctr = 0
end_tag = ['</s>']
with codecs.open(train_file) as infile:
    for line in infile:
        if line.strip() == '':
            doc_ctr += 1
            continue
        words = line.strip().split()
        sentences.append(' '.join(words))
        which_document.append(doc_ctr)
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
# load file containing vague terms that lack definition
with codecs.open(vague_lack_definition_file) as infile:
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

vague_sents = []
    
for i in range(len(word_id_seqs)):
    word_id_seq = word_id_seqs[i]
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
    if vague_flag == 1: 
        total_vague_sents += 1
        if which_document[i] in document_indices:
            vague_sents.append(sentences[i])
        
with open(vague_sents_csv, 'w') as f:
    f.write('sentence1,sentence2,sentence3,sentence4,sentence5\n')
    count = 0;
    for sent in vague_sents:
        f.write('"' + sent + '"')
        count += 1
        if count % 5 == 0:
            f.write('\n')
        else:
            f.write(',')
        
print('total vague terms: %d' % (total_vague_terms))
print('total vague sentences: %d' % (total_vague_sents))
print('total terms: %d' % (total_terms))
           
        
print('done')





































