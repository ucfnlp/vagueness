#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import math
import yaml
import numpy
import codecs
import operator
import h5py
import gensim
from gensim.models.word2vec import Word2Vec
from numpy import nan_to_num
from odo.backends.pandas import categorical

numpy.random.seed(123)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#if using theano, then comment the next line and uncomment the following 2 lines
float_type = numpy.float32        
# import theano
# float_type = theano.config.floatX

train_file = '../data/Privacy_Sentences.txt'
word_id_file = '../data/train.h5'
dict_file = '../data/words.dict'
embedding_file = '../data/GoogleNews-vectors-negative300.bin'
vague_file = '../data/vague_terms'
dataset_file = '../data/annotated_dataset.h5'
embedding_weights_file = '../data/annotated_embedding_weights.h5'
clean_data_json = '../data/clean_data.json'
vague_phrases_file = '../data/vague_phrases.txt'

 
vocab_size = 5000
embedding_dim = 300
maxlen = 50
batch_size = 128
val_samples = batch_size * 10
train_ratio = 0.8
vague_phrase_threshold = 2
min_vague_score = 1
max_vague_score = 5

'''
Parameters
----------------
sentence: string
vague_phrases: dict, representing the counts of each vague phrase in the sentence

Output
----------------
labels: list of integers (0 or 1), one element for each word in the sentence
    0 = not a vague word
    1 = vague word
'''
def labelVagueWords(sentence, vague_phrases):
    labels = [0] * len(sentence)
    for phrase, count in vague_phrases.iteritems():
        if count >= vague_phrase_threshold:
            phrase = phrase.strip().split()
            word_idx = 0
            for i in range(len(sentence)):
                if i + len(phrase) >= len(sentence): break
                if sentence[i:i+len(phrase)] ==  phrase:
                    labels[i:i+len(phrase)] = [1] * len(phrase)
    if len(labels) != len(sentence):
        raise ValueError('len labels does not equal len sentence')
    return labels

# read in existing dictionary created by preprocess_unannotated.py
print('loading dictionary')
d = {}
with open(dict_file) as f:
    for line in f:
       (val, key) = line.split()
       d[val] = int(key)

# Reads in the JSON data
with open(clean_data_json) as f:
    json_str = f.read()
data = yaml.safe_load(json_str)

# calculate statistics
total_vague_terms = 0
total_terms = 0
total_vague_sents = 0
stds = []

# load file, one sentence per line
sentences = []
Y_sentence = []
Y_word = []
sentence_doc_ids = []
vague_phrases = {}
start_tag = ['<s>']
end_tag = ['</s>']
for doc in data['docs']:
    for sent in doc['vague_sentences']:
        words = sent['sentence_str'].strip().split()
        if len(words) == 0:
            continue
        words = start_tag + words + end_tag
        sentences.append(' '.join(words))
        
        # Get the sentence-level scores
        scores = map(int, sent['scores'])
        Y_sentence.append(numpy.nan_to_num(numpy.average(scores)))
        
        # Get word-level vagueness
        word_labels = labelVagueWords(words, sent['vague_phrases'])
        Y_word.append(word_labels)
        
        # Store the document ID
        sentence_doc_ids.append(int(doc['id']))
        
        # Calculate statistics
        total_terms += len(word_labels)
        num_vague_terms = sum(x == 1 for x in word_labels)
        total_vague_terms += num_vague_terms
        if num_vague_terms > 0: 
            total_vague_sents += 1
        std = numpy.std(scores)
        stds.append(std)
        for phrase, count in sent['vague_phrases'].iteritems():
            if count >= vague_phrase_threshold:
                vague_phrases[phrase] = vague_phrases.get(phrase, 0) + 1
                
# Print statistics
print('total vague sentences: %d' % (total_vague_sents))
print('total number of sentences in train file: %d' % len(sentences))
print('total vague terms: %d' % (total_vague_terms))
print('total terms: %d' % (total_terms))
print('average standard deviation of scores for each sentence: %f' % (numpy.average(stds)))


# plt.hist(Y_sentence)
# plt.title("Sentence-Level Vagueness Score Distribution")
# plt.xlabel("Score")
# plt.ylabel("Number of Sentences")
# plt.show()

# convert from float to category (possible categories: {0,1,2,3})
for idx, item in enumerate(Y_sentence):
    res = math.floor(item)
    if res == max_vague_score:
        res = max_vague_score-1
    res -= 1
    Y_sentence[idx] = res
plt.hist(Y_sentence, bins=max_vague_score-min_vague_score, range=(min_vague_score-1, max_vague_score-1))
plt.title("Sentence-Level Vagueness Score Distribution")
plt.xlabel("Score")
plt.ylabel("Number of Sentences")
plt.show(block=False)


sorted_vague_phrases = sorted(vague_phrases.items(), key=operator.itemgetter(1), reverse=True)
with open(vague_phrases_file, 'w') as f:
    for phrase, count in sorted_vague_phrases:
        if phrase != '':
            f.write(phrase + ': ' + str(count) + '\n')
            
word_id_seqs = []
for sent in sentences:
    words = sent.lower().split()
    word_id_seq = []
    for word in words:
        if (not d.has_key(word)) or (d[word] >= vocab_size):
            word_id_seq.append(0)
        else:
            word_id_seq.append(d[word])
    word_id_seqs.append(word_id_seq)
        
        
    
# # tokenize, create vocabulary
# tokenizer = Tokenizer(nb_words=vocab_size, filters=' ')
# tokenizer.fit_on_texts(sentences)
# word_id_seqs = tokenizer.texts_to_sequences(sentences)
# print('finished creating the dictionary')
    
# # output dictionary
# with codecs.open(dict_file, 'w') as outfile:
#     for word, idx in sorted(tokenizer.word_index.items(), key=operator.itemgetter(1)):
#         outfile.write('%s %d\n' % (word, idx))
#     
# # output list of word ids
# total_word_ids = 0
# my_word_ids = []
# for word_id_seq in word_id_seqs:
#     for word_id in word_id_seq[-maxlen-1:]: #TODO
#         total_word_ids += 1
#         my_word_ids.append(word_id)
#     
# outfile = h5py.File(word_id_file, 'w')
# states = outfile.create_dataset('words', data=numpy.array(my_word_ids))
# outfile.flush()
# outfile.close()

# # prepare embedding weights
# word2vec_model = Word2Vec.load_word2vec_format(embedding_file, binary=True)
# embedding_weights = numpy.zeros((vocab_size, embedding_dim), dtype=float_type)
#    
# n_words_in_word2vec = 0
# n_words_not_in_word2vec = 0
#    
# for word, idx in tokenizer.word_index.items():
#     if idx < vocab_size:
#         try: 
#             embedding_weights[idx,:] = word2vec_model[word]
#             n_words_in_word2vec += 1
#         except:
#             embedding_weights[idx,:] = 0.01 * numpy.random.randn(1, embedding_dim).astype(float_type)
#             n_words_not_in_word2vec += 1
# print('%d words found in word2vec, %d are not' % (n_words_in_word2vec, n_words_not_in_word2vec))
# outfile = h5py.File(embedding_weights_file, 'w')
# outfile.create_dataset('embedding_weights', data=embedding_weights)
# outfile.flush()
# outfile.close()

# Pad X and Y
X = word_id_seqs
X_padded = pad_sequences(X, maxlen=maxlen, padding='post')
Y_padded_word = pad_sequences(Y_word, maxlen=maxlen, padding='post')
Y_sentence = numpy.asarray(Y_sentence, dtype=numpy.int32)
Y_padded_word = Y_padded_word.reshape(Y_padded_word.shape[0], Y_padded_word.shape[1], 1)

# shuffle Documents
doc_ids = set()
for doc in data['docs']:
    doc_ids.add(int(doc['id']))
doc_ids = list(doc_ids)
numpy.random.shuffle(doc_ids)
train_len = int(train_ratio*len(doc_ids))
train_doc_ids = doc_ids[:train_len]
test_doc_ids = doc_ids[train_len:]

# Split into train and test, keeping documents together
train_indices = []
test_indices = []
for i in range(len(sentence_doc_ids)):
    doc = sentence_doc_ids[i]
    if doc in train_doc_ids:
        train_indices.append(i)
    elif doc in test_doc_ids:
        test_indices.append(i)
    else:
        raise ValueError('Document id was not in either the train set nor the test set')
train_X = X_padded[train_indices]
train_Y_word = Y_padded_word[train_indices]
train_Y_sentence = Y_sentence[train_indices]
test_X = X_padded[test_indices]
        
test_Y_word = Y_padded_word[test_indices]
test_Y_sentence = Y_sentence[test_indices]

#shuffle
        
permutation = numpy.random.permutation(train_X.shape[0])
train_X = train_X[permutation]
train_Y_word = train_Y_word[permutation]
train_Y_sentence = train_Y_sentence[permutation]
permutation = numpy.random.permutation(test_X.shape[0])
test_X = test_X[permutation]
test_Y_word = test_Y_word[permutation]
test_Y_sentence = test_Y_sentence[permutation]

# Save preprocessed dataset to file
outfile = h5py.File(dataset_file, 'w')
outfile.create_dataset('train_X', data=train_X)
outfile.create_dataset('train_Y_word', data=train_Y_word)
outfile.create_dataset('train_Y_sentence', data=train_Y_sentence)
outfile.create_dataset('test_X', data=test_X)
outfile.create_dataset('test_Y_word', data=test_Y_word)
outfile.create_dataset('test_Y_sentence', data=test_Y_sentence)
outfile.flush()
outfile.close()

print('done')

























