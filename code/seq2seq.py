#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json
import yaml
import numpy
import theano
import codecs
import operator
import h5py
import gensim
from gensim.models.word2vec import Word2Vec
from numpy import nan_to_num
import operator

numpy.random.seed(123)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from metrics import performance

train_file = '../data/Privacy_Sentences.txt'
word_id_file = '../data/train.h5'
dict_file = '../data/words.dict'
embedding_file = '../data/GoogleNews-vectors-negative300.bin'
vague_file = '../data/vague_terms'
dataset_file = '../data/dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
clean_data_json = '../data/clean_data.json'
vague_phrases_file = '../data/vague_phrases.txt'
Y_words_file = '../predictions/seq2seq_Y_words.txt'
predict_words_file = '../predictions/seq2seq_predictions_words.txt'
dictionary_file = '../data/words.dict'

 
vocab_size = 1000
embedding_dim = 300
maxlen = 50
batch_size = 128
val_samples = batch_size * 10
train_ratio = 0.8
vague_phrase_threshold = 2

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
end_tag = ['</s>']
for doc in data['docs']:
    for sent in doc['vague_sentences']:
        words = sent['sentence_str'].strip().split() + end_tag
        sentences.append(' '.join(words))
        scores = map(int, sent['scores'])
        Y_sentence.append(numpy.nan_to_num(numpy.average(scores)))
        word_labels = labelVagueWords(words, sent['vague_phrases'])
        Y_word.append(word_labels)
        sentence_doc_ids.append(int(doc['id']))
        
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
print('total vague sentences: %d' % (total_vague_sents))
print('total number of sentences in train file: %d' % len(sentences))
print('total vague terms: %d' % (total_vague_terms))
print('total terms: %d' % (total_terms))
print('average standard deviation of scores for each sentence: %f' % (numpy.average(stds)))

plt.hist(Y_sentence)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
# plt.show()

sorted_vague_phrases = sorted(vague_phrases.items(), key=operator.itemgetter(1), reverse=True)
with open(vague_phrases_file, 'w') as f:
    for phrase, count in sorted_vague_phrases:
        if phrase != '':
            f.write(phrase + ': ' + str(count) + '\n')
        
    
# tokenize, create vocabulary
tokenizer = Tokenizer(nb_words=vocab_size, filters=' ')
tokenizer.fit_on_texts(sentences)
word_id_seqs = tokenizer.texts_to_sequences(sentences)
print('finished creating the dictionary')
    
# output dictionary
# with codecs.open(dict_file, 'w') as outfile:
#     for word, idx in sorted(tokenizer.word_index.items(), key=operator.itemgetter(1)):
#         outfile.write('%s %d\n' % (word, idx))
    
# output list of word ids
total_word_ids = 0
my_word_ids = []
for word_id_seq in word_id_seqs:
    for word_id in word_id_seq[-maxlen-1:]: #TODO
        total_word_ids += 1
        my_word_ids.append(word_id)
    
# outfile = h5py.File(word_id_file, 'w')
# states = outfile.create_dataset('words', data=numpy.array(my_word_ids))
# outfile.flush()
# outfile.close()

X = word_id_seqs

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
# outfile = h5py.File(embedding_weights_file, 'w')
# outfile.create_dataset('embedding_weights', data=embedding_weights)
# outfile.flush()
# outfile.close()

        
           





# 
# phrase_word_ids = []
# for sent in vague_phrases:
#     phrases = []
#     for phrase, count in sent:
#         if count >= vague_phrase_threshold:
#             words = phrase.strip().split()
#             word_ids = []
#             for w in words: 
#                 word_ids.append(tokenizer.word_index[w] if w in tokenizer.word_index else 0)
             


X_padded = pad_sequences(X, maxlen=maxlen)
Y_padded_word = pad_sequences(Y_word, maxlen=maxlen)
Y_sentence = numpy.asarray(Y_sentence)
Y_padded_word = Y_padded_word.reshape(Y_padded_word.shape[0], Y_padded_word.shape[1], 1)

doc_ids = set()
for doc in data['docs']:
    doc_ids.add(int(doc['id']))
doc_ids = list(doc_ids)
numpy.random.shuffle(doc_ids)
train_len = int(train_ratio*len(doc_ids))
train_doc_ids = doc_ids[:train_len]
test_doc_ids = doc_ids[train_len:]

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






'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
'''


import numpy
import theano
import h5py

numpy.random.seed(123)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, merge, Input, TimeDistributed, Bidirectional
from keras.layers import Embedding, LSTM, GRU
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras import metrics
from keras.callbacks import EarlyStopping

from batch_generator import batch_generator
from objectives import RankNet_mean


from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range


dataset_file = '../data/dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'


fast = False
 
vocab_size = 1000
embedding_dim = 300
maxlen = 50
hidden_dim = 512
batch_size = 128
nb_epoch = 200
samples_per_epoch = None
 
if fast:
    nb_epoch = 1
    samples_per_epoch = 100
    
# print('loading embedding weights')
# with h5py.File(embedding_weights_file, 'r') as hf:
#     embedding_weights = hf['embedding_weights'][:]
#     
# print('loading training and test data')
# with h5py.File(dataset_file, 'r') as data_file:
#     train_X = data_file['train_X'][:]
#     train_Y = data_file['train_Y_word'][:]
#     test_X = data_file['test_X'][:]
#     test_Y = data_file['test_Y_word'][:]

train_Y = train_Y_word
test_Y = test_Y_word
    
train_X[train_X >= vocab_size] = 0
test_X[test_X >= vocab_size] = 0
    
if not samples_per_epoch:
    samples_per_epoch = train_X.shape[0]
        
howmuch = 5000
x_train = train_X[:howmuch]
# y_train = train_X[:howmuch]
x_val = test_X[:howmuch]
# y_val = test_X[:howmuch]

y_train = []
for sentence in x_train:
    new_sentence = []
    for word_id in sentence:
        if word_id != 0:
            new_sentence.append(word_id)
    y_train.append(new_sentence)
y_train = pad_sequences(y_train, maxlen=maxlen, padding='post')

y_val = []
for sentence in x_val:
    new_sentence = []
    for word_id in sentence:
        if word_id != 0:
            new_sentence.append(word_id)
    y_val.append(new_sentence)
y_val = pad_sequences(y_val, maxlen=maxlen, padding='post')

y_train = y_train.reshape((-1, maxlen, 1))
y_val = y_val.reshape((-1, maxlen, 1))

new_x_train = []
for sentence in x_train:
    new_sentence = []
    for word_id in sentence:
        if word_id != 0:
            new_sentence.insert(0, word_id)
    new_x_train.append(new_sentence)
new_x_train = pad_sequences(new_x_train, maxlen=maxlen)
x_train = new_x_train

new_x_val = []
for sentence in x_val:
    new_sentence = []
    for word_id in sentence:
        if word_id != 0:
            new_sentence.insert(0, word_id)
    new_x_val.append(new_sentence)
new_x_val = pad_sequences(new_x_val, maxlen=maxlen)
x_val = new_x_val

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 32
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')
model = Sequential()

# model.add(Input(shape=(maxlen,), dtype='int32'))

model.add(Embedding(vocab_size, 
              embedding_dim, 
              input_length=maxlen, 
              weights=[embedding_weights], 
              trainable=False))
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE))
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(maxlen))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(vocab_size)))
model.add(layers.Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# # Train the model each generation and show predictions against the validation
# # dataset.
# for iteration in range(1, 1000):
#     print()
#     print('-' * 50)
#     print('Iteration', iteration)
#     model.fit(x_train, y_train,
#               batch_size=BATCH_SIZE,
#             nb_epoch=1,
#             validation_data=(x_val, y_val))
    
earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
        nb_epoch=1000,
        callbacks=[earlyStopping], validation_split=0.2,)

X = x_val
Y = y_val
Y = Y.reshape((Y.shape[0], Y.shape[1]))
    
probs = model.predict(X, verbose=1)
one_hot_predictions = numpy.zeros((probs.shape[0], probs.shape[1]))
for i in range(len(probs)):
    for j in range(len(probs[i])):
        one_hot_predictions[i][j] = numpy.argmax(probs[i][j])
        
predict = one_hot_predictions
accuracy, _, _, _ = performance(predict, Y)
print('Accuracy:\t' + str(accuracy))
        
d = {}
with open(dictionary_file) as f:
    for line in f:
       (val, key) = line.split()
       d[int(key)] = val
       
out = ''
for i in range(Y.shape[0]):
    line = ''
    for j in range(Y.shape[1]):
        idx = Y[i][j]
        if idx == 0:
            continue
        word = d[idx]
        line += word
#         if Y[i][j] == 1:
#             line += '*'
        line += ' '
    line += '\n'
    out += line
with open(Y_words_file, 'w') as f:
    f.write(out)
out = ''
for i in range(predict.shape[0]):
    line = ''
    for j in range(predict.shape[1]):
        idx = predict[i][j]
        if idx == 0:
            continue
        word = d[idx]
        line += word
#         if predict[i][j] == 1:
#             line += '*'
        line += ' '
    line += '\n'
    out += line
with open(predict_words_file, 'w') as f:
    f.write(out)
    
a = 0
    
    
#     # Select 10 samples from the validation set at random so we can visualize
#     # errors.
#     for i in range(10):
#         ind = np.random.randint(0, len(x_val))
#         rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
#         preds = model.predict_classes(rowx, verbose=0)
#         q = ctable.decode(rowx[0])
#         correct = ctable.decode(rowy[0])
#         guess = ctable.decode(preds[0], calc_argmax=False)
#         print('Q', q[::-1] if INVERT else q)
#         print('T', correct)
#         if correct == guess:
#             print(colors.ok + '☑' + colors.close, end=" ")
#         else:
#             print(colors.fail + '☒' + colors.close, end=" ")
#         print(guess)
#         print('---')