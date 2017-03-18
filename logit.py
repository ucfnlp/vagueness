#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy
import theano
import h5py
import codecs
import operator

numpy.random.seed(123)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, merge, Input, TimeDistributed, Flatten
from keras.layers import Embedding, LSTM, GRU
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2, activity_l2
from keras.preprocessing.text import Tokenizer
from keras import optimizers

from metrics import performance
from batch_generator import batch_generator

model_file = 'model_V_logit.h5'
dataset_file = 'dataset.h5'
embedding_weights_file = 'embedding_weights.h5'
vague_file = 'vague_terms'
train_file = 'Privacy_Sentences.txt'
 
fast = False
 
vocab_size = 5000
embedding_dim = 300
maxlen = 50
hidden_dim = 512
batch_size = 128
nb_epoch = 500
samples_per_epoch = None
train_ratio = 0.8

def splitPerc(l):
    splits = numpy.cumsum([train_ratio*100, (1-train_ratio)*100])/100.
    if splits[-1] != 1:
        raise ValueError("percents don't add up to 100")
    # split doesn't need last percent, it will just take what is left
    return numpy.split(l, splits[:-1]*len(l))

def shuffleTogether(X, Y):
    permutation = numpy.random.permutation(X.shape[0])
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    return shuffled_X, shuffled_Y
 
if fast:
    nb_epoch = 1
    samples_per_epoch = 100
    
print('loading embedding weights')
with h5py.File(embedding_weights_file, 'r') as hf:
    embedding_weights = hf['embedding_weights'][:]
    
# print('loading training and test data')
# with h5py.File(word_id_file, 'r') as hf:
#     my_word_ids = hf['words'][:]

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
print('finished creating the dictionary')
    
# load file containing vague terms
vague_terms = []
with codecs.open(vague_file) as infile:
    for line in infile:
        words = line.strip().split()
        if len(words) != 1:
            continue
        vague_terms.append(words[0])
        
    
X = numpy.arange(1,vocab_size)
Y = numpy.zeros((X.shape))
for word, idx in sorted(tokenizer.word_index.items(), key=operator.itemgetter(1)):
    if idx > vocab_size:
        break
    if word in vague_terms:
        Y[idx-1] = 1
permutation = numpy.random.permutation(X.shape[0])
X = X[permutation]
Y = Y[permutation]
X1 = X[Y==1]
Y1 = Y[Y==1]
X0 = X[Y==0]
Y0 = Y[Y==0]
train_X1, test_X1 = splitPerc(X1)
train_Y1, test_Y1 = splitPerc(Y1)
train_X0, test_X0 = splitPerc(X0)
train_Y0, test_Y0 = splitPerc(Y0)
# train_X0 = train_X0[:1000]
# train_Y0 = train_Y0[:1000]
train_X = numpy.concatenate((train_X1, train_X0))
train_Y = numpy.concatenate((train_Y1, train_Y0))
test_X = numpy.concatenate((test_X1, test_X0))
test_Y = numpy.concatenate((test_Y1, test_Y0))
train_X, train_Y = shuffleTogether(train_X, train_Y)
test_X, test_Y = shuffleTogether(test_X, test_Y)

        
        
# build model
print('building model')
my_input = Input(shape=(1,), dtype='int32')

embedded = Embedding(vocab_size, 
              embedding_dim, 
              input_length=1,
              weights=[embedding_weights], 
              dropout=0.2,
              trainable=False)(my_input)
              
flatten = Flatten()(embedded)
               
forwards = Dense(1, activation='sigmoid')(flatten)

# forwards = GRU(hidden_dim,
#                return_sequences=True,
#                dropout_W=0.2,
#                dropout_U=0.2,
#                W_regularizer=l2(0.1),
#                U_regularizer=l2(0.1),
#                b_regularizer=l2(0.1))(embedded)
# 
# output = Dropout(0.5)(forwards)
# 
# output_lm = TimeDistributed(Dense(vocab_size, activation='softmax', W_regularizer=l2(0.1), b_regularizer=l2(0.1)), name='loss_lm')(output)
# output_vague = TimeDistributed(Dense(1, activation='sigmoid', W_regularizer=l2(0.1), b_regularizer=l2(0.1)), name='loss_vague')(output)

model = Model(input=my_input, output=[forwards])
optimizer = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_X, train_Y, nb_epoch=nb_epoch)
model.save(model_file)

predictions = model.predict(test_X)
predictions = numpy.round(predictions)
accuracy, precision, recall, f1score = performance(predictions, test_Y)
numpy.savetxt('predictions.txt', predictions, delimiter=',', fmt='%d')
numpy.savetxt('Y.txt', test_Y, delimiter=',', fmt='%d')
print('Accuracy:\t' + str(accuracy))
print('Precision:\t' + str(precision))
print('Recall:\t\t' + str(recall))
print('F1 score:\t' + str(f1score))

print('done')














































