#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy
import theano
import h5py

numpy.random.seed(123)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, merge, Input, TimeDistributed, Bidirectional, Flatten
from keras.layers import Embedding, LSTM, GRU
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2, activity_l2
# from keras.metrics import precision, recall, f1score, binary_accuracy
from keras import metrics

from metrics import performance
from batch_generator import batch_generator
from objectives import RankNet_mean

model_file = 'model_nn_time_distributed.h5'
dataset_file = 'dataset.h5'
embedding_weights_file = 'embedding_weights.h5'
 
fast = False
 
vocab_size = 5000
embedding_dim = 300
maxlen = 50
hidden_dim = 512
batch_size = 128
nb_epoch = 5
samples_per_epoch = None
 
if fast:
    nb_epoch = 1
    samples_per_epoch = 100
    
print('loading embedding weights')
with h5py.File(embedding_weights_file, 'r') as hf:
    embedding_weights = hf['embedding_weights'][:]
    
print('loading training and test data')
with h5py.File(dataset_file, 'r') as data_file:
    train_X_padded = data_file['train_X'][:]
    train_Y_padded = data_file['train_Y'][:]
    train_Y_padded_vague = data_file['train_Y_vague'][:]
    
train_X_padded = train_X_padded.flatten()
train_Y_padded_vague = train_Y_padded_vague.flatten()
if not samples_per_epoch:
    samples_per_epoch = train_X_padded.shape[0]
        
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
hidden = Dense(100, activation='sigmoid')(flatten)
forwards = Dense(1, activation='sigmoid')(hidden)

model = Model(input=my_input, output=[forwards])
# optimizer = optimizers.RMSprop(lr=0.001)
optimizer = 'rmsprop'
model.compile(optimizer=optimizer,
            loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(batch_generator(train_X_padded, train_Y_padded_vague, batch_size), 
                    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch)
model.save(model_file)

print('done')

















































