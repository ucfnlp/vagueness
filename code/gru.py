#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
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
# from keras.metrics import precision, recall, f1score, binary_accuracy
from keras import metrics
from keras.callbacks import EarlyStopping

from metrics import performance
from batch_generator import batch_generator
from objectives import RankNet_mean

model_file = '../models/model.h5'
dataset_file = '../data/dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
 
fast = False
 
vocab_size = 5000
embedding_dim = 300
maxlen = 50
hidden_dim = 512
batch_size = 128
nb_epoch = 200
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
    train_Y_padded_vague = data_file['train_Y_word'][:]
    
howmany = 5000
train_X_padded = train_X_padded[:howmany]
train_Y_padded_vague = train_Y_padded_vague[:howmany]
 
# train_X_padded = numpy.reshape(train_X_padded, (1, train_X_padded.shape[0]))
# train_Y_padded_vague = numpy.reshape(train_Y_padded_vague, (1, train_Y_padded_vague.shape[0]))
# batch_size = 1
    
if not samples_per_epoch:
    samples_per_epoch = train_X_padded.shape[0]
        
# build model
print('building model')
my_input = Input(shape=(maxlen,), dtype='int32')

embedded = Embedding(vocab_size, 
              embedding_dim, 
              input_length=maxlen, 
              weights=[embedding_weights], 
              dropout=0.2,
              trainable=False)(my_input)
              
forwards = Bidirectional(GRU(hidden_dim,
               return_sequences=True,
               dropout_W=0.2,
               dropout_U=0.2))(embedded)

output = Dropout(0.5)(forwards)

#     output_lm = TimeDistributed(Dense(vocab_size, activation='softmax'), name='loss_lm')(output)
output_vague = TimeDistributed(Dense(1, activation='sigmoid'), name='loss_vague')(output)

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

model = Model(input=my_input, output=[output_vague])
model.compile(optimizer='rmsprop',
              loss={'loss_vague': 'binary_crossentropy'},
              loss_weights={'loss_vague': 1.},
              metrics=['accuracy','precision','recall'],
              sample_weight_mode='temporal')

get_hidden_layer = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[2].output])

# val_ratio = 0.8
# val_len = val_ratio * train_X_padded.shape[0]
# val_X = train_X_padded[:val_len]
# val_Y = train_Y_padded_vague[:val_len]
# train_X_padded = train_X_padded[val_len:]
# train_Y_padded_vague = train_Y_padded_vague[val_len:]
# sample_weights = (val_Y * 40 + 1).reshape((val_Y.shape[0], val_Y.shape[1]))
# earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
# model.fit_generator(batch_generator(train_X_padded, train_Y_padded_vague, batch_size), 
#                     samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
#                     callbacks=[earlyStopping], validation_data=(val_X, val_Y) )

earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
model.fit(train_X_padded, train_Y_padded_vague, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[earlyStopping], 
          validation_split=0.2, validation_data=None, shuffle=True,
          class_weight=None, sample_weight=None)
# model.fit(train_X_padded, train_Y_padded_vague,
#                     batch_size=batch_size, nb_epoch=nb_epoch)
model.save(model_file)

print('done')

















































