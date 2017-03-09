#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy
import theano
import h5py

numpy.random.seed(123)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, merge, Input, TimeDistributed
from keras.layers import Embedding, LSTM, GRU
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2, activity_l2

from metrics import performance
from batch_generator import batch_generator

model_file = 'model_V_weighted_loss.h5'
dataset_file = 'dataset.h5'
embedding_weights_file = 'embedding_weights.h5'
 
fast = False
 
vocab_size = 5000
embedding_dim = 300
maxlen = 50
hidden_dim = 512
batch_size = 128
nb_epoch = 30
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
              
forwards = GRU(hidden_dim,
               return_sequences=True,
               dropout_W=0.2,
               dropout_U=0.2)(embedded)

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
              loss={'loss_vague':'binary_crossentropy'},
              loss_weights={'loss_vague': 1.},
              metrics=['accuracy'],
              sample_weight_mode='temporal')

get_hidden_layer = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[2].output])
sample_weight = (train_Y_padded_vague * 40) + 1
model.fit_generator(batch_generator(train_X_padded, train_Y_padded_vague, batch_size), 
                    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch)
model.save(model_file)

print('done')

















































