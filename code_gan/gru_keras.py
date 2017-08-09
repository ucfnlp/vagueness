#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function
import numpy
import tensorflow as tf
import h5py
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils

numpy.random.seed(123)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, merge, Input, TimeDistributed, Bidirectional
from keras.layers import Embedding, LSTM, GRU, Flatten
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras import metrics
from keras.callbacks import EarlyStopping

# from batch_generator import batch_generator
# from objectives import RankNet_mean

model_file = '../models/model.h5'
dataset_file = '../data/dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
dictionary_file = '../data/words.dict'
 
fast = False
 
vocab_size = 5000
embedding_dim = 300
maxlen = 50
hidden_dim = 512
batch_size = 128
nb_epoch = 1
samples_per_epoch = None

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('EPOCHS', 5000,
                            'Num epochs.')
tf.app.flags.DEFINE_integer('LATENT_SIZE', 512,
                            'Size of both the hidden state of RNN and random vector z.')
tf.app.flags.DEFINE_integer('SEQUENCE_LEN', 50,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('EMBEDDING_SIZE', 300,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('PATIENCE', 200,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('BATCH_SIZE', 128,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_string('CELL_TYPE', 'GRU',
                            'Which RNN cell for the RNNs.')
 
if fast:
    nb_epoch = 1
    samples_per_epoch = 100
    
print('loading embedding weights')
with h5py.File(embedding_weights_file, 'r') as hf:
    embedding_weights = hf['embedding_weights'][:]
    
print('loading dictionary')
d = {}
with open(dictionary_file) as f:
    for line in f:
       (val, key) = line.split()
       d[int(key)] = val
    
print('loading training and test data')
with h5py.File(dataset_file, 'r') as data_file:
    train_X = data_file['train_X'][:]
    train_Y = data_file['train_Y'][:]
    test_X = data_file['test_X'][:]
    test_Y = data_file['test_Y'][:]

    
# howmany = 5000
# train_X = train_X[:howmany]
# train_Y = train_Y[:howmany]
    
if not samples_per_epoch:
    samples_per_epoch = train_X.shape[0]
        
# build model
print('building model')
my_input = Input(shape=(maxlen,), dtype='int32')
 
embedded = Embedding(vocab_size, 
              embedding_dim, 
              input_length=maxlen, 
              weights=[embedding_weights], 
              dropout=0.2,
              trainable=False)(my_input)
               
# forwards = Bidirectional(GRU(hidden_dim,
#                return_sequences=True,
#                dropout_W=0.2,
#                dropout_U=0.2))(embedded)
forwards = GRU(hidden_dim,
               return_sequences=True,
               dropout_W=0.2,
               dropout_U=0.2)(embedded)
 
output = Dropout(0.5)(forwards)
 
# output_vague = TimeDistributed(Dense(1, activation='sigmoid'), name='loss_vague')(output)
output_lm = TimeDistributed(Dense(vocab_size, activation='softmax'), name='loss_lm')(output)



model = Model(input=my_input, output=[output_lm])
model.compile(optimizer='rmsprop',
              loss={'loss_lm': 'categorical_crossentropy'},
              loss_weights={'loss_lm': 1.},
              metrics=['accuracy'],
              sample_weight_mode='temporal')

get_hidden_layer = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[2].output])



# val_ratio = 0.8
# val_len = val_ratio * train_X.shape[0]
# val_X = train_X[:val_len]
# val_Y = train_Y[:val_len]
# train_X = train_X[val_len:]
# train_Y = train_Y[val_len:]
# sample_weights = (val_Y * 40 + 1).reshape((val_Y.shape[0], val_Y.shape[1]))
# earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
# model.fit_generator(batch_generator(train_X, train_Y, batch_size), 
#                     samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
#                     callbacks=[earlyStopping], validation_data=(val_X, val_Y) )

def idx_to_categorical(y, num_categories):
    categorical_y = numpy.array(np_utils.to_categorical(y.flatten(), num_categories))
    categorical_y = categorical_y.reshape(-1, y.shape[1], num_categories)
    return categorical_y

def batch_generator(x, y):
    data_len = x.shape[0]
    for i in range(0, data_len, batch_size):
        batch_x = x[i:min(i+batch_size,data_len)]
        batch_y = y[i:min(i+batch_size,data_len)]
        batch_y = idx_to_categorical(batch_y, vocab_size)
        yield batch_x, batch_y, i, data_len

for cur_epoch in range(nb_epoch):
    for batch_x, batch_y, cur, data_len in batch_generator(train_X, train_Y):
        loss, accuracy = model.train_on_batch(batch_x, batch_y)
        test_batch_x = test_X[:batch_size]
        test_batch_y = test_Y[:batch_size]
        logits = model.predict_on_batch(test_batch_x)
        preds = numpy.argmax(logits, axis=2)
        for i in range(min(2, len(preds))):
            for j in range(len(preds[i])):
                if test_batch_y[i][j] == 0:
                    print '<>',
                else:
                    word = d[test_batch_y[i][j]]
                    print word,
                print '\t\t',
                if preds[i][j] == 0:
                    print '<>',
                else:
                    word = d[preds[i][j]]
                    print word,
                print '\n'
            print '\n'
        print(preds)
        print('Iter: {}'.format(cur_epoch))
        print('Instance ', cur, ' out of ', data_len)
        print('Loss ', loss)
        print('Accuracy ', accuracy)
        
model.save(model_file)

print('done')

















































