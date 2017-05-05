#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy
import h5py

numpy.random.seed(123)
from keras.models import Sequential, Model, load_model

from metrics import performance
from batch_generator import batch_generator

predict_file = '../predictions/predictions.h5'
model_file = '../models/model.h5'
dataset_file = '../data/dataset.h5'
Y_words_file = '../predictions/Y_words.txt'
predict_words_file = '../predictions/predictions_words.txt'
dictionary_file = '../data/words.dict'
 
batch_size = 128
val_samples = batch_size


model = load_model(model_file)


print('loading data')
with h5py.File(dataset_file, 'r') as data_file:
    train_X_padded = data_file['train_X'][:]
    train_Y_padded_vague = data_file['train_Y_word'][:]
    test_X_padded = data_file['test_X'][:]
    test_Y_padded_vague = data_file['test_Y_word'][:]

howmany = 5000
X = train_X_padded[:howmany]
Y = train_Y_padded_vague[:howmany]

# X = numpy.reshape(X, (1, X.shape[0]))
# Y = numpy.reshape(Y, (1, Y.shape[0]))

#test
outfile = h5py.File(predict_file, 'w')
# probs = outfile.create_dataset('Y_predict_vague', (X.shape[0], X.shape[1], 1))
probs = outfile.create_dataset('Y_predict_vague', Y.shape)
# idx = 0
# while idx < X.shape[0] - (X.shape[0] % val_samples):
# # for i in range(2):
#     print('Test: ' + str(idx) + '/' + str(X.shape[0]))
#     end = min(idx + val_samples, X.shape[0])
#     probs[idx:end] = model.predict_generator(
#     batch_generator(X, Y, batch_size), 
#     val_samples)
#     idx += val_samples
# predict = numpy.round(probs)

probs = model.predict(X)
predict = numpy.round(probs)



accuracy, precision, recall, f1score = performance(predict, Y)
print('Accuracy:\t' + str(accuracy))
print('Precision:\t' + str(precision))
print('Recall:\t\t' + str(recall))
print('F1 score:\t' + str(f1score))
outfile.flush()
outfile.close()



if not X.shape[0] == predict.shape[0]:
    print('X len (' + str(X.shape[0])
          + ') does not equal predict length (' + str(predict.shape[0]) + ')')

d = {}
with open(dictionary_file) as f:
    for line in f:
       (val, key) = line.split()
       d[int(key)] = val
       
out = ''
for i in range(X.shape[0]):
    line = ''
    for j in range(X.shape[1]):
        idx = X[i][j]
        if idx == 0:
            continue
        word = d[idx]
        line += word
        if Y[i][j] == 1:
            line += '*'
        line += ' '
    line += '\n'
    out += line
with open(Y_words_file, 'w') as f:
    f.write(out)
out = ''
for i in range(X.shape[0]):
    line = ''
    for j in range(X.shape[1]):
        idx = X[i][j]
        if idx == 0:
            continue
        word = d[idx]
        line += word
        if predict[i][j] == 1:
            line += '*'
        line += ' '
    line += '\n'
    out += line
with open(predict_words_file, 'w') as f:
    f.write(out)
print('done')














