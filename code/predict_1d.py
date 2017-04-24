#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy
import h5py

numpy.random.seed(123)
from keras.models import Sequential, Model, load_model

from metrics import performance
from batch_generator import batch_generator

predict_file = 'predictions.h5'
model_file = 'model_nn_time_distributed.h5'
dataset_file = 'dataset.h5'
Y_words_file = 'test_Y_words_1d.txt'
predict_words_file = 'test_predict_words_1d.txt'
 
fast = False
 
batch_size = 128
val_samples = batch_size * 10


model = load_model(model_file)

d = {}
with open("words.dict") as f:
    for line in f:
       (val, key) = line.split()
       d[int(key)] = val

print('loading data')
with h5py.File(dataset_file, 'r') as data_file:
    train_X_padded = data_file['train_X'][:]
    train_Y_padded = data_file['train_Y'][:]
    train_Y_padded_vague = data_file['train_Y_vague'][:]
    test_X_padded = data_file['test_X'][:]
    test_Y_padded = data_file['test_Y'][:]
    test_Y_padded_vague = data_file['test_Y_vague'][:]

X = test_X_padded
Y = test_Y_padded_vague

X = X.flatten()
Y = Y.flatten()

#test
outfile = h5py.File(predict_file, 'w')
predict = outfile.create_dataset('Y_predict_vague', (X.shape[0], 1))
idx = 0
while idx < X.shape[0] - (X.shape[0] % val_samples):
# for i in range(2):
    print('Test: ' + str(idx) + '/' + str(X.shape[0]))
    end = min(idx + val_samples, X.shape[0])
    predict[idx:end] = model.predict_generator(
    batch_generator(X, Y, batch_size), 
    val_samples)
    idx += val_samples
predict = numpy.round(predict)
# numpy.set_printoptions(threshold=numpy.nan)
# numpy.savetxt(predict_file, predict, delimiter=',', fmt='%d')
# numpy.savetxt(test_Y_vague_file, Y, delimiter=',', fmt='%d')
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

out = ''
line_ctr = 0
for i in range(X.shape[0]):
    line = ''
    idx = X[i]
    if idx == 0:
        continue
    word = d[idx]
    line += word
    if Y[i] == 1:
        line += '*'
    line += ' '
    line_ctr += 1
    if line_ctr >= 20:
        line += '\n'
    out += line
with open(Y_words_file, 'w') as f:
    f.write(out)
out = ''
for i in range(X.shape[0]):
    line = ''
    idx = X[i]
    if idx == 0:
        continue
    word = d[idx]
    line += word
    if predict[i] == 1:
        line += '*'
    line += ' '
    line_ctr += 1
    if line_ctr >= 20:
        line += '\n'
    out += line
with open(predict_words_file, 'w') as f:
    f.write(out)
print('done')














