#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy
import h5py

numpy.random.seed(123)
from keras.models import Sequential, Model, load_model
from sklearn.metrics import mean_absolute_error

from metrics import performance
from batch_generator import batch_generator

predict_file = '../predictions/predictions.h5'
model_file = '../models/model.h5'
dataset_file = '../data/dataset.h5'
sentence_pred_file = '../predictions/sentence.txt'
 
batch_size = 128
val_samples = batch_size


model = load_model(model_file)


print('loading data')
with h5py.File(dataset_file, 'r') as data_file:
    train_X_padded = data_file['train_X'][:]
    train_Y_padded_vague = data_file['train_Y_sentence'][:]
    test_X_padded = data_file['test_X'][:]
    test_Y_padded_vague = data_file['test_Y_sentence'][:]

howmany = 5000
X = test_X_padded[:howmany]
Y = test_Y_padded_vague[:howmany]

simple_mean_predictions = [numpy.mean(train_Y_padded_vague)]*len(Y)
comparative_mae = mean_absolute_error(Y, simple_mean_predictions)
print('Mean Absolute Error for simple mean: ' + str(comparative_mae))

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

predict = model.predict(X)
predict = predict.flatten()


mae = mean_absolute_error(Y, predict)
print('Mean Absolute Error:\t' + str(mae))
outfile.flush()
outfile.close()



if not Y.shape[0] == predict.shape[0]:
    print('Y len (' + str(Y.shape[0])
          + ') does not equal predict length (' + str(predict.shape[0]) + ')')

with open(sentence_pred_file, 'w') as f:
    for i in range(len(predict)):
        f.write(str(predict[i]) + '\t' + str(Y[i]) + '\n')
print('done')














