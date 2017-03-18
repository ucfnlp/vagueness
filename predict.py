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
predict_vague_file = 'predictions_vague.h5'
model_file = 'model_V_weighted_loss.h5'
dataset_file = 'dataset.h5'
embedding_weights_file = 'embedding_weights.h5'
Y_vague_predict_file = 'Y_vague_predict.out'
test_Y_vague_file = 'test_Y_padded_vague.out'
 
fast = False
 
batch_size = 128
val_samples = batch_size * 10


model = load_model(model_file)

print('loading test data')
with h5py.File(dataset_file, 'r') as data_file:
    test_X_padded = data_file['test_X'][:]
    test_Y_padded = data_file['test_Y'][:]
    test_Y_padded_vague = data_file['test_Y_vague'][:]

#test
outfile = h5py.File(predict_file, 'w')
Y_vague_predict = outfile.create_dataset('Y_predict_vague', (test_X_padded.shape[0], test_X_padded.shape[1], 1))
idx = 0
while idx < test_X_padded.shape[0] - (test_X_padded.shape[0] % val_samples):
# for i in range(2):
    print('Test: ' + str(idx) + '/' + str(test_X_padded.shape[0]))
    end = min(idx + val_samples, test_X_padded.shape[0])
    Y_vague_predict[idx:end] = model.predict_generator(
    batch_generator(test_X_padded, test_Y_padded_vague, batch_size), 
    val_samples)
    idx += val_samples
Y_vague_predict = numpy.round(Y_vague_predict)
numpy.set_printoptions(threshold=numpy.nan)
numpy.savetxt(Y_vague_predict_file, Y_vague_predict, delimiter=',', fmt='%d')
numpy.savetxt(test_Y_vague_file, test_Y_padded_vague, delimiter=',', fmt='%d')
accuracy, precision, recall, f1score = performance(Y_vague_predict, test_Y_padded_vague)
print('Accuracy:\t' + str(accuracy))
print('Precision:\t' + str(precision))
print('Recall:\t\t' + str(recall))
print('F1 score:\t' + str(f1score))
outfile.flush()
outfile.close()















