#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import h5py
import tensorflow as tf

numpy.random.seed(123)

from metrics import performance

predict_file = '../predictions/predictions.h5'
model_file = '../models/tf_lm_model'
dataset_file = '../data/dataset.h5'
predict_words_file = '../predictions/tf_predictions_and_y_words.txt'
dictionary_file = '../data/words.dict'

 
batch_size = 128
val_samples = batch_size


print('loading data')
with h5py.File(dataset_file, 'r') as data_file:
    train_X_padded = data_file['train_X'][:]
    train_Y_padded_vague = data_file['train_Y'][:]
    test_X_padded = data_file['test_X'][:]
    test_Y_padded_vague = data_file['test_Y'][:]

# howmany = 50
# X = test_X_padded[:howmany]
# Y = test_Y_padded_vague[:howmany]
X = test_X_padded
Y = test_Y_padded_vague

# X = numpy.reshape(X, (1, X.shape[0]))
# Y = numpy.reshape(Y, (1, Y.shape[0]))

#test
outfile = h5py.File(predict_file, 'w')
# probs = outfile.create_dataset('Y_predict_vague', (X.shape[0], X.shape[1], 1))
predict = outfile.create_dataset('Y_predict_vague', Y.shape, dtype='int32')
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

def batch_generator(x, y):
    data_len = x.shape[0]
    for i in range(0, data_len, batch_size):
        batch_x = x[i:min(i+batch_size,data_len)]
        batch_y = y[i:min(i+batch_size,data_len)]
#         batch_y = idx_to_categorical(batch_y, vocab_size)
        yield batch_x, batch_y, i, data_len

print 'predicting...'
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('gru-model.meta')
    new_saver.restore(sess, 'gru-model')
    # tf.get_collection() returns a list. In this example we only want the
    # first one.
    inputs = tf.get_collection('inputs')[0]
    predictions = tf.get_collection('predictions')[0]
    for batch_x, batch_y, cur, data_len in batch_generator(X, Y):
        print 'Predicting instance ', cur, ' out of ', data_len
        batch_predictions = sess.run([predictions], feed_dict={inputs:batch_x})
#         batch_predictions = numpy.argmax(batch_predictions, axis=2)
        batch_predictions = batch_predictions[0]
    #     batch_predictions = batch_predictions.flatten()
        predict[cur:cur+len(batch_predictions)] = batch_predictions



accuracy, precision, recall, f1score = performance(predict, Y)
print('Accuracy:\t' + str(accuracy))
print('Precision:\t' + str(precision))
print('Recall:\t\t' + str(recall))
print('F1 score:\t' + str(f1score))



if not X.shape[0] == predict.shape[0]:
    print('X len (' + str(X.shape[0])
          + ') does not equal predict length (' + str(predict.shape[0]) + ')')

d = {}
with open(dictionary_file) as f:
    for line in f:
       (val, key) = line.split()
       d[int(key)] = val
       
print 'printing to ' + predict_words_file
out = ''
for i in range(len(predict)):
    for j in range(len(predict[i])):
        if Y[i][j] == 0:
            out += '<>'
        else:
            word = d[Y[i][j]]
            out += word
        out += '\t\t'
        if predict[i][j] == 0:
            out += '<>'
        else:
            word = d[predict[i][j]]
            out += word
        out += '\n'
    out += '\n'
with open(predict_words_file, 'w') as f:
    f.write(out)
    
outfile.flush()
outfile.close()
print('done')














