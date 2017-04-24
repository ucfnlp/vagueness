#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy

numpy.random.seed(123)

# data generator
def batch_generator(X_padded, Y_padded_vague, batch_size):
    
    samples_per_epoch = X_padded.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    
    shuffle_index = numpy.arange(Y_padded_vague.shape[0])
    numpy.random.shuffle(shuffle_index)
        
    while True:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        
        X_batch = X_padded[index_batch]
        Y_batch_vague = Y_padded_vague[index_batch]
        sample_weights = (Y_batch_vague * 40 + 1).reshape((Y_batch_vague.shape[0]))
        
        counter += 1
        if counter == number_of_batches:
            numpy.random.shuffle(shuffle_index)
            counter = 0

        yield (X_batch, [Y_batch_vague], sample_weights)