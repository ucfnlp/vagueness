# -*- coding: utf-8 -*-


from __future__ import print_function
from keras.models import Sequential
from keras import layers
from keras import backend as K
import numpy as np

def gomax(x):
    return K.argmax(x, axis=-1)

def gomax_output_shape(input_shape):
    shape = list(input_shape)
    shape = shape[:len(shape)-1]
    return tuple(shape)

model = Sequential()
model.add(layers.Lambda(gomax,input_shape=(5,),output_shape=gomax_output_shape))
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
x = np.array([[0,1,2,1,0],[2,6,2,2,2]])
y = np.array([1])
predict = model.predict(x)
print(predict)
