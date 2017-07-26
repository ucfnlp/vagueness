# The RNN logic is written using Keras's functional API.
# Which means we use Keras layers instead of theano/tensorflow ops
from keras.layers import *
from keras.models import *
from recurrentshop import *

x_t = Input((1,)) # The input to the RNN at time t
h_tm1 = Input((10,))  # Previous hidden state
o_tm1 = Input((3,)) # Previous output

x_t = Embedding(1, 5)(x_t)
a_t = Flatten()(x_t)
input_t = concatenate([a_t, o_tm1], axis=-1)

# Compute new hidden state
h_t = add([Dense(10)(input_t), Dense(10, use_bias=False)(h_tm1)])

# tanh activation
h_t = Activation('tanh')(h_t)

# Compute new output state from hidden state
o_t = Dense(3)(h_t)

# tanh activation
o_t = Activation('tanh')(o_t)

# Build the RNN
rnn = RecurrentModel(input=x_t, initial_states=[h_tm1, o_tm1], output=o_t, final_states=[h_t, o_t])

# rnn is a standard Keras `Recurrent` instance. RecuurentModel also accepts arguments such as unroll, return_sequences etc

# Run the RNN over a random sequence

x = Input((7))
y = rnn(x)

model = Model(x, y)
a = np.random.random((1, 7))
predict = model.predict(a)
print('done')