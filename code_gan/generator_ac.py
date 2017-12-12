import numpy as np
import tensorflow as tf
from seq2seq import  embedding_rnn_decoder
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import utils

FLAGS = tf.app.flags.FLAGS
start_symbol_index = 2

def generator(z, c, initial_vague_terms, embedding_matrix, keep_prob, gumbel_mu, gumbel_sigma):
    
    with tf.variable_scope("G_"):
        cell = utils.create_cell(keep_prob)
#         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        dims = tf.stack([tf.shape(c)[0],])
        start_symbol_input = [tf.fill(dims, start_symbol_index) for i in range(FLAGS.SEQUENCE_LEN)]
        initial_state = tf.contrib.rnn.LSTMStateTuple(tf.zeros([dims[0], FLAGS.LATENT_SIZE]), tf.zeros([dims[0], FLAGS.LATENT_SIZE]))
        gumbel_noise = z if FLAGS.GUMBEL else None
        weights = tf.get_variable("output_weights", shape=[FLAGS.LATENT_SIZE, FLAGS.VOCAB_SIZE],
           initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("output_biases", shape=[FLAGS.VOCAB_SIZE],
           initializer=tf.zeros_initializer())
        
        if FLAGS.USE_VAGUE_VECTOR:
            vague_terms = tf.get_variable(initializer=tf.constant(initial_vague_terms, dtype=tf.float32), dtype=tf.float32, name='vague_terms')
            def create_vague_weights(vague_terms, c):
                a = tf.tile(vague_terms, dims)
                b = tf.reshape(a,[-1,FLAGS.VOCAB_SIZE])
                vague_weights = tf.multiply(b,tf.cast(tf.reshape(c - 1, [-1,1]),tf.float32))
                return vague_weights
            vague_weights = create_vague_weights(vague_terms, c)
        else:
            vague_weights = None
        
        outputs, states, samples, probs, logits, pure_logits, inps = embedding_rnn_decoder(start_symbol_input,   # is this ok? I'm not sure what giving 0 inputs does (although it should be completely ignoring inputs)
                                  initial_state,
                                  cell,
                                  FLAGS.VOCAB_SIZE,
                                  FLAGS.EMBEDDING_SIZE,
                                  output_projection=(weights,biases),
                                  feed_previous=True,
                                  update_embedding_for_previous=True,
                                  vague_weights=vague_weights,
                                  embedding_matrix=embedding_matrix,
                                  gumbel=gumbel_noise,
                                  gumbel_mu=gumbel_mu,
                                  gumbel_sigma=gumbel_sigma)

        pure_logits = tf.stack(pure_logits, axis=1)
        logits = tf.stack(logits, axis=1)
        samples = tf.cast(tf.stack(samples, axis=1), tf.int32)
        probs = tf.stack(probs, axis=1)
        outputs = tf.stack(outputs, axis=1)
        
        ''' Used for clipping all words after <eos> word '''
        EOS_idx=utils.get_EOS_idx(samples)
        m=tf.sequence_mask(EOS_idx, FLAGS.SEQUENCE_LEN, dtype=tf.float32)
        samples=tf.multiply(samples,tf.cast(m, tf.int32))
                                                    
        if FLAGS.GUMBEL:
            logits = logits / FLAGS.TAU
        x = tf.nn.softmax(logits)         # and does softmax make sense here in between gen and discr?
#         x = [tf.nn.tanh(logit) for logit in weighted_logits]
        ''' Used for clipping all words after <eos> word '''
        x = tf.multiply(x, tf.expand_dims(m, axis=-1))
    
        x_emb = tf.matmul(tf.reshape(x, [-1, FLAGS.VOCAB_SIZE]), embedding_matrix)
        x_emb = tf.reshape(x_emb, [-1, FLAGS.SEQUENCE_LEN, FLAGS.EMBEDDING_SIZE])

        return x, samples, probs, EOS_idx, logits, pure_logits, vague_weights, inps, m, outputs, x_emb




























