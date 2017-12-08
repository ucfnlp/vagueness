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
                                  sample_from_distribution=FLAGS.SAMPLE,
                                  vague_weights=vague_weights,
                                  embedding_matrix=embedding_matrix,
                                  hidden_noise_std_dev=None,
                                  vocab_noise_std_dev=None,
                                  gumbel=gumbel_noise,
                                  gumbel_mu=gumbel_mu,
                                  gumbel_sigma=gumbel_sigma)
#                                   class_embedding=c_embedding),

        samples = tf.cast(tf.stack(samples, axis=1), tf.int32)
        probs = tf.stack(probs, axis=1)
        
        ''' Used for clipping all words after <eos> word '''
        batch_size = tf.stack([tf.shape(c)[0],])
        eos = tf.fill(batch_size, 3)
        eos = tf.reshape(eos, [-1, 1])
        d=tf.concat([samples,eos],1)
        B=tf.cast(tf.argmax(tf.cast(tf.equal(d, 3), tf.int32), axis=1), tf.int32)
        m=tf.sequence_mask(B, tf.shape(samples)[1], dtype=tf.int32)
        samples=tf.multiply(samples,m)
         
        o=tf.reshape(m,[-1,FLAGS.SEQUENCE_LEN,1])
        n = tf.tile(o,[1,1,FLAGS.VOCAB_SIZE])
        u=tf.cast(tf.unstack(n,axis=1),tf.float32)
#         u=None
        
#         logits = [tf.matmul(output, W) + b for output in outputs] #TODO add vague vocabulary, and remove class embedding
#         weighted_logits = [tf.add(logit, vague_weights) for logit in logits]
#         if FLAGS.VOCAB_NOISE_STD_DEV != 0:
#           weighted_logits = [utils.gaussian_noise_layer(wl, std=FLAGS.VOCAB_NOISE_STD_DEV) for wl in weighted_logits]
        if FLAGS.GUMBEL:
            logits = [logit/FLAGS.TAU for logit in logits]
        x = [tf.nn.softmax(logit) for logit in logits] # is this softmaxing over the right dimension? this turns into 3D
                                                                # and does softmax make sense here in between gen and discr?
#         x = [tf.nn.tanh(logit) for logit in weighted_logits]
        ''' Used for clipping all words after <eos> word '''
        for i in range(len(x)):
            x[i] = tf.multiply(x[i], u[i])

        return x, samples, probs, u, m, logits, pure_logits, vague_weights, inps




























