import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def cnn(embeddings, keep_prob):
    
    embeddings_expanded = tf.expand_dims(embeddings, -1)
    
    with tf.variable_scope('cnn'):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        filter_sizes = list(map(int, FLAGS.FILTER_SIZES.split(",")))
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, FLAGS.EMBEDDING_SIZE, 1, FLAGS.NUM_FILTERS]
                W = tf.get_variable('W', shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[FLAGS.NUM_FILTERS]))
                conv = tf.nn.conv2d(
                    embeddings_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, FLAGS.SEQUENCE_LEN - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        
        # Combine all the pooled features
        num_filters_total = FLAGS.NUM_FILTERS * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        
        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob)
    
    return h_drop