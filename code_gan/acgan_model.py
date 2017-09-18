import numpy as np
import tensorflow as tf
from generator_ac import generator
from discriminator_ac import discriminator
import param_names
import utils

start_symbol_index = 2

FLAGS = tf.app.flags.FLAGS

class ACGANModel(object):
    
    def __init__(self, vague_terms, params):
        """ init the model with hyper-parameters etc """
        self.vague_terms = tf.constant(vague_terms, dtype=tf.float32)
        self.params = params
        
    def run_D_train_step(self, sess, batch_x, batch_c, z, batch_fake_c):
        to_return = [self.D_solver, self.D_loss, self.D_real_acc, self.D_fake_acc, 
            self.D_real_class_acc, self.D_fake_class_acc]
        return sess.run(to_return,
                    feed_dict={self.real_x: batch_x,
                               self.real_c: batch_c,
                               self.fake_c: batch_fake_c,
                               self.z: z})
    
    def run_G_train_step(self, sess, batch_x, batch_c, z, batch_fake_c):
        to_return = [self.G_solver, self.G_loss, self.samples, self.probs, self.merged]
        return sess.run(to_return,
                    feed_dict={self.real_x: batch_x,
                               self.real_c: batch_c,
                               self.fake_c: batch_fake_c,
                               self.z: z})
        
    def run_test(self, sess, batch_x):
        to_return = self.D_class_logit_real
        return sess.run(to_return,
                        feed_dict={self.real_x: batch_x})
        
    def run_samples(self, sess, batch_fake_c, batch_z):
        return sess.run(self.samples, feed_dict={self.fake_c: batch_fake_c,
                                                 self.z: batch_z})
        
    def get_variables(self, sess):
        return sess.run(tf.trainable_variables())
    
    def assign_variables(self, sess):
        sess.run(self.assign_ops)
        
    def get_global_step(self):
        return self.global_step.eval()
    
    def set_global_step(self, value):
        self.global_step.assign(value).eval()
        
    def _add_placeholder(self):
        if FLAGS.SAMPLE:
            self.real_x = tf.placeholder(tf.int32, shape=[None, FLAGS.SEQUENCE_LEN])
        else:
            self.real_x = tf.placeholder(tf.float32, shape=[None, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
        self.real_c = tf.placeholder(tf.int32, [None,], 'class')
        self.fake_c = tf.placeholder(tf.int32, [None,], 'class')
        self.z = tf.placeholder(tf.float32, [None, FLAGS.LATENT_SIZE], name='z')
        self.dims = tf.stack([tf.shape(self.fake_c)[0],])
        self.start_symbol_input = [tf.fill(self.dims, start_symbol_index) for i in xrange(FLAGS.SEQUENCE_LEN)]
        
        def create_vague_weights(vague_terms, fake_c):
            a = tf.tile(vague_terms, self.dims)
            b = tf.reshape(a,[-1,FLAGS.VOCAB_SIZE])
            vague_weights = tf.multiply(b,tf.cast(tf.reshape(self.fake_c - 1, [-1,1]),tf.float32))
            return vague_weights
        self.vague_weights = create_vague_weights(self.vague_terms, self.fake_c)
        self.embedding_matrix = tf.Variable(tf.random_normal([FLAGS.VOCAB_SIZE, FLAGS.EMBEDDING_SIZE]), name='embedding_matrix')
        
    def _add_acgan(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            G_sample, self.samples, self.probs, self.u = generator(self.z, self.fake_c, self.vague_weights, self.start_symbol_input, self.embedding_matrix) #TODO move to generator
            self.D_real, self.D_logit_real, self.D_class_logit_real = discriminator(self.real_x, self.embedding_matrix)
            tf.get_variable_scope().reuse_variables()
            if FLAGS.SAMPLE:
                self.D_fake, self.D_logit_fake, self.D_class_logit_fake = discriminator(self.samples, self.embedding_matrix)
            else:
                self.D_fake, self.D_logit_fake, self.D_class_logit_fake = discriminator(G_sample, self.embedding_matrix)
        a=0
        
    def _add_loss(self):
        self.D_real_acc = tf.cast(utils.tf_count(tf.round(self.D_real), 1), tf.float32) / tf.cast(
            tf.shape(self.D_real)[0], tf.float32)
        self.D_fake_acc = tf.cast(utils.tf_count(tf.round(self.D_fake), 0), tf.float32) / tf.cast(
            tf.shape(self.D_fake)[0], tf.float32)
        self.D_real_class_acc = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(
            self.D_class_logit_real, axis=1), tf.int32), self.real_c), tf.float32)) /  tf.cast(
                tf.shape(self.real_c)[0], tf.float32)
        self.D_fake_class_acc = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(
            self.D_class_logit_fake, axis=1), tf.int32), self.fake_c), tf.float32)) /  tf.cast(
                tf.shape(self.fake_c)[0], tf.float32)
    
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logit_real, labels=tf.ones_like(self.D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logit_fake, labels=tf.zeros_like(self.D_logit_fake)))
        D_loss_class_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.D_class_logit_real, labels=self.real_c))
        D_loss_class_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.D_class_logit_real, labels=self.fake_c))
        self.D_loss = D_loss_real + D_loss_fake + D_loss_class_real + D_loss_class_fake
        G_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logit_fake, labels=tf.ones_like(self.D_logit_fake)))
        self.G_loss = D_loss_class_fake + G_loss_fake
        
    def _add_assignment_ops(self):
        tvars = tf.trainable_variables()
        tvar_names = [var.name for var in tvars]
        self.assign_ops = []
        for pair in param_names.GAN_PARAMS.VARIABLE_PAIRS:
            append = False
        #     if pair[1] == param_names.GEN_GRU_GATES_WEIGHTS or pair[1] == param_names.GEN_GRU_CANDIDATE_WEIGHTS:
        #         append = True
            self.assign_ops.append(utils.assign_variable_op(self.params, tvars, pair[0], pair[1], append=append))
        
    def _add_optimizer(self):
        tvars = tf.trainable_variables()
        theta_D = [var for var in tvars if 'D_' in var.name]
        theta_G = [var for var in tvars if 'G_' in var.name]
        theta_D.append(self.embedding_matrix)
        theta_G.append(self.embedding_matrix)
        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=theta_D)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=theta_G)
        for var in theta_D:
            utils.variable_summaries(var) 
            print var
        for var in theta_G:
            utils.variable_summaries(var) 
            print var
        
    def _add_saver_and_summary(self):
        self.global_step = tf.Variable(-1, name='global_step', trainable=False)
        self.saver = tf.train.Saver()
        for var in tf.trainable_variables():
            utils.variable_summaries(var)
        self.merged = tf.summary.merge_all()
        
    
    def build_graph(self, include_optimizer=True):
        self._add_placeholder()
        self._add_acgan()
        self._add_loss()
        self._add_assignment_ops()
        if include_optimizer:
            self._add_optimizer()
        self._add_saver_and_summary()

    


    

