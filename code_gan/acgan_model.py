import numpy as np
import tensorflow as tf
from generator_ac import generator
from discriminator_ac import discriminator
import param_names
import utils


FLAGS = tf.app.flags.FLAGS

class ACGANModel(object):
    
    def __init__(self, vague_terms, params):
        """ init the model with hyper-parameters etc """
        self.vague_terms = vague_terms
        self.params = params
        self.is_built = False
        
    def run_D_train_step(self, sess, batch_x, batch_c, z, batch_fake_c):
        to_return = [self.D_solver, self.D_loss, self.D_real_acc, self.D_fake_acc, 
            self.D_real_class_acc, self.D_fake_class_acc, self.D_loss_real, 
            self.D_loss_fake, self.D_loss_class_real, self.D_loss_class_fake]
        return sess.run(to_return,
                    feed_dict={self.real_x: batch_x,
                               self.real_c: batch_c,
                               self.fake_c: batch_fake_c,
                               self.z: z,
                               self.keep_prob: FLAGS.KEEP_PROB})
    
    def run_G_train_step(self, sess, z, batch_fake_c):
        to_return = [self.G_solver, self.G_loss, self.samples, self.probs, self.merged, 
                     self.logits, self.pure_logits, self.inps]
        return sess.run(to_return,
                    feed_dict={self.fake_c: batch_fake_c,
                               self.z: z,
                               self.keep_prob: FLAGS.KEEP_PROB})
        
    def run_test(self, sess, batch_x):
        to_return = self.D_class_logit_real
        return sess.run(to_return,
                        feed_dict={self.real_x: batch_x,
                               self.keep_prob: 1})
        
    def run_val(self, sess, batch_x, batch_y):
        to_return = self.D_loss_class_real
        return sess.run(to_return,
                    feed_dict={self.real_x: batch_x,
                               self.real_c: batch_y,
                               self.keep_prob: 1})
        
    def run_samples(self, sess, batch_fake_c, batch_z):
        return sess.run(self.samples, feed_dict={self.fake_c: batch_fake_c,
                                                 self.z: batch_z,
                                                 self.keep_prob: 1})
        
    def run_summary(self, sess):
        return sess.run(self.merged)
        
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
        self.z = tf.placeholder(tf.float32, [None, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE], name='z')
        
        # Initialization doesn't matter here, since the embedding matrix is
        # being replaced with the pretrained parameters
        self.embedding_matrix = tf.get_variable(shape=[FLAGS.VOCAB_SIZE, FLAGS.EMBEDDING_SIZE],
                            initializer=tf.contrib.layers.xavier_initializer(), name='embedding_matrix')
        self.gumbel_mu = tf.get_variable(name='gumbel_mu', initializer=tf.constant(0.), trainable=False)
        self.gumbel_sigma = tf.get_variable(name='gumbel_sigma', initializer=tf.constant(1.), trainable=False)
        self.keep_prob = tf.placeholder(tf.float32)
        
    def _add_acgan(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            self.G_sample, self.samples, self.probs, self.u, self.m, self.logits, self.pure_logits, self.vague_weights, self.inps = generator(self.z, self.fake_c, self.vague_terms,
                 self.embedding_matrix, self.keep_prob, self.gumbel_mu, self.gumbel_sigma) #TODO move to generator
            self.D_real, self.D_logit_real, self.D_class_logit_real = discriminator(self.real_x, 
                 self.embedding_matrix, self.keep_prob)
            tf.get_variable_scope().reuse_variables()
            if FLAGS.SAMPLE:
                self.D_fake, self.D_logit_fake, self.D_class_logit_fake = discriminator(self.samples,
                     self.embedding_matrix, self.keep_prob)
            else:
                self.D_fake, self.D_logit_fake, self.D_class_logit_fake = discriminator(self.G_sample,
                     self.embedding_matrix, self.keep_prob)
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
    
#         self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=self.D_logit_real, labels=tf.ones_like(self.D_logit_real)))
#         self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=self.D_logit_fake, labels=tf.zeros_like(self.D_logit_fake)))
        self.D_loss_real = -tf.reduce_mean(tf.log(tf.sigmoid(self.D_logit_real)))
        self.D_loss_fake = -tf.reduce_mean(tf.log(1-tf.sigmoid(self.D_logit_fake)))
#         ones = tf.ones(tf.stack([tf.shape(self.D_logit_real)[0],]), dtype=tf.int32)
#         zeros = tf.zeros(tf.stack([tf.shape(self.D_logit_fake)[0],]), dtype=tf.int32)
#         self.D_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#             logits=self.D_logit_real, labels=ones))
#         self.D_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#             logits=self.D_logit_fake, labels=zeros))
        self.D_loss_class_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.D_class_logit_real, labels=self.real_c))
        self.D_loss_class_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.D_class_logit_fake, labels=self.fake_c))
        self.D_loss = FLAGS.SOURCE_LOSS_WEIGHT*(self.D_loss_real + self.D_loss_fake) + (
            self.D_loss_class_real)*FLAGS.REAL_CLASS_LOSS_WEIGHT + self.D_loss_class_fake*FLAGS.FAKE_CLASS_LOSS_WEIGHT
#         self.G_loss_fake = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=self.D_logit_fake, labels=tf.ones_like(self.D_logit_fake)))
        self.G_loss_fake = -tf.reduce_mean(tf.log(tf.sigmoid(self.D_logit_fake)))
        self.G_loss = FLAGS.FAKE_CLASS_LOSS_WEIGHT*self.D_loss_class_fake + FLAGS.SOURCE_LOSS_WEIGHT*self.G_loss_fake
        
        tvars   = tf.trainable_variables() 
        theta_D = [var for var in tvars if 'D_' in var.name]
        theta_G = [var for var in tvars if 'G_' in var.name]
#         theta_D.append(self.embedding_matrix)
#         theta_G.append(self.embedding_matrix)
        D_lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in theta_D if 'bias' not in v.name ]) * FLAGS.L2_LAMBDA
        G_lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in theta_G if 'bias' not in v.name ]) * FLAGS.L2_LAMBDA
        self.D_loss += D_lossL2
        self.G_loss += G_lossL2
        
    def _add_assignment_ops(self):
        tvars = tf.trainable_variables()
        tvar_names = [var.name for var in tvars]
        self.assign_ops = []
        if FLAGS.CELL_TYPE == 'LSTM':
            gan_params = param_names.GAN_LSTM_PARAMS
        else:
            gan_params = param_names.GAN_PARAMS
        for pair in gan_params.VARIABLE_PAIRS:
            append = False
        #     if pair[1] == param_names.GEN_GRU_GATES_WEIGHTS or pair[1] == param_names.GEN_GRU_CANDIDATE_WEIGHTS:
        #         append = True
            self.assign_ops.append(utils.assign_variable_op(self.params, pair[0], pair[1], append=append))
        
    def _add_optimizer(self):
        tvars = tf.trainable_variables()
        theta_D = [var for var in tvars if 'D_' in var.name]
        theta_G = [var for var in tvars if 'G_' in var.name]
#         theta_G = [var for var in tvars if 'G_' in var.name or 'D_' in var.name]
        if FLAGS.TRAIN_EMBEDDING:
            theta_D.append(self.embedding_matrix)
            theta_G.append(self.embedding_matrix)
        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=theta_D)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=theta_G)
        for var in theta_D:
            print (var)
        for var in theta_G:
            print (var)
        
    def _add_saver_and_summary(self):
        self.global_step = tf.Variable(-1, name='global_step', trainable=False)
        utils.variable_summaries(tf.trainable_variables())
        self.merged = tf.summary.merge_all()
        
    
    def build_graph(self, include_optimizer=True):
        self._add_placeholder()
        self._add_acgan()
        self._add_loss()
        self._add_assignment_ops()
        if include_optimizer:
            self._add_optimizer()
        self._add_saver_and_summary()
        self.is_built = True

    


    

