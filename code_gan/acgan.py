import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq, rnn_decoder, embedding_rnn_decoder, sequence_loss
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import matplotlib.pyplot as plt
import os
from metrics import performance
import time
import h5py
from docutils.nodes import generated
from generator_ac import generator
from discriminator_ac import discriminator
import param_names
import utils

prediction_file = '../predictions/predictions_gan'
y_file = 'y.txt'
prediction_words_file = '../predictions/predictions_words_gan'
summary_file = '/home/logan/tmp'
model_file = '../models/gan_model'
dataset_file = '../data/annotated_dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
dictionary_file = '../data/words.dict'
# train_variables_file = '../models/tf_enc_dec_variables.npz'
train_variables_file = '../models/tf_lm_variables (copy).npz'
ckpt_dir = '../models/gan_ckpts'
gan_variables_file = '../models/tf_gan_variables.npz'
start_symbol_index = 2
use_checkpoint = False

# np.random.seed(123)

start_time = time.time()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('EPOCHS', 5000,
                            'Num epochs.')
tf.app.flags.DEFINE_integer('VOCAB_SIZE', 5000,
                            'Number of words in the vocabulary.')
tf.app.flags.DEFINE_integer('LATENT_SIZE', 512,
                            'Size of both the hidden state of RNN and random vector z.')
tf.app.flags.DEFINE_integer('SEQUENCE_LEN', 50,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('EMBEDDING_SIZE', 300,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('PATIENCE', 200,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('BATCH_SIZE', 128,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('NUM_CLASSES', 4,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('CLASS_EMBEDDING_SIZE', 100,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_string('CELL_TYPE', 'GRU',
                            'Which RNN cell for the RNNs.')


'''
--------------------------------

LOAD DATA

--------------------------------
'''

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

print('loading model parameters')
params = np.load(train_variables_file)

print('loading embedding weights')
with h5py.File(embedding_weights_file, 'r') as hf:
    embedding_weights = hf['embedding_weights'][:]
    
print('loading training and test data')
with h5py.File(dataset_file, 'r') as data_file:
    train_x = data_file['train_X'][:]
    train_y = data_file['train_Y_sentence'][:]
    

print('loading dictionary')
d = {}
vocab_size = 0
with open(dictionary_file) as f:
    for line in f:
       (val, key) = line.split()
       d[int(key)] = val
       vocab_size += 1


print train_x
for i in range(min(5, len(train_x))):
    for j in range(len(train_x[i])):
        if train_x[i][j] == 0:
            continue
        word = d[train_x[i][j]]
        print word + ' ',
    print '(' + str(train_y[i]) + ')\n'

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def batch_generator(x, y):
    data_len = x.shape[0]
    for i in range(0, data_len, FLAGS.BATCH_SIZE):
        x_batch = x[i:min(i+FLAGS.BATCH_SIZE,data_len)]
        x_batch_transpose = np.transpose(x_batch)
        x_batch_one_hot = np.eye(FLAGS.VOCAB_SIZE)[x_batch_transpose.astype(int)]
        y_batch = y[i:min(i+FLAGS.BATCH_SIZE,data_len)]
        yield x_batch_one_hot, y_batch, i, data_len

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean ' + var.name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev ' + var.name, stddev)
    tf.summary.scalar('max ' + var.name, tf.reduce_max(var))
    tf.summary.scalar('min ' + var.name, tf.reduce_min(var))
    tf.summary.histogram('histogram ' + var.name, var)


'''
--------------------------------

MODEL

--------------------------------
'''
real_x = [tf.placeholder(tf.float32, shape=[None, FLAGS.VOCAB_SIZE]) for i in xrange(FLAGS.SEQUENCE_LEN)]
real_c = tf.placeholder(tf.int32, [None,], 'class')
fake_c = tf.placeholder(tf.int32, [None,], 'class')
z = tf.placeholder(tf.float32, [None, FLAGS.LATENT_SIZE], name='z')
dims = tf.stack([tf.shape(real_x[0])[0],])
start_symbol_input = [tf.fill(dims, start_symbol_index) for i in xrange(FLAGS.SEQUENCE_LEN)]

def sample_Z(m, n):
    return np.zeros((m, n))
#     return np.random.normal(size=[m, n])

def sample_C(m):
    return np.random.randint(low=0, high=FLAGS.NUM_CLASSES, size=m)

    
with tf.variable_scope(tf.get_variable_scope()) as scope:
    G_sample, samples, probs = generator(z, fake_c, start_symbol_input)
    D_real, D_logit_real, D_class_logit_real = discriminator(real_x)
    tf.get_variable_scope().reuse_variables()
    D_fake, D_logit_fake, D_class_logit_fake = discriminator(G_sample)
    D_real_acc = tf.cast(tf_count(tf.round(D_real), 1), tf.float32) / tf.cast(tf.shape(D_real)[0], tf.float32)
    D_fake_acc = tf.cast(tf_count(tf.round(D_fake), 0), tf.float32) / tf.cast(tf.shape(D_fake)[0], tf.float32)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss_class_real = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_class_logit_real, labels=real_c)
    D_loss_class_fake = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_class_logit_real, labels=fake_c)
    D_loss = D_loss_real + D_loss_fake + D_loss_class_real + D_loss_class_fake
    G_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    G_loss = D_loss_class_fake + G_loss_fake

tvars = tf.trainable_variables()
tvar_names = [var.name for var in tvars]
assign_ops = []
for pair in param_names.GAN_PARAMS.VARIABLE_PAIRS:
    append = False
    if pair[1] == param_names.GEN_GRU_GATES_WEIGHTS or pair[1] == param_names.GEN_GRU_CANDIDATE_WEIGHTS:
        append = True
    assign_ops.append(utils.assign_variable_op(params, tvars, pair[0], pair[1], append=append))

theta_D = [var for var in tvars if 'D_' in var.name]
theta_G = [var for var in tvars if 'G_' in var.name]
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(-G_loss, var_list=theta_G)
for var in theta_D:
    variable_summaries(var) 
    print var
for var in theta_G:
    variable_summaries(var) 
    print var
    
global_step = tf.Variable(-1, name='global_step', trainable=False)
saver = tf.train.Saver()
    
merged = tf.summary.merge_all()


'''
--------------------------------

MAIN

--------------------------------
'''

def save_samples_to_file(generated_sequences, epoch):
    with open(prediction_file + '_epoch_' + str(epoch), 'w') as f:
        np.savetxt(f, generated_sequences, fmt='%d\t')
    with open(prediction_words_file + '_epoch_' + str(epoch), 'w') as f:
        for i in range(len(generated_sequences)):
            for j in range(len(generated_sequences[i])):
                if generated_sequences[i][j] == 0:
                    f.write('<UNK> ')
                else:
                    word = d[generated_sequences[i][j]]
                    f.write(word + ' ')
            f.write('\n')

if not os.path.exists('out/'):
    os.makedirs('out/')

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
    tf.global_variables_initializer().run()
    sess.run(assign_ops)
    min_test_cost = np.inf
    num_mistakes = 0
    
    if use_checkpoint:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

    start = global_step.eval() + 1 # get last global_step and start the next one
    print "Start from:", start
    
    batch_x, batch_y, _, _ = batch_generator(train_x, train_y).next()
    train_dict = {i: d for i, d in zip(real_x, batch_x)}
    train_dict[real_c] = batch_y
    train_dict[z] = sample_Z(batch_x.shape[1], FLAGS.LATENT_SIZE)
    train_dict[fake_c] = sample_C(batch_x.shape[1])
    batch_samples = sess.run(samples, feed_dict=train_dict)
    save_samples_to_file(batch_samples, 'pre')
    
    xaxis = 0
    step = 0
    for cur_epoch in range(FLAGS.EPOCHS):
        for batch_x, batch_y, cur, data_len in batch_generator(train_x, train_y):
            train_dict = {i: d for i, d in zip(real_x, batch_x)}
            train_dict[real_c] = batch_y
            train_dict[z] = sample_Z(batch_x.shape[1], FLAGS.LATENT_SIZE)
            train_dict[fake_c] = sample_C(batch_x.shape[1])
        
            D_loss_curr, real_acc, fake_acc, summary = sess.run([D_loss, D_real_acc, D_fake_acc, merged], feed_dict=train_dict)
            for j in range(1):
                batch_samples, batch_probs, G_loss_curr = sess.run([samples, probs, G_loss], feed_dict=train_dict)
    
            if cur_epoch % 1 == 0:
                train_writer.add_summary(summary, step)
                step += 1
                generated_sequences = batch_samples
                print('Iter: {}'.format(cur_epoch))
                print('Instance ', cur, ' out of ', data_len)
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print('D real acc: ', real_acc, ' D fake acc: ', fake_acc)
                print('Samples', generated_sequences)
                print()
                for i in range(min(3, len(generated_sequences))):
                    for j in range(len(generated_sequences[i])):
                        if generated_sequences[i][j] == 0:
                            print '<UNK> ',
                        else:
                            word = d[generated_sequences[i][j]]
                            print word + ' ',
                    print '\n'
                
        if cur_epoch % 1 == 0:
            save_samples_to_file(generated_sequences, cur_epoch)
        
        print 'saving model to file:'
        global_step.assign(cur_epoch).eval() # set and update(eval) global_step with index, i
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
        vars = sess.run(tvars)
        variables = dict(zip(tvar_names, vars))
        np.savez(gan_variables_file, **variables)
        
    train_writer.close()
        
    save_samples_to_file(generated_sequences, cur_epoch)
        
print('Execution time: ', time.time() - start_time)
    
    
    
    
    