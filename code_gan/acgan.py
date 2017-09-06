import numpy as np
import tensorflow as tf
from seq2seq import basic_rnn_seq2seq, rnn_decoder, embedding_rnn_decoder, sequence_loss
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
import matplotlib.pyplot as plt
import os
import time
import h5py
from docutils.nodes import generated
from generator_ac import generator
from discriminator_ac import discriminator
import param_names
import utils
import acgan_model
import argparse
from sklearn import metrics

prediction_file = '../predictions/predictions_acgan'
y_file = 'y_acgan.txt'
prediction_words_file = '../predictions/predictions_words_acgan'
summary_file = '/home/logan/tmp'
model_file = '../models/gan_model'
dataset_file = '../data/annotated_dataset.h5'
embedding_weights_file = '../data/embedding_weights.h5'
dictionary_file = '../data/words.dict'
# train_variables_file = '../models/tf_enc_dec_variables.npz'
train_variables_file = '../models/tf_lm_variables (copy).npz'
ckpt_dir = '../models/acgan_ckpts'
gan_variables_file = ckpt_dir + '/tf_acgan_variables_'
vague_terms_file = '../data/vague_terms'
use_checkpoint = False

# np.random.seed(123)

start_time = time.time()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('EPOCHS', 50,
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
tf.app.flags.DEFINE_integer('BATCH_SIZE', 64,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('NUM_CLASSES', 4,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_integer('CLASS_EMBEDDING_SIZE', 1,
                            'Max length for each sentence.')
tf.app.flags.DEFINE_string('CELL_TYPE', 'GRU',
                            'Which RNN cell for the RNNs.')
tf.app.flags.DEFINE_string('MODE', 'TRAIN',
                            'Whether to run in train or test mode.')

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
    test_x = data_file['test_X'][:]
    test_y = data_file['test_Y_sentence'][:]
print 'Number of training instances: ' + str(train_y[0])

print('loading dictionary')
d = {}
word_to_id = {}
with open(dictionary_file) as f:
    for line in f:
       (word, id) = line.split()
       d[int(id)] = word
       word_to_id[word] = int(id)
       
print('loading vague terms vector')
vague_terms = np.zeros((FLAGS.VOCAB_SIZE))
with open(vague_terms_file) as f:
    for line in f:
        words = line.split()
        if not len(words) == 1:
            print('excluded', words, 'because it is not 1 word:')
            continue
        word = words[0]
        if not word_to_id.has_key(word):
            print(word, 'is not in dictionary')
            continue
        id = word_to_id[word]
        if id >= vague_terms.shape[0]:
            print(word, 'is out of vocabulary')
            continue
        vague_terms[id] = 1




print train_x
for i in range(min(5, len(train_x))):
    for j in range(len(train_x[i])):
        if train_x[i][j] == 0:
            continue
        word = d[train_x[i][j]]
        print word + ' ',
    print '(' + str(train_y[i]) + ')\n'

def batch_generator(x, y, batch_size=FLAGS.BATCH_SIZE):
    data_len = x.shape[0]
    for i in range(0, data_len, batch_size):
        x_batch = x[i:min(i+batch_size,data_len)]
        x_batch_transpose = np.transpose(x_batch)
        x_batch_one_hot = np.eye(FLAGS.VOCAB_SIZE)[x_batch_transpose.astype(int)]
        x_batch_one_hot_reshaped = x_batch_one_hot.reshape([-1,FLAGS.SEQUENCE_LEN,FLAGS.VOCAB_SIZE])
        y_batch = y[i:min(i+batch_size,data_len)]
        yield x_batch_one_hot_reshaped, y_batch, i, data_len

'''
--------------------------------

MAIN

--------------------------------
'''

def save_samples_to_file(generated_sequences, batch_fake_c, epoch):
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
            f.write('(' + str(batch_fake_c[i]) + ')\n\n')
            
def sample_Z(m, n):
    return np.zeros((m, n))
#     return np.random.normal(size=[m, n])

def sample_C(m):
    return np.random.randint(low=0, high=FLAGS.NUM_CLASSES, size=m)

if not os.path.exists('out/'):
    os.makedirs('out/')
    
def train(model):
    print 'building graph'
    model.build_graph(include_optimizer=True)
    print 'training'
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
        tf.global_variables_initializer().run()
        model.assign_variables(sess)
        min_test_cost = np.inf
        num_mistakes = 0
        
        if use_checkpoint:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print ckpt.model_checkpoint_path
                model.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
    
        start = model.get_global_step() + 1 # get last global_step and start the next one
        print "Start from:", start
        
        batch_x, batch_y, _, _ = batch_generator(train_x, train_y).next()
        batch_fake_c = np.zeros([FLAGS.BATCH_SIZE], dtype=np.int32)
        batch_z = sample_Z(FLAGS.BATCH_SIZE, FLAGS.LATENT_SIZE)
        batch_samples = model.run_samples(sess, batch_fake_c, batch_z)
        save_samples_to_file(batch_samples, batch_fake_c, 'pre')
        
        xaxis = 0
        step = 0
        for cur_epoch in range(start, FLAGS.EPOCHS):
            for batch_x, batch_y, cur, data_len in batch_generator(train_x, train_y):
                batch_z = sample_Z(batch_x.shape[0], FLAGS.LATENT_SIZE)
                batch_fake_c = sample_C(batch_x.shape[0])
                _, D_loss_curr, real_acc, fake_acc, real_class_acc, fake_class_acc, summary = model.run_D_train_step(
                    sess, batch_x, batch_y, batch_z, batch_fake_c)
                for j in range(1):
                    _, G_loss_curr, batch_samples, batch_probs = model.run_G_train_step(
                        sess, batch_x, batch_y, batch_z, batch_fake_c)
        
                if cur_epoch % 1 == 0:
                    train_writer.add_summary(summary, step)
                    step += 1
                    generated_sequences = batch_samples
                    print('Iter: {}'.format(cur_epoch))
                    print('Instance ', cur, ' out of ', data_len)
                    print('D loss: {:.4}'. format(D_loss_curr))
                    print('G_loss: {:.4}'.format(G_loss_curr))
                    print('D real acc: ', real_acc, ' D fake acc: ', fake_acc)
                    print('D real class acc: ', real_class_acc, ' D fake class acc: ', fake_class_acc)
                    print('Samples', generated_sequences)
                    print()
                    for i in range(min(3, len(generated_sequences))):
                        for j in range(len(generated_sequences[i])):
                            if generated_sequences[i][j] == 0:
                                print '<UNK> ',
                            else:
                                word = d[generated_sequences[i][j]]
                                print word + ' ',
                        print '(' + str(batch_fake_c[i]) + ')\n'
                    
            if cur_epoch % 1 == 0:
                save_samples_to_file(generated_sequences, batch_fake_c, cur_epoch)
            
            print 'saving model to file:'
    #         global_step.assign(cur_epoch).eval() # set and update(eval) global_step with index, cur_epoch
            model.set_global_step(cur_epoch)
    #         saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
            model.saver.save(sess, ckpt_dir + "/model.ckpt", global_step=cur_epoch)
    #         vars = sess.run(tvars)
            vars = model.get_variables(sess)
            tvar_names = [var.name for var in tf.trainable_variables()]
            variables = dict(zip(tvar_names, vars))
            np.savez(gan_variables_file + str(cur_epoch), **variables)
            
        train_writer.close()
            
        save_samples_to_file(generated_sequences, batch_fake_c, cur_epoch)
    
    
def test(model):
    print 'building graph'
    model.build_graph(include_optimizer=False)
    print 'testing'
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            model.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        predictions = []
        for batch_x, batch_y, cur, data_len in batch_generator(test_x, test_y, batch_size=1):
#         for batch_x, batch_y, cur, data_len in batch_generator(test_x, test_y):
            batch_predictions = model.run_test(sess, batch_x)
            predictions.append(batch_predictions)
            print('Instance ', cur, ' out of ', data_len)
        predictions = np.concatenate(predictions)
        predictions_indices = np.argmax(predictions, axis=1)
        print ('Accuracy', metrics.accuracy_score(test_y, predictions_indices))
        print metrics.classification_report(test_y,predictions_indices)
        a=1
        
    
def main(unused_argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="run in train mode",
                        action="store_true")
    args = parser.parse_args()
    model = acgan_model.ACGANModel(vague_terms, params)
    
    if args.train:
        train(model)
    else:
        test(model)
    
            
    print('Execution time: ', (time.time() - start_time)/3600., ' hours')
    
if __name__ == '__main__':
  tf.app.run()
    
    
    
    