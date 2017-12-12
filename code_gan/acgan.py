from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import time
import h5py
import utils
import load
import param_names
import acgan_model
import argparse
import sys
from tqdm import tqdm

prediction_folder = os.path.join('..','predictions')
prediction_words_file = os.path.join('predictions_words_acgan')
summary_file = os.path.join('/home','logan','tensorboard')
# train_variables_file = os.path.join('../models/tf_enc_dec_variables.npz')
models_folder = os.path.join('..','models')
default_model_name = 'acgan'
default_lm_name = 'lm_l2_1e-4'
num_folds = 5

start_time = time.time()

FLAGS = tf.app.flags.FLAGS


parser = argparse.ArgumentParser()
parser.add_argument("--train_only", help="run in train mode only",
                    action="store_true")
parser.add_argument("--test_only", help="run in test mode only",
                    action="store_true")
parser.add_argument("--one_fold", help="perform only on one fold instead of five-fold cross validation",
                    action="store_true")
parser.add_argument("--resume", help="Resume training from last saved checkpoint",
                    action="store_true")
parser.add_argument('--name', default=default_model_name, help='Optional name of model, and affects where to save model')
parser.add_argument('--lm_name', default=default_lm_name, help='Optional name of language model, which tells where to load LM parameters')  
parser.add_argument('--EPOCHS', default=50, type=int,
                            help='Num epochs.')
parser.add_argument('--VOCAB_SIZE', default=10000, type=int,
                            help='Number of words in the vocabulary.')
parser.add_argument('--LATENT_SIZE', default=512, type=int,
                            help='Size of both the hidden state of RNN and random vector z.')
parser.add_argument('--SEQUENCE_LEN', default=50, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--EMBEDDING_SIZE', default=300, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--PATIENCE', default=5, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--BATCH_SIZE', default=64, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--NUM_CLASSES', default=4, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--CLASS_EMBEDDING_SIZE', default=1, type=int,
                            help='Max length for each sentence.')
parser.add_argument('--CELL_TYPE', default='LSTM', type=str,
                            help='Which RNN cell for the RNNs.')
parser.add_argument('--MODE', default='TRAIN', type=str,
                            help='Whether to run in train or test mode.')
parser.add_argument('--RANDOM_SEED', default=123, type=int,
                            help='Random seed used for numpy and tensorflow (dropout, sampling)')
parser.add_argument('--USE_CNN', default=True,
                            help='Whether to use CNN or RNN')
parser.add_argument('--FILTER_SIZES', default='3,4,5', type=str,
                            help="Comma-separated filter sizes (default: '3,4,5')")
parser.add_argument('--NUM_FILTERS', default=128,  type=int,
                            help='Number of filters per filter size (default: 128)')
parser.add_argument('--KEEP_PROB', default=0.5, type=float,
                            help='Dropout probability of keeping a node')
parser.add_argument('--SOURCE_LOSS_WEIGHT', default=1, type=float,
                            help='How much importance that fake/real contributes to the total loss.')
parser.add_argument('--REAL_CLASS_LOSS_WEIGHT', default=1, type=float,
                            help='How much importance that real instances\' class loss contributes to the total loss.')
parser.add_argument('--FAKE_CLASS_LOSS_WEIGHT', default=1, type=float,
                            help='How much importance that real instances\' class loss contributes to the total loss.')
parser.add_argument('--USE_VAGUE_VECTOR', default=True, type=bool,
                            help='Use regular GAN without class. Ignores class loss and vague terms vector.')
parser.add_argument('--VANILLA_GAN', default=False, type=bool,
                            help='Use regular GAN without class. Ignores class loss and vague terms vector.')
parser.add_argument('--SHARE_EMBEDDING', default=True, type=bool,
                            help='Whether the discriminator and generator should share their embedding parameters')
parser.add_argument('--TRAIN_EMBEDDING', default=False, type=bool,
                            help='Whether the embedding parameters should be trained')
parser.add_argument('--TRAIN_GENERATOR', default=True, type=bool,
                            help='Whether to train the generator\'s parameters')
parser.add_argument('--L2_LAMBDA', default=1e-4, type=float,
                            help='L2 regularization lambda parameter')
parser.add_argument('--CHECKPOINT', default=-1, type=int,
                            help='Which checkpoint model to load when running on test set. -1 means the last model.')
parser.add_argument('--GUMBEL', default=True, type=bool,
                            help='Whether to use Gumbel-Softmax Relaxation')
parser.add_argument('--TAU', default=0.5, type=float,
                            help='Tau parameter used for gumbel softmax relaxation technique. Used only if GUMBEL=True.')
parser.add_argument('--PRINT_SAMPLES', default=True, type=bool,
                            help='Whether to print generated samples')
parser.add_argument('--TRUNCATE_GENERATED_EOS', default=True, type=bool,
                            help='Whether to truncate all words after End Of Sentence symbol')
parser.add_argument('--MASK_OUT_EOS_SYMBOL', default=True, type=bool,
                            help='Whether to truncate all words after End Of Sentence symbol')
parser.add_argument('--RANDOM_START_WORD_INSTEAD_OF_GUMBEL', default=False, type=bool,
                            help='Replaces start of sentence symbol with a random word, and turns off gumbel')
args = parser.parse_args()

if args.VANILLA_GAN:
    args.SOURCE_LOSS_WEIGHT = 1.
    args.REAL_CLASS_LOSS_WEIGHT = 0.
    args.FAKE_CLASS_LOSS_WEIGHT = 0.
    args.USE_VAGUE_VECTOR = False
    args.EPOCHS = 5

for arg_name, arg_value in vars(args).iteritems():
    if type(arg_value) == bool:
        tf.app.flags.DEFINE_boolean(arg_name, arg_value, docstring='')
    elif type(arg_value) == int:
        tf.app.flags.DEFINE_integer(arg_name, arg_value, docstring='')
    elif type(arg_value) == float:
        tf.app.flags.DEFINE_float(arg_name, arg_value, docstring='')
    elif type(arg_value) == str:
        tf.app.flags.DEFINE_string(arg_name, arg_value, docstring='')
        
tf.set_random_seed(FLAGS.RANDOM_SEED)
np.random.seed(FLAGS.RANDOM_SEED)
''' Store model using sampling in a different location ''' 
print('sw ', FLAGS.SOURCE_LOSS_WEIGHT, 'rcw', FLAGS.REAL_CLASS_LOSS_WEIGHT, 
      'fcw', FLAGS.FAKE_CLASS_LOSS_WEIGHT, 'uvv', FLAGS.USE_VAGUE_VECTOR)


'''
--------------------------------

LOAD DATA

--------------------------------
'''
ckpt_dir = os.path.join(models_folder, args.name)

''' Make directories for model files and prediction files '''
utils.create_dirs(ckpt_dir, num_folds)
utils.create_dirs(os.path.join(prediction_folder, FLAGS.name), num_folds)

params = load.load_pretrained_params(FLAGS.lm_name)
d, word_to_id = load.load_dictionary()
vague_terms = load.load_vague_terms_vector(word_to_id, FLAGS.VOCAB_SIZE)

Metrics = utils.Metrics()

'''
--------------------------------

MAIN

--------------------------------
'''

def save_samples_to_file(generated_sequences, batch_fake_c, fold_num, epoch):
    fold_prediction_dir = os.path.join(prediction_folder, FLAGS.name, str(fold_num))
    file_name = os.path.join(fold_prediction_dir, prediction_words_file + '_epoch_' + str(epoch))
    with open(file_name, 'w') as f:
        for i in range(len(generated_sequences)):
            for j in range(len(generated_sequences[i])):
                if generated_sequences[i][j] == 0:
                    f.write('_ ')
                else:
                    word = d[generated_sequences[i][j]]
                    f.write(word + ' ')
            f.write('(' + str(batch_fake_c[i]) + ')\n\n')
            
def sample_Z(size):
    return np.random.gumbel(size=size)
#     return np.random.normal(size=[m, n], scale=FLAGS.NOISE_STD_DEV)

def sample_C(m):
    return np.random.randint(low=0, high=FLAGS.NUM_CLASSES, size=m)

def feed_dict(model, batch_x, batch_y, batch_z, batch_fake_c, batch_weights):
    return {model.real_x: batch_x,
                               model.real_c: batch_y,
                               model.fake_c: batch_fake_c,
                               model.z: batch_z,
                               model.keep_prob: FLAGS.KEEP_PROB,
                               model.real_weights: batch_weights}

def validate(sess, model, val_x, val_y, val_weights):
    sum_cost = 0
    for batch_x, batch_y, batch_weights, cur, data_len in utils.batch_generator(val_x, val_y, val_weights, one_hot=False, actually_zero=True):
        batch_cost = model.run_val(sess, batch_x, batch_y, batch_weights)
        sum_cost += batch_cost * len(batch_y)  # Add up cost based on how many instances in batch
    val_cost = sum_cost / len(val_y)
    return val_cost
    
def train(model, train_x, train_y, train_weights, val_x, val_y, val_weights, fold_num):
    print ('building graph')
    if not model.is_built:
        model.build_graph(include_optimizer=True)
    print ('training')
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(os.path.join(summary_file, FLAGS.name), sess.graph)
        saver = tf.train.Saver(max_to_keep=5)
        tf.global_variables_initializer().run()
        model.assign_variables(sess)
        
        fold_ckpt_dir = os.path.join(ckpt_dir,str(fold_num))
        gan_variables_file = os.path.join(fold_ckpt_dir,'tf_acgan_variables_')
        if FLAGS.resume:
            ckpt = tf.train.get_checkpoint_state(fold_ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print (ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
    
        start = model.get_global_step() + 1 # get last global_step and start the next one
        print ("Start from:", start)
        
        batch_x, batch_y, batch_weights, _, _ = utils.batch_generator(train_x, train_y, weights=train_weights, one_hot=False, actually_zero=True).next()
        batch_fake_c = np.zeros([FLAGS.BATCH_SIZE], dtype=np.int32)
        batch_z = sample_Z([FLAGS.BATCH_SIZE, FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
        batch_samples = model.run_samples(sess, batch_fake_c, batch_z)
        summary = model.run_summary(sess)
        train_writer.add_summary(summary, -1)
        save_samples_to_file(batch_samples, batch_fake_c, fold_num, 'pre')
        
        step = 0
        min_val_cost = np.inf
        num_mistakes = 0
        for cur_epoch in tqdm(range(start, FLAGS.EPOCHS), desc='Fold ' + str(fold_num) + ': Epoch', total=FLAGS.EPOCHS-start):
            disc_steps = 1
            step_ctr = 0
            for batch_x, batch_y, batch_weights, _, _ in tqdm(utils.batch_generator(train_x, train_y, weights=train_weights, one_hot=False, actually_zero=True), desc='Batch', total=len(train_y)/FLAGS.BATCH_SIZE):
                batch_z = sample_Z([batch_x.shape[0], FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
                batch_fake_c = sample_C(batch_x.shape[0])
                for j in range(1):
                    _, D_loss_curr, real_acc, fake_acc, real_class_acc, fake_class_acc, real_loss, fake_loss, real_class_loss, fake_class_loss = model.run_D_train_step(
                        sess, batch_x, batch_y, batch_z, batch_fake_c, batch_weights)
                step_ctr += 1
                if step_ctr == disc_steps:
                    step_ctr = 0
                    if FLAGS.TRAIN_GENERATOR:
                        for j in range(1):
                            batch_z = sample_Z([batch_x.shape[0], FLAGS.SEQUENCE_LEN, FLAGS.VOCAB_SIZE])
                            g_batch_fake_c = sample_C(batch_x.shape[0])
                            _, G_loss_curr, batch_samples, batch_probs, summary, logits, pure_logits, inps, outputs  = model.run_G_train_step(
                                sess, batch_z, g_batch_fake_c)
                            generated_sequences = batch_samples
                    else:
                        summary = model.run_summary(sess) 
                    
                    if FLAGS.PRINT_SAMPLES:
                        out = ''
                        out += 'Fold: {:}\t'. format(fold_num)
                        out += 'Epoch: {:}\t'. format(cur_epoch)
                        out += 'D loss: {:.4}\t'. format(D_loss_curr)
                        if FLAGS.TRAIN_GENERATOR:
                            out += 'G_loss: {:.4}\t'.format(G_loss_curr)
                        out += 'Real S acc: {:.4}\t'.format(real_acc) 
                        out += ' Fake S acc: {:.4}\t'.format(fake_acc)
                        out += 'Real C acc: {:.4}\t'.format(real_class_acc)
                        out += ' Fake C acc: {:.4}\n'.format(fake_class_acc)
                        tqdm.write(out)
    #                     out += 'Samples', generated_sequences)
                        if FLAGS.TRAIN_GENERATOR:
                            out = ''
                            for i in range(min(6, len(generated_sequences))):
                                for j in range(len(generated_sequences[i])):
                                    if generated_sequences[i][j] == 0:
                                        out += '_ '
                                    else:
                                        word = d[generated_sequences[i][j]]
                                        out += word + ' '
                                out += '(' + str(g_batch_fake_c[i]) + ')\n\n'
                            tqdm.write(out)
                    projection_w = utils.eval_variable(param_names.GEN_OUTPUT_WEIGHTS)
                    projection_b = utils.eval_variable(param_names.GEN_OUTPUT_BIASES)
                    emb = utils.eval_variable(param_names.GAN_EMBEDDING)
                    out = 'Proj W: {:.6}\t'.format(np.mean(np.absolute(projection_w)))
                    out += 'Proj B: {:.6}\t'.format(np.mean(np.absolute(projection_b)))
                    out += 'Emb: {:.6}\t'.format(np.mean(np.absolute(emb)))
                    tqdm.write(out)
                        
                    if step % 10 == 0:
                        train_writer.add_summary(summary, step)
                    step += 1
            if cur_epoch % 1 == 0:
                if FLAGS.TRAIN_GENERATOR:
                    save_samples_to_file(generated_sequences, g_batch_fake_c, fold_num, cur_epoch)
            
            print ('saving model to file:')
            model.set_global_step(cur_epoch)
            saver.save(sess, os.path.join(fold_ckpt_dir,'model.ckpt'), global_step=cur_epoch)
            vars = model.get_variables(sess)
            tvar_names = [var.name for var in tf.trainable_variables()]
            variables = dict(zip(tvar_names, vars))
            np.savez(gan_variables_file, **variables)
            
            val_cost = validate(sess, model, val_x, val_y, val_weights)
            print('Val Loss: ', val_cost)
            if val_cost < min_val_cost:
                min_val_cost = val_cost
            else:
                num_mistakes += 1
            if num_mistakes >= FLAGS.PATIENCE:
                print ('Stopping early at epoch: ', cur_epoch)
                break
            
        train_writer.close()
    
def test(model, test_x, test_y, test_weights, fold_num):
    print ('building graph')
    if not model.is_built:
        model.build_graph(include_optimizer=False)
    print ('testing')
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)
        tf.global_variables_initializer().run()
        fold_ckpt_dir = os.path.join(ckpt_dir,str(fold_num))
        if FLAGS.CHECKPOINT == -1:
            ckpt = tf.train.get_checkpoint_state(fold_ckpt_dir)
            if not ckpt:
                raise Exception('Could not find saved model in: ' + fold_ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print (ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        else:
            ckpt_model_path = os.path.join(fold_ckpt_dir, 'model.ckpt-'+str(FLAGS.CHECKPOINT))
            print (ckpt_model_path)
            saver.restore(sess, ckpt_model_path) # restore all variables
        predictions = []
        for batch_x, batch_y, batch_weights, _, _ in tqdm(utils.batch_generator(test_x, test_y, test_weights, batch_size=1, one_hot=False, actually_zero=True), desc='Fold ' + str(fold_num) + ':Testing', total=len(test_y)):
            batch_predictions = model.run_test(sess, batch_x, batch_weights)
            predictions.append(batch_predictions)
        predictions = np.concatenate(predictions)
        predictions_indices = np.argmax(predictions, axis=1)
        Metrics.print_and_save_metrics(test_y, predictions_indices)
    
        
def run_on_fold(mode, model, fold_num):
    train_x, _, train_y, train_weights, val_x, _, val_y, val_weights, test_x, _, test_y, test_weights = load.load_annotated_data(fold_num)
    if mode == 'train':
        train(model, train_x, train_y, train_weights, val_x, val_y, val_weights, fold_num)
    else:
        test(model, test_x, test_y, test_weights, fold_num)
        
def run_in_mode(model, mode, one_fold):
    if one_fold:
        run_on_fold(mode, model, 0)
    else:
        for fold_num in range(num_folds):
            run_on_fold(mode, model, fold_num)
    if mode == 'test':
        Metrics.print_metrics_for_all_folds()
    
def main(unused_argv):
    if args.train_only and args.test_only: raise Exception('provide only one mode')
    train = args.train_only or not args.test_only
    test = not args.train_only or args.test_only
    model = acgan_model.ACGANModel(vague_terms, params)
    if train:
        run_in_mode(model, 'train', args.one_fold)
    if test:
        run_in_mode(model, 'test', args.one_fold)
        

    localtime = time.asctime( time.localtime(time.time()) )
    print ("Finished at: ", localtime     )
    print('Execution time: ', (time.time() - start_time)/3600., ' hours')
    
if __name__ == '__main__':
  tf.app.run()
    
    
    
    