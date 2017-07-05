import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from seq2seq import basic_rnn_seq2seq, rnn_decoder, embedding_rnn_decoder, sequence_loss
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell
from metrics import performance

prediction_file = 'predictions.txt'
y_file = 'y.txt'
summary_file = '/home/logan/tmp'

EPOCHS = 5000
PRINT_STEP = 10
NUM_TRAIN = 100
NUM_TEST = 1000
LATENT_SIZE = 200
SEQUENCE_LEN = 20
VOCAB_SIZE = 100
EMBEDDING_SIZE = 50
PATIENCE = 200
BATCH_SIZE = 128
cell_type = 1
np.random.seed(123)


if cell_type == 2:
    len_x = LATENT_SIZE/2
else:
    len_x = LATENT_SIZE
# Each input is a latent vector of size LATENT_SIZE
x = np.random.randint(1, VOCAB_SIZE, size=(NUM_TRAIN+NUM_TEST, len_x))
for i in range(len(x)):
    for j in range(1,len(x[i])):
        x[i][j] = (x[i][j-1] + (VOCAB_SIZE/10+1)) % VOCAB_SIZE
# Each target is a sequence based on the latent vector
y = np.zeros((NUM_TRAIN+NUM_TEST, SEQUENCE_LEN))
# sum of each row
row_sum = np.sum(x, axis=1)
# To calculate the y at time t, take the sum of the latent vector,
# then divide by x[t], then add y[t-1]
for r in range(len(y)):
    for c in range(len(y[r])):
        y[r,c] = x[r, c % x.shape[1]]
# for r in range(len(y)):
#     for c in range(len(y[r])):
#         temp = row_sum[r] / x[r,c%LATENT_SIZE]
#         if c == 0:
#             previous = 0
#         else:
#             previous = y[r,c-1]
#         y[r,c] = temp + previous

y_shifted = np.roll(y, 1)
y_shifted[:,0] = 0
    
train_x = x[:NUM_TRAIN]
train_y = y[:NUM_TRAIN]
train_y_shifted = y_shifted[:NUM_TRAIN]

test_x = x[NUM_TRAIN:]
test_y = y[NUM_TRAIN:]
test_y_shifted = y_shifted[NUM_TRAIN:]

def batch_generator(x,y,y_shifted):
    for i in range(len(x)):
        yield x[i:min(i+BATCH_SIZE,len(x))], y[i:min(i+BATCH_SIZE,len(x))], y_shifted[i:min(i+BATCH_SIZE,len(x))]
    yield None, None, None

data = train_x
target = train_y

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    
class Decoder:
    def __init__(self, params):
        self.build_model()

    def build_model(self):
        x_ = tf.placeholder(tf.float32, [None, data.shape[1]], name='x_')
        c_init = tf.fill(tf.shape(x_), 0.0)
        y_ = [tf.placeholder(tf.int64, [None,], name='y_' + str(i)) for i in xrange(target.shape[1])]
        inputs = [tf.placeholder(tf.int64, [None,], name='input_' + str(i)) for i in xrange(target.shape[1])]
        weights = [tf.fill(tf.shape(y_[0]), 1.0) for i in xrange(target.shape[1])]
        
        self.x_ = x_
        self.y_ = y_
        self.inputs = inputs
        
        if cell_type == 2:
            cell = BasicLSTMCell(num_units=data.shape[1], activation=tf.nn.tanh, state_is_tuple=False)
        elif cell_type == 0:
            cell = BasicRNNCell(num_units=data.shape[1], activation=tf.nn.tanh)
        elif cell_type == 1:
            cell = GRUCell(num_units=data.shape[1], activation=tf.nn.tanh)
        
        W = tf.Variable(tf.random_normal([LATENT_SIZE, VOCAB_SIZE]), name='W')    
        b = tf.Variable(tf.random_normal([VOCAB_SIZE]), name='b')    
        variable_summaries(W) 
        variable_summaries(b)
        
        with tf.variable_scope("decoder"):
            outputs, states = embedding_rnn_decoder(inputs,
                                      x_,
                                      cell,
                                      VOCAB_SIZE,
                                      EMBEDDING_SIZE,
                                      output_projection=(W,b),
                                      feed_previous=False,
                                      update_embedding_for_previous=True)
            tf.get_variable_scope().reuse_variables()
            outputs_fed_previous, states_fed_previous = embedding_rnn_decoder(inputs,
                                      x_,
                                      cell,
                                      VOCAB_SIZE,
                                      EMBEDDING_SIZE,
                                      output_projection=(W,b),
                                      feed_previous=True,
                                      update_embedding_for_previous=True)
        
        def calcCost(y_, outputs):
        
            z = []
            y = []
            for t in range(len(outputs)):
                output_t = outputs[t]
                z.append(tf.matmul(output_t, W) + b)
                y.append(tf.arg_max(z[t], -1))
            cost = sequence_loss(z,
                          y_,
                          weights)
        
            return cost, y
        
        cost, y = calcCost(y_, outputs)
        cost_fed_previous, y_fed_previous = calcCost(y_, outputs_fed_previous)
        train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)
        merged = tf.summary.merge_all()
        
        self.cost = cost
        self.y = y
        self.cost_fed_previous = cost_fed_previous
        self.y_fed_previous = y_fed_previous
        self.train_op = train_op
        self.merged = merged
        
    def train(self):
        with tf.Session() as sess:
            K.set_session(sess)
            train_writer = tf.summary.FileWriter(summary_file + '/train', sess.graph)
            tf.global_variables_initializer().run()
            train_dict = {i: d for i, d in zip(self.y_, np.transpose(target))}
            temp = {i: d for i, d in zip(self.inputs, np.transpose(train_y_shifted))}
            train_dict.update(temp)
            train_dict[self.x_] = data
            test_dict = {i: d for i, d in zip(self.y_, np.transpose(test_y))}
            temp = {i: d for i, d in zip(self.inputs, np.transpose(test_y_shifted))}
            test_dict.update(temp)
            test_dict[self.x_] = test_x
            min_test_cost = np.inf
            num_mistakes = 0
            for i in range(EPOCHS):
                # train for 1 epoch
                sess.run(self.train_op, feed_dict=train_dict)
                
        #         batcher = batch_generator(train_x, train_y, train_y_shifted)
        #         batch_x, batch_y, batch_y_shifted = next(batcher)
        #         while batch_x != None:
        #             train_dict = {i: d for i, d in zip(y_, np.transpose(batch_y))}
        #             temp = {i: d for i, d in zip(inputs, np.transpose(batch_y_shifted))}
        #             train_dict.update(temp)
        #             train_dict[x_] = batch_x
        #             sess.run(train_op, feed_dict=train_dict)
        #             batch_x, batch_y, batch_y_shifted = next(batcher)
                
                # print loss
                if i % PRINT_STEP == 0:
                    train_cost, train_predictions, summary = sess.run([self.cost, self.y_fed_previous, self.merged], feed_dict=train_dict)
                    train_writer.add_summary(summary, i)
                    test_cost, test_predictions = sess.run([self.cost_fed_previous, self.y_fed_previous], feed_dict=test_dict)
                    train_predictions = np.transpose(np.array(train_predictions))
                    train_accuracy,_,_,_ = performance(train_predictions, train_y)
                    test_predictions = np.transpose(np.array(test_predictions))
                    test_accuracy,_,_,_ = performance(test_predictions, test_y)
                    print('training cost:', train_cost, 'training accuracy:', train_accuracy)
                    print('test cost:', test_cost, 'training accuracy:', test_accuracy)
                    
                if test_cost < min_test_cost:
                    num_mistakes = 0
                    min_test_cost = test_cost
                else:
                    num_mistakes += 1
                if num_mistakes >= PATIENCE:
                    break
                
            train_writer.close()
                
            # test on train set
            response = sess.run(self.y_fed_previous, feed_dict=train_dict)
            response = np.array(response)
            response = np.transpose(response)
            acc,_,_,_ = performance(response, train_y)
            print('Train accuracy: ', acc)
            
            # test on test set
            response = sess.run(self.y_fed_previous, feed_dict=test_dict)
            response = np.array(response)
            response = np.transpose(response)
            acc,_,_,_ = performance(response, test_y)
            print('Test accuracy: ', acc)
            #print to files
            with open(prediction_file, 'w') as f:
                np.savetxt(f, response, fmt='%d\t')
            with open(y_file, 'w') as f:
                np.savetxt(f, test_y, fmt='%d\t')
                
decoder = Decoder(None)
decoder.train()
print('done')
    
    
    
    
    