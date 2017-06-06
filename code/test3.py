import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(dtype=tf.float32, shape = [2,4]) #batchsize: 2, stepsize: 4
rnn = tf.contrib.rnn.BasicRNNCell(10)
state = rnn.zero_state(2, dtype=tf.float32) 
y,state = rnn(x, state) #ERROR OCCURS AT THIS LINE
tf.initialize_all_variables().run()
res = sess.run(y, feed_dict={x: [[1,2,1,1],[0,0,0,1]]})

print(res)