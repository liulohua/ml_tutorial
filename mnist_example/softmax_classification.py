import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt
import os

# init parameters
num_classes = 10
learning_rate = 0.5
num_train_steps = 2000
batch_size = 512
data_dir = "mnist_data"
train_dir = 'train_dir'

if not os.path.exists(train_dir):
  os.mkdir(train_dir)

# load mnist dataset
mnist = read_data_sets(data_dir, one_hot=True)

# create graph
x = tf.placeholder(tf.float32, shape=[None, 28*28])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

logits = tf.layers.dense(x, units=num_classes)
loss = tf.losses.softmax_cross_entropy(y_, logits)
tf.summary.scalar('loss', loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)

probabilities = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(probabilities, axis=1),
                        tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# create initialization operation
init_op = tf.global_variables_initializer()

# create summary operation
merged_summary_op = tf.summary.merge_all()

sess = tf.Session()
sess.run(init_op) # make sure to initialize variables
summary_writer = tf.summary.FileWriter(logdir=train_dir, graph=sess.graph)

total_step = 0
for step in range(num_train_steps):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  _, loss_val, summary = sess.run([train_op, loss, merged_summary_op],
                         feed_dict={x:batch_xs, y_:batch_ys})
  total_step += 1
  if step % 10 == 0:
    print(step, loss_val)
    summary_writer.add_summary(summary, total_step)

print('====================Test Accuracy====================')
for step in range(10):
  batch_xs, batch_ys = mnist.test.next_batch(batch_size)
  accuracy_val = sess.run([accuracy],
                         feed_dict={x:batch_xs, y_:batch_ys})
  print(step, accuracy_val)

print('====================Test Result====================')
batch_xs, batch_ys = mnist.test.next_batch(batch_size)
probabilities_val = sess.run([probabilities],
                         feed_dict={x:batch_xs, y_:batch_ys})
# for i in range(10):
#   plt.figure()
#   plt.imshow(batch_xs[i].reshape((28, 28)))
#   plt.title('I guess the number is %d' % np.argmax(batch_ys[i]))
#   plt.show()
