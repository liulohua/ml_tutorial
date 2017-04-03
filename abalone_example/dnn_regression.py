import tensorflow as tf
import numpy as np


def dnn_model(x, training=False):
  hidden1 = tf.layers.dense(x, units=10, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=hidden1, rate=0.7, training=training)
  hidden2 = tf.layers.dense(hidden1, units=10, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=hidden2, rate=0.7, training=training)
  hidden3 = tf.layers.dense(hidden2, units=10, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=hidden3, rate=0.7, training=training)
  logits = tf.layers.dense(dropout, units=1)
  return logits


train_file = 'Data/abalone_train.csv'
test_file = 'Data/abalone_predict.csv'
columns = ["Length", "Diameter", "Height", "Whole_Weight", "Shucked_Weight",
           "Viscera_Weight", "Shell_Weight", "Age"]
num_features = 7

# load data
train_set = tf.contrib.learn.datasets.base.load_csv_without_header(
  filename=train_file, target_dtype=np.float32, features_dtype=np.float32)

num_train_data = len(train_set.data)

test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
  filename=test_file, target_dtype=np.float32, features_dtype=np.float32)

# create graph
x = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name='x')
y_ = tf.placeholder(dtype=tf.float32, shape=None, name='x')

y = dnn_model(x, training=True)
yyy = dnn_model(x, training=False)

loss = tf.reduce_mean(tf.square(y - y_))

global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

lr = tf.train.exponential_decay(0.5,
                                global_step,
                                1000,
                                0.1,
                                staircase=True)
optimizer = tf.train.AdadeltaOptimizer(lr)
train_op = optimizer.minimize(loss, global_step=global_step)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

# batch_size = 256
for i in range(3000):
  # batch_xs_indices = np.random.choice(num_train_data, batch_size)
  # batch_xs = train_set.data[batch_xs_indices, :]
  # batch_ys = np.random.choice(train_set.target, batch_size)
  lr_val, _, loss_value = sess.run([lr, train_op, loss],
                           feed_dict={x:train_set.data, y_:train_set.target})
  if i % 10 == 0:
    print('loss value: %.2f' % loss_value)
    print(lr_val)

# testing
predicted_age, actual_age = \
sess.run([yyy, y_], feed_dict={x:train_set.data[0:10], y_:train_set.target[0:10]})
print('predicted age', predicted_age.transpose().astype(np.int32))
print('actual age', actual_age.astype(np.int32))

sess.close()