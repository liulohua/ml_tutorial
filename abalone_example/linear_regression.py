import tensorflow as tf
import numpy as np

train_file = 'Data/abalone_train.csv'
test_file = 'Data/abalone_predict.csv'
columns = ["Length", "Diameter", "Height", "Whole_Weight", "Shucked_Weight",
           "Viscera_Weight", "Shell_Weight", "Age"]
num_features = 7

tf.contrib.learn.DNNClassifier
# load data
train_set = tf.contrib.learn.datasets.base.load_csv_without_header(
  filename=train_file, target_dtype=np.float32, features_dtype=np.float32)

num_train_data = len(train_set.data)

test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
  filename=test_file, target_dtype=np.float32, features_dtype=np.float32)

# create graph
x = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name='x')
y_ = tf.placeholder(dtype=tf.float32, shape=None, name='x')

w = tf.Variable(tf.random_normal([num_features, 1]), dtype=tf.float32, name='w')
b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name = 'b')

y = tf.add(tf.matmul(x, w), b)

loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.AdadeltaOptimizer(0.1)
train_op = optimizer.minimize(loss)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for i in range(1000):
  _, loss_value = sess.run([train_op, loss],
                           feed_dict={x:train_set.data, y_:train_set.target})
  if i % 10 == 0:
    print('loss value: %.2f' % loss_value)

# testing
predicted_age, actual_age = \
sess.run([y, y_], feed_dict={x:test_set.data, y_:test_set.target})
print('predicted age', predicted_age.transpose().astype(np.int32))
print('actual age', actual_age.astype(np.int32))

sess.close()