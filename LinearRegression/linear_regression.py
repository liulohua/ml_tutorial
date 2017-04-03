import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# prepare data
x_min = -10.
x_max = 10.
num_samples = 100

batch_xs = np.random.uniform(x_min, x_max, num_samples)
a = 10.
c = 5.
y_noise_sigma = 3.
batch_ys = a + c * batch_xs + np.random.randn(len(batch_xs)) * y_noise_sigma

plt.plot(batch_xs, batch_ys, "bo", markersize=6)

# create graph
x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
y_ = tf.placeholder(dtype=tf.float32, shape=None, name='x')

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='w')
b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name = 'b')

y = w * x + b

###############################################################################
# tunable parameters
learning_rate = 0.1 # option 1
learning_rate = 0.01 # option 2

num_iter_steps = 10
num_iter_steps = 100
num_iter_steps = 1000
###############################################################################

loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for i in range(num_iter_steps):
  _, loss_value = sess.run([train_op, loss],
                           feed_dict={x:batch_xs, y_:batch_ys})
  if i % 100 == 0:
    print('loss value: %.2f' % loss_value)

# testing
w_val, b_val = sess.run([w, b])
plt.plot(np.linspace(x_min, x_max, 1000),
        (b_val + w_val * np.linspace(x_min, x_max, 1000)), 'r', linewidth=2)
plt.grid()
plt.show()

sess.close()