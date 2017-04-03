import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fun(w, b):
  # prepare data
  x_min = -10.
  x_max = 10.
  m = 100
  
  x = np.random.uniform(x_min, x_max, m)
  true_w = 5.
  true_b = 5.
  y_noise_sigma = 3.
  y_ = true_w * x + true_b# + np.random.randn(len(x)) * y_noise_sigma

  y = w * x + b
  error = np.mean(np.square(y - y_) / 2.0)

  return error


# error_surface = np.zeros([100, 100])
# w_count = 0
# for w in np.linspace(-10, 20, 100):
#   b_count = 0
#   for b in np.linspace(-10, 20, 100):
#     y = w * x + b
#     error = np.mean(np.square(y - y_) / 2.0)
#     error_surface[w_count, b_count] = error
#     b_count += 1
#   w_count += 1

fig = plt.figure()
ax = Axes3D(fig)
w = np.linspace(4, 6, 100)
b = np.linspace(2, 8, 100)
W, B = np.meshgrid(w, b)
E = np.array([fun(w, b) for w, b in zip(np.ravel(W), np.ravel(B))])
E = E.reshape(W.shape)

ax.plot_surface(W, B, E, rstride=1, cstride=1, cmap=plt.cm.hot)

ax.set_xlabel('W Label')
ax.set_ylabel('B Label')
ax.set_zlabel('E Label')

plt.show()
