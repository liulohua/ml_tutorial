import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.ensemble import ExtraTreesClassifier
from skimage.feature import hog
import matplotlib.pyplot as plt

# init parameters
num_classes = 10
learning_rate = 0.5
num_train_steps = 2000
batch_size = 512
data_dir = "mnist_data"


def func():
  mnist = read_data_sets(data_dir, one_hot=False, reshape=False)

  classifier = ExtraTreesClassifier(n_estimators=50,
                                    # max_depth=20,
                                    # bootstrap=True,
                                    min_samples_split=2,
                                    n_jobs=8,
                                    verbose=2)

  train_hog = np.zeros((mnist.train.num_examples, 1152))
  for i in range(mnist.train.num_examples):
    # print(i)
    fd, hog_image = hog(np.squeeze(mnist.train.images[i]),
                        orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2), visualise=True)
    train_hog[i] = fd
  
  classifier.fit(train_hog, mnist.train.labels)
  
  test_hog = np.zeros((mnist.test.num_examples, 1152))
  for i in range(mnist.test.num_examples):
    fd, hog_image = hog(np.squeeze(mnist.test.images[i]),
                        orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2), visualise=True)
    test_hog[i] = fd
  preds = classifier.predict(test_hog).astype(np.int32)
  targets = mnist.test.labels.astype(np.int32)
  accuracy = np.mean(np.equal(preds, targets))
  
  print(accuracy)


if __name__ == '__main__':
  # load mnist dataset
  func()