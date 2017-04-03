import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import os

# init parameters
root_data_dir = 'abalone_data'
train_file = os.path.join(root_data_dir, 'abalone_train.csv')
test_file  = os.path.join(root_data_dir, 'abalone_test.csv')
# test_file  = os.path.join(root_data_dir, 'abalone_predict.csv')

col_names = ['length', 'diameter', 'height',
             'whole_weight', 'shucked_weight',
             'viscera_weight', 'shell_weight', 'age']

# load data
df_train = pd.read_csv(train_file, names=col_names)
df_test  = pd.read_csv(test_file, names=col_names)

train_data = df_train.as_matrix()
test_data  = df_test.as_matrix()

features_train, targets_train = train_data[:, :-1], train_data[:, -1]
features_test, targets_test = test_data[:, :-1], test_data[:, -1]

# train ExtTreesRegressor
# classifier = ExtraTreesRegressor(n_estimators=500,
#                                  max_depth=20,
#                                  min_samples_split=5,
#                                  verbose=2)
# train NaiveGaussianBayes
# classifier = GaussianNB()

# train svm
classifier = svm.SVR()

classifier.fit(features_train , targets_train)
preds = classifier.predict(features_test).astype(np.int32)
print([(pred, target) for pred, target in zip(preds, targets_test.astype(np.int32))])



