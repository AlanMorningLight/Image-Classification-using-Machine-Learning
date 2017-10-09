"""
A script for testing the feedforward neural network.
"""

from __future__ import print_function

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

from fnn import FNNClassifier

print(__doc__)

# Load data
_base_path = ('the data folder')
_X_train_path = _base_path + 'train_x.bin'
_y_train_path = _base_path + 'train_y.csv'
X = np.fromfile(_X_train_path, dtype='uint8').reshape((100000, -1))
y = np.loadtxt(_y_train_path, dtype='int8', delimiter=',', usecols=(1,),
               skiprows=1)

# Normalize data
X = preprocessing.scale(X)
scaler = preprocessing.StandardScaler().fit(X)

clf = FNNClassifier()

# Set the parameters by cross-validation
parameters = [{'do_dropout': [False],
               'learning_rate': [0.3],
               'momentum_coeff': [0.9],
               'hidden_layer_sizes': [(20, 20)],
               'batch_size': [25],
               'max_iter': [15]}]

inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=42)

clf = GridSearchCV(clf, parameters, cv=inner_cv, n_jobs=-1)
clf.fit(X, y)

print("Best parameters set found:")
print(clf.best_params_, end='\n\n')
print("Best score:", clf.best_score_, end='\n\n')
print("Grid scores on development sets:", end='\n\n')
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("{0:.3f} (+/-{1:.3f}) for {2}".format(mean, std * 2, params))
    print()

nested_score = cross_val_score(clf, X, y=y, cv=outer_cv, n_jobs=-1)
print('Cross validated mean:', nested_score.mean())

