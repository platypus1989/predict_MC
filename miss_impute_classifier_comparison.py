#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from predict_MC import predict_MC
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability = True),
    SVC(gamma=2, C=1, probability = True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]

n_samples = 1000
X, y = make_classification(n_samples = n_samples, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(n_samples = n_samples, noise=0.3, random_state=0),
            make_circles(n_samples = n_samples, noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

dataset_names = ['moon', 'circle', 'linear']            

accuracy_no_miss = pd.DataFrame(np.empty([len(names), len(datasets)]))
accuracy_no_miss.index = names
accuracy_no_miss.columns = dataset_names

# iterate over datasets
for j, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    for i, name, clf in zip(range(len(names)), names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        accuracy_no_miss.iloc[i,j] = score
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(score)).rjust(50))
        
    print('\n')


miss_index = rng.choice(X_test.shape[0], int(n_samples/10))

accuracy_median_impute = pd.DataFrame(np.empty([len(names), len(datasets)]))
accuracy_median_impute.index = names
accuracy_median_impute.columns = dataset_names

# iterate over datasets
for j, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    train_median = np.median(X_train[:,0])
    X_test[miss_index,1] = train_median

    for i, name, clf in zip(range(len(names)), names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        accuracy_median_impute.iloc[i,j] = score
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(score)).rjust(50))
        
    print('\n')

    
accuracy_mean_impute = pd.DataFrame(np.empty([len(names), len(datasets)]))
accuracy_mean_impute.index = names
accuracy_mean_impute.columns = dataset_names

# iterate over datasets
for j, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    train_mean = np.mean(X_train[:,0])
    X_test[miss_index,1] = train_mean

    for i, name, clf in zip(range(len(names)), names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        accuracy_mean_impute.iloc[i,j] = score
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(score)).rjust(50))
        
    print('\n')

    
    
accuracy_zero_impute = pd.DataFrame(np.empty([len(names), len(datasets)]))
accuracy_zero_impute.index = names
accuracy_zero_impute.columns = dataset_names

# iterate over datasets
for j, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    train_zero = 0
    X_test[miss_index,1] = train_zero

    for i, name, clf in zip(range(len(names)), names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        accuracy_zero_impute.iloc[i,j] = score
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(score)).rjust(50))
        
    print('\n')

    
accuracy_MC_impute = pd.DataFrame(np.empty([len(names), len(datasets)]))
accuracy_MC_impute.index = names
accuracy_MC_impute.columns = dataset_names

# iterate over datasets
for j, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    X_test[miss_index,1] = np.nan
    for i, name, clf in zip(range(len(names)), names, classifiers):
        clf.fit(X_train, y_train)
        pred = predict_MC(clf, X_train, X_test)
        score = accuracy_score(y_test, pred)
        accuracy_MC_impute.iloc[i,j] = score
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(score)).rjust(50))
        
    print('\n')    
    

accuracy_no_miss, accuracy_median_impute, accuracy_mean_impute, accuracy_zero_impute, accuracy_MC_impute   

accuracy_array = np.array([accuracy_median_impute.values, accuracy_mean_impute.values, 
                           accuracy_zero_impute.values, accuracy_MC_impute.values])
accuracy_array.argmax(axis=0)


accuracy_MC_order = pd.DataFrame(accuracy_array.shape[0] - accuracy_array.argsort(axis=0).argsort(axis=0)[3,:,:])
accuracy_MC_order.index = names
accuracy_MC_order.columns = dataset_names


sns.set()

myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

ax = sns.heatmap(accuracy_MC_order, cmap=cmap, linewidths=.5, linecolor='lightgray')

# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1.2, 2, 2.8])
colorbar.set_ticklabels(['1', '2', '3'])

# X - Y axis labels
ax.set_ylabel('Classifiers')
ax.set_xlabel('Simulation Scheme')

# Only y-axis labels need their rotation set, x-axis labels already have a rotation of 0
_, labels = plt.yticks()
plt.setp(labels, rotation=0)

plt.show()
