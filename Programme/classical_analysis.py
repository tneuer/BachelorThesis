#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : classical_analysis.py
    # Creation Date : Don 09 Aug 2018 11:47:37 CEST
    # Last Modified : Mit 19 Sep 2018 16:54:00 CEST
    # Description : Short script to do a superficial "classical" statistical analysis of the data
"""
#==============================================================================

import h5py
import pickle

import numpy as np
import pandas as pd

from sklearn import linear_model, neighbors, svm, metrics
from sklearn.preprocessing import OneHotEncoder

nr_train = 50000
nr_test = 10000
PCA = False

# Load test data
print("Read test...")

datafolder = "../data/"
with h5py.File("{}Processed_Val_mu_263093.h5".format(datafolder), "r") as valData:
    test_x = valData["Val"][:nr_test]

with open("{}Val_mu_labels_263093.pickle".format(datafolder), "rb") as f:
    test_y = pickle.load(f)
    test_y = test_y[:nr_test]

# Load train data
print("Read data...")
with h5py.File("{}Processed_Train_mu_400000.h5".format(datafolder), "r") as Data:
    data = Data["Train"][:nr_train]
if PCA:
    print("PCA data...")
    data = inc_pca.transform(data)

print("Read labels...")
with open("{}Train_mu_labels_400000.pickle".format(datafolder), "rb") as f:
    labels = pickle.load(f)
    labels = labels[:nr_train]


nr_features = test_x.shape[1]


# Fitting logistic regression model
print("\nFitting LogReg")

logreg = linear_model.LogisticRegression()
logreg.fit(data, labels)

print("Predict LogReg")
pred = logreg.predict(test_x)
predProba = logreg.predict_proba(test_x)
predProba = predProba[:, 1]

acc = metrics.accuracy_score(test_y, pred)
roc_auc = metrics.roc_auc_score(test_y, predProba)
cm = metrics.confusion_matrix(test_y, pred)

print("Accuracy:", acc)
print("Roc_Auc:", roc_auc)
print("Confusion matrix:")
print(cm)

#Fitting SVM classifier
print("\nFitting SVC")
svc = svm.SVC()
svc.fit(data, labels)

print("Predict SVC")
pred = svc.predict(test_x)

acc = metrics.accuracy_score(test_y, pred)
cm = metrics.confusion_matrix(test_y, pred)

print("Accuracy:", acc)
print("Confusion matrix:")
print(cm)

#Fitting k nearest neighbours
print("\nFitting kNN")

nearneigh = neighbors.KNeighborsClassifier()
nearneigh.fit(data, labels)

print("Predict kNN")
pred = nearneigh.predict(test_x)
predProba = nearneigh.predict_proba(test_x)
predProba = predProba[:, 1]

acc = metrics.accuracy_score(test_y, pred)
roc_auc = metrics.roc_auc_score(test_y, predProba)
cm = metrics.confusion_matrix(test_y, pred)

print("Accuracy:", acc)
print("Roc_Auc:", roc_auc)
print("Confusion matrix:")
print(cm)




































































