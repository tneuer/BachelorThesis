#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : nn_featureImportance.py
    # Creation Date : Fre 10 Aug 2018 18:26:52 CEST
    # Last Modified : Son 12 Aug 2018 12:16:41 CEST
    # Description :
"""
#==============================================================================

import h5py
import time
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from nn_model import feed_forward_nn, autoencoder
from sklearn.preprocessing import OneHotEncoder
from baUtils import get_sg_and_bg

nr_examples = 50000
kturns = 10

with open("../data/Val_mu_labels_263093.pickle", "rb") as f:
    test_y = pickle.load(f)
    test_y = test_y[:nr_examples]

with h5py.File("../data/Processed_Val_mu_263093.h5", "r") as valData:
    test_x = valData["Val"][:nr_examples]

with open("../data/inc_pca_mu_354.pickle", "rb") as f:
    inc_pca = pickle.load(f)

with open("../data/featureNames.pickle", "rb") as f:
    feaureNames = np.array(pickle.load(f))

enc = OneHotEncoder(sparse=False)
test_y = enc.fit_transform(test_y.reshape(-1,1))

nr_features = test_x.shape[1]
nr_classes = test_y.shape[1]

print(np.unique(feaureNames))
raise

with tf.Session() as sess:

    new_saver = tf.train.import_meta_graph('../allOutputs/Outputs3/FC/nn_logs18/Models/NNModel-0.meta')
    new_saver.restore(sess, '../allOutputs/Outputs3/FC/nn_logs18/Models/NNModel-69')

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    tr_phase = graph.get_tensor_by_name("trainPhase:0")
    accuracy = graph.get_tensor_by_name("Accuracy/accuracy:0")
    pred_proba = graph.get_tensor_by_name("Model/predProba:0")
    correct = graph.get_tensor_by_name("Accuracy/correct:0")
    cost = graph.get_tensor_by_name("Model/cost:0")

    scores = []
    times = []

    baseline = accuracy.eval({x: test_x, y: test_y, tr_phase: 1})

    for i in np.arange(nr_features):
        score = []
        test_x_shuffle = np.copy(test_x)
        for k in np.arange(kturns):
            start = time.clock()
            test_x_shuffle[:, i] = np.random.permutation(test_x[:, i])
            s = accuracy.eval({x: test_x_shuffle, y: test_y, tr_phase: 1})
            score.append(s)
            times.append(time.clock()-start)

        if i % 10 == 0:
            meanPerRound = np.mean(times)
            remaining = np.round((nr_features-i)*kturns * meanPerRound, 4)
            print(i, "/", nr_features, "features processed; Remaining: ", remaining, "s.")
        scores.append(np.mean(score))

    importance = np.argsort(scores)
    scores = (np.array(scores)-baseline)[importance]
    scores = np.round(scores/max(scores)*100, 2)

    impFeatures = pd.DataFrame({"Features": feaureNames[importance], "Scores": scores, "ID": importance})
    impFeatures.to_csv("../NNImportanceAccuracy.csv", index=False)
























































































