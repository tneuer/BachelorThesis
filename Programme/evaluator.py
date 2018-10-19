#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : evaluator.py
    # Creation Date : Don 08 Mär 2018 16:21:10 CET
    # Last Modified : Don 08 Mär 2018 23:42:07 CET
    # Description :
"""
#==============================================================================


import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os.path
import h5py
import gc
import sys
import re

from nn_model import feed_forward_nn, autoencoder
from sklearn.preprocessing import OneHotEncoder
from pipeliner import batch_pipeline, logtransform_standard_pipeline
from baUtils import get_sg_and_bg

with open("../data/Train_labels_400000.pickle", "rb") as f:
    labels = pickle.load(f)
with open("../data/Val_labels_263093.pickle", "rb") as f:
    test_y = pickle.load(f)
    test_y = test_y[:1000]

with h5py.File("../data/Processed_Val_263093.h5", "r") as valData:
    test_x = valData["Val"][:1000]

with open("../data/featureNames.pickle", "rb") as f:
    featureNames = pickle.load(f)

with open("../data/inc_pca354.pickle", "rb") as f:
    inc_pca = pickle.load(f)

enc = OneHotEncoder(sparse=False)
labels = enc.fit_transform(labels.reshape(-1,1))
test_y = enc.transform(test_y.reshape(-1,1))

test_x = inc_pca.transform(test_x)

nr_features = test_x.shape[1]
nr_classes = labels.shape[1]


with tf.Session() as sess:
    with open("../batchNorm/Outputs/FC/nn_logs3/Models/checkpoint", "r") as f:
        lines = f.readlines()
        pattern = "\".*\""
        for line in lines[1:]:
            regExp = re.search(pattern, line)
            print(regExp.group(0))

    new_saver = tf.train.import_meta_graph('../batchNorm/Outputs/FC/nn_logs3/Models/NNModel-0.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('../batchNorm/Outputs/FC/nn_logs3/Models/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    tr_phase = graph.get_tensor_by_name("trainPhase:0")
    accuracy = graph.get_tensor_by_name("Accuracy/accuracy:0")
    pred_proba = graph.get_tensor_by_name("Model/predProba:0")
    correct = graph.get_tensor_by_name("Accuracy/correct:0")
    cost = graph.get_tensor_by_name("Model/cost:0")

    acc = accuracy.eval({x: test_x, y: test_y, tr_phase: 1})
    print(acc)

    pred = pred_proba.eval({x: test_x, tr_phase: 1})
    pred = np.array([np.argmax(p) for p in pred])
    lbls = np.array([np.argmax(l) for l in test_y])

    corr = sum(np.equal(pred, lbls))

    print(corr)
    print()























































































