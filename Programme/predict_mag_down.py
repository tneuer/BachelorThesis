#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : predict_mad_down.py
    # Creation Date : Don 05 Apr 2018 00:20:43 CEST
    # Last Modified : Mit 29 Aug 2018 17:51:02 CEST
    # Description :
"""
#==============================================================================

import h5py
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from nn_model import feed_forward_nn, autoencoder
from sklearn.preprocessing import OneHotEncoder
from baUtils import get_sg_and_bg

with open("../data/Val_md_labels_264972.pickle", "rb") as f:
    test_y = pickle.load(f)
    test_y = test_y[:]

with h5py.File("../data/Processed_Val_md_264972.h5", "r") as valData:
    test_x = valData["Val"][:]


enc = OneHotEncoder(sparse=False)
test_y = enc.fit_transform(test_y.reshape(-1,1))

nr_features = test_x.shape[1]
nr_classes = test_y.shape[1]

sgname = "../data/Val_md_MC_157677.pickle"
bgname = "../data/Val_md_hf_107295.pickle"

origData, test_y = get_sg_and_bg(sgname, bgname, labels=True, shuffle=True, random_state=42)


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
    histogram = "Z0_MM"

    pred = pred_proba.eval({x: test_x, tr_phase: 1})
    pred = np.array([np.argmax(p) for p in pred])

    signal = np.where(pred==1)[0]
    bg = np.where(pred==0)[0]

    truesignal = np.where(test_y==1)[0]
    truebg = np.where(test_y==0)[0]

    truepos = np.logical_and(test_y==1, pred==1)
    trueneg = np.logical_and(test_y==0, pred==0)
    falsepos = np.logical_and(test_y==0, pred==1)
    falseneg = np.logical_and(test_y==1, pred==0)

    bincuts = np.array([10, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 60.0, 120.0])*1000

    x = [origData[histogram][truepos], origData[histogram][falsepos], origData[histogram][trueneg], origData[histogram][falseneg]]
    n, edges, _ = plt.hist(x, histtype = "bar", stacked = True, bins=bincuts, color = ["#0b0db0", "#7172e1", "#b21000","#e17b71" ], label = ["true signal", "false signal", "true background", "false background"])

    count_trueSig = n[0]
    count_predSig = n[0] + n[1]
    count_trueBg = n[2]
    count_predBg = n[2] + n[3]
    count_correct = n[0] + n[2]
    count_signal = n[0] + n[3]
    count_bg = n[1] + n[2]
    count_all = n[0] + n[1] + n[2] + n[3]

    predsg = list(count_predSig/count_all)
    predbg = list(count_predBg/count_all)
    predsg.insert(0, predsg[0])
    predbg.insert(0, predbg[0])

    predFsg = list(n[1]/count_all)
    predFbg = list(n[3]/count_all)
    predFsg.insert(0, predFsg[0])
    predFbg.insert(0, predFbg[0])

    pd.DataFrame({"bg": predbg, "signal": predsg, "bg_err": predFbg, "signal_err":predFsg}).to_csv("dataPredictionTransverseDown.csv", index=False)


    acc = accuracy.eval({x: test_x, y: test_y, tr_phase: 1})
    print("Accuracy on Mag down: ", acc)





