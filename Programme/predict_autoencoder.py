#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : predict_autoencoder.py
    # Creation Date : Mit 04 Apr 2018 23:18:37 CEST
    # Last Modified : Don 12 Apr 2018 18:09:54 CEST
    # Description :
"""
#==============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import h5py

from xgboost import XGBClassifier, plot_importance
from pipeliner import batch_pipeline, logtransform_standard_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn import svm
from baUtils import get_sg_and_bg

plt.style.use("./matplotlib_LHCb.mplstyle")

fnr = 2
mnr = 2

network = '../allOutputs/AutoEncoding/AutoEncoder{}/nn_logs{}/Models/'.format(fnr, mnr)

with open("../data/Val_mu_labels_263093.pickle", "rb") as f:
    test_y = pickle.load(f)
    test_y = test_y[:]

with h5py.File("../data/Processed_Val_mu_263093.h5", "r") as valData:
    test_x = valData["Val"][:]

with open("../data/inc_pca_mu_354.pickle", "rb") as f:
    inc_pca = pickle.load(f)

# test_x = inc_pca.transform(test_x)

nr_features = test_x.shape[1]

train_x, val_x, train_y, val_y = train_test_split(test_x, test_y, test_size=0.1)


def classify_AE():
    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('{}NNModel-0.meta'.format(network))
        new_saver.restore(sess, '{}NNModel-Best'.format(network))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        pred = graph.get_tensor_by_name("Model/predict:0")

        train_x = pred.eval({x: train_x})
        val_x = pred.eval({x: val_x})
        print(train_x.shape, train_y.shape)

    clf = svm.SVC(kernel="rbf")
    clf.fit(train_x, train_y)
    pred = clf.predict(val_x)
    print("Accuracy exponential SVM: ", accuracy_score(val_y, pred))

    clf = svm.SVC(kernel="linear")
    clf.fit(train_x, train_y)
    pred = clf.predict(val_x)
    print("Accuracy Linear SVM: ", accuracy_score(val_y, pred))

    plt.figure()
    plt.plot(val_x[val_y==1, 0], val_x[val_y==1, 1], "o", alpha=0.2, label="Signal")
    plt.plot(val_x[val_y==0, 0], val_x[val_y==0, 1], "o", alpha=0.2, label="Background")

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none')
    
    plt.legend()
    plt.show()

    model = XGBClassifier(max_depth=3, n_estimators=500, silent=True)
    model.fit(train_x, train_y)

    # make predictions for test data
    y_pred = model.predict(val_x)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(val_y, predictions)

    print("Accuracy XGB: ", accuracy)


def savefig(list_of_plots, path):
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(path)
    for fig in list_of_plots:
        pdf.savefig(fig, orientation="landscape")
    pdf.close()


def plot_cuts():
    sgname = "../data/Val_mu_MC_156388.pickle"
    bgname = "../data/Val_mu_hf_106705.pickle"

    origData, labels = get_sg_and_bg(sgname, bgname, labels=True, shuffle=True, random_state=42)

    figs = []
    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('{}NNModel-0.meta'.format(network))
        new_saver.restore(sess, '{}NNModel-Best'.format(network))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        pred = graph.get_tensor_by_name("Model/predict:0")

        predictions = pred.eval({x: test_x})

    origData["TrafoX"] = predictions[:, 0]
    origData["TrafoY"] = predictions[:, 1]

    Xcuts = [(-1, -0.9375), (-0.92, -0.75), (-0.60, -0.25), (0.2, 0.43), (0.61, 0.75), (0.85, 1)]

    fig1 = plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(predictions[labels==1, 0], predictions[labels==1, 1], "o", markersize=3, alpha=0.2, label="Signal")
    plt.plot(predictions[labels==0, 0], predictions[labels==0, 1], "o", markersize=3, alpha=0.2, label="Background")
    plt.title("2D rep")
    plt.legend()
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.grid()

    sg_dists = []
    bg_dists = []
    colors = []
    allcolors = ["#ff0000", "#6100ff", "#007c2d", "#6ad5fc", "#eeff00", "#d166ff"]

    for i, cut in enumerate(Xcuts):
        c = allcolors[i]
        plt.axvline(x=cut[0], color=c)
        plt.axvline(x=cut[1], color=c)

        data = origData[np.equal(origData["TrafoX"]<cut[1], origData["TrafoX"]>cut[0])]
        lbls = labels[np.equal(origData["TrafoX"]<cut[1], origData["TrafoX"]>cut[0])]

        sg_dists.append(data[lbls==1]["Z0_MM"])
        bg_dists.append(data[lbls==0]["Z0_MM"])
        colors.append(c)

    plt.subplot(1, 3, 2)
    plt.hist(sg_dists, histtype="step", bins=50, color=colors, label=[str(cut) for cut in Xcuts])
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.title("Signal mass histogram")
    plt.xlabel("Mass")
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.hist(bg_dists, histtype="step", bins=50, color=colors, label=[str(cut) for cut in Xcuts])
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.title("Background mass histogram")
    plt.xlabel("Mass")
    plt.ylabel("Count")


    Ycuts = [(0.29, 0.51), (-0.29, -0.17), (-0.41, -0.29), (-1, -0.78)]
    fig2 = plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(predictions[labels==1, 0], predictions[labels==1, 1], "o", markersize=3, alpha=0.2, label="Signal")
    plt.plot(predictions[labels==0, 0], predictions[labels==0, 1], "o", markersize=3, alpha=0.2, label="Background")
    plt.title("2D rep")
    plt.legend()
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.grid()

    sg_dists = []
    bg_dists = []
    colors = []
    allcolors = ["#ff0000", "#6100ff", "#007c2d", "#6ad5fc", "#eeff00", "#d166ff"]

    for i, cut in enumerate(Ycuts):
        c = allcolors[i]
        plt.axhline(y=cut[0], color=c)
        plt.axhline(y=cut[1], color=c)

        data = origData[np.equal(origData["TrafoY"]<cut[1], origData["TrafoY"]>cut[0])]
        lbls = labels[np.equal(origData["TrafoY"]<cut[1], origData["TrafoY"]>cut[0])]

        sg_dists.append(data[lbls==1]["Z0_MM"])
        bg_dists.append(data[lbls==0]["Z0_MM"])
        colors.append(c)

    plt.subplot(1, 3, 2)
    plt.hist(sg_dists, histtype="step", bins=50, color=colors, label=[str(cut) for cut in Xcuts])
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.title("Signal mass histogram")
    plt.xlabel("Mass")
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.hist(bg_dists, histtype="step", bins=50, color=colors, label=[str(cut) for cut in Xcuts])
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.title("Background mass histogram")
    plt.xlabel("Mass")
    plt.ylabel("Count")


    Xcuts = [(-1, -0.9375), (-0.92, -0.75), (-0.60, -0.25), (0.2, 0.43), (0.61, 0.75), (0.85, 1)]
    Ycuts = [(0.29, 0.51), (-0.29, -0.17), (-0.41, -0.29), (-1, -0.78)]

    XYcuts = [(-0.6, -0.25, 0.29, 0.51), (0.2, 0.43, 0.29, 0.51), (-0.6, -0.25, -0.29, -0.17), (0.2, 0.43, -0.29, -0.17)]
    fig3 = plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(predictions[labels==1, 0], predictions[labels==1, 1], "o", markersize=3, alpha=0.2, label="Signal")
    plt.plot(predictions[labels==0, 0], predictions[labels==0, 1], "o", markersize=3, alpha=0.2, label="Background")
    plt.title("2D rep")
    plt.legend()
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.grid()

    sg_dists = []
    bg_dists = []
    colors = []
    allcolors = ["#ff0000", "#6100ff", "#007c2d", "#6ad5fc", "#eeff00", "#d166ff"]

    for i, cut in enumerate(XYcuts):
        c = allcolors[i]
        plt.plot([cut[0], cut[0]], [cut[2], cut[3]], color=c)
        plt.plot([cut[0], cut[1]], [cut[3], cut[3]], color=c)
        plt.plot([cut[1], cut[1]], [cut[3], cut[2]], color=c)
        plt.plot([cut[1], cut[0]], [cut[2], cut[2]], color=c)

        data_tmp = origData[np.equal(origData["TrafoX"]<cut[1], origData["TrafoX"]>cut[0])]
        data = data_tmp[np.equal(data_tmp["TrafoY"]<cut[3], data_tmp["TrafoY"]>cut[2])]
        lbls = labels[np.equal(origData["TrafoX"]<cut[1], origData["TrafoX"]>cut[0])]
        lbls = lbls[np.equal(data_tmp["TrafoY"]<cut[3], data_tmp["TrafoY"]>cut[2])]

        sg_dists.append(data[lbls==1]["Z0_MM"])
        bg_dists.append(data[lbls==0]["Z0_MM"])
        colors.append(c)

    plt.subplot(1, 3, 2)
    plt.hist(sg_dists, histtype="step", bins=50, color=colors, label=[str(cut) for cut in Xcuts])
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.title("Signal mass histogram")
    plt.xlabel("Mass")
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.hist(bg_dists, histtype="step", bins=50, color=colors, label=[str(cut) for cut in Xcuts])
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.title("Background mass histogram")
    plt.xlabel("Mass")
    plt.ylabel("Count")


    figs = [fig1, fig2, fig3]
    plt.show()
    # savefig(figs, "../figures/CutsAutoencoder{}_{}.pdf".format(fnr, mnr))

plot_cuts()










































































