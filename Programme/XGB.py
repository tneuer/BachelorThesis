#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : XGB.py
    # Creation Date : Sam 24 Feb 2018 15:19:14 CET
    # Last Modified : Son 18 Mär 2018 15:44:42 CET
    # Description :
"""
#==============================================================================

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import pipeliner
import time
import h5py
import re

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from baUtils import get_sg_and_bg
from pipeliner import logtransform_standard_pipeline, batch_pipeline

plt.style.use("./matplotlib_LHCb.mplstyle")

def savefig(list_of_plots, path):
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(path)
    for fig in list_of_plots:
        pdf.savefig(fig)
    pdf.close()

#load data
with open("../data/Train_labels_400000.pickle", "rb") as f:
    labels = pickle.load(f)
with open("../data/Val_labels_263093.pickle", "rb") as f:
    test_y = pickle.load(f)
    test_y = test_y[:]

with h5py.File("../data/Processed_Val_263093.h5", "r") as valData:
    test_x = valData["Val"][:]

with open("../data/featureNames.pickle", "rb") as f:
    featureNames = pickle.load(f)

with open("../data/inc_pca354.pickle", "rb") as f:
    inc_pca = pickle.load(f)

PCA = True
print("PCA: {}".format(PCA))

if PCA:
    test_x = inc_pca.transform(test_x)
instances = [300000] #50, 100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 200000, 250000, 300000, 350000, 400000]

accuracies = []
aucs = []
f1_scores = []
accuraciesTrain = []
aucsTrain = []
f1_scoresTrain = []

figs = []
times = []

# for instance in instances:
#     with h5py.File("../data/Processed_Train_400000.h5", "r") as data:
# 
#         print("Instances: ", instance)
# 
#         start = time.clock()
#         train_x = data["Train"][:instance]
#         train_y = labels[:instance]
#         print("Read-Time: ", time.clock()-start)
# 
#         nr_examples = train_x.shape[0]
#         nr_features = test_x.shape[1]
# 
#         if PCA:
#             train_x = inc_pca.transform(train_x)
# 
#         # fit model no training data
#         model = XGBClassifier(max_depth=3, n_estimators=100, silent=True)
#         model.fit(train_x, train_y)
# 
# 
#         # make predictions for test data
#         y_pred = model.predict(test_x)
#         predictions = [round(value) for value in y_pred]
# 
#         accuracy = accuracy_score(test_y, predictions)
#         auc_score = roc_auc_score(test_y, predictions)
#         f1 = f1_score(test_y, predictions)
#         print("Accuracy: %.2f%%" % (accuracy * 100.0))
# 
#         accuracies.append(accuracy)
#         aucs.append(auc_score)
#         f1_scores.append(f1)
# 
#         t = time.clock()-start
#         fig = plt.figure(figsize=(20,10))
# 
#         if PCA:
#             plot_importance(model, max_num_features=30)
#             fImportance = model._Booster.get_fscore()
# 
#             keys = []
#             values = []
#             for key, value in fImportance.items():
#                 keys.append(key)
#                 values.append(value)
# 
#             sortInd = np.argsort(values)[::-1][:30]
# 
#             values = np.array(values)[sortInd]
#             keys = np.array(keys)[sortInd]
#             fNames = keys
# 
#         else:
#             fImportance = model._Booster.get_fscore()
#             keys = []
#             values = []
#             for key, value in fImportance.items():
#                 keys.append(key)
#                 values.append(value)
# 
#             keys = [int(key[1:]) for key in keys]
#             sortInd = np.argsort(values)[::-1][:30]
# 
#             values = np.array(values)[sortInd]
#             keys = np.array(keys)[sortInd]
#             fNames = np.array(featureNames)[keys]
# 
#             bars = plt.bar(np.arange(len(values)), values)
#             for i, bar in enumerate(bars):
#                 height = bar.get_height()
#                 width = bar.get_x() + bar.get_width()/2
#                 plt.text(width, height, values[i], ha="center", va="bottom")
#             plt.xticks(np.arange(len(values)), fNames, rotation=70, size=9)
#             plt.ylabel("F-score")
#             plt.title("Instances: {}, Accuracy: {}, ROC AUC: {}, Time: {}".format(instance, np.round(accuracy,4)*100, np.round(auc_score, 3), np.round(t, 3)))
# 
#         plt.draw()
#         y_pred = model.predict(train_x)
#         predictions = [round(value) for value in y_pred]
#         
#         accuracy = accuracy_score(train_y, predictions)
#         auc_score = roc_auc_score(train_y, predictions)
#         f1 = f1_score(train_y, predictions)
#         
#         accuraciesTrain.append(accuracy)
#         aucsTrain.append(auc_score)
#         f1_scoresTrain.append(f1)
#         
#         times.append(t)
#         figs.append(fig)
#         print("Fit-Time: ", t, "\n")
# 
# savefig(figs, "../figures/XGB_PCA_Importances_{}.pdf".format(instances[-1]))
# 
# plt.figure()
# plt.plot(instances, accuracies, "o-", label="Test")
# plt.plot(instances, accuraciesTrain, "o-", label="Train")
# plt.legend()
# plt.xlabel("#Instances")
# plt.xscale("log")
# plt.ylabel("Accuracy")
# plt.title("Accuracy Development")
# plt.savefig("../figures/XGBShuffle_accuracies.png")
# plt.draw()
# 
# plt.figure()
# plt.plot(instances, aucs, "o-", label="Test")
# plt.plot(instances, aucsTrain, "o-", label="Train")
# plt.legend()
# plt.xlabel("#Instances")
# plt.xscale("log")
# plt.ylabel("AUC")
# plt.title("AUC Development")
# plt.savefig("../figures/XGBShuffle_aucs.png")
# plt.draw()
# 
# plt.figure()
# plt.plot(instances, f1_scores, "o-", label="Test")
# plt.plot(instances, f1_scoresTrain, "o-", label="Train")
# plt.legend()
# plt.xlabel("#Instances")
# plt.xscale("log")
# plt.ylabel("F1 Score")
# plt.title("F Score Development")
# plt.savefig("../figures/XGBShuffle_fScores.png")
# plt.draw()
# 
# plt.figure()
# plt.plot(instances, times, "o-")
# plt.xlabel("#Instances")
# plt.xscale("log")
# plt.ylabel("Fit time")
# plt.title("Fit time Development")
# plt.savefig("../figures/XGBShuffle_fitTime.png")
# 
# with open("../data/XGBShuffle_Stats.pickle", "wb") as f:
#     pickle.dump([instances, (accuracies, accuraciesTrain), (aucs, aucsTrain), (f1_scores, f1_scoresTrain), times, (fNames, values)], f)


# with open("../data/XGB_Stats.pickle", "rb") as f:
#     stats = pickle.load(f)
#     instances = stats[0]
#     accuracies = stats[1]
#     aucs = stats[2]
#     f1_scores = stats[3]
#     times = stats[4]
# 
# model._Booster.save_model("XGBBooster_PCA_{}.model".format(instances[-1]))
# with open("XGBModel_PCA_{}.pickle".format(instances[-1]), "wb") as f:
#     pickle.dump(model, f)
# 
# Fit model using each importance as a threshold
# thresholds = np.sort(np.unique(model.feature_importances_))[::-1]
# thresh_limit = 100 if 100 < len(thresholds) else len(thresholds)
# thresholds = thresholds[:thresh_limit]
# nr_feat = []
# accs = []
# rocs = []
# print(len(thresholds))
# with open("XGB_PCA_thresholds_{}.txt".format(instances[-1]), "w") as f:
#     for thresh in thresholds:
#         # select features using threshold
#         selection = SelectFromModel(model, threshold=thresh, prefit=True)
#         select_x_train = selection.transform(train_x)
# 
#         # train model
#         selection_model = XGBClassifier()
#         selection_model.fit(select_x_train, train_y)
# 
#         # eval model
#         select_x_test = selection.transform(test_x)
#         y_pred = selection_model.predict(select_x_test)
#         predictions = [round(value) for value in y_pred]
#         accuracy = accuracy_score(test_y, predictions)
#         roc = roc_auc_score(test_y, predictions)
#         txt = "Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_x_train.shape[1], accuracy*100.0)
#         nr_feat.append(select_x_train.shape[1])
#         accs.append(accuracy)
#         rocs.append(roc)
#         print(txt)
#         f.write(txt+"\n\n")
# 
# plt.figure(figsize=(20,10))
# plt.plot(nr_feat, accs, "o-")
# plt.axhline(y=accuracies[-1], c="red", linewidth=1, zorder=0, linestyle="--")
# plt.xlabel("#Features")
# plt.ylabel("Accuracy")
# plt.xscale("log")
# plt.title("#Examples: {}".format(instances[-1]))
# plt.grid()
# plt.savefig("../figures/XGB_PCA_accsFeature.pdf")
# plt.draw()
# 
# plt.figure(figsize=(20,10))
# plt.plot(nr_feat, rocs, "o-")
# plt.axhline(y=aucs[-1], c="red", linewidth=1, zorder=0, linestyle="--")
# plt.xlabel("#Features")
# plt.ylabel("ROC AUC")
# plt.xscale("log")
# plt.title("#Examples: {}".format(instances[-1]))
# plt.grid()
# plt.savefig("../figures/XGB_PCA_rocsFeature.pdf")
# plt.draw()
# 
# plt.figure(figsize=(20,10))
# plt.plot(nr_feat[:20], accs[:20], "o-")
# plt.axhline(y=accuracies[-1], c="red", linewidth=1, zorder=0, linestyle="--")
# plt.xlabel("#Features")
# plt.ylabel("Accuracy")
# plt.title("#Examples: {}".format(instances[-1]))
# plt.grid()
# plt.savefig("../figures/XGB_PCA_accsFeature20.pdf")
# plt.draw()
# 
# plt.figure(figsize=(20,10))
# plt.plot(nr_feat[:20], rocs[:20], "o-")
# plt.axhline(y=aucs[-1], c="red", linewidth=1, zorder=0, linestyle="--")
# plt.xlabel("#Features")
# plt.ylabel("ROC AUC")
# plt.title("#Examples: {}".format(instances[-1]))
# plt.grid()
# plt.savefig("../figures/XGB_PCA_rocsFeature20.pdf")
# plt.show()


with open("./XGBModel_300000.pickle", "rb") as f:
    model = pickle.load(f)


fImportance = model._Booster.get_fscore()
keys = []
values = []
for key, value in fImportance.items():
    keys.append(key)
    values.append(value)

keys = [int(key[1:]) for key in keys]
sortInd = np.argsort(values)[::-1]#[:30]

values = np.array(values)[sortInd]
keys = np.array(keys)[sortInd]
fNames = np.array(featureNames)[keys]

accuracyDict = {}
with open("./XGB_thresholds_300000.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        pattern = "n=([0-9]+)"
        nr = re.findall(pattern, line)

        pattern = "[0-9][0-9]\.[0-9][0-9]%"
        acc = re.findall(pattern, line)
        if nr:
            accuracyDict[int(nr[0])] = acc[0]

print(accuracyDict)

with open("./XGBImportantFeatures.txt", "w") as f:
    for i, name in enumerate(fNames):
        f.write("{}) {}\n".format(i+1, name))

        if i+1 in accuracyDict:
            f.write("---"*5 + "Contained --> Accuracy: {}\n".format(accuracyDict[i+1]))

