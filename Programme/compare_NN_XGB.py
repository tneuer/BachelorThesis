#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : compare_NN_XGB.py
    # Creation Date : Mit 04 Apr 2018 10:01:09 CEST
    # Last Modified : Son 19 Aug 2018 11:41:09 CEST
    # Description :
"""
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import pandas as pd

plt.style.use("./matplotlib_LHCb.mplstyle")

accuracyDict = {}
with open("../allOutputs/XGB/XGB_thresholds_300000.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        pattern = "n=([0-9]+)"
        nr = re.findall(pattern, line)

        pattern = "[0-9][0-9]\.[0-9][0-9]"
        acc = re.findall(pattern, line)
        if nr:
            accuracyDict[int(nr[0])] = acc[0]

NN_summary = pd.read_csv("../allOutputs/Featurewise/FC/summary.csv")
features_NN = NN_summary["feature_size"]

accuracies_NN = NN_summary["max_acc"]

plt.figure()
plt.plot(features_NN.values, accuracies_NN.values, marker="o", label="Neural Networks")
plt.plot(features_NN.values, [float(accuracyDict[i]) for i in features_NN], marker="o", label="XGBoost")
plt.xlabel("#Features")
plt.ylabel("Accuracy [%]")
plt.ylim([0,100])
plt.legend()
plt.show()



