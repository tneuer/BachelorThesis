#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : correlations.py
    # Creation Date : Mit 11 Apr 2018 17:38:20 CEST
    # Last Modified : Don 12 Apr 2018 11:02:47 CEST
    # Description :
"""
#==============================================================================

import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from baUtils import get_sg_and_bg

data = "../data/Processed_Train_mu_400000.h5"
labels = "../data/Train_mu_labels_400000.pickle"

with h5py.File(data, "r") as f:
    data = f["Train"][:50000]

with open(labels, "rb") as f:
    labels = pickle.load(f)
    labels = labels[:50000]

with open("../data/featureNamesDict.pickle", "rb") as f:
    fNames = pickle.load(f)

def plot_correlation(f1, f2, show=True):
    f1_sig = data[labels==1, fNames[f1]]
    f1_bg = data[labels==0, fNames[f1]]
    f2_sig = data[labels==1, fNames[f2]]
    f2_bg = data[labels==0, fNames[f2]]

    fig = plt.figure()
    plt.plot(f1_sig, f2_sig, "ro", alpha=0.2, label="Signal")
    plt.plot(f1_bg, f2_bg, "bo", alpha=0.2, label="Background")
    plt.title("Correlation {} - {}".format(f1, f2))
    plt.xlabel(f1)
    plt.ylabel(f2)
    if show:
        plt.show()
    return fig

f1s = ["muminus_PT", "muminus_PT", ]
f2s = ["muplus_TrEta", "muminus_TrPhi"]

# for f1, f2 in zip(f1s, f2s):
#     plot_correlation(f1, f2, show=False)

newVar = []
newVarNr = []
pattern = "tracks_[a-zA-Z]+"
for key in fNames:
    reg = re.findall(pattern, key)
    if reg:
        reg = reg[0][7:]
        if reg not in newVar:
            newVarNr.append(fNames[key])
            newVar.append(reg)

print("Calculating correlations...")
data = pd.DataFrame(data[:, :])
corrMatrix = data.corr()
diag = np.zeros(corrMatrix.shape, int)
np.fill_diagonal(diag, 1)
corrMatrix -= diag
plt.matshow(corrMatrix, cmap=plt.cm.seismic)

newVarNr = [nr-0.5 for nr in newVarNr if nr < corrMatrix.shape[0] ]
for nr in newVarNr:
    plt.axhline(y=nr, color="r")
    plt.axvline(x=nr, color="r")
plt.colorbar()
titlestring = [str((var, int(nr+0.5))) for var, nr in zip(newVar, newVarNr)]
plt.title("  ".join(titlestring))
plt.savefig("../figures/correlations.png")

plt.show()














































































