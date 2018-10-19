#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : compare_data_iso.py
    # Creation Date : Mon 20 Aug 2018 00:22:57 CEST
    # Last Modified : Mon 20 Aug 2018 00:35:49 CEST
    # Description :
"""
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import os.path
import h5py

plt.style.use("./matplotlib_LHCb.mplstyle")

with h5py.File("../data/Processed_Iso_Data_mu_500000.h5", "r") as h5:
    data = h5["Data"][:]

def savefig(list_of_plots, path):
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(path)
    for fig in list_of_plots:
        pdf.savefig(fig)
    pdf.close()

def build_full_output(stats, histogram, nr, acc, lastEpoch):
    for i in range(100):
        filename2 = "./Figures/fig{}_{}_acc{}_last{}.pdf".format(i, nr, str(acc)[0], str(lastEpoch)[0])
        if not os.path.isfile(filename2):

            build_figures(stats, path=filename2, histogram=histogram, nr=nr, acc=acc, lastEpoch=lastEpoch)
            print("Log and figures file: {}_{}".format(i, nr))
            break

with open("../data/IsoFeaturesOrder_mu_2367829.pickle", "rb") as f:
    iso_features = pickle.load(f)

data = pd.DataFrame(data, columns = iso_features)

data = data[["muminus_PT", "muminus_TrEta", "muminus_iso", "muminus_MINIP", "muplus_PT", "muplus_TrEta", "muplus_iso", "muplus_MINIP"]]

data["Max_Iso"] = data[["muminus_iso", "muplus_iso"]].max(axis=1)
data["Max_MINIP"] = data[["muminus_MINIP", "muplus_MINIP"]].max(axis=1)

real = data[["muminus_PT", "muminus_TrEta", "muplus_PT", "muplus_TrEta", "Max_Iso", "Max_MINIP"]]



with open("../data/Val_Iso_mu_labels_263093.pickle", "rb") as f:
    test_y = pickle.load(f)

with h5py.File("../data/Processed_Iso_Val_mu_263093.h5", "r") as valData:
    data = valData["Val"][:]

data = pd.DataFrame(data, columns = iso_features)

data = data[["muminus_PT", "muminus_TrEta", "muminus_iso", "muminus_MINIP", "muplus_PT", "muplus_TrEta", "muplus_iso", "muplus_MINIP"]]

data["Max_Iso"] = data[["muminus_iso", "muplus_iso"]].max(axis=1)
data["Max_MINIP"] = data[["muminus_MINIP", "muplus_MINIP"]].max(axis=1)

sim = data[["muminus_PT", "muminus_TrEta", "muplus_PT", "muplus_TrEta", "Max_Iso", "Max_MINIP"]]


for col in real.columns.values:
    plt.figure()
    plt.title(col)
    plt.hist(real[col], density=True, label="real", bins=30)
    plt.legend()
    plt.draw()

    plt.figure()
    plt.title(col)
    plt.hist(sim[col], density=True, label="sim", bins=30)
    plt.legend()
    plt.draw()

    plt.show()

plt.show()

