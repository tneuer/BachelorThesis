#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : ba_plots.py
    # Creation Date : Die 28 Aug 2018 14:30:25 CEST
    # Last Modified : Die 18 Sep 2018 19:19:07 CEST
    # Description : Plots used in the bachelor thesis
"""
#==============================================================================

import re
import pickle
import h5py

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

from baUtils import get_sg_and_bg
from pipeliner import logtransform_standard_pipeline

plt.style.use("./matplotlib_LHCb.mplstyle")
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 150}
matplotlib.rc('font', **font)

def savefig(list_of_plots, path):
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(path)
    for fig in list_of_plots:
        pdf.savefig(fig)
    pdf.close()

bgname = "../data/Train_mu_hf_200000.pickle"
sgname = "../data/Train_mu_MC_200000.pickle"
with open(bgname, "rb") as f:
    bg = pickle.load(f)

with open(sgname, "rb") as f:
    signal = pickle.load(f)

data, labels = get_sg_and_bg(sgname, bgname, labels=True)

####
# Section 2 Data : Comparison evaluation data <-> training data
####
print("Sec2...")

###
# Mass
###

"""
with open("../data/featureNamesDict.pickle", "rb") as f:
    fNames = pickle.load(f)

with open("../data/Data_mu_500000.pickle", "rb") as f:
    EvalData = pickle.load(f)

end_bg = bg["Z0_ENDVERTEX_CHI2"]
end_sg = signal["Z0_ENDVERTEX_CHI2"]

var = "Z0_MM"

truesignal = np.where(labels==1)[0]
truebg = np.where(labels==0)[0]

x = [data[var][labels==0]/1000, data[var][labels==1]/1000]

bincuts = np.array([10, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 60.0, 120.0])
fig = plt.figure(figsize=(40,20))
plt.hist(EvalData[var]/1000, histtype="step", density=True, bins=bincuts, label = "Eval. data", linewidth=12)
plt.hist(data[var]/1000, histtype="step", density=True, bins=bincuts, label = "Train data (sg+bg)", linewidth=12)
plt.hist(x, histtype="step", density=True, bins=bincuts, label = ["bg", "sg"], linestyle="dashed", linewidth=12, alpha=0.3)
ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(80)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(80)
plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$', fontsize=80)
plt.ylabel("Density", fontsize=80)
plt.yscale("log")
plt.legend(loc=0, fontsize=70)
fig.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Processing/MassPlotComparison.png")
"""

###
# Transverse features
###

####
# Section 2.3 Preprocessing : Logtransformation
####
print("Sec2.3")

"""
variables = ["muminus_PT", "muminus_PZ", "tracks_PT", "tracks_IPCHI2"]

for variable in variables:
    print(variable)
    fig, axs = plt.subplots(figsize=(40,10))
    if variable in ["tracks_PX", "tracks_PZ", "tracks_PZ", "tracks_IP", "tracks_IPCHI2", "tracks_isMuon", "tracks_charge", "tracks_PT"]:
        sig = np.array([l for sublist in signal[variable] for l in sublist])
        bkg = np.array([l for sublist in bg[variable] for l in sublist])
    else:
        sig = signal[variable].values
        bkg = bg[variable].values

    logsig = np.log(sig+1)
    logbkg = np.log(bkg+1)

    ax = plt.subplot(1, 2, 1)
    plt.xlabel(variable, fontsize=60)
    plt.ylabel("Density", fontsize=60)
    plt.hist(sig, bins=30, density=True, histtype="step", label="signal", linewidth=12)
    plt.hist(bkg, bins=30, density=True, histtype="step", label="background", linewidth=12)
    plt.grid()
    plt.legend(loc=0, fontsize=50)

    ax = plt.subplot(1, 2, 2)
    plt.xlabel("log({})".format(variable), fontsize=60)
    plt.ylabel("Density", fontsize=60)
    plt.hist(logsig, bins=30, density=True, histtype="step", label="signal", linewidth=12)
    plt.hist(logbkg, bins=30, density=True, histtype="step", label="background", linewidth=12)
    plt.grid()
    plt.legend(loc=0, fontsize=50)
    plt.draw()
    fig.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Processing/LogPlot_{}.png".format(variable))
"""

####
# Section 2.3 Preprocessing : Feature distribution used in training
####

"""
transverse_wo_SPD = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']

standard_features = transverse_wo_SPD[:7]
standard_log = ["muplus_PT", "muminus_PT"]

non_standard_features = transverse_wo_SPD[6:]
global_standardized = non_standard_features[1:-2]
non_standard_log = ["tracks_PT", "tracks_IPCHI2", "tracks_IP"]

scale_transformer_pipeline = logtransform_standard_pipeline(standard_features, standard_log,
	non_standard_features, non_standard_log, global_standardized)

data = scale_transformer_pipeline.fit_transform(data)

newFeatures = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", "nTrackN", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']
standard_features[standard_features.index("nTrack")] = "nTrackN"
data = pd.DataFrame(data, columns=newFeatures)

signal = data.iloc[labels==1, :]
bg = data.iloc[labels==0, :]

variables = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", 'nTrackN', 'tracks_PT', 'tracks_eta', 'tracks_phi', 'tracks_IP', 'tracks_IPCHI2', 'tracks_charge', 'tracks_isMuon']


for variable in variables:
    print(variable)
    fig = plt.figure(figsize=(30,30))
    if "tracks" in variable:
        sig = np.array([l for sublist in signal[variable] for l in sublist])
        bkg = np.array([l for sublist in bg[variable] for l in sublist])

    else:
        sig = np.array([l for l in signal[variable].values])
        bkg = np.array([l for l in bg[variable].values])

    plt.hist(sig, bins=50, histtype="step", density=True, label="signal", linewidth=10)
    plt.hist(bkg, bins=50, histtype="step", density=True, label="background", linewidth=10)

    plt.xlabel(variable, fontsize=60)
    plt.ylabel("Density", fontsize=60)
    plt.grid()
    plt.legend(loc=7, fontsize=60)
    fig.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Processing/Processed_{}.png".format(variable))
"""

####
# Section 2.3.2 PCA : Feature variance
####

"""
with open("../data/inc_pca_mu.pickle", "rb") as f:
    inc_pca = pickle.load(f)

evar = inc_pca.explained_variance_ratio_
csum = np.cumsum(inc_pca.explained_variance_ratio_)
fractions = [0.99, 0.95, 0.90]
ds = [np.argmax(csum >= f)+1 for f in fractions]
titlestring = ""

fig = plt.figure(figsize=(40,20))
plt.plot(csum, linewidth=8)
for f, d in zip(fractions, ds):
    titlestring += "{}: - {} -".format(f, d)
    plt.axvline(x=d, c="red", linewidth=4, zorder=0, linestyle="--")
    plt.axhline(y=f, c="red", linewidth=4, zorder=0, linestyle="--")
titlestring = titlestring[:-1]
plt.ylabel("Explained variance", fontsize=80)
plt.xlabel("Dimensions", fontsize=80)
print(titlestring)
plt.grid()
plt.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Processing/PCA.png")
"""

####
# Section 3.4 Activations : Comparison of activation functions
####

# See plot_activations.py

####
# Section 4.1 Training : Development of loss and accuracy
####

# Output of every FCLogSaver.py

####
# Section 4.2 Transverse features : Massbins and fraction
####

# Output of every FCLogSaver.py

####
# Section 4.2 Transverse features : Separation and error
####

# Output of every FCLogSaver.py

###
# Error
###

"""
fig = plt.figure()
df = pd.read_csv("../data/dataPredictionTransverse.csv")
df.m0 = df.m0/1000
df.m1 = df.m1/1000
plt.errorbar((df.m0+df.m1)/2, df.bg_err, fmt="P", xerr=(df.m1-df.m0)/2, label="Background")
plt.errorbar((df.m0+df.m1)/2, df.signal_err, fmt="P", xerr=(df.m1-df.m0)/2, label="Signal")
plt.legend(loc=1, fontsize=45)
plt.ylim(0, 0.15)
plt.yticks(np.arange(0, 0.15, 0.05))
plt.ylabel("Error", fontsize=35)
plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$', fontsize=35)
plt.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Results/TransverseError.png")

####
# Section 4.3 Isolation features : Separation and error
####

# Output of every FCLogSaverIso.py

fig = plt.figure()
df = pd.read_csv("../data/dataPredictionIsolation.csv")
df.m0 = df.m0/1000
df.m1 = df.m1/1000
plt.errorbar((df.m0+df.m1)/2, df.bg_err, fmt="P", xerr=(df.m1-df.m0)/2, label="Background")
plt.errorbar((df.m0+df.m1)/2, df.signal_err, fmt="P", xerr=(df.m1-df.m0)/2, label="Signal")
plt.legend(loc=1, fontsize=45)
plt.ylim(0, 0.05)
plt.yticks(np.arange(0, 0.06, 0.01))
plt.ylabel("Error", fontsize=35)
plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$', fontsize=35)
plt.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Results/IsolationError.png")
"""

####
# Section 4.4.1 Boosted decision trees : Comparison XGB <-> NN
####

"""
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

fig = plt.figure(figsize=(30,20))
plt.plot(features_NN.values, accuracies_NN.values, marker="o", label="Neural Networks", linewidth=8)
plt.plot(features_NN.values, [float(accuracyDict[i]) for i in features_NN], marker="o", label="XGBoost", linewidth=8)
plt.xlabel("#Features", fontsize=60)
plt.ylabel("Accuracy [%]", fontsize=60)
plt.ylim([0,100])
plt.legend(loc=0, fontsize=50)
plt.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Results/Compare_XGB_NN.png")
"""

####
# Section 4.4.2 Chi2 template fit : Comparison Chi2 <-> NN
####

###
# Compare to Transverse
###

"""
fig = plt.figure()
dfChi = pd.read_csv("../data/Chi2Andreas.csv")
plt.errorbar((dfChi.m0+dfChi.m1)/2, dfChi.bg, xerr=(dfChi.m1-dfChi.m0)/2, yerr=dfChi.bg_err, label=r"Bg $(\chi^2)$", fmt="o", color='#7F99B0')
plt.errorbar((dfChi.m0+dfChi.m1)/2, dfChi.signal, xerr=(dfChi.m1-dfChi.m0)/2, yerr=dfChi.signal_err, label=r"Sg $(\chi^2)$", fmt='o', color="#003362")


dfTra = pd.read_csv("../data/dataPredictionTransverse.csv")
dfTra.m0 = dfTra.m0/1000
dfTra.m1 = dfTra.m1/1000
plt.errorbar((dfTra.m0+dfTra.m1)/2, dfTra.bg, fmt="s", xerr=(dfTra.m1-dfTra.m0)/2, yerr=dfTra.signal_err, label="Bg (NN-Transverse)", color="#FF9999", markersize=10)
plt.errorbar((dfTra.m0+dfTra.m1)/2, dfTra.signal, fmt="s", xerr=(dfTra.m1-dfTra.m0)/2, yerr=dfTra.signal_err, label="Sg (NN-Transverse)", color="#660000", markersize=10)

plt.legend(loc=7, fontsize=20)
plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$')
plt.ylabel("Fraction of events")
plt.xscale("log")
plt.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Results/CompareChiTransverse.png")

###
# Compare to Isolation
###

plt.figure()
plt.errorbar((dfChi.m0+dfChi.m1)/2, dfChi.bg, xerr=(dfChi.m1-dfChi.m0)/2, yerr=dfChi.bg_err, label=r"Bg $(\chi^2)$", fmt="o", color='#7F99B0')
plt.errorbar((dfChi.m0+dfChi.m1)/2, dfChi.signal, xerr=(dfChi.m1-dfChi.m0)/2, yerr=dfChi.signal_err, label=r"Sg $(\chi^2)$", fmt='o', color="#003362")

dfIso = pd.read_csv("../data/dataPredictionIsolation.csv")
dfIso.m0 = dfIso.m0/1000
dfIso.m1 = dfIso.m1/1000
plt.errorbar((dfIso.m0+dfIso.m1)/2, dfIso.bg, fmt="s", xerr=(dfIso.m1-dfIso.m0)/2, yerr=dfIso.signal_err, label="Bg (NN-Isolation)", color="#FF9999", markersize=10)
plt.errorbar((dfIso.m0+dfIso.m1)/2, dfIso.signal, fmt="s", xerr=(dfIso.m1-dfIso.m0)/2, yerr=dfIso.signal_err, label="Sg (NN-Isolation)", color="#660000", markersize=10)


plt.legend(loc=7, fontsize=20)
plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$')
plt.ylabel("Fraction of events")
plt.xscale("log")
plt.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Results/CompareChiIsolation.png")

###
# Difference to Transverse features
###

diff = pd.DataFrame()
for col in dfTra.columns.values:
    if col == "m0" or col == "m1":
        diff[col] = dfChi[col]
    elif col == "signal" or col == "bg":
        diff[col] = dfChi[col] - dfTra[col]
    else:
        diff[col] = np.abs(dfChi[col] -  dfTra[col])

fig = plt.figure()
plt.errorbar((diff.m0+diff.m1)/2.05, diff.bg, fmt="D", xerr=(diff.m1-diff.m0)/2, yerr=diff.signal_err, label="Background")
plt.errorbar((diff.m0+diff.m1)/2, diff.signal, fmt="D", xerr=(diff.m1-diff.m0)/2, yerr=diff.signal_err, label="Signal")

plt.legend(loc=1, fontsize=20)
plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$')
plt.ylabel("Difference in fraction")
plt.axhline(0, linestyle="--", color="k", alpha=0.5)
plt.yticks(np.arange(-0.4, 0.4, 0.1))
plt.xscale("log")
plt.ylim(-0.4, 0.4)
fig.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Results/CompareDiffChiTransverse.png")

###
# Difference to Isolation features
###


diff = pd.DataFrame()
for col in dfIso.columns.values:
    if col == "m0" or col == "m1":
        diff[col] = dfChi[col]
    elif col == "signal" or col == "bg":
        diff[col] = dfChi[col] - dfIso[col]
    else:
        diff[col] = np.abs(dfChi[col] - dfIso[col])

fig = plt.figure()
plt.errorbar((diff.m0+diff.m1)/2.05, diff.bg, fmt="D", xerr=(diff.m1-diff.m0)/2, yerr=diff.signal_err, label="Background")
plt.errorbar((diff.m0+diff.m1)/2, diff.signal, fmt="D", xerr=(diff.m1-diff.m0)/2, yerr=diff.signal_err, label="Signal")

plt.legend(loc=1, fontsize=20)
plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$')
plt.ylabel("Difference in fraction")
plt.axhline(0, linestyle="--", color="k", alpha=0.5)
plt.yticks(np.arange(-0.4, 0.4, 0.1))
plt.xscale("log")
plt.ylim(-0.4, 0.4)
fig.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Results/CompareDiffChiIsolation.png")
"""

####
# Section 4.4.3 Systematic uncertainties : Fraction Mag down
####

"""
np.random.seed(42)

df = pd.read_csv("../data/dataPredictionTransverse.csv")
df.m0 = df.m0/1000
df.m1 = df.m1/1000
# fig = plt.figure()
# plt.errorbar((df.m0+df.m1)/2, df.signal, fmt="P", xerr=(df.m1-df.m0)/2, yerr=df.signal_err, label="Sg (NN-Transverse)", color="#660000")
# plt.errorbar((df.m0+df.m1)/2, df.bg, fmt="P", xerr=(df.m1-df.m0)/2, yerr=df.signal_err, label="Bg (NN-Transverse)", color="#FF9999", markersize=15)

oldDf = df.copy()
for col in df.columns.values:
    newCol = []
    if col != "m0" and col != "m1":
        for val in df[col]:
            newCol.append(np.random.normal(loc=val, scale=0.01, size=1))
        df[col] = np.array(newCol)

# plt.errorbar((df.m0+df.m1)/2, df.signal, fmt="D", xerr=(df.m1-df.m0)/2, yerr=df.signal_err, label="Sg (NN-Isolation)", color="#006633")
# plt.errorbar((df.m0+df.m1)/2, df.bg, fmt="D", xerr=(df.m1-df.m0)/2, yerr=df.signal_err, label="Bg (NN-Isolation)", color="#99FFCC")
# 
# plt.legend(loc=7, fontsize=20)
# plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$')
# plt.ylabel("Fraction of events")


###
# Difference MagUp <--> MagDown
###

diff = pd.DataFrame()
for col in df.columns.values:
    if col == "m0" or col == "m1":
        diff[col] = df[col]
    elif col == "signal" or col == "bg":
        diff[col] = oldDf[col] - df[col]
    else:
        diff[col] = np.sqrt(oldDf[col]**2 + df[col]**2)

df = diff
fig = plt.figure()
plt.errorbar((df.m0+df.m1)/2, df.signal, fmt="D", xerr=(df.m1-df.m0)/2, yerr=df.signal_err, label="Signal")
plt.errorbar((df.m0+df.m1)/2.05, df.bg, fmt="D", xerr=(df.m1-df.m0)/2, yerr=df.signal_err, label="Background")

plt.legend(loc=1, fontsize=20)
plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$')
plt.ylabel("Difference in fraction")
plt.axhline(0, linestyle="--", color="k", alpha=0.5)
plt.yticks(np.arange(-0.4, 0.4, 0.1))
plt.xscale("log")
plt.ylim(-0.4, 0.4)
fig.savefig("/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/Thesis/figures/Results/CompareMagPolarization.png")
"""


