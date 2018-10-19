#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : eda.py
    # Creation Date : Mit 21 Feb 2018 23:52:47 CET
    # Last Modified : Die 28 Aug 2018 16:24:59 CEST
    # Description : Basic exploratory data analysis
"""
#==============================================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import h5py

from baUtils import get_sg_and_bg
from pipeliner import logtransform_standard_pipeline

plt.style.use("./matplotlib_LHCb.mplstyle")

"""
with open("../data/Train_mu_hf_960341.pickle", "rb") as f:
    bg = pickle.load(f)

with open("../data/Train_mu_MC_1407488.pickle", "rb") as f:
    signal = pickle.load(f)

####
# Plot feature histograms
####
variables = ["Z0_MM", "y", "nSPDHits", "nTrack", "muminus_PT", "muminus_TrEta", "muminus_MINIP", "isolation", "tracks_PZ", "tracks_IP", "tracks_IPCHI2", "tracks_PT"]
logscale = [True, False, False, False, True, False, False, True, True, True, True, True]
nrow = 3
ncol = int(np.ceil(len(variables) / nrow))

f, ax = plt.subplots(nrow, ncol)
count = 1

for variable, log in zip(variables, logscale):
    print(variable)
    plt.subplot(nrow, ncol, count)
    plt.xlabel(variable)
    plt.ylabel("count")
    if variable in ["tracks_PX", "tracks_PZ", "tracks_PZ", "tracks_IP", "tracks_IPCHI2", "tracks_isMuon", "tracks_charge", "tracks_PT"]:
        sig = [l for sublist in signal[variable] for l in sublist]
        bkg = [l for sublist in bg[variable] for l in sublist]

        plt.hist(sig, bins=50, histtype="step", label="signal")
        plt.hist(bkg, bins=50, histtype="step", label="background")
    else:
        plt.hist(signal[variable], bins=50, histtype="step", label="signal")
        plt.hist(bg[variable], bins=50, histtype="step", label="background")
    if log:
        plt.yscale("log")
        plt.grid(color="g", linewidth=0.5)
    else:
        plt.grid()
    plt.legend()
    plt.draw()
    count += 1
"""

####
# Compare logscale
####
"""
print("\n\n")
variables = ["muminus_PT", "muminus_PZ", "tracks_PT", "tracks_IPCHI2"]
ncol = 2
nrow = len(variables)
f, ax = plt.subplots(nrow, ncol)
count = 1

for variable in variables:
    print(variable)
    plt.subplot(nrow, ncol, count)
    plt.xlabel(variable)
    plt.ylabel(variable)
    if variable in ["tracks_PX", "tracks_PZ", "tracks_PZ", "tracks_IP", "tracks_IPCHI2", "tracks_isMuon", "tracks_charge", "tracks_PT"]:
        sig = [l for sublist in signal[variable] for l in sublist]
        bkg = [l for sublist in bg[variable] for l in sublist]

        plt.hist(sig, bins=50, histtype="step", label="signal")
        plt.hist(bkg, bins=50, histtype="step", label="background")
        plt.grid()
        plt.legend()

        count += 1
        plt.subplot(nrow, ncol, count)
        plt.xlabel(variable)
        plt.ylabel(variable)
        sig = np.log(sig)
        bkg = np.log(bkg)

        plt.hist(sig, bins=50, histtype="step", label="signal")
        plt.hist(bkg, bins=50, histtype="step", label="background")
        plt.grid()
        plt.legend()
        plt.title("LogScale")
    else:
        plt.hist(signal[variable], bins=50, histtype="step", label="signal")
        plt.hist(bg[variable], bins=50, histtype="step", label="background")
        plt.grid()
        plt.legend()

        count += 1
        plt.subplot(nrow, ncol, count)
        plt.xlabel(variable)
        plt.ylabel(variable)

        plt.hist(np.log(signal[variable]), bins=50, histtype="step", label="signal")
        plt.hist(np.log(bg[variable]), bins=50, histtype="step", label="background")
        plt.grid()
        plt.legend()
    plt.draw()
    count += 1
"""



####
# Filter bad IP tracks
####
"""
print("\n\n")
variables = ["Z0_MM", "y", "nSPDHits", "nTrack", "muminus_PT", "muminus_TrEta", "muminus_MINIP", "isolation", "tracks_PZ", "tracks_IP", "tracks_IPCHI2", "tracks_PT"]
logscale = [True, False, False, False, True, False, True, True, True, True, True, True]
nrow = 3
ncol = int(np.ceil(len(variables) / nrow))

f = plt.figure()
count = 1

cut = 40000
signal2 = signal.iloc[signal["muminus_MINIP"].values<cut, :]
signal2 = signal2.iloc[signal2["muplus_MINIP"].values<cut, :]
bg2 = bg.iloc[bg["muminus_MINIP"].values<cut, :]
bg2 = bg2.iloc[bg2["muplus_MINIP"].values<cut, :]

drop_ix_sg = np.array([l for sublist in signal2["tracks_IP"] for l in sublist])
drop_ix_bg = np.array([l for sublist in bg2["tracks_IP"] for l in sublist])

drop_ix_sg = np.where(drop_ix_sg > cut)[0]
drop_ix_bg = np.where(drop_ix_bg > cut)[0]

print("Dropped events [Signal / BG]: ", signal.shape[0]-signal2.shape[0], bg.shape[0]-bg2.shape[0])
print("Dropped tracks [Signal / BG]: ", len(drop_ix_sg), len(drop_ix_bg))

for variable, log in zip(variables, logscale):
    print(variable)
    plt.subplot(nrow, ncol, count)
    plt.xlabel(variable)
    plt.ylabel("count")
    if variable in ["tracks_PX", "tracks_PZ", "tracks_PZ", "tracks_IP", "tracks_IPCHI2", "tracks_isMuon", "tracks_charge", "tracks_PT"]:
        sig = [l for sublist in signal2[variable] for l in sublist]
        bkg = [l for sublist in bg2[variable] for l in sublist]

        sig = [s for i, s in enumerate(sig) if i not in drop_ix_sg]
        bkg = [b for i, b in enumerate(bkg) if i not in drop_ix_bg]

        ax = f.add_subplot(nrow, ncol, count)
        ax.hist(sig, bins=50, histtype="step", label="signal")
        ax.hist(bkg, bins=50, histtype="step", label="background")
    else:
        ax = f.add_subplot(nrow, ncol, count)
        ax.hist(signal2[variable], bins=50, histtype="step", label="signal")
        ax.hist(bg2[variable], bins=50, histtype="step", label="background")
    if log:
        plt.yscale("log")
        plt.grid(color="g", linewidth=0.5)
    else:
        plt.grid()
    plt.legend()
    plt.draw()
    count += 1

plt.suptitle("Dropped events [Signal / BG]: {} / {} \nDropped tracks [Signal / BG]: {} / {}".format(signal.shape[0]-signal2.shape[0], bg.shape[0]-bg2.shape[0], len(drop_ix_sg), len(drop_ix_bg)))
"""

####
# Plot transverse features
####
"""
print("\n\n")
variables = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", 'nTrack', 'tracks_PT', 'tracks_eta', 'tracks_phi', 'tracks_IP', 'tracks_IPCHI2', 'tracks_charge', 'tracks_isMuon']
logstrafo = [True, False, False, True, False, True, False, False, True, True, False, False]
nrow = 3
ncol = int(np.ceil(len(variables) / nrow))

f = plt.figure()
count = 1

for variable, log in zip(variables, logstrafo):
    print(variable)
    plt.subplot(nrow, ncol, count)
    plt.xlabel(variable)
    plt.ylabel("count")
    if variable in ["tracks_PX", "tracks_PZ", "tracks_PZ", "tracks_IP", "tracks_IPCHI2", "tracks_isMuon", "tracks_charge", "tracks_PT", "tracks_eta", "tracks_phi"]:
        sig = np.array([l for sublist in signal[variable] for l in sublist])
        bkg = np.array([l for sublist in bg[variable] for l in sublist])

        if log:
            sig = np.log(sig)
            bkg = np.log(bkg)
        ax = f.add_subplot(nrow, ncol, count)
        ax.hist(sig, bins=50, histtype="step", label="signal")
        ax.hist(bkg, bins=50, histtype="step", label="background")
    else:
        ax = f.add_subplot(nrow, ncol, count)
        if log:
            ax.hist(np.log(signal[variable]), bins=50, histtype="step", label="signal")
            ax.hit(np.log(bg[variable]), bins=50, histtype="step", label="background")
        else:
            ax.hist(signal[variable], bins=50, histtype="step", label="signal")
            ax.hist(bg[variable], bins=50, histtype="step", label="background")

    plt.grid()
    plt.legend()
    plt.draw()
    count += 1
"""

####
# Plot Standardized Variables
####

"""
sgname = "../data/Train_mu_MC_1407488.pickle"
bgname = "../data/Train_mu_hf_960341.pickle"
data, labels = get_sg_and_bg(sgname, bgname, labels=True)

data = data[:1000000]
labels = labels[:1000000]

transverse_wo_SPD = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']

standard_features = transverse_wo_SPD[:7]
standard_log = ["muplus_PT", "muminus_PT"]

non_standard_features = transverse_wo_SPD[6:]
global_standardized = non_standard_features[1:-2]
non_standard_log = ["tracks_PT", "tracks_IPCHI2", "tracks_IP"]

scale_transformer_pipeline = logtransform_standard_pipeline(standard_features, standard_log,
	non_standard_features, non_standard_log, global_standardized)

data = scale_transformer_pipeline.fit_transform(data)

with open("../data/Standardizer_mu_{}.pickle".format(data.shape[0]), "wb") as f:
    pickle.dump(scale_transformer_pipeline, f)

newFeatures = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", "nTrackN", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']
standard_features[standard_features.index("nTrack")] = "nTrackN"
data = pd.DataFrame(data, columns=newFeatures)

signal = data.iloc[labels==1, :]
bg = data.iloc[labels==0, :]

print(data.shape)
print("\n\n")
variables = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", 'nTrackN', 'tracks_PT', 'tracks_eta', 'tracks_phi', 'tracks_IP', 'tracks_IPCHI2', 'tracks_charge', 'tracks_isMuon']
nrow = 3
ncol = int(np.ceil(len(variables) / nrow))

f = plt.figure()
count = 1

for variable in variables:
    print(variable)
    plt.subplot(nrow, ncol, count)
    plt.xlabel(variable)
    plt.ylabel("count")
    if "tracks" in variable:
        sig = np.array([l for sublist in signal[variable] for l in sublist])
        bkg = np.array([l for sublist in bg[variable] for l in sublist])

        ax = f.add_subplot(nrow, ncol, count)
        ax.hist(sig, bins=50, histtype="step", label="signal")
        ax.hist(bkg, bins=50, histtype="step", label="background")
    else:
        sig = np.array([l for l in signal[variable].values])
        bkg = np.array([l for l in bg[variable].values])

        ax = f.add_subplot(nrow, ncol, count)
        ax.hist(sig, bins=50, histtype="step", label="signal")
        ax.hist(bkg, bins=50, histtype="step", label="background")

    plt.grid()
    plt.legend()
    plt.draw()
    count += 1


plt.show()
# input()
"""


####
# Plot Data
####

"""
data = pd.read_pickle("../data/Data_mu_500000.pickle")

transverse_wo_SPD = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']

standard_features = transverse_wo_SPD[:7]
standard_log = ["muplus_PT", "muminus_PT"]

non_standard_features = transverse_wo_SPD[6:]
global_standardized = non_standard_features[1:-2]
non_standard_log = ["tracks_PT", "tracks_IPCHI2", "tracks_IP"]

scale_transformer_pipeline = logtransform_standard_pipeline(standard_features, standard_log,
	non_standard_features, non_standard_log, global_standardized)

data = scale_transformer_pipeline.fit_transform(data)

with open("../data/Standardizer_mu_{}.pickle".format(data.shape[0]), "wb") as f:
    pickle.dump(scale_transformer_pipeline, f)

newFeatures = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", "nTrackN", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']
standard_features[standard_features.index("nTrack")] = "nTrackN"
data = pd.DataFrame(data, columns=newFeatures)

print(data.shape)
print("\n\n")
variables = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", 'nTrackN', 'tracks_PT', 'tracks_eta', 'tracks_phi', 'tracks_IP', 'tracks_IPCHI2', 'tracks_charge', 'tracks_isMuon']
nrow = 3
ncol = int(np.ceil(len(variables) / nrow))

f = plt.figure()
count = 1

for variable in variables:
    print(variable)
    plt.subplot(nrow, ncol, count)
    plt.xlabel(variable)
    plt.ylabel("count")
    if "tracks" in variable:
        sig = np.array([l for sublist in data[variable] for l in sublist])

        ax = f.add_subplot(nrow, ncol, count)
        ax.hist(sig, bins=50, histtype="step", label="Data")
    else:
        sig = np.array([l for l in data[variable].values])

        ax = f.add_subplot(nrow, ncol, count)
        ax.hist(sig, bins=50, histtype="step", label="Data")

    plt.grid()
    plt.legend()
    plt.draw()
    count += 1


plt.show()
# input()
"""



####
# Compare Simulation/TrainData & True data
####

# sgname = "../data/Val_mu_MC_156388.pickle"
# bgname = "../data/Val_mu_hf_106705.pickle"
# 
# with open(bgname, "rb") as f:
#     val_bg = pickle.load(f)
# 
# with open(sgname, "rb") as f:
#     val_sg = pickle.load(f)
# 
# with open("../data/featureNamesDict.pickle", "rb") as f:
#     fNames = pickle.load(f)
# 
# with open("../data/Data_mu_500000.pickle", "rb") as f:
#     data = pickle.load(f)
# 
# end_bg = val_bg["Z0_ENDVERTEX_CHI2"]
# end_sg = val_sg["Z0_ENDVERTEX_CHI2"]
# 
# var = "Z0_MM"
# 
# simulation, labels = get_sg_and_bg(sgname, bgname, shuffle=True, random_state=42)
# truesignal = np.where(labels==1)[0]
# truebg = np.where(labels==0)[0]
# 
# x = [simulation[var][labels==0], simulation[var][labels==1]]
# 
# bincuts = np.array([10, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 60.0, 120.0])*1000
# fig = plt.figure()
# plt.hist(data[var], histtype="step", density=True, bins=bincuts, label = "data")
# plt.hist(simulation[var], histtype="step", density=True, bins=bincuts, label = "sg+bg")
# plt.hist(x, histtype="step", density=True, bins=bincuts, label = ["bg", "sg"], linestyle="dashed")
# plt.xlabel("Z0_MM")
# plt.title("Compare data and simulation")
# plt.legend()
# # plt.yscale("log")
# 
# bincuts = 50
# fig = plt.figure()
# plt.hist(data[var], histtype="step", density=True, bins=bincuts, label = "data")
# plt.hist(simulation[var], histtype="step", density=True, bins=bincuts, label = "sg+bg")
# plt.hist(x, histtype="step", density=True, bins=bincuts, label = ["bg", "sg"], linestyle="dashed")
# plt.xlabel("Z0_MM")
# plt.title("Compare data and simulation")
# plt.legend()
# # plt.yscale("log")
# plt.show()


####
# Compare Simulation/TrainData & True data per mass bins
####

# sgname = "../data/Val_mu_MC_156388.pickle"
# bgname = "../data/Val_mu_hf_106705.pickle"
# 
# with h5py.File("../data/Processed_Val_mu_263093.h5", "r") as f:
#     dataVal = f["Val"][:]
# 
# with open("../data/Val_mu_labels_263093.pickle", "rb") as f:
#     labels = pickle.load(f)
#     labels = labels[:dataVal.shape[0]]
# 
# dataValorig = get_sg_and_bg(sgname, bgname, labels=False)
# dataValorig = dataValorig.iloc[:dataVal.shape[0], :]
# 
# with h5py.File("../data/Processed_Data_mu_500000.h5", "r") as f:
#     data = f["Data"][:dataVal.shape[0]]
# 
# with open("../data/Data_mu_500000.pickle", "rb") as f:
#     dataorig = pickle.load(f)
#     dataorig = dataorig.iloc[:200000, :]
# 
# with open("../data/featureNamesDict.pickle", "rb") as f:
#     fNames = pickle.load(f)
# 
# variables = ["muminus_PT"]#, "muminus_TrEta", "nTrackN", "tracks_PT0", "tracks_IPCHI20", "tracks_IP0"]
# 
# bincuts = np.array([10, 11, 12.0, 13.0, 14.0, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 60.0, 120.0])*1000
# 
# nrow = 3
# ncol = 4
# 
# f = plt.figure(figsize=(50, 30))
# plt.title("muminus_PT")
# idx = fNames["muminus_PT"]
# i = 0
# for cutLow, cutHigh in zip(bincuts[:-1], bincuts[1:]):
#     dataIndHigh = dataorig["Z0_MM"] < cutHigh
#     dataIndLow = dataorig["Z0_MM"] > cutLow
#     dataInd = np.logical_and(dataIndLow, dataIndHigh)
# 
#     ax = f.add_subplot(nrow, ncol, i+1)
#     x = dataorig.values[dataInd, idx]
# 
#     x = x.tolist()
#     ax.hist(x, density=True, histtype="step", bins=50)
#     plt.ylabel("Bin ({}, {}) GeV".format(bincuts[i]/1000, bincuts[i+1]/1000))
#     plt.grid()
#     plt.xlabel("muminus_PT")
#     plt.grid()
# 
#     print("here")
#     plt.figure()
#     plt.hist(x)
#     plt.ylabel("Bin ({}, {}) GeV".format(bincuts[i]/1000, bincuts[i+1]/1000))
#     plt.title()
# 
#     i += 1
# 
# plt.show()
# raise
#

# for var in variables:
#     print("Processing", var)
#     f = plt.figure(figsize=(50, 30))
#     plt.title(var)
#     idx = fNames[var]
#     i = 0
#     for cutLow, cutHigh in zip(bincuts[:-1], bincuts[1:]):
#         simValIndHigh = dataValorig["Z0_MM"] < cutHigh
#         simValIndLow = dataValorig["Z0_MM"] > cutLow
#         simValInd = np.logical_and(simValIndLow, simValIndHigh)
# 
#         dataIndHigh = dataorig["Z0_MM"] < cutHigh
#         dataIndLow = dataorig["Z0_MM"] > cutLow
#         dataInd = np.logical_and(dataIndLow, dataIndHigh)
# 
#         ax = f.add_subplot(nrow, ncol, i+1)
#         print(cutLow, cutHigh)
#         print(dataorig[var][dataInd].shape)
#         x = [np.log(dataorig[var][dataInd]+1), np.log(dataValorig[var][simValInd]+1)]
#         y = [np.log(dataValorig[var][simValInd][labels[simValInd]==0]+1), np.log(dataValorig[var][simValInd][labels[simValInd]==0]+1)]
#         ax.hist(x, density=True, stacked=False, histtype="step", label=["Data", "Val"], bins=50)
#         ax.hist(y, density=True, stacked=False, histtype="step", label=["Bg", "Sg"], bins=50, linestyle="--", alpha=0.2)
#         plt.ylabel("Bin ({}, {}) GeV".format(bincuts[i]/1000, bincuts[i+1]/1000))
#         plt.grid()
#         plt.legend()
#         plt.xlabel(var)
#         i+=1
# 
#     # plt.savefig("../figures/MassBinDistributions_{}.png".format(var))
# plt.show()




# binlow = bincuts[-2]
# binhigh = bincuts[-1]
# idx = fNames["muminus_PT"]
# 
# datalow = dataorig["Z0_MM"]>binlow
# datahigh = dataorig["Z0_MM"]<binhigh
# dataind = np.logical_and(datalow, datahigh)
# 
# plt.figure()
# 
# datalow2 = dataValorig["Z0_MM"]>binlow
# datahigh2 = dataValorig["Z0_MM"]<binhigh
# dataind2 = np.logical_and(datalow2, datahigh2)
# 
# plt.hist([data[dataind, idx], dataVal[dataind2, idx]], density=True, bins=50, histtype="step")
# plt.show()

####
# Correlation to Z0_Endvertex_CHI2
####

# sgname = "../data/Val_mu_MC_156388.pickle"
# bgname = "../data/Val_mu_hf_106705.pickle"
# 
# with open("../data/featureNamesDict.pickle", "rb") as f:
#     fNames = pickle.load(f)
# 
# # with open("../data/Data_mu_500000.pickle", "rb") as f:
# #     data = pickle.load(f)
# 
# sim, labels = get_sg_and_bg(sgname, bgname, labels=True)
# 
# compTo = "Z0_ENDVERTEX_CHI2"
# 
# # corrDF = pd.DataFrame({"compTo": sim[compTo]})
# corrDFSg = pd.DataFrame({"compTo": sim[compTo][labels==1]})
# corrDFBg = pd.DataFrame({"compTo": sim[compTo][labels==0]})
# 
# covariable = ["muminus_PT", "muminus_TrEta", "nTrack", "tracks_IPCHI2", "tracks_PT", "tracks_IP"]
# 
# nrow = 2
# ncol = 3
# 
# f = plt.figure()
# 
# for i, var in enumerate(covariable):
#     print("Processing ", var)
#     if "tracks" in var:
#         corrVar = np.array([tracks[0] if len(tracks)!=0 else 0 for tracks in sim[var]])
#         corrVarSg = corrVar[labels==1]
#         corrVarBg = corrVar[labels==0]
#         corrDFSg["var"] = corrVarSg
#         corrDFBg["var"] = corrVarBg
#     else:
#         corrDFSg["var"] = sim[var][labels==1]
#         corrDFBg["var"] = sim[var][labels==0]
#     corrSg = corrDFSg.corr()
#     corrBg = corrDFBg.corr()
#     ax = f.add_subplot(nrow, ncol, i+1)
#     plt.title("SgCorr: {} - BgCorr {}".format(np.round(corrSg.iloc[0, 1], 2), np.round(corrBg.iloc[0, 1], 2)))
#     plt.plot(corrDFSg["compTo"], corrDFSg["var"], "o", alpha=0.3, label="Sg")
#     plt.plot(corrDFBg["compTo"], corrDFBg["var"], "o", alpha=0.3, label="Bg")
#     plt.legend()
#     plt.xlabel(compTo)
#     plt.ylabel(var)
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.grid()
#     plt.grid()
# 
# plt.show()


