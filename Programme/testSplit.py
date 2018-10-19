#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : testSplit.py
    # Creation Date : Mit 21 Feb 2018 15:11:30 CET
    # Last Modified : Die 20 Mär 2018 12:34:54 CET
    # Description : In this script the train/val/test split is performed and a subsample is created. Also global cuts are performed.
    Special operations:
        - Insert PT
"""
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re

from root_pandas import read_root
from sklearn.model_selection import train_test_split
from pipeliner import InputPT

####
# Do for MC Sample
####
data = read_root("../data/dy_tuple_12_md_MC.root")

#### Cuts
instances = []
instances.append(data.shape[0])
data = data.drop(data[data["Z0_MM"] > 120000].index)
instances.append(data.shape[0])
data = data.drop(data[data["Z0_MM"] < 10000].index)
instances.append(data.shape[0])
data = data.drop(data[data["nSPDHits"] > 600].index)
instances.append(data.shape[0])
data = data.drop(data[data["nTrack"] > 150].index)
instances.append(data.shape[0])
data = data.drop(data[data["y"] > 4.5].index)
instances.append(data.shape[0])
data = data.drop(data[data["y"] < 2].index)
instances.append(data.shape[0])
data = data.drop(data[data["muminus_TrPChi2"] < 0.001].index)
instances.append(data.shape[0])
data = data.drop(data[data["muplus_TrPChi2"] < 0.001].index)
instances.append(data.shape[0])

cut = 1000
data = data.drop(data[data["muminus_MINIP"] > cut].index)
instances.append(data.shape[0])
data = data.drop(data[data["muplus_MINIP"] > cut].index)
instances.append(data.shape[0])

print(instances)
print(np.array(instances)[:-1]-np.array(instances)[1:])
tvars = [c for c in data.columns.values if "tracks_" in c]

a = 0
def clean_bad_IP(row):
    global a
    a += 1
    if a % 1000 == 0:
        print(a, "/", instances[-1])
    good_ix = np.where(row["tracks_IP"]<cut)[0]
    row["nTrack"] = len(good_ix)
    for t in tvars:
        row[t] = row[t][good_ix]

    return row

data = data.apply(clean_bad_IP, axis=1)
data["labels"] = 1
print("Dropped events signal IP: ", instances[-2]-instances[-1])
print("Dropped events signal all: ", instances[0]-instances[-1])

#### Calculate PT
print("Calculate PT...")

pt = InputPT()
data = pt.transform(data)

#### Create train-test-set
train, test = train_test_split(data, test_size=0.2, random_state=42)

train, val = train_test_split(train, test_size=0.1, random_state=31415)

print("Sizes MC (train/val/test): ", train.shape, val.shape, test.shape)
with open("../data/Train_md_MC_{}.pickle".format(train.shape[0]), "wb") as f:
    pickle.dump(train, f)

with open("../data/Val_md_MC_{}.pickle".format(val.shape[0]), "wb") as f:
    pickle.dump(val, f)

with open("../data/Test_md_MC_{}.pickle".format(test.shape[0]), "wb") as f:
    pickle.dump(test, f)

#### Create smaller dataframe out of trainset

sample = train.sample(200000, random_state=1103)

with open("../data/Train_md_MC_200000.pickle", "wb") as f:
    pickle.dump(sample, f)



####
# Do for heavy flavor set
####

data = read_root("../data/dy_tuple_12_md_hf.root")

#### Cuts
instances = []
instances.append(data.shape[0])
data = data.drop(data[data["Z0_MM"] > 120000].index)
instances.append(data.shape[0])
data = data.drop(data[data["Z0_MM"] < 10000].index)
instances.append(data.shape[0])
data = data.drop(data[data["nSPDHits"] > 600].index)
instances.append(data.shape[0])
data = data.drop(data[data["nTrack"] > 150].index)
instances.append(data.shape[0])
data = data.drop(data[data["y"] > 4.5].index)
instances.append(data.shape[0])
data = data.drop(data[data["y"] < 2].index)
instances.append(data.shape[0])
data = data.drop(data[data["muminus_TrPChi2"] < 0.001].index)
instances.append(data.shape[0])
data = data.drop(data[data["muplus_TrPChi2"] < 0.001].index)
instances.append(data.shape[0])

cut = 1000
data = data.drop(data[data["muminus_MINIP"] > cut].index)
instances.append(data.shape[0])
data = data.drop(data[data["muplus_MINIP"] > cut].index)
instances.append(data.shape[0])

print(instances)
print(np.array(instances)[:-1]-np.array(instances)[1:])
tvars = [c for c in data.columns.values if "tracks_" in c]

a = 0
def clean_bad_IP(row):
    global a
    a += 1
    if a % 1000 == 0:
        print(a, "/", instances[-1])
    good_ix = np.where(row["tracks_IP"]<cut)[0]
    row["nTrack"] = len(good_ix)
    for t in tvars:
        row[t] = row[t][good_ix]

    return row

data = data.apply(clean_bad_IP, axis=1)
data["labels"] = 0
print("Dropped events background IP: ", instances[-2]-instances[-1])
print("Dropped events background all: ", instances[0]-instances[-1])

#### Calculate PT
print("Calculate PT...")

pt = InputPT()
data = pt.transform(data)
data["weight"] = 1

#### Create train-test-set
train, test = train_test_split(data, test_size=0.2, random_state=42)

train, val = train_test_split(train, test_size=0.1, random_state=31415)

print("Sizes hf (train/val/test): ", train.shape, val.shape, test.shape)
with open("../data/Train_md_hf_{}.pickle".format(train.shape[0]), "wb") as f:
    pickle.dump(train, f)

with open("../data/Val_md_hf_{}.pickle".format(val.shape[0]), "wb") as f:
    pickle.dump(val, f)

with open("../data/Test_md_hf_{}.pickle".format(test.shape[0]), "wb") as f:
    pickle.dump(test, f)

#### Create smaller dataframe out of trainset

sample = train.sample(200000, random_state=11)

with open("../data/Train_md_hf_200000.pickle", "wb") as f:
    pickle.dump(sample, f)




