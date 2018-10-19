#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : baUtils.py
    # Creation Date : Sam 24 Feb 2018 16:07:59 CET
    # Last Modified : Mit 29 Aug 2018 17:34:18 CEST
    # Description : Some often used uitlity functions
"""
#==============================================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def get_sg(fname, labels=True):
    sg = pd.read_pickle(fname)

    if labels:
        lbls = np.ones(sg.shape[0])
        return sg, lbls
    else:
        return sg

def get_bg(fname, labels=True):
    bg = pd.read_pickle(fname)

    if labels:
        lbls = np.zeros(bg.shape[0])
        return bg, lbls
    else:
        return bg

def get_sg_and_bg(file_sig, file_bkg, labels=True, shuffle=True, random_state=42):
    if labels:
        sgx, sgy = get_sg(file_sig, labels=True)
        bgx, bgy = get_bg(file_bkg, labels=True)
    else:
        sgx = get_sg(file_sig, labels=False)
        bgx = get_bg(file_bkg, labels=False)

    miss_in_bg = [col for col in sgx.columns.values if col not in bgx.columns.values]

    miss_in_signal = [col for col in bgx.columns.values if col not in sgx.columns.values]

    print("Dropped in bg: {}\n\n Dropped in signal: {}".format(miss_in_bg, miss_in_signal))
    sgx.drop(miss_in_bg, inplace = True, axis = 1)
    bgx.drop(miss_in_signal, inplace = True, axis = 1)
    data = pd.concat((sgx, bgx))
    indices = np.arange(data.shape[0])

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
        data = data.iloc[indices, :]

    if labels:
        labels = np.concatenate((sgy, bgy))[indices]
        return data, labels
    else:
        return data




