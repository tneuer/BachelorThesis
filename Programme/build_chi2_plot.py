#!/home/thomas/anaconda2/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : build_chi2_plot.py
    # Creation Date : Son 19 Aug 2018 10:24:43 CEST
    # Last Modified : Mon 20 Aug 2018 00:50:39 CEST
    # Description :
"""
#==============================================================================

import pickle

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("./matplotlib_LHCb.mplstyle")

df = pd.read_csv("../data/Chi2Andreas.csv")
plt.errorbar((df.m0+df.m1)/2, df.signal, xerr=(df.m1-df.m0)/2, yerr=df.signal_err, label="Signal", fmt='o')
plt.errorbar((df.m0+df.m1)/2, df.bg, xerr=(df.m1-df.m0)/2, yerr=df.bg_err, label="Background", fmt='o')


df = pd.read_csv("../data/dataPredictionTransverse.csv")
plt.plot((df.m0+df.m1)/2000, df.signal, "s", color="green", label="Signal")
plt.plot((df.m0+df.m1)/2000, df.bg, "s", color="green", label="Background")


df = pd.read_csv("../data/dataPredictionIsolation.csv")
plt.plot((df.m0+df.m1)/2000, df.signal, "*", color="blue", label="Signal")
plt.plot((df.m0+df.m1)/2000, df.bg, "*", color="blue", label="Background")


plt.legend()
plt.xlabel(r'$M_{\mu\mu} [GeV/c^2]$')
plt.ylabel("Fraction of events")
plt.show()

