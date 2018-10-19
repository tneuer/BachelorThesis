#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : generate_helpful_plots.py
    # Creation Date : Die 07 Aug 2018 21:41:29 CEST
    # Last Modified : Mon 17 Sep 2018 22:02:47 CEST
    # Description : Script were useful plotting utilities are generated.
"""
#==============================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.style.use("./matplotlib_LHCb.mplstyle")

def compare_activations(actvs_dict, x_range=np.linspace(-2, 2, 300)):
    """ Generate plot comparing activations and their derivatives

    Parameters
    ----------
    actvs_dict : dic
        dictionary with key=name of activation function. The value is a function which
        returns the value of the activation and its derivative.
    x_range : np.array
        x values to be plotted
    Returns
    -------
    """
    path = "/media/tneuer/TOSHIBA EXT/MyData/Uni/BachelorThesis/workingDir/figures/NN"

    y_vals = []
    dy_vals = []
    names = []
    for key in actvs_dict:
        y = np.array([actvs_dict[key](x) for x in x_range])
        y_val = y[:, 0]
        dy_val = y[:, 1]

        y_vals.append(y_val)
        dy_vals.append(dy_val)
        names.append(key)

    plt.figure()
    for y, name in zip(y_vals, names):
        plt.plot(x_range, y, label=name)

    plt.legend(prop={'size': 20})
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.figure()
    for dy, name in zip(dy_vals, names):
        plt.plot(x_range, dy, label=name)

    plt.legend(prop={'size': 20})
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("df(x)/dx")
    plt.show()

    # plt.savefig(path)



def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    dy = y * (1 - y)
    return y, dy

def tanh(x):
    y = np.tanh(x)
    dy = 1 - y**2
    return y, dy

def softsign(x):
    y = x / (1 + np.abs(x))
    dy = 1 / (1 + np.abs(x))**2
    return y, dy

def relu(x):
    y = max(0, x)
    dy = 1 if x>0 else 0
    return y, dy

def prelu(x, a=0.1):
    y = x if x>0 else a*x
    dy = 1 if x>0 else a
    return y, dy

def elu(x, a=0.6):
    y = a*(np.exp(x) - 1) if x<0 else x
    dy = 1 if x>0 else y+a
    return y, dy


if __name__ == "__main__":
    actvs = [sigmoid, tanh, relu, elu, prelu]
    names = ["Sigmoid", "Tanh", "Relu"]#, "Elu", "PRelu"]
    actvs_dict = {}

    for name, f in zip(names, actvs):
        actvs_dict[name] = f

    compare_activations(actvs_dict)




