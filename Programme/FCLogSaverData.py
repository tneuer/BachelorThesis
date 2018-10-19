import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as mpatches
import pickle
import time
import os.path
import argparse
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import tensorflow as tf
import h5py
import re

plt.style.use("./matplotlib_LHCb.mplstyle")

with h5py.File("../data/Processed_Data_mu_500000.h5", "r") as h5:
    data = h5["Data"][:]

with open("../data/inc_pca_mu_354.pickle", "rb") as f:
    inc_pca = pickle.load(f)

with open("../data/featureNamesDict.pickle", "rb") as f:
    fNames = pickle.load(f)

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

def build_figures(stats, nr, path = None, histogram = "Z0_MM", acc=False, lastEpoch=False):
    epochs = np.arange(stats["epochs"])

    figs = []

    print("Process cost...")
    fig0 = plt.figure()
    plt.plot(epochs, stats["losses"][0], label = "Train cost")
    try:
        plt.plot(epochs, stats["val_losses"][0], label = "Validation cost")
    except KeyError:
        pass
    plt.legend(loc = "upper right")
    plt.title("Structure: {}, Act.: {}, batch_size: {}, \nepochs: {}, opt.: {}".format(stats["nodes"], stats["activations"], stats["batch"], stats["epochs"], stats["optimizer"]))
    figs += [fig0]

    print("Process accuracies...")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']*2
    fig2 = plt.figure()
    plt.plot(epochs, stats["acc"][0], label = "Test Accuracy")
    plt.plot(epochs, stats["train_acc"][0], "--", label = "Training Accuracy")
    plt.legend(loc = "upper left")
    plt.title("Structure: {}, Act.: {}, batch_size: {}, \nepochs: {}, opt.: {}".format(stats["nodes"],  stats["activations"], stats["batch"], stats["epochs"], stats["optimizer"]))
    figs += [fig2]

    if histogram is not None:
        figs += create_histograms(stats, histogram, nr, acc, lastEpoch)

    savefig(figs, path)

def create_histograms(stats, histograms, nr, acc, lastEpoch):
    print("Process histograms...")
    hists = []
    curdir = os.path.abspath(os.path.curdir)

    global data
    origData = pd.read_pickle("../data/Data_mu_500000.pickle")

    if stats["PCA"]:
        test_x = inc_pca.transform(data)
    else:
        test_x = data

    nr_examples = test_x.shape[0]
    folder = "./nn_logs{}".format(nr)

    columns = ['runNumber', 'eventNumber', 'nCandidate', 'totCandidates', 'nTracks',
        'nSPDHits', 'nLongTracks', 'Z0_MM', 'Z0_PT', 'Z0_ENDVERTEX_CHI2',
        'Z0_ENDVERTEX_NDOF', 'y', 'isolation', 'muplus_MINIP', 'muplus_P', 'muplus_PT',
        'muplus_PX', 'muplus_PY', 'muplus_PZ', 'muplus_TrEta', 'muplus_TrPChi2',
        'muplus_TrPhi', 'muplus_cpt_0.1', 'muplus_cpt_0.5', 'muminus_MINIP',
        'muminus_P', 'muminus_PT', 'muminus_PX', 'muminus_PY', 'muminus_PZ',
        'muminus_TrEta', 'muminus_TrPChi2', 'muminus_TrPhi', 'muminus_cpt_0.1',
        'muminus_cpt_0.5', 'nTrack', 'tracks_PT' 'tracks_PX', 'tracks_PY',
        'tracks_PZ', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi',
        'tracks_charge', 'tracks_isMuon', 'weight']

    drawn = []
    for histogram in histograms:
        if histogram not in columns:
            print("!!!\n {} does not in exist in this dataframe. IGNORED.\n!!!".format(histogram))
            histograms.remove(histogram)
        else:
            drawn.append(False)


    with open(folder+"/Models/checkpoint", "r") as f, tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(folder+'/Models/NNModel-0.meta')
        lines = f.readlines()
        pattern = "nn_logs.*"
        for ei, line in enumerate(lines[1:]):
            if lastEpoch and not ei+1 == len(lines)-1:
                continue

            regExp = re.search(pattern, line)
            filename = regExp.group(0)[:-1]
            regExp = re.search("[0-9]+$", filename)
            e = int(regExp.group(0))

            new_saver.restore(sess, filename)
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            y = graph.get_tensor_by_name("y:0")
            tr_phase = graph.get_tensor_by_name("trainPhase:0")
            pred_proba = graph.get_tensor_by_name("Model/predProba:0")
            pred = pred_proba.eval({x: test_x, tr_phase: 1})
            pred = np.array([np.argmax(p) for p in pred])

            signal = pred==1
            bg = pred==0
            for h, histogram in enumerate(histograms):
                bincuts = np.array([10, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 60.0, 120.0])*1000 if histogram=="Z0_MM" else 50

                x = [origData[histogram][signal], origData[histogram][bg]]
                fig = plt.figure()
                n, edges, _ = plt.hist(x, histtype="bar", stacked=True, bins=bincuts, color = ["#0b0db0", "#b21000"], label = ["pred signal", "pred background"])
                plt.xlabel(histogram)
                plt.title("Epoch: {}".format(e))
                plt.legend()
                # plt.yscale("log")
                hists += [[int("1{}{}0".format(h, ei)), fig]]

                if acc:
                    fig = plt.figure()
                    n, edges, _ = plt.hist(x, histtype="bar", stacked=False, bins=bincuts)
                    count_predSig = n[0]
                    count_predBg = n[1]
                    count_all = n[0] + n[1]

                    predsg = list(count_predSig/count_all)
                    predbg = list(count_predBg/count_all)

                    pred_data = pd.DataFrame({"m0": bincuts[:-1], "m1": bincuts[1:], "signal": predsg, "bg": predbg})

                    pred_data.to_csv("../data/dataPredictionTransverse.csv")

                    predbg.insert(0, predbg[0])
                    predsg.insert(0, predsg[0])
                    fig = plt.figure()
                    plt.step(edges, predbg, color="#ff8a65", label="Pred. Bg", where="pre", linestyle="--")
                    plt.step(edges, predsg, color="navy", label="Pred. Sg", where="pre")
                    plt.xlabel(histogram)
                    plt.title("Predicted rate - Epoch: {}".format(e))
                    plt.legend(loc=8, fontsize = "small")
                    plt.ylim([0,1])
                    hists += [[int("1{}{}2".format(h, ei)), fig]]

    hists = list(np.array(sorted(hists, key=lambda x: x[0]))[:, 1])
    return hists


def main():
    def str2bool(inpt):
        if inpt in ["n", "N", "False", "F", "f", "no"]:
            inpt = False
        else:
            inpt = True
        return inpt
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", type=int, default=None, help="stats number, default: None")
    parser.add_argument("--histogram", type=str, default="Z0_MM", help="variable to plot a histogram, features are seperated by a ':', default: 'Z0_MM'")
    parser.add_argument("--acc", type=str2bool, default=True, help="Presentation of histograms in terms of correct classification, default: True")
    parser.add_argument("--last", type=str2bool, default=True, help="True, if only last epoch should be shown, default: False")
    parser.add_argument("--sortby", type=str, default="roc_auc", help="Sort value for summary. Possible are: 'max_acc' (default), 'roc_auc' and 'F1'")
    args = parser.parse_args()
    
    curdir = os.path.abspath(os.path.curdir)
    if args.nr is None:
        stats_nr = []
        user_input = 0
        while True:
            user_input = input("Stats number: ")
            if user_input == "":
                break
            elif float(user_input) < 0:
                stats_nr = [i for i in range(100)]
                break
            try:
                user_input = int(user_input)
                stats_nr += [user_input]
            except:
                print("Invalid input. Insert integer!")
    
        for nr in stats_nr:
            filename = "{}/nn_logs{}/stats{}.pickle".format(curdir, nr, nr)
            with open(filename, "rb") as f:
                stats = pickle.load(f)
                build_full_output(stats, args.histogram.split(":"), nr, args.acc, args.last)
    
    else:
        filename = "{}/nn_logs{}/stats{}.pickle".format(curdir, args.nr, args.nr)
        with open(filename, "rb") as f:
            stats = pickle.load(f)
            build_full_output(stats, args.histogram.split(":"), args.nr, args.acc, args.last)


    # nmbrs=[0]
    # histograms = "Z0_MM:nSPDHits:y"
    # for nmbr in nmbrs:
    #     filename = "./nn_logs{}/stats{}.pickle".format(nmbr, nmbr)
    #     with open(filename, "rb") as f:
    #         stats = pickle.load(f)
    #         build_full_output(stats, histograms.split(":"), nmbr, True, False)



if __name__ == "__main__":
    main()
