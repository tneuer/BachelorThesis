import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
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

for i in range(100):
    newDoc = ""
    bestLine = ""
    try:
        with open("./nn_logs{}/Models/checkpoint".format(i), "r") as f:
            lines = f.readlines()
            pattern = "Best"
            for line in lines:
                regExp = re.findall(pattern, line)
                if regExp:
                    bestLine = line
                else:
                    newDoc += line
            newDoc += bestLine
    except FileNotFoundError:
        continue
    with open("./nn_logs{}/Models/checkpoint".format(i), "w") as f:
        f.write(newDoc)

with open("../../../data/Val_mu_labels_263093.pickle", "rb") as f:
    test_y = pickle.load(f)

with h5py.File("../../../data/Processed_Val_mu_263093.h5", "r") as valData:
    data = valData["Val"][:]

with open("../../../data/inc_pca_mu_354.pickle", "rb") as f:
    inc_pca = pickle.load(f)

with open("../../../data/featureNamesDict.pickle", "rb") as f:
    featureDict = pickle.load(f)

def get_sg(fname, labels=True):
    with open(fname, "rb") as f:
        sg = pickle.load(f)

    if labels:
        lbls = np.ones(sg.shape[0])
        return sg, lbls
    else:
        return sg

def get_bg(fname, labels=True):
    with open(fname, "rb") as f:
        bg = pickle.load(f)

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

nr_features = data.shape[1]
nr_classes = len(list(np.unique(test_y)))

def savefig(list_of_plots, path):
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(path)
    for fig in list_of_plots:
        pdf.savefig(fig)
    pdf.close()


def build_csv_summary(stats_nr, filename, sortby):
    if type(stats_nr) != list:
        stats_nr = [stats_nr]

    folders = ["./nn_logs{}".format(nr) for nr in stats_nr]
    out_file = filename + ".csv"

    columns = ["stats_nr", "NNType", "nodes", "activations", "DropOut", "max_acc", "max_acc_std", "roc_auc", "F1", "acc", "std", "train_examples", "test_examples", "feature_size", "optimizer", "epochs", "time", "fcost", "batchNorm", "PCA"]
    out_df = pd.DataFrame()

    collective_dict = {}
    for col in columns:
        collective_dict[col] = []

    for folder,statsNr in zip(folders, stats_nr):
        try:
            with open(folder+"/stats{}.pickle".format(statsNr), "rb") as f:
                in_dict = pickle.load(f)
        except FileNotFoundError:
            continue

        roc_auc = {}
        F1 = {}

        if in_dict["PCA"]:
            test_x_total = inc_pca.transform(data)
        else:
            test_x_total = data

        with open(folder+"/Models/checkpoint", "r") as f, tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(folder+'/Models/NNModel-0.meta')
            lines = f.readlines()
            pattern = "NNModel.*"
            for line in lines[1:]:
                regExp = re.search(pattern, line)
                filename = regExp.group(0)[:-1]
                filename = folder + "/Models/" + filename
                try:
                    regExp = re.search("[0-9]+$", filename)
                    e = int(regExp.group(0))
                except AttributeError:
                    e = in_dict["epochs"]-20

                new_saver.restore(sess, filename)
                graph = tf.get_default_graph()
                x = graph.get_tensor_by_name("x:0")
                y = graph.get_tensor_by_name("y:0")
                tr_phase = graph.get_tensor_by_name("trainPhase:0")
                pred_proba = graph.get_tensor_by_name("Model/predProba:0")

                feature_nrs = np.array([featureDict[feat] for feat in in_dict["features"]])
                test_x = test_x_total[:, feature_nrs]

                pred = pred_proba.eval({x: test_x, tr_phase: 1})
                pred = np.array([np.argmax(p) for p in pred])

                fpr, tpr, _ = roc_curve(test_y, pred)
                roc_auc["{}".format(e)] = auc(fpr, tpr)

                #F1
                f1 = f1_score(test_y, pred)
                F1["{}".format(e)]= f1

        tf.reset_default_graph()
        max_auc = max(roc_auc.values())
        max_F1 = max(F1.values())

        for col in columns:
            try:
                if col == "acc":
                    collective_dict["max_acc"] += [np.round(np.mean(np.amax(in_dict[col], axis=1)), 4) * 100]
                    collective_dict["max_acc_std"] += [np.round(np.std(np.amax(in_dict[col], axis=1)), 4) * 100]
                    collective_dict[col] += [np.round(np.mean(in_dict[col][:,-1]), 4) * 100]
                    collective_dict["std"] += [np.round(np.std(in_dict[col][:,-1]), 4) * 100]
                elif col == "train_examples":
                    collective_dict[col] += [in_dict[col]]
                    collective_dict["test_examples"] += [int(in_dict["test_size"])]
                elif col == "features":
                    collective_dict[col] += [len(in_dict[col])]
                elif col in ["max_acc", "max_acc_std", "std", "test_examples"]:
                    pass
                elif col == "time":
                    collective_dict[col] += [np.round(in_dict[col], 0)]
                elif col == "roc_auc":
                    collective_dict[col] += [round(max_auc, 4)]
                elif col == "F1":
                    collective_dict[col] += [round(max_F1, 4)]
                else:
                    collective_dict[col] += [in_dict[col]]
            except KeyError:
                collective_dict[col] += ["Not implemented"]

        print(folder+" used for summary.")


    for col in columns:
        out_df[col] = collective_dict[col]

    out_df.sort_values(by=sortby, ascending = False, inplace = True)
    out_df.to_csv(out_file)
    print("\n" + out_file + " successfully built.\n")


def build_full_output(stats, histogram, nr, acc, lastEpoch):
    for i in range(100):
        filename1 = "./Logs/log{}_{}.txt".format(i, nr)
        filename2 = "./Figures/fig{}_{}_acc{}_last{}.pdf".format(i, nr, str(acc)[0], str(lastEpoch)[0])
        if not os.path.isfile(filename1) and not os.path.isfile(filename2):
            with open(filename1, "w") as f:

                f.write("Start date: {}, Corresponding stats: {}".format(time.strftime("%c"), nr))
                f.write("\n"+"#"*10)
                f.write("\n\tNetwork Information")
                f.write("\n"+"#"*10)

                f.write("\n\nNetwork type: {}".format(stats["NNType"]))
                f.write("\n\nStructure: {}".format(stats["nodes"]))
                f.write("\nActivations: {}".format(stats["activations"]))
                f.write("\nDropout: {}".format(stats["DropOut"]))


                f.write("\n\n"+"#"*10)
                f.write("\n\tTraining Information")
                f.write("\n"+"#"*10)

                f.write("\n\nTrainingSet: {}, Epochs: {}, batch_size: {}".format(stats["train_examples"], stats["epochs"], stats["batch"]))
                f.write("\n\nOptimizer: {}".format(stats["optimizer"]))
                f.write("\n\n#Features: {} - {}".format(len(stats["features"]), stats["feature_size"]))
                f.write("\n\nStandardized Features: {}".format(stats["standFeatures"]))
                f.write("\n\nLog-transformed Features: {}".format(stats["LogTransformed"]))
                f.write("\n\nFeatures: {}".format(stats["features"]))
                try:
                    f.write("\n\nApplied cuts: {}".format(stats["cuts"]))
                except KeyError:
                    pass


                f.write("\n\n"+"#"*10)
                f.write("\n\tEvaluation Information")
                f.write("\n"+"#"*10)

                test_accs = stats["acc"][:,-1]
                f.write("\n\nFinal test accuracy: {}".format(test_accs))
                f.write("\nK-Fold test accuracy: {} pm {}".format(np.mean(test_accs), np.std(test_accs)))

                max_test_acc = np.amax(stats["acc"], axis=1)
                f.write("\n\nMax test accuracies: {} --- Epoch: {}".format(max_test_acc, np.argmax(stats["acc"], axis=1)))
                f.write("\nK-Fold max accuracy: {} pm {}".format(np.mean(max_test_acc), np.std(max_test_acc)))


                losses = stats["losses"][:, -1]
                f.write("\nK-Fold cost: {} pm {}".format(np.mean(losses), np.std(losses)))
                try:
                    val_losses = stats["val_losses"][:, -1]
                    f.write("\nK-Fold val_cost: {} pm {}".format(np.mean(val_losses), np.std(val_losses)))
                except KeyError:
                    pass

                f.write("\n\n{}".format(create_confusion_matrix(stats, nr)))
                f.write("\n\n"+"#"*10)
                f.write("\n\tEvaluation Details")
                f.write("\n"+"#"*10)

                f.write("\n\nLosses: {}".format(stats["losses"]))
                try:
                    f.write("\n\nVal_Losses: {}".format(stats["val_losses"]))
                except KeyError:
                    pass
                f.write("\n\nTraining Accuracy: {}".format(stats["train_acc"]))
                f.write("\n\nTest Accuracy: {}".format(stats["acc"]))

                t = stats["time"]
                f.write("\n\nTraining Time: {}s, {}min, {}h".format(t, t/60, t/3600))

            build_figures(stats, path=filename2, histogram=histogram, nr=nr, acc=acc, lastEpoch=lastEpoch)
            print("Log and figures file: {}_{}".format(i, nr))
            break


def create_confusion_matrix(stats, nr):
    confMatrices = "Confusion matrix | Precision | Recall: \n"
    if stats["PCA"]:
        test_x_total = inc_pca.transform(data)
    else:
        test_x_total = data

    folder = "./nn_logs{}".format(nr)
    with open(folder+"/Models/checkpoint", "r") as f, tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(folder+'/Models/NNModel-0.meta')
        lines = f.readlines()
        pattern = "NNModel.*"
        for line in lines[1:]:
            regExp = re.search(pattern, line)
            filename = regExp.group(0)[:-1]
            filename = folder + "/Models/" + filename
            try:
                regExp = re.search("[0-9]+$", filename)
                e = int(regExp.group(0))
            except AttributeError:
                e = stats["epochs"]-20

            new_saver.restore(sess, filename)
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            y = graph.get_tensor_by_name("y:0")
            tr_phase = graph.get_tensor_by_name("trainPhase:0")
            pred_proba = graph.get_tensor_by_name("Model/predProba:0")

            feature_nrs = np.array([featureDict[feat] for feat in stats["features"]])
            test_x = test_x_total[:, feature_nrs]

            pred = pred_proba.eval({x: test_x, tr_phase: 1})
            pred = np.array([np.argmax(p) for p in pred])

            confMatrices = confMatrices + "{} Epoch {}".format("---"*10, e)
            confMatrix = confusion_matrix(test_y, pred)
            confMatrices += "\n\n{}".format(str(confMatrix))

            confMatrix = confMatrix / confMatrix.sum(axis=0, keepdims=True)
            confMatrices += "\n\n{}".format(str(confMatrix))

            confMatrix = confMatrix / confMatrix.sum(axis=1, keepdims=True)
            confMatrices += "\n\n{}\n".format(str(confMatrix))
    tf.reset_default_graph()


    return confMatrices


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

    margin_q = np.linspace(-1, 1, 50)
    figMargin = plt.figure()

    if stats["PCA"]:
        test_x_total = inc_pca.transform(data)
    else:
        test_x_total = data

    folder = "./nn_logs{}".format(nr)

    with open(folder+"/Models/checkpoint", "r") as f, tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(folder+'/Models/NNModel-0.meta')
        lines = f.readlines()
        pattern = "NNModel.*"
        for line in lines[1:]:
            regExp = re.search(pattern, line)
            filename = regExp.group(0)[:-1]
            filename = folder + "/Models/" + filename
            try:
                regExp = re.search("[0-9]+$", filename)
                e = int(regExp.group(0))
            except AttributeError:
                e = stats["epochs"]-20

            new_saver.restore(sess, filename)
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            y = graph.get_tensor_by_name("y:0")
            tr_phase = graph.get_tensor_by_name("trainPhase:0")
            pred_proba = graph.get_tensor_by_name("Model/predProba:0")

            feature_nrs = np.array([featureDict[feat] for feat in stats["features"]])
            test_x = test_x_total[:, feature_nrs]

            pred = pred_proba.eval({x: test_x, tr_phase: 1})

            factor = np.array([int(i) if i else int(i)-1 for i in np.equal(np.argmax(pred, axis = 1), test_y)])

            margins = np.max(pred, axis = 1) * factor

            margin_p = [len(np.where(margins<=q)[0])/len(margins) for q in margin_q ]
            plt.plot(margin_q, margin_p, label = "Epoch {}".format(e))
            plt.title("Margin CDF")
            plt.xlabel("Margin")
            plt.ylabel("CDF / Fraction")
            plt.xticks(np.linspace(-1, 1, 11))
            plt.yticks(np.linspace(0, 1, 11))
            plt.legend()

    tf.reset_default_graph()

    figs += [figMargin]

    figs += create_sg_bg_separation_and_ROC(stats, nr, lastEpoch)
    if histogram is not None:
        figs += create_histograms(stats, histogram, nr, acc, lastEpoch)


    savefig(figs, path)


def create_sg_bg_separation_and_ROC(stats, nr, lastEpoch):
    print("Process ROC...")

    ROCs = []
    folder = "./nn_logs{}".format(nr)

    if stats["PCA"]:
        test_x_total = inc_pca.transform(data)
    else:
        test_x_total = data

    with open(folder+"/Models/checkpoint", "r") as f, tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(folder+'/Models/NNModel-0.meta')
        lines = f.readlines()
        pattern = "NNModel.*"
        for ei, line in enumerate(lines[1:]):
            if lastEpoch and not (ei == 0 or ei+1 == len(lines)-1):
                continue

            regExp = re.search(pattern, line)
            filename = regExp.group(0)[:-1]
            filename = folder + "/Models/" + filename
            try:
                regExp = re.search("[0-9]+$", filename)
                e = int(regExp.group(0))
            except AttributeError:
                e = stats["epochs"]-20

            new_saver.restore(sess, filename)
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            y = graph.get_tensor_by_name("y:0")
            tr_phase = graph.get_tensor_by_name("trainPhase:0")
            pred_proba = graph.get_tensor_by_name("Model/predProba:0")

            feature_nrs = np.array([featureDict[feat] for feat in stats["features"]])
            test_x = test_x_total[:, feature_nrs]

            pred = pred_proba.eval({x: test_x, tr_phase: 1})

            bgDist = pred[np.where(test_y==0)[0]][:,0]
            sgDist = pred[np.where(test_y==1)[0]][:,0]

            fig, ax = plt.subplots(nrows = 1, ncols = 2)

            plt.subplot(1, 2, 1)
            x = [bgDist, sgDist]
            plt.hist(x, histtype = "step", bins =75, label = ["background", "signal"])
            plt.title("Bg-Sg-Prediction Dist.")
            plt.xlabel("Prediction")
            plt.legend(loc="upper center")

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(nr_classes):
                fpr[i], tpr[i], _ = roc_curve(test_y, pred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.subplot(1, 2, 2)
            plt.plot(fpr[1], tpr[1], color='darkorange', lw=2, label='ROC area = %0.4f' % roc_auc[1])
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC, Epoch: {}'.format(e))
            plt.legend(loc="lower right")
            ROCs.append(fig)

    return ROCs


def create_histograms(stats, histograms, nr, acc, lastEpoch):
    print("Process histograms...")
    hists = []
    curdir = os.path.abspath(os.path.curdir)

    sgname = "../../../data/Val_mu_MC_156388.pickle"
    bgname = "../../../data/Val_mu_hf_106705.pickle"

    if stats["PCA"]:
        test_x_total = inc_pca.transform(data)
    else:
        test_x_total = data

    origData, test_y = get_sg_and_bg(sgname, bgname, labels=True, shuffle=True, random_state=42)

    nr_examples = test_x_total.shape[0]
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
        pattern = "NNModel.*"
        for ei, line in enumerate(lines[1:]):
            if lastEpoch and not (ei == 0 or ei+1 == len(lines)-1):
                continue

            regExp = re.search(pattern, line)
            filename = regExp.group(0)[:-1]
            filename = folder + "/Models/" + filename
            try:
                regExp = re.search("[0-9]+$", filename)
                e = int(regExp.group(0))
            except AttributeError:
                e = stats["epochs"]-20

            new_saver.restore(sess, filename)
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            y = graph.get_tensor_by_name("y:0")
            tr_phase = graph.get_tensor_by_name("trainPhase:0")
            pred_proba = graph.get_tensor_by_name("Model/predProba:0")

            feature_nrs = np.array([featureDict[feat] for feat in stats["features"]])
            test_x = test_x_total[:, feature_nrs]

            pred = pred_proba.eval({x: test_x, tr_phase: 1})
            pred = np.array([np.argmax(p) for p in pred])

            signal = np.where(pred==1)[0]
            bg = np.where(pred==0)[0]

            truesignal = np.where(test_y==1)[0]
            truebg = np.where(test_y==0)[0]

            truepos = np.logical_and(test_y==1, pred==1)
            trueneg = np.logical_and(test_y==0, pred==0)
            falsepos = np.logical_and(test_y==0, pred==1)
            falseneg = np.logical_and(test_y==1, pred==0)

            for h, histogram in enumerate(histograms):
                bincuts = np.array([10, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 60.0, 120.0])*1000 if histogram=="Z0_MM" else 50
                if not drawn[h]:
                    x = [origData[histogram][test_y==1], origData[histogram][test_y==0]]
                    fig = plt.figure()
                    plt.hist(x, histtype = "bar", stacked = "True", bins=bincuts, color = ["#0b0db0", "#b21000"], label = ["signal", "background"])
                    plt.xlabel(histogram)
                    plt.title("True distribution")
                    plt.legend()
                    plt.yscale("log")
                    hists += [[-1, fig]]
                    fig = plt.figure()
                    drawn[h] = True

                x = [origData[histogram][truepos], origData[histogram][falsepos], origData[histogram][trueneg], origData[histogram][falseneg]]
                fig = plt.figure()
                plt.hist(x, histtype = "bar", stacked = True, bins=bincuts, color = ["#0b0db0", "#7172e1", "#b21000","#e17b71" ], label = ["true signal", "false signal", "true background", "false background"])
                plt.xlabel(histogram)
                plt.title("Epoch: {}".format(e))
                plt.legend()
                plt.yscale("log")
                hists += [[int("1{}{}0".format(h, ei)), fig]]

                if acc:
                    fig = plt.figure()
                    n, edges, _ = plt.hist(x, histtype = "bar", stacked = False, bins=bincuts)
                    count_trueSig = n[0]
                    count_predSig = n[0] + n[1]
                    count_trueBg = n[2]
                    count_predBg = n[2] + n[3]
                    count_correct = n[0] + n[2]
                    count_signal = n[0] + n[3]
                    count_bg = n[1] + n[2]
                    count_all = n[0] + n[1] + n[2] + n[3]

                    acc_sg = list(count_trueSig / count_signal)
                    acc_bg = list(count_trueBg / count_bg)
                    acc_al = list(count_correct / count_all)

                    acc_sg.insert(0, acc_sg[0])
                    acc_bg.insert(0, acc_bg[0])
                    acc_al.insert(0, acc_al[0])
                    fig = plt.figure()
                    plt.step(edges, acc_bg, color ="#ff8a65", label= "Recall: Bg", where="pre", linestyle="--")
                    plt.step(edges, acc_al, color ="#7dfd6a", label= "Recall: Total", where="pre", linestyle="--")
                    plt.step(edges, acc_sg, color="navy", label="Recall: Signal", where="pre")
                    plt.xlabel(histogram)
                    plt.title("Recall - Epoch: {}".format(e))
                    plt.legend(loc=8, fontsize = "small")
                    plt.ylim([0,1])
                    hists += [[int("1{}{}1".format(h, ei)), fig]]

                    predsg = list(count_predSig/count_all)
                    predbg = list(count_predBg/count_all)

                    predsg.insert(0, predsg[0])
                    predbg.insert(0, predbg[0])
                    fig = plt.figure()
                    plt.step(edges, predbg, color="#ff8a65", label="Pred. Bg", where="pre", linestyle="--")
                    plt.step(edges, predsg, color="navy", label="Pred. Sg", where="pre")
                    plt.xlabel(histogram)
                    plt.title("Predicted rate - Epoch: {}".format(e))
                    plt.legend(loc=8, fontsize = "small")
                    plt.ylim([0,1])
                    hists += [[int("1{}{}2".format(h, ei)), fig]]

                    predFsg = list(n[1]/count_all)
                    predFbg = list(n[3]/count_all)

                    predFsg.insert(0, predFsg[0])
                    predFbg.insert(0, predFbg[0])
                    fig = plt.figure()
                    plt.step(edges[:-1], predFbg, color="#ff8a65", label="False negative", where="pre", linestyle="--")
                    plt.step(edges[:-1], predFsg, color="navy", label="False positive", where="pre")
                    plt.xlabel(histogram)
                    plt.title("False rate - Epoch: {}".format(e))
                    plt.legend(loc=8, fontsize = "small")
                    plt.ylim([0,1])
                    hists += [[int("1{}{}3".format(h, ei)), fig]]

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
    parser.add_argument("--summary", type=str2bool, default=True, help="True if summary is requested, default: True")
    parser.add_argument("--filename", type=str, default="summary", help="filename of summary, default: 'summary'")
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
    
        if args.summary:
            build_csv_summary(stats_nr, args.filename, args.sortby)
        else:
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
