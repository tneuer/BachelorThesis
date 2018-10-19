import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as mpatches
import pickle
import time
import os.path
import argparse
import h5py
import re
import tensorflow as tf

plotted_points = 20000
style.use("ggplot")

for i in range(100):
    newDoc = ""
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
    test_y = test_y[:plotted_points]

with h5py.File("../../../data/Processed_Val_mu_263093.h5", "r") as valData:
    data = valData["Val"][:plotted_points]

with open("../../../data/inc_pca_mu_354.pickle", "rb") as f:
    inc_pca = pickle.load(f)

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


def build_csv_summary(stats_nr, filename):
    if type(stats_nr) != list:
        stats_nr = [stats_nr]

    curdir = os.path.abspath(os.path.curdir)
    stats_files = ["{}/nn_logs{}/stats{}.pickle".format(curdir, nr, nr) for nr in stats_nr]
    out_file = filename + ".csv"

    columns = ["stats_nr", "NNType", "nodes", "activations", "DropOut", "min_loss", "min_loss_std", "losses", "std", "train_examples", "test_examples", "feature_size", "optimizer", "epochs", "time", "fcost"]
    out_df = pd.DataFrame()

    collective_dict = {}
    for col in columns:
        collective_dict[col] = []

    print("")
    for stats_file in stats_files:
        try:
            with open(stats_file, "rb") as f:
                in_dict = pickle.load(f)

                for col in columns:
                    try:
                        if col == "losses":
                            collective_dict["min_loss"] += [np.round(np.mean(np.amin(in_dict[col], axis=1)), 4)]
                            collective_dict["min_loss_std"] += [np.round(np.std(np.amin(in_dict[col], axis=1)), 4)]
                            collective_dict["losses"] += [np.round(np.mean(in_dict[col][:,-1]), 4)]
                            collective_dict["std"] += [np.round(np.std(in_dict[col][:,-1]), 4)]
                        elif col == "train_examples":
                            collective_dict[col] += [in_dict[col]]
                            collective_dict["test_examples"] += [int(in_dict["test_size"])]
                        elif col == "features":
                            collective_dict[col] += [len(in_dict[col])]
                        elif col in ["min_loss", "min_loss_std", "std", "test_examples"]:
                            pass
                        elif col == "time":
                            collective_dict[col] += [np.round(in_dict[col], 0)]
                        else:
                            collective_dict[col] += [in_dict[col]]
                    except:
                        collective_dict[col] += ["Not implemented"]

            print(stats_file+" used for summary.")
        except:
            pass

    for col in columns:
        out_df[col] = collective_dict[col]

    out_df.sort_values(by="min_loss", ascending = False, inplace = True)
    out_df.to_csv(out_file)
    print("\n" + out_file + " successfully built.\n")


def build_full_output(stats, nr, lastEpoch):
    for i in range(100):
        filename1 = "./Logs/log{}_{}.txt".format(i, nr)
        filename2 = "./Figures/fig{}_{}_last{}.pdf".format(i, nr, str(lastEpoch)[0])
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

                f.write("\n\n"+"#"*10)
                f.write("\n\tEvaluation Information")
                f.write("\n"+"#"*10)

                f.write("\n\nMin loss: {}".format( min(stats["losses"][0])))
                f.write("\n\nMin val loss: {}".format( min(stats["val_losses"][0])))

                f.write("\n\nLast loss: {}".format( stats["losses"][0][-1]))
                f.write("\n\nLast val loss: {}".format( stats["val_losses"][0][-1]))

                f.write("\n\n"+"#"*10)
                f.write("\n\tEvaluation Details")
                f.write("\n"+"#"*10)

                f.write("\n\nLosses: {}".format(stats["losses"]))
                f.write("\n\nValidation losses: {}".format(stats["val_losses"]))

                t = stats["time"]
                f.write("\n\nTraining Time: {}s, {}min, {}h".format(t, t/60, t/3600))

            build_figures(stats, path=filename2, nr=nr, lastEpoch=lastEpoch)
            print("Log and figures file: {}_{}".format(i, nr))
            break


def build_figures(stats, path, nr, lastEpoch):

    figs = []

    fig0 = plt.figure()
    plt.plot(stats["losses"][0], label = "Train cost")
    plt.plot(stats["val_losses"][0], label = "Test cost")

    plt.legend(loc = "upper right")
    plt.title("Structure: {}, Act.: {}, batch_size: {}, \nepochs: {}, opt.: {}".format(stats["nodes"], stats["activations"], stats["batch"], stats["epochs"], stats["optimizer"]))
    figs += [fig0]

    print("Process plot...")

    folder = "./nn_logs{}".format(nr)

    if stats["PCA"]:
        test_x = inc_pca.transform(data)
    else:
        test_x = data

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
            pred = graph.get_tensor_by_name("Model/predict:0")
            print(line)
            pred = pred.eval({x: test_x})

            sg = pred[test_y==1, :]
            bg = pred[test_y==0, :]
            fig = plt.figure()
            plt.plot(sg[:, 0], sg[:, 1], "o", markersize=1, alpha=0.2, label="Signal")
            plt.plot(bg[:, 0], bg[:, 1], "o", markersize=1, alpha=0.2, label="Background")
            plt.title("First two components --- Epoch: {}".format(e))
            plt.ylabel("2nd component")
            plt.xlabel("1st component")
            plt.legend()

            figs.append(fig)

    print("Saving...")
    savefig(figs, path)

def main():
    def str2bool(inpt):
        if inpt in ["n", "N", "False", "F", "f", "no"]:
            inpt = False
        else:
            inpt = True
        return inpt

    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", type=int, default=None, help="stats number, default None")
    parser.add_argument("--summary", type=str2bool, default=True, help="True if summary is requested, default True")
    parser.add_argument("--filename", type=str, default="summary", help="filename of summary")
    parser.add_argument("--last", type=str2bool, default=False, help="True, if only last epoch should be shown, default: False")
    args = parser.parse_args()

    curdir = os.path.abspath(os.path.curdir)

    if args.nr == None:
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
            build_csv_summary(stats_nr, args.filename)
        else:
            for nr in stats_nr:
                filename = "{}/nn_logs{}/stats{}.pickle".format(curdir, nr, nr)
                with open(filename, "rb") as f:
                    stats = pickle.load(f)
                    build_full_output(stats, nr, args.last)

    else:
        filename = "{}/nn_logs{}/stats{}.pickle".format(curdir, args.nr, args.nr)
        with open(filename, "rb") as f:
            stats = pickle.load(f)
            build_full_output(stats, args.nr, args.last)


if __name__ == "__main__":
    main()
