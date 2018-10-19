#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : keras.py
    # Creation Date : Fre 02 Mär 2018 18:52:17 CET
    # Last Modified : Son 15 Apr 2018 16:00:26 CEST
    # Description : implementation of networks in keras
"""
#==============================================================================

import numpy as np
import pandas as pd
import pickle
import time
import h5py
import os
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import OneHotEncoder
from pipeliner import batch_pipeline, logtransform_standard_pipeline
from baUtils import get_sg_and_bg
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import roc_auc_score, f1_score

from keras import backend as K

#############################

several_nn = [[100, 100, 50], [100, 100, 50], [250, 100], [50]]
activations_list = [["elu"]*3, ["elu"]*3, ["elu"]*2, ["elu"]]
DropOut_list = [[None]*3, [None]*3, [None]*2, [None]]

nr_examples_list = [400000]
hm_epochs_list = [5]
batch_size_list = [256]
optimizer_names = [["Adam", 0.001]]

BN_list = [False, True, False, False]
PCA = True
folder = "../allOutputs/keras/"
datafolder = "../data"
#############################

print("Read test labels")
with open("{}/Val_labels_263093.pickle".format(datafolder), "rb") as f:
    test_y = pickle.load(f)

print("Read test...")
with h5py.File("{}/Processed_Val_263093.h5".format(datafolder), "r") as valData:
    test_x = valData["Val"][:]

with open("{}/inc_pca354.pickle".format(datafolder), "rb") as f:
    inc_pca = pickle.load(f)

enc = OneHotEncoder(sparse=False)
test_y = enc.fit_transform(test_y.reshape(-1,1))

transverse_wo_SPD = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']

standard_features = transverse_wo_SPD[:7]
standard_log = ["muplus_PT", "muminus_PT"]

non_standard_features = transverse_wo_SPD[6:]
global_standardized = non_standard_features[1:-2]
non_standard_log = ["tracks_PT", "tracks_IPCHI2", "tracks_IP"]

if PCA:
    print("PCA test")
    test_x = inc_pca.transform(test_x)

nr_classes = len(test_y[0])
nr_features = test_x.shape[1]

def check_and_correct_parameters(models):
    """Checks user input for errors.

    Every list needs to be the same length as 'several_nn' or one. Also DropOut needs to have the
    same number of layers as stated in 'several_nn'. If this is not the case an error is raised.

    Parameters
    ----------
    models : int
        number of specified models; length of 'several_nn'

    Returns
    -------
    No value is returned, only error is raised.
    """
    parameter_dict = {"Activations": activations_list, "DropOuts": DropOut_list, "Epochs": hm_epochs_list, "batch_sizes": batch_size_list, "Optimizers": optimizer_names, "BatchNorm": BN_list, "nr_examples": nr_examples_list}

    for key, value in parameter_dict.items():
        l = len(value)
        if l != models and l != 1:
            errmess = "1 needed" if models == 1 else "1 or {} needed".format(models)
            raise AssertionError("Wrong length of {}. {}. Given: {}".format(key, errmess, l))
        elif l == 1:
            value *= models

    nodeLen = np.array([len(node) for node in several_nn])
    dropLen = np.array([len(drop) if drop!= None else len(several_nn[i]) for i, drop in enumerate(parameter_dict["DropOuts"])])
    actLen = np.array([len(avt) for avt in parameter_dict["Activations"]])

    dropequal = np.where(~np.equal(nodeLen, dropLen))[0]
    actequal = np.where(~np.equal(nodeLen, actLen))[0]

    assert len(dropequal) == 0, ("\nFollowing DropOuts have the wrong length: {}\nNeeded: {} (node length)\nGiven: {}".format(dropequal, nodeLen[dropequal], dropLen[dropequal]))

    assert len(actequal) == 0, ("\nFollowing activations have the wrong length: {}\nNeeded: {} (nodelength)\nGiven: {}".format(actequal, nodeLen[actequal], actLen[actequal]))

def train_neural_network():
    start = time.clock()
    for i in range(200):
        checkIfExists = folder + "nn_logs{}".format(i)
        if not os.path.isdir(checkIfExists):
            os.makedirs(checkIfExists)
            os.makedirs(checkIfExists+"/Models")
            stats_nr = i
            print("Folder nn_logs{} generated.".format(i))
            break

    model = Sequential()
    model.add(Dense(layers[0], input_dim=nr_features))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(activations[0]))
    if DropOut[0] is not None:
        model.add(Dropout(DropOut[0]))
    i=0
    try:
        for nodes, act in zip(layers[1:], activations[1:]):
            model.add(Dense(nodes))
            if bn:
                model.add(BatchNormalization())
            model.add(Activation(act))
            if DropOut[i] is not None:
                model.add(Dropout(DropOut[i]))
            i+=1
    except IndexError:
        pass

    model.add(Dense(nr_classes))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation("softmax"))

    optim = eval("keras.optimizers.{}(lr={})".format(optimizer_name[0], optimizer_name[1]))
    model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["acc"])

    print(model.summary())
    with open("{}/ModelSummary.txt".format(checkIfExists), "w") as f:
        model.summary(print_fn=lambda x: f.write(x+"\n"))

    Checkpointer = ModelCheckpoint("{}/BestModel_{}.h5".format(checkIfExists, nr_examples), save_best_only=True)
    Stoper = EarlyStopping(patience=20)
    Tensorboarder = TensorBoard(log_dir=checkIfExists, histogram_freq=3, write_graph=True, write_images=True)

    class ROC_AUC(keras.callbacks.Callback):
        def __init__(self, te_x, te_y):
            self.roc_aucs = []

            self.val_x, self.val_y = te_x, te_y

        def on_epoch_end(self, epoch, logs={}):
            y_pred = self.model.predict_proba(self.val_x, verbose=0)
            score = roc_auc_score(self.val_y[:, 1], y_pred[:, 1])
            self.roc_aucs.append(score)

    class Regular_Checkpoint(keras.callbacks.Callback):
        def __init__(self, filepath, interval):
            self.filepath = filepath
            self.interval = interval

        def on_epoch_end(self, epoch, logs={}):
            if epoch % self.interval == 0:
                print("Checkpoint saved!")
                self.model.save("{}/checkpoint_{}.h5".format(self.filepath, epoch))
                self.model.save_weights("{}/checkpoint_{}_weights.h5".format(self.filepath, epoch))


    ROCer = ROC_AUC(test_x, test_y)
    reg_check = Regular_Checkpoint("{}".format(checkIfExists), int(hm_epochs/5))

    history = model.fit(data, labels, epochs=hm_epochs, batch_size=batch_size, shuffle=True,
            validation_data=(test_x, test_y), callbacks=[Checkpointer, Stoper, Tensorboarder, ROCer, reg_check], verbose=2)

    y_pred = [np.argmax(value) for value in model.predict(test_x)]
    acc = accuracy_score(y_pred, test_y[:, 1])

    print("Accuracy: ", round(acc,4)*100)

    model.save("{}/FinalFCModel_{}.h5".format(checkIfExists, nr_examples))
    model.save_weights("{}/FCModelWeights_{}.h5".format(checkIfExists, nr_examples))
    json_string = model.to_json()

    with open("{}/FinalFCModel_{}.json".format(checkIfExists, nr_examples), "w") as f:
        f.write(json_string)

    losses = history.history["loss"]
    val_losses = history.history["val_loss"]
    accs = history.history["acc"]
    val_accs = history.history["val_acc"]

    plt.figure()
    plt.plot(losses, label="train")
    plt.plot(val_losses, label="test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("{}/loss.png".format(checkIfExists))

    plt.figure()
    plt.plot(accs, label="train")
    plt.plot(val_accs, label="test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("{}/accuracy.png".format(checkIfExists))

    plt.figure()
    plt.plot(ROCer.roc_aucs)
    plt.title("Roc Auc")
    plt.xlabel("Epoch")
    plt.ylabel("ROC_AUC_Score")
    plt.savefig("{}/rocs.png".format(checkIfExists))
    # plt.show()

    t = time.clock()-start
    stats_log = {"losses": losses, "val_losses": val_losses, "acc": val_accs, "train_acc": accs, "nodes": layers, "roc_auc": ROCer.roc_aucs, \
    "batch": batch_size, "epochs": hm_epochs, "activations": activations, "optimizer": optimizer_name, \
    "features": transverse_wo_SPD, "train_examples": nr_examples, "time": t, \
    "test_size": test_x.shape[0], "DropOut": DropOut, \
    "feature_size": nr_features, \
    "standFeatures": standard_features + global_standardized, \
    "LogTransformed": standard_log + non_standard_log, \
    "stats_nr": stats_nr, "batchNorm": bn, "PCA": PCA}

    filename1 = "{}/stats{}.pickle".format(checkIfExists, stats_nr)
    with open(filename1, "wb") as f:
        pickle.dump(stats_log, f)

    print("\nGenerated: {}\n".format(filename1))

    K.clear_session()


if __name__ == "__main__":
    models = len(several_nn)
    counter = 1

    check_and_correct_parameters(models)

    #### Fetch all values needed to process the data and train the network
    for nodes in several_nn:
        DropOut = DropOut_list[counter-1]
        hm_epochs = hm_epochs_list[counter-1]
        batch_size = batch_size_list[counter-1]
        activations = activations_list[counter-1]
        optimizer_name = optimizer_names[counter-1]
        bn = BN_list[counter-1]
        nr_examples = nr_examples_list[counter-1]
        layers = several_nn[counter-1]

        #### Training
        print("Read data...")
        with h5py.File("{}/Processed_Train_400000.h5".format(datafolder), "r") as Data:
            data = Data["Train"][:nr_examples]
            if PCA:
                print("PCA data...")
                data = inc_pca.transform(data)
                print(data.shape)

        with open("{}/Train_labels_400000.pickle".format(datafolder), "rb") as f:
            labels = pickle.load(f)
            labels = labels[:nr_examples]
            labels = enc.transform(labels.reshape(-1,1))

        train_neural_network()

        counter += 1
