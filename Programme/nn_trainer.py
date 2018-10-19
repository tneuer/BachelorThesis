"""
    @author: Thomas Neuer (tneuer)

    This script trains the neural networks specified by the user. Many different networks can be
    trained, one after another (TODO: parallel implementation)

    The user input needs to be specified only the first part after all the imports. The rest is handled
    by the script (TODO: Function or class implementation)

    The following inputs can be specified:

    5)* networkTypes: indicated which networks are trained:
        1: feed forward neural network
        2: autoencoder with identity in last layer
        3: autoencoder with user specified activation in last layer
    6) several_nn: list of lists. All other lists need to be as long as this list or one. Contains the network
        architecture. Length of sublist determines the number of layers and the integer value determines the number
        of nodes per layer.
    7)* activations_list: list of lists with strings containing which activation function should be used.
    8)* DropOut_list: list of list with the keep probability per alyer and per network
    9)* hm_epochs_list: number of epochs trained on the data
    12)* batch_size_list: number of batch size per neural network.
    13)* optimizer_names: used optimizer for the diminuation of the cost function. Available are:
        - Adam - learning rate
        - AdaGrad - learning rate
        - Momentum - learning rate - momentum factor
        - RMS - learning rate
    18)* BN_list : true, if batch normalisation should be applied
    19) folder: Specify destination folder for output. Different for every cluster and needs to be cheked.

    All inputs marked with '*' need to be either of length one or equal to len(several_nn). If it is one
    the same option is used for every network.
"""

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os.path
import h5py
import gc
import sys

from nn_model import feed_forward_nn, autoencoder
from sklearn.preprocessing import OneHotEncoder
from pipeliner import batch_pipeline, logtransform_standard_pipeline
from baUtils import get_sg_and_bg

######################
"""
Only this part needs to be filled by the user.
"""
######################

# nr_examples = 2367829
networkTypes = [1]

several_nn = [[100, 50, 10, 2]] #, [100, 50], [100], [250], [50], [300, 300, 250, 100]]
activations_list = [["elu"]*4] #+ [["elu"]*2] + [["elu"]]*3 + [["elu"]*4]]
DropOut_list = [None]

hm_epochs_list = [200]
batch_size_list = [256]
optimizer_names = [["Adam", 0.0007]]
nr_examples_list = [400000]
earlyStopping = 20

BN_list = [False]
PCA = False

folder = "../allOutputs/AutoEncoding/"
datafolder = "../data/"
######################

network_type_list = [["FC", "AutoID", "AutoUser"][t] for t in networkTypes]
fcost_list = [["Cross-entropy", "LS-to-self", "LS-to-self"][t] for t in networkTypes]

transverse_wo_SPD = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']

standard_features = transverse_wo_SPD[:7]
standard_log = ["muplus_PT", "muminus_PT"]

non_standard_features = transverse_wo_SPD[6:]
global_standardized = non_standard_features[1:-2]
non_standard_log = ["tracks_PT", "tracks_IPCHI2", "tracks_IP"]

# Load test data
print("Read test...")
with h5py.File("{}Processed_Val_mu_263093.h5".format(datafolder), "r") as valData:
    test_x = valData["Val"][:]

with open("{}Val_mu_labels_263093.pickle".format(datafolder), "rb") as f:
    test_y = pickle.load(f)

with open("{}inc_pca_mu_354.pickle".format(datafolder), "rb") as f:
    inc_pca = pickle.load(f)

enc = OneHotEncoder(sparse=False)
test_y = enc.fit_transform(test_y.reshape(-1,1))

if PCA:
    print("PCA test")
    test_x = inc_pca.transform(test_x)

nr_features = test_x.shape[1]
nr_classes = test_y.shape[1]

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
    parameter_dict = {"Network Types": network_type_list, "Activations": activations_list, \
    "DropOuts": DropOut_list, "Epochs": hm_epochs_list, \
    "batch_sizes": batch_size_list, "Optimizers": optimizer_names, "Cost": fcost_list, "BatchNorm": BN_list, "Nr_examples": nr_examples_list}

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


def train_neural_network(x):
    """Trains the neural network

    Core function which trains the network. Makes heavy use of the nn_model module and uses all inputs.

    Parameters
    ----------
    x : tf.placeholder
        has the same dimension as the input vector and is used by tensorflow to represent exactly this vector.

    data : DataSet
        DataSet obect which contains ALL relevant information about the data, like labels and chunks. This one is
        iterated through while training the network.

    Returns
    -------
    No return values, but it saves different output and analysis obects in the specified folder ('folder' variable in
    header). The stats object contains all information given in pythonic language. Testsize, labels, chunks, processing
    properties,... The other two folders which are created contain the train and test set information used by tensorboard.
    Select those in the bash command $tensorboard --logdir=nn_logs.
    """

    starter = time.clock()
    with tf.name_scope("Model"):
        if network_type == "FC":
            prediction = feed_forward_nn(x, nr_examples, nr_features, nr_classes, nodes, activations, DropOut, bn, tr_phase)
            pred_proba = tf.nn.softmax(prediction, name="predProba")
            cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y), name="cost" )
            saveDirectory = "{}Outputs/FC/".format(folder)

        elif network_type == "AutoID":
            prediction, encoded = autoencoder(x, nr_features, nodes, activations, DropOut, lastLayer="ID")
            pred = tf.identity(encoeded, name="predict")
            cost = tf.reduce_mean(tf.pow(x - prediction, 2), name="cost")
            saveDirectory = "{}Outputs/AutoEncoder/".format(folder)
        elif network_type == "AutoUser":
            prediction, encoded = autoencoder(x, nr_features, nodes, activations, DropOut, lastLayer="User")
            pred = tf.identity(encoded, name="predict")
            cost = tf.reduce_mean(tf.pow(x - prediction, 2), name="cost")
            saveDirectory = "{}Outputs/AutoEncoder/".format(folder)
        else:
            raise TypeError("Invalid network type. Enter FC or Auto")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if optimizer_name[0] == "Adam":
                optimizer = tf.train.AdamOptimizer(optimizer_name[1]).minimize(cost)
            elif optimizer_name[0] == "Adagrad":
                optimizer = tf.train.AdagradOptimizer(optimizer_name[1]).minimize(cost)
            elif optimizer_name[0] == "Momentum":
                optimizer = tf.train.MomentumOptimizer(optimizer_name[1], optimizer_name[2]).minimize(cost)
            elif optimizer_name[0] == "RMS":
                optimizer = tf.train.RMSPropOptimizer(optimizer_name[1]).minimize(cost)
            else:
                print(optimizer_name[0] == "Adam")
                raise TypeError("Not a valid Optimizer given. Choose from Adam, Adagrad, Momentum or RMS!")

        tf.summary.scalar("cost", cost)

    with tf.name_scope("Accuracy"):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1), name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, "float64"), name="accuracy")

        tf.summary.scalar("TestAccuracy", accuracy)


    losses = []
    val_losses = []
    train_acc = []
    test_acc = []
    times = []

    losses.append([])
    val_losses.append([])
    train_acc.append([])
    test_acc.append([])

    saver = tf.train.Saver(max_to_keep=10)
    curr_min_loss = np.inf
    earlyStoppingCounter = 0
    with tf.Session() as sess: #config=tf.ConfigProto(log_device_placement=True)

        for i in range(200):
            checkIfExists = saveDirectory + "nn_logs{}".format(i)
            if not os.path.isdir(checkIfExists):

                Trainwriter = tf.summary.FileWriter("{}/train".format(checkIfExists), sess.graph)
                Testwriter = tf.summary.FileWriter("{}/test".format(checkIfExists), sess.graph)
                merged = tf.summary.merge_all()
                stats_nr = i
                print("Folder  nn_logs{} generated.".format(i))
                break

        sess.run(tf.global_variables_initializer())

        np.random.seed(42)
        indices = np.arange(nr_examples)
        for epoch in range(hm_epochs):
            s = time.clock()
            epoch_loss = 0

            np.random.shuffle(indices)
            if (epoch % 20 == 0 or epoch == hm_epochs-1):
                print("Model saved.")
                save_graph = not bool(epoch)
                saver.save(sess, "{}/Models/NNModel".format(checkIfExists), global_step=epoch, write_meta_graph=save_graph)

            tr_acc = []
            # with h5py.File("../data/Processed_Train_400000.h5", "r") as hf:
            for start in range(0, nr_examples, batch_size):
                # print(start, "/ ", nr_examples)
                batch_ix = sorted(list(indices[start:(start+batch_size)]))
                batch_x = data[batch_ix]
                batch_y = labels[batch_ix]

                if network_type == "FC":
                    _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y, tr_phase: 1})
                    tr_acc.append(accuracy.eval({x:batch_x, y: batch_y, tr_phase: 1}))
                elif network_type == "AutoUser" or network_type == "AutoID":
                    _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x})
                epoch_loss += c

            if network_type == "FC":
                tr_acc = np.mean(tr_acc)
                te_acc = accuracy.eval({x:test_x, y: test_y, tr_phase: 1})
                epoch_val_loss = sess.run([cost], feed_dict = {x: test_x, y: test_y, tr_phase: 0})[0]

            elif network_type == "AutoUser" or network_type == "AutoID":
                correct = "-"
                accuracy = "-"
                tr_acc = 0
                te_acc = 0
                epoch_val_loss = sess.run([cost], feed_dict = {x: test_x, y: test_y, tr_phase: 0})[0]

            losses[0] += [epoch_loss/(nr_examples/batch_size)]
            val_losses[0] += [epoch_val_loss]
            train_acc[0] += [tr_acc]
            test_acc[0] += [te_acc]

            print("Model", counter, "/", models, \
            " Epoch ", epoch+1, "/", hm_epochs, " Loss val/train: ", round(val_losses[0][-1], 5), round(losses[0][-1], 5), " Accuracy val/train: ", round(te_acc, 3), round(tr_acc,3))

            summary = sess.run([merged], feed_dict = {x: test_x, y: test_y, tr_phase: 1})[0]
            Testwriter.add_summary(summary, epoch+1)

            times.append(time.clock()-s)
            print("Remaining time: ", round(np.mean(times)/60*(hm_epochs-epoch-1), 3), "min")

            if epoch_val_loss < curr_min_loss:
                earlyStoppingCounter = 0
                curr_min_loss = epoch_val_loss
                save_graph = False
                saver.save(sess, "{}/Models/NNModel-Best".format(checkIfExists), write_meta_graph=save_graph)
            else:
                earlyStoppingCounter += 1
                if earlyStoppingCounter == earlyStopping:
                    print("\nEarly Stopping after {} epochs.\n".format(epoch))
                    save_graph = False
                    saver.save(sess, "{}/Models/NNModel".format(checkIfExists), global_step=epoch, write_meta_graph=save_graph)
                    break

            print(earlyStoppingCounter, "/", earlyStopping, "round(s) without improvement.")

        #Testing of the network
        if network_type == "FC":
            with tf.name_scope("Accuracy_Final"):
                print("Final Accuracy: ", accuracy.eval({x:test_x, y: test_y, tr_phase: 1}))


    losses = np.array(losses)
    val_losses = np.array(val_losses)
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    t = time.clock() - starter

    stats_log = {"losses": losses, "val_losses": val_losses, "acc": test_acc, "train_acc": train_acc, "nodes": nodes, \
    "batch": batch_size, "epochs": hm_epochs, "activations": activations, "optimizer": optimizer_name, \
    "features": transverse_wo_SPD, "train_examples": nr_examples, "time": t, \
    "test_size": test_x.shape[0], "DropOut": DropOut, "NNType": network_type, \
    "feature_size": nr_features, "fcost": fcost, \
    "standFeatures": standard_features + global_standardized, \
    "LogTransformed": standard_log + non_standard_log, \
    "stats_nr": stats_nr, "batchNorm": bn, "PCA": PCA}

    filename1 = "{}/stats{}.pickle".format(checkIfExists, stats_nr)
    with open(filename1, "wb") as f:
        pickle.dump(stats_log, f)

    print("\nGenerated: {}\n".format(filename1))
    tf.reset_default_graph()


if __name__ == "__main__":
    models = len(several_nn)
    counter = 1

    check_and_correct_parameters(models)

    #### Fetch all values needed to process the data and train the network
    for nodes in several_nn:
        DropOut = DropOut_list[counter-1]
        hm_epochs = hm_epochs_list[counter-1]
        batch_size = batch_size_list[counter-1]
        network_type = network_type_list[counter-1]
        fcost = fcost_list[counter-1]
        activations = activations_list[counter-1]
        optimizer_name = optimizer_names[counter-1]
        bn = BN_list[counter-1]
        nr_examples = nr_examples_list[counter-1]

        #### Preprocessing
        x = tf.placeholder('float', [None, nr_features], name="x")
        y = tf.placeholder('float', name="y")
        tr_phase = tf.placeholder(tf.bool, name="trainPhase")

        print("Read data...")
        with h5py.File("{}Processed_Train_mu_400000.h5".format(datafolder), "r") as Data:
            data = Data["Train"][:nr_examples]
        if PCA:
            print("PCA data...")
            data = inc_pca.transform(data)

        print("Read labels...")
        with open("{}Train_mu_labels_400000.pickle".format(datafolder), "rb") as f:
            labels = pickle.load(f)
            labels = labels[:nr_examples]
            labels = enc.transform(labels.reshape(-1,1))

        #### Training
        train_neural_network(x)

        counter += 1

























