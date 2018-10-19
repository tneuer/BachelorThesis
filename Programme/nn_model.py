"""
@autho: Thomas Neuer (tneuer)

In this module, different network types are implemented with tensorflow, including:
    1) a feed forward network, with possibility on Drop out
    2) an autoencoder where every layer can be specified (tanh works, but then the data needs to be downscaled)
    3) an autoencoder where the last layer is the identity (instead of tanh for example) in order to cover
    the whole range of real values.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def feed_forward_nn(data, nr_features, n_classes, nodes, activations, DropOut, BN, train_phase):
    """Implement a feed forward neural network.

    The simplest but also often an effective neural network model. A DropOut can be added
    in order to improve generalisation. Only the hidden layers need to be specified.
    There are also many lines of code which are fed into tensorboard which can then be used
    to further analyze the network performnce. Inclued are:
        * the mean of the biases and weights in every layer
        * the histogram and distribution of weights and biases in every layer

    Parameters
    ----------
    data : np.matrix
        feature matrix or batches, where one row is an observation and a column is a feature.

    nr_features : int
        number of features.

    n_classes : int
        number of classes (output groups).

    nodes : list
        The length of this list determines the number of hidden layers (only those need to be specified).
        Every element in this list needs to be an integer which determines the number of nodes per layer.

    activations : list
        A list of activation functions. Needs to have the same length as 'nodes'. All regular tf.train
        activation functions are supported.

    DropOut : list
        A list of floats between 0 and 1. Determines the keep probability per layer. This list needs to have
        the same length as 'nodes'.

    BN : boolean
        True, then batch normalisation is applied

    train_phase : boolean
        Only considered if batch normalisation is true. Rebuild network for test phase (train_phase = False)

    Returns
    -------
    curr_ipt : np.matrix
        Entries of last layer WITHOUT activation function (often softmax).
    """

    depth = len(nodes) + 1
    nodes.insert(0, nr_features)
    nodes.append(n_classes)
    activation = ["tf.nn.{}".format(actv) for actv in activations]
    weights = {}
    biases = {}

    for i in range(depth):
        weights["layer{}".format(i)] = tf.Variable(tf.random_normal([nodes[i], nodes[i+1]], stddev=np.sqrt(4/(nodes[i]+nodes[i+1])) ))

        biases["layer{}".format(i)] = tf.Variable(tf.random_normal([nodes[i+1]]))

        tf.summary.histogram("Weights{}".format(i), weights["layer{}".format(i)])
        mean_w = tf.reduce_mean(weights["layer{}".format(i)])
        tf.summary.scalar("mean_weights{}".format(i), mean_w)
        tf.summary.histogram("Biases{}".format(i), biases["layer{}".format(i)])
        mean_b = tf.reduce_mean(biases["layer{}".format(i)])
        tf.summary.scalar("mean_biases{}".format(i), mean_b)
    del nodes[0]; del nodes[-1]

    curr_ipt = data
    for i in range(depth-1):
        with tf.name_scope("Layer{}".format(i)):
            layer = tf.add(tf.matmul(curr_ipt, weights["layer{}".format(i)]), biases["layer{}".format(i)])
            if BN:
                try:
                    layer = tf.contrib.layers.batch_norm(layer, center=True, scale=True, is_training=train_phase, epsilon =0.01)
                except ValueError:
                    layer = tf.contrib.layers.batch_norm(layer, center=True, scale=True, is_training=train_phase, reuse = True, epsilon=0.01)

            curr_ipt = eval(activation[i])(layer)

            if DropOut != None:
                tf.nn.dropout(curr_ipt, DropOut[i])

    with tf.name_scope("Layer{}".format(depth-1)):
        curr_ipt = tf.matmul(curr_ipt, weights["layer{}".format(depth-1)]) + biases["layer{}".format(depth-1)]
    return curr_ipt


def autoencoder(data, nr_features, nodes, activations, DropOut, lastLayer = "ID"):
    """Implement an autoencoder neural network.

    A form of neural networks which tries to map the input via a 1-1 transformation on itself.
    First a dimensionality reduction is tried to obtain. the last layer should have two nodes,
    in order to be presentable in a 2D plot. Only the architecture of the encoder needs to
    be specified.
    There are also many lines of code which are fed into tensorboard which can then be used
    to further analyze the network performnce. Inclued are:
        * the mean of the biases and weights in every layer
        * the histogram and distribution of weights and biases in every layer

    Parameters
    ----------
    data : np.matrix
        feature matrix or batches, where one row is an observation and a column is a feature.

    nr_features : int
        number of features.

    nodes : list
        The length of this list determines the number of hidden layers (only those need to be specified).
        Every element in this list needs to be an integer which determines the number of nodes per layer.

    activations : list
        A list of activation functions. Needs to have the same length as 'nodes'. All regular tf.train
        activation functions are supported.

    DropOut : list
        A list of floats between 0 and 1. Determines the keep probability per layer. This list needs to have
        the same length as 'nodes'.

    lastLayer : str ["ID", else "user"]
        Determines which activation function should be used in the last layer. If 'ID' is chosen, the last
        layer performs an identity operation. Else the same activation function as in the first layer is applied.

    Returns
    -------
    curr_ipt : np.matrix
        Entries of last layer WITHOUT activation function (often softmax).
    """

    depth = len(nodes)
    nodes.insert(0, nr_features)
    weights = {}
    biases = {}

    for i in range(depth):
        weights["encoder_layer{}".format(i)] = tf.Variable(tf.random_normal([nodes[i], nodes[i+1]], stddev=np.sqrt(4/(nodes[i]+nodes[i+1])) ))
        weights["decoder_layer{}".format(depth-i-1)] = tf.Variable(tf.random_normal([nodes[i+1], nodes[i]], stddev=np.sqrt(4/(nodes[i]+nodes[i+1])) ))

        biases["encoder_layer{}".format(i)] = tf.Variable(tf.random_normal([nodes[i+1]]))
        biases["decoder_layer{}".format(depth-i-1)] = tf.Variable(tf.random_normal([nodes[i]]))

        tf.summary.histogram("EncoderWeights{}".format(i), weights["encoder_layer{}".format(i)])
        mean_w = tf.reduce_mean(weights["encoder_layer{}".format(i)])
        tf.summary.scalar("mean_EncoderWeights{}".format(i), mean_w)
        tf.summary.histogram("EncoderBiases{}".format(i), biases["encoder_layer{}".format(i)])
        mean_b = tf.reduce_mean(biases["encoder_layer{}".format(i)])
        tf.summary.scalar("mean_EncoderBiases{}".format(i), mean_b)

        tf.summary.histogram("DecoderWeights{}".format(depth-i-1), weights["decoder_layer{}".format(depth-i-1)])
        mean_w = tf.reduce_mean(weights["decoder_layer{}".format(depth-i-1)])
        tf.summary.scalar("mean_DecoderWeights{}".format(depth-i-1), mean_w)
        tf.summary.histogram("DecoderBiases{}".format(depth-i-1), biases["decoder_layer{}".format(depth-i-1)])
        mean_b = tf.reduce_mean(biases["decoder_layer{}".format(depth-i-1)])
        tf.summary.scalar("mean_DecoderBiases{}".format(depth-i-1), mean_b)
    del nodes[0]

    activation = ["tf.nn.{}".format(actv) for actv in activations]
    def encoder(x):
        curr_ipt = x
        for i in range(depth):
            layer = tf.add(tf.matmul(curr_ipt, weights["encoder_layer{}".format(i)]), biases["encoder_layer{}".format(i)])
            curr_ipt = eval(activation[i])(layer)

            if DropOut != None:
                tf.nn.dropout(curr_ipt, DropOut[i])

        return curr_ipt

    def decoder(x):
        curr_ipt = x
        ddepth = depth-1 if lastLayer == "ID" else depth
        for i in range(ddepth):
            layer = tf.add(tf.matmul(curr_ipt, weights["decoder_layer{}".format(i)]), biases["decoder_layer{}".format(i)])
            curr_ipt = eval(activation[depth-i-1])(layer)

            if DropOut != None:
                tf.nn.dropout(curr_ipt, DropOut[i])

        if lastLayer == "ID":
            layer = tf.add(tf.matmul(curr_ipt, weights["decoder_layer{}".format(ddepth)]), biases["decoder_layer{}".format(ddepth)])
            curr_ipt = layer

            if DropOut != None:
                tf.nn.dropout(curr_ipt, DropOut[ddepth])
        return curr_ipt

    encoder_run = encoder(data)
    decoder_run = decoder(encoder_run)

    output = decoder_run

    return output, encoder_run








