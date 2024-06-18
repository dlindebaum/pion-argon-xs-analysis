# Created 19/01/24
# Dennis Lindebaum
# GNN models

import os
import pickle
import copy
import numpy as np
import awkward as ak
import tensorflow as tf
import tensorflow_gnn as tfgnn
import matplotlib.pyplot as plt
from python.gnn import DataPreparation
from apps.cex_toy_parameters import PlotCorrelationMatrix as plot_confusion_matrix
# from python.analysis import Plots

def generic_dense_layer(number, activation="relu", dropout=0.1, regulariser=8e-5):
    """A Dense layer with regularization (L2 and Dropout)."""
    regularizer = tf.keras.regularizers.l2(regulariser)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout)
    ])

def beam_collection_layer(
        message_dimension,
        final_dimension,
        dropout=0.1,
        regulariser=8e-5):
    """Update the beam node with beam connection PFO data."""
    beam_collection_update = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "beam": tfgnn.keras.layers.NodeSetUpdate(
                {"beam_connections": tfgnn.keras.layers.SimpleConv(
                    sender_edge_feature=tfgnn.HIDDEN_STATE,
                    message_fn=generic_dense_layer(
                        message_dimension,
                        dropout=dropout, regulariser=regulariser),
                    reduce_type="sum",
                    receiver_tag=tfgnn.SOURCE)},
                tfgnn.keras.layers.NextStateFromConcat(
                    generic_dense_layer(
                        final_dimension,
                        dropout=dropout, regulariser=regulariser))
            )
        })
    return beam_collection_update

def pfo_update_layer(
        message_dimension,
        final_dimension,
        dropout=0.1,
        regulariser=8e-5):
    """
    Update to PFOs with neighbouring PFO data.
    
    This step does the following:
    - For each neighour edge + connected PFO: concatenate the hidden
        states of this node, the neighbour edge, then connected PFO.
    - Perform a dense layer with `message_dimension` outputs over this
        concatenated set of data.
    - Pool (sum) all the values calculated above for every connected
        edge/PFO pair.
    - Concatenate this pooled data with this node's hidden state.
    - Perform a dense layer with output size `final_dimensions` over
        this concatenated data.

    Parameters
    ----------
    message_dimension : int
        Dimensions of the vector produced by the dense layer acting
        over the PFo/edge covolutions.
    final_dimension : int
        Dimensions of the final output of the layer.
    dropout : float
        Probablitity that a given weight will not get used in the
        training.
    regulariser : float
        Penalisation term for having large magnitude weights (L2
        regularisation).
    
    Returns
    -------
    tfgnn.keras.layers.GraphUpdate
        Graph update layer.
    """
    pfo_node_update = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "pfo": tfgnn.keras.layers.NodeSetUpdate(
                {"neighbours": tfgnn.keras.layers.SimpleConv(
                    sender_edge_feature=tfgnn.HIDDEN_STATE,
                    message_fn=generic_dense_layer(
                        message_dimension,
                        dropout=dropout, regulariser=regulariser),
                    reduce_type="sum",
                    receiver_tag=tfgnn.TARGET)},
                tfgnn.keras.layers.NextStateFromConcat(
                    generic_dense_layer(
                        final_dimension,
                        dropout=dropout, regulariser=regulariser))
            )
        })
    return pfo_node_update

def _neighbour_update_layer(
        final_dims,
        dropout=0.1,
        regulariser=8e-5):
    """Update neighbour edges using connected PFO data"""
    neighbours_edge_update = tfgnn.keras.layers.GraphUpdate(
        edge_sets={
            "neighbours": tfgnn.keras.layers.EdgeSetUpdate(
               tfgnn.keras.layers.NextStateFromConcat(generic_dense_layer(final_dims))
            )})
    return neighbours_edge_update

def _beam_conn_update_layer():
    beam_conn_edge_update = tfgnn.keras.layers.GraphUpdate(
        edge_sets={
            "beam_connections": tfgnn.keras.layers.EdgeSetUpdate(
                tfgnn.keras.layers.NextStateFromConcat(dense(beam_edge_next_dim))
            )})
    return beam_conn_edge_update
