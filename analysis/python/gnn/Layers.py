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

def parse_constructor(constructor: list, parameters: dict):
    """
    Take a list of constructors (classes which build layers in this
    module) and layer parameters to create two lists, the first
    containing the functions themselves which define the layers, and
    the second containing a list indicating which output the function
    at the corresponding index should correspond to, or None if it is
    not an output.

    Note that by default, this assumes that the message passing steps
    etc. of the model all have the same dimensions (for residual
    states), and that the message dimensions/heads are the same, and
    are thus stored in the parameters. This values can be overwritten
    on a per layer basis by overwriting the corresponding argument in
    the constructor. I.e. the number of heads in the beam collection
    step could be changed with the following constructor:
    ```
    Layers.beam_collection(beam_node_dim=64,
                           beam_message_heads=16,
                           beam_message_channels=32)
    ```

    Parameters
    ----------
    constructor : list
        List of `LayerConstructor`s which define the layers of the
        model in order.
    parameters : dict
        Dictionary containing shared layer properties (i.e. dropout
        rates, or regulariser values).

    Returns
    -------
    functions : list
        List containing the function of the layer to be run on the
        graph as produced by the previous non-output layer, or a
        `tf.keras.Input` layer for the first function.
    outputs : list
        List containing None if a layer isn't an output, or a string
        containing the name of the output otherwise.
    """
    functions = []
    outputs = []
    for layer in constructor:
        if isinstance(layer, LoopConstructor):
            funcs, outs = parse_constructor(layer.constructor, parameters)
            functions += funcs
            outputs += outs
        else:
            functions.append(layer.get_func(parameters))
            outputs.append(layer.output_name)
    return functions, outputs

class LayerConstructor():
    def __init__(self, *output_name, message_type="GATv2", final_step=True, **kwargs):
        self.do_final_loop_step = final_step
        if len(output_name) == 0:
            self.output_name=None
        elif len(output_name) == 1:
            self.output_name = output_name[0]
        else:
            raise TypeError(
                f"{type(self).__name__} takes at most 1 positional argument")
        known_messages = ["conv", "gatv2"]
        if message_type.lower() not in known_messages:
            raise ValueError(f"Unknown message type: {message_type}. "
                             + f"Must be one of {known_messages}")
        self.message_type = message_type.lower()
        self.additional_args = kwargs
        self.repr_kwargs = self.additional_args.copy()
        return
    
    def _remove_default_kwargs_from_repr(self, default_kwargs):
        for key, val in default_kwargs.items():
            if self.repr_kwargs[key] == val:
                del self.repr_kwargs[key]
        return 

    def __repr__(self):
        repr_string = f"{type(self).__name__}("
        if self.output_name is not None:
            repr_string += f"{self.output_name}, "
        if not self.do_final_loop_step:
            repr_string += "final_step=False, "
        for key, val in self.additional_args.items():
            repr_string += f"{key}={val}, "
        if repr_string[-2:] == ", ":
            repr_string = repr_string[:-2]
        repr_string += ")"
        return repr_string

    def get_func(self, parameters):
        parameters.update(self.additional_args)
        return self._func(**parameters)

    def _func(self, **kwargs):
        pass

    def load_from_repr(self, repr_string):
        pass

class LoopConstructor():
    def __init__(self, loop_constructor, loops=1):
        self.looped = True
        self.constructor_repr = repr(loop_constructor)
        self.loops = loops
        self.constructor = []
        for i in range(self.loops):
            last_loop = i == (self.loops-1)
            for layer in loop_constructor:
                if layer.do_final_loop_step or not last_loop:
                    self.constructor.append(layer)
        return

    def __repr__(self):
        return (f"{type(self).__name__}({self.constructor_repr}, "
                + f"loops={self.loops})")

class Setup(LayerConstructor):
    def _func(self, **kwargs):
        def layer(graph):
            return graph.merge_batch_to_components()
        return layer

class InitialState(LayerConstructor):
    def __init__(
            self, *output_name,
            pfo_hidden=None,
            neighbours_hidden=None,
            beam_connections_hidden=None,
            **kwargs):
        kwargs.update({"pfo_hidden": pfo_hidden,
                       "neighbours_hidden": neighbours_hidden,
                       "beam_connections_hidden": beam_connections_hidden})
        super().__init__(*output_name, **kwargs)
        self._remove_default_kwargs_from_repr({
            "pfo_hidden": None,
            "neighbours_hidden": None,
            "beam_connections_hidden": None})
        return

    def _func(self, **kwargs):
        return map_to_initial_state_layer(**kwargs)

class ReadoutNode(LayerConstructor):
    def __init__(
            self, *output_name,
            which_nodes="beam",
            **kwargs):
        kwargs.update({"which_nodes":which_nodes})
        super().__init__(*output_name, **kwargs)
        self._remove_default_kwargs_from_repr({
            "which_nodes": "beam"})
        return
    
    def _func(self, **kwargs):
        return tfgnn.keras.layers.Readout(node_set_name=which_nodes)(graph)

class ReadoutClassifyNode(LayerConstructor):
    def __init__(
            self, *output_name,
            n_outputs=None, # If None, copy data without applying dense layer
            hidden=None,
            which_nodes="pfo",
            which_feature=tfgnn.HIDDEN_STATE,
            **kwargs):
        kwargs.update({"n_outputs": n_outputs,
                       "hidden": hidden,
                       "which_nodes":which_nodes,
                       "which_feature": which_feature})
        super().__init__(*output_name, **kwargs)
        self._remove_default_kwargs_from_repr({
            "n_outputs": 1,
            "hidden": None,
            "which_nodes": "pfo",
            "which_feature": tfgnn.HIDDEN_STATE})
        return
    
    def _func(self, **kwargs):
        return classifer_and_readout(
            "node_sets_fn",
            which_data = kwargs["which_nodes"],
            **kwargs)

class ReadoutEdge(LayerConstructor):
    def __init__(
            self, *output_name,
            which_edges="neighbours",
            **kwargs):
        kwargs.update({"which_edges":which_edges})
        super().__init__(*output_name, **kwargs)
        self._remove_default_kwargs_from_repr({
            "which_edges": "neighbours"})
        return
    
    def _func(self, **kwargs):
        return tfgnn.keras.layers.Readout(edge_set_name=which_edges)(graph)

class ReadoutClassifyEdge(LayerConstructor):
    def __init__(
            self, *output_name,
            n_outputs=1,
            hidden=None,
            which_edges="neighbours",
            which_feature=tfgnn.HIDDEN_STATE,
            **kwargs):
        kwargs.update({"n_outputs": n_outputs,
                       "hidden": hidden,
                       "which_edges":which_edges,
                       "which_feature": which_feature})
        super().__init__(*output_name, **kwargs)
        self._remove_default_kwargs_from_repr({
            "n_outputs": 1,
            "hidden": None,
            "which_edges": "neighbours",
            "which_feature": tfgnn.HIDDEN_STATE})
        return
    
    def _func(self, **kwargs):
        return classifer_and_readout(
            "edge_sets_fn",
            which_data = kwargs["which_edges"],
            **kwargs)

class NodeUpdate(LayerConstructor):
    def __init__(
            self, *output_name,
            next_state="residual",
            **kwargs):
        if not hasattr(self, "kwarg_dict"):
            raise NotImplementedError(
                "Class should have a kwarg_dict indicating which "
                + "kwargs to pick out for dimensions. This is an "
                + "improperly initialised class.")
        known_next_states = ["residual", "concat"]
        if next_state.lower() not in known_next_states:
            raise ValueError(f"Unknown next state: {next_state}. "
                             + f"Must be one of {known_next_states}")
        kwargs.update({"next_state": next_state})
        super().__init__(*output_name, **kwargs)
        self._remove_default_kwargs_from_repr({
            "next_state": "residual"})
        return
    
    def _func(self, **kwargs):
        final_dim_key = self.kwarg_dict["final_dim"]
        if kwargs["next_state"] == "concat":
            next_state_func = tfgnn.keras.layers.NextStateFromConcat(
                generic_dense_layer(kwargs[final_dim_key], **kwargs))
        else: # "next_state" == "residual"
            next_state_func = tfgnn.keras.layers.ResidualNextState(
                generic_dense_layer(kwargs[final_dim_key], **kwargs))
        if self.message_type == "gatv2":
            message_func = gatv2_message(
                kwargs[self.kwarg_dict["gatv2_heads"]],
                kwargs[self.kwarg_dict["gatv2_channels"]],
                **kwargs)
        else: # self.message_type == "conv"
            message_func = convolution_message(
                kwargs[self.kwarg_dict["conv_dim"]], **kwargs)
        return tfgnn.keras.layers.GraphUpdate(
            node_sets={
                self.kwarg_dict["node"]: tfgnn.keras.layers.NodeSetUpdate(
                    {self.kwarg_dict["edges"]: message_func},
                    next_state_func)})

class BeamCollection(NodeUpdate):
    def __init__(self, *args, **kwargs):
        self.kwarg_dict = {
            "final_dim"     : "beam_node_dim",
            "gatv2_heads"   : "beam_message_heads",
            "gatv2_channels": "beam_message_channels",
            "node"          : "beam", 
            "edges"         : "beam_connections"}
        return super().__init__(*args, *kwargs)

class PFOUpdate(NodeUpdate):
    def __init__(self, *args, **kwargs):
        self.kwarg_dict = {
            "final_dim"     : "node_dim",
            "gatv2_heads"   : "message_heads",
            "gatv2_channels": "message_channels",
            "node"          : "pfo", 
            "edges"         : "neighbours"}
        return super().__init__(*args, *kwargs)

class EdgeUpdate(LayerConstructor):
    def __init__(
            self, *output_name,
            next_state="residual",
            **kwargs):
        if not hasattr(self, "kwarg_dict"):
            raise NotImplementedError(
                "Class should have a kwarg_dict indicating which "
                + "kwargs to pick out for dimensions. This is an "
                + "improperly initialised class.")
        known_next_states = ["residual", "concat"]
        if next_state.lower() not in known_next_states:
            raise ValueError(f"Unknown next state: {next_state}. "
                             + f"Must be one of {known_next_states}")
        kwargs.update({"next_state": next_state})
        super().__init__(*output_name, **kwargs)
        self._remove_default_kwargs_from_repr({
            "next_state": "residual"})
        return
    
    def _func(self, **kwargs):
        final_dim_key = self.kwarg_dict["final_dim"]
        if kwargs["next_state"] == "concat":
            next_state_func = tfgnn.keras.layers.NextStateFromConcat(
                generic_dense_layer(kwargs[final_dim_key], **kwargs))
        else: # "next_state" == "residual"
            next_state_func = tfgnn.keras.layers.ResidualNextState(
                generic_dense_layer(kwargs[final_dim_key], **kwargs))
        return tfgnn.keras.layers.GraphUpdate(
            edge_sets={
                self.kwarg_dict["edge"]: tfgnn.keras.layers.EdgeSetUpdate(
                    next_state_func,
                    edge_input_feature = tfgnn.HIDDEN_STATE,
                    node_input_tags = (tfgnn.SOURCE, tfgnn.TARGET),
                    node_input_feature = tfgnn.HIDDEN_STATE)})

class NeighbourUpdate(EdgeUpdate):
    def __init__(self, *args, **kwargs):
        self.kwarg_dict = {
            "final_dim": "edge_dim",
            "edge"     : "neighbours"}
        return super().__init__(*args, *kwargs)

class BeamConnectionUpdate(EdgeUpdate):
    def __init__(self, *args, **kwargs):
        self.kwarg_dict = {
            "final_dim": "beam_edge_dim",
            "edge"     : "beam_connections"}
        return super().__init__(*args, *kwargs)

class Dense(LayerConstructor):
    def __init__(
            self, *output_name,
            depth=None,
            n_layers=1,
            **kwargs):
        if depth is None:
            raise TypeError("depth kwarg must be specified.")
        kwargs.update({"depth": depth,
                       "n_layers": n_layers})
        super().__init__(*output_name, **kwargs)
        return

    def _func(self, **kwargs):
        depth = kwargs["depth"]
        n_layers = kwargs["n_layers"]
        dropout_rate = kwargs["dropout_rate"]
        layers_list = []
        for i in range(n_layers):
            layers_list += [dense_core(depth, **kwargs)]
            if dropout_rate == 0.0:
                # Don't include dropout on the final layer if this is output
                if (self.output_name is not None) or (i < (n_layers-1)):
                    layers_list += [tf.keras.layers.Dropout(dropout_rate)]
        return tf.keras.Sequential(layers_list)

def dense_core(number, activation="relu", regularisation=None, **kwargs):
    layer_kwargs = {"activation": activation}
    if regularisation is not None:
        regulariser = tf.keras.regularizers.l2(regularisation)
        layer_kwargs.update({"kernel_regularizer": regulariser,
                             "bias_regularizer": regulariser})
    return tf.keras.layers.Dense(
        number,
        **layer_kwargs)

def generic_dense_layer(number, activation="relu", dropout_rate=0., regularisation=None, **kwargs):
    """A Dense layer with regularization (L2 and Dropout)."""
    if dropout_rate == 0.:
        return dense_core(number, activation="relu", regularisation=None, **kwargs)
    return tf.keras.Sequential([
        dense_core(number, activation="relu", regularisation=None, **kwargs),
        tf.keras.layers.Dropout(dropout_rate)])

def convolution_message(message_dim, message_reduction="sum", **kwargs):
    return tfgnn.keras.layers.SimpleConv(
        message_fn=generic_dense_layer(
            message_dim,
            **kwargs),
        reduce_type=message_reduction,
        receiver_tag=tfgnn.SOURCE,
        receiver_feature=tfgnn.HIDDEN_STATE,
        sender_node_feature=tfgnn.HIDDEN_STATE,
        sender_edge_feature=tfgnn.HIDDEN_STATE,
        combine_type="concat")

def gatv2_message(
        heads, channels,
        use_bias=True,
        **kwargs):
    if "dropout_rate" in kwargs.keys():
        dropout = kwargs["dropout_rate"]
    else:
        dropout == 0.0
    regulariser = None
    if regularisation in kwargs.keys():
        if regularisation is not None:
            regulariser = tf.keras.regularizers.l2(kwargs["regularisation"])
    return tfgnn.models.gat_v2.GATv2Conv(
        num_heads = heads,
        per_head_channels = channels,
        receiver_tag = tfgnn.SOURCE,
        receiver_feature = tfgnn.HIDDEN_STATE,
        sender_node_feature = tfgnn.HIDDEN_STATE,
        sender_edge_feature = tfgnn.HIDDEN_STATE,
        use_bias = use_bias,
        edge_dropout = dropout,
        attention_activation = 'leaky_relu',
        heads_merge_type = 'concat',
        activation = 'relu', 
        kernel_initializer = regulariser,
        kernel_regularizer = regulariser)

def multi_dense_layers(depth=None, n_layers=None, **kwargs):
    if depth is None:
        raise TypeError("depth must be set")
    if n_layers is None:
        raise TypeError("n_layers must be set")
    def layers_func(input):
        output = input
        for _ in range(n_layers):
            output = generic_dense_layer(depth, **kwargs)(output)
        return output
    return layers_func

def get_initial_edge_state_func(
        edge_dim=None,
        beam_edge_dim=None,
        neighbours_hidden=None,
        beam_connections_hidden=None,
        **kwargs):
    if edge_dim is None:
        raise TypeError("edge_dim must be set")
    if beam_edge_dim is None:
        raise TypeError("beam_edge_dim must be set")
    def set_initial_edge_state(edge_set, edge_set_name):
        if edge_set_name == "neighbours":
            data = edge_set[tfgnn.HIDDEN_STATE]
            if neighbours_hidden is not None:
                (depth, count) = neighbours_hidden
                data = multi_dense_layers(
                    depth=depth, n_layers=count, **kwargs)(data)
            return generic_dense_layer(edge_dim, **kwargs)(data)
        elif edge_set_name == "beam_connections":
            data = edge_set[tfgnn.HIDDEN_STATE]
            if beam_connections_hidden is not None:
                (depth, count) = beam_connections_hidden
                data = multi_dense_layers(
                    depth=depth, n_layers=count, **kwargs)(data)
            return generic_dense_layer(beam_edge_dim, **kwargs)(data)
        else:
            raise ValueError(f"Unknown edges: {edge_set_name}")
    return set_initial_edge_state

def get_initial_node_state_func(
        node_dim=None,
        pfo_hidden=None,
        **kwargs):
    if node_dim is None:
        raise TypeError("node_dim must be set")
    def set_initial_node_state(node_set, node_set_name):
        if node_set_name == "beam":
            return tfgnn.keras.layers.MakeEmptyFeature()(node_set)
        elif node_set_name == "pfo":
            data = node_set[tfgnn.HIDDEN_STATE]
            if pfo_hidden is not None:
                (depth, count) = pfo_hidden
                data = multi_dense_layers(
                    depth=depth, n_layers=count, **kwargs)(data)
            return generic_dense_layer(node_dim, **kwargs)(data)
        else:
            raise ValueError(f"Unknown nodes: {node_set_name}")
    return set_initial_node_state

def map_to_initial_state_layer(
        node_dim=None,
        edge_dim=None,
        beam_edge_dim=None,
        pfo_hidden=None,
        neighbours_hidden=None,
        beam_connections_hidden=None,
        **kwargs):
    if node_dim is None:
        raise TypeError("node_dim must be set")
    if edge_dim is None:
        raise TypeError("edge_dim must be set")
    if beam_edge_dim is None:
        raise TypeError("beam_edge_dim must be set")
    kwargs.update({"node_dim": node_dim,
                   "edge_dim": edge_dim,
                   "beam_edge_dim": beam_edge_dim,
                   "pfo_hidden": pfo_hidden,
                   "neighbours_hidden": neighbours_hidden,
                   "beam_connections_hidden": beam_connections_hidden})
    return tfgnn.keras.layers.MapFeatures(
        node_sets_fn = get_initial_node_state_func(**kwargs),
        edge_sets_fn = get_initial_edge_state_func(**kwargs))

# def node_classifer_and_readout(
#         hidden=None,
#         which_feature=tfgnn.HIDDEN_STATE,
#         which_nodes="pfo",
#         n_outputs=1,
#         **kwargs):
#     def classifier_layer(node_set, node_set_name):
#       if node_set_name == which_nodes:
#         data = node_set[which_feature]
#         if hidden is not None:
#             (depth, count) = hidden
#             data = multi_dense_layers(
#                 depth=depth, n_layers=count, **kwargs)(data)
#         # No args, so no dropout etc. on classifer layer
#         return generic_dense_layer(n_outputs)(data)
#       return None
#     classifier = tfgnn.keras.layers.MapFeatures(node_sets_fn=classifier_layer)
#     def readout(graph):
#       updated = classifier(graph)
#       return tf.RaggedTensor.from_row_lengths(
#          values=updated.node_sets[which_nodes].features[tfgnn.HIDDEN_STATE],
#          row_lengths=updated.node_sets[which_nodes].sizes
#         ).with_row_splits_dtype(tf.int64)
#     return readout

# def edge_classifer_and_readout(
#         hidden=None,
#         which_feature=tfgnn.HIDDEN_STATE,
#         which_edges="neighbours",
#         n_outputs=1,
#         **kwargs):
#     def classifier_layer(edge_set, edge_set_name):
#       if edge_set_name == which_edges:
#         data = edge_set[which_feature]
#         if hidden is not None:
#             (depth, count) = hidden
#             data = multi_dense_layers(
#                 depth=depth, n_layers=count, **kwargs)(data)
#         # No args, so no dropout etc. on classifer layer
#         return generic_dense_layer(n_outputs)(data)
#       return None
#     classifier = tfgnn.keras.layers.MapFeatures(edge_sets_fn=classifier_layer)
#     def readout(graph):
#       updated = classifier(graph)
#       return tf.RaggedTensor.from_row_lengths(
#          values=updated.node_sets[which_nodes].features[tfgnn.HIDDEN_STATE],
#          row_lengths=updated.node_sets[which_nodes].sizes
#         ).with_row_splits_dtype(tf.int64)
#     return readout

def classifer_and_readout(
        map_func_kwarg,
        hidden=None,
        which_feature=tfgnn.HIDDEN_STATE,
        which_data=None,
        n_outputs=1,
        **kwargs):
    def classifier_layer(data_set, data_set_name):
        if data_set_name == which_data:
            data = data_set[which_feature]
            if hidden is not None:
                (depth, count) = hidden
                data = multi_dense_layers(
                    depth=depth, n_layers=count, **kwargs)(data)
            # No args, so no dropout etc. on classifer layer
            return generic_dense_layer(n_outputs)(data)
        return None
    map_kwargs = {map_func_kwarg: classifier_layer}
    classifier = tfgnn.keras.layers.MapFeatures(**map_kwargs)
    def readout(graph):
        updated = classifier(graph)
        return tf.RaggedTensor.from_row_lengths(
                values=updated.node_sets[
                    which_nodes].features[tfgnn.HIDDEN_STATE],
                row_lengths=updated.node_sets[which_nodes].sizes
            ).with_row_splits_dtype(tf.int64)
    return readout

def convolution_message_passer():
    return tfgnn.keras.layers.SimpleConv(
        sender_edge_feature=tfgnn.HIDDEN_STATE,
        message_fn=dense(beam_message_dim),
        reduce_type="sum",
        receiver_tag=tfgnn.SOURCE)

def gatv2_message_passer():
    return tfgnn.models.gat_v2.GATv2Conv(
        num_heads = beam_message_heads,
        per_head_channels = beam_message_channels,
        receiver_tag = tfgnn.SOURCE,
        receiver_feature = tfgnn.HIDDEN_STATE,
        sender_node_feature = tfgnn.HIDDEN_STATE,
        sender_edge_feature = tfgnn.HIDDEN_STATE,
        use_bias = False,
        edge_dropout = dropout_rate,
        attention_activation = 'leaky_relu',
        heads_merge_type = 'concat',
        activation = 'relu')

# def beam_collection_layer(
#         message_dimension,
#         final_dimension,
#         dropout=0.1,
#         regulariser=8e-5):
#     """Update the beam node with beam connection PFO data."""
#     beam_collection_update = tfgnn.keras.layers.GraphUpdate(
#         node_sets={
#             "beam": tfgnn.keras.layers.NodeSetUpdate(
#                 {"beam_connections": tfgnn.keras.layers.SimpleConv(
#                     sender_edge_feature=tfgnn.HIDDEN_STATE,
#                     message_fn=generic_dense_layer(
#                         message_dimension,
#                         dropout=dropout, regulariser=regulariser),
#                     reduce_type="sum",
#                     receiver_tag=tfgnn.SOURCE)},
#                 tfgnn.keras.layers.NextStateFromConcat(
#                     generic_dense_layer(
#                         final_dimension,
#                         dropout=dropout, regulariser=regulariser))
#             )
#         })
#     return beam_collection_update

# def pfo_update_layer(
#         message_dimension,
#         final_dimension,
#         dropout=0.1,
#         regulariser=8e-5):
#     """
#     Update to PFOs with neighbouring PFO data.
    
#     This step does the following:
#     - For each neighour edge + connected PFO: concatenate the hidden
#         states of this node, the neighbour edge, then connected PFO.
#     - Perform a dense layer with `message_dimension` outputs over this
#         concatenated set of data.
#     - Pool (sum) all the values calculated above for every connected
#         edge/PFO pair.
#     - Concatenate this pooled data with this node's hidden state.
#     - Perform a dense layer with output size `final_dimensions` over
#         this concatenated data.

#     Parameters
#     ----------
#     message_dimension : int
#         Dimensions of the vector produced by the dense layer acting
#         over the PFo/edge covolutions.
#     final_dimension : int
#         Dimensions of the final output of the layer.
#     dropout : float
#         Probablitity that a given weight will not get used in the
#         training.
#     regulariser : float
#         Penalisation term for having large magnitude weights (L2
#         regularisation).
    
#     Returns
#     -------
#     tfgnn.keras.layers.GraphUpdate
#         Graph update layer.
#     """
#     pfo_node_update = tfgnn.keras.layers.GraphUpdate(
#         node_sets={
#             "pfo": tfgnn.keras.layers.NodeSetUpdate(
#                 {"neighbours": tfgnn.keras.layers.SimpleConv(
#                     sender_edge_feature=tfgnn.HIDDEN_STATE,
#                     message_fn=generic_dense_layer(
#                         message_dimension,
#                         dropout=dropout, regulariser=regulariser),
#                     reduce_type="sum",
#                     receiver_tag=tfgnn.TARGET)},
#                 tfgnn.keras.layers.NextStateFromConcat(
#                     generic_dense_layer(
#                         final_dimension,
#                         dropout=dropout, regulariser=regulariser))
#             )
#         })
#     return pfo_node_update

# def _neighbour_update_layer(
#         final_dims,
#         dropout=0.1,
#         regulariser=8e-5):
#     """Update neighbour edges using connected PFO data"""
#     neighbours_edge_update = tfgnn.keras.layers.GraphUpdate(
#         edge_sets={
#             "neighbours": tfgnn.keras.layers.EdgeSetUpdate(
#                tfgnn.keras.layers.NextStateFromConcat(generic_dense_layer(final_dims))
#             )})
#     return neighbours_edge_update

# def _beam_conn_update_layer():
#     beam_conn_edge_update = tfgnn.keras.layers.GraphUpdate(
#         edge_sets={
#             "beam_connections": tfgnn.keras.layers.EdgeSetUpdate(
#                 tfgnn.keras.layers.NextStateFromConcat(dense(beam_edge_next_dim))
#             )})
#     return beam_conn_edge_update
