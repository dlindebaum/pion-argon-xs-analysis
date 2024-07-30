# Created 19/01/24
# Dennis Lindebaum
# GNN models

import os
import pickle
import warnings
import copy
import numpy as np
import awkward as ak
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.graph import pool_ops
import matplotlib.pyplot as plt
from python.gnn import DataPreparation

__version__ = "1.2.0"

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

class StdGraphPieceReducer(pool_ops.GraphPieceReducer):
    """
    Implements standard deviation pooling from one graph piece
    Will return sqrt(eps) if no inputs
    """

    def __init__(self, *args, eps=1e-5, **kwargs):
        self.eps = eps
        super().__init__(*args, **kwargs)

    def unsorted_segment_op(self,
                            values,
                            segment_ids,
                            num_segments):
        """Implements subclass API."""
        squares = tf.math.square(values)
        x2_mean = tf.math.unsorted_segment_mean(squares, segment_ids, num_segments)
        x_mean2 = tf.math.square(tf.math.unsorted_segment_mean(values, segment_ids, num_segments))
        var = tf.math.subtract(x2_mean, x_mean2)
        relu_var = (tf.math.abs(var) + var) / 2 # eqution for ReLU activation
        # From PNA paper [arXiv:2004.05718], use relu to avoid -ve from
        # numerical errors and add an epsilon for differentiability
        # print(np.min(relu_var))
        return tf.math.sqrt(relu_var + self.eps)

class PrincipleNeighbourAggregator(tfgnn.keras.layers.AnyToAnyConvolutionBase):
    def __init__(
            self,
            units,
            activation="relu", dropout_rate=0., regularisation=None,
            receiver_tag = None,
            receiver_feature = tfgnn.HIDDEN_STATE,
            sender_node_feature = tfgnn.HIDDEN_STATE,
            sender_edge_feature = None,
            aggregators = ["mean", "min", "max", "std"],
            # scalar_alphas = [0., 1., -1.],
            **kwargs):
        # Min an max treated differently to ensure non-inifite results
        #   when no neighbours:
        # Use the pool_to_receiver with reduce_type=f"{min/max}_no_inf"
        #   to return zero for no inputs.
        self.aggregator_defs = {
            "mean": pool_ops.MeanGraphPieceReducer().reduce,
            "std" : StdGraphPieceReducer().reduce}
        self.inbuilt_aggs = ["min", "max"]
        self.aggregators = aggregators
        self.message_units = units
        # self.scalar_alphas = scalar_alphas

        super().__init__(
            receiver_tag = receiver_tag,
            receiver_feature = receiver_feature,
            sender_node_feature = sender_node_feature,
            sender_edge_feature = sender_edge_feature,
            extra_receiver_ops = {agg: self.aggregator_defs[agg]
                                  for agg in self.aggregators
                                  if agg not in self.inbuilt_aggs},
            **kwargs)
        # generic_dense_layer defined in this module
        self._message_fn = generic_dense_layer(
            units, activation=activation,
            dropout_rate=dropout_rate, regularisation=regularisation)

    def scalar(self, counts, alpha):
        """
        Implements calculation of a scaler for each neighbourhood
        based on the scalar calculation method suggested by Corso et al.
        in [arXiv:2004.05718].

        For a neighbourhood of size d, calculate the scalar S as:
        S(d) = (log(d + 1)/delta)^alpha
        With delta as the E[log(d+1)] over the training set.
        """
        log_counts = tf.math.log(counts + 1)

        # Not sure what dimension this should be for edge cases,
        # expect only 1 input dim.
        average_count = tf.math.reduce_mean(log_counts, axis=0)
        scalars = tf.math.pow(log_counts/average_count, alpha)
        # Expand dims to allow broadcasting onto featureful final dimension
        return tf.expand_dims(scalars, axis=-1)

    def get_config(self):
        return dict(
            units=self.message_units,
            aggregators=self.aggregators,
            # scalar_alphas=self.scalar_alphas,
            **super().get_config())

    def convolve(
            # Call arguements from AnyToAnyConvolutionBase parent
            self, *,
            sender_node_input, sender_edge_input, receiver_input,
            broadcast_from_sender_node, broadcast_from_receiver, 
            pool_to_receiver,
            extra_receiver_ops,
            training):
        inputs = []
        if sender_node_input is not None:
            inputs.append(broadcast_from_sender_node(sender_node_input))
        if sender_edge_input is not None:
            inputs.append(sender_edge_input)
        if receiver_input is not None:
            inputs.append(broadcast_from_receiver(receiver_input))
        messages = self._message_fn(tf.concat(inputs, axis=-1))
        aggrs = []
        for agg in self.aggregators:
            if agg in self.inbuilt_aggs:
                # Use inbuilt method for min/max to avoid infinite
                #  values when no inputs present
                aggrs.append(pool_to_receiver(messages, reduce_type=f"{agg}_no_inf"))
            else:
                # With not inputs: mean is 0, std is sqrt(1e-5) (default)
                aggrs.append(extra_receiver_ops[agg](messages))
        return tf.concat(aggrs, axis=-1)
        # counts = pool_to_receiver(messages, reduce_type="_count")
        # scalars = [self.scalar(counts, a) for a in self.scalar_alphas]
        # # tfgnn automatically adds in the initial state when using 
        # #   tfgnn.NextStateFromConcat
        # return tf.concat([aggrs * s for s in scalars], axis=-1)
        

class LayerConstructor():
    def __init__(self, *output_name, final_step=True, **kwargs):
        self.do_final_loop_step = final_step
        if len(output_name) == 0:
            self.output_name=None
        elif len(output_name) == 1:
            self.output_name = output_name[0]
        else:
            raise TypeError(
                f"{type(self).__name__} takes at most 1 positional argument")
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
            repr_string += f"{repr(self.output_name)}, "
        if not self.do_final_loop_step:
            repr_string += "final_step=False, "
        for key, val in self.repr_kwargs.items():
            repr_string += f"{key}={self._kwarg_repr(val)}, "
        if repr_string[-2:] == ", ":
            repr_string = repr_string[:-2]
        repr_string += ")"
        return repr_string

    def _kwarg_repr(self, val):
        return repr(val)

    def get_func(self, parameters):
        params = parameters.copy()
        params.update(self.additional_args)
        return self._func(**params)

    def _func(self, **kwargs):
        pass

class LoopConstructor():
    def __init__(self, loop_constructor, loops=1, final_step=True):
        self.do_final_loop_step = final_step
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

class NormaliseHiddenFeatures(LayerConstructor):
    def __init__(
            self, *output_name,
            pfo_mean=None, pfo_std=None,
            neighbours_mean=None, neighbours_std=None,
            beam_connections_mean=None, beam_connections_std=None,
            **kwargs):
        kwargs.update({"pfo_mean": pfo_mean,
                       "pfo_std": pfo_std,
                       "neighbours_mean": neighbours_mean,
                       "neighbours_std": neighbours_std,
                       "beam_connections_mean": beam_connections_mean,
                       "beam_connections_std": beam_connections_std})
        required_args = ["pfo_mean", "pfo_std",
                         "neighbours_mean", "neighbours_std"]
        for req in required_args:
            if kwargs[req] is None:
                raise ValueError(f"{req} must be supplied for normalisation")
        if kwargs["beam_connections_mean"] is None:
            warnings.warn("Beam connection normalisation not applied, "
                          "only valid for graphs without a beam node")
        super().__init__(*output_name, **kwargs)
        return

    def _kwarg_repr(self, val):
        if isinstance(val, np.ndarray):
            # Add np. to get np.array
            res = "np." + repr(val)
            # Get rid of extra \n as most excess spaces
            res = res.replace(r"\n", "")
            res = [s.strip() for s in res.split(",")]
            res = ", ".join(res)
            # Add np. to dtype (for i.e. np.float32)
            res = "dtype=np.".join(res.split("dtype="))
        else:
            res = repr(val)
        return res

    def _func(self, **kwargs):
        def pfo_normaliser(node_set, node_set_name):
            feats = node_set.get_features_dict()
            if node_set_name == "pfo":
                data = feats[tfgnn.HIDDEN_STATE]
                data = ((data - kwargs["pfo_mean"])
                        / kwargs["pfo_std"])
                feats[tfgnn.HIDDEN_STATE] = data
            return feats
        def edge_normaliser(edge_set, edge_set_name):
            feats = edge_set.get_features_dict()
            if edge_set_name == "neighbours":
                data = feats[tfgnn.HIDDEN_STATE]
                data = ((data - kwargs["neighbours_mean"])
                        / kwargs["neighbours_std"])
                feats[tfgnn.HIDDEN_STATE] = data
            elif edge_set_name == "beam_connections":
                # This is only reached if beam connections exits
                data = feats[tfgnn.HIDDEN_STATE]
                data = ((data - kwargs["beam_connections_mean"])
                        / kwargs["beam_connections_std"])
                feats[tfgnn.HIDDEN_STATE] = data
            return feats
        return tfgnn.keras.layers.MapFeatures(
            node_sets_fn = pfo_normaliser,
            edge_sets_fn = edge_normaliser)

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
        return
    
    def _func(self, **kwargs):
        return tfgnn.keras.layers.Readout(node_set_name=kwargs["which_nodes"])

class ReadoutClassifyNode(LayerConstructor):
    def __init__(
            self, *output_name,
            n_outputs=1,
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
            "which_feature": tfgnn.HIDDEN_STATE})
        return
    
    def _func(self, **kwargs):
        return node_classifer_and_readout(**kwargs)

class ReadoutEdge(LayerConstructor):
    def __init__(
            self, *output_name,
            which_edges="neighbours",
            **kwargs):
        kwargs.update({"which_edges":which_edges})
        super().__init__(*output_name, **kwargs)
        return
    
    def _func(self, **kwargs):
        return tfgnn.keras.layers.Readout(edge_set_name=kwargs["which_edges"])

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
            "which_feature": tfgnn.HIDDEN_STATE})
        return
    
    def _func(self, **kwargs):
        return edge_classifer_and_readout(**kwargs)

class NodeUpdate(LayerConstructor):
    def __init__(
            self, *output_name,
            message_type="GATv2",
            next_state="residual",
            **kwargs):
        if not hasattr(self, "kwarg_dict"):
            raise NotImplementedError(
                "Class should have a kwarg_dict indicating which "
                + "kwargs to pick out for dimensions. This is an "
                + "improperly initialised class.")
        known_messages = ["conv", "gatv2", "pna"]
        if message_type.lower() not in known_messages:
            raise ValueError(f"Unknown message type: {message_type}. "
                             + f"Must be one of {known_messages}")
        self.message_type = message_type.lower()
        known_next_states = ["residual", "concat"]
        if next_state.lower() not in known_next_states:
            raise ValueError(f"Unknown next state: {next_state}. "
                             + f"Must be one of {known_next_states}")
        kwargs.update({"next_state": next_state,
                       "message_type": message_type})
        super().__init__(*output_name, **kwargs)
        self._remove_default_kwargs_from_repr({
            "next_state": "residual"})
        return
    
    def _func(self, **kwargs):
        final_dim_key = self.kwarg_dict["final_dim"]
        # No residual state for PNA
        if kwargs["next_state"] == "concat" or self.message_type == "pna":
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
        elif self.message_type == "pna":
            # PNA tpye doesn't use a standard residual type block,
            #   instead include the initial state in the features sent
            #   to the final layer
            # Currently using the "residual like" by default
            #   (as in the PNA paper)
            message_func = pna_message(
                kwargs[self.kwarg_dict["conv_dim"]],
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
            "conv_dim"      : "beam_message_dim",
            "gatv2_heads"   : "beam_message_heads",
            "gatv2_channels": "beam_message_channels",
            "node"          : "beam", 
            "edges"         : "beam_connections"}
        return super().__init__(*args, **kwargs)

class PFOUpdate(NodeUpdate):
    def __init__(self, *args, **kwargs):
        self.kwarg_dict = {
            "final_dim"     : "node_dim",
            "conv_dim"      : "message_dim",
            "gatv2_heads"   : "message_heads",
            "gatv2_channels": "message_channels",
            "node"          : "pfo", 
            "edges"         : "neighbours"}
        return super().__init__(*args, **kwargs)

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
        return super().__init__(*args, **kwargs)

class BeamConnectionUpdate(EdgeUpdate):
    def __init__(self, *args, **kwargs):
        self.kwarg_dict = {
            "final_dim": "beam_edge_dim",
            "edge"     : "beam_connections"}
        return super().__init__(*args, **kwargs)

class Dense(LayerConstructor):
    def __init__(
            self, *output_name,
            depth=None,
            n_layers=1,
            dropout_rate=0.0,
            **kwargs):
        if depth is None:
            raise TypeError("depth kwarg must be specified.")
        kwargs.update({"depth": depth,
                       "n_layers": n_layers,
                       "dropout_rate": dropout_rate})
        super().__init__(*output_name, **kwargs)
        self._remove_default_kwargs_from_repr({
            "dropout_rate": 0.0})
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
        return dense_core(number, activation=activation, regularisation=regularisation, **kwargs)
    return tf.keras.Sequential([
        dense_core(number, activation=activation, regularisation=regularisation, **kwargs),
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
        dropout=0.0,
        regularisation=None,
        **kwargs):
    if regularisation is not None:
        regulariser = tf.keras.regularizers.l2(regularisation)
    else:
        regulariser = None
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
        kernel_initializer = None,
        kernel_regularizer = regulariser)

def pna_message(
        units,
        aggregators = ["mean", "min", "max", "std"],
        activation = "relu",
        dropout_rate = 0.0,
        regularisation = None,
        **kwargs):
    # # if "dropout_rate" in kwargs.keys():
    # #     dropout = kwargs["dropout_rate"]
    # # else:
    # #     dropout = 0.0
    # # regulariser = None
    # # if "regularisation" in kwargs.keys():
    # #     if kwargs["regularisation"] is not None:
    # #         regulariser = tf.keras.regularizers.l2(kwargs["regularisation"])
    # # Can't keep excess arugments here
    # pna_kwargs = {
    # #     "dropout": dropout,
    # #     "regulariser": regulariser,
    #     "aggregators": ["mean", "min", "max", "std"]}
    # keep_args = ["activation", "aggregators",
    #              "regularisation", "dropout_rate"]
    # for arg in keep_args:
    #     if arg in kwargs.keys():
    #         pna_kwargs.update({arg: kwargs[arg]})
    return PrincipleNeighbourAggregator(
        units,
        receiver_tag = tfgnn.SOURCE,
        receiver_feature = tfgnn.HIDDEN_STATE,
        sender_node_feature = tfgnn.HIDDEN_STATE,
        sender_edge_feature = tfgnn.HIDDEN_STATE,
        aggregators = aggregators,
        activation = activation,
        dropout_rate = dropout_rate,
        regularisation = regularisation)

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

def node_classifer_and_readout(
        hidden=None,
        which_feature=tfgnn.HIDDEN_STATE,
        which_nodes="pfo",
        n_outputs=1,
        **kwargs):
    def classifier_layer(node_set, node_set_name):
      if node_set_name == which_nodes:
        data = node_set[which_feature]
        if hidden is not None:
            (depth, count) = hidden,
            data = multi_dense_layers(
                depth=depth, n_layers=count, **kwargs)(data)
        # No args, so no dropout etc. on classifer layer
        return generic_dense_layer(n_outputs)(data)
      return None
    classifier = tfgnn.keras.layers.MapFeatures(node_sets_fn=classifier_layer)
    def readout(graph):
      updated = classifier(graph)
      return tf.RaggedTensor.from_row_lengths(
         values=updated.node_sets[which_nodes].features[tfgnn.HIDDEN_STATE],
         row_lengths=updated.node_sets[which_nodes].sizes
        ).with_row_splits_dtype(tf.int64)
    return readout

def edge_classifer_and_readout(
        hidden=None,
        which_feature=tfgnn.HIDDEN_STATE,
        which_edges="neighbours",
        n_outputs=1,
        **kwargs):
    def classifier_layer(edge_set, edge_set_name):
      if edge_set_name == which_edges:
        data = edge_set[which_feature]
        if hidden is not None:
            (depth, count) = hidden
            data = multi_dense_layers(
                depth=depth, n_layers=count, **kwargs)(data)
        # No args, so no dropout etc. on classifer layer
        return generic_dense_layer(n_outputs)(data)
      return None
    classifier = tfgnn.keras.layers.MapFeatures(edge_sets_fn=classifier_layer)
    def readout(graph):
      updated = classifier(graph)
      return tf.RaggedTensor.from_row_lengths(
         values=updated.edge_sets[which_edges].features[tfgnn.HIDDEN_STATE],
         row_lengths=updated.edge_sets[which_edges].sizes
        ).with_row_splits_dtype(tf.int64)
    return readout

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
