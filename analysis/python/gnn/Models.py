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
from matplotlib.colors import LogNorm
from python.gnn import DataPreparation
# from python.gnn.model_plots import *
from apps.cex_toy_parameters import PlotCorrelationMatrix as plot_confusion_matrix
# from python.analysis import Plots

# =====================================================================
#                                Storage

def _get_dict_repr(dict):
    """Returns the dict with string representations of values"""
    return {key: repr(val) for  key, val in dict.items()}

def _convert_losses_to_config(dict):
    """Calls `.get_config()` on loss/metric type params in the dict"""
    new_dict = copy.copy(dict)
    keys = ["metrics"]#, "callbacks"]
    loss = new_dict["loss"]
    if isinstance(loss, list):
        new_dict["loss"] = [ l.get_config() for l in loss ]
    else:
        new_dict["loss"] = new_dict["loss"].get_config()
    def get_vals_func(item):
        try:
            return item.name
        except:
            return [get_vals_func(i) for i in item]
    for k in keys:
        new_dict[k] = get_vals_func(new_dict[k])
        # try:
        #     new_dict[k] = [val.name for val in new_dict[k]]
        # except:
        #     new_dict[k] = new_dict[k].name
    new_dict["callbacks"] = repr(new_dict["callbacks"])
    return new_dict

def _convert_config_to_losses(dict):
    """Gets loss/metric instances from stored config params"""
    dict["loss"] = tf.keras.losses.Loss.from_config(dict["loss"])
    # dict["metrics"] = _get_one_config(tf.keras.losses.Loss,
    #                                   dict["metrics"])
    # dict["callbacks"] = _get_one_config(tf.keras.losses.Loss,
    #                                     dict["callbacks"])
    return dict

def make_model_paths(folder_path):
    """
    Create a dictionary containing the default paths for model storage
    given the supplied folder.

    Dictionary contains `"folder_path"`, `"params_path"`,
    `"text_params_path"`, and `"model_path"`.

    Parameters
    ----------
    folder_path : str
        Path to main folder storing the data.

    Returns
    -------
    dict
        Dictionary containing default paths to model data.
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return {"folder_path": folder_path,
            "params_path": os.path.join(folder_path, "hyper_params.pkl"),
            "text_params_path": os.path.join(folder_path,
                                             "hyper_params_repr.txt"),
            "model_path": os.path.join(folder_path, "model.tf"),
            "history_path": os.path.join(folder_path, "train_history.pkl")}

def generate_hyper_params(
        folder_path,
        loss,
        metrics,
        callbacks,
        learning_rate,
        n_epochs,
        steps_per_epoch,
        extra_losses=None,
        loss_weights=None,
        class_weights=None,
        data_folder=None,
        training_batch=None,
        training_shuffle=None
    ):
    """
    Creates a dictionary storing the training details of a model.

    Parameters
    ----------
    folder_path : str
        Path to folder in which to save the model.
    loss : tf.keras.losses.Loss
        Loss function used in training.
    metrics : tf.keras.metrics.Metric
        Metrics used in training evaluation.
    callbacks : tf.keras.callbacks.Callback
        Callbacks active during the training process.
    learning_rate : float
        Learning rate the model is complied at.
    n_epochs : int
        Maximum number of epochs trained for.
    steps_per_epoch : int
        Number of batches to train on per epoch.
    extra_losses : list, optional
        List containing extra loss information that should be included
        in the losses list. Requires the input graph to have extra
        `truth_info` selected. See examples underneath for available
        extra losses. Default is None.
    data_folder : str, optional
        Folder containing the data records used to load the training
        data. Default is None.
    training_batch : int, optional
        Number of events per batch. Default is None.
    training_shuffle : int, optional
        Shuffle this many parameters from the dataset before selecting
        each batch. Default is None.

    Returns
    -------
    dict:
        Dictionary containing the training hyper-parameters.
    
    Examples
    --------
    Avaiable losses come from labels in the network used by the extra
    `truth_info`.
    
    Context available information is:
    [`mc_pions`, `mc_photons`, `mc_pi0s`, `bt_pions`, `bt_photons`,
    `bt_pi0s`]
    Note bt_... properties are only available for data type graphs.
    
    PFO available information is:
    [`beam_daughter`, `beam_granddaughter`, `pi0_granddaughter`,
    `beam_related`, `beam_relevant`, `pion`, `photon`, `beam_pion`,
    `beam_photon`, `pi0`, `beam_pi0`]
    Note the final two properties are only available for MC type data.

    Neighbour connection available information is:
    [`true_pi0`, `beam_pi0`]

    Beam connections available information is:
    [`true_daughter`, `true_granddaughter`, `pi0_granddaughter`,
    `beam_related`, `beam_relevant`]
    """
    hyper_params = {
        "loss": loss,
        "metrics": metrics,
        "callbacks": callbacks,
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "steps_per_epoch": steps_per_epoch,
        "extra_losses": extra_losses,
        "loss_weights": loss_weights,
        "class_weights": class_weights,
        "data_folder": data_folder,
        "training_batch": training_batch,
        "training_shuffle": training_shuffle}
    hyper_params.update(make_model_paths(folder_path))
    save_params = _convert_losses_to_config(hyper_params)
    with open(hyper_params["params_path"], 'wb') as f:
        pickle.dump(save_params, f)
    val_repr = _get_dict_repr(hyper_params)
    with open(hyper_params["text_params_path"], 'w') as f:
        f.write(repr(val_repr).replace(", ", ",\n "))
    return hyper_params

def load_hyper_params(paths_dict):
    with open(paths_dict["params_path"], 'rb') as f:
        params = pickle.load(f)
    return params

# =====================================================================
#                            Data formatting

def format_data(train, val, batch_size=32, shuffle_size=128):
    if shuffle_size is not None:
        train = train.shuffle(buffer_size=shuffle_size)
    train_batched = train.ragged_batch(batch_size=batch_size).repeat()
    val_batched = val.ragged_batch(batch_size=batch_size)
    return train_batched, val_batched

def load_data_from_hyper_params(hyper_params):
    if ((hyper_params["data_folder"] is None)
        or (hyper_params["training_batch"] is None)):
        raise ValueError("Hyper-parameters do not contain data details")
    data_paths = DataPreparation.create_filepath_dictionary(
        hyper_params["data_folder"])
    train_ds, val_ds = DataPreparation.load_record(
        data_paths["schema_path"],
        [data_paths["train_path"], data_paths["val_path"]],
        extra_losses=hyper_params["extra_losses"])
    return format_data(train_ds, val_ds,
                       hyper_params["training_batch"],
                       hyper_params["training_shuffle"])

def get_spec_from_hyper_params(hyper_params):
    if ((hyper_params["data_folder"] is None)
        or (hyper_params["training_batch"] is None)):
        raise ValueError("Hyper-parameters do not contain data details")
    data_paths = DataPreparation.create_filepath_dictionary(
        hyper_params["data_folder"])
    train_ds = DataPreparation.load_record(
        data_paths["schema_path"],
        data_paths["train_path"])
    return train_ds.element_spec[0]


# =====================================================================
#                                Losses

class CategoricalCrossentropyUnclassBin(
        tf.keras.losses.CategoricalCrossentropy):
    def __init__(
            self,
            unclass_weight = 0.5,
            name='categorical_crossentropy_unclass_bin',
            **kwargs):
        self.unclass_weight=unclass_weight
        return super().__init__(name = name, **kwargs)
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        # return super().__call__(y_true, y_pred[..., :-1], sample_weight=sample_weight)
        # print(y_true)
        # print(y_pred)
        # unclass_pred = y_pred[..., -1:]*self.unclass_weight
        unclass_pred = self.unclass_weight
        pred_weighting = (y_true * unclass_pred) + ((1-y_true) * (1-unclass_pred)/(y_true.shape[-1] - 1))
        # print(pred_weighting)
        # the_loss = super().__call__(y_true, y_pred[..., :-1] + pred_weighting * y_pred[..., -1:], sample_weight=sample_weight)
        # print(the_loss)
        # return the_loss
        # print(pred_weighting.shape)
        # print(y_pred[..., :-1].shape)
        # print((y_pred[..., :-1]*pred_weighting).shape)
        return super().__call__(y_true, y_pred[..., :-1] + pred_weighting * y_pred[..., -1:], sample_weight=sample_weight)
    
class CategoricalCrossentropyUnclassWeight(
        tf.keras.losses.CategoricalCrossentropy):
    def __init__(
            self,
            unclass_weight = 0.5,
            name='categorical_crossentropy_unclass_weight',
            **kwargs):
        self.unclass_weight=unclass_weight
        return super().__init__(name = name, **kwargs)
    
    def __call__(self, y_true, y_pred):
        uncless_val = y_pred[..., -1:]
        weighting = 2 - (uncless_val ** 2)
        return super().__call__(y_true, y_pred[..., :-1], sample_weight=weighting)


# =====================================================================
#                               Model creation

def _generic_dense_layer(number, activation="relu", dropout=0.1, regulariser=8e-5):
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

def _beam_collection_layer(
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
                    message_fn=_generic_dense_layer(
                        message_dimension,
                        dropout=dropout, regulariser=regulariser),
                    reduce_type="sum",
                    receiver_tag=tfgnn.SOURCE)},
                tfgnn.keras.layers.NextStateFromConcat(
                    _generic_dense_layer(
                        final_dimension,
                        dropout=dropout, regulariser=regulariser))
            )
        })
    return beam_collection_update

def _pfo_update_layer(
        message_dimension,
        final_dimension,
        dropout=0.1,
        regulariser=8e-5):
    """Update to PFOs with neighbouring PFO data"""
    pfo_node_update = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "pfo": tfgnn.keras.layers.NodeSetUpdate(
                {"neighbours": tfgnn.keras.layers.SimpleConv(
                    sender_edge_feature=tfgnn.HIDDEN_STATE,
                    message_fn=_generic_dense_layer(
                        message_dimension,
                        dropout=dropout, regulariser=regulariser),
                    reduce_type="sum",
                    receiver_tag=tfgnn.TARGET)},
                tfgnn.keras.layers.NextStateFromConcat(
                    _generic_dense_layer(
                        final_dimension,
                        dropout=dropout, regulariser=regulariser))
            )
        })
    return pfo_node_update

def _neighbour_update_layer():
    neighbours_edge_update = tfgnn.keras.layers.GraphUpdate(
        edge_sets={
            "neighbours": tfgnn.keras.layers.EdgeSetUpdate(
               tfgnn.keras.layers.NextStateFromConcat(dense(edge_next_dim))
            )})
    return neighbours_edge_update

def _beam_conn_update_layer():
    beam_conn_edge_update = tfgnn.keras.layers.GraphUpdate(
        edge_sets={
            "beam_connections": tfgnn.keras.layers.EdgeSetUpdate(
                tfgnn.keras.layers.NextStateFromConcat(dense(beam_edge_next_dim))
            )})
    return beam_conn_edge_update

def build_evt_classifer_model_data_with_momentum(
    graph_tensor_spec,
    # Dimensions of initial states.
    beam_dim=16,
    node_dim=16,
    mom_dim=18,
    beam_edge_dim=16,
    edge_dim=16,
    # Node setup
    n_node_layers=1,
    n_node_hidden_depth=128,
    # Dimensions for message passing.
    message_dim=64,
    next_state_dim=64,
    beam_message_dim=64,
    beam_next_dim=64,
    edge_next_dim=16,
    beam_edge_next_dim=16,
    # Dimension for the logits.
    final_layers = 2,
    final_layer_nodes = 32,
    num_classes=4,
    # Number of message passing steps.
    num_message_passing=2,
    # Other hyperparameters.
    l2_regularization=8e-5,
    dropout_rate=0.4,
    # # Names of sets to use
    # node_sets_name="pfo",
    # edge_sets_name="neighbours"
):
  # This helper function is just a short-hand for the code below.
  def dense(units, activation="relu"):
    """A Dense layer with regularization (L2 and Dropout)."""
    regularizer = tf.keras.regularizers.l2(l2_regularization)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout_rate)
    ])

  # Model building with Keras's Functional API starts with an input object
  # (a placeholder for the eventual inputs). Here is how it works for
  # GraphTensors:
  input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
  # print(input_graph)
  # IMPORTANT: All TF-GNN modeling code assumes a GraphTensor of shape []
  # in which the graphs of the input batch have been merged to components of
  # one contiguously indexed graph. (There are no edges between components,
  # so no information flows between them.)
  graph = input_graph.merge_batch_to_components()
  
  def make_node_internal(node_input):
    res = node_input
    for _ in range(n_node_layers):
      res = dense(n_node_hidden_depth)(res)
    return tf.keras.layers.Dense(node_dim)(res)
  # Nodes and edges have one-hot encoded input features. Sending them through
  # a Dense layer effectively does a lookup in a trainable embedding table.
  def set_initial_node_state(node_set, node_set_name):
    if node_set_name == "beam":
      return tfgnn.keras.layers.MakeEmptyFeature()(node_set)
    # Since we only have one node set, we can ignore node_set_name.
    # elif node_set_name == "pfo":
    type_vec = make_node_internal(node_set[tfgnn.HIDDEN_STATE])
    mom_vec = tf.keras.layers.Dense(mom_dim)(node_set["momentum"])
    return {tfgnn.HIDDEN_STATE: type_vec, "momentum": mom_vec}
  def set_initial_edge_state(edge_set, edge_set_name):
    if edge_set_name == "beam_connections":
      return tf.keras.layers.Dense(beam_edge_dim)(edge_set[tfgnn.HIDDEN_STATE])
    return tf.keras.layers.Dense(edge_dim)(edge_set[tfgnn.HIDDEN_STATE])
  # print(graph)
  graph = tfgnn.keras.layers.MapFeatures(
      node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(
          graph)

  def beam_collection_layer():
    beam_collection_update = tfgnn.keras.layers.GraphUpdate(
      node_sets={
        "beam": tfgnn.keras.layers.NodeSetUpdate(
          {"beam_connections": tfgnn.keras.layers.SimpleConv(
            sender_edge_feature=tfgnn.HIDDEN_STATE,
            message_fn=dense(beam_message_dim),
            reduce_type="sum",
            receiver_tag=tfgnn.SOURCE)},
          tfgnn.keras.layers.NextStateFromConcat(dense(beam_next_dim)))})
    return beam_collection_update

  def neighbour_update_layer():
    neighbours_edge_update = tfgnn.keras.layers.GraphUpdate(
      edge_sets={
        "neighbours": tfgnn.keras.layers.EdgeSetUpdate(
          tfgnn.keras.layers.NextStateFromConcat(dense(edge_next_dim))
        )})
    return neighbours_edge_update
  
  def neighbour_momentum_update_layer():
    neighbours_edge_update = tfgnn.keras.layers.GraphUpdate(
      edge_sets={
        "neighbours": tfgnn.keras.layers.EdgeSetUpdate(
          tfgnn.keras.layers.NextStateFromConcat(dense(edge_next_dim)),
          node_input_feature = "momentum"
        )})
    return neighbours_edge_update

  def beam_conn_update_layer():
    beam_conn_edge_update = tfgnn.keras.layers.GraphUpdate(
      edge_sets={
        "beam_connections": tfgnn.keras.layers.EdgeSetUpdate(
          tfgnn.keras.layers.NextStateFromConcat(dense(beam_edge_next_dim))
        )})
    return beam_conn_edge_update
  
  def pfo_update_layer():
    pfo_node_update = tfgnn.keras.layers.GraphUpdate(
      node_sets={
        "pfo": tfgnn.keras.layers.NodeSetUpdate(
          {"neighbours": tfgnn.keras.layers.SimpleConv(
            sender_edge_feature=tfgnn.HIDDEN_STATE,
            message_fn=dense(message_dim),
            reduce_type="sum",
            receiver_tag=tfgnn.TARGET)},
          tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)))})
    return pfo_node_update

  def momentum_gather_layer():
    mom_layer = tfgnn.keras.layers.GraphUpdate(
      node_sets={
        "pfo": tfgnn.keras.layers.NodeSetUpdate(
          {"neighbours": tfgnn.keras.layers.SimpleConv(
            sender_edge_feature=tfgnn.HIDDEN_STATE,
            message_fn=dense(message_dim),
            reduce_type="sum",
            receiver_tag=tfgnn.TARGET,
            sender_node_feature="momentum")},
          tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)),
          node_input_feature="momentum")})
    return mom_layer

  graph = beam_collection_layer()(graph)
  
  for i in range(num_message_passing):
    graph = pfo_update_layer()(graph)
    if i < num_message_passing - 1:
      graph = neighbour_momentum_update_layer()(graph)
      graph = neighbour_update_layer()(graph)
    graph = beam_collection_layer()(graph)
    # if i < num_message_passing - 1:
    #   graph = momentum_gather_layer()(graph)
  
  # After the GNN has computed a context-aware representation of the "atoms",
  # the model reads out a representation for the graph as a whole by averaging
  # (pooling) nde states into the graph context. The context is global to each
  # input graph of the batch, so the first dimension of the result corresponds
  # to the batch dimension of the inputs (same as the labels).
  # readout_features = tfgnn.keras.layers.Pool(
  #     tfgnn.CONTEXT, "mean", node_set_name="beam")(graph)
  # readout_features = graph.node_sets["beam"][tfgnn.HIDDEN_STATE]
  readout_features = tfgnn.keras.layers.Readout(node_set_name="beam")(graph)
  # print(readout_features)

  for _ in range(final_layers - 1):
    readout_features = tf.keras.layers.Dense(final_layer_nodes)(readout_features)

  # Put a linear classifier on top (not followed by dropout).
  # Use logits over soft max since its supposed to improve numerical stability
  logits = tf.keras.layers.Dense(num_classes)(readout_features)

  # Build a Keras Model for the transformation from input_graph to logits.
  return tf.keras.Model(inputs=[input_graph], outputs=[logits])

def build_gvn_model(
        graph_tensor_spec,
        # Dimensions of initial states.
        node_dim=16,
        edge_dim=16,
        # Dimensions for message passing.
        message_dim=64,
        next_state_dim=64,
        # Dimension for the logits.
        num_classes=4,
        # Number of message passing steps.
        num_message_passing=2,
        # Other hyperparameters.
        l2_regularization=5e-4,
        dropout_rate=0.5,
        # # Names of sets to use
        # node_sets_name="pfo",
        # edge_sets_name="neighbours"
    ):
    # Model building with Keras's Functional API starts with an input object
    # (a placeholder for the eventual inputs). Here is how it works for
    # GraphTensors:
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)

    # IMPORTANT: All TF-GNN modeling code assumes a GraphTensor of shape []
    # in which the graphs of the input batch have been merged to components of
    # one contiguously indexed graph. (There are no edges between components,
    # so no information flows between them.)
    graph = input_graph.merge_batch_to_components()
    
    """TODO Add more layers here - a few Dence layers to get some sensible encoding"""
    # Nodes and edges have one-hot encoded input features. Sending them through
    # a Dense layer effectively does a lookup in a trainable embedding table.
    def set_initial_node_state(node_set, *, node_set_name):
        # Since we only have one node set, we can ignore node_set_name.
        return tf.keras.layers.Dense(node_dim)(node_set[tfgnn.HIDDEN_STATE])
    def set_initial_edge_state(edge_set, *, edge_set_name):
        return tf.keras.layers.Dense(edge_dim)(edge_set[tfgnn.HIDDEN_STATE])
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(
            graph)

    # This helper function is just a short-hand for the code below.
    def dense(units, activation="relu"):
        """A Dense layer with regularization (L2 and Dropout)."""
        regularizer = tf.keras.regularizers.l2(l2_regularization)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    # The GNN core of the model does `num_message_passing` many updates of node
    # states conditioned on their neighbors and the edges connecting to them.
    # More precisely:
    #  - Each edge computes a message by applying a dense layer `message_fn`
    #    to the concatenation of node states of both endpoints (by default)
    #    and the edge's own unchanging feature embedding.
    #  - Messages are summed up at the common TARGET nodes of edges.
    #  - At each node, a dense layer is applied to the concatenation of the old
    #    node state with the summed edge inputs to compute the new node state.
    # Each iteration of the for-loop creates new Keras Layer objects, so each
    # round of updates gets its own trainable variables.
    for _ in range(num_message_passing):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "pfo": tfgnn.keras.layers.NodeSetUpdate(
                    {"neighbours": tfgnn.keras.layers.SimpleConv(
                        sender_edge_feature=tfgnn.HIDDEN_STATE,
                        message_fn=dense(message_dim),
                        reduce_type="sum",
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)))}
        )(graph)
    
    # After the GNN has computed a context-aware representation of the "atoms",
    # the model reads out a representation for the graph as a whole by averaging
    # (pooling) nde states into the graph context. The context is global to each
    # input graph of the batch, so the first dimension of the result corresponds
    # to the batch dimension of the inputs (same as the labels).
    readout_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="pfo")(graph)

    # Put a linear classifier on top (not followed by dropout).
    logits = tf.keras.layers.Dense(num_classes)(readout_features)

    # This one isn't logits, need to work out exactly what it is though...
    # logits = tf.keras.layers.Dense(num_classes, activation="softmax")(readout_features)

    # Build a Keras Model for the transformation from input_graph to logits.
    return tf.keras.Model(inputs=[input_graph], outputs=[logits])

# =====================================================================
#                            Model training

def compile_and_train(
        model,
        hyper_params,
        batched_train,
        batched_val,
        print_summary=True):
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=hyper_params["learning_rate"]),
        loss=hyper_params["loss"],
        loss_weights = hyper_params["loss_weights"],
        metrics=hyper_params["metrics"])
    if print_summary:
        model.summary()
    history = model.fit(batched_train,
                        class_weight=hyper_params["class_weights"],
                        steps_per_epoch=hyper_params["steps_per_epoch"],
                        epochs=hyper_params["epochs"],
                        callbacks=hyper_params["callbacks"],
                        validation_data=batched_val)
    # This save only saves a model which can do inference, it doesn't
    #   have full (i.e. fitting) functionality
    # model.export(hyper_params["model_path"])
    # Saving not currently working - need to use strings as dftype formatting?
    model.save(hyper_params["model_path"], save_format="tf", overwrite=False)
    # with open(hyper_params["history_path"], "wb") as f:
    #     pickle.dump(history, f)
    return history

# =====================================================================
#                           Model evaluation
#    Plotting functions for evaluation imported from gnn.model_plots

def convert_labels_to_truth(labels):
    size = labels.take(1).get_single_element().shape[0]
    index = np.arange(size)
    if size == 1:
        index += 1
    return np.array([lab @ index for lab in labels.as_numpy_iterator()])

def _parse_classification_labels(labels, pred, truth):
    if labels is None:
        n_indicies = int(max(np.max(pred), np.max(truth)) + 1)
        labels = [f"class index {i}" for i in range(n_indicies)]
    return labels

def create_summary_text(
        pred_index,
        truth_index,
        classification_labels=None,
        print_results=False):
    correct_mask = pred_index == truth_index
    labels = _parse_classification_labels(
        classification_labels, pred_index, truth_index)
    summary_text = f"{correct_mask.size} events searched.\n"
    summary_text += f"{np.sum(correct_mask)} events correctly classified "
    summary_text += f"({np.sum(np.logical_not(correct_mask))} false), "
    summary_text += f"for an efficiency of "
    summary_text += f"{100*np.sum(correct_mask)/correct_mask.size:.2f}%.\n"
    for i in range(len(labels)):
        eff = (100*np.sum(pred_index[truth_index==i] == i)
               / np.sum(truth_index==i))
        purity = (100*np.sum(truth_index[pred_index==i] == i)
                  / np.sum(pred_index==i))
        summary_text += f"For {labels[i]}: "
        summary_text += f"{eff:.2f}% efficiency, {purity:.2f}% purity (product: {eff*purity/100:.2f}%).\n"
    if print_results:
        print(summary_text)
    return summary_text

def plot_summary_information(pred_index, truth_index, plot_config):
    correct_mask = pred_index == truth_index
    n_classes = max(np.max(pred_index), np.max(truth_index))+1
    shared_kwargs = {"type": "hist",
                     "density":False,
                     "bins": np.arange(n_classes+1)}
    plot_config.setup_figure()
    plt.hist(
        pred_index[correct_mask],
        **plot_config.gen_kwargs(label="Correct predicitions", index=0,
                                 **shared_kwargs))
    plt.hist(
        pred_index[np.logical_not(correct_mask)],
        **plot_config.gen_kwargs(label="Failed predictions", index=1,
                                 **shared_kwargs))
    plt.hist(truth_index,
        **plot_config.gen_kwargs(label="Truth distribution", index=2,
                                 **shared_kwargs))
    plot_config.format_axis(xlabel="Classification index", ylabel="Count")
    plot_config.end_plot()

# CAREFUL with the following plotting functions, they have a bunch of
#   stupid special cases which are nicely centrally unified
def get_predicitions(
        model,
        schema_path, test_path,
        pred_index=0, other_truth=None
    ):
    if other_truth is not None:
        other_truth = [other_truth]
    test_record = DataPreparation.load_record(schema_path, test_path,
                                              extra_losses=other_truth)
    test_data = test_record.map(lambda data, label : data)
    test_data = test_data.batch(batch_size=32)
    predictions = model.predict(test_data, batch_size=32)
    if isinstance(predictions, list):
        # First predicition is always the classification
        predictions = predictions[pred_index]
    if other_truth is None:
        test_label = test_record.map(lambda data, label : label)
        truth = convert_labels_to_truth(test_label)
    elif other_truth[0] == "reco_class":
        test_label = test_record.map(lambda data, label : label[-1])
        truth = convert_labels_to_truth(test_label)
    else:
        truth = test_record.map(lambda data, label : label[-1])
        truth = ak.Array([t.numpy() for t in truth])
    return predictions, truth

def _hist_plot(plot_config, sig, bkg, bins=None, label=None, norm=False):
    title = "Normalised" if norm else "Unnormalised"
    y_lab = "Density" if norm else "Count"
    if bins is None:
        bins = plot_config.get_bins(
            np.concatenate((sig_preds, bkg_preds)),
            array=True)
    plot_config.setup_figure(title=title)
    plt.hist(
        sig, **plot_config.gen_kwargs(
            index=0, bins=bins, label=label,
            density=False, histtype="step"))
    plt.hist(
        bkg, **plot_config.gen_kwargs(
            index=1, bins=bins, label="Background",
            density=norm, histtype="step"))
    plot_config.format_axis(xlog=False, ylog=True,
                            xlabel="Prediction", ylabel=y_lab)
    return plot_config.end_plot()

def _sig_bkg_hist_plotter(sig_preds, bkg_preds, sig_label, plot_config):
    joint_data = np.concatenate((sig_preds, bkg_preds))
    d_max = np.max(joint_data)
    d_min = np.min(joint_data)
    print(f"Found predictions ranging between {d_min:.3f} and {d_max:.3f}.")
    if d_min >= 0. and d_max <= 1. and (d_max - d_min)>0.1:
        bins = np.linspace(0, 1, 31)
    else:
        bins = plot_config.get_bins(joint_data, array=True)
    _hist_plot(plot_config, sig_preds, bkg_preds,
               bins=bins, label=sig_label, norm=False)
    _hist_plot(plot_config, sig_preds, bkg_preds,
               bins=bins, label=sig_label, norm=True)
    return

def plot_binary_event_classification_dist(model, path_params, plot_config):
    preds, truths = get_predicitions(
        model, path_params["schema_path"], path_params["test_path"])
    bkg_preds = preds[:, 0][truths==0.]
    sig_preds = preds[:, 0][truths==1.]
    return _sig_bkg_hist_plotter(sig_preds, bkg_preds, "CEx.", plot_config)

def plot_binary_extra_loss_dist(
        model,
        extra_loss, loss_index,
        path_params, plot_config
    ):
    preds, truths = get_predicitions(
        model,
        path_params["schema_path"], path_params["test_path"],
        pred_index=loss_index, other_truth=extra_loss)
    if isinstance(preds, tf.RaggedTensor):
        preds = np.hstack(tf.squeeze(preds, axis=-1))
    else:
        preds = ak.ravel(preds)
    truths = ak.ravel(truths)
    bkg_preds = preds[truths==0.]
    sig_preds = preds[truths==1.]
    return _sig_bkg_hist_plotter(sig_preds, bkg_preds, extra_loss, plot_config)

def plot_regression_extra_loss_dist(
        model,
        extra_loss, loss_index,
        path_params, plot_config,
        logits=False, mask=None
    ):
    preds, truths = get_predicitions(
        model,
        path_params["schema_path"], path_params["test_path"],
        pred_index=loss_index, other_truth=extra_loss)
    if isinstance(preds, tf.RaggedTensor):
        preds = np.hstack(tf.squeeze(preds, axis=-1))
    else:
        preds = ak.ravel(preds)
    truths = ak.ravel(truths)
    if mask is not None:
        preds = preds[mask]
        truths = truths[mask]
    plot_config.setup_figure(title=extra_loss.replace('_', ' ').title())
    pred_bins = plot_config.get_bins(preds, array=True)
    max_true = max(truths)
    true_bins = np.linspace(-0.25, max_true + 0.25, int(max_true)*2 + 2)
    plt.hist2d(
        truths, preds, norm=LogNorm(), bins=(true_bins, pred_bins))
    # plt.scatter(
    #     truths, preds, **plot_config.gen_kwargs(index=0, marker="x"))
    plot_config.format_axis(xlog=False, ylog=False,
                            xlabel="True count", ylabel="Prediction")
    return plot_config.end_plot()

def plot_confusion_extra_loss(
        model,
        extra_loss, loss_index,
        path_params,
        classification_labels=None,
        plot_config=None
    ):
    preds, truth_index = get_predicitions(
        model,
        path_params["schema_path"], path_params["test_path"],
        pred_index=loss_index, other_truth=extra_loss)
    pred_index = np.where(
        preds == np.max(preds, axis=1)[:, np.newaxis])[1]
    labels = _parse_classification_labels(classification_labels,
                                          pred_index, truth_index)
    _ = create_summary_text(
        pred_index, truth_index,
        classification_labels=labels,
        print_results=True)
    confustion_mat = create_confusion_matrix(pred_index, truth_index)
    plot_confusion_matrix(
        confustion_mat,
        labels, labels)
    if plot_config is not None:
        plot_summary_information(pred_index, truth_index, plot_config)
    return pred_index, truth_index

def plot_confusion_main_vs_reco_loss(
        model,
        extra_loss, loss_index,
        path_params,
        classification_labels=None,
        plot_config=None
    ):
    _, truth_index = get_predicitions(
        model,
        path_params["schema_path"], path_params["test_path"],
        pred_index=loss_index, other_truth=extra_loss)
    preds, _ = get_predicitions(
        model,
        path_params["schema_path"], path_params["test_path"])
    pred_index = np.where(
        preds == np.max(preds, axis=1)[:, np.newaxis])[1]
    labels = _parse_classification_labels(classification_labels,
                                          pred_index, truth_index)
    _ = create_summary_text(
        pred_index, truth_index,
        classification_labels=labels,
        print_results=True)
    confustion_mat = create_confusion_matrix(pred_index, truth_index)
    plot_confusion_matrix(
        confustion_mat,
        labels, labels)
    if plot_config is not None:
        plot_summary_information(pred_index, truth_index, plot_config)
    return pred_index, truth_index


def create_confusion_matrix(predictions, truth):
    n_inds = int(1 + max(np.max(truth), np.max(predictions)))
    true = n_inds * truth
    inds, counts = np.unique(true+predictions, return_counts=True)
    inds = inds.astype(int)
    res = np.zeros((n_inds, n_inds), dtype=int)
    res[inds//n_inds, inds%n_inds] = counts
    return res

def evaluate_model(
        model,
        schema_path,
        test_path,
        classification_labels=None,
        plot_config=None):
    predictions, truth_index = get_predicitions(
        model, schema_path, test_path)
    pred_index = np.where(
        predictions == np.max(predictions, axis=1)[:, np.newaxis])[1]
    labels = _parse_classification_labels(classification_labels,
                                          pred_index, truth_index)
    _ = create_summary_text(
        pred_index, truth_index,
        classification_labels=labels,
        print_results=True)
    confustion_mat = create_confusion_matrix(pred_index, truth_index)
    plot_confusion_matrix(
        confustion_mat,
        labels, labels)
    if plot_config is not None:
        plot_summary_information(pred_index, truth_index, plot_config)
    return pred_index, truth_index

def region_dist(
        axis,
        selected_index, true_index,
        predictions, labels,
        plot_config,
        expand_bins=True):
    if predictions.size == 0:
        return
    bins = plot_config.get_bins(predictions[:, selected_index], array=True)
    for i in range(predictions.shape[-1]):
        data = predictions[:, i]
        if expand_bins:
            this_bins = plot_config.expand_bins(bins, data)
        else:
            this_bins = bins
        kwargs = plot_config.gen_kwargs(
                type="hist", index=i, label=labels[i],
                bins=this_bins)
        if i == true_index:
            kwargs["ls"] = ":"
        elif i == selected_index:
            kwargs["lw"] = 4
        else:
            kwargs["ls"] = "--"
        axis.hist(data, **kwargs)
    return

def _get_region_masks(pred, truth, n):
    region_masks = np.full((n, n, truth.size), False, dtype=bool)
    for i in range(4):
        for j in range(4):
            region_masks[i, j] = np.logical_and(pred==i, truth==j)
    return region_masks

def per_region_dists(
        model,
        schema_path,
        test_path,
        plot_config,
        classification_labels=None,
        expand_bins=True):
    predictions, truth_index = get_predicitions(
        model, schema_path, test_path)
    pred_index = np.where(
        predictions == np.max(predictions, axis=1)[:, np.newaxis])[1]
    labels = _parse_classification_labels(classification_labels,
                                          pred_index, truth_index)
    n_regions = predictions.shape[-1]
    region_masks = _get_region_masks(pred_index, truth_index, n_regions)
    _, axes = plot_config.setup_figure(n_regions, n_regions, figsize=(24, 20))
    for reco_i in range(n_regions):
        for true_i in range(n_regions):
            ax = axes[3-true_i, reco_i]
            these_preds = predictions[region_masks[reco_i, true_i]]
            region_dist(
                ax,
                reco_i, true_i,
                these_preds, labels,
                plot_config,
                expand_bins=expand_bins)
            plot_config.format_axis(
                ax, xlabel="GNN output",
                ylabel=f"Count ({region_masks[reco_i, true_i].sum()} total)")
            ax.legend()
    return plot_config.end_plot()
    

def simple_plot_history(history, init_point=0):
    for k, hist in history.history.items():
        plt.plot(hist[init_point:])
        plt.title(k)
        plt.show()
    return


# =====================================================================
#               Testing archtecture on the MUTAG dataset

_mutag_callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=32, restore_best_weights=True)
_mutag_metrics = [
    tf.keras.metrics.AUC(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.BinaryAccuracy()]

def test_architecture_on_MUTAG(model, hyper_params, mutag_folder):
    train_ds, val_ds = DataPreparation.get_mutag_data(mutag_folder)
    mutag_params = hyper_params.copy()
    mutag_params["callbacks"] = _mutag_callbacks
    mutag_params["metrics"] = _mutag_metrics
    mutag_params["epochs"] = 2048
    train_ds_batched, val_ds_batched = format_data(
        train_ds, val_ds, 32, None)
    hist = compile_and_train(model, mutag_params,
                             train_ds_batched, val_ds_batched)
    simple_plot_history(hist)
    return hist

def get_MUTAG_input_spec(hyper_params, mutag_folder):
    train_ds, val_ds = DataPreparation.get_mutag_data(mutag_folder)
    if hyper_params["training_batch"] is None:
        bs = 32
    else:
        bs = hyper_params["training_batch"]
    train_ds_batched, _ = format_data(
        train_ds, val_ds, bs, hyper_params["training_shuffle"])
    # [0] to get data (not label) spec
    return train_ds_batched.element_spec[0]

def get_MUTAG_predicitions(model, mutag_folder, which="val"):
    train_ds, val_ds = DataPreparation.get_mutag_data(mutag_folder)
    use_ds = train_ds if which.lower() == "train" else val_ds
    test_data = use_ds.map(lambda data, label : data)
    test_data = test_data.batch(batch_size=32)
    test_label = use_ds.map(lambda data, label : label)
    truth_index = convert_labels_to_truth(test_label)
    predictions = model.predict(test_data, batch_size=32)
    return predictions, truth_index
