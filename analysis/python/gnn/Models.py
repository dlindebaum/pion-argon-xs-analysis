# Created 19/01/24
# Dennis Lindebaum
# GNN models

import os
import pickle
import json
import copy
import warnings
import numpy as np
import awkward as ak
import tensorflow as tf
import tensorflow_gnn as tfgnn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from python.gnn import DataPreparation, Layers
# from python.gnn.model_plots import *
from python.analysis.Plots import PlotConfusionMatrix as plot_confusion_matrix
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
    weights_folder = os.path.join(folder_path, "checkpoints")
    if not os.path.exists(weights_folder):
        os.mkdir(weights_folder)
    return {"folder_path": folder_path,
            "params_path": os.path.join(
                folder_path, "hyper_params.pkl"),
            "text_params_path": os.path.join(
                folder_path, "hyper_params_repr.txt"),
            "model_path": os.path.join(
                folder_path, "model.tf"),
            "model_params_path": os.path.join(
                folder_path, "model_params.txt"),
            "weights_path": os.path.join(
                weights_folder, "weights"),
            "history_path": os.path.join(
                folder_path, "train_history.pkl")}

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
    """Load a hyper parameters dictionary"""
    with open(paths_dict["params_path"], 'rb') as f:
        params = pickle.load(f)
    return params

# def load_model(path, from_weights=True):
#     pass
    # if isinstance(path, dict):
    #     if from_weights:
    #         path = path["weights_path"]
    #     else:
    #         path = path["model_path"]
    # else:
    #     tf = not path[-4:] == ".dll"
    # if tf:
    #     return tf.keras.models.load_model(path)
    # with open(path, "rb") as f:
    #     return dill.load(f)
            

# =====================================================================
#                            Data formatting

def format_data(train, val, batch_size=32, shuffle_size=128):
    """Prepare training/validation datasets for training"""
    if shuffle_size is not None:
        train = train.shuffle(buffer_size=shuffle_size)
    train_batched = train.ragged_batch(batch_size=batch_size).repeat()
    val_batched = val.ragged_batch(batch_size=batch_size)
    return train_batched, val_batched

def load_data_from_hyper_params(hyper_params):
    """
    Load the training and validation sets in hyper_params, and format
    for training
    
    Parameters
    ----------
    hyper_params : dict
        Hyper parameters, including the schema, training, and
        validation paths, training batch formatting and any
        additional losses.
    
    Returns
    tf.Dataset
        Training dataset.
    tf.Dataset
        Validation dataset.
    """
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
    """Load the graph element spec dictated by the hyper parameters"""
    if ((hyper_params["data_folder"] is None)
        or (hyper_params["training_batch"] is None)):
        raise ValueError("Hyper-parameters do not contain data details")
    data_paths = DataPreparation.create_filepath_dictionary(
        hyper_params["data_folder"])
    train_ds = DataPreparation.load_record(
        data_paths["schema_path"],
        "") # tf doesn't try to load the train_ds as an actual file
        #   until you ask for an element. Thus passing nothing here
        #    works to handling removal of extra losses.
        # A proper method would required poping "classification", and
        #    any other extra losses out of the schema itself
    # raw_schema = tfgnn.create_graph_spec_from_schema_pb(
    #     tfgnn.read_schema(data_paths["schema_path"]))
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

example_params = {
    "node_dim": 8,
    "beam_node_dim": 8,
    "edge_dim": 4,
    "beam_edge_dim": 4,
    # Node setup
    "n_node_layers": 3,
    "n_node_hidden_depth": 64,
    # Dimensions for message passing.
    "message_heads": 8,
    "message_channels": 4,
    "beam_message_heads": 8,
    "beam_message_channels": 4,
    # edge_readout_dim=16,
    # edge_readout_hidden_layers=2,
    # Dimension for the logits.
    "final_hidden_layers": 0,
    "final_layer_nodes": 16,
    "num_classes": 4,
    # Classificatino only layer after regression readout
    "class_readout_dim": 16,
    # Number of message passing steps.
    "num_message_passing": 2,
    # Other hyperparameters.
    "regularisation": 8e-5,
    "dropout_rate": 0.1
}

example_message_passing = [
    Layers.PFOUpdate(),
    Layers.NeighbourUpdate(final_step=False),
    Layers.BeamCollection(),
    Layers.BeamConnectionUpdate(final_step=False)
]

example_constructor = [
    Layers.Setup(),
    Layers.InitialState(pfo_hidden=(64, 3),
                        neighbours_hidden=None,
                        beam_connections_hidden=None),
    # Layer is an output if given positional argument
    Layers.ReadoutClassifyNode("pion_id", which_nodes="pfo"),
    Layers.ReadoutClassifyNode("photon_id", which_nodes="pfo"),
    Layers.BeamCollection(next_state="concat"),
    # Loop constructor creates a sub loop
    Layers.LoopConstructor(example_message_passing,
                           loops=2),
    Layers.ReadoutClassifyEdge("pi0_id", which_edges="neighbours"),
    Layers.ReadoutNode(which_nodes="beam"),
    Layers.Dense(depth=16, n_layers=1),
    Layers.Dense("pion_count", depth=1),
    Layers.Dense("pi0_count", depth=1),
    Layers.Dense(depth=16, n_layers=1),
    Layers.Dense("reco_classification", depth=4),
    Layers.Dense("classifier", depth=4)
]

example_outputs = ["classifier",
                   "pion_count", "pi0_count",
                   "pion_id", "photon_id", "pi0_id",
                   "reco_classification"]

def create_normaliser_from_data(data_path_params, erf=False):
    """Load the corresponding normalisation as a normalising layer"""
    if isinstance(data_path_params, str):
        norms_path = data_path_params
    else:
        norms_path = data_path_params["norm_path"]
    with open(norms_path, "r") as f:
        json_dict = json.load(f)
    norms_dict = DataPreparation._norms_json_formatter(
        json_dict, invert=True)
    return Layers.NormaliseHiddenFeatures(**norms_dict, use_err_func=erf)

def construct_model(
        hyper_params,
        constructor, parameters, outputs,
        model_type="GATv2", save=True):
    """
    Constructs a GNN model using a list of Layers in `constructor`,
    formatted to be compatible with the hyper parameters, with global
    parameters in `parameters`, to generate the outputs indexed in
    `outputs`.

    The Layers in constructor should be instances of a class in
    `Layers` referencing the `LayerConstructor` or `LoopConstructor`
    base classes.
    `outputs` shold be a list referencing the any strings given as
    first arguments to any Layers in `constructor`, allows mapping the
    output from any layer in the constructor to any desired output.

    Parameters
    ----------
    hyper_params : dict
        Dictionary containing hyper parameters.
    constructor : list
        List of Layers defining the model.
    parameters : dict
        Dictionary of global layer parameters. These will be passed as
        kwargs to all input layers of the Model (unless sepcified in
        the Layer initialisation).
    outputs : list
        List of strings referencing any identification strings in the
        Layers to control the output specifications.
    model_type : str, optional
        Type of the model to be constructed. Default is "GATv2".
    save : bool, optional
        Whether to save the model parameters to a file. Default is
        True.

    Returns
    -------
    tf.keras.Model
        The constructed TensorFlow Keras model.
    """
    if save:
        model_params_dict = {
            "model_type": model_type,
            "model_parameters": parameters,
            "model_constructor": [repr(c) for c in constructor],
            "model_outputs": outputs,
            "layers_version": Layers.__version__}
        with open(hyper_params["model_params_path"], "w") as f:
            json.dump(model_params_dict, f, indent=4)
    layer_funcs=[]
    output_index=[]
    layer_funcs, which_outputs = Layers.parse_constructor(constructor, parameters)
    which_outputs = [outputs.index(o) if o in outputs else o for o in which_outputs]
    model_out = outputs.copy()
    input_graph_spec = get_spec_from_hyper_params(hyper_params)
    input_graph = tf.keras.layers.Input(type_spec=input_graph_spec)
    graph = input_graph
    for func, output_ind in zip(layer_funcs, which_outputs):
        if output_ind is None:
            graph = func(graph)
        else:
            model_out[output_ind] = func(graph)
    return tf.keras.Model(inputs=[input_graph], outputs=model_out)

def _parse_string_kwargs(string, kwarg_types=None):
    kwargs = {}
    data = [s.strip() for s in string.split(",") if len(s) > 0]
    for i, d in enumerate(data):
        if "=" in d:
            key, val = d.split("=")
        else:
            key = list(kwarg_types.keys())[i]
            val = d
        kwargs[key] = kwarg_types[key](val)
    return kwargs

def load_layer(layer_repr):
    if layer_repr[:16] == "LoopConstructor(":
        # If loop constructor, first argument is a list of layers
        constructors = []
        this_constructor = ""
        curr_depth = 0
        curr_ind = 17
        while curr_depth >= 0:
            char = layer_repr[curr_ind]
            curr_ind += 1
            if char == "[" or char == "(":
                curr_depth += 1
            elif char == "]" or char == ")":
                curr_depth -= 1
                if curr_depth == -1:
                    constructors.append(this_constructor)
                    continue
            elif curr_depth == 0:
                if char == ",":
                    constructors.append(this_constructor)
                    this_constructor = ""
                    continue
                elif char == " ":
                    continue
            this_constructor += char
        constructors = [load_layer(c) for c in constructors]
        extra_args = layer_repr[curr_ind:-1]
        expected_kwargs = {"loops": int, "final_step": bool}
        loop_kwargs = _parse_string_kwargs(extra_args, expected_kwargs)
        return Layers.LoopConstructor(constructors, **loop_kwargs)
    else:
        return eval("Layers." + layer_repr)

def _split_ver_patch(version):
    major, minor, patch = version.split(".")
    return major, ".".join([minor, patch])

def _check_version(version_model, version_module):
    maj1, min1 = _split_ver_patch(version_model)
    maj2, min2 = _split_ver_patch(version_module)
    if maj1 != maj2:
        raise ValueError(
            f"Model constructed with Layers version {version_model}, "
            + f"but trying to load with version {version_module}")
    if min1 != min2:
        warnings.warn(f"Layer versions {version_model} (model), "
                      + f"{version_module} (current module) do not "
                      + "match, may be unexpected behaviour.")
    return

def add_id_output_to_loaded_model(model):
    """
    Append the event ID information of a graph to the end of the a
    model's output.

    Parameters
    ----------
    model : tf.keras.Model
        A loaded model.
    
    Returns
    tf.keras.Model
        Model with the final output as the event ID information.
    """
    input = model.input
    id_out = input.merge_batch_to_components().context.features["id"]
    model_outs = model(input)
    if not isinstance(model, list):
        model_outs = [model_outs]
    return tf.keras.Model(inputs=[input], outputs=model(input) + [id_out])


def load_model_from_file(model_folder, new_norm=None, new_data_folder=None, construct_only=False):
    """
    Loads a model contained in the supplied folder.

    This is done by reconstructing the model from the model_params.txt
    file within the folder, and then reloading the corresponding
    weights from the checkpoints/weights data.

    The normalisation layer may be substituted by passing new_norm as a
    string path to normalisation parameters json, dictionary of data
    path parameters, or an already constructed NormaliseHiddenFeatures
    layer.

    Parameters
    ----------
    model_folder : str
        Path to folder containing the model data.
    new_norm : str, dict, or NormaliseHiddenFeatures, optional
        If passed, any existing NormaliseHiddenFeatures layers with a
        new NormaliseHiddenFeatures layer with the supplied parameters.
        Parameters are supplied string path to normalisation parameters
        json, a dictionary of data path parameters including
        "norm_path", or an already constructed NormaliseHiddenFeatures
        layer. Default is None.
    new_data_folder : str, optional
        If passed, change the input spec of the graph to the spec of
        the data found in the supplied data folder. This is intended to
        be used to swap to a data type graph structure, which does not
        include any truth information. Default is None.

    Returns
    -------
    tf.keras.Model
        Copy of the model which was saved.
    """
    paths_dict = make_model_paths(model_folder)
    with open(paths_dict["model_params_path"], "r") as f:
        model_properties = json.load(f)
    load_version = model_properties["layers_version"]
    _check_version(load_version, Layers.__version__)
    params = model_properties["model_parameters"]
    outputs = model_properties["model_outputs"]
    model_type = model_properties["model_type"]
    constructor = [load_layer(rep) for rep in model_properties["model_constructor"]]
    if new_norm is not None:
        if isinstance(new_norm, Layers.NormaliseHiddenFeatures):
            new_layer = new_norm
        else:
            new_layer = create_normaliser_from_data(new_norm)
        for i, layer in enumerate(constructor):
            if isinstance(layer, Layers.NormaliseHiddenFeatures):
                constructor[i] = new_layer
    try:
        with open(paths_dict["params_path"], 'rb') as f:
            hyper_params = pickle.load(f)
    except EOFError:
        print("Pickle file seems to have been overwritten (multiple trains?)."
              + " Loading from text...")
        with open(paths_dict["text_params_path"], "r") as f:
            hyper_params = eval(f.read().replace('\n', ''))
            # Bad saving practice means the path items we want are
            #   strings like "'string'", so the leading/trailing
            #   "'"-s must be removed.
            for key, val in hyper_params.items():
                if "path" in key or "folder" in key:
                    hyper_params[key] = val[1:-1]
    # This is used by the model to get the input graph spec, so editing
    #   this allows the model to accept graphs without the excess truth
    #   data. This part doesn't have any loaded weights.
    if new_data_folder is not None:
        hyper_params["data_folder"] = new_data_folder
    model = construct_model(
        hyper_params,
        constructor,
        params,
        outputs,
        model_type=model_type,
        save=False)
    if not construct_only:
        model.load_weights(paths_dict["weights_path"])
    return model

# =====================================================================
#                            Model training

def compile_and_train(
        model,
        hyper_params,
        batched_train,
        batched_val,
        print_summary=True,
        partial_train=False):
    """
Compiles and trains a TensorFlow Keras model based on the provided
hyper parameters and datasets.

Parameters
----------
model : tf.keras.Model
    The TensorFlow Keras model to be compiled and trained.
hyper_params : dict
    Dictionary containing hyper-parameters for compiling and training
    the model, as generated by `Models.generate_hyper_params`.
batched_train : tf.data.Dataset
    Batched training dataset.
batched_val : tf.data.Dataset
    Batched validation dataset.
print_summary : bool, optional
    Whether to print the model summary. Default is True.
partial_train : bool, optional
    If True, save the weights as "_partial" at the end of training.
    This can be used as a training checkpoint to load later.  Intent is
    to allow changing loss weightings partly through the process.
    Default is False.

Returns
-------
history : tf.keras.callbacks.History
    A record of training loss values and metrics values at successive
    epochs, as well as validation loss values and validation metrics
    values.
"""
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
    if not partial_train:
        model.save_weights(hyper_params["weights_path"])
        model.save(hyper_params["model_path"], save_format="tf", overwrite=False)
    else:
        partial_weights_path = hyper_params["weights_path"] + "_partial"
        model.save_weights(partial_weights_path)
    with open(hyper_params["history_path"], "wb") as f:
        pickle.dump(history.history, f)
    return history

# =====================================================================
#                           Model evaluation
#    Plotting functions for evaluation imported from gnn.model_plots

def convert_labels_to_truth(labels):
    """Convert 1-hot true classification vector to a channel number"""
    size = labels.take(1).get_single_element().shape[0]
    index = np.arange(size)
    if size == 1:
        index += 1
    return np.array([lab @ index for lab in labels.as_numpy_iterator()])

def _parse_classification_labels(labels, pred, truth=None):
    """If labels is passed as None, create defaul label names"""
    if labels is None:
        n_indicies = int(np.max(pred) + 1)
        if truth is not None:
            n_indicies = max(n_indicies, int(np.max(truth) + 1))
        labels = [f"class index {i}" for i in range(n_indicies)]
    return labels

def create_summary_text(
        pred_index,
        truth_index,
        classification_labels=None,
        print_results=False):
    """Create a string summarising the performance of the network"""
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

def _type_concatenate(arrs):
    """Perform concatenation appropriate to the data type"""
    if isinstance(arrs[0], tf.RaggedTensor):
        return tf.concat(arrs, axis=0)
    elif isinstance(arrs[0], ak.Array):
        return ak.concatenate(arrs, axis=0)
    return np.concatenate(arrs, axis=0)

# def pred_decorator(func, has_truth=False):
#     if has_truth:
#         def decorated(pred, truth):
#             return func(pred)
#         return decorated
#     else:
#         return func

# def gen_model_pred_map(index, added_id, has_truth):
#     if (index < 0) and added_id:
#         index -= 1
#         warnings.warn('Offset negative requested index to account for '
#                       + 'additional ID output. If this is undesired, '
#                       + 'add 1, or explicitly request "id"')
#     @pred_decorator(has_truth=has_truth)
#     def map_func(predictions):
#         return predictions[index]
#     return map_func

# def gen_id_pred_map(has_truth):
#     @pred_decorator(has_truth=has_truth)
#     def map_func(predictions):
#         return predictions[-1]
#     return map_func

# def gen_truth_map(index):
#     def map_func(preds, truth):
#         return truth[index]
#     return map_func

# def contains_truth_request(signature):
#     return any([sig in DataPreparation.known_truths + ["classification"]])

def record_mapper(has_truth, truth_out_indicies):
    if has_truth:
        def map_func(data, truth):
            return data, [truth[ind] if ind is not None else None
                          for ind in truth_out_indicies]
    else:
        def map_func(data):
            return data, truth_out_indicies

def get_predictions_signature(
        signature, model,
        schema_path, test_path,
        start_ind=None, n_graphs=None):
    """
    Get model outputs as specified by a signature list.

    Allowed signatures are:
     - int: These are treated as an index of which model prediction to
       get. Note 0 is the main classifier. Also note, if using negative
       integers with "id" also appearing in the signature, this will be
       offset by -1 to account for the additional id output.
     - "id": Get the array of event IDs for confirming a match.
     - "classification": Classification label used during training
     - str: If not "id" or "classification", must be one of
       DataPreparation.known_truths.

    Parameters
    ----------
    signature : list[str|int]
        List indicating what outputs should be provided, allowed
        signatures hinted above.
    model : tf.keras.Model
        Model for which to generate predictions (could be passed as
        None if no prediction requests in the signature).
    schema_path : str
        Path to find the schema for grpah loading.
    test_path : str
        Path to the graphs to be loaded.
    start_ind : int, optional
        Index of graph to begin loading from, or the first graph if
        None. Default is None
    n_graphs : int, optional
        Number of graphs to load, or all remaing graphs in file if
        None/larger than total graphs present. Default is None.
    """
    has_truth = False
    add_id = "id" in signature
    pred_indicies = []
    truth_indicies = []
    extra_losses = []
    curr_truth_index = 1
    for val in signature:
        if val in DataPreparation.known_truths:
            has_truth=True
            pred_indicies.append(None)
            truth_indicies.append(curr_truth_index)
            curr_truth_index += 1
            extra_losses.append(val)
        elif val == "classification":
            has_truth=True
            pred_indicies.append(None)
            truth_indicies.append(0)
        elif val == "id":
            pred_indicies.append(-1)
            truth_indicies.append(None)
        elif val < 0:
            pred_indicies.append(val - int(add_id))
            truth_indicies.append(None)
        else:
            pred_indicies.append(val)
            truth_indicies.append(None)
    if add_id:
        pred_model = add_id_output_to_loaded_model(model)
    else:
        pred_model = model
    test_data = DataPreparation.load_record(
        schema_path, test_path,
        no_label = (not has_truth),
        extra_losses = extra_losses,
        start_ind=start_ind, n_graphs=n_graphs)
    graphs, output = test_record.map(
        record_mapper(has_truth, truth_indicies))
    if all([ind is None for ind in pred_indicies]):
        # Return output before prediction if no predictions present
        return output
    graphs = graphs.batch(batch_size=512)
    preds = pred_model.predict(graphs, batch_size=512)
    for output_ind, pred_ind in enumerate(pred_indicies):
        if pred_ind is not None:
            output[output_ind] = preds[pred_ind]
    return pred_ind


    # # These two lines are inefficient, requires looking through the
    # # list 3 times. But it is much simpler to understand compared to
    # # tracking mutable objects.
    # add_id = "id" in signature
    # get_truth = contains_truth_request
    # curr_truth_index = 1  # Start at one, since 0 is always class label
    # map_funcs = []
    # extra_losses = []
    # for val in signature:
    #     if val in DataPreparation.known_truths:
    #         map_funcs.append(gen_truth_map(curr_truth_index))
    #         curr_truth_index += 1
    #         extra_losses.append(val)
    #     elif val == "classification":
    #         map_funcs.append(gen_truth_map(0))
    #     elif val == "id":
    #         gen_id_pred_map(get_truth)
    #     else:
    #         try:
    #             map_funcs.append(gen_model_pred_map(val, add_id, get_truth))
    #         except:
    #             raise ValueError(f"Unknown signature type: {val}")
    # if add_id:
    #     pred_model = add_id_output_to_loaded_model(model)
    # else:
    #     pred_model = model
    # test_data = DataPreparation.load_record(
    #     schema_path, test_path,
    #     no_label = (not get_truth),
    #     extra_losses = extra_losses,
    #     start_ind=start_ind, n_graphs=n_graphs)
    # test_data = test_record.map(lambda data, label : data)
    # test_data = test_data.batch(batch_size=512)
    # predictions = model.predict(test_data, batch_size=512)
    # pass

# CAREFUL with the following plotting functions, they have a bunch of
#   stupid special cases which are nicely centrally unified
def get_predictions(
        model,
        schema_path, test_path,
        pred_index=0, other_truth=None,
        return_truth=True):
    """
    Load the predictions of the passed model, along with some truth.

    Note: only valid for MC type data (including true labels).
    For data without true labels (or to simulate not ture labels), use
    the `get_data_predictions` function.

    Parameters
    ----------
    model : tf.keras.Model
        Model to generate predictions.
    schema_path : str or list
        Path to the schema to describing the graph data. If passed as a
        list, each graph in the list will be loaded, and the results
        concatenated together.
    test_path : str or list
        Path to the graph data record. If passed as a list, it must
        have the same length as schema path. The results from each
        graph will be found and concatenated together.
    pred_index : int, optional
        Which graph prediction to return. Refers to the output given by
        the model structure. Default is 0 (this should always be the
        main classification prediction).
    other_truth : str, optional
        If passed, istead of the classification loss, return a
        different loss parameter available in the graph structure. Loss
        must reference a loss available in
        `DataPreparation._make_decode_func`. If this isn't a
        classification type loss, the result will be returned as an
        awkward array. Default is None.
    return_truth : bool, optional
        If True, return the loss truth component, else return only the
        prediction. Default is True.

    Returns
    -------
    np.ndarray
        Prediction of the model at `pred_index`.
    np.ndarray or ak.Array
        Truth value defined by `other_truth`, else the main
        classification loss.
    """
    if other_truth is not None:
        other_truth = [other_truth]
    if isinstance(test_path, list):
        if ((not isinstance(schema_path, list))
            or (not len(schema_path) == len(test_path))):
            raise TypeError(
                "If test_path is a list, "
                "schema_path must be an equal length list")
        results = []
        for s_path, t_path in zip(schema_path, test_path):
            results.append(_get_predictions_core(
                model, s_path, t_path,
                pred_index=pred_index, other_truth=other_truth,
                return_truth=return_truth))
        if not return_truth:
            return _type_concatenate(results)
        else:
            preds = _type_concatenate([r[0] for r in results])
            truths = _type_concatenate([r[1] for r in results])
            return preds, truths
    else:
        return _get_predictions_core(
            model, schema_path, test_path,
            pred_index=pred_index, other_truth=other_truth,
            return_truth=return_truth)

def _get_predictions_core(
        model,
        schema_path, test_path,
        pred_index=0, other_truth=None,
        return_truth=True):
    test_record = DataPreparation.load_record(schema_path, test_path,
                                              extra_losses=other_truth)
    test_data = test_record.map(lambda data, label : data)
    test_data = test_data.batch(batch_size=512)
    predictions = model.predict(test_data, batch_size=512)
    if isinstance(predictions, list):
        # First predicition is always the classification
        predictions = predictions[pred_index]
    if not return_truth:
        return predictions
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

def get_data_predictions(
        model,
        schema_path, test_path,
        start_ind=None, n_graphs=None,
        which_pred=0):
    """
    Load the predictions of the passed model, along with the event IDs.

    Note: this does not return any truth information, for that use
    `get_predictions`. This function will however work, even if truth
    information is present.

    Parameters
    ----------
    model : tf.keras.Model
        Model to generate predictions.
    schema_path : str or list
        Path to the schema to describing the graph data. If passed as a
        list, each graph in the list will be loaded, and the results
        concatenated together.
    test_path : str or list
        Path to the graph data record. If passed as a list, it must
        have the same length as schema path. The results from each
        graph will be found and concatenated together.
    start_ind : int, optional
        Which graph in the record to start reading from. If None, read
        from the first file. Default is None.
    n_graphs : int, optional
        How many graphs to read from the data file. If None, read until
        the end of the file. Default is None.
    which_pred : int, optional
        Which prediction index to return. Corresponds to the list of
        GNN model outputs. Default is 0 (corresponding to the main
        classifier output).
        
    Returns
    -------
    np.ndarray
        Prediction of the model at `pred_index`.
    np.ndarray
        Array of unique identifiers of the events. The should be
        checked with the expected IDs when relating to ntuples.
    """
    id_model = add_id_output_to_loaded_model(model)
    if isinstance(test_path, list):
        if ((not isinstance(schema_path, list))
            or (not len(schema_path) == len(test_path))):
            raise TypeError(
                "If test_path is a list, "
                "schema_path must be an equal length list")
        results = []
        for s_path, t_path in zip(schema_path, test_path):
            results.append(_get_data_predictions_core(
                id_model, s_path, t_path,
                start_ind=start_ind, n_graphs=n_graphs,
                which_pred=which_pred))
        preds = np.concatenate([r[0] for r in results], axis=0)
        ids = np.concatenate([r[1] for r in results], axis=0)
    else:
        preds, ids = _get_data_predictions_core(
            id_model, schema_path, test_path,
            start_ind=start_ind, n_graphs=n_graphs,
            which_pred=which_pred)
    return preds, ids

def _get_data_predictions_core(
        id_model,
        schema_path, test_path,
        start_ind=None, n_graphs=None,
        which_pred=0):
    test_data = DataPreparation.load_record(
        schema_path, test_path,
        no_label=True,
        start_ind=start_ind, n_graphs=n_graphs)
    # test_data = test_data.map(lambda data, label : data)
    test_data = test_data.batch(batch_size=512)
    predictions = id_model.predict(test_data)
    return predictions[which_pred], predictions[-1]

def _get_paths_from_params(path_params, which_data="test"):
    """Extract the schema and data paths from (list of) path params"""
    if isinstance(path_params, list):
        schema = [p["schema_path"] for p in path_params]
        data = [p[f"{which_data}_path"] for p in path_params]
    else:
        schema = path_params["schema_path"]
        data = path_params[f"{which_data}_path"]
    return schema, data


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
    preds, truths = get_predictions(
        model, path_params["schema_path"], path_params["test_path"])
    bkg_preds = preds[:, 0][truths==0.]
    sig_preds = preds[:, 0][truths==1.]
    return _sig_bkg_hist_plotter(sig_preds, bkg_preds, "CEx.", plot_config)

def plot_binary_extra_loss_dist(
        model,
        extra_loss, loss_index,
        path_params, plot_config,
        which_data="test"):
    schema, data = _get_paths_from_params(path_params, which_data=which_data)
    preds, truths = get_predictions(
        model, schema, data,
        pred_index=loss_index, other_truth=extra_loss)
    if isinstance(preds, tf.RaggedTensor):
        preds = preds.merge_dims(0, -1).numpy()
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
        logits=False, mask=None,
        do_title=False,
        which_data="test"):
    schema, data = _get_paths_from_params(path_params, which_data=which_data)
    preds, truths = get_predictions(
        model, schema, data,
        pred_index=loss_index, other_truth=extra_loss)
    if isinstance(preds, tf.RaggedTensor):
        preds = preds.merge_dims(0, -1).numpy()
    else:
        preds = ak.ravel(preds)
    truths = ak.ravel(truths)
    if mask is not None:
        preds = preds[mask]
        truths = truths[mask]
    if do_title:
        title=extra_loss.replace('_', ' ').title()
    else:
        print(extra_loss.replace('_', ' ').title())
        title=None
    if "pions" in extra_loss.lower():
        particle_str = r"$\pi^\pm$ "
    elif "pi0s" in extra_loss.lower():
        particle_str = r"$\pi^0$ "
    else:
        particle_str = ""
    if "bt" in extra_loss.lower():
        truth_str = "backtracked "
    else:
        truth_str = ""
    plot_config.setup_figure(title=title)
    pred_bins = plot_config.get_bins(preds, array=True)
    max_true = max(truths)
    offset=-0.4
    width=0.8
    true_bins = np.linspace(offset, max_true + offset+0.5, int(max_true)*2 + 2)
    true_bins[1::2] += width-0.5
    plt.hist2d(
        truths, preds, norm=LogNorm(), bins=(true_bins, pred_bins))
    # plt.scatter(
    #     truths, preds, **plot_config.gen_kwargs(index=0, marker="x"))
    plot_config.format_axis(xlog=False, ylog=False,
                            xlabel=f"True {truth_str}{particle_str}count",
                            ylabel=f"Predicted {particle_str}count")
    maj_ticks = np.arange(max_true+1)
    plt.gca().set_xticks(maj_ticks)
    plt.gca().set_xticks(true_bins, minor=True)
    plt.colorbar(label = "Count")
    return plot_config.end_plot()

def plot_confusion_extra_loss(
        model,
        extra_loss, loss_index,
        path_params,
        classification_labels=None,
        plot_config=None,
        which_data="test"):
    schema, data = _get_paths_from_params(path_params, which_data=which_data)
    preds, truth_index = get_predictions(
        model, schema, data,
        pred_index=loss_index, other_truth=extra_loss)
    pred_index = np.argmax(preds, axis=1)
    labels = _parse_classification_labels(classification_labels,
                                          pred_index, truth_index)
    _ = create_summary_text(
        pred_index, truth_index,
        classification_labels=labels,
        print_results=True)
    confustion_mat = create_confusion_matrix(pred_index, truth_index)
    plot_confusion_matrix(
        confustion_mat,
        labels, labels,
        x_label="Predicted process", y_label="True process")
    if plot_config is not None:
        plot_summary_information(pred_index, truth_index, plot_config)
    return pred_index, truth_index

def plot_confusion_main_vs_reco_loss(
        model,
        extra_loss, loss_index,
        path_params,
        classification_labels=None,
        plot_config=None,
        which_data="test"):
    schema, data = _get_paths_from_params(path_params, which_data=which_data)
    _, truth_index = get_predictions(
        model, schema, data,
        pred_index=loss_index, other_truth=extra_loss)
    preds, _ = get_predictions(
        model,
        path_params["schema_path"], path_params["test_path"])
    pred_index = np.argmax(preds, axis=1)
    labels = _parse_classification_labels(classification_labels,
                                          pred_index, truth_index)
    _ = create_summary_text(
        pred_index, truth_index,
        classification_labels=labels,
        print_results=True)
    confustion_mat = create_confusion_matrix(pred_index, truth_index)
    plot_confusion_matrix(
        confustion_mat,
        labels, labels,
        x_label="Predicted process", y_label="True process")
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
        path_params,
        classification_labels=None,
        plot_config=None,
        which_data="test"):
    schema, data = _get_paths_from_params(path_params, which_data=which_data)
    predictions, truth_index = get_predictions(
        model, schema, data)
    pred_index = np.argmax(predictions, axis=1)
    labels = _parse_classification_labels(classification_labels,
                                          pred_index, truth_index)
    _ = create_summary_text(
        pred_index, truth_index,
        classification_labels=labels,
        print_results=True)
    confustion_mat = create_confusion_matrix(pred_index, truth_index)
    plot_confusion_matrix(
        confustion_mat,
        labels, labels,
        x_label="Predicted process", y_label="True process")
    if plot_config is not None:
        plot_summary_information(pred_index, truth_index, plot_config)
    return pred_index, truth_index

def template_dists(
        model,
        path_params,
        plot_config,
        classification_labels=None,
        expand_bins=True,
        which_data="test",
        split_figs=False):
    schema, data = _get_paths_from_params(path_params, which_data=which_data)
    predictions, truth_index = get_predictions(
        model, schema, data)
    if split_figs:
        template_dists_from_preds_multi_fig(
        predictions, truth_index, plot_config,
        classification_labels=classification_labels)
    else:
        return template_dists_from_preds(
            predictions, truth_index, plot_config,
            classification_labels=classification_labels)

def template_dists_from_preds(
        predictions,
        truth_indicies,
        plot_config,
        classification_labels=None):
    labels = _parse_classification_labels(
        classification_labels, np.argmax(predictions, axis=1), truth_indicies)
    n_regions = predictions.shape[-1]
    fig_dims = int(np.ceil(n_regions**0.5))
    _, axes = plot_config.setup_figure(fig_dims, fig_dims)
    for i, temp in enumerate(labels):
        ax = axes[fig_dims-1-(i//2), i%fig_dims]
        ax.set_title(f"{temp} template")
        temp_preds = predictions[truth_indicies == i]
        for j, lab in enumerate(labels):
            lw = 3 + 2*(temp == lab)
            bins = plot_config.get_bins(temp_preds[:, j])
            ax.hist(
                temp_preds[:, j],
                **plot_config.gen_kwargs(
                    type="hist", index=j, bins=bins, label=lab, lw=lw))
        plot_config.format_axis(ax, xlabel="GNN score", ylabel = "Count")
        ax.legend()
    return plot_config.end_plot()

def template_dists_from_preds_multi_fig(
        predictions,
        truth_indicies,
        plot_config,
        classification_labels=None):
    labels = _parse_classification_labels(
        classification_labels, np.argmax(predictions, axis=1), truth_indicies)
    n_regions = predictions.shape[-1]
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)
    range_pred = max_pred - min_pred
    if (min_pred>=0) and (max_pred<=1) and (range_pred>=0.5):
        bins=np.linspace(0,1,int(30/range_pred))
    else:
        bins = plot_config.get_bins(predictions.flatten(), array=True)
    for i, temp in enumerate(labels):
        plot_config.setup_figure(size=[1/n_regions])
        temp_preds = predictions[truth_indicies == i]
        for j, lab in enumerate(labels):
            lw = 1.5 + 1*(temp == lab)
            plt.hist(
                temp_preds[:, j],
                **plot_config.gen_kwargs(
                    type="hist", index=j, bins=bins, label=lab, lw=lw))
        plot_config.format_axis(xlabel="GNN score", ylabel = "Count")
        plt.legend()
    return plot_config.end_plot()

def region_dist(
        axis,
        selected_index, true_index,
        predictions, labels,
        plot_config,
        expand_bins=True,
        min_bin_size=None):
    if predictions.size == 0:
        return
    if expand_bins:
        bin_data = predictions.flatten()
    else:
        bin_data = predictions[:, selected_index]
    bins = plot_config.get_bins(bin_data, array=True)
    if min_bin_size is not None:
        if bins.size <= min_bin_size:
            bins = np.linspace(bins[0], bins[-1], min_bin_size)
    for i in range(predictions.shape[-1]):
        data = predictions[:, i]
        this_bins = bins
        # if expand_bins:
        #     this_bins = plot_config.expand_bins(bins, data)
        # else:
        #     this_bins = bins
        kwargs = plot_config.gen_kwargs(
                type="hist", index=i, label=labels[i],
                bins=this_bins)
        if i == selected_index:
            kwargs["lw"] = 4
            # kwargs["ls"] = ":"
        elif i == true_index:
            kwargs["lw"] = 3
            # kwargs["ls"] = "--"
        else:
            kwargs["lw"] = 2
            # kwargs["ls"] = "--"
        axis.hist(data, **kwargs)
    return

def _get_region_masks(pred, truth, n):
    region_masks = np.full((n, n, truth.size), False, dtype=bool)
    for i in range(n):
        for j in range(n):
            region_masks[i, j] = np.logical_and(pred==i, truth==j)
    return region_masks

def per_region_dists(
        model,
        path_params,
        plot_config,
        classification_labels=None,
        expand_bins=True,
        which_data="test"):
    schema, data = _get_paths_from_params(path_params, which_data=which_data)
    predictions, truth_index = get_predictions(
        model, schema, data)
    pred_index = np.argmax(predictions, axis=1)
    labels = _parse_classification_labels(classification_labels,
                                          pred_index, truth_index)
    n_regions = predictions.shape[-1]
    region_masks = _get_region_masks(pred_index, truth_index, n_regions)
    _, axes = plot_config.setup_figure(n_regions+1, n_regions+1, figsize=(32, 24))
    for reco_i in range(n_regions):
        for true_i in range(n_regions):
            ax = axes[n_regions-1-true_i, reco_i]
            these_preds = predictions[region_masks[reco_i, true_i]]
            region_dist(
                ax,
                reco_i, true_i,
                these_preds, labels,
                plot_config,
                expand_bins=expand_bins,
                min_bin_size=20)
            plot_config.format_axis(
                ax, xlabel="GNN output",
                ylabel=f"Count ({region_masks[reco_i, true_i].sum()} total)")
            ax.legend()
        ax = axes[-1, reco_i]
        all_true_mask = region_masks[reco_i, 0]
        for i in range(1, n_regions):
            all_true_mask = np.logical_or(all_true_mask, region_masks[reco_i, i])
        these_preds = predictions[all_true_mask]
        region_dist(
                ax,
                reco_i, -1,
                these_preds, labels,
                plot_config,
                expand_bins=expand_bins,
                min_bin_size=20)
        plot_config.format_axis(
                ax, xlabel="GNN output",
                ylabel=f"Count ({region_masks[reco_i, :].sum()} total)")
        ax.legend()
    for true_i in range(n_regions):
        ax = axes[n_regions-1-true_i, -1]
        all_reco_mask = region_masks[0, true_i]
        for i in range(1, n_regions):
            all_reco_mask = np.logical_or(all_reco_mask, region_masks[i, true_i])
        these_preds = predictions[all_reco_mask]
        # these_preds = predictions[np.sum(region_masks, axis=0)[true_i]]
        region_dist(
                ax,
                -1, true_i,
                these_preds, labels,
                plot_config,
                expand_bins=expand_bins,
                min_bin_size=20)
        plot_config.format_axis(
                ax, xlabel="GNN output",
                ylabel=f"Count ({region_masks[:, true_i].sum()} total)")
        ax.legend()
    return plot_config.end_plot()

def total_score_dist(
        model,
        path_params,
        plot_config,
        classification_labels=None,
        expand_bins=True,
        which_data="test"):
    schema, data = _get_paths_from_params(path_params, which_data=which_data)
    predictions, _ = get_data_predictions(
        model, schema, data)
    return total_score_dist_from_preds(
        predictions,
        plot_config,
        classification_labels=classification_labels,
        expand_bins=expand_bins)

def total_score_dist_from_preds(
        predictions,
        plot_config,
        classification_labels=None,
        expand_bins=True):
    pred_index = np.argmax(predictions, axis=1)
    labels = _parse_classification_labels(classification_labels,
                                          pred_index, None)
    bins = plot_config.get_bins(predictions.flatten(), array=True)
    plot_config.setup_figure()
    for i in range(predictions.shape[-1]):
        data = predictions[:, i]
        plt.hist(data, **plot_config.gen_kwargs(
            type="hist", index=i, label=labels[i],
            bins=bins))
    plot_config.format_axis(
                xlabel="GNN output",
                ylabel="Count")
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
