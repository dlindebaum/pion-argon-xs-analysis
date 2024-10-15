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
from apps.cex_toy_parameters import PlotCorrelationMatrix as plot_confusion_matrix
# from python.analysis import Plots


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
#   stupid special cases which could be nicely centrally unified
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
    