#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:47

Author: Shyam Bhuller

Description: Selection studies for the charge exchange analysis.
"""
import argparse
import os
import warnings

from rich import print as rprint
from python.analysis import (
    Master, Plots, Processing, cross_section, Utils)
from python.gnn import Models, DataPreparation
import python.analysis.SelectionTools as st
import apps.cex_beam_selection_studies as beam_selection
from tensorflow.python.framework.errors_impl import NotFoundError

import awkward as ak
import numpy as np
import pandas as pd

def run(i, file, n_events, start, selected_events, args) -> dict:
    events = Master.Data(file, n_events, start, args["nTuple_type"], args["pmom"])
    output = {
        "name" : file,
        "predictions": None,
        "ids": None,
        "truth_regions": None
    }
    is_mc = (not args["data"])
    events = beam_selection.BeamPionSelection(events, args, is_mc)
    evt_ids = DataPreparation.make_evt_ids(events)
    graph_path_params = DataPreparation.create_filepath_dictionary(args["graph"])
    gnn_model = Models.load_model_from_file(
        # Model path
        args["gnn_model_path"],
    # Where to find the schema so model knows to look for truth or not
        new_data_folder=graph_path_params["folder_path"],
    # Change input PFO property normalisations before running the GNN
        new_norm=args["graph_norm"])
    # If training sample, still grab IDs, but enforce no predictions
    if args["train_sample"]:
        gnn_scores = None # Enforce no predictions on train sample
        try:
            _, loaded_evt_ids = Models.get_data_predictions(
                gnn_model,
                [graph_path_params["schema_path"],
                 graph_path_params["schema_path"]],
                [graph_path_params["train_path"],
                 graph_path_params["val_path"]])
        except NotFoundError:
            warnings.warn("Was specified as train sample, but can't find "
                          + f"train and val data for file {events.filename}\n"
                          + "Skipping...")
            loaded_evt_ids = evt_ids
    else:
        gnn_scores, loaded_evt_ids = Models.get_data_predictions(
        gnn_model,
        graph_path_params["schema_path"],
        graph_path_params["test_path"])
    # Confirm graphs match, and ordering is the same
    assert evt_ids.shape == loaded_evt_ids.shape
    assert np.all(evt_ids == loaded_evt_ids)
    output["predictions"] = gnn_scores
    output["ids"] = loaded_evt_ids
    if is_mc and (not args["train_sample"]):
        _, trained_regions = Models.get_predictions(
            gnn_model,
            graph_path_params["schema_path"],
            graph_path_params["test_path"])
        output["truth_regions"] = trained_regions
    return output

def MakeRegionPlots(outputs_mc : dict, outputs_data : dict, outDir : str):
    """ Correlation matrices for truth and reco region selection.

    Args:
        outputs_mc_masks (dict): mc masks for each region
        outputs_data_masks (dict): data masks for each region
        outDir (str): output directory
    """
    plt_cfg = Plots.PlotConfig()
    with Plots.PlotBook(outDir + "regions.pdf") as pdf:
        mc_confusion = Models.create_confusion_matrix(
            np.argmax(outputs_mc["predictions"], axis=1),
            outputs_mc["truth_regions"])
        Plots.PlotConfusionMatrix(
            mc_confusion, title="MC GNN confusion",
            x_label="Prediction", y_label="Truth")
        pdf.Save()
        plt_cfg.TITLE = "MC GNN predictions"
        Models.total_score_dist_from_preds(outputs_mc["predictions"], plt_cfg)
        pdf.Save()
        plt_cfg.TITLE = "MC GNN predictions by process"
        Models.template_dists_from_preds(
            outputs_mc["predictions"], outputs_mc["truth_regions"], plt_cfg)
        pdf.Save()
        plt_cfg.TITLE = "Data GNN predictions"
        Models.total_score_dist_from_preds(outputs_data["predictions"], plt_cfg)
        pdf.Save()
    return

def list_to_dict(outputs : list) -> dict:
    """
    Changes a list of dictionaries, to a dictionary of lists
    """
    for i, this_dict in enumerate(outputs):
        if i == 0:
            merged_output = this_dict.copy()
            for key, val in merged_output.items():
                # if val is not None:
                merged_output[key] = [merged_output[key]]
        else:
            for key, val in this_dict.items():
                # if val is not None:
                merged_output[key].append(val)
    # for final_key in merged_output[dtype]:
    #     if final_key == "name":
    #         continue
    #     else:
    #         merged_output[d_type][key] = np.concatenate(
    #             merged_output[d_type][key], axis=0)
    rprint("merged_output")
    rprint(merged_output)
    return merged_output

def concat_output_lists(outputs : dict) -> dict:
    res = outputs.copy()
    for key, val in outputs.items():
        no_none = [v for v in val if v is not None]
        if key == "name":
            continue
        elif len(no_none) != 0:
            res[key] = np.concatenate(no_none, axis=0)
    return res

def save_gnn_preds(output : dict, out : str):
    """
    Save GNN predictions masks to dill file

    Args:
        output (dict): masks
        out (str): output file directory
    """
    os.makedirs(out, exist_ok = True)
    files = output["name"]
    to_save = {}
    for name, res in output.items():
        if name != "name":  # if not looking at the filenames
            if res is None:  # data file with no truth information
                continue
            to_save[name] = {}
            for f, obj in zip(files, res):
                to_save[name].update({f: obj})
    for name in to_save.keys():
        Master.SaveObject(out + f"gnn_{name}.dill", to_save[name])
    return

def _get_graph_info(events, args, sample):
    for file_dict in args["ntuple_files"][sample]:
        if file_dict["file"] == events.filename:
            graph_path = file_dict["graph"]
            norm_path = file_dict["graph_norm"]
            break
    graph_path_params = DataPreparation.create_filepath_dictionary(graph_path)
    return graph_path_params, norm_path

@Master.timer
def get_gnn_results(events, args, is_mc):
    sample = "mc" if is_mc else "data"
    args_c = Utils.args_to_dict(args)
    if "gnn_results" in args_c.keys():
        predictions = args_c["gnn_results"][sample]["predictions"][events.filename]
        ids = args_c["gnn_results"][sample]["ids"][events.filename]
    else:
        warnings.warn(
            "Predicting using GNN, beware this causes hanging combined "
            + "with multi-processing (manual interruption required)")
        graph_path_params, norm_path = _get_graph_info(events, args, sample)
        evt_ids = DataPreparation.make_evt_ids(events)
        gnn_model = Models.load_model_from_file(
            args["gnn_model_path"],
            new_data_folder=graph_path_params["folder_path"],
            new_norm=norm_path)
        predictions, ids = Models.get_data_predictions(
            gnn_model,
            graph_path_params["schema_path"],
            graph_path_params["test_path"])
        # Confirm graphs match, and ordering is the same
        assert np.all(evt_ids == ids)
    return predictions, ids

@Master.timer
def get_truth_regions(events, args):
    args_c = Utils.args_to_dict(args)
    if "gnn_results" in args_c.keys():
        if not ("truth_regions" in args_c["gnn_results"]["mc"].keys()):
            raise Exception("Can't find 'truth_regions' in the stored "
                            + "information.")
        truth_regions = args_c["gnn_results"]["mc"]["truth_regions"][events.filename]
    else:
        warnings.warn(
            "Predicting using GNN, beware this causes hanging combined "
            + "with multi-processing (manual interruption required)")
        graph_path_params, norm_path = _get_graph_info(events, args, sample)
        gnn_model = Models.load_model_from_file(
            args["gnn_model_path"],
            new_data_folder=graph_path_params["folder_path"],
            new_norm=norm_path)
        _, truth_regions = Models.get_predictions(
            gnn_model,
            graph_path_params["schema_path"],
            graph_path_params["test_path"])
    return truth_regions

@Master.timer
def main(args):
    cross_section.PlotStyler.SetPlotStyle(extend_colors = True)
    outdir = args.out + "region_selection/"
    cross_section.os.makedirs(outdir, exist_ok = True)

    output_mc = list_to_dict(
        Processing.ApplicationProcessing(
            ["mc"], outdir, args, run, False, "output_mc",
            # tf prediction silently hangs if trying multiprocessing
            batchless=True)["mc"])

    output_data = None
    if "data" in args.ntuple_files:
        if len(args.ntuple_files["data"]) > 0:
            if args.mc_only is False:
                output_data = list_to_dict(
                    Processing.ApplicationProcessing(
                        ["data"], outdir, args, run, False, "output_data",
                        # tf prediction fails if trying multiprocessing
                        batchless=True)["data"])

    # save masks used in selection
    save_gnn_preds(output_mc, args.out + "predictions_mc/")
    if output_data is not None:
        save_gnn_preds(output_data, args.out + "predictions_data/")

    # output directories
    os.makedirs(outdir + "plots/", exist_ok = True)
    MakeRegionPlots(
        concat_output_lists(output_mc),
        concat_output_lists(output_data),
        outdir + "plots/")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Applies beam particle selection, PFO selection, produces tables and basic plots.",
        formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Config(parser, required = True)
    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Regen(parser)

    parser.add_argument(
        "--mc", dest = "mc_only", action = "store_true",
        help = "Only analyse the MC file.")

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)

    rprint(vars(args))
    main(args)