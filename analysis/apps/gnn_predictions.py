#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:47

Author: Shyam Bhuller

Description: Selection studies for the charge exchange analysis.
"""
import argparse
import os
import copy

from rich import print as rprint
from python.analysis import Master, BeamParticleSelection, EventSelection, PFOSelection, Plots, shower_merging, Processing, Tags, cross_section
from python.gnn import DataPreparation, Models
from apps.cex_analysis_input import BeamPionSelection, args_to_dict

import awkward as ak
import numpy as np
import pandas as pd

def run_get_predictions(i, file, n_events, start, selected_events, args) -> dict:
    if not (start == 0):
        raise NotImplementedError(
            "GNN predictions does not support batch running")
    graph_path_dict = DataPreparation.create_filepath_dictionary(args["graph"])
    if not os.path.exists(graph_path_dict["test_path"]):
        print(f'Graph data: "{args["graph"]}"\n'
              + "does not include test data, skipping...")
        return {file: np.empty((0, 4))} # {file: None}
    events = Master.Data(file, nTuple_type=args["nTuple_type"], target_momentum=args["pmom"])
    if not (n_events == ak.count(events.eventNum)):
        raise NotImplementedError(
            "GNN predictions does not support batch running")
    events = BeamPionSelection(
        events,
        args_to_dict(args),
        (not args["data"]))# is_mc = not is_data
    preds = get_predictions(graph_path_dict, args["gnn_model_path"], events)
    return {file: preds}

def get_predictions(graph_paths_dict, model_path, filtered_events):
    model = Models.load_model_from_file(
        model_path, new_data_folder=graph_paths_dict["folder_path"])
    print(f"Model loaded: {model_path}")
    predictions, ids = Models.get_data_predictions(
        model,
        graph_paths_dict["schema_path"], graph_paths_dict["test_path"])
    event_ids = np.array(
        [filtered_events.run, filtered_events.subRun, filtered_events.eventNum]).T
    if not np.all(ids == event_ids):
        raise ValueError("IDs of the predictions do not "
                         + "match those of the GNN predictions")
    return predictions

# def run_fetch_graphs(i, file, n_events, start, selected_events, args) -> dict:
#     events = Master.Data(file, n_events, start, args["nTuple_type"], args["pmom"])

#     output = {
#         "name" : file,
#         "fiducial" : None,
#         "beam" : None,
#         "null_pfo" : None,
#         "pi" : None,
#         "photon" : None,
#         "loose_pi" : None,
#         "loose_photon" : None,
#         "pi0" : None,
#         "regions" : None
#     }

#     if args["data"] == True:
#         selection_args = "data_arguments"
#     else:
#         selection_args = "mc_arguments"

#     if "beam_selection" in args:
#         print("beam particle selection")

#         if "TrueFiducialCut" in args["beam_selection"]["selections"]:
#             output_fd, table_fd, fd_masks = AnalyseBeamFiducialCut(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args])
#             output["fiducial"] = {"data" : output_fd, "table" : table_fd, "masks" : fd_masks}

#         output_beam, table_beam, beam_masks = AnalyseBeamSelection(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args]) # events are cut after this
#         output["beam"] = {"data" : output_beam, "table" : table_beam, "masks" : beam_masks}

#     if "valid_pfo_selection" in args:
#         print("PFO pre-selection")
#         good_PFO_mask = PFOSelection.GoodShowerSelection(events)
#         good_PFO_cut_table = Master.CutTable.CutHandler(events, tags = None)
#         good_PFO_cut_table.add_mask(good_PFO_mask, "GoodShowerSelection")
#         events.Filter([good_PFO_mask])
#         output["null_pfo"] = {"table" : good_PFO_cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False), "masks" : {"ValidPFOSelection" : good_PFO_mask}}

#     # this selection only needs to be done for shower merging ntuple because the PDSPAnalyser ntuples only store daughter PFOs, not the beam as well.
#     if events.nTuple_type == Master.Ntuple_Type.SHOWER_MERGING:
#         #* beam particle daughter selection 
#         mask = PFOSelection.BeamDaughterCut(events)
#         events.Filter([mask])

#     if "piplus_selection" in args:
#         print("pion selection")
#         output_pip, table_pip = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["piplus_selection"]["selections"], args["piplus_selection"][selection_args])        
#         pip_masks = CreatePFOMasks(events, args["piplus_selection"], selection_args)
#         output["pi"] = {"data" : output_pip, "table" : table_pip, "masks" : pip_masks}

#     if "loose_pion_selection" in args:
#         print("loose pion selection")
#         output_loose_pip, table_loose_pip = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["loose_pion_selection"]["selections"], args["loose_pion_selection"][selection_args])
#         loose_pip_masks = CreatePFOMasks(events, args["loose_pion_selection"], selection_args)
#         output["loose_pi"] = {"data" : output_loose_pip, "table" : table_loose_pip, "masks" : loose_pip_masks}

#     if "loose_photon_selection" in args:
#         print("loose photon selection")
#         output_loose_photon, table_loose_photon = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["loose_photon_selection"]["selections"], args["loose_photon_selection"][selection_args])
#         loose_photon_masks = CreatePFOMasks(events, args["loose_photon_selection"], selection_args)
#         output["loose_photon"] = {"data" : output_loose_photon, "table" : table_loose_photon, "masks" : loose_photon_masks}

#     if "photon_selection" in args:
#         print("photon selection")
#         output_photon, table_photon = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["photon_selection"]["selections"], args["photon_selection"][selection_args])
#         photon_masks = CreatePFOMasks(events, args["photon_selection"], selection_args)
#         output["photon"] = {"data" : output_photon, "table" : table_photon, "masks" : photon_masks}

#         photon_selection_mask = None
#         for m in photon_masks:
#             if photon_selection_mask is None:
#                 photon_selection_mask = photon_masks[m]
#             else:
#                 photon_selection_mask = photon_selection_mask & photon_masks[m]

#         if "pi0_selection" in args:
#             print("pi0 selection")
#             output_pi0, table_pi0 = AnalysePi0Selection(events.Filter(returnCopy = True), args["data"], args["pi0_selection"]["selections"], args["pi0_selection"][selection_args], photon_selection_mask)
#             pi0_masks = CreatePFOMasks(events, args["pi0_selection"], selection_args, {"photon_mask" : photon_selection_mask})
#             output["pi0"] = {"data" : output_pi0, "table" : table_pi0, "masks" : pi0_masks}

#         print("regions")
#         truth_regions, reco_regions = AnalyseRegions(events, photon_selection_mask, args["data"], args["shower_correction"]["correction"], args["shower_correction"]["correction_params"])

#         regions  = {
#             "truth_regions"       : truth_regions,
#             "reco_regions"        : reco_regions
#         }
#         output["regions"] = regions
#     return output

@Master.timer
def main(args):
    if args.gnn_model_path is None:
        raise ValueError('GNN model not supplied, populate the "GNN_DATA": '
                         + '{"model_path": _ } section of the config.')
    
    outdir = args.out + "gnn_data/"
    cross_section.os.makedirs(outdir, exist_ok = True)
    # Ensure we aren't batching (hopefully no memory issues...):
    args.batches = None
    args.events = None
    args.threads = None
    # Separate MC and data to ensure they get separate save files
    predictions_mc = cross_section.ApplicationProcessing(
            ["mc"], # Which set of data to look at
            outdir, # Where to save outputs
            args, # Arguments from config
            run_get_predictions, # function to run
            True, # Automatically merge
            outname="predictions_mc",
            batchless=True # Batching causes Model.predict to fail
        )["mc"] #Default return is a dict indexed by the first argument
    if (args.has_data and (not args.mc_only)):
        predictions_data = cross_section.ApplicationProcessing(
                ["data"], # Which set of data to look at
                outdir, # Where to save outputs
                args, # Arguments from config
                run_get_predictions, # function to run
                True, # Automatically merge
                outname="predictions_data",
                batchless=True # Batching causes Model.predict to fail
            )["data"]



    # shower_merging.SetPlotStyle(extend_colors = True)

    # output_mc = MergeSelectionMasks(MergeOutputs(cross_section.RunProcess(args.ntuple_files["mc"], False, args, run, False)))
    # output_data = None
    # if "data" in args.ntuple_files:
    #     if len(args.ntuple_files["data"]) > 0:
    #         if args.mc_only is False:
    #             output_data = MergeSelectionMasks(MergeOutputs(cross_section.RunProcess(args.ntuple_files["data"], True, args, run, False)))

    # # tables
    # MakeTables(output_mc, args.out + "tables_mc/", "mc")
    # if output_data is not None: MakeTables(output_data, args.out + "tables_data/", "data")

    # # save masks used in selection
    # SaveMasks(output_mc, args.out + "masks_mc/")
    # if output_data is not None: SaveMasks(output_data, args.out + "masks_data/")

    # # output directories
    # os.makedirs(args.out + "plots/", exist_ok = True)

    # # plots
    # if output_mc["fiducial"]:
    #     MakeBeamSelectionPlots(output_mc["fiducial"]["data"], output_data["fiducial"]["data"] if output_data else None, args.out + "plots/", norm = args.norm, book_name = "fiducial")

    # if output_mc["beam"]: #* this is assuming you apply the same cuts as Data and MC (which is implictly assumed for now)
    #     MakeBeamSelectionPlots(output_mc["beam"]["data"], output_data["beam"]["data"] if output_data else None, args.out + "plots/", norm = args.norm, book_name = "beam")

    # for i in ["pi", "photon", "loose_pi", "loose_photon"]:
    #     if output_mc[i]:
    #         MakePFOSelectionPlots(output_mc[i]["data"], output_data[i]["data"] if output_data else None, args.out + "plots/", norm = args.norm, book_name = i)

    # if output_mc["loose_pi"]:
    #     MakePFOSelectionPlotsConsdensed(
    #         output_mc["pi"]["data"],
    #         output_mc["loose_pi"]["data"],
    #         output_data["pi"]["data"] if output_data else None,
    #         output_data["loose_pi"]["data"] if output_data else None,
    #         args.out + "plots/",
    #         norm = args.norm,
    #         book_name = "pi_both"
    #         )

    # if output_mc["loose_photon"]:
    #     MakePFOSelectionPlotsConsdensed(
    #         output_mc["photon"]["data"],
    #         output_mc["loose_photon"]["data"],
    #         output_data["photon"]["data"] if output_data else None,
    #         output_data["loose_photon"]["data"] if output_data else None,
    #         args.out + "plots/",
    #         norm = args.norm,
    #         book_name = "photon_both"
    #         )

    # if output_mc["pi0"]:
    #     MakePi0SelectionPlots(output_mc["pi0"]["data"], output_data["pi0"]["data"] if output_data else None, args.out + "plots/", norm = args.norm, nbins = args.nbins)
    # if output_mc["regions"]:
    #     MakeRegionPlots(output_mc["regions"], output_data["regions"] if output_data else None, args.out + "plots/")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Config(parser, required = True)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Regen(parser)

    parser.add_argument("--mc", dest = "mc_only", action = "store_true", help = "Only analyse the MC file.")

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)

    rprint(vars(args))
    main(args)