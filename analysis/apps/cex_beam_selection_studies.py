#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:47

Author: Shyam Bhuller

Description: Selection studies for the charge exchange analysis.
"""
import argparse
import os

from rich import print as rprint
from particle import Particle
from python.analysis import (
    Master, BeamParticleSelection, EventSelection, PFOSelection, Plots,
    shower_merging, Processing, Tags, cross_section, EnergyTools, Utils,
    Slicing)
import python.analysis.SelectionTools as st

import awkward as ak
import numpy as np
import pandas as pd

x_label = {
    "track_score_all" : "Track score", 
    "TrackScoreCut" : "Track score",
    "NHitsCut" : "Number of hits",
    "PiPlusSelection" : "Median $dE/dX$ / MeV cm^{-1}",
    "BeamParticleDistanceCut" : "$d$ / cm",
    "BeamParticleIPCut" : "$b$ / cm",
    "Chi2ProtonSelection" : "$(\chi^{2}/ndf)_{p}$",
    "PiBeamSelection" : "True particle ID",
    "APA3Cut" : "Beam end position $z$ / cm",
    "TrueFiducialCut" : "True beam end position $z$ / cm",
    "FiducialStart" : "True beam end position $z$ / cm",
    "PandoraTagCut" : "Pandora tag",
    "DxyCut" : "$\delta_{xy}$",
    "DzCut" : "$\delta_{z}$",
    "CosThetaCut" : "$\cos(\\theta)$",
    "MichelScoreCut" : "Michel score",
    "MedianDEdXCut" : "Median $dE/dX$ / MeV cm^{-1}",
    "BeamScraperCut" : "$r_{inst}$",
    "TrackLengthSelection" : "l / cm"
}
y_scale = {
    "track_score_all" : "log",
    "TrackScoreCut" : "linear",
    "NHitsCut" : "linear",
    "PiPlusSelection" : "log",
    "BeamParticleDistanceCut" : "linear",
    "BeamParticleIPCut" : "linear",
    "Chi2ProtonSelection" : "linear",
    "PiBeamSelection" : None,
    "APA3Cut" : "log",
    "TrueFiducialCut" : "linear",
    "FiducialStart" : "linear",
    "PandoraTagCut" : None,
    "DxyCut" : "log",
    "DzCut" : "log",
    "CosThetaCut" : "log",
    "MichelScoreCut" : "log",
    "MedianDEdXCut" : "log",
    "BeamScraperCut" : "linear",
    "TrackLengthSelection" : "linear"
}
x_range = {
    "track_score_all" : None,
    "TrackScoreCut" : [0, 1],
    "NHitsCut" : [0, 500],
    "PiPlusSelection" : [0, 10],
    "BeamParticleDistanceCut" : [0, 150],
    "BeamParticleIPCut" : [0, 100],
    "Chi2ProtonSelection" : [0, 600],
    "PiBeamSelection" : None,
    "PandoraTagCut" : None,
    "APA3Cut" : [0, 700],
    "TrueFiducialCut" : [0, 700],
    "FiducialStart" : [0, 700],
    "DxyCut" : [0, 5],
    "DzCut" : [-5, 5],
    "CosThetaCut" : [0.9, 1],
    "MichelScoreCut" : [0, 1],
    "MedianDEdXCut" : [1.5, 3],
    "BeamScraperCut" : [0, 5],
    "TrackLengthSelection" : [0, 400],

}
nbins = {
    "track_score_all" : 50,
    "TrackScoreCut" : 50,
    "NHitsCut" : 50,
    "PiPlusSelection" : 50,
    "BeamParticleDistanceCut" : 50,
    "BeamParticleIPCut" : 50,
    "Chi2ProtonSelection" : 50,
    "PiBeamSelection" : 50,
    "APA3Cut" : 50,
    "TrueFiducialCut" : 50,
    "FiducialStart" : 50,
    "PandoraTagCut" : 50,
    "DxyCut" : 50,
    "DzCut" : 50,
    "CosThetaCut" : 50,
    "MichelScoreCut" : 50,
    "MedianDEdXCut" : 50,
    "BeamScraperCut" : 50,
    "TrackLengthSelection" : 50
}
ncols = {
    "track_score_all" : 2,
    "TrackScoreCut" : 2,
    "NHitsCut" : 2,
    "PiPlusSelection" : 2,
    "BeamParticleDistanceCut" : 2,
    "BeamParticleIPCut" : 2,
    "Chi2ProtonSelection" : 2,
    "PiBeamSelection" : 2,
    "APA3Cut" : 2,
    "TrueFiducialCut" : 2,
    "FiducialStart" : 2,
    "PandoraTagCut" : 2,
    "DxyCut" : 2,
    "DzCut" : 2,
    "CosThetaCut" : 2,
    "MichelScoreCut" : 2,
    "MedianDEdXCut" : 2,
    "BeamScraperCut" : 2,
    "TrackLengthSelection" : 2
}
truncate = {
    "track_score_all" : True,
    "TrackScoreCut" : True,
    "NHitsCut" : False,
    "PiPlusSelection" : False,
    "BeamParticleDistanceCut" : True,
    "BeamParticleIPCut" : True,
    "Chi2ProtonSelection" : True,
    "PiBeamSelection" : False,
    "APA3Cut" : False,
    "TrueFiducialCut" : False,
    "FiducialStart" : False,
    "PandoraTagCut" : False,
    "DxyCut" : False,
    "DzCut" : False,
    "CosThetaCut" : False,
    "MichelScoreCut" : False,
    "MedianDEdXCut" : False,
    "BeamScraperCut" : False,
    "BeamScraperCut" : False,
    "TrackLengthSelection" : False
}

def loop_through_masks(
        counts, binner, mask_dict,
        init_e, end_e, in_tpc, true_beam=None,
        proc_mask=slice(None)):
    curr_mask = ak.ones_like(
        list(mask_dict.values())[0], dtype=bool)
    for m_name, mask in mask_dict.items():
        curr_mask = np.logical_and(curr_mask, mask)
        if true_beam is not None:
            true_arg = (true_beam[curr_mask[proc_mask]],)
        else:
            true_arg = ()
        new_counts = binner.energies_to_multi_dim_hist(
            init_e[curr_mask[proc_mask]], end_e[curr_mask[proc_mask]],
            in_tpc[curr_mask[proc_mask]], *true_arg)
        counts.update({m_name: new_counts})
    return counts, curr_mask

def get_multi_dim_counts_reco(
        init_energies, end_energies, in_tpc,
        multi_dim_bins, args, outputs,
        proc_mask=None):
    if proc_mask is None:
        proc_mask = np.ones_like(init_energies, dtype=bool)
    init_energies = init_energies[proc_mask]
    end_energies = end_energies[proc_mask]
    in_tpc = in_tpc[proc_mask]
    fid_masks = outputs["fiducial"]
    beam_masks = outputs["beam"]
    pfo_masks = outputs["null_pfo"]
    if fid_masks is not None:
        fid_mask = ak.ones_like(
            list(fid_masks["masks"].values())[0], dtype=bool)
        for m in fid_masks["masks"].values():
            fid_mask = np.logical_and(fid_mask, m)
        fid_lab = "Fiducial"
    else:
        fid_mask = ak.ones_like(
            list(beam_masks["masks"].values())[0], dtype=bool)
        fid_lab = "NoSelection"
    fid_counts = multi_dim_bins.energies_to_multi_dim_hist(
        init_energies[fid_mask[proc_mask]],
        end_energies[fid_mask[proc_mask]], in_tpc[fid_mask[proc_mask]])
    counts = {fid_lab: fid_counts}
    counts, beam_mask = loop_through_masks(
        counts, multi_dim_bins, beam_masks["masks"],
        init_energies[fid_mask[proc_mask]],
        end_energies[fid_mask[proc_mask]],
        in_tpc[fid_mask[proc_mask]], proc_mask=proc_mask[fid_mask])
    # counts, _ = loop_through_masks(
    #     counts, multi_dim_bins, pfo_masks["masks"],
    #     init_energies[fid_mask][beam_mask], end_energies[fid_mask][beam_mask],
    #     in_tpc[fid_mask][beam_mask])
    # curr_mask = ak.ones_like(
    #     list(beam_masks["masks"].values())[0], dtype=bool)
    # for m_name, mask in beam_masks["masks"].items():
    #     curr_mask = np.logical_and(curr_mask, mask)
    #     new_counts = multi_dim_bins.energies_to_multi_dim_hist(
    #         init_energies[fid_mask][curr_mask],
    #         end_energies[fid_mask][curr_mask],
    #         in_tpc[fid_mask][curr_mask])
    #     counts.update({m_name: new_counts})
    # beam_mask = curr_mask
    # curr_mask = ak.ones_like(
    #     list(pfo_masks["masks"].values())[0], dtype=bool)
    # for m_name, mask in pfo_masks["masks"].items():
    #     curr_mask = np.logical_and(curr_mask, mask)
    #     new_counts = multi_dim_bins.energies_to_multi_dim_hist(
    #         init_energies[fid_mask][beam_mask][curr_mask],
    #         end_energies[fid_mask][beam_mask][curr_mask],
    #         in_tpc[fid_mask][beam_mask][curr_mask])
    #     counts.update({m_name: new_counts})
    return counts

def get_multi_dim_counts_truth(
        init_energies, end_energies, in_tpc, true_pion_mask,
        multi_dim_bins, args, outputs, fiducial_truth_mask,
        proc_mask=None):
    if proc_mask is None:
        proc_mask = np.ones_like(init_energies, dtype=bool)
    init_energies = init_energies[proc_mask]
    end_energies = end_energies[proc_mask]
    in_tpc = in_tpc[proc_mask]
    true_pion_mask = true_pion_mask[proc_mask]
    fiducial_truth_mask = fiducial_truth_mask[proc_mask]
    pre_cut_lab = "FiducialTruth"
    pre_cut_counts = multi_dim_bins.energies_to_multi_dim_hist(
        init_energies[fiducial_truth_mask],
        end_energies[fiducial_truth_mask],
        in_tpc[fiducial_truth_mask],
        true_pion_mask[fiducial_truth_mask])
    counts = {pre_cut_lab: pre_cut_counts}
    fid_masks = outputs["fiducial"]
    beam_masks = outputs["beam"]
    pfo_masks = outputs["null_pfo"]
    counts, fid_mask = loop_through_masks(
        counts, multi_dim_bins, fid_masks["masks"],
        init_energies, end_energies,
        in_tpc, true_pion_mask, proc_mask=proc_mask)
    counts, beam_mask = loop_through_masks(
        counts, multi_dim_bins, beam_masks["masks"],
        init_energies[fid_mask[proc_mask]],
        end_energies[fid_mask[proc_mask]], in_tpc[fid_mask[proc_mask]],
        true_pion_mask[fid_mask[proc_mask]],
        proc_mask=proc_mask[fid_mask])
    # counts, _ = loop_through_masks(
    #     counts, multi_dim_bins, pfo_masks["masks"],
    #     init_energies[fid_mask][beam_mask], end_energies[fid_mask][beam_mask],
    #     in_tpc[fid_mask][beam_mask], true_pion_mask[fid_mask][beam_mask])
    return counts

# def sum_dict_masks(dictionary):
#     return {k: ak.sum(v) for k, v in dictionary.items()}

# def get_process_counts(events, args_dict, true_pion_beam=True):
#     assert len(args_dict["gnn_region_labels"]) == 3, "Not implemented for arbitrary regions"
#     proc_masks = EventSelection.create_3_regions_from_evts(events)
#     if true_pion_beam:
#         true_pion_mask = events.trueParticles.pdg[..., 0] == 211
#         proc_masks = {k: np.logical_and(true_pion_mask, v) for k, v in proc_masks.items()}
#     return sum_dict_masks(proc_masks)

# def get_proc_counts_reco(
#         events, args, outputs):
#     fid_masks = outputs["fiducial"]
#     beam_masks = outputs["beam"]
#     # pfo_masks = outputs["null_pfo"]
#     proc_counts = get_process_counts(events, args, true_pion_beam=False)
#     if fid_masks is not None:
#         fid_mask = ak.ones_like(
#             list(fid_masks["masks"].values())[0], dtype=bool)
#         for m in fid_masks["masks"].values():
#             fid_mask = np.logical_and(fid_mask, m)
#         fid_lab = "Fiducial"
#     else:
#         fid_mask = ak.ones_like(
#             list(beam_masks["masks"].values())[0], dtype=bool)
#         fid_lab = "NoSelection"
#     proc_counts = {k: v[fid_mask] for k, v in proc_counts.items()}
#     fid_counts = sum_dict_masks(proc_counts)
#     counts = {fid_lab: fid_counts}
#     curr_mask = ak.ones_like(
#         list(mask_dict.values())[0], dtype=bool)
#     for m_name, mask in mask_dict.items():
#         curr_mask = np.logical_and(curr_mask, mask)
#         counts.update(sum_dict_masks({k: np.logical_and(curr_mask, v)
#                                       for k, v in proc_counts.items()}))
#     return counts

def AnalyseBeamFiducialCut(events : Master.Data, beam_instrumentation : bool, functions : dict, args : dict):
    output = {}
    masks = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueBeamParticleTags(events) if beam_instrumentation is False else None)
    output["no_selection"] = st.MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))
    fiducial_cuts = ["TrueFiducialCut", "FiducialStart", "APA3Cut"]
    masks = {k : functions[k](events, **v) for k, v in args.items() if k in fiducial_cuts}
    if len(masks) != 0:
    # if "TrueFiducialCut" in args:
    #     masks = {k : functions[k](events, **v) for k, v in args.items() if k in ["APA3Cut", "TrueFiducialCut"]}
        # for a in ["APA3Cut", "TrueFiducialCut"]:
        for a in masks.keys():
            mask, property = functions[a](events, **args[a], return_property = True)
            cut_table.add_mask(mask, a)
            cut_values = args[a]["cut"] if "cut" in args[a] else None
            operations = args[a]["op"] if "op" in args[a] else None
            output[a] = st.MakeOutput(property, Tags.GenerateTrueBeamParticleTags(events), cut_values, operations)

            events.Filter([mask], [mask])
            output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* true particle population
    output["final_tags"] = st.MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))

    df = cut_table.get_table(init_data_name = "no selection", pfos = False, percent_remain = False, relative_percent = False, ave_per_event = False)

    return output, df, masks


def AnalyseBeamSelection(events : Master.Data, beam_instrumentation : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame, dict]:
    """ Manually applies the beam selection while storing the value being cut on, cut values and truth tags in order to do plotting
        and produce performance tables.

    Args:
        events (Master.Data): events to look at
        beam_instrumentation (bool): use beam instrumentation for beam particle selection
        beam_quality_fits (dict): fit values for the beam quality selection

    Returns:
        dict: output data.
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueBeamParticleTags(events) if beam_instrumentation is False else None)

    output["no_selection"] = st.MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))
    fiducial_cuts = ["TrueFiducialCut", "FiducialStart"]
    masks = {k : functions[k](events, **v) for k, v in args.items() if k not in fiducial_cuts}

    #* pi+ beam selection
    mask = BeamParticleSelection.PiBeamSelection(events, **args["PiBeamSelection"])
    cut_table.add_mask(mask, "PiBeamSelection")
    counts = Tags.GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i].mask)
    output["PiBeamSelection"] = st.MakeOutput(counts, Tags.GenerateTrueBeamParticleTags(events), None, None)
    events.Filter([mask], [mask])
    output["PiBeamSelection"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    for a in args:
        if a in ["PiBeamSelection"] + fiducial_cuts: continue
        if ("TrueFiducialCut" in args) and (a == "APA3Cut"):
            # should have applied fiducial cuts first
            del masks["APA3Cut"]
            continue 
        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = st.MakeOutput(property, Tags.GenerateTrueBeamParticleTags(events), cut_values, operations)

        if (a == "BeamScraperCut") and (beam_instrumentation == False):
            if beam_instrumentation == False:
                scraper_id = EnergyTools.IsScraper(events, args[a]["fits"])
                output[a + "_scraper_tag"] = st.MakeOutput(None, Tags.BeamScraperTag(scraper_id), None, None)

        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* true particle population
    output["final_tags"] = st.MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))

    df = cut_table.get_table(init_data_name = "no selection", pfos = False, percent_remain = False, relative_percent = False, ave_per_event = False)
    return output, df, masks

def AnalyseSelectionEfficiency(efficiency_counts):
    init_name, init_count = list(efficiency_counts.items())[0]
    valid_starts = ["FiducialTruth", "Fiducial", "NoSelection"]
    if "FiducialTruth" in efficiency_counts.keys():
        assert init_name == "FiducialTruth", f"Must have fiducial truth cut first, found {init_name}"
    assert init_name in valid_starts, f"Invalid start count: {init_name}"
    return {name : count/init_count for name, count in efficiency_counts.items()}

def run(i, file, n_events, start, selected_events, args) -> dict:
    events = Master.Data(file, n_events, start, args["nTuple_type"], args["pmom"])

    output = {
        "name" : file,
        "fiducial" : None,
        "beam" : None,
        "null_pfo" : None,
        "efficiencies": None
    }

    if args["data"] == True:
        selection_args = "data_arguments"
    else:
        selection_args = "mc_arguments"
        truth_multi_bins = Slicing.MultiDimBins(
            args["energy_slices"].bin_edges_with_overflow, True, True)
        truth_init_e = EnergyTools.TrueInitialEnergyFiducial(
            events, args["fiducial_volume"]["start"])
        truth_end_e, truth_in_tpc = EnergyTools.TrueEndEnergyFiducial(
            events, args["fiducial_volume"]["end"])
        true_pion_mask = events.trueParticles.pdg[..., 0] == 211
        truth_fiducial_mask = BeamParticleSelection.TrueFiducialCut(
            events, True, cut=args["fiducial_volume"]["start"], op=">")
        assert len(args["gnn_region_labels"]) == 3, "Not implemented process efficiencies for arbitrary regions"
        truth_masks = EventSelection.create_3_regions_from_evts(events)

    reco_multi_bins = Slicing.MultiDimBins(
        args["energy_slices"].bin_edges_with_overflow, True, False)
    reco_init_e = EnergyTools.BetheBloch.InteractingKE(
                EnergyTools.KE(events.recoParticles.beam_inst_P,
                               Particle.from_pdgid(211).mass),
                (args["fiducial_volume"]["start"]
                 * np.ones_like(events.recoParticles.beam_inst_P)),
                25)
    reco_end_e = reco_init_e - EnergyTools.RecoDepositedEnergyFiducial(
            events, reco_init_e, args["fiducial_volume"]["end"])
    reco_in_tpc = (events.recoParticles.beam_endPos_SCE.z
                   < args["fiducial_volume"]["end"])

    if "beam_selection" in args:
        print("beam particle selection")

        if (
                ("TrueFiducialCut" in args["beam_selection"]["selections"])
                or ("FiducialStart" in args["beam_selection"]["selections"])):
            output_fd, table_fd, fd_masks = AnalyseBeamFiducialCut(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args])
            output["fiducial"] = {"data" : output_fd, "table" : table_fd, "masks" : fd_masks}

        output_beam, table_beam, beam_masks = AnalyseBeamSelection(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args]) # events are cut after this
        output["beam"] = {"data" : output_beam, "table" : table_beam, "masks" : beam_masks}

    if "valid_pfo_selection" in args:
        print("PFO pre-selection (valid PFO cuts)")
        good_PFO_mask = PFOSelection.GoodShowerSelection(events)
        good_PFO_cut_table = Master.CutTable.CutHandler(events, tags = None)
        good_PFO_cut_table.add_mask(good_PFO_mask, "GoodShowerSelection")
        events.Filter([good_PFO_mask])
        output["null_pfo"] = {"table" : good_PFO_cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False), "masks" : {"ValidPFOSelection" : good_PFO_mask}}
    # print({name: len(mask) for name, mask in output["fiducial"]["masks"].items()})
    # print({name: len(mask) for name, mask in output["beam"]["masks"].items()})
    # print({name: len(mask) for name, mask in output["null_pfo"]["masks"].items()})
    # selection_masks = output["beam"]["masks"].copy()
    # selection_masks.update(output["null_pfo"]["masks"])
    reco_efficiency_counts = get_multi_dim_counts_reco(
        reco_init_e, reco_end_e, reco_in_tpc,
        reco_multi_bins, args, output)
    efficiencies = {"reco": {
        "count": reco_efficiency_counts,
        "efficiency": AnalyseSelectionEfficiency(reco_efficiency_counts)}}
    if not args["data"]:
        truth_efficiency_counts = get_multi_dim_counts_truth(
            truth_init_e, truth_end_e, truth_in_tpc, true_pion_mask,
            truth_multi_bins, args, output, truth_fiducial_mask)
        efficiencies.update({"truth": {
            "count": truth_efficiency_counts,
            "efficiency": AnalyseSelectionEfficiency(truth_efficiency_counts)}})
        proc_info = {}
        for p, m in truth_masks.items():
            true_pure_proc_counts = get_multi_dim_counts_truth(
                truth_init_e, truth_end_e, truth_in_tpc,
                true_pion_mask, truth_multi_bins, args, output,
                truth_fiducial_mask, proc_mask=m)
            true_all_proc_counts = get_multi_dim_counts_truth(
                truth_init_e, truth_end_e, truth_in_tpc,
                ak.ones_like(true_pion_mask, dtype=bool),
                truth_multi_bins, args, output, truth_fiducial_mask,
                proc_mask=m)
            reco_proc_counts = get_multi_dim_counts_reco(
                reco_init_e, reco_end_e, reco_in_tpc,
                reco_multi_bins, args, output, proc_mask=m)
            proc_info.update({p: {
                "truth_pure_count": true_pure_proc_counts,
                "truth_all_count": true_all_proc_counts,
                "reco_count": reco_proc_counts}})
        efficiencies.update({"process": proc_info})
    output["efficiencies"] = efficiencies
    return output

@Master.timer
def BeamPionSelection(
        events : Master.Data,
        args : argparse.Namespace | dict, is_mc : bool) -> Master.Data:
    """
    Apply beam pion selection to ntuples. Loads stored masks if
    present (cex_beam_selection_studies already run), else applies the
    fiducial, beam, and valid PFO selections without generating plots
    or saving masks.

    Args:
        events (Master.Data): analysis ntuple
        args (argparse.Namespace): analysis configuration
        is_mc (bool): is the ntuple mc or data?

    Returns:
        Master.Data: selected events.
    """
    args_c = Utils.args_to_dict(args)

    events_copy = events.Filter(returnCopy = True)
    if is_mc:
        selection_args = "mc_arguments"
        sample = "mc"
    else:
        selection_args = "data_arguments"
        sample = "data"

    if "beam_selection_masks" in args_c:
        masks = args_c["beam_selection_masks"][sample]
        if ("fiducial" in masks) and (len(masks["fiducial"]) > 0):
            mask = st.CombineMasks(masks["fiducial"][events.filename])
            events_copy.Filter([mask], [mask])
        mask = st.CombineMasks(masks["beam"][events.filename])
        events_copy.Filter([mask], [mask])
    else:
        for s in args_c["beam_selection"]["selections"]:
            mask = args_c["beam_selection"]["selections"][s](events_copy, **args_c["beam_selection"][selection_args][s])
            events_copy.Filter([mask], [mask])
            print(events_copy.cutTable.get_table())

    if "valid_pfo_selection" in args_c:
        if args_c["valid_pfo_selection"] is True:
            if "beam_selection_masks" in args:
                events_copy.Filter([args_c["beam_selection_masks"][sample]['null_pfo'][events.filename]['ValidPFOSelection']]) # apply PFO preselection here
            else:
                events_copy.Filter(PFOSelection.GoodShowerSelection(events))
    return events_copy

def MakeBeamSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, book_name = "beam"):
    """ Beam particle selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """
    norm = False if output_data is None else norm
    with Plots.PlotBook(outDir + f"{book_name}.pdf") as pdf:

        for p in output_mc:
            if p in x_label:
                if p == "PiBeamSelection":
                    Plots.PlotTags(output_mc["PiBeamSelection"]["tags"], "True particle ID")
                    pdf.Save()
                elif p == "PandoraTagCut":
                    if output_data is None:
                        Plots.PlotBar(output_mc[p]["value"], xlabel = x_label[p])
                    else:
                        Plots.PlotBarComparision(output_mc[p]["value"], output_data[p]["value"], label_1 = "MC", label_2 = "Data", xlabel = x_label[p], fraction = True, barlabel = True)
                    pdf.Save()

                elif p == "APA3Cut":
                    for l in ["linear", "log"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = l, x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = st.CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "TrueFiducialCut":
                    for l in ["linear", "log"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = None, norm = norm, y_scale = l, x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = st.CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "BeamScraperCut":
                    for t in ["BeamScraperCut", "BeamScraperCut_scraper_tag"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[t]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = st.CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                else:
                    Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                    if p == "CosThetaCut":
                        arrow_l = 0.02
                    else:
                        arrow_l = st.CalculateArrowLength(output_mc[p]["value"], x_range[p])
                    Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = arrow_l, arrow_loc = 0.7, color = "k")
                    pdf.Save()

        Plots.PlotTags(output_mc["final_tags"]["tags"], "True particle ID")
        pdf.Save()
    Plots.plt.close("all")
    return

def MakeEfficiencyPlots(output_mc : dict, output_data : dict, outDir : str, args, book_name = "sel_efficiency"):
    """ Beam particle selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """
    overflow_bins = args.energy_slices.bin_edges_with_overflow
    truth_multi_binner = Slicing.MultiDimBins(
        overflow_bins, True, True)
    reco_multi_binner = Slicing.MultiDimBins(
        overflow_bins, True, False)
    count_max = max([np.max(v) for v in output_mc["truth"]["count"].values()])
    with Plots.PlotBook(outDir + f"{book_name}.pdf") as pdf:
        Plots.multi_dim_2d_plot(
            output_mc["truth"]["count"]["FiducialTruth"], overflow_bins, truth_multi_binner,
            title="Truth fiducial MC truth count", norm="log",
            vmax=count_max, vmin=1, pdf=pdf)
        # pdf.Save()
        Plots.multi_dim_2d_plot(
            output_mc["truth"]["efficiency"]["FiducialTruth"], overflow_bins, truth_multi_binner,
            title="Truth fiducial MC truth efficiency", norm="linear",
            vmax=2, vmin=0, pdf=pdf, cmap="coolwarm")
        # pdf.Save()
        for p in output_mc["reco"]["count"]:
            if "Fiducial" not in p:
                Plots.multi_dim_2d_plot(
                    output_mc["truth"]["count"][p], overflow_bins, truth_multi_binner,
                    title=p + " MC truth count", norm="log",
                    vmax=count_max, vmin=1, pdf=pdf)
                # pdf.Save()
                Plots.multi_dim_2d_plot(
                    output_mc["truth"]["efficiency"][p], overflow_bins, truth_multi_binner,
                    title=p + " MC truth efficiency", norm="linear",
                    vmax=2, vmin=0, pdf=pdf, cmap="coolwarm")
                # pdf.Save()
            Plots.multi_dim_2d_plot(
                output_mc["reco"]["count"][p], overflow_bins, reco_multi_binner,
                title=p + " MC reco count", norm="log",
                vmax=count_max, vmin=1, pdf=pdf)
            # pdf.Save()
            Plots.multi_dim_2d_plot(
                output_mc["reco"]["efficiency"][p], overflow_bins, reco_multi_binner,
                title=p + " MC reco efficiency", norm="linear",
                vmax=1, vmin=0, pdf=pdf)
            # pdf.Save()
            if output_data is not None:
                Plots.multi_dim_2d_plot(
                    output_data["reco"]["count"][p], overflow_bins, reco_multi_binner,
                    title=p + " data reco count", norm="log",
                    vmax=count_max, vmin=1, pdf=pdf)
                # pdf.Save()
                Plots.multi_dim_2d_plot(
                    output_data["reco"]["efficiency"][p], overflow_bins, reco_multi_binner,
                    title=p + " data reco efficiency", norm="linear",
                    vmax=1, vmin=0, pdf=pdf)
                # pdf.Save()
    Plots.plt.close("all")
    return

@Master.timer
def main(args):
    cross_section.PlotStyler.SetPlotStyle(extend_colors = True)
    outdir = args.out + "beam_selection/"
    cross_section.os.makedirs(outdir, exist_ok = True)

    output_mc = st.MergeSelectionMasks(st.MergeOutputs(Processing.ApplicationProcessing(["mc"], outdir, args, run, False, "output_mc")["mc"]))

    output_data = None
    if "data" in args.ntuple_files:
        if len(args.ntuple_files["data"]) > 0:
            if args.mc_only is False:
                output_data = st.MergeSelectionMasks(st.MergeOutputs(Processing.ApplicationProcessing(["data"], outdir, args, run, False, "output_data")["data"]))

    # tables
    st.MakeTables(output_mc, args.out + "tables_mc/", "mc")
    if output_data is not None: st.MakeTables(output_data, args.out + "tables_data/", "data")

    # save masks used in selection
    st.SaveMasks(output_mc, args.out + "masks_mc/")
    if output_data is not None: st.SaveMasks(output_data, args.out + "masks_data/")

    st.SaveEfficiencies(output_mc, args.out + "efficiency_mc/")

    # output directories
    os.makedirs(outdir + "plots/", exist_ok = True)

    # plots
    if output_mc["fiducial"]:
        MakeBeamSelectionPlots(output_mc["fiducial"]["data"], output_data["fiducial"]["data"] if output_data else None, outdir + "plots/", norm = args.norm, book_name = "fiducial")

    if output_mc["beam"]: #* this is assuming you apply the same cuts as Data and MC (which is implictly assumed for now)
        MakeBeamSelectionPlots(output_mc["beam"]["data"], output_data["beam"]["data"] if output_data else None, outdir + "plots/", norm = args.norm, book_name = "beam")
    
    MakeEfficiencyPlots(output_mc["efficiencies"], output_data["efficiencies"] if output_data else None, outdir + "plots/", args, book_name = "sel_efficiency")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Config(parser, required = True)
    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Regen(parser)

    parser.add_argument("--mc", dest = "mc_only", action = "store_true", help = "Only analyse the MC file.")

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)

    rprint(vars(args))
    main(args)