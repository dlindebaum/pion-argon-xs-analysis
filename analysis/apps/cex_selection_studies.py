#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:47

Author: Shyam Bhuller

Description: Selection studies for the charge exchange analysis.
"""
import argparse
import json
import os

from rich import print as rprint
from python.analysis import Master, BeamParticleSelection, EventSelection, PFOSelection, Plots, shower_merging, Processing, Tags, cross_section

import awkward as ak
import numpy as np
import pandas as pd


def MakeOutput(value : ak.Array, tags : Tags.Tags, cuts : list = [], fs_tags : Tags.Tags = None) -> dict:
    """ Output dictionary for multiprocessing class, if we want to be fancy this can be a dataclass instead.
        Generally the data stored in this dictionary should be before any cuts are applied, for plotting purposes.

    Args:
        value (ak.Array): value which is being studied
        tags (Tags.Tags): tags which categorise the value
        cuts (list, optional): cut values used on this data. Defaults to [].
        fs_tags (Tags.Tags, optional): final state tags to the data. Defaults to None.

    Returns:
        dict: dictionary of data.
    """
    return {"value" : value, "tags" : tags, "cuts" : cuts, "fs_tags" : fs_tags}


def AnalyseBeamSelection(events : Master.Data, beam_instrumentation : bool, beam_quality_fits : dict, functions : dict, args : dict) -> tuple[dict, pd.DataFrame, dict]:
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
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueBeamParticleTags(events))

    output["no_selection"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, EventSelection.GenerateTrueFinalStateTags(events))

    #* pi+ beam selection
    mask = BeamParticleSelection.PiBeamSelection(events, beam_instrumentation)
    cut_table.add_mask(mask, "PiBeamSelection")
    counts = Tags.GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i].mask)
    output["PiBeamSelection"] = MakeOutput(counts, Tags.GenerateTrueBeamParticleTags(events), None)
    events.Filter([mask], [mask])
    output["PiBeamSelection"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    for a in args:
        if a == "PiBeamSelection": continue
        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        output[a] = MakeOutput(property, Tags.GenerateTrueBeamParticleTags(events), cut_values)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)    


    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, EventSelection.GenerateTrueFinalStateTags(events))

    df = cut_table.get_table(init_data_name = "no selection", pfos = False, percent_remain = False, relative_percent = False, ave_per_event = False)
    print(df)
    return output, df, cut_table.get_masks_dict()


def AnalysePiPlusSelection(events : Master.Data, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse the daughter pi+ selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTagsPiPlus(events))

    # this selection only needs to be done for shower merging ntuple because the PDSPAnalyser ntuples only store daughter PFOs, not the beam as well.
    if events.nTuple_type == Master.Ntuple_Type.SHOWER_MERGING:
        #* beam particle daughter selection 
        mask = PFOSelection.BeamDaughterCut(events)
        cut_table.add_mask(mask, "track_score_all")
        output["track_score_all"] = MakeOutput(events.recoParticles.trackScore, Tags.GenerateTrueParticleTagsPiPlus(events)) # keep a record of the track score to show the cosmic muon background
        events.Filter([mask])

    for a in args:
        if a == "NHitsCut":
            output[a + "_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [], [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        output[a] = MakeOutput(property, Tags.GenerateTrueParticleTagsPiPlus(events), cut_values)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueParticleTagsPiPlus(events), None)
    
    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePhotonCandidateSelection(events : Master.Data, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse the photon candidate selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTagsPi0Shower(events))

    for a in args:
        if a in ["NHitsCut", "BeamParticleIPCut"]:
            output[a + "_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [], [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        output[a] = MakeOutput(property, Tags.GenerateTrueParticleTagsPi0Shower(events), cut_values)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueParticleTagsPi0Shower(events), None)

    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePi0Selection(events : Master.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse the pi0 selection.

    Args:
        events (Master.Data): events to look at
        data (bool): is the ntuple file data or MC
        correction (callable, optional): shower energy correction. Defaults to None.
        correction_params (dict, optional): shower energy correction parameters. Defaults to None.

    Returns:
        dict: output data
    """    
    def null_tag():
        tag = Tags.Tags()
        tag["null"] = Tags.Tag(mask = (events.eventNum < -1))
        return tag

    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = EventSelection.GenerateTrueFinalStateTags(events))

    photonCandidates = PFOSelection.InitialPi0PhotonSelection(events) # repeat the photon candidate selection, but only require the mask

    for a in args:
        if a == "Pi0MassSelection":
            mask, property = functions[a](events, **args[a], return_property = True, photon_mask = photonCandidates)
            output["mass_event_tag"] = MakeOutput(None, EventSelection.GenerateTrueFinalStateTags(events), None)
        else:
            mask, property = functions[a](events, **args[a], return_property = True, photon_mask = photonCandidates)
        if a != "NPhotonCandidateSelection":
            mask = ak.flatten(mask)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
        output[a] = MakeOutput(property, tags, cut_values)
        events.Filter([mask], [mask])
        photonCandidates = photonCandidates[mask]
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* final counts
    output["event_tag"] = MakeOutput(None, EventSelection.GenerateTrueFinalStateTags(events), None)
    tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["final_tags"] = MakeOutput(None, tags, None) 

    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, pfos = False)
    print(df)
    return output, df


def AnalyseRegions(events : Master.Data, photon_mask : ak.Array, is_data : bool, correction : callable = None, correction_params : dict = None) -> tuple[dict, dict]:
    """ Create masks which desribe the truth and reco regions for various exlusive cross sections.

    Args:
        events (Master.Data): events to look at
        photon_mask (ak.Array): mask for pi0 photon shower candidates
        is_data (bool): is this a data ntuple?
        correction (callable, optional): shower energy correction. Defaults to None.
        correction_params (dict, optional): shower energy correction parameters. Defaults to None.

    Returns:
        tuple[dict, dict]: truth and reco regions
    """
    truth_regions = EventSelection.create_regions(events.trueParticles.nPi0, events.trueParticles.nPiPlus) if is_data == False else None

    reco_pi0_counts = EventSelection.count_pi0_candidates(events, exactly_two_photons = True, photon_mask = photon_mask, correction = cross_section.EnergyCorrection.shower_energy_correction[correction], correction_params = correction_params)
    reco_pi_plus_counts_mom_cut = EventSelection.count_charged_pi_candidates(events, energy_cut = None)
    reco_regions = EventSelection.create_regions(reco_pi0_counts, reco_pi_plus_counts_mom_cut)
    return truth_regions, reco_regions


def CreatePFOMasks(mc : Master.Data, selections : dict, args_type : str, extra_args : dict = None):
    masks = {}
    for n, c, v in zip(selections["selections"].keys(), selections["selections"].values(), selections[args_type].values()):
        if extra_args is not None:
            v = {**v, **extra_args}
        masks[n] = c(mc, **v)
    return masks

# @Processing.log_process
def run(i, file, n_events, start, selected_events, args) -> dict:
    events = Master.Data(file, nEvents = n_events, start = start, nTuple_type = args["ntuple_type"]) # load data

    if args["data"] == True:
        beam_quality_fit_values = args["data_beam_quality_fit"]
        selection_args = "data_arguments"
    else:
        beam_quality_fit_values = args["mc_beam_quality_fit"]
        selection_args = "mc_arguments"

    #* shower energy correction
    if args["correction_params"] is not None:
        with open(args["correction_params"], "r") as f:
            correction_params = json.load(f)
    else:
        correction_params = None

    print("beam particle selection")
    output_beam, table_beam, beam_masks = AnalyseBeamSelection(events, args["data"], beam_quality_fit_values, args["beam_selection"]["selections"], args["beam_selection"][selection_args]) # events are cut after this

    print("PFO pre-selection")
    good_PFO_mask = PFOSelection.GoodShowerSelection(events)
    events.Filter([good_PFO_mask])

    print("pion selection")
    output_pip, table_pip = AnalysePiPlusSelection(events.Filter(returnCopy = True), args["piplus_selection"]["selections"], args["piplus_selection"][selection_args]) # pass the PFO selections a copy of the event
    pip_masks = CreatePFOMasks(events, args["piplus_selection"], selection_args)

    print("photon selection")
    output_photon, table_photon = AnalysePhotonCandidateSelection(events.Filter(returnCopy = True), args["photon_selection"]["selections"], args["photon_selection"][selection_args])
    photon_masks = CreatePFOMasks(events, args["photon_selection"], selection_args)

    photon_selection_mask = None
    for m in photon_masks:
        if photon_selection_mask is None:
            photon_selection_mask = photon_masks[m]
        else:
            photon_selection_mask = photon_selection_mask & photon_masks[m]

    print("pi0 selection")
    output_pi0, table_pi0 = AnalysePi0Selection(events.Filter(returnCopy = True), args["data"], args["pi0_selection"]["selections"], args["pi0_selection"][selection_args])

    pi0_masks = CreatePFOMasks(events, args["pi0_selection"], selection_args, {"photon_mask" : photon_selection_mask})

    print("regions")
    truth_regions, reco_regions = AnalyseRegions(events, photon_selection_mask, args["data"], args["correction"], correction_params)

    regions  = {
        "truth_regions"       : truth_regions,
        "reco_regions"        : reco_regions
    }

    output = {
        "beam" : {"data" : output_beam, "table" : table_beam, "masks" : beam_masks},
        "null_pfo" : {"masks" : {"ValidPFOSelection" : good_PFO_mask}},
        "pip" : {"data" : output_pip, "table" : table_pip, "masks" : pip_masks},
        "photon" : {"data" : output_photon, "table" : table_photon, "masks" : photon_masks},
        "pi0" : {"data" : output_pi0, "table" : table_pi0, "masks" : pi0_masks},
        "regions" : regions 
    }
    return output


def MakeBeamSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float):
    """ Beam particle selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """

    norm = False if output_data is None else norm
    with Plots.PlotBook(outDir + "beam.pdf") as pdf:

        bar_data = []
        for tag in output_mc["PiBeamSelection"]["value"]:
            bar_data.extend([tag] * output_mc["PiBeamSelection"]["value"][tag])
        Plots.PlotBar(bar_data, xlabel = "True particle ID")
        pdf.Save()

        if output_data is None:
            Plots.PlotBar(output_mc["PandoraTagCut"]["value"], xlabel = "Pandora tag")
        else:
            Plots.PlotBarComparision(output_mc["PandoraTagCut"]["value"], output_data["PandoraTagCut"]["value"], label_1 = "MC", label_2 = "Data", xlabel = "pandora beam tag", fraction = True, barlabel = True)
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["DxyCut"]["value"], output_mc["DxyCut"]["tags"], data2 = output_data["DxyCut"]["value"], bins = args.nbins, x_label = "$\delta_{xy}$", y_scale = "log", x_range = [0, 5], norm = norm)
        else:
            Plots.PlotTagged(output_mc["DxyCut"]["value"], output_mc["DxyCut"]["tags"], bins = args.nbins, x_label = "$\delta_{xy}$", y_scale = "log", x_range = [0, 5], norm = norm)
        Plots.DrawCutPosition(output_mc["DxyCut"]["cuts"], arrow_length = 1, face = "left")
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["DzCut"]["value"], output_mc["DzCut"]["tags"], data2 = output_data["DzCut"]["value"], bins = args.nbins, x_label = "$\delta_{z}$", y_scale = "log", x_range = [-10, 10], norm = norm)
        else:
            Plots.PlotTagged(output_mc["DzCut"]["value"], output_mc["DzCut"]["tags"], bins = args.nbins, x_label = "$\delta_{z}$", y_scale = "log", x_range = [-10, 10], norm = norm)
        Plots.DrawCutPosition(min(output_mc["DzCut"]["cuts"]), arrow_length = 1, face = "right")
        Plots.DrawCutPosition(max(output_mc["DzCut"]["cuts"]), arrow_length = 1, face = "left")
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["CosThetaCut"]["value"], output_mc["CosThetaCut"]["tags"], data2 = output_data["CosThetaCut"]["value"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = args.nbins, x_range = [0.9, 1], norm = norm)
        else:
            Plots.PlotTagged(output_mc["CosThetaCut"]["value"], output_mc["CosThetaCut"]["tags"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = args.nbins, x_range = [0.9, 1], norm = norm)

        Plots.DrawCutPosition(output_mc["CosThetaCut"]["cuts"], arrow_length = 0.02)
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["APA3Cut"]["value"], output_mc["APA3Cut"]["tags"], data2 = output_data["APA3Cut"]["value"], bins = args.nbins, x_label = "Beam end position z (cm)", x_range = [0, 700], norm = norm)
        else:
            Plots.PlotTagged(output_mc["APA3Cut"]["value"], output_mc["APA3Cut"]["tags"], bins = args.nbins, x_label = "Beam end position z (cm)", x_range = [0, 700], norm = norm)

        Plots.DrawCutPosition(output_mc["APA3Cut"]["cuts"], face = "left", arrow_length = 50)
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["MichelScoreCut"]["value"], output_mc["MichelScoreCut"]["tags"], data2 = output_data["MichelScoreCut"]["value"], x_range = (0, 1), y_scale = "log", bins = args.nbins, x_label = "Michel score", ncols = 2, norm = norm)
        else:
            Plots.PlotTagged(output_mc["MichelScoreCut"]["value"], output_mc["MichelScoreCut"]["tags"], x_range = (0, 1), y_scale = "log", bins = args.nbins, x_label = "Michel score", ncols = 2, norm = norm)
        Plots.DrawCutPosition(output_mc["MichelScoreCut"]["cuts"], face = "left")
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["MedianDEdXCut"]["value"], output_mc["MedianDEdXCut"]["tags"], data2 = output_data["MedianDEdXCut"]["value"], bins = args.nbins, y_scale = "log", x_range = [0, 10], x_label = "Median $dE/dX$ (MeV/cm)", norm = norm)
        else:
            Plots.PlotTagged(output_mc["MedianDEdXCut"]["value"], output_mc["MedianDEdXCut"]["tags"], bins = args.nbins, y_scale = "log", x_range = [0, 10], x_label = "Median $dE/dX$ (MeV/cm)", norm = norm)
        Plots.DrawCutPosition(output_mc["MedianDEdXCut"]["cuts"], face = "left", arrow_length = 2)
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["BeamScraperCut"]["value"], output_mc["BeamScraperCut"]["tags"], data2 = output_data["BeamScraperCut"]["value"], bins = args.nbins, y_scale = "log", x_range = [0, 10], x_label = "$r_{inst}$ (cm)", norm = norm)
        else:
            Plots.PlotTagged(output_mc["BeamScraperCut"]["value"], output_mc["BeamScraperCut"]["tags"], bins = args.nbins, y_scale = "log", x_range = [0, 10], x_label = "$r_{inst}$ (cm)", norm = norm)
        Plots.DrawCutPosition(output_mc["BeamScraperCut"]["cuts"], face = "left", arrow_length = 2)
        pdf.Save()

        bar_data = []
        for t in output_mc["final_tags"]["tags"]:
            bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
        Plots.PlotBar(bar_data, xlabel = "True particle ID")
        pdf.Save()
    return


def MakePiPlusSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float):
    """ Pi plus selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """

    norm = False if output_data is None else norm
    with Plots.PlotBook(outDir + "piplus.pdf") as pdf:
        if "track_score_all" in output_mc:
            if output_data:
                Plots.PlotTagged(output_mc["track_score_all"]["value"], output_mc["track_score_all"]["tags"], data2 = output_data["track_score_all"]["value"], y_scale = "log", x_label = "track score", norm = norm)
            else:
                Plots.PlotTagged(output_mc["track_score_all"]["value"], output_mc["track_score_all"]["tags"], y_scale = "log", x_label = "track score", norm = norm)
            pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["TrackScoreCut"]["value"], output_mc["TrackScoreCut"]["tags"], data2 = output_data["TrackScoreCut"]["value"], x_range = [0, 1], y_scale = "linear", bins = args.nbins, ncols = 4, x_label = "track score", norm = norm)
        else:
            Plots.PlotTagged(output_mc["TrackScoreCut"]["value"], output_mc["TrackScoreCut"]["tags"], x_range = [0, 1], y_scale = "linear", bins = args.nbins, ncols = 4, x_label = "track score", norm = norm)
        
        Plots.DrawCutPosition(output_mc["TrackScoreCut"]["cuts"], face = "right")
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["NHitsCut"]["value"], output_mc["NHitsCut"]["tags"], data2 = output_data["NHitsCut"]["value"], bins = args.nbins, ncols = 2, x_range = [0, 500], x_label = "nHits", norm = norm)
        else:
            Plots.PlotTagged(output_mc["NHitsCut"]["value"], output_mc["NHitsCut"]["tags"], bins = args.nbins, ncols = 2, x_range = [0, 500], x_label = "nHits", norm = norm)
        pdf.Save()

        Plots.PlotHist2DImshowMarginal(ak.ravel(output_mc["NHitsCut"]["value"]), ak.ravel(output_mc["NHitsCut_completeness"]["value"]), ylabel = "completeness", xlabel = "nHits", x_range = [0, 500], bins = 50, norm = "column", c_scale = "linear")
        Plots.DrawCutPosition(output_mc["NHitsCut"]["cuts"], arrow_length = 100, arrow_loc = 0.1, color = "C6")
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["PiPlusSelection"]["value"], output_mc["PiPlusSelection"]["tags"], data2 = output_data["PiPlusSelection"]["value"], ncols = 2, x_range = [0, 5], x_label = "median $dEdX$ (MeV/cm)", bins = args.nbins, norm = norm)
        else:
            Plots.PlotTagged(output_mc["PiPlusSelection"]["value"], output_mc["PiPlusSelection"]["tags"], ncols = 2, x_range = [0, 5], x_label = "median $dEdX$", bins = args.nbins, norm = norm)
        Plots.DrawCutPosition(min(output_mc["PiPlusSelection"]["cuts"]), arrow_length = 0.5, face = "right")
        Plots.DrawCutPosition(max(output_mc["PiPlusSelection"]["cuts"]), arrow_length = 0.5, face = "left")
        pdf.Save()

        bar_data = []
        for t in output_mc["final_tags"]["tags"]:
            bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
        Plots.PlotBar(bar_data, xlabel = "true particle ID")
        pdf.Save()
    return


def MakePhotonCandidateSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float):
    """ Photon candidate plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """
    
    norm = False if output_data is None else norm
    with Plots.PlotBook(outDir + "photon.pdf") as pdf:    
        if output_data:
            Plots.PlotTagged(output_mc["EMScoreCut"]["value"], output_mc["EMScoreCut"]["tags"], data2 = output_data["EMScoreCut"]["value"], bins = args.nbins, x_range = [0, 1], ncols = 4, x_label = "em score", norm = norm)
        else:
            Plots.PlotTagged(output_mc["EMScoreCut"]["value"], output_mc["EMScoreCut"]["tags"], bins = args.nbins, x_range = [0, 1], ncols = 4, x_label = "em score", norm = norm)

        Plots.DrawCutPosition(output_mc["EMScoreCut"]["cuts"])
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["NHitsCut"]["value"], output_mc["NHitsCut"]["tags"], data2 = output_data["NHitsCut"]["value"], bins = args.nbins, x_label = "number of hits", x_range = [0, 1000], norm = norm, truncate = True)
        else:
            Plots.PlotTagged(output_mc["NHitsCut"]["value"], output_mc["NHitsCut"]["tags"], bins = args.nbins, x_label = "number of hits", x_range = [0, 1000], norm = norm, truncate = True)
        Plots.DrawCutPosition(output_mc["NHitsCut"]["cuts"], arrow_length = 100)
        pdf.Save()

        Plots.PlotHist2DImshowMarginal(ak.ravel(output_mc["NHitsCut"]["value"]), ak.ravel(output_mc["NHitsCut_completeness"]["value"]), bins = 50, y_range = [0, 1],x_range = [0, 800], ylabel = "completeness", xlabel = "number of hits", norm = "column")
        Plots.DrawCutPosition(80, flip = False, arrow_length = 100, color = "C6")
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["BeamParticleDistanceCut"]["value"], output_mc["BeamParticleDistanceCut"]["tags"], data2 = output_data["BeamParticleDistanceCut"]["value"], bins = 31, x_range = [0, 93], x_label = "distance from PFO start to beam end position (cm)", norm = norm, truncate = True)
        else:
            Plots.PlotTagged(output_mc["BeamParticleDistanceCut"]["value"], output_mc["BeamParticleDistanceCut"]["tags"], bins = 31, x_range = [0, 93], x_label = "distance from PFO start to beam end position (cm)", norm = norm, truncate = True)
        Plots.DrawCutPosition(min(output_mc["BeamParticleDistanceCut"]["cuts"]), arrow_length = 30)
        Plots.DrawCutPosition(max(output_mc["BeamParticleDistanceCut"]["cuts"]), face = "left", arrow_length = 30)
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["BeamParticleIPCut"]["value"], output_mc["BeamParticleIPCut"]["tags"], data2 = output_data["BeamParticleIPCut"]["value"], bins = args.nbins, x_range = [0, 50], x_label = "impact parameter wrt beam (cm)", norm = norm, truncate = True)
        else:
            Plots.PlotTagged(output_mc["BeamParticleIPCut"]["value"], output_mc["BeamParticleIPCut"]["tags"], bins = args.nbins, x_range = [0, 50], x_label = "impact parameter wrt beam (cm)", norm = norm, truncate = True)
        Plots.DrawCutPosition(output_mc["BeamParticleIPCut"]["cuts"], arrow_length = 20, face = "left")
        pdf.Save()

        Plots.PlotHist2DImshowMarginal(ak.ravel(output_mc["BeamParticleIPCut"]["value"]), ak.ravel(output_mc["BeamParticleIPCut_completeness"]["value"]), x_range = [0, 40], y_range = [0, 1], bins = 20, norm = "column", c_scale = "linear", ylabel = "completeness", xlabel = "impact parameter wrt beam (cm)")
        Plots.DrawCutPosition(20, arrow_length = 20, face = "left", color = "red")
        pdf.Save()

        bar_data = []
        for t in output_mc["final_tags"]["tags"]:
            bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
        Plots.PlotBar(bar_data, xlabel = "true particle ID")
        pdf.Save()
    return


def MakePi0SelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float):
    """ Pi0 selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """
    norm = False if output_data is None else norm

    with Plots.PlotBook(outDir + "pi0.pdf") as pdf:
        if output_data is not None:
            scale = ak.count(output_data["NPhotonCandidateSelection"]["value"]) / ak.count(output_mc["NPhotonCandidateSelection"]["value"])

            n_photons_scaled = []
            u, c = np.unique(output_mc["NPhotonCandidateSelection"]["value"], return_counts = True)
            for i, j in zip(u, c):
                n_photons_scaled.extend([i]* int(scale * j))

            Plots.PlotBarComparision(n_photons_scaled, output_data["NPhotonCandidateSelection"]["value"], xlabel = "number of $\pi^{0}$ photon candidates", label_1 = "MC", label_2 = "Data", fraction = True, barlabel = False)
        else:
            Plots.PlotBar(output_mc["NPhotonCandidateSelection"]["value"], xlabel = "number of $\pi^{0}$ photon candidates")
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["Pi0MassSelection"]["tags"], data2 = output_data["Pi0MassSelection"]["value"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
        else:
            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["Pi0MassSelection"]["tags"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
        Plots.DrawCutPosition(min(output_mc["Pi0MassSelection"]["cuts"]), face = "right", arrow_length = 50)
        Plots.DrawCutPosition(max(output_mc["Pi0MassSelection"]["cuts"]), face = "left", arrow_length = 50)
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["mass_event_tag"]["tags"], data2 = output_data["Pi0MassSelection"]["value"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
        else:
            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["mass_event_tag"]["tags"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
        Plots.DrawCutPosition(min(output_mc["Pi0MassSelection"]["cuts"]), face = "right", arrow_length = 50)
        Plots.DrawCutPosition(max(output_mc["Pi0MassSelection"]["cuts"]), face = "left", arrow_length = 50)
        pdf.Save()

        if output_data:
            Plots.PlotTagged(output_mc["Pi0OpeningAngleSelection"]["value"], output_mc["Pi0OpeningAngleSelection"]["tags"], data2 = output_data["Pi0OpeningAngleSelection"]["value"], bins = args.nbins, x_label = "Opening angle (rad)", norm = norm)
        else:
            Plots.PlotTagged(output_mc["Pi0OpeningAngleSelection"]["value"], output_mc["Pi0OpeningAngleSelection"]["tags"], bins = args.nbins, x_label = "Opening angle (rad)", norm = norm)
        Plots.DrawCutPosition(min(output_mc["Pi0OpeningAngleSelection"]["cuts"]) * np.pi / 180, face = "right", arrow_length = 0.25)
        Plots.DrawCutPosition(max(output_mc["Pi0OpeningAngleSelection"]["cuts"]) * np.pi / 180, face = "left", arrow_length = 0.25)
        pdf.Save()

    return


def MakeRegionPlots(outputs_mc_masks : dict, outputs_data_masks : dict, outDir : str):
    """ Correlation matrices for truth and reco region selection.

    Args:
        outputs_mc_masks (dict): mc masks for each region
        outputs_data_masks (dict): data masks for each region
        outDir (str): output directory
    """
    with Plots.PlotBook(outDir + "regions.pdf") as pdf:
        # Visualise the regions
        Plots.plot_region_data(outputs_mc_masks["truth_regions"], compare_max=0, title="truth regions")
        pdf.Save()
        Plots.plot_region_data(outputs_mc_masks["reco_regions"], compare_max=0, title="reco regions")
        pdf.Save()
        # Compare the regions
        Plots.compare_truth_reco_regions(outputs_mc_masks["reco_regions"], outputs_mc_masks["truth_regions"], title="")
        pdf.Save()

        if outputs_data_masks is not None:
            Plots.plot_region_data(outputs_data_masks["reco_regions"], compare_max=0, title="reco regions")
            pdf.Save()


def MergeOutputs(outputs : list) -> dict:
    """ Merge multiprocessing output into a single dictionary.

    Args:
        outputs (list): outputs from multiprocessing job.

    Returns:
        dict: merged output
    """
    merged_output = {}

    for output in outputs:
        for selection in output:
            if selection not in merged_output:
                merged_output[selection] = {}

            for o in output[selection]:
                if o not in merged_output[selection]:
                    merged_output[selection][o] = output[selection][o]
                    continue
                if output[selection][o] is None: continue
                if selection == "regions":
                    for m in output[selection][o]:
                        merged_output[selection][o][m] = ak.concatenate([merged_output[selection][o][m], output[selection][o][m]])
                else:
                    if o == "table":
                        tmp = merged_output[selection][o] + output[selection][o]
                        tmp.Name = merged_output[selection][o].Name
                        merged_output[selection][o] = tmp
                    elif o == "masks":
                        for c in output[selection][o]:
                            merged_output[selection][o][c] = ak.concatenate([merged_output[selection][o][c], output[selection][o][c]])
                    else:
                        for c in output[selection][o]:
                            if output[selection][o][c]["value"] is not None:
                                if c in ["PiBeamSelection"]:
                                    for i in merged_output[selection][o][c]["value"]:
                                        merged_output[selection][o][c]["value"][i] = merged_output[selection][o][c]["value"][i] + output[selection][o][c]["value"][i] 
                                else:
                                    merged_output[selection][o][c]["value"] = ak.concatenate([merged_output[selection][o][c]["value"], output[selection][o][c]["value"]]) 

                            if output[selection][o][c]["tags"] is not None:
                                for t in merged_output[selection][o][c]["tags"]:
                                    merged_output[selection][o][c]["tags"][t].mask = ak.concatenate([merged_output[selection][o][c]["tags"][t].mask, output[selection][o][c]["tags"][t].mask])

                            if output[selection][o][c]["fs_tags"] is not None:
                                for t in merged_output[selection][o][c]["fs_tags"]:
                                    merged_output[selection][o][c]["fs_tags"][t].mask = ak.concatenate([merged_output[selection][o][c]["fs_tags"][t].mask, output[selection][o][c]["fs_tags"][t].mask])

    rprint("merged_output")
    rprint(merged_output)
    return merged_output


def MakeTables(output : dict, out : str, sample : str):
    """ Create cutflow tables.

    Args:
        output (dict): output data
        out (str): output name
        sample (str): sample name i.e. mc or data
    """
    for s in output:
        if "table" in output[s]:
            outdir = out + s + "/"
            os.makedirs(outdir, exist_ok = True)
            df = output[s]["table"]
            purity = pd.concat([df["Name"], df.iloc[:, df.columns.to_list().index("Name") + 1:].div(df.iloc[:, df.columns.to_list().index("Name") + 1], axis = 0)], axis = 1)
            efficiency = pd.concat([df["Name"], df.iloc[:, df.columns.to_list().index("Name") + 1:].div(df.iloc[0, df.columns.to_list().index("Name") + 1:], axis = 1)], axis = 1)
            df.style.to_latex(outdir + s + "_counts.tex")
            purity.style.to_latex(outdir + s + "_purity.tex")
            efficiency.style.to_latex(outdir + s + "_efficiency.tex")
    return


def SaveMasks(output : dict, out : str):
    os.makedirs(out, exist_ok = True)
    for head in output:
        if "masks" not in output[head]: continue 
        cross_section.SaveSelection(out + f"{head}_selection_masks.dill", output[head]["masks"])

@Master.timer
def main(args):
    shower_merging.SetPlotStyle(extend_colors = True)

    func_args = vars(args)
    func_args["data"] = False

    output_mc = MergeOutputs(Processing.mutliprocess(run, [args.mc_file], args.batches, args.events, func_args, args.threads)) # run the main analysing method

    output_data = None
    if args.mc_only != True:
        if args.data_file is not None:
            func_args["data"] = True
            output_data = MergeOutputs(Processing.mutliprocess(run, [args.data_file], args.batches, args.events, func_args, args.threads)) # run the main analysing method

    # tables
    MakeTables(output_mc, args.out + "tables_mc/", "mc")
    if output_data is not None: MakeTables(output_data, args.out + "tables_data/", "data")

    # save masks used in selection
    SaveMasks(output_mc, args.out + "masks_mc/")
    if output_data is not None: SaveMasks(output_data, args.out + "masks_data/")

    # output directories
    os.makedirs(args.out + "plots/", exist_ok = True)

    # plots
    MakeBeamSelectionPlots(output_mc["beam"]["data"], output_data["beam"]["data"] if output_data else None, args.out + "plots/", norm = args.norm)
    MakePiPlusSelectionPlots(output_mc["pip"]["data"], output_data["pip"]["data"] if output_data else None, args.out + "plots/", norm = args.norm)
    MakePhotonCandidateSelectionPlots(output_mc["photon"]["data"], output_data["photon"]["data"] if output_data else None, args.out + "plots/", norm = args.norm)
    MakePi0SelectionPlots(output_mc["pi0"]["data"], output_data["pi0"]["data"] if output_data else None, args.out + "plots/", norm = args.norm)
    MakeRegionPlots(output_mc["regions"], output_data["regions"] if output_data else None, args.out + "plots/")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Ntuples(parser, True)
    cross_section.ApplicationArguments.BeamQualityCuts(parser, True)
    cross_section.ApplicationArguments.BeamSelection(parser)
    cross_section.ApplicationArguments.ShowerCorrection(parser)
    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Plots(parser)
    cross_section.ApplicationArguments.Config(parser)

    parser.add_argument("--mc", dest = "mc_only", action = "store_true", help = "Only analyse at MC file.")

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)

    rprint(vars(args))
    main(args)