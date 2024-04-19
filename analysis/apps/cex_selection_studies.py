#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:47

Author: Shyam Bhuller

Description: Selection studies for the charge exchange analysis.
"""
import argparse
import os

from rich import print as rprint
from python.analysis import Master, BeamParticleSelection, EventSelection, PFOSelection, Plots, shower_merging, Processing, Tags, cross_section

import awkward as ak
import numpy as np
import pandas as pd

x_label = {
    "track_score_all" : "track score", 
    "TrackScoreCut" : "track score",
    "NHitsCut" : "number of hits",
    "PiPlusSelection" : "median $dEdX$ (MeV/cm)",
    "BeamParticleDistanceCut" : "$d$ (cm)",
    "BeamParticleIPCut" : "$b$ (cm)",
    "Chi2ProtonSelection" : "$(\chi^{2}/ndf)_{p}$",
    "PiBeamSelection" : "True particle ID",
    "APA3Cut" : "Beam end position z (cm)",
    "TrueFiducialCut" : "True beam end position z (cm)",
    "PandoraTagCut" : "Pandora tag",
    "DxyCut" : "$\delta_{xy}$",
    "DzCut" : "$\delta_{z}$",
    "CosThetaCut" : "$\cos(\\theta)$",
    "MichelScoreCut" : "Michel score",
    "MedianDEdXCut" : "Median $dE/dX$ (MeV/cm)",
    "BeamScraperCut" : "$r_{inst}$",
}
y_scale = {
    "track_score_all" : "log",
    "TrackScoreCut" : "linear",
    "NHitsCut" : "linear",
    "PiPlusSelection" : "linear",
    "BeamParticleDistanceCut" : "linear",
    "BeamParticleIPCut" : "linear",
    "Chi2ProtonSelection" : "linear",
    "PiBeamSelection" : None,
    "APA3Cut" : "linear",
    "TrueFiducialCut" : "linear",
    "PandoraTagCut" : None,
    "DxyCut" : "log",
    "DzCut" : "log",
    "CosThetaCut" : "log",
    "MichelScoreCut" : "log",
    "MedianDEdXCut" : "log",
    "BeamScraperCut" : "linear",
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
    "DxyCut" : [0, 5],
    "DzCut" : [-5, 5],
    "CosThetaCut" : [0.9, 1],
    "MichelScoreCut" : [0, 1],
    "MedianDEdXCut" : [1.5, 3],
    "BeamScraperCut" : [0, 5],
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
    "PandoraTagCut" : 50,
    "DxyCut" : 50,
    "DzCut" : 50,
    "CosThetaCut" : 50,
    "MichelScoreCut" : 50,
    "MedianDEdXCut" : 50,
    "BeamScraperCut" : 50,
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
    "PandoraTagCut" : 2,
    "DxyCut" : 2,
    "DzCut" : 2,
    "CosThetaCut" : 2,
    "MichelScoreCut" : 2,
    "MedianDEdXCut" : 2,
    "BeamScraperCut" : 2,
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
    "PandoraTagCut" : False,
    "DxyCut" : False,
    "DzCut" : False,
    "CosThetaCut" : False,
    "MichelScoreCut" : False,
    "MedianDEdXCut" : False,
    "BeamScraperCut" : False,
}


def CalculateArrowLength(value, x_range):
    if x_range is None:
        l = ak.max(value)
    else:
        l = max(x_range)
    return 0.1 * l


def MakeOutput(value : ak.Array, tags : Tags.Tags, cuts : list = [], op : list = [], fs_tags : Tags.Tags = None) -> dict:
    """ Output dictionary for multiprocessing class, if we want to be fancy this can be a dataclass instead.
        Generally the data stored in this dictionary should be before any cuts are applied, for plotting purposes.

    Args:
        value (ak.Array): value which is being studied
        tags (Tags.Tags): tags which categorise the value
        cuts (list, optional): cut values used on this data. Defaults to [].
        cuts (list, optional): operation of cut. Defaults to [].
        fs_tags (Tags.Tags, optional): final state tags to the data. Defaults to None.

    Returns:
        dict: dictionary of data.
    """
    return {"value" : value, "tags" : tags, "cuts" : cuts, "op" : op, "fs_tags" : fs_tags}


def AnalyseBeamFiducialCut(events : Master.Data, beam_instrumentation : bool, functions : dict, args : dict):
    output = {}
    masks = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueBeamParticleTags(events) if beam_instrumentation is False else None)
    output["no_selection"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))
    if "TrueFiducialCut" in args:
        masks = {k : functions[k](events, **v) for k, v in args.items() if k in ["APA3Cut", "TrueFiducialCut"]}
        for a in ["APA3Cut", "TrueFiducialCut"]:
            mask, property = functions[a](events, **args[a], return_property = True)
            cut_table.add_mask(mask, a)
            cut_values = args[a]["cut"] if "cut" in args[a] else None
            operations = args[a]["op"] if "op" in args[a] else None
            output[a] = MakeOutput(property, Tags.GenerateTrueBeamParticleTags(events), cut_values, operations)

            events.Filter([mask], [mask])
            output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))

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

    output["no_selection"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))

    masks = {k : functions[k](events, **v) for k, v in args.items()}

    #* pi+ beam selection
    mask = BeamParticleSelection.PiBeamSelection(events, **args["PiBeamSelection"])
    cut_table.add_mask(mask, "PiBeamSelection")
    counts = Tags.GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i].mask)
    output["PiBeamSelection"] = MakeOutput(counts, Tags.GenerateTrueBeamParticleTags(events), None, None)
    events.Filter([mask], [mask])
    output["PiBeamSelection"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    for a in args:
        if a in ["PiBeamSelection", "TrueFiducialCut"]: continue
        if ("TrueFiducialCut" in args) and (a == "APA3Cut"): continue # should have applied fiducial cuts first
        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = MakeOutput(property, Tags.GenerateTrueBeamParticleTags(events), cut_values, operations)

        if (a == "BeamScraperCut") and (beam_instrumentation == False):
            if beam_instrumentation == False:
                scraper_id = cross_section.IsScraper(events, args[a]["fits"])
                output[a + "_scraper_tag"] = MakeOutput(None, Tags.BeamScraperTag(scraper_id), None, None)

        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)
        
    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))

    df = cut_table.get_table(init_data_name = "no selection", pfos = False, percent_remain = False, relative_percent = False, ave_per_event = False)
    return output, df, masks


def AnalysePFOSelection(events : Master.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse PFO selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTagsInterestingPFOs(events))

    for a in args:
        if a in ["NHitsCut", "BeamParticleIPCut"]:
            output[a + "_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = MakeOutput(property, Tags.GenerateTrueParticleTagsInterestingPFOs(events), cut_values, operations)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueParticleTagsInterestingPFOs(events), None, None)
    
    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePiPlusSelection(events : Master.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse the daughter pi+ selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTagsPiPlus(events))

    for a in args:
        if a == "NHitsCut":
            output[a + "_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = MakeOutput(property, Tags.GenerateTrueParticleTagsPiPlus(events), cut_values, operations)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueParticleTagsPiPlus(events), None, None)
    
    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePhotonCandidateSelection(events : Master.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse the photon candidate selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTagsPi0Shower(events) if data is False else None)

    for a in args:
        if a in ["NHitsCut", "BeamParticleIPCut"]:
            output[a + "_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = MakeOutput(property, Tags.GenerateTrueParticleTagsPi0Shower(events), cut_values, operations)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueParticleTagsPi0Shower(events), None, None)

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
    photonCandidates = PFOSelection.InitialPi0PhotonSelection(events) # repeat the photon candidate selection, but only require the mask
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GeneratePi0Tags(events, photonCandidates) if data is False else None)

    for a in args:
        if a == "Pi0MassSelection":
            mask, property = functions[a](events, **args[a], return_property = True, photon_mask = photonCandidates)
            output["mass_event_tag"] = MakeOutput(None, EventSelection.GenerateTrueFinalStateTags(events), None, None)
        else:
            mask, property = functions[a](events, **args[a], return_property = True, photon_mask = photonCandidates)
        if a != "NPhotonCandidateSelection":
            mask = ak.flatten(mask)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
        output[a] = MakeOutput(property, tags, cut_values, operations)
        events.Filter([mask], [mask])
        photonCandidates = photonCandidates[mask]
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* final counts
    output["event_tag"] = MakeOutput(None, EventSelection.GenerateTrueFinalStateTags(events), None, None)
    tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["final_tags"] = MakeOutput(None, tags, None, None)

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
        correction_params (str, optional): shower energy correction parameters file. Defaults to None.

    Returns:
        tuple[dict, dict]: truth and reco regions
    """
    truth_regions = EventSelection.create_regions(events.trueParticles.nPi0, events.trueParticles.nPiPlus + events.trueParticles.nPiMinus) if is_data == False else None

    if correction_params is None:
        params = None
    else:
        params = cross_section.LoadConfiguration(correction_params)

    reco_pi0_counts = EventSelection.count_pi0_candidates(events, exactly_two_photons = True, photon_mask = photon_mask, correction = cross_section.EnergyCorrection.shower_energy_correction[correction], correction_params = params)
    reco_pi_plus_counts_mom_cut = EventSelection.count_charged_pi_candidates(events, energy_cut = None)
    reco_regions = EventSelection.create_regions(reco_pi0_counts, reco_pi_plus_counts_mom_cut)
    return truth_regions, reco_regions


def CreatePFOMasks(sample : Master.Data, selections : dict, args_type : str, extra_args : dict = None) -> dict[np.array]:
    """ Create PFO masks to save to file.

    Args:
        mc (Master.Data): sample.
        selections (dict): PFO selections.
        args_type (str): use Data or MC arguments.
        extra_args (dict, optional): any additional arguments to add. Defaults to None.

    Returns:
        masks: dictionary of masks.
    """
    masks = {}
    for n, c, v in zip(selections["selections"].keys(), selections["selections"].values(), selections[args_type].values()):
        if extra_args is not None:
            v = {**v, **extra_args}
        masks[n] = c(sample, **v)
    return masks

# @Processing.log_process
def run(i, file, n_events, start, selected_events, args) -> dict:
    events = Master.Data(file, nEvents = n_events, start = start, nTuple_type = args["ntuple_type"]) # load data
    output = {
        "fiducial" : None,
        "beam" : None,
        "null_pfo" : None,
        "pi" : None,
        "photon" : None,
        "loose_pi" : None,
        "loose_photon" : None,
        "pi0" : None,
        "regions" : None
    }

    if args["data"] == True:
        selection_args = "data_arguments"
    else:
        selection_args = "mc_arguments"

    if "beam_selection" in args:
        print("beam particle selection")

        if "TrueFiducialCut" in args["beam_selection"]["selections"]:
            output_fd, table_fd, fd_masks = AnalyseBeamFiducialCut(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args])
            output["fiducial"] = {"data" : output_fd, "table" : table_fd, "masks" : fd_masks}

        output_beam, table_beam, beam_masks = AnalyseBeamSelection(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args]) # events are cut after this
        output["beam"] = {"data" : output_beam, "table" : table_beam, "masks" : beam_masks}

    if "valid_pfo_selection" in args:
        print("PFO pre-selection")
        good_PFO_mask = PFOSelection.GoodShowerSelection(events)
        good_PFO_cut_table = Master.CutTable.CutHandler(events, tags = None)
        good_PFO_cut_table.add_mask(good_PFO_mask, "GoodShowerSelection")
        events.Filter([good_PFO_mask])
        output["null_pfo"] = {"table" : good_PFO_cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False), "masks" : {"ValidPFOSelection" : good_PFO_mask}}

    # this selection only needs to be done for shower merging ntuple because the PDSPAnalyser ntuples only store daughter PFOs, not the beam as well.
    if events.nTuple_type == Master.Ntuple_Type.SHOWER_MERGING:
        #* beam particle daughter selection 
        mask = PFOSelection.BeamDaughterCut(events)
        events.Filter([mask])

    if "piplus_selection" in args:
        print("pion selection")
        # output_pip, table_pip = AnalysePiPlusSelection(events.Filter(returnCopy = True), args["data"], args["piplus_selection"]["selections"], args["piplus_selection"][selection_args]) # pass the PFO selections a copy of the event        
        output_pip, table_pip = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["piplus_selection"]["selections"], args["piplus_selection"][selection_args])        
        pip_masks = CreatePFOMasks(events, args["piplus_selection"], selection_args)
        output["pi"] = {"data" : output_pip, "table" : table_pip, "masks" : pip_masks}

    if "loose_pion_selection" in args:
        print("loose pion selection")
        output_loose_pip, table_loose_pip = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["loose_pion_selection"]["selections"], args["loose_pion_selection"][selection_args])
        loose_pip_masks = CreatePFOMasks(events, args["loose_pion_selection"], selection_args)
        output["loose_pi"] = {"data" : output_loose_pip, "table" : table_loose_pip, "masks" : loose_pip_masks}

    if "loose_photon_selection" in args:
        print("loose photon selection")
        output_loose_photon, table_loose_photon = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["loose_photon_selection"]["selections"], args["loose_photon_selection"][selection_args])
        loose_photon_masks = CreatePFOMasks(events, args["loose_photon_selection"], selection_args)
        output["loose_photon"] = {"data" : output_loose_photon, "table" : table_loose_photon, "masks" : loose_photon_masks}

    if "photon_selection" in args:
        print("photon selection")
        # output_photon, table_photon = AnalysePhotonCandidateSelection(events.Filter(returnCopy = True), args["data"], args["photon_selection"]["selections"], args["photon_selection"][selection_args])
        output_photon, table_photon = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["photon_selection"]["selections"], args["photon_selection"][selection_args])
        photon_masks = CreatePFOMasks(events, args["photon_selection"], selection_args)
        output["photon"] = {"data" : output_photon, "table" : table_photon, "masks" : photon_masks}

        photon_selection_mask = None
        for m in photon_masks:
            if photon_selection_mask is None:
                photon_selection_mask = photon_masks[m]
            else:
                photon_selection_mask = photon_selection_mask & photon_masks[m]

        if "pi0_selection" in args:
            print("pi0 selection")
            output_pi0, table_pi0 = AnalysePi0Selection(events.Filter(returnCopy = True), args["data"], args["pi0_selection"]["selections"], args["pi0_selection"][selection_args])
            pi0_masks = CreatePFOMasks(events, args["pi0_selection"], selection_args, {"photon_mask" : photon_selection_mask})
            output["pi0"] = {"data" : output_pi0, "table" : table_pi0, "masks" : pi0_masks}

        print("regions")
        truth_regions, reco_regions = AnalyseRegions(events, photon_selection_mask, args["data"], args["shower_correction"]["correction"], args["shower_correction"]["correction_params"])

        regions  = {
            "truth_regions"       : truth_regions,
            "reco_regions"        : reco_regions
        }
        output["regions"] = regions
    return output


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
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "TrueFiducialCut":
                    for l in ["linear", "log"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = None, norm = norm, y_scale = l, x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "BeamScraperCut":
                    for t in ["BeamScraperCut", "BeamScraperCut_scraper_tag"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[t]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                else:
                    Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                    if p == "CosThetaCut":
                        arrow_l = 0.02
                    else:
                        arrow_l = CalculateArrowLength(output_mc[p]["value"], x_range[p])
                    Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = arrow_l, arrow_loc = 0.7, color = "k")
                    pdf.Save()

        Plots.PlotTags(output_mc["final_tags"]["tags"], "True particle ID")
        pdf.Save()
    Plots.plt.close("all")
    return


def MakePFOSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, book_name : str):
    norm = False if output_data is None else norm

    with Plots.PlotBook(outDir + book_name) as pdf:

        for p in output_mc:
            if p in x_label:
                Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.5, color = "k")
                pdf.Save()
                if f"{p}_completeness" in output_mc:
                    Plots.PlotHist2DImshowMarginal(ak.ravel(output_mc[p]["value"]), ak.ravel(output_mc[f"{p}_completeness"]["value"]), ylabel = "completeness", xlabel = x_label[p], x_range = x_range[p], bins = nbins[p], norm = "column", c_scale = "linear")
                    Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.1, color = "C6")
                    pdf.Save()
        Plots.PlotTags(output_mc["final_tags"]["tags"], xlabel = "true particle ID")
        pdf.Save()
    Plots.plt.close("all")
    return


def MakePFOSelectionPlotsConsdensed(output_mc : dict, output_mc_loose : dict, output_data : dict, output_data_loose : dict, outDir : str, norm : float, book_name : str):
    norm = False if output_data is None else norm

    with Plots.PlotBook(outDir + book_name) as pdf:

        for p in output_mc_loose:
            if p in x_label:
                Plots.PlotTagged(output_mc_loose[p]["value"], output_mc_loose[p]["tags"], data2 = output_data_loose[p]["value"] if output_data_loose else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                
                for c, mc in zip(["red", "magenta"], [output_mc, output_mc_loose]):
                    Plots.DrawMultiCutPosition(mc[p]["cuts"], face = mc[p]["op"], arrow_length = CalculateArrowLength(mc[p]["value"], x_range[p]), arrow_loc = 0.5, color = c)
                
                pdf.Save()
                if f"{p}_completeness" in output_mc:
                    Plots.PlotHist2DImshowMarginal(ak.ravel(mc[p]["value"]), ak.ravel(mc[f"{p}_completeness"]["value"]), ylabel = "completeness", xlabel = x_label[p], x_range = x_range[p], bins = nbins[p], norm = "column", c_scale = "linear")

                    for c, mc in zip(["red", "magenta"], [output_mc, output_mc_loose]):
                        Plots.DrawMultiCutPosition(mc[p]["cuts"], face = mc[p]["op"], arrow_length = CalculateArrowLength(mc[p]["value"], x_range[p]), arrow_loc = 0.1, color = c)
                
                    pdf.Save()

        for mc in [output_mc, output_mc_loose]:
            Plots.PlotTags(mc["final_tags"]["tags"], xlabel = "true particle ID")
            pdf.Save()
    Plots.plt.close("all")
    return


def MakePi0SelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, nbins : int):
    """ Pi0 selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """
    norm = False if output_data is None else norm

    with Plots.PlotBook(outDir + "pi0.pdf") as pdf:
        if "NPhotonCandidateSelection" in output_mc:
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

        if "Pi0MassSelection" in output_mc:
            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["Pi0MassSelection"]["tags"], data2 = output_data["Pi0MassSelection"]["value"] if output_data else None, bins = nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm, ncols = 1)
            Plots.DrawMultiCutPosition(output_mc["Pi0MassSelection"]["cuts"], face = output_mc["Pi0MassSelection"]["op"], arrow_length = 50)
            pdf.Save()

            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["mass_event_tag"]["tags"], data2 = output_data["Pi0MassSelection"]["value"] if output_data else None, bins = nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm, ncols = 1)
            Plots.DrawMultiCutPosition(output_mc["Pi0MassSelection"]["cuts"], face = output_mc["Pi0MassSelection"]["op"], arrow_length = 50)
            pdf.Save()

        if "Pi0OpeningAngleSelection" in output_mc:
            Plots.PlotTagged(output_mc["Pi0OpeningAngleSelection"]["value"], output_mc["Pi0OpeningAngleSelection"]["tags"], data2 = output_data["Pi0OpeningAngleSelection"]["value"] if output_data else None, bins = nbins, x_label = "Opening angle (rad)", norm = norm, ncols = 1)

            Plots.DrawMultiCutPosition((np.array(output_mc["Pi0OpeningAngleSelection"]["cuts"]) * np.pi / 180).tolist(), face = output_mc["Pi0OpeningAngleSelection"]["op"], arrow_length = 0.25)
            pdf.Save()
    Plots.plt.close("all")
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
    Plots.plt.close("all")
    return

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
            if output[selection] is None:
                continue
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
            df = df.rename(columns = {i : i.replace(" remaining", "") for i in df})
            print(df)
            purity = pd.concat([df["Name"], df.iloc[:, df.columns.to_list().index("Name") + 1:].div(df.iloc[:, df.columns.to_list().index("Name") + 1], axis = 0)], axis = 1)
            efficiency = pd.concat([df["Name"], df.iloc[:, df.columns.to_list().index("Name") + 1:].div(df.iloc[0, df.columns.to_list().index("Name") + 1:], axis = 1)], axis = 1)
            
            df.style.hide(axis = "index").to_latex(outdir + s + "_counts.tex")
            purity.style.hide(axis = "index").to_latex(outdir + s + "_purity.tex")
            efficiency.style.hide(axis = "index").to_latex(outdir + s + "_efficiency.tex")
    return


def SaveMasks(output : dict, out : str):
    """ Save selection masks to dill file

    Args:
        output (dict): masks
        out (str): output file directory
    """
    os.makedirs(out, exist_ok = True)
    for head in output:
        if "masks" not in output[head]: continue 
        cross_section.SaveObject(out + f"{head}_selection_masks.dill", output[head]["masks"])

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
    if output_mc["fiducial"]:
        MakeBeamSelectionPlots(output_mc["fiducial"]["data"], output_data["fiducial"]["data"] if output_data else None, args.out + "plots/", norm = args.norm, book_name = "fiducial")

    if output_mc["beam"]: #* this is assuming you apply the same cuts as Data and MC (which is implictly assumed for now)
        MakeBeamSelectionPlots(output_mc["beam"]["data"], output_data["beam"]["data"] if output_data else None, args.out + "plots/", norm = args.norm, book_name = "beam")

    for i in ["pi", "photon", "loose_pi", "loose_photon"]:
        if output_mc[i]:
            MakePFOSelectionPlots(output_mc[i]["data"], output_data[i]["data"] if output_data else None, args.out + "plots/", norm = args.norm, book_name = i)

    if output_mc["loose_pi"]:
        MakePFOSelectionPlotsConsdensed(
            output_mc["pi"]["data"],
            output_mc["loose_pi"]["data"],
            output_data["pi"]["data"] if output_data else None,
            output_data["loose_pi"]["data"] if output_data else None,
            args.out + "plots/",
            norm = args.norm,
            book_name = "pi_both"
            )

    if output_mc["loose_photon"]:
        MakePFOSelectionPlotsConsdensed(
            output_mc["photon"]["data"],
            output_mc["loose_photon"]["data"],
            output_data["photon"]["data"] if output_data else None,
            output_data["loose_photon"]["data"] if output_data else None,
            args.out + "plots/",
            norm = args.norm,
            book_name = "photon_both"
            )

    if output_mc["pi0"]:
        MakePi0SelectionPlots(output_mc["pi0"]["data"], output_data["pi0"]["data"] if output_data else None, args.out + "plots/", norm = args.norm, nbins = args.nbins)
    if output_mc["regions"]:
        MakeRegionPlots(output_mc["regions"], output_data["regions"] if output_data else None, args.out + "plots/")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Config(parser, required = True)
    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Plots(parser)

    parser.add_argument("--mc", dest = "mc_only", action = "store_true", help = "Only analyse the MC file.")

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)

    rprint(vars(args))
    main(args)