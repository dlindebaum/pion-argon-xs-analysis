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
from python.analysis import Master, BeamParticleSelection, EventSelection, PFOSelection, Plots, shower_merging, vector, Processing, Tags, cross_section

import awkward as ak
from matplotlib.backends.backend_pdf import PdfPages
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


def AnalyseBeamSelection(events : Master.Data, beam_instrumentation : bool, beam_quality_fits : dict, beam_scraper_fits : dict, args : argparse.Namespace) -> tuple[dict, pd.DataFrame]:
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
    cut_table.add_mask(mask, "pi_beam")
    counts = Tags.GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i].mask)
    output["pi_beam"] = MakeOutput(counts, Tags.GenerateTrueBeamParticleTags(events), None)
    events.Filter([mask], [mask])
    output["pi_beam"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* calo size cut
    mask = BeamParticleSelection.CaloSizeCut(events)
    cut_table.add_mask(mask, "calo_size")
    output["calo_size"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None)
    events.Filter([mask], [mask])
    output["calo_size"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* beam pandora tag selection
    mask = BeamParticleSelection.PandoraTagCut(events) # create the mask for the cut
    cut_table.add_mask(mask, "pandora_tag")
    output["pandora_tag"] = MakeOutput(events.recoParticles.beam_pandora_tag, Tags.GenerateTrueBeamParticleTags(events), [args["beam_selection"]["mc_arguments"][1]["cut"]]) # store the data to cut, cut value and truth tags
    events.Filter([mask], [mask]) # apply the cut
    output["pandora_tag"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events) # store the final state truth tags after the cut is applied

    #* dxy cut
    dxy = (((events.recoParticles.beam_startPos_SCE.x - beam_quality_fits["mu_x"]) / beam_quality_fits["sigma_x"])**2 + ((events.recoParticles.beam_startPos_SCE.y - beam_quality_fits["mu_y"]) / beam_quality_fits["sigma_y"])**2)**0.5
    mask = dxy < 3
    cut_table.add_mask(mask, "dxy")
    output["dxy"] = MakeOutput(dxy, Tags.GenerateTrueBeamParticleTags(events), [3], EventSelection.GenerateTrueFinalStateTags(events))
    print(f"dxy cut: {BeamParticleSelection.CountMask(mask)}")
    events.Filter([mask], [mask])
    output["dxy"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* dz cut
    delta_z = (events.recoParticles.beam_startPos_SCE.z - beam_quality_fits["mu_z"]) / beam_quality_fits["sigma_z"]
    mask = (delta_z > -3) & (delta_z < 3)
    cut_table.add_mask(mask, "dz")
    output["dz"] = MakeOutput(delta_z, Tags.GenerateTrueBeamParticleTags(events), [-3, 3], EventSelection.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    print(f"dz cut: {BeamParticleSelection.CountMask(mask)}")
    output["dz"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* beam direction
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos_SCE, events.recoParticles.beam_startPos_SCE))
    beam_dir_mu = vector.normalize(vector.vector(beam_quality_fits["mu_dir_x"], beam_quality_fits["mu_dir_y"], beam_quality_fits["mu_dir_z"]))
    beam_costh = vector.dot(beam_dir, beam_dir_mu)
    mask = beam_costh > 0.95
    cut_table.add_mask(mask, "cos_theta")
    output["cos_theta"] = MakeOutput(beam_costh, Tags.GenerateTrueBeamParticleTags(events), [0.95], EventSelection.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["cos_theta"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* APA3 cut
    mask = BeamParticleSelection.APA3Cut(events)
    cut_table.add_mask(mask, "beam_endPos_z")
    output["beam_endPos_z"] = MakeOutput(events.recoParticles.beam_endPos_SCE.z, Tags.GenerateTrueBeamParticleTags(events), [220], EventSelection.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["beam_endPos_z"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* michel score cut
    score = ak.where(events.recoParticles.beam_nHits != 0, events.recoParticles.beam_michelScore / events.recoParticles.beam_nHits, -999)
    mask = BeamParticleSelection.MichelScoreCut(events)
    cut_table.add_mask(mask, "michel_score")
    output["michel_score"] = MakeOutput(score, Tags.GenerateTrueBeamParticleTags(events), [0.55], EventSelection.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["michel_score"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* median dE/dX
    mask = BeamParticleSelection.MedianDEdXCut(events)
    cut_table.add_mask(mask, "median_dEdX")
    median = PFOSelection.Median(events.recoParticles.beam_dEdX)
    output["median_dEdX"] = MakeOutput(median, Tags.GenerateTrueBeamParticleTags(events), [2.4], EventSelection.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["median_dEdX"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    # #* beam scraper
    # mask = BeamParticleSelection.BeamScraper(events, **args["beam_selection"]["mc_arguments"][-1])
    # cut_table.add_mask(mask, "beam_scraper")

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, EventSelection.GenerateTrueFinalStateTags(events))
    return output, cut_table.get_table(init_data_name = "no selection", pfos = False, percent_remain = False, relative_percent = False, ave_per_event = False)


def AnalysePiPlusSelection(events : Master.Data) -> tuple[dict, pd.DataFrame]:
    """ Analyse the daughter pi+ selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTags(events))

    # this selection only needs to be done for shower merging ntuple because the PDSPAnalyser ntuples only store daughter PFOs, not the beam as well.
    if events.nTuple_type == Master.Ntuple_Type.SHOWER_MERGING:
        #* beam particle daughter selection 
        mask = PFOSelection.BeamDaughterCut(events)
        cut_table.add_mask(mask, "track_score_all")
        output["track_score_all"] = MakeOutput(events.recoParticles.trackScore, Tags.GenerateTrueParticleTags(events)) # keep a record of the track score to show the cosmic muon background
        events.Filter([mask])

    #* track score selection
    mask = PFOSelection.TrackScoreCut(events)
    cut_table.add_mask(mask, "track_score")
    output["track_score"] = MakeOutput(events.recoParticles.trackScore, Tags.GenerateTrueParticleTags(events), [0.5])
    events.Filter([mask])

    #* nHits cut
    mask = PFOSelection.NHitsCut(events, 20)
    cut_table.add_mask(mask, "nHits")
    output["nHits"] = MakeOutput(events.recoParticles.nHits, Tags.GenerateTrueParticleTags(events), [20])
    output["completeness"] = MakeOutput(events.trueParticlesBT.completeness, Tags.GenerateTrueParticleTags(events))
    events.Filter([mask])

    #* median dEdX    
    mask = PFOSelection.PiPlusSelection(events)
    cut_table.add_mask(mask, "median_dEdX")
    output["median_dEdX"] = MakeOutput(PFOSelection.Median(events.recoParticles.track_dEdX), Tags.GenerateTrueParticleTags(events), [0.5, 2.8])
    events.Filter([mask])

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueParticleTags(events), None)
    return output, cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)


def AnalysePhotonCandidateSelection(events : Master.Data) -> tuple[dict, pd.DataFrame]:
    """ Analyse the photon candidate selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTags(events))

    #* em shower score cut
    output["em_score"] = MakeOutput(events.recoParticles.emScore, Tags.GenerateTrueParticleTags(events), [0.5])
    mask = PFOSelection.EMScoreCut(events, 0.5)
    cut_table.add_mask(mask, "em_score")
    events.Filter([mask])

    #* nHits cut
    output["nHits"] = MakeOutput(events.recoParticles.nHits, Tags.GenerateTrueParticleTags(events), [80])
    output["nHits_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [], [])
    mask = PFOSelection.NHitsCut(events, 80)
    cut_table.add_mask(mask, "nHits")
    events.Filter([mask])

    #* distance to beam cut
    dist = PFOSelection.find_beam_separations(events)
    output["beam_separation"] = MakeOutput(dist, Tags.GenerateTrueParticleTags(events), [3, 90])
    mask = PFOSelection.BeamParticleDistanceCut(events, [3, 90])
    cut_table.add_mask(mask, "beam_separation")
    events.Filter([mask])

    #* impact parameter
    ip = PFOSelection.find_beam_impact_parameters(events)
    output["impact_parameter"] = MakeOutput(ip, Tags.GenerateTrueParticleTags(events), [20])
    output["impact_parameter_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [], [])
    mask = PFOSelection.BeamParticleIPCut(events)
    cut_table.add_mask(mask, "impact_parameter")
    events.Filter([mask])

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueParticleTags(events), None)
    return output, cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)


def AnalysePi0Selection(events : Master.Data, data : bool = False, correction : callable = None, correction_params : dict = None) -> tuple[dict, pd.DataFrame]:
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

    #* number of photon candidate
    n_photons = ak.sum(photonCandidates, -1)
    mask = n_photons == 2
    cut_table.add_mask(mask, "n_photons")
    output["n_photons"] = MakeOutput(n_photons, None, [2])
    events.Filter([mask], [mask]) # technically is an event level cut as we only try to find 1 pi0 in the final state, two is more complicated
    photonCandidates = photonCandidates[mask]

    #* invariant mass
    shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photonCandidates)

    if correction is None:
        le = shower_pairs.reco_lead_energy
        se = shower_pairs.reco_sub_energy
    else:
        le = correction(shower_pairs.reco_lead_energy, **correction_params)
        se = correction(shower_pairs.reco_sub_energy, **correction_params)

    mass = shower_pairs.Mass(le, se, shower_pairs.reco_angle)
    mass = ak.fill_none(ak.pad_none(mass, 1, -1), -999, -1)
    mask = (mass > 50) & (mass < 250)
    mask = ak.flatten(mask) # 1 pi0
    cut_table.add_mask(mask, "mass")
    tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["mass"] = MakeOutput(mass, tags, [50, 250])
    output["mass_event_tag"] = MakeOutput(None, EventSelection.GenerateTrueFinalStateTags(events), None)
    events.Filter([mask], [mask])
    photonCandidates = photonCandidates[mask]

    #* opening angle
    shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photonCandidates)
    angle = ak.fill_none(ak.pad_none(shower_pairs.reco_angle, 1, -1), -999, -1)
    mask = (angle > (10 * np.pi / 180)) & (angle < (80 * np.pi / 180))
    mask = ak.flatten(mask) # 1 pi0
    tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["angle"] = MakeOutput(angle, tags, [(10 * np.pi / 180), (80 * np.pi / 180)])
    cut_table.add_mask(mask, "angle")
    events.Filter([mask], [mask])
    photonCandidates = photonCandidates[mask]

    #* final counts
    output["event_tag"] = MakeOutput(None, EventSelection.GenerateTrueFinalStateTags(events), None)
    tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["final_tags"] = MakeOutput(None, tags, None) 

    return output, cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, pfos = False)


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


# @Processing.log_process
def run(i, file, n_events, start, selected_events, args) -> dict:
    events = Master.Data(file, nEvents = n_events, start = start, nTuple_type = args["ntuple_type"]) # load data

    if args["data"] == True:
        beam_quality_fit_values = args["data_beam_quality_fit"]
    else:
        beam_quality_fit_values = args["mc_beam_quality_fit"]

    do_scraper_cut = any(s == BeamParticleSelection.BeamScraper for s in args["beam_selection"]["selections"]) # remove with new cut handling

    #* shower energy correction
    if args["correction_params"] is not None:
        with open(args["correction_params"], "r") as f:
            correction_params = json.load(f)
    else:
        correction_params = None

    print("beam particle selection")
    output_beam, table_beam = AnalyseBeamSelection(events, args["data"], beam_quality_fit_values, args["mc_beam_scraper_fit"], args) # events are cut after this

    print("PFO pre-selection")
    good_PFO_mask = PFOSelection.GoodShowerSelection(events)
    events.Filter([good_PFO_mask])

    print("pion selection")
    output_pip, table_pip = AnalysePiPlusSelection(events.Filter(returnCopy = True)) # pass the PFO selections a copy of the event

    print("photon selection")
    photon_selection_mask = PFOSelection.InitialPi0PhotonSelection(events)
    output_photon, table_photon = AnalysePhotonCandidateSelection(events.Filter(returnCopy = True))

    print("pi0 selection")
    output_pi0, table_pi0 = AnalysePi0Selection(events.Filter(returnCopy = True), args["data"], cross_section.EnergyCorrection.shower_energy_correction[args["correction"]], correction_params)

    print("regions")
    truth_regions, reco_regions = AnalyseRegions(events, photon_selection_mask, args["data"], args["correction"], correction_params)

    regions  = {
        "truth_regions"       : truth_regions,
        "reco_regions"        : reco_regions
    }

    output = {
        "beam" : {"data" : output_beam, "table" : table_beam},
        "pip" : {"data" : output_pip, "table" : table_pip},
        "photon" : {"data" : output_photon, "table" : table_photon},
        "pi0" : {"data" : output_pi0, "table" : table_pi0},
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
    with PdfPages(outDir + "beam.pdf") as pdf:

        bar_data = []
        for tag in output_mc["pi_beam"]["value"]:
            bar_data.extend([tag] * output_mc["pi_beam"]["value"][tag])
        Plots.PlotBar(bar_data, xlabel = "True particle ID")
        pdf.savefig()

        if output_data is None:
            Plots.PlotBar(output_mc["pandora_tag"]["value"], xlabel = "Pandora tag")
        else:
            Plots.PlotBarComparision(output_mc["pandora_tag"]["value"], output_data["pandora_tag"]["value"], label_1 = "MC", label_2 = "Data", xlabel = "pandora beam tag", fraction = True, barlabel = True)
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["dxy"]["value"], output_mc["dxy"]["tags"], data2 = output_data["dxy"]["value"], bins = args.nbins, x_label = "$\delta_{xy}$", y_scale = "log", x_range = [0, 5], norm = norm)
        else:
            Plots.PlotTagged(output_mc["dxy"]["value"], output_mc["dxy"]["tags"], bins = args.nbins, x_label = "$\delta_{xy}$", y_scale = "log", x_range = [0, 5], norm = norm)
        Plots.DrawCutPosition(output_mc["dxy"]["cuts"][0], arrow_length = 1, face = "left")
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["dz"]["value"], output_mc["dz"]["tags"], data2 = output_data["dz"]["value"], bins = args.nbins, x_label = "$\delta_{z}$", y_scale = "log", x_range = [-10, 10], norm = norm)
        else:
            Plots.PlotTagged(output_mc["dz"]["value"], output_mc["dz"]["tags"], bins = args.nbins, x_label = "$\delta_{z}$", y_scale = "log", x_range = [-10, 10], norm = norm)
        Plots.DrawCutPosition(min(output_mc["dz"]["cuts"]), arrow_length = 1, face = "right")
        Plots.DrawCutPosition(max(output_mc["dz"]["cuts"]), arrow_length = 1, face = "left")
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["cos_theta"]["value"], output_mc["cos_theta"]["tags"], data2 = output_data["cos_theta"]["value"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = args.nbins, x_range = [0.9, 1], norm = norm)
        else:
            Plots.PlotTagged(output_mc["cos_theta"]["value"], output_mc["cos_theta"]["tags"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = args.nbins, x_range = [0.9, 1], norm = norm)

        Plots.DrawCutPosition(output_mc["cos_theta"]["cuts"][0], arrow_length = 0.02)
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["beam_endPos_z"]["value"], output_mc["beam_endPos_z"]["tags"], data2 = output_data["beam_endPos_z"]["value"], bins = args.nbins, x_label = "Beam end position z (cm)", x_range = [0, 700], norm = norm)
        else:
            Plots.PlotTagged(output_mc["beam_endPos_z"]["value"], output_mc["beam_endPos_z"]["tags"], bins = args.nbins, x_label = "Beam end position z (cm)", x_range = [0, 700], norm = norm)

        Plots.DrawCutPosition(output_mc["beam_endPos_z"]["cuts"][0], face = "left", arrow_length = 50)
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["michel_score"]["value"], output_mc["michel_score"]["tags"], data2 = output_data["michel_score"]["value"], x_range = (0, 1), y_scale = "log", bins = args.nbins, x_label = "Michel score", ncols = 2, norm = norm)
        else:
            Plots.PlotTagged(output_mc["michel_score"]["value"], output_mc["michel_score"]["tags"], x_range = (0, 1), y_scale = "log", bins = args.nbins, x_label = "Michel score", ncols = 2, norm = norm)
        Plots.DrawCutPosition(output_mc["michel_score"]["cuts"][0], face = "left")
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], data2 = output_data["median_dEdX"]["value"], bins = args.nbins, y_scale = "log", x_range = [0, 10], x_label = "Median $dE/dX$ (MeV/cm)", norm = norm)
        else:
            Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], bins = args.nbins, y_scale = "log", x_range = [0, 10], x_label = "Median $dE/dX$ (MeV/cm)", norm = norm)

        Plots.DrawCutPosition(output_mc["median_dEdX"]["cuts"][0], face = "left", arrow_length = 2)
        pdf.savefig()

        bar_data = []
        for t in output_mc["final_tags"]["tags"]:
            bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
        Plots.PlotBar(bar_data, xlabel = "True particle ID")
        pdf.savefig()
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
    with PdfPages(outDir + "piplus.pdf") as pdf:
        if "track_score_all" in output_mc:
            if output_data:
                Plots.PlotTagged(output_mc["track_score_all"]["value"], output_mc["track_score_all"]["tags"], data2 = output_data["track_score_all"]["value"], y_scale = "log", x_label = "track score", norm = norm)
            else:
                Plots.PlotTagged(output_mc["track_score_all"]["value"], output_mc["track_score_all"]["tags"], y_scale = "log", x_label = "track score", norm = norm)
            pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["track_score"]["value"], output_mc["track_score"]["tags"], data2 = output_data["track_score"]["value"], x_range = [0, 1], y_scale = "linear", bins = args.nbins, ncols = 4, x_label = "track score", norm = norm)
        else:
            Plots.PlotTagged(output_mc["track_score"]["value"], output_mc["track_score"]["tags"], x_range = [0, 1], y_scale = "linear", bins = args.nbins, ncols = 4, x_label = "track score", norm = norm)
        
        Plots.DrawCutPosition(output_mc["track_score"]["cuts"][0], face = "right")
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], data2 = output_data["nHits"]["value"], bins = args.nbins, ncols = 2, x_range = [0, 500], x_label = "nHits", norm = norm)
        else:
            Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], bins = args.nbins, ncols = 2, x_range = [0, 500], x_label = "nHits", norm = norm)
        pdf.savefig()

        Plots.PlotHist2DImshowMarginal(ak.ravel(output_mc["nHits"]["value"]), ak.ravel(output_mc["completeness"]["value"]), ylabel = "completeness", xlabel = "nHits", x_range = [0, 500], bins = 50, norm = "column", c_scale = "linear")
        Plots.DrawCutPosition(output_mc["nHits"]["cuts"][0], arrow_length = 100, arrow_loc = 0.1, color = "C6")
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], data2 = output_data["median_dEdX"]["value"], ncols = 2, x_range = [0, 5], x_label = "median $dEdX$ (MeV/cm)", bins = args.nbins, norm = norm)
        else:
            Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], ncols = 2, x_range = [0, 5], x_label = "median $dEdX$", bins = args.nbins, norm = norm)
        Plots.DrawCutPosition(min(output_mc["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "right")
        Plots.DrawCutPosition(max(output_mc["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "left")
        pdf.savefig()

        bar_data = []
        for t in output_mc["final_tags"]["tags"]:
            bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
        Plots.PlotBar(bar_data, xlabel = "true particle ID")
        pdf.savefig()
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
    with PdfPages(outDir + "photon.pdf") as pdf:    
        if output_data:
            Plots.PlotTagged(output_mc["em_score"]["value"], output_mc["em_score"]["tags"], data2 = output_data["em_score"]["value"], bins = args.nbins, x_range = [0, 1], ncols = 4, x_label = "em score", norm = norm)
        else:
            Plots.PlotTagged(output_mc["em_score"]["value"], output_mc["em_score"]["tags"], bins = args.nbins, x_range = [0, 1], ncols = 4, x_label = "em score", norm = norm)

        Plots.DrawCutPosition(output_mc["em_score"]["cuts"][0])
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], data2 = output_data["nHits"]["value"], bins = args.nbins, x_label = "number of hits", x_range = [0, 1000], norm = norm)
        else:
            Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], bins = args.nbins, x_label = "number of hits", x_range = [0, 1000], norm = norm)
        Plots.DrawCutPosition(output_mc["nHits"]["cuts"][0], arrow_length = 100)
        pdf.savefig()

        Plots.PlotHist2DImshowMarginal(ak.ravel(output_mc["nHits"]["value"]), ak.ravel(output_mc["nHits_completeness"]["value"]), bins = 50, y_range = [0, 1],x_range = [0, 800], ylabel = "completeness", xlabel = "number of hits", norm = "column")
        Plots.DrawCutPosition(80, flip = False, arrow_length = 100, color = "C6")
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["beam_separation"]["value"], output_mc["beam_separation"]["tags"], data2 = output_data["beam_separation"]["value"], bins = 31, x_range = [0, 93], x_label = "distance from PFO start to beam end position (cm)", norm = norm)
        else:
            Plots.PlotTagged(output_mc["beam_separation"]["value"], output_mc["beam_separation"]["tags"], bins = 31, x_range = [0, 93], x_label = "distance from PFO start to beam end position (cm)", norm = norm)
        Plots.DrawCutPosition(min(output_mc["beam_separation"]["cuts"]), arrow_length = 30)
        Plots.DrawCutPosition(max(output_mc["beam_separation"]["cuts"]), face = "left", arrow_length = 30)
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["impact_parameter"]["value"], output_mc["impact_parameter"]["tags"], data2 = output_data["impact_parameter"]["value"], bins = args.nbins, x_range = [0, 50], x_label = "impact parameter wrt beam (cm)", norm = norm)
        else:
            Plots.PlotTagged(output_mc["impact_parameter"]["value"], output_mc["impact_parameter"]["tags"], bins = args.nbins, x_range = [0, 50], x_label = "impact parameter wrt beam (cm)", norm = norm)
        Plots.DrawCutPosition(output_mc["impact_parameter"]["cuts"][0], arrow_length = 20, face = "left")
        pdf.savefig()

        Plots.PlotHist2DImshowMarginal(ak.ravel(output_mc["impact_parameter"]["value"]), ak.ravel(output_mc["impact_parameter_completeness"]["value"]), x_range = [0, 40], y_range = [0, 1], bins = 20, norm = "column", c_scale = "linear", ylabel = "completeness", xlabel = "impact parameter wrt beam (cm)")
        Plots.DrawCutPosition(20, arrow_length = 20, face = "left", color = "red")
        pdf.savefig()

        bar_data = []
        for t in output_mc["final_tags"]["tags"]:
            bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
        Plots.PlotBar(bar_data, xlabel = "true particle ID")
        pdf.savefig()
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

    with PdfPages(outDir + "pi0.pdf") as pdf:
        if output_data is not None:
            scale = ak.count(output_data["n_photons"]["value"]) / ak.count(output_mc["n_photons"]["value"])

            n_photons_scaled = []
            u, c = np.unique(output_mc["n_photons"]["value"], return_counts = True)
            for i, j in zip(u, c):
                n_photons_scaled.extend([i]* int(scale * j))

            Plots.PlotBarComparision(n_photons_scaled, output_data["n_photons"]["value"], xlabel = "number of $\pi^{0}$ photon candidates", label_1 = "MC", label_2 = "Data", fraction = True, barlabel = False)
        else:
            Plots.PlotBar(output_mc["n_photons"]["value"], xlabel = "number of $\pi^{0}$ photon candidates")
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["mass"]["value"], output_mc["mass"]["tags"], data2 = output_data["mass"]["value"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
        else:
            Plots.PlotTagged(output_mc["mass"]["value"], output_mc["mass"]["tags"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
        Plots.DrawCutPosition(min(output_mc["mass"]["cuts"]), face = "right", arrow_length = 50)
        Plots.DrawCutPosition(max(output_mc["mass"]["cuts"]), face = "left", arrow_length = 50)
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["mass"]["value"], output_mc["mass_event_tag"]["tags"], data2 = output_data["mass"]["value"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
        else:
            Plots.PlotTagged(output_mc["mass"]["value"], output_mc["mass_event_tag"]["tags"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
        Plots.DrawCutPosition(min(output_mc["mass"]["cuts"]), face = "right", arrow_length = 50)
        Plots.DrawCutPosition(max(output_mc["mass"]["cuts"]), face = "left", arrow_length = 50)
        pdf.savefig()

        if output_data:
            Plots.PlotTagged(output_mc["angle"]["value"], output_mc["angle"]["tags"], data2 = output_data["angle"]["value"], bins = args.nbins, x_label = "Opening angle (rad)", norm = norm)
        else:
            Plots.PlotTagged(output_mc["angle"]["value"], output_mc["angle"]["tags"], bins = args.nbins, x_label = "Opening angle (rad)", norm = norm)
        Plots.DrawCutPosition(min(output_mc["angle"]["cuts"]), face = "right", arrow_length = 0.25)
        Plots.DrawCutPosition(max(output_mc["angle"]["cuts"]), face = "left", arrow_length = 0.25)
        pdf.savefig()

    return


def MakeRegionPlots(outputs_mc_masks : dict, outputs_data_masks : dict, outDir : str):
    """ Correlation matrices for truth and reco region selection.

    Args:
        outputs_mc_masks (dict): mc masks for each region
        outputs_data_masks (dict): data masks for each region
        outDir (str): output directory
    """
    with PdfPages(outDir + "regions.pdf") as pdf:
        # Visualise the regions
        Plots.plot_region_data(outputs_mc_masks["truth_regions"], compare_max=0, title="truth regions")
        pdf.savefig()
        Plots.plot_region_data(outputs_mc_masks["reco_regions"], compare_max=0, title="reco regions")
        pdf.savefig()
        # Compare the regions
        Plots.compare_truth_reco_regions(outputs_mc_masks["reco_regions"], outputs_mc_masks["truth_regions"], title="")
        pdf.savefig()

        if outputs_data_masks is not None:
            Plots.plot_region_data(outputs_mc_masks["reco_regions"], compare_max=0, title="reco regions")
            pdf.savefig()


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
                    else:
                        for c in output[selection][o]:
                            if output[selection][o][c]["value"] is not None:
                                if c in ["pi_beam"]:
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
                            # we shouldn't need to merge cuts because this should be the same for all events

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

@Master.timer
def main(args):
    shower_merging.SetPlotStyle(extend_colors = True)

    func_args = vars(args)
    func_args["data"] = False

    output_mc = MergeOutputs(Processing.mutliprocess(run, [args.mc_file], args.batches, args.events, func_args, args.threads)) # run the main analysing method

    output_data = None
    # if args.data_file is not None:
    #     func_args["data"] = True
    #     output_data = MergeOutputs(Processing.mutliprocess(run, [args.data_file], args.batches, args.events, func_args, args.threads)) # run the main analysing method
    # tables
    MakeTables(output_mc, args.out + "tables_mc/", "mc")
    # if output_data is not None: MakeTables(output_data, args.out + "tables_data/", "data")

    # output directories
    os.makedirs(args.out + "plots/", exist_ok = True)

    # exit()
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

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)

    rprint(vars(args))
    main(args)