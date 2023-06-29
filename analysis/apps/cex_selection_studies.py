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


def AnalyseBeamSelection(events : Master.Data, beam_instrumentation : bool, beam_quality_fits : dict) -> dict:
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

    output["no_selection"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, Tags.GenerateTrueFinalStateTags(events))

    #* calo size cut
    mask = BeamParticleSelection.CaloSizeCut(events)
    output["calo_size"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None)
    events.Filter([mask], [mask])
    output["calo_size"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* pi+ beam selection
    mask = BeamParticleSelection.PiBeamSelection(events, beam_instrumentation)
    counts = Tags.GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i].mask)
    output["pi_beam"] = MakeOutput(counts, Tags.GenerateTrueBeamParticleTags(events), None)
    events.Filter([mask], [mask])
    output["pi_beam"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* beam pandora tag selection
    mask = BeamParticleSelection.PandoraTagCut(events) # create the mask for the cut
    output["pandora_tag"] = MakeOutput(events.recoParticles.beam_pandora_tag, Tags.GenerateTrueBeamParticleTags(events), [13]) # store the data to cut, cut value and truth tags
    events.Filter([mask], [mask]) # apply the cut
    output["pandora_tag"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events) # store the final state truth tags after the cut is applied

    #* dxy cut
    dxy = (((events.recoParticles.beam_startPos.x - beam_quality_fits["mu_x"]) / beam_quality_fits["sigma_x"])**2 + ((events.recoParticles.beam_startPos.y - beam_quality_fits["mu_y"]) / beam_quality_fits["sigma_y"])**2)**0.5
    mask = dxy < 3
    output["dxy"] = MakeOutput(dxy, Tags.GenerateTrueBeamParticleTags(events), [3], Tags.GenerateTrueFinalStateTags(events))
    print(f"dxy cut: {BeamParticleSelection.CountMask(mask)}")
    events.Filter([mask], [mask])
    output["dxy"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* dz cut
    delta_z = (events.recoParticles.beam_startPos.z - beam_quality_fits["mu_z"]) / beam_quality_fits["sigma_z"]
    mask = (delta_z > -3) & (delta_z < 3)
    output["dz"] = MakeOutput(delta_z, Tags.GenerateTrueBeamParticleTags(events), [-3, 3], Tags.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    print(f"dz cut: {BeamParticleSelection.CountMask(mask)}")
    output["dz"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* beam direction
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))
    beam_dir_mu = vector.normalize(vector.vector(beam_quality_fits["mu_dir_x"], beam_quality_fits["mu_dir_y"], beam_quality_fits["mu_dir_z"]))
    beam_costh = vector.dot(beam_dir, beam_dir_mu)
    mask = beam_costh > 0.95
    output["cos_theta"] = MakeOutput(beam_costh, Tags.GenerateTrueBeamParticleTags(events), [0.95], Tags.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["cos_theta"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* APA3 cut
    mask = BeamParticleSelection.APA3Cut(events)
    output["beam_endPos_z"] = MakeOutput(events.recoParticles.beam_endPos.z, Tags.GenerateTrueBeamParticleTags(events), [220], Tags.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["beam_endPos_z"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* michel score cut
    score = ak.where(events.recoParticles.beam_nHits != 0, events.recoParticles.beam_michelScore / events.recoParticles.beam_nHits, -999)
    mask = BeamParticleSelection.MichelScoreCut(events)
    output["michel_score"] = MakeOutput(score, Tags.GenerateTrueBeamParticleTags(events), [0.55], Tags.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["michel_score"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* median dE/dX
    mask = BeamParticleSelection.MedianDEdXCut(events)
    median = PFOSelection.Median(events.recoParticles.beam_dEdX)
    output["median_dEdX"] = MakeOutput(median, Tags.GenerateTrueBeamParticleTags(events), [2.4], Tags.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["median_dEdX"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, Tags.GenerateTrueFinalStateTags(events))
    return output


def AnalysePiPlusSelection(events : Master.Data) -> dict:
    """ Analyse the daughter pi+ selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}

    # this selection only needs to be done for shower merging ntuple because the PDSPAnalyser ntuples only store daughter PFOs, not the beam as well.
    if events.nTuple_type == Master.Ntuple_Type.SHOWER_MERGING:
        #* beam particle daughter selection 
        mask = PFOSelection.BeamDaughterCut(events)
        output["track_score_all"] = MakeOutput(events.recoParticles.trackScore, Tags.GenerateTrueParticleTags(events)) # keep a record of the track score to show the cosmic muon background
        events.Filter([mask])

    #* track score selection
    mask = PFOSelection.TrackScoreCut(events)
    output["track_score"] = MakeOutput(events.recoParticles.trackScore, Tags.GenerateTrueParticleTags(events), [0.5])
    events.Filter([mask])

    #* nHits cut
    mask = PFOSelection.NHitsCut(events, 20)
    output["nHits"] = MakeOutput(events.recoParticles.nHits, Tags.GenerateTrueParticleTags(events), [20])
    output["completeness"] = MakeOutput(events.trueParticlesBT.completeness, Tags.GenerateTrueParticleTags(events))
    events.Filter([mask])

    #* median dEdX    
    mask = PFOSelection.PiPlusSelection(events)
    output["median_dEdX"] = MakeOutput(PFOSelection.Median(events.recoParticles.track_dEdX), Tags.GenerateTrueParticleTags(events), [0.5, 2.8])
    events.Filter([mask])

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueParticleTags(events), None)
    return output


def AnalysePhotonCandidateSelection(events : Master.Data) -> dict:
    """ Analyse the photon candidate selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}

    #* em shower score cut
    output["em_score"] = MakeOutput(events.recoParticles.emScore, Tags.GenerateTrueParticleTags(events), [0.5])
    mask = PFOSelection.EMScoreCut(events, 0.5)
    events.Filter([mask])

    #* nHits cut
    output["nHits"] = MakeOutput(events.recoParticles.nHits, Tags.GenerateTrueParticleTags(events), [80])
    output["nHits_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [], [])
    mask = PFOSelection.NHitsCut(events, 80)
    events.Filter([mask])

    #* distance to beam cut
    dist = PFOSelection.find_beam_separations(events)
    output["beam_separation"] = MakeOutput(dist, Tags.GenerateTrueParticleTags(events), [3, 90])
    mask = PFOSelection.BeamParticleDistanceCut(events, [3, 90])
    events.Filter([mask])

    #* impact parameter
    ip = PFOSelection.find_beam_impact_parameters(events)
    output["impact_parameter"] = MakeOutput(ip, Tags.GenerateTrueParticleTags(events), [20])
    output["impact_parameter_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [], [])
    mask = PFOSelection.BeamParticleIPCut(events)
    events.Filter([mask])

    #* true particle population
    output["final_tags"] = MakeOutput(None, Tags.GenerateTrueParticleTags(events), None)
    return output


def AnalysePi0Selection(events : Master.Data, data : bool = False, correction = None, correction_params : dict = None) -> dict:
    """ Analyse the pi0 selection.

    Args:
        events (Master.Data): events to look at
        data (bool): is the ntuple file data or MC
        energy_correction_factor (float, optional): linear correction factor to shower energies. Defaults to 1.

    Returns:
        dict: output data
    """
    def null_tag():
        tag = Tags.Tags()
        tag["null"] = Tags.Tag(mask = (events.eventNum < -1))
        return tag

    output = {}

    photonCandidates = PFOSelection.InitialPi0PhotonSelection(events) # repeat the photon candidate selection, but only require the mask

    #* number of photon candidate
    n_photons = ak.sum(photonCandidates, -1)
    mask = n_photons == 2
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
    tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["mass"] = MakeOutput(mass, tags, [50, 250])
    output["mass_event_tag"] = MakeOutput(None, Tags.GenerateTrueFinalStateTags(events), None)
    events.Filter([mask], [mask])
    photonCandidates = photonCandidates[mask]

    #* opening angle
    shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photonCandidates)
    angle = ak.fill_none(ak.pad_none(shower_pairs.reco_angle, 1, -1), -999, -1)
    mask = (angle > (10 * np.pi / 180)) & (angle < (80 * np.pi / 180))
    mask = ak.flatten(mask) # 1 pi0
    tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["angle"] = MakeOutput(angle, tags, [(10 * np.pi / 180), (80 * np.pi / 180)])
    events.Filter([mask], [mask])
    photonCandidates = photonCandidates[mask]

    #* final counts
    output["event_tag"] = MakeOutput(None, Tags.GenerateTrueFinalStateTags(events), None)
    tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["final_tags"] = MakeOutput(None, tags, None) 

    return output

def AnalyseRegions(events : Master.Data, photon_mask : ak.Array, is_data : bool, correction = None, correction_params : dict = None):
    truth_regions = EventSelection.create_regions(events.trueParticles.nPi0, events.trueParticles.nPiPlus) if is_data == False else None

    reco_pi0_counts = EventSelection.count_pi0_candidates(events, exactly_two_photons = True, photon_mask = photon_mask, correction = correction, correction_params = correction_params)
    reco_pi_plus_counts_mom_cut = EventSelection.count_charged_pi_candidates(events, energy_cut = None)
    reco_regions = EventSelection.create_regions(reco_pi0_counts, reco_pi_plus_counts_mom_cut)
    return truth_regions, reco_regions


# @Processing.log_process
def run(i, file, n_events, start, selected_events, args):
    events = Master.Data(file, nEvents = n_events, start = start, nTuple_type = args["ntuple_type"]) # load data

    if args["data"] == True:
        fit_file = args["data_beam_quality_fit"]
    else:
        fit_file = args["mc_beam_quality_fit"]

    #* beam quality cuts
    with open(fit_file, "r") as f:
        fit_values = json.load(f)

    #* shower energy correction
    if args["correction_params"] is not None:
        with open(args["correction_params"], "r") as f:
            correction_params = json.load(f)
    else:
        correction_params = None

    print("beam particle selection")
    beam_selection_mask = BeamParticleSelection.CreateDefaultSelection(events, args["data"], fit_values, return_table = False) # make this premptively to save masks to file
    output_beam = AnalyseBeamSelection(events, args["data"], fit_values) # events are cut after this

    print("PFO pre-selection")
    good_PFO_mask = PFOSelection.GoodShowerSelection(events) 
    events.Filter([good_PFO_mask])

    print("pion selection")
    pi_plus_selection_mask = PFOSelection.DaughterPiPlusSelection(events)
    output_pip = AnalysePiPlusSelection(events.Filter(returnCopy = True)) # pass the PFO selections a copy of the event

    print("photon selection")
    photon_selection_mask = PFOSelection.InitialPi0PhotonSelection(events)
    output_photon = AnalysePhotonCandidateSelection(events.Filter(returnCopy = True))

    print("pi0 selection")
    pi0_selection_mask = EventSelection.Pi0Selection(events, photon_selection_mask, correction = args["correction"], correction_params = correction_params)
    output_pi0 = AnalysePi0Selection(events.Filter(returnCopy = True), args["data"], args["correction"], correction_params)

    print("regions")
    truth_regions, reco_regions = AnalyseRegions(events, photon_selection_mask, args["data"], args["correction"], correction_params)

    masks  = {
        "beam_selection"      : beam_selection_mask,
        "valid_pfo_selection" : good_PFO_mask,
        "pi_plus_selection"   : pi_plus_selection_mask,
        "photon_selection"    : photon_selection_mask,
        "pi0_selection"       : pi0_selection_mask,
        "truth_regions"       : truth_regions,
        "reco_regions"        : reco_regions
    } # keep masks which select events/PFOs for applying the selection without computing each cut every time

    output = {
        "beam" : output_beam,
        "pip" : output_pip,
        "photon" : output_photon,
        "pi0" : output_pi0,
        "masks" : masks
    }
    return output


def MakeBeamSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float):
    """ Beam particle selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
    """

    norm = False if output_data is None else norm

    bar_data = []
    for tag in output_mc["pi_beam"]["value"]:
        bar_data.extend([tag] * output_mc["pi_beam"]["value"][tag])
    Plots.PlotBar(bar_data, xlabel = "True particle ID")
    Plots.Save("pi_beam", outDir)

    if output_data is None:
        Plots.PlotBar(output_mc["pandora_tag"]["value"], xlabel = "Pandora tag")
    else:
        Plots.PlotBarComparision(output_mc["pandora_tag"]["value"], output_data["pandora_tag"]["value"], label_1 = "MC", label_2 = "Data", xlabel = "pandora beam tag", fraction = True, barlabel = True)
    Plots.Save("pandora_tag", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["michel_score"]["value"], output_mc["michel_score"]["tags"], data2 = output_data["michel_score"]["value"], x_range = (0, 1), y_scale = "log", bins = args.nbins, x_label = "Michel score", ncols = 2, norm = norm)
    else:
        Plots.PlotTagged(output_mc["michel_score"]["value"], output_mc["michel_score"]["tags"], x_range = (0, 1), y_scale = "log", bins = args.nbins, x_label = "Michel score", ncols = 2, norm = norm)
    Plots.DrawCutPosition(output_mc["michel_score"]["cuts"][0], face = "left")
    Plots.Save("michel_score", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["dxy"]["value"], output_mc["dxy"]["tags"], data2 = output_data["dxy"]["value"], bins = args.nbins, x_label = "$\delta_{xy}$", y_scale = "log", x_range = [0, 5], norm = norm)
    else:
        Plots.PlotTagged(output_mc["dxy"]["value"], output_mc["dxy"]["tags"], bins = args.nbins, x_label = "$\delta_{xy}$", y_scale = "log", x_range = [0, 5], norm = norm)
    Plots.DrawCutPosition(output_mc["dxy"]["cuts"][0], arrow_length = 1, face = "left")
    Plots.Save("dxy", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["dz"]["value"], output_mc["dz"]["tags"], data2 = output_data["dz"]["value"], bins = args.nbins, x_label = "$\delta_{z}$", y_scale = "log", x_range = [-10, 10], norm = norm)
    else:
        Plots.PlotTagged(output_mc["dz"]["value"], output_mc["dz"]["tags"], bins = args.nbins, x_label = "$\delta_{z}$", y_scale = "log", x_range = [-10, 10], norm = norm)
    Plots.DrawCutPosition(min(output_mc["dz"]["cuts"]), arrow_length = 1, face = "right")
    Plots.DrawCutPosition(max(output_mc["dz"]["cuts"]), arrow_length = 1, face = "left")
    Plots.Save("dz", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["cos_theta"]["value"], output_mc["cos_theta"]["tags"], data2 = output_data["cos_theta"]["value"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = args.nbins, x_range = [0.9, 1], norm = norm)
    else:
        Plots.PlotTagged(output_mc["cos_theta"]["value"], output_mc["cos_theta"]["tags"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = args.nbins, x_range = [0.9, 1], norm = norm)

    Plots.DrawCutPosition(output_mc["cos_theta"]["cuts"][0], arrow_length = 0.02)
    Plots.Save("cos_theta", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["beam_endPos_z"]["value"], output_mc["beam_endPos_z"]["tags"], data2 = output_data["beam_endPos_z"]["value"], bins = args.nbins, x_label = "Beam end position z (cm)", x_range = [0, 700], norm = norm)
    else:
        Plots.PlotTagged(output_mc["beam_endPos_z"]["value"], output_mc["beam_endPos_z"]["tags"], bins = args.nbins, x_label = "Beam end position z (cm)", x_range = [0, 700], norm = norm)

    Plots.DrawCutPosition(output_mc["beam_endPos_z"]["cuts"][0], face = "left", arrow_length = 50)
    Plots.Save("beam_endPos_z", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], data2 = output_data["median_dEdX"]["value"], bins = args.nbins, y_scale = "log", x_range = [0, 10], x_label = "Median $dE/dX$ (MeV/cm)", norm = norm)
    else:
        Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], bins = args.nbins, y_scale = "log", x_range = [0, 10], x_label = "Median $dE/dX$ (MeV/cm)", norm = norm)

    Plots.DrawCutPosition(output_mc["median_dEdX"]["cuts"][0], face = "left", arrow_length = 2)
    Plots.Save("median_dEdX", outDir)

    bar_data = []
    for t in output_mc["final_tags"]["tags"]:
        bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
    Plots.PlotBar(bar_data, xlabel = "True particle ID")
    Plots.Save("final_tags", outDir)
    return


def MakePiPlusSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float):
    """ Pi plus selection plots.

    Args:
        output (dict): data to plot
        outDir (str): output directory
    """

    norm = False if output_data is None else norm

    if "track_score_all" in output_mc:
        if output_data:
            Plots.PlotTagged(output_mc["track_score_all"]["value"], output_mc["track_score_all"]["tags"], data2 = output_data["track_score_all"]["value"], y_scale = "log", x_label = "track score", norm = norm)
        else:
            Plots.PlotTagged(output_mc["track_score_all"]["value"], output_mc["track_score_all"]["tags"], y_scale = "log", x_label = "track score", norm = norm)
        Plots.Save("track_score_all", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["track_score"]["value"], output_mc["track_score"]["tags"], data2 = output_data["track_score"]["value"], x_range = [0, 1], y_scale = "linear", bins = args.nbins, ncols = 4, x_label = "track score", norm = norm)
    else:
        Plots.PlotTagged(output_mc["track_score"]["value"], output_mc["track_score"]["tags"], x_range = [0, 1], y_scale = "linear", bins = args.nbins, ncols = 4, x_label = "track score", norm = norm)
    
    Plots.DrawCutPosition(output_mc["track_score"]["cuts"][0], face = "right")
    Plots.Save("track_score", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], data2 = output_data["nHits"]["value"], bins = args.nbins, ncols = 2, x_range = [0, 500], x_label = "nHits", norm = norm)
    else:
        Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], bins = args.nbins, ncols = 2, x_range = [0, 500], x_label = "nHits", norm = norm)
    
    Plots.Save("nHits", outDir)

    Plots.PlotHist2D(ak.ravel(output_mc["completeness"]["value"]), ak.ravel(output_mc["nHits"]["value"]), xlabel = "completeness", ylabel = "nHits", y_range = [0, 500], bins = args.nbins)
    Plots.DrawCutPosition(output_mc["nHits"]["cuts"][0], flip = True, arrow_length = 100, arrow_loc = 0.1, color = "red")
    Plots.Save("nHits_vs_completeness", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], data2 = output_data["median_dEdX"]["value"], ncols = 2, x_range = [0, 5], x_label = "median $dEdX$ (MeV/cm)", bins = args.nbins, norm = norm)
    else:
        Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], ncols = 2, x_range = [0, 5], x_label = "median $dEdX$", bins = args.nbins, norm = norm)
    Plots.DrawCutPosition(min(output_mc["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "right")
    Plots.DrawCutPosition(max(output_mc["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "left")
    Plots.Save("median_dEdX", outDir)


    bar_data = []
    for t in output_mc["final_tags"]["tags"]:
        bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
    Plots.PlotBar(bar_data, xlabel = "true particle ID")
    Plots.Save("true_particle_ID", outDir)
    return


def MakePhotonCandidateSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float):
    """ Photon candidate plots.

    Args:
        output (dict): data to plot
        outDir (str): output directory
    """
    
    norm = False if output_data is None else norm
    
    if output_data:
        Plots.PlotTagged(output_mc["em_score"]["value"], output_mc["em_score"]["tags"], data2 = output_data["em_score"]["value"], bins = args.nbins, x_range = [0, 1], ncols = 4, x_label = "em score", norm = norm)
    else:
        Plots.PlotTagged(output_mc["em_score"]["value"], output_mc["em_score"]["tags"], bins = args.nbins, x_range = [0, 1], ncols = 4, x_label = "em score", norm = norm)

    Plots.DrawCutPosition(output_mc["em_score"]["cuts"][0])
    Plots.Save("em_score", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], data2 = output_data["nHits"]["value"], bins = args.nbins, x_label = "number of hits", x_range = [0, 1000], norm = norm)
    else:
        Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], bins = args.nbins, x_label = "number of hits", x_range = [0, 1000], norm = norm)
    Plots.DrawCutPosition(output_mc["nHits"]["cuts"][0], arrow_length = 100)
    Plots.Save("nHits", outDir)

    Plots.PlotHist2D(ak.ravel(output_mc["nHits_completeness"]["value"]), ak.ravel(output_mc["nHits"]["value"]), bins = args.nbins, x_range = [0, 1],y_range = [0, 1000], xlabel = "completeness", ylabel = "number of hits")
    Plots.DrawCutPosition(output_mc["nHits"]["cuts"][0], flip = True, arrow_length = 100, color = "red")
    Plots.Save("nHits_vs_completeness", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["beam_separation"]["value"], output_mc["beam_separation"]["tags"], data2 = output_data["beam_separation"]["value"], bins = args.nbins, x_range = [0, 150], x_label = "distance from PFO start to beam end position (cm)", norm = norm)
    else:
        Plots.PlotTagged(output_mc["beam_separation"]["value"], output_mc["beam_separation"]["tags"], bins = args.nbins, x_range = [0, 150], x_label = "distance from PFO start to beam end position (cm)", norm = norm)
    Plots.DrawCutPosition(min(output_mc["beam_separation"]["cuts"]), arrow_length = 30)
    Plots.DrawCutPosition(max(output_mc["beam_separation"]["cuts"]), face = "left", arrow_length = 30)
    Plots.Save("beam_separation", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["impact_parameter"]["value"], output_mc["impact_parameter"]["tags"], data2 = output_data["impact_parameter"]["value"], bins = args.nbins, x_range = [0, 50], x_label = "impact parameter wrt beam (cm)", norm = norm)
    else:
        Plots.PlotTagged(output_mc["impact_parameter"]["value"], output_mc["impact_parameter"]["tags"], bins = args.nbins, x_range = [0, 50], x_label = "impact parameter wrt beam (cm)", norm = norm)
    Plots.DrawCutPosition(output_mc["impact_parameter"]["cuts"][0], arrow_length = 20, face = "left")
    Plots.Save("impact_parameter", outDir)

    Plots.PlotHist2D(ak.ravel(output_mc["impact_parameter_completeness"]["value"]), ak.ravel(output_mc["impact_parameter"]["value"]), bins = args.nbins, x_range = [0, 1], y_range = [0, 50], xlabel = "completeness", ylabel = "impact parameter wrt beam (cm)")
    Plots.DrawCutPosition(output_mc["impact_parameter"]["cuts"][0], arrow_loc = 0.2, arrow_length = 20, face = "left", flip = True, color ="red")
    Plots.Save("impact_parameter_vs_completeness", outDir)


    bar_data = []
    for t in output_mc["final_tags"]["tags"]:
        bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
    Plots.PlotBar(bar_data, xlabel = "true particle ID")
    Plots.Save("true_particle_ID", outDir)
    return


def MakePi0SelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float):
    """ Pi0 selection plots.

    Args:
        output (dict): data to plot
        outDir (str): output directory
    """
    norm = False if output_data is None else norm

    if output_data is not None:
        scale = ak.count(output_data["n_photons"]["value"]) / ak.count(output_mc["n_photons"]["value"])

        n_photons_scaled = []
        u, c = np.unique(output_mc["n_photons"]["value"], return_counts = True)
        for i, j in zip(u, c):
            n_photons_scaled.extend([i]* int(scale * j))

        Plots.PlotBarComparision(n_photons_scaled, output_data["n_photons"]["value"], xlabel = "number of $\pi^{0}$ photon candidates", label_1 = "MC", label_2 = "Data", fraction = True, barlabel = False)
    else:
        Plots.PlotBar(output_mc["n_photons"]["value"], xlabel = "number of $\pi^{0}$ photon candidates")
    Plots.Save("n_photons", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["angle"]["value"], output_mc["angle"]["tags"], data2 = output_data["angle"]["value"], bins = args.nbins, x_label = "Opening angle (rad)", norm = norm)
    else:
        Plots.PlotTagged(output_mc["angle"]["value"], output_mc["angle"]["tags"], bins = args.nbins, x_label = "Opening angle (rad)", norm = norm)
    Plots.DrawCutPosition(min(output_mc["angle"]["cuts"]), face = "right", arrow_length = 0.25)
    Plots.DrawCutPosition(max(output_mc["angle"]["cuts"]), face = "left", arrow_length = 0.25)
    Plots.Save("angle", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["mass"]["value"], output_mc["mass"]["tags"], data2 = output_data["mass"]["value"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
    else:
        Plots.PlotTagged(output_mc["mass"]["value"], output_mc["mass"]["tags"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
    Plots.DrawCutPosition(min(output_mc["mass"]["cuts"]), face = "right", arrow_length = 50)
    Plots.DrawCutPosition(max(output_mc["mass"]["cuts"]), face = "left", arrow_length = 50)
    Plots.Save("mass_pi0_tags", outDir)

    if output_data:
        Plots.PlotTagged(output_mc["mass"]["value"], output_mc["mass_event_tag"]["tags"], data2 = output_data["mass"]["value"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
    else:
        Plots.PlotTagged(output_mc["mass"]["value"], output_mc["mass_event_tag"]["tags"], bins = args.nbins, x_label = "Invariant mass (MeV)", x_range = [0, 500], norm = norm)
    Plots.DrawCutPosition(min(output_mc["mass"]["cuts"]), face = "right", arrow_length = 50)
    Plots.DrawCutPosition(max(output_mc["mass"]["cuts"]), face = "left", arrow_length = 50)
    Plots.Save("mass_fs_tags", outDir)
    return

def MakeRegionPlots(outputs_mc_masks : dict, outputs_data_masks : dict, out : str):
    # Visualise the regions
    Plots.plot_region_data(outputs_mc_masks["truth_regions"], compare_max=0, title="truth regions")
    Plots.Save(out + "mc_truth_regions")
    Plots.plot_region_data(outputs_mc_masks["reco_regions"], compare_max=0, title="reco regions")
    Plots.Save(out + "mc_reco_regions")
    # Compare the regions
    Plots.compare_truth_reco_regions(outputs_mc_masks["reco_regions"], outputs_mc_masks["truth_regions"], title="")
    Plots.Save(out + "mc_truth_vs_reco_regions")

    if outputs_data_masks is not None:
        Plots.plot_region_data(outputs_mc_masks["reco_regions"], compare_max=0, title="reco regions")
        Plots.Save(out + "data_reco_regions")


def CalcualteCountMetrics(counts : pd.DataFrame) -> tuple:
    """ Calculates purity and efficiency from event counts.

    Args:
        counts (pd.DataFrame): event counts dataframe

    Returns:
        tuple: purity and efficiency data frames
    """
    efficiency = counts.div(counts.no_selection, 0)
    purity = counts / counts.loc["total"]
    return purity, efficiency    


def MakeFinalStateTables(output : dir, outDir : str):
    """ Make performance tables based on the final state truth tags.

    Args:
        output (dir): output data
        outDir (str): save location
    """
    table = {}

    for o in output:
        if o == "final_tags" : continue
        row = {}
        for t in output[o]["fs_tags"]:
            count = BeamParticleSelection.CountMask(output[o]["fs_tags"][t].mask)
            row["total"] = count[0]
            row[t] = count[1]

        table[o] = row
    event_counts = pd.DataFrame(table)
    purity, efficiency = CalcualteCountMetrics(event_counts)
    rprint(event_counts)
    rprint(purity)
    rprint(efficiency)

    event_counts.T.to_latex(outDir + "final_state_event_counts.tex")
    purity.T.to_latex(outDir + "final_state_purity.tex")
    efficiency.T.to_latex(outDir + "final_state_efficiency.tex")
    return


def PiPlusBranchingFractions(output : dir, outDir : str):
    """ Make performance tables which roughly represents the branching fractions of
        pi+ + Ar interactions.

    Args:
        output (dir): output data
        outDir (str): save location
    """
    table = {}

    for o in ["no_selection", "final_tags"]:
        row = {}
        pi_mask = output[o]["tags"]["$\\pi^{+}$:inel"].mask
        for t in output[o]["fs_tags"]:
            count = BeamParticleSelection.CountMask(output[o]["fs_tags"][t].mask & pi_mask)
            row["total"] = ak.sum(pi_mask)
            row[t] = count[1]
        table[o] = row
    event_counts = pd.DataFrame(table)
    purity, efficiency = CalcualteCountMetrics(event_counts)
    rprint(event_counts)
    rprint(purity)
    rprint(efficiency)

    event_counts.T.to_latex(outDir + "pi_final_state_counts.tex")
    purity.T.to_latex(outDir + "pi_final_state_purity.tex")
    efficiency.T.to_latex(outDir + "pi_final_state_efficiency.tex")


def MakeParticleTables(output : dir, outDir : str, exclude = [], type : str = "mc"):
    """ Make performance tables based on the true particle tags.

    Args:
        output (dir): output data
        outDir (str): save location
    """
    table = {}

    for o in output:
        if o in exclude: continue
        row = {"total" : 0}
        print(o)
        for t in output[o]["tags"]:
            if output[o]["tags"][t] is None: continue
            count = PFOSelection.CountMask(output[o]["tags"][t].mask)
            row["total"] = count[0]
            row[t] = count[1]
            table[o] = row
    event_counts = pd.DataFrame(table)

    event_counts.columns = event_counts.columns[:-1].insert(0, "no_selection")

    rprint(event_counts)
    event_counts.T.to_latex(outDir + "particle_counts.tex")
    if type == "mc":
        purity, efficiency = CalcualteCountMetrics(event_counts)
        rprint(purity)
        rprint(efficiency)
        purity.T.to_latex(outDir + "particle_purity.tex")
        efficiency.T.to_latex(outDir + "particle_efficiency.tex")
    return


def MergeOutputs(outputs : list) -> dict:
    """ Merge multiprocessing output into a single dictionary.

    Args:
        outputs (list): outputs from multiprocessing job.

    Returns:
        dict: merged output
    """
    rprint("outputs")
    rprint(outputs[0])

    merged_output = {}

    for output in outputs:
        for selection in output:
            if selection not in merged_output:
                merged_output[selection] = {}
            print(selection)
            if selection == "masks":
                for o in output[selection]:
                    if o not in merged_output[selection]:
                        merged_output[selection][o] = output[selection][o]
                        continue
                    print(o)
                    if output[selection][o] is None: continue
                    elif type(output[selection][o]) == dict:
                        for m in output[selection][o]:
                            print(merged_output[selection][o][m])
                            print(output[selection][o][m])
                            merged_output[selection][o][m] = ak.concatenate([merged_output[selection][o][m], output[selection][o][m]])
                    else:
                        merged_output[selection][o] = ak.concatenate([merged_output[selection][o], output[selection][o]])
            else:
                for o in output[selection]:
                    if o not in merged_output[selection]:
                        merged_output[selection][o] = output[selection][o]
                        continue
                    else:
                        print(o)
                        if output[selection][o]["value"] is not None:
                            if o in ["pi_beam"]:
                                for i in merged_output[selection][o]["value"]:
                                    merged_output[selection][o]["value"][i] = merged_output[selection][o]["value"][i] + output[selection][o]["value"][i] 
                            else:
                                merged_output[selection][o]["value"] = ak.concatenate([merged_output[selection][o]["value"], output[selection][o]["value"]]) 

                        if output[selection][o]["tags"] is not None:
                            for t in merged_output[selection][o]["tags"]:
                                merged_output[selection][o]["tags"][t].mask = ak.concatenate([merged_output[selection][o]["tags"][t].mask, output[selection][o]["tags"][t].mask])

                        if output[selection][o]["fs_tags"] is not None:
                            for t in merged_output[selection][o]["fs_tags"]:
                                merged_output[selection][o]["fs_tags"][t].mask = ak.concatenate([merged_output[selection][o]["fs_tags"][t].mask, output[selection][o]["fs_tags"][t].mask])
                        # we shouldn't need to merge cuts because this should be the same for all events

    rprint("merged_output")
    rprint(merged_output)
    return merged_output


def MakeTables(output : dict, out : str, sample : "str"):
    # output directories
    os.makedirs(out + "daughter_pi/", exist_ok = True)
    os.makedirs(out + "beam/", exist_ok = True)
    os.makedirs(out + "photon/", exist_ok = True)
    os.makedirs(out + "pi0/", exist_ok = True)

    MakeFinalStateTables(output["beam"], out + "beam/")
    MakeParticleTables(output["pip"]       , out + "daughter_pi/", ["completeness"], sample)
    MakeParticleTables(output["photon"]    , out + "photon/", ["nHits_completeness", "impact_parameter_completeness"], sample)
    MakeParticleTables(output["pi0"]       , out + "pi0/", ["n_photons", "event_tag", "mass_event_tag"], sample)
    if sample == "mc":
        MakeParticleTables(output["beam"]      , out + "beam/", ["no_selection"], sample)
        PiPlusBranchingFractions(output["beam"], out + "beam/")
    return

@Master.timer
def main(args):
    shower_merging.SetPlotStyle(extend_colors = True)

    func_args = vars(args)
    func_args["data"] = False
    print(func_args)

    output_mc = MergeOutputs(Processing.mutliprocess(run, [args.mc_file], args.batches, args.events, func_args, args.threads)) # run the main analysing method

    output_data = None
    if args.data_file is not None:
        func_args["data"] = True
        output_data = MergeOutputs(Processing.mutliprocess(run, [args.data_file], args.batches, args.events, func_args, args.threads)) # run the main analysing method

    # tables
    MakeTables(output_mc, args.out + "tables_mc/", "mc")
    if output_data is not None: MakeTables(output_data, args.out + "tables_data/", "data")

    # output directories
    os.makedirs(args.out + "plots/daughter_pi/", exist_ok = True)
    os.makedirs(args.out + "plots/beam/", exist_ok = True)
    os.makedirs(args.out + "plots/photon/", exist_ok = True)
    os.makedirs(args.out + "plots/pi0/", exist_ok = True)
    os.makedirs(args.out + "plots/regions/", exist_ok = True)

    # plots
    MakeBeamSelectionPlots(output_mc["beam"], output_data["beam"] if output_data else None, args.out + "plots/beam/", norm = args.norm)
    MakePiPlusSelectionPlots(output_mc["pip"], output_data["pip"] if output_data else None, args.out + "plots/daughter_pi/", norm = args.norm)
    MakePhotonCandidateSelectionPlots(output_mc["photon"], output_data["photon"] if output_data else None, args.out + "plots/photon/", norm = args.norm)
    MakePi0SelectionPlots(output_mc["pi0"], output_data["pi0"] if output_data else None, args.out + "plots/pi0/", norm = args.norm)
    MakeRegionPlots(output_mc["masks"], output_data["masks"] if output_data else None, args.out + "plots/regions/")

    # masks
    cross_section.SaveSelection(args.out + args.mc_file[0].split("/")[-1].split(".")[0] + "_selection_masks.dill", output_mc["masks"])
    if output_data:
        cross_section.SaveSelection(args.out + args.data_file[0].split("/")[-1].split(".")[0] + "_selection_masks.dill",output_data["masks"])
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