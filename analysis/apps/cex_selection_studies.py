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
from python.analysis import Master, BeamParticleSelection, PFOSelection, Plots, shower_merging, vector, Processing, Tags

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


def AnalyseBeamSelection(events : Master.Data, beam_instrumentation : bool, beam_quality_fits : str) -> dict:
    """ Manually applies the beam selection while storing the value being cut on, cut values and truth tags in order to do plotting
        and produce performance tables.

    Args:
        events (Master.Data): events to look at
        beam_instrumentation (bool): use beam instrumentation for beam particle selection
        beam_quality_fits (str): fit values for the beam quality selection

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

    #* beam quality cuts
    with open(beam_quality_fits, "r") as f:
        fit_values = json.load(f)

    #* dxy cut
    dxy = (((events.recoParticles.beam_startPos.x - fit_values["mu_x"]) / fit_values["sigma_x"])**2 + ((events.recoParticles.beam_startPos.y - fit_values["mu_y"]) / fit_values["sigma_y"])**2)**0.5
    mask = dxy < 3
    output["dxy"] = MakeOutput(dxy, Tags.GenerateTrueBeamParticleTags(events), [3], Tags.GenerateTrueFinalStateTags(events))
    print(f"dxy cut: {BeamParticleSelection.CountMask(mask)}")
    events.Filter([mask], [mask])
    output["dxy"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* dz cut
    delta_z = (events.recoParticles.beam_startPos.z - fit_values["mu_z"]) / fit_values["sigma_z"]
    mask = (delta_z > -3) & (delta_z < 3)
    output["dz"] = MakeOutput(delta_z, Tags.GenerateTrueBeamParticleTags(events), [-3, 3], Tags.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    print(f"dz cut: {BeamParticleSelection.CountMask(mask)}")
    output["dz"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* beam direction
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))
    beam_dir_mu = vector.normalize(vector.vector(fit_values["mu_dir_x"], fit_values["mu_dir_y"], fit_values["mu_dir_z"]))
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


def AnalysePi0Selection(events : Master.Data, data : bool = False, energy_correction_factor : float = 1) -> dict:
    """ Analyse the pi0 selection.

    Args:
        events (Master.Data): events to look at
        data (bool): is the ntuple file data or MC
        energy_correction_factor (float, optional): linear correction factor to shower energies. Defaults to 1.

    Returns:
        dict: output data
    """
    output = {}

    photonCandidates = PFOSelection.InitialPi0PhotonSelection(events) # repeat the photon candidate selection, but only require the mask

    #* number of photon candidate
    n_photons = ak.sum(photonCandidates, -1)
    mask = n_photons == 2
    output["n_photons"] = MakeOutput(n_photons, None, [2])
    events.Filter([mask], [mask]) # technically is an event level cut as we only try to find 1 pi0 in the final state, two is more complicated
    photonCandidates = photonCandidates[mask]
    
    #* opening angle
    shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photonCandidates)
    angle = ak.flatten(shower_pairs.reco_angle)
    mask = (angle > (10 * np.pi / 180)) & (angle < (80 * np.pi / 180))
    tags = None if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["angle"] = MakeOutput(angle, tags, [(10 * np.pi / 180), (80 * np.pi / 180)])
    events.Filter([mask], [mask])
    photonCandidates = photonCandidates[mask]

    #* invariant mass
    shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photonCandidates)
    mass = ak.flatten(shower_pairs.reco_mass) / energy_correction_factor # see shower_correction.ipynb
    mask = (mass > 50) & (mass < 250)
    tags = None if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["mass"] = MakeOutput(mass, tags, [50, 250])
    output["event_tag"] = MakeOutput(None, Tags.GenerateTrueFinalStateTags(events), None)
    events.Filter([mask], [mask])
    photonCandidates = photonCandidates[mask]

    return output

# @Processing.log_process
def run(i, file, n_events, start, selected_events, args):
    events = Master.Data(file, nEvents = n_events, start = start, nTuple_type = args["ntuple_type"]) # load data

    if args["data"] == True:
        fit_file = args["data_beam_quality_fit"]
    else:
        fit_file = args["mc_beam_quality_fit"]

    print("beam particle selection")
    output_beam = AnalyseBeamSelection(events, args["data"], fit_file) # events are cut after this

    print("PFO pre-selection")
    events.Filter([PFOSelection.GoodShowerSelection(events)])

    print("pion selection")
    output_pip = AnalysePiPlusSelection(events.Filter(returnCopy = True)) # pass the PFO selections a copy of the event

    print("photon selection")
    output_photon = AnalysePhotonCandidateSelection(events.Filter(returnCopy = True))

    print("pi0 selection")
    output_pi0 = AnalysePi0Selection(events.Filter(returnCopy = True), args["data"], args["shower_correction_factor"])

    output = {
        "beam" : output_beam,
        "pip" : output_pip,
        "photon" : output_photon,
        "pi0" : output_pi0
    }

    return output


def MakeBeamSelectionPlots(output_mc : dict, output_data : dict, outDir : str):
    """ Beam particle selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
    """

    norm = False if output_data is None else True

    bar_data = []
    for tag in output_mc["pi_beam"]["value"]:
        bar_data.extend([tag] * output_mc["pi_beam"]["value"][tag])
    Plots.PlotBar(bar_data, xlabel = "True particle ID")
    Plots.Save("pi_beam", outDir)

    if output_data is None:
        Plots.PlotBar(output_mc["pandora_tag"]["value"], xlabel = "Pandora tag")
    else:
        scale = ak.count(output_data["pandora_tag"]["value"]) / ak.count(output_mc["pandora_tag"]["value"])

        pandora_tag_scaled = []
        u, c = np.unique(output_mc["pandora_tag"]["value"], return_counts = True)
        for i, j in zip(u, c):
            pandora_tag_scaled.extend([i]* int(scale * j))

        Plots.PlotBarComparision(pandora_tag_scaled, output_data["pandora_tag"]["value"], label_1 = "MC", label_2 = "Data", xlabel = "pandora tag")
    Plots.Save("pandora_tag", outDir)

    Plots.PlotTagged(output_mc["michel_score"]["value"], output_mc["michel_score"]["tags"], data2 = output_data["michel_score"]["value"], x_range = (0, 1), y_scale = "log", bins = args.nbins, x_label = "Michel score", ncols = 3, norm = norm)
    Plots.DrawCutPosition(output_mc["michel_score"]["cuts"][0], face = "left")
    Plots.Save("michel_score", outDir)

    Plots.PlotTagged(output_mc["dxy"]["value"], output_mc["dxy"]["tags"], data2 = output_data["dxy"]["value"], bins = args.nbins, x_label = "$dxy$", y_scale = "log", x_range = [0, 5], norm = norm)
    Plots.DrawCutPosition(output_mc["dxy"]["cuts"][0], arrow_length = 1, face = "left")
    Plots.Save("dxy", outDir)

    Plots.PlotTagged(output_mc["dz"]["value"], output_mc["dz"]["tags"], data2 = output_data["dz"]["value"], bins = args.nbins, x_label = "$dz$", y_scale = "log", x_range = [-10, 10], norm = norm)
    Plots.DrawCutPosition(min(output_mc["dz"]["cuts"]), arrow_length = 1, face = "right")
    Plots.DrawCutPosition(max(output_mc["dz"]["cuts"]), arrow_length = 1, face = "left")
    Plots.Save("dz", outDir)

    Plots.PlotTagged(output_mc["cos_theta"]["value"], output_mc["cos_theta"]["tags"], data2 = output_data["cos_theta"]["value"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = args.nbins, x_range = [0.9, 1], norm = norm)
    Plots.DrawCutPosition(output_mc["cos_theta"]["cuts"][0], arrow_length = 0.02)
    Plots.Save("cos_theta", outDir)

    Plots.PlotTagged(output_mc["beam_endPos_z"]["value"], output_mc["beam_endPos_z"]["tags"], data2 = output_data["beam_endPos_z"]["value"], bins = args.nbins, x_label = "Beam end position z (cm)", x_range = [-100, 600], norm = norm)
    Plots.DrawCutPosition(output_mc["beam_endPos_z"]["cuts"][0], face = "left", arrow_length = 50)
    Plots.Save("beam_endPos_z", outDir)

    Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], data2 = output_data["median_dEdX"]["value"], bins = args.nbins, y_scale = "log", x_range = [0, 10], x_label = "Median $dE/dX$ (MeV/cm)", norm = norm)
    Plots.DrawCutPosition(output_mc["median_dEdX"]["cuts"][0], face = "left", arrow_length = 2)
    Plots.Save("median_dEdX", outDir)

    bar_data = []
    for t in output_mc["final_tags"]["tags"]:
        bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
    Plots.PlotBar(bar_data, xlabel = "True particle ID")
    Plots.Save("final_tags", outDir)
    return


def MakePiPlusSelectionPlots(output_mc : dict, output_data : dict, outDir : str):
    """ Pi plus selection plots.

    Args:
        output (dict): data to plot
        outDir (str): output directory
    """

    norm = False if output_data is None else True

    if "track_score_all" in output_mc:
        Plots.PlotTagged(output_mc["track_score_all"]["value"], output_mc["track_score_all"]["tags"], data2 = output_data["track_score_all"]["value"], y_scale = "log", x_label = "track score", norm = norm)
        Plots.Save("track_score_all", outDir)

    Plots.PlotTagged(output_mc["track_score"]["value"], output_mc["track_score"]["tags"], data2 = output_data["track_score"]["value"], x_range = [0, 1], y_scale = "log", bins = args.nbins, ncols = 5, x_label = "track score", norm = norm)
    Plots.DrawCutPosition(output_mc["track_score"]["cuts"][0], face = "right")
    Plots.Save("track_score", outDir)


    Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], data2 = output_data["nHits"]["value"], bins = args.nbins, ncols = 5, x_range = [0, 500], x_label = "nHits", norm = norm)
    Plots.Save("nHits", outDir)

    Plots.PlotHist2D(ak.ravel(output_mc["completeness"]["value"]), ak.ravel(output_mc["nHits"]["value"]), xlabel = "completeness", ylabel = "nHits", y_range = [0, 500], bins = args.nbins)
    Plots.DrawCutPosition(output_mc["nHits"]["cuts"][0], flip = True, arrow_length = 100, arrow_loc = 0.1, color = "red")
    Plots.Save("nHits_vs_completeness", outDir)

    Plots.PlotTagged(output_mc["median_dEdX"]["value"], output_mc["median_dEdX"]["tags"], data2 = output_data["median_dEdX"]["value"], ncols = 3, x_range = [0, 6], x_label = "median $dEdX$", bins = args.nbins, norm = norm)
    Plots.DrawCutPosition(min(output_mc["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "right")
    Plots.DrawCutPosition(max(output_mc["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "left")
    Plots.Save("median_dEdX", outDir)


    bar_data = []
    for t in output_mc["final_tags"]["tags"]:
        bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
    Plots.PlotBar(bar_data, xlabel = "true particle ID")
    Plots.Save("true_particle_ID", outDir)
    return

def MakePhotonCandidateSelectionPlots(output_mc : dict, output_data : dict, outDir : str):
    """ Photon candidate plots.

    Args:
        output (dict): data to plot
        outDir (str): output directory
    """
    
    norm = False if output_data is None else True
    
    Plots.PlotTagged(output_mc["em_score"]["value"], output_mc["em_score"]["tags"], data2 = output_data["em_score"]["value"], bins = args.nbins, x_range = [0, 1], ncols = 5, x_label = "em score", norm = norm)
    Plots.DrawCutPosition(output_mc["em_score"]["cuts"][0])
    Plots.Save("em_score", outDir)

    Plots.PlotTagged(output_mc["nHits"]["value"], output_mc["nHits"]["tags"], data2 = output_data["nHits"]["value"], bins = args.nbins, x_label = "number of hits", x_range = [0, 1000], norm = norm)
    Plots.DrawCutPosition(output_mc["nHits"]["cuts"][0], arrow_length = 100)
    Plots.Save("nHits", outDir)

    Plots.PlotHist2D(ak.ravel(output_mc["nHits_completeness"]["value"]), ak.ravel(output_mc["nHits"]["value"]), bins = args.nbins, x_range = [0, 1],y_range = [0, 1000], xlabel = "completeness", ylabel = "number of hits")
    Plots.DrawCutPosition(output_mc["nHits"]["cuts"][0], flip = True, arrow_length = 100, color = "red")
    Plots.Save("nHits_vs_completeness", outDir)

    Plots.PlotTagged(output_mc["beam_separation"]["value"], output_mc["beam_separation"]["tags"], data2 = output_data["beam_separation"]["value"], bins = args.nbins, x_range = [0, 200], x_label = "distance from PFO start to beam end position (cm)", norm = norm)
    Plots.DrawCutPosition(min(output_mc["beam_separation"]["cuts"]), arrow_length = 30)
    Plots.DrawCutPosition(max(output_mc["beam_separation"]["cuts"]), face = "left", arrow_length = 30)
    Plots.Save("beam_separation", outDir)

    Plots.PlotTagged(output_mc["impact_parameter"]["value"], output_mc["impact_parameter"]["tags"], data2 = output_data["impact_parameter"]["value"], bins = args.nbins, x_label = "impact parameter wrt beam (cm)", norm = norm)
    Plots.DrawCutPosition(output_mc["impact_parameter"]["cuts"][0], arrow_length = 20, face = "left")
    Plots.Save("impact_parameter", outDir)

    Plots.PlotHist2D(ak.ravel(output_mc["impact_parameter_completeness"]["value"]), ak.ravel(output_mc["impact_parameter"]["value"]), bins = args.nbins, x_range = [0, 1], xlabel = "completeness", ylabel = "impact parameter wrt beam (cm)")
    Plots.DrawCutPosition(output_mc["impact_parameter"]["cuts"][0], arrow_loc = 0.2, arrow_length = 20, face = "left", flip = True, color ="red")
    Plots.Save("impact_parameter_vs_completeness", outDir)


    bar_data = []
    for t in output_mc["final_tags"]["tags"]:
        bar_data.extend([t] * ak.sum(output_mc["final_tags"]["tags"][t].mask))
    Plots.PlotBar(bar_data, xlabel = "true particle ID")
    Plots.Save("true_particle_ID", outDir)
    return


def MakePi0SelectionPlots(output_mc : dict, output_data : dict, outDir : str):
    """ Pi0 selection plots.

    Args:
        output (dict): data to plot
        outDir (str): output directory
    """
    norm = False if output_data is None else True

    if output_data is not None:
        scale = ak.count(output_data["n_photons"]["value"]) / ak.count(output_mc["n_photons"]["value"])

        n_photons_scaled = []
        u, c = np.unique(output_mc["n_photons"]["value"], return_counts = True)
        for i, j in zip(u, c):
            n_photons_scaled.extend([i]* int(scale * j))

        Plots.PlotBarComparision(n_photons_scaled, output_data["n_photons"]["value"], xlabel = "number of $\pi^{0}$ photon candidates", label_1 = "MC", label_2 = "Data")
    else:
        Plots.PlotBar(output_mc["n_photons"]["value"], xlabel = "number of $\pi^{0}$ photon candidates")
    Plots.Save("n_photons", outDir)

    Plots.PlotTagged(output_mc["angle"]["value"], output_mc["angle"]["tags"], data2 = output_data["angle"]["value"], bins = args.nbins, x_label = "Opening angle (rad)", norm = norm)
    Plots.DrawCutPosition(min(output_mc["angle"]["cuts"]), face = "right", arrow_length = 0.25)
    Plots.DrawCutPosition(max(output_mc["angle"]["cuts"]), face = "left", arrow_length = 0.25)
    Plots.Save("angle", outDir)

    Plots.PlotTagged(output_mc["mass"]["value"], output_mc["mass"]["tags"], data2 = output_data["mass"]["value"], bins = args.nbins, x_label = "Invariant mass (MeV)", norm = norm)
    Plots.DrawCutPosition(min(output_mc["mass"]["cuts"]), face = "right", arrow_length = 50)
    Plots.DrawCutPosition(max(output_mc["mass"]["cuts"]), face = "left", arrow_length = 50)
    Plots.Save("mass_pi0_tags", outDir)

    Plots.PlotTagged(output_mc["mass"]["value"], output_mc["event_tag"]["tags"], data2 = output_data["mass"]["value"], bins = args.nbins, x_label = "Invariant mass (MeV)", norm = norm)
    Plots.DrawCutPosition(min(output_mc["mass"]["cuts"]), face = "right", arrow_length = 50)
    Plots.DrawCutPosition(max(output_mc["mass"]["cuts"]), face = "left", arrow_length = 50)
    Plots.Save("mass_fs_tags", outDir)

    return

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
        pi_mask = output[o]["tags"]["$\\pi^{+}$"].mask
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


def MakeParticleTables(output : dir, outDir : str, exclude = []):
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

    purity, efficiency = CalcualteCountMetrics(event_counts)
    rprint(event_counts)
    rprint(purity)
    rprint(efficiency)

    event_counts.T.to_latex(outDir + "particle_counts.tex")
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
    rprint(outputs)

    merged_output = {}

    for output in outputs:
        for selection_output in output:
            if selection_output not in merged_output:
                merged_output[selection_output] = {}
            for o in output[selection_output]:
                if o not in merged_output[selection_output]:
                    merged_output[selection_output][o] = output[selection_output][o]
                    continue
                else:
                    print(o)
                    merged_output[selection_output][o]["value"] = ak.concatenate([merged_output[selection_output][o]["value"], output[selection_output][o]["value"]]) 
                    for t in merged_output[selection_output][o]["tags"]:
                        merged_output[selection_output][o]["tags"][t].mask = ak.concatenate([merged_output[selection_output][o]["tags"][t].mask, output[selection_output][o]["tags"][t]].mask)
                    for t in merged_output[selection_output][o]["tags"]:
                        merged_output[selection_output][o]["fs_tags"][t].mask = ak.concatenate([merged_output[selection_output][o]["fs_tags"][t].mask, output[selection_output][o]["fs_tags"][t]].mask)
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
    if sample == "mc":
        MakeParticleTables(output["beam"]      , out + "beam/", ["no_selection"])
        PiPlusBranchingFractions(output["beam"], out + "beam/")
        MakeParticleTables(output["pip"]       , out + "daughter_pi/", ["completeness"])
        MakeParticleTables(output["photon"]    , out + "photon/", ["nHits_completeness", "impact_parameter_completeness"])
        MakeParticleTables(output["pi0"]       , out + "pi0/", ["n_photons", "event_tag"])
    return


def main(args):
    shower_merging.SetPlotStyle(extend_colors = True)

    func_args = vars(args)
    func_args["data"] = False
    print(func_args)

    output_mc = MergeOutputs(Processing.mutliprocess(run, args.mc_file, args.batches, args.events, func_args, args.threads)) # run the main analysing method

    output_data = None
    if args.data_file is not None:
        func_args["data"] = True
        output_data = MergeOutputs(Processing.mutliprocess(run, args.data_file, args.batches, args.events, func_args, args.threads)) # run the main analysing method

    # tables
    MakeTables(output_mc, args.out + "tables_mc/", "mc")
    if output_data is not None: MakeTables(output_data, args.out + "tables_data/", "data")

    # output directories
    os.makedirs(args.out + "plots/daughter_pi/", exist_ok = True)
    os.makedirs(args.out + "plots/beam/", exist_ok = True)
    os.makedirs(args.out + "plots/photon/", exist_ok = True)
    os.makedirs(args.out + "plots/pi0/", exist_ok = True)

    # plots
    MakeBeamSelectionPlots(output_mc["beam"], output_data["beam"], args.out + "plots/beam/")
    MakePiPlusSelectionPlots(output_mc["pip"], output_data["pip"], args.out + "plots/daughter_pi/")
    MakePhotonCandidateSelectionPlots(output_mc["photon"], output_data["photon"], args.out + "plots/photon/")
    MakePi0SelectionPlots(output_mc["pi0"], output_data["pi0"], args.out + "plots/pi0/")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "mc_file", nargs = "+", help = "MC NTuple file to study.")
    parser.add_argument("-d", "--data-file", dest = "data_file", nargs = "+", help = "Data Ntuple to study")
    parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {Master.Ntuple_Type._member_map_}.", required = True)

    parser.add_argument("--mc_beam_quality_fit", dest = "mc_beam_quality_fit", type = str, help = "mc fit values for the beam quality cut.", required = True)
    parser.add_argument("--data_beam_quality_fit", dest = "data_beam_quality_fit", type = str, default = None, help = "data fit values for the beam quality cut.")
    parser.add_argument("--shower_correction_factor", dest = "shower_correction_factor", type = str, help = "linear shower energy correction factor.", default = 1)

    parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "number of batches to split n tuple files into when parallel processing processing data.")
    parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "number of events to process when parallel processing data.")

    parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "number of threads to use when processsing")

    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save plots")
    parser.add_argument("--nbins", dest = "nbins", type = int, default = 50, help = "number of bins to make for histogram plots.")
    parser.add_argument("-a", "--annotation", dest = "annotation", type = str, default = None, help = "annotation to add to plots")

    args = parser.parse_args()

    if args.out is None:
        if len(args.mc_file) == 1:
            args.out = args.mc_file[0].split("/")[-1].split(".")[0] + "/"
        else:
            args.out = "selection_studies/" #? how to make a better name for multiple input files?
    if args.out[-1] != "/": args.out += "/"
    if args.data_file is not None and args.data_beam_quality_fit is None:
        raise Exception("beam quality fit values for data are required")

    rprint(vars(args))
    main(args)