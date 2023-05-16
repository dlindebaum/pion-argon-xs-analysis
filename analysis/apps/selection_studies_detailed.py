#!/usr/bin/env python3
import argparse
import json
import os

from rich import print as rprint
from python.analysis import Master, BeamParticleSelection, PFOSelection, Plots, shower_merging, vector, Processing, Tags

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from particle import Particle


def MakeOutput(value, tags, cuts = [], fs_tags = None) -> dict:
    return {"value" : value, "tags" : tags, "cuts" : cuts, "fs_tags" : fs_tags}


def AnalyseBeamSelection(events : Master.Data, beam_quality_fits : str) -> dict:
    output = {}

    output["no_selection"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, Tags.GenerateTrueFinalStateTags(events))

    mask = BeamParticleSelection.CaloSizeCut(events)
    output["calo_size"] = MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None)
    events.Filter([mask], [mask])
    output["calo_size"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* beam pandora tag selection
    mask = BeamParticleSelection.PandoraTagCut(events)
    output["pandora_tag"] = MakeOutput(events.recoParticles.beam_pandora_tag, Tags.GenerateTrueBeamParticleTags(events), [13])
    events.Filter([mask], [mask])
    output["pandora_tag"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

    #* michel score cut
    score = ak.where(events.recoParticles.beam_nHits != 0, events.recoParticles.beam_michelScore / events.recoParticles.beam_nHits, -999)
    mask = BeamParticleSelection.MichelScoreCut(events)
    output["michel_score"] = MakeOutput(score, Tags.GenerateTrueBeamParticleTags(events), [0.55], Tags.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["michel_score"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)

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

    #* median dE/dX
    mask = BeamParticleSelection.MedianDEdXCut(events)
    median = PFOSelection.Median(events.recoParticles.beam_dEdX)
    output["median_dEdX"] = MakeOutput(median, Tags.GenerateTrueBeamParticleTags(events), [2.4], Tags.GenerateTrueFinalStateTags(events))
    events.Filter([mask], [mask])
    output["median_dEdX"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)


    #* pi+ beam selection
    mask = BeamParticleSelection.PiBeamSelection(events)
    counts = Tags.GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i].mask)
    output["pi_beam"] = MakeOutput(counts, Tags.GenerateTrueBeamParticleTags(events), None)
    events.Filter([mask], [mask])
    output["pi_beam"]["fs_tags"] = Tags.GenerateTrueFinalStateTags(events)


    #* true particle population
    tags = Tags.GenerateTrueBeamParticleTags(events)
    for t in tags:
        tags[t] = ak.sum(tags[t].mask) # turn mask into counts
    output["final_tags"] = tags
    return output


def AnalysePiPlusSelection(events : Master.Data) -> dict:
    # for now just do the daughter pi+ cuts
    output = {}

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
    tags = Tags.GenerateTrueParticleTags(events)
    for t in tags:
        tags[t] = ak.sum(tags[t].mask) # turn mask into counts
    output["final_tags"] = tags
    return output


def AnalysePhotonCandidateSelection(events : Master.Data) -> dict:
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
    tags = Tags.GenerateTrueParticleTags(events)
    for t in tags:
        tags[t] = ak.sum(tags[t].mask) # turn mask into counts
    output["final_tags"] = tags

    return output


def AnalysePi0Selection(events : Master.Data) -> dict:
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
    mass = ak.flatten(shower_pairs.reco_mass)
    mask = (mass > 50) & (mass < 250)
    output["mass"] = MakeOutput(mass, Tags.GeneratePi0Tags(events), [50, 250])
    output["event_tag"] = MakeOutput(None, Tags.GenerateTrueFinalStateTags(events), None)
    events.Filter([mask], [mask])

    photonCandidates = photonCandidates[mask]

    #* opening angle
    shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photonCandidates)
    angle = ak.flatten(shower_pairs.reco_angle)
    mask = (angle > (10 * np.pi / 180)) & (angle < (80 * np.pi / 180))
    output["angle"] = MakeOutput(angle, Tags.GeneratePi0Tags(events), [(10 * np.pi / 180), (80 * np.pi / 180)])
    events.Filter([mask], [mask])
    photonCandidates = photonCandidates[mask]
    return output

# @Processing.log_process
def run(i, file, n_events, start, selected_events, args):
    events = Master.Data(file, nEvents = n_events, start = start, nTuple_type = args["ntuple_type"]) # load data

    events.Filter([PFOSelection.GoodShowerSelection(events)])

    output_beam = AnalyseBeamSelection(events, args["beam_quality_fit"]) # events are cut after this

    output_pip = AnalysePiPlusSelection(events.Filter(returnCopy = True)) # pass the PFO selections a copy of the event

    output_photon = AnalysePhotonCandidateSelection(events.Filter(returnCopy = True))

    output_pi0 = AnalysePi0Selection(events.Filter(returnCopy = True))

    output = {
        "beam" : output_beam,
        "pip" : output_pip,
        "photon" : output_photon,
        "pi0" : output_pi0
    }

    return output


def MakeBeamSelectionPlots(output : dict, outDir : str):
    bar_data = []
    for tag in output["pi_beam"]["value"]:
        bar_data.extend([tag] * output["pi_beam"]["value"][tag])
    Plots.PlotBar(bar_data, xlabel = "True particle ID")
    Plots.Save("pi_beam", outDir)

    Plots.PlotBar(output["pandora_tag"]["value"], xlabel = "Pandora tag")
    Plots.Save("pandora_tag", outDir)

    Plots.PlotTagged(output["michel_score"]["value"], output["michel_score"]["tags"], x_range = (0, 1), y_scale = "log", bins = 50, x_label = "Michel score", ncols = 3)
    Plots.DrawCutPosition(output["michel_score"]["cuts"][0], face = "left")
    Plots.Save("michel_score", outDir)

    Plots.PlotTagged(output["dxy"]["value"], output["dxy"]["tags"], bins = 50, x_label = "$dxy$", y_scale = "log", x_range = [0, 10])
    Plots.DrawCutPosition(output["dxy"]["cuts"][0], arrow_length = 1, face = "left")
    Plots.Save("dxy", outDir)

    Plots.PlotTagged(output["dz"]["value"], output["dz"]["tags"], bins = 50, x_label = "$dz$", y_scale = "log", x_range = [-10, 10])
    Plots.DrawCutPosition(min(output["dz"]["cuts"]), arrow_length = 1, face = "right")
    Plots.DrawCutPosition(max(output["dz"]["cuts"]), arrow_length = 1, face = "left")
    Plots.Save("dz", outDir)

    Plots.PlotTagged(output["cos_theta"]["value"], output["cos_theta"]["tags"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = 50, x_range = [0.9, 1])
    Plots.DrawCutPosition(output["cos_theta"]["cuts"][0], arrow_length = 0.02)
    Plots.Save("cos_theta", outDir)

    Plots.PlotTagged(output["beam_endPos_z"]["value"], output["beam_endPos_z"]["tags"], x_label = "Beam end position z (cm)")
    Plots.DrawCutPosition(output["beam_endPos_z"]["cuts"][0], face = "left", arrow_length = 50)
    Plots.Save("beam_endPos_z", outDir)

    Plots.PlotTagged(output["median_dEdX"]["value"], output["median_dEdX"]["tags"], y_scale = "log", x_range = [0, 10], x_label = "Median $dE/dX$ (MeV/cm)")
    Plots.DrawCutPosition(output["median_dEdX"]["cuts"][0], face = "left", arrow_length = 2)
    Plots.Save("median_dEdX", outDir)

    bar_data = []
    for tag in output["final_tags"]:
        bar_data.extend([tag]*output["final_tags"][tag])
    Plots.PlotBar(bar_data, xlabel = "True particle ID")
    Plots.Save("final_tags", outDir)
    return


def MakePiPlusSelectionPlots(output : dict, outDir : str):
    if "track_score_all" in output:
        Plots.PlotTagged(output["track_score_all"]["value"], output["track_score_all"]["tags"], y_scale = "log", x_label = "track score")
        Plots.Save("track_score_all", outDir)

    Plots.PlotTagged(output["track_score"]["value"], output["track_score"]["tags"], x_range = [0, 1], y_scale = "log", bins = 50, ncols = 5, x_label = "track score")
    Plots.DrawCutPosition(output["track_score"]["cuts"][0], face = "right")
    Plots.Save("track_score", outDir)


    Plots.PlotTagged(output["nHits"]["value"], output["nHits"]["tags"], bins = 50, ncols = 5, x_range = [0, 500], x_label = "nHits")
    Plots.Save("nHits", outDir)

    Plots.PlotHist2D(ak.ravel(output["completeness"]["value"]), ak.ravel(output["nHits"]["value"]), xlabel = "completeness", ylabel = "nHits", y_range = [0, 500], bins = 50)
    Plots.DrawCutPosition(output["nHits"]["cuts"][0], flip = True, arrow_length = 100, arrow_loc = 0.1, color = "red")
    Plots.Save("nHits_vs_completeness", outDir)

    Plots.PlotTagged(output["median_dEdX"]["value"], output["median_dEdX"]["tags"], ncols = 3, x_range = [0, 6], x_label = "median $dEdX$", bins = 50)
    Plots.DrawCutPosition(min(output["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "right")
    Plots.DrawCutPosition(max(output["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "left")
    Plots.Save("median_dEdX", outDir)

    bar_data = []
    for tag in output["final_tags"]:
        bar_data.extend([tag]*output["final_tags"][tag])
    Plots.PlotBar(bar_data, xlabel = "true particle ID")
    Plots.Save("true_particle_ID", outDir)
    return

def MakePhotonCandidateSelectionPlots(output : dict, outDir : str):
    Plots.PlotTagged(output["em_score"]["value"], output["em_score"]["tags"], bins = 50, x_range = [0, 1], ncols = 5, x_label = "em score")
    Plots.DrawCutPosition(output["em_score"]["cuts"][0])
    Plots.Save("em_score", outDir)

    Plots.PlotTagged(output["nHits"]["value"], output["nHits"]["tags"], bins = 50, x_label = "number of hits", x_range = [0, 1000])
    Plots.DrawCutPosition(output["nHits"]["cuts"][0], arrow_length = 100)
    Plots.Save("nHits", outDir)

    Plots.PlotHist2D(ak.ravel(output["nHits_completeness"]["value"]), ak.ravel(output["nHits"]["value"]), bins = 50, x_range = [0, 1],y_range = [0, 1000], xlabel = "completeness", ylabel = "number of hits")
    Plots.DrawCutPosition(output["nHits"]["cuts"][0], flip = True, arrow_length = 100, color = "red")
    Plots.Save("nHits_vs_completeness", outDir)

    Plots.PlotTagged(output["beam_separation"]["value"], output["beam_separation"]["tags"], bins = 50, x_range = [0, 200], x_label = "distance from PFO start to beam end position (cm)")
    Plots.DrawCutPosition(min(output["beam_separation"]["cuts"]), arrow_length = 30)
    Plots.DrawCutPosition(max(output["beam_separation"]["cuts"]), face = "left", arrow_length = 30)
    Plots.Save("beam_separation", outDir)

    Plots.PlotTagged(output["impact_parameter"]["value"], output["impact_parameter"]["tags"], bins = 50, x_label = "impact parameter wrt beam (cm)")
    Plots.DrawCutPosition(output["impact_parameter"]["cuts"][0], arrow_length = 20, face = "left")
    Plots.Save("impact_parameter", outDir)

    Plots.PlotHist2D(ak.ravel(output["impact_parameter_completeness"]["value"]), ak.ravel(output["impact_parameter"]["value"]), bins = 50, x_range = [0, 1], xlabel = "completeness", ylabel = "impact parameter wrt beam (cm)")
    Plots.DrawCutPosition(output["impact_parameter"]["cuts"][0], arrow_loc = 0.2, arrow_length = 20, face = "left", flip = True, color ="red")
    Plots.Save("impact_parameter_vs_completeness", outDir)

    bar_data = []
    for tag in output["final_tags"]:
        bar_data.extend([tag]*output["final_tags"][tag])
    Plots.PlotBar(bar_data, xlabel = "true particle ID")
    Plots.Save("true_particle_ID", outDir)
    return


def MakePi0SelectionPlots(output : dict, outDir : str):
    Plots.PlotBar(output["n_photons"]["value"], xlabel = "number of $\pi^{0}$ photon candidates")
    Plots.Save("n_photons", outDir)

    Plots.PlotTagged(output["angle"]["value"], output["angle"]["tags"], bins = 50, x_label = "Opening angle (rad)")
    Plots.DrawCutPosition(min(output["angle"]["cuts"]), face = "right", arrow_length = 0.25)
    Plots.DrawCutPosition(max(output["angle"]["cuts"]), face = "left", arrow_length = 0.25)
    Plots.Save("angle", outDir)

    Plots.PlotTagged(output["mass"]["value"], output["mass"]["tags"], bins = 50, x_label = "Invariant mass (MeV)")
    Plots.DrawCutPosition(min(output["mass"]["cuts"]), face = "right", arrow_length = 50)
    Plots.DrawCutPosition(max(output["mass"]["cuts"]), face = "left", arrow_length = 50)
    Plots.Save("mass_pi0_tags", outDir)

    Plots.PlotTagged(output["mass"]["value"], output["event_tag"]["tags"], bins = 50, x_label = "Invariant mass (MeV)")
    Plots.DrawCutPosition(min(output["mass"]["cuts"]), face = "right", arrow_length = 50)
    Plots.DrawCutPosition(max(output["mass"]["cuts"]), face = "left", arrow_length = 50)
    Plots.Save("mass_fs_tags", outDir)

    return

def CalcualteCountMetrics(counts : pd.DataFrame):
    efficiency = counts.div(counts.no_selection, 0)
    purity = counts / counts.loc["total"]
    return purity, efficiency    


def MakeFinalStateTables(output : dir, outDir : str):
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


def MakeParticleTables(output : dir, outDir : str, exclude = []):
    table = {}

    for o in output:
        if o in exclude: continue
        row = {"total" : 0}
        if o == "final_tags":
            for t in output[o]:
                row["total"] = row["total"] + output[o][t]
                row[t] = output[o][t]
        else:
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


def MergeOutputs(outputs : list):
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
                    if o in ["final_tags"]:
                        for t in merged_output[selection_output][o]:
                            merged_output[selection_output][o][t] = merged_output[selection_output][o][t] + output[selection_output][o][t]
                    else:
                        merged_output[selection_output][o]["value"] = ak.concatenate([merged_output[selection_output][o]["value"], output[selection_output][o]["value"]]) 
                        for t in merged_output[selection_output][o]["tags"]:
                            merged_output[selection_output][o]["tags"][t].mask = ak.concatenate([merged_output[selection_output][o]["tags"][t].mask, output[selection_output][o]["tags"][t]].mask)
                        for t in merged_output[selection_output][o]["tags"]:
                            merged_output[selection_output][o]["fs_tags"][t].mask = ak.concatenate([merged_output[selection_output][o]["fs_tags"][t].mask, output[selection_output][o]["fs_tags"][t]].mask)
                        # we shouldn't need to merge cuts because this should be the same for all events

    rprint("merged_output")
    rprint(merged_output)
    return merged_output


def main(args):
    shower_merging.SetPlotStyle(extend_colors = True)
    os.makedirs(args.out + "selection_plots/daughter_pi/", exist_ok = True)
    os.makedirs(args.out + "selection_plots/beam/", exist_ok = True)
    os.makedirs(args.out + "selection_plots/photon/", exist_ok = True)
    os.makedirs(args.out + "selection_plots/pi0/", exist_ok = True)

    output = MergeOutputs(Processing.mutliprocess(run, args.file, args.batches, args.events, vars(args), args.threads)) # run the main analysing method
    MakeFinalStateTables(output["beam"], args.out + "selection_plots/beam/")
    MakeParticleTables(output["beam"], args.out + "selection_plots/beam/", ["no_selection"])
    MakeParticleTables(output["pip"], args.out + "selection_plots/daughter_pi/", ["completeness"])
    MakeParticleTables(output["photon"], args.out + "selection_plots/photon/", ["nHits_completeness", "impact_parameter_completeness"])
    MakeParticleTables(output["pi0"], args.out + "selection_plots/pi0/", ["n_photons", "event_tag"])

    MakePiPlusSelectionPlots(output["pip"], args.out + "selection_plots/daughter_pi/")
    MakeBeamSelectionPlots(output["beam"], args.out + "selection_plots/beam/")
    MakePhotonCandidateSelectionPlots(output["photon"], args.out + "selection_plots/photon/")
    MakePi0SelectionPlots(output["pi0"], args.out + "selection_plots/pi0/")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", nargs = "+", help = "NTuple file to study.")
    parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {Master.Ntuple_Type._member_map_}.", required = True)

    parser.add_argument("--beam_quality_fit", dest = "beam_quality_fit", type = str, help = "fit values for the beam quality cut.", required = True)

    parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "number of batches to split n tuple files into when parallel processing processing data.")
    parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "number of events to process when parallel processing data.")

    parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "number of threads to use when processsing")

    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save plots")
    parser.add_argument("-a", "--annotation", dest = "annotation", type = str, default = None, help = "annotation to add to plots")

    args = parser.parse_args()

    if args.out is None:
        if len(args.file) == 1:
            args.out = args.file[0].split("/")[-1].split(".")[0] + "/"
        else:
            args.out = "selection_studies/" #? how to make a better name for multiple input files?
    if args.out[-1] != "/": args.out += "/"

    rprint(vars(args))
    main(args)