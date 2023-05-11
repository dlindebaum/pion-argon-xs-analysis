#!/usr/bin/env python3
import argparse
import json
import os

from rich import print as rprint
from python.analysis import Master, BeamParticleSelection, PFOSelection, Plots, shower_merging, vector, Processing, Tags

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from particle import Particle


def MakeOutput(value, tags, cuts = []) -> dict:
    return {"value" : value, "tags" : tags, "cuts" : cuts}


def AnalyseBeamSelection(events : Master.Data):
    output = {}

    mask = BeamParticleSelection.CaloSizeCut(events) # not plot needed for this
    events.Filter([mask], [mask])

    #* pi+ beam selection
    mask = BeamParticleSelection.PiBeamSelection(events)
    counts = Tags.GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i])
    output["pi_beam"] = counts
    events.Filter([mask], [mask])

    #* beam pandora tag selection
    mask = BeamParticleSelection.PandoraTagCut(events)
    output["pandora_tag"] = MakeOutput(events.recoParticles.beam_pandora_tag, Tags.GenerateTrueBeamParticleTags(events), [13])
    events.Filter([mask], [mask])

    #* michel score cut
    score = ak.where(events.recoParticles.beam_nHits != 0, events.recoParticles.beam_michelScore / events.recoParticles.beam_nHits, -999)
    mask = BeamParticleSelection.MichelScoreCut(events)
    output["michel_score"] = MakeOutput(score, Tags.GenerateTrueBeamParticleTags(events), [0.55])
    events.Filter([mask], [mask])

    #* beam quality cuts
    with open(args["beam_quality_fit"], "r") as f:
        fit_values = json.load(f)

    #* dxy cut
    dxy = (((events.recoParticles.beam_startPos.x - fit_values["mu_x"]) / fit_values["sigma_x"])**2 + ((events.recoParticles.beam_startPos.y - fit_values["mu_y"]) / fit_values["sigma_y"])**2)**0.5
    mask = dxy < 3
    output["dxy"] = MakeOutput(dxy, Tags.GenerateTrueBeamParticleTags(events), [3])
    print(f"dxy cut: {BeamParticleSelection.CountMask(mask)}")
    events.Filter([mask], [mask])

    #* dz cut
    delta_z = (events.recoParticles.beam_startPos.z - fit_values["mu_z"]) / fit_values["sigma_z"]
    mask = (delta_z > -3) & (delta_z < 3)
    output["dz"] = MakeOutput(delta_z, Tags.GenerateTrueBeamParticleTags(events), [-3, 3])
    events.Filter([mask], [mask])
    print(f"dz cut: {BeamParticleSelection.CountMask(mask)}")

    #* beam direction
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))
    beam_dir_mu = vector.normalize(vector.vector(fit_values["mu_dir_x"], fit_values["mu_dir_y"], fit_values["mu_dir_z"]))
    beam_costh = vector.dot(beam_dir, beam_dir_mu)
    mask = beam_costh > 0.95
    output["cos_theta"] = MakeOutput(beam_costh, Tags.GenerateTrueBeamParticleTags(events), [0.95])
    events.Filter([mask], [mask])

    #* APA3 cut
    mask = BeamParticleSelection.APA3Cut(events)
    output["beam_endPos_z"] = MakeOutput(events.recoParticles.beam_endPos.z, Tags.GenerateTrueBeamParticleTags(events), [220])
    events.Filter([mask], [mask])

    #* median dE/dX
    mask = BeamParticleSelection.MedianDEdXCut(events)
    median = PFOSelection.Median(events.recoParticles.beam_dEdX)
    output["median_dEdX"] = MakeOutput(median, Tags.GenerateTrueBeamParticleTags(events), [2.4])
    events.Filter([mask], [mask])

    #* true particle population
    tags = Tags.GenerateTrueBeamParticleTags(events)
    for t in tags:
        tags[t] = ak.sum(tags[t]) # turn mask into counts
    output["final_tags"] = tags
    return output


def AnalysePiPlusSelection(events : Master.Data):
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
        tags[t] = ak.sum(tags[t]) # turn mask into counts
    output["final_tags"] = tags
    return output

# @Processing.log_process
def run(i, file, n_events, start, selected_events, args):
    events = Master.Data(file, nEvents = n_events, start = start, nTuple_type = args["ntuple_type"]) # load data

    output_beam = AnalyseBeamSelection(events) # events are cut after this

    output_pip = AnalysePiPlusSelection(events) # events are cut after this

    output = {
        "beam" : output_beam,
        "pip" : output_pip,
    }

    return output


def MakeBeamSelectionPlots(output : dict, outDir : str):
    bar_data = []
    for tag in output["pi_beam"]:
        bar_data.extend([tag] * output["pi_beam"][tag])
    Plots.PlotBar(bar_data, xlabel = "true particle ID")
    Plots.Save("pi_beam", outDir)

    Plots.PlotBar(output["pandora_tag"]["value"])
    Plots.Save("pandora_tag", outDir)

    Plots.PlotTagged(output["michel_score"]["value"], output["michel_score"]["tags"], x_range = (0, 1), y_scale = "log", bins = 50, x_label = "michel score", ncols = 3)
    Plots.DrawCutPosition(output["michel_score"]["cuts"][0], face = "left")
    Plots.Save("michel_score", outDir)

    Plots.PlotTagged(output["dxy"]["value"], output["dxy"]["tags"], bins = 50, x_label = "$dxy$ (cm)", y_scale = "log", x_range = [0, 10])
    Plots.DrawCutPosition(output["dxy"]["cuts"][0], arrow_length = 1, face = "left")
    Plots.Save("dxy", outDir)

    Plots.PlotTagged(output["dz"]["value"], output["dz"]["tags"], bins = 50, x_label = "$dz$ (cm)", y_scale = "log", x_range = [0, 10])
    Plots.DrawCutPosition(min(output["dz"]["cuts"]), arrow_length = 1, face = "right")
    Plots.DrawCutPosition(max(output["dz"]["cuts"]), arrow_length = 1, face = "left")
    Plots.Save("dz", outDir)

    Plots.PlotTagged(output["cos_theta"]["value"], output["cos_theta"]["tags"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = 50, x_range = [0.9, 1])
    Plots.DrawCutPosition(output["cos_theta"]["cuts"][0], arrow_length = 0.02)
    Plots.Save("cos_theta", outDir)

    Plots.PlotTagged(output["beam_endPos_z"]["value"], output["beam_endPos_z"]["tags"], x_label = "Beam end position z (cm)")
    Plots.DrawCutPosition(output["beam_endPos_z"]["cuts"][0], face = "left", arrow_length = 50)
    Plots.Save("beam_endPos_z", outDir)

    Plots.PlotTagged(output["median_dEdX"]["value"], output["median_dEdX"]["tags"], y_scale = "log", x_range = [0, 10], x_label = "median $dE/dX$ (MeV/cm)")
    Plots.DrawCutPosition(output["median_dEdX"]["cuts"][0], face = "left", arrow_length = 2)
    Plots.Save("median_dEdX", outDir)

    bar_data = []
    for tag in output["final_tags"]:
        bar_data.extend([tag]*output["final_tags"][tag])
    Plots.PlotBar(bar_data, xlabel = "true particle ID")
    Plots.Save("true_particle_ID", outDir)
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
                    if o in ["final_tags", "pi_beam"]:
                        for t in merged_output[selection_output][o]:
                            merged_output[selection_output][o][t] = merged_output[selection_output][o][t] + output[selection_output][o][t]
                    else:
                        merged_output[selection_output][o]["value"] = ak.concatenate([merged_output[selection_output][o]["value"], output[selection_output][o]["value"]])
                        merged_tags = {}
                        for t in merged_output[selection_output][o]["tags"]:
                            merged_tags[t] = ak.concatenate([merged_output[selection_output][o]["tags"][t], output[selection_output][o]["tags"][t]])
                        merged_output[selection_output][o]["tags"] = merged_tags
                        # we shouldn't need to merge cuts because this should be the same for all events

    rprint("merged_output")
    rprint(merged_output)
    return merged_output


def main(args):
    shower_merging.SetPlotStyle(extend_colors = True)
    os.makedirs(args.out + "selection_plots/daughter_pi/", exist_ok = True)
    os.makedirs(args.out + "selection_plots/beam/", exist_ok = True)

    output = MergeOutputs(Processing.mutliprocess(run, args.file, args.batches, args.events, vars(args), args.threads)) # run the main analysing method
    MakePiPlusSelectionPlots(output["pip"], args.out + "selection_plots/daughter_pi/")
    MakeBeamSelectionPlots(output["beam"], args.out + "selection_plots/beam/")
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