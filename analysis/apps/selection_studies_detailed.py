#!/usr/bin/env python3
import argparse
import os

from rich import print as rprint
from python.analysis import Master, BeamParticleSelection, PFOSelection, Plots, shower_merging, vector, Processing

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from particle import Particle
from scipy.optimize import curve_fit


def DrawCutPosition(value : float, arrow_loc : float = 0.8, arrow_length : float = 0.2, face : str = "right", flip : bool = False, color = "black"):
    """ Illustrates a cut on a plot. Direction of the arrow indidcates which portion of the plot passes the cut.

    Args:
        value (float): value of the cut
        arrow_loc (float, optional): where along the line to place the arrow. Defaults to 0.8.
        arrow_length (float, optional): length of the arrow, must be in units of the cut. Defaults to 0.2.
        face (str, optional): which way the arrow faces. Defaults to "right".
        flip (bool, optional): flip the arrow to the y axis. Defaults to False.
        color (str, optional): colour of the line and arrow. Defaults to "black".
    """

    if face == "right":
        face_factor = 1
    elif face == "left":
        face_factor = -1
    else:
        raise Exception("face must be left or right")

    xy0 = (value - face_factor * (value/1500), arrow_loc)
    xy1 = (value - (value/1500) + face_factor * arrow_length, arrow_loc)
    transform = ("data", "axes fraction")

    if flip:
        xy0 = tuple(reversed(xy0))
        xy1 = tuple(reversed(xy1))
        transform = tuple(reversed(transform))

        plt.axhline(value, color = color)
    else:
        plt.axvline(value, color = color)

    plt.annotate("", xy = xy1, xytext = xy0, arrowprops=dict(facecolor = color, edgecolor = color, arrowstyle = "->"), xycoords= transform)


def PlotTagged(data : np.array, tags : dict, bins = 100, range : list = None, y_scale : str = "linear", x_label : str = "", loc : str = "best", ncols : int = 2):
    """ Makes a stacked histogram and splits the sample based on tags.

    Args:
        data (np.array): data to plot
        tags (dict): tags for the data, values should be a mask.
        bins (int, optional): number of bins. Defaults to 100.
        range (list, optional): plot range. Defaults to None.
        y_scale (str, optional): y axis scale. Defaults to "linear".
        x_label (str, optional): x label. Defaults to "".
        loc (str, optional): legend location. Defaults to "best".
        ncols (int, optional): number of columns in legend. Defaults to 2.
    """
    split_data = [ak.ravel(data[tags[t]]) for t in tags]
    Plots.PlotHist(split_data, stacked = True, label = list(tags.keys()), bins = bins, y_scale = y_scale, xlabel = x_label, range = range)
    plt.legend(loc = loc, ncols = ncols)


def ParticleMasks(pdgs : ak.Array, to_tag : list) -> dict:
    """ produces a dictionry of masks based on particle pdg codes to tag specified by the user.

    Args:
        pdgs (ak.Array): array of pdg codes
        to_tag (list): particle pgd codes to tag

    Returns:
        dict: particle tags
    """
    masks = {}
    for t in to_tag:
        masks["$" + Particle.from_pdgid(t).latex_name + "$"] = pdgs == t
    return masks


def OtherMask(masks : dict) -> ak.Array:
    """ Creates a mask which selects indices not already tagged by a set of masks.

    Args:
        masks (dict): masks which tag data

    Returns:
        ak.Array: mask which tags any untagged data.
    """
    other = None
    for m in masks:
        if other is None:
            other = masks[m]
        else:
            other = other | masks[m]
    return ~other


def GenerateTrueParticleTags(events : Master.Data) -> dict:
    """ Creates true particle tags with boolean masks. Does this for all PFOs.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: tags
    """
    particles_to_tag = [
        211, -211, 13, -13, 11, -11, 22, 2212, 321
    ] # anything not in this list is tagged as other

    masks = ParticleMasks(events.trueParticlesBT.pdg, particles_to_tag)

    masks["other"] = OtherMask(masks)
    return masks


def GenerateTrueBeamParticleTags(events : Master.Data) -> dict:
    """ Creates true particle tags with boolean masks for beam particles.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: tags
    """
    particles_to_tag = [
        211, -13, -11, 2212, 321
    ] # anything not in this list is tagged as other

    masks = ParticleMasks(events.trueParticlesBT.beam_pdg, particles_to_tag)
    masks["other"] = OtherMask(masks)
    return masks


def MakeOutput(value, tags, cuts = []):
    return {"value" : value, "tags" : tags, "cuts" : cuts}


def AnalyseBeamSelection(events : Master.Data):
    output = {}

    mask = BeamParticleSelection.CaloSizeCut(events) # not plot needed for this
    events.Filter([mask], [mask])

    #* pi+ beam selection
    mask = BeamParticleSelection.PiBeamSelection(events)
    counts = GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i])
    output["pi_beam"] = counts
    events.Filter([mask], [mask])

    #* beam pandora tag selection
    mask = BeamParticleSelection.PandoraTagCut(events)
    output["pandora_tag"] = MakeOutput(events.recoParticles.beam_pandora_tag, GenerateTrueBeamParticleTags(events), [13])
    events.Filter([mask], [mask])

    #* michel score cut
    score = ak.where(events.recoParticles.beam_nHits != 0, events.recoParticles.beam_michelScore / events.recoParticles.beam_nHits, -999)
    mask = BeamParticleSelection.MichelScoreCut(events)
    output["michel_score"] = MakeOutput(score, GenerateTrueBeamParticleTags(events), [0.55])
    events.Filter([mask], [mask])

    #* beam quality cuts
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def fit_gaussian(data, bins, range = None):
        if range is None:
            range = [min(data), max(data)]
        y, bins_edges = np.histogram(np.array(data), bins = bins, range = range)
        bin_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
        return curve_fit(gaussian, bin_centers, y, p0 = (0, ak.mean(data), ak.std(data)))

    mu = {}
    sigma = {}
    for i in ["x", "y", "z"]:
        popt, _ = fit_gaussian(events.recoParticles.beam_startPos[i], bins = 100)
        mu[i] = popt[1]
        sigma[i] =popt[2]

    print(mu)
    print(sigma)

    #* dxy cut
    dxy = (((events.recoParticles.beam_startPos.x - mu["x"]) / sigma["x"])**2 + ((events.recoParticles.beam_startPos.y - mu["y"]) / sigma["y"])**2)**0.5
    mask = dxy < 3
    output["dxy"] = MakeOutput(dxy, GenerateTrueBeamParticleTags(events), [3])
    print(f"dxy cut: {BeamParticleSelection.CountMask(mask)}")
    events.Filter([mask], [mask])

    #* dz cut
    delta_z = (events.recoParticles.beam_startPos.z - mu["z"]) / sigma["z"]
    mask = (delta_z > -3) & (delta_z < 3)
    output["dz"] = MakeOutput(delta_z, GenerateTrueBeamParticleTags(events), [-3, 3])
    events.Filter([mask], [mask])
    print(f"dz cut: {BeamParticleSelection.CountMask(mask)}")

    #* beam direction
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))

    mu_dir = {}
    sigma_dir = {}
    for i in ["x", "y", "z"]:
        popt, _ = fit_gaussian(beam_dir[i], bins = 50)
        mu_dir[i] = popt[1]
        sigma_dir[i] = popt[2]

    print(mu_dir)
    print(sigma_dir)

    beam_dir_mu = vector.normalize(vector.vector(mu_dir["x"], mu_dir["y"], mu_dir["z"]))
    beam_costh = vector.dot(beam_dir, beam_dir_mu)
    mask = beam_costh > 0.95
    output["cos_theta"] = MakeOutput(beam_costh, GenerateTrueBeamParticleTags(events), [0.95])
    events.Filter([mask], [mask])

    output["fit_values"] = {
        "start_pos" : {"mu": mu, "sigma" : sigma},
        "direction" : {"mu": mu_dir, "sigma" : sigma_dir}
    }
    print(output["fit_values"])

    #* APA3 cut
    mask = BeamParticleSelection.APA3Cut(events)
    output["beam_endPos_z"] = MakeOutput(events.recoParticles.beam_endPos.z, GenerateTrueBeamParticleTags(events), [220])
    events.Filter([mask], [mask])

    #* median dE/dX
    mask = BeamParticleSelection.MedianDEdXCut(events)
    median = PFOSelection.Median(events.recoParticles.beam_dEdX)
    output["median_dEdX"] = MakeOutput(median, GenerateTrueBeamParticleTags(events), [2.4])
    events.Filter([mask], [mask])

    #* true particle population
    tags = GenerateTrueBeamParticleTags(events)
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
        output["track_score_all"] = MakeOutput(events.recoParticles.trackScore, GenerateTrueParticleTags(events)) # keep a record of the track score to show the cosmic muon background
        events.Filter([mask], [mask])

    #* track score selection
    mask = PFOSelection.TrackScoreCut(events)
    output["track_score"] = MakeOutput(events.recoParticles.trackScore, GenerateTrueParticleTags(events), [0.5])
    events.Filter([mask], [mask])

    #* nHits cut
    mask = PFOSelection.NHitsCut(events, 20)
    output["nHits"] = MakeOutput(events.recoParticles.nHits, GenerateTrueParticleTags(events), [20])
    output["completeness"] = MakeOutput(events.trueParticlesBT.completeness, GenerateTrueParticleTags(events))
    events.Filter([mask], [mask])

    #* median dEdX    
    mask = PFOSelection.PiPlusSelection(events)
    output["median_dEdX"] = MakeOutput(PFOSelection.Median(events.recoParticles.track_dEdX), GenerateTrueParticleTags(events), [0.5, 2.8])
    events.Filter([mask], [mask])

    #* true particle population
    tags = GenerateTrueParticleTags(events)
    for t in tags:
        tags[t] = ak.sum(tags[t]) # turn mask into counts
    output["final_tags"] = tags
    return output

# @Processing.log_process
def run(i, file, n_events, start, selected_events, args):
    events = Master.Data(file, nEvents = n_events, start = start, nTuple_type = args["ntuple_type"]) # load data

    # do the beam particle selection (will expand on this later)
    # shower_merging.Selection(events, "reco", "reco", veto_daughter_pip = False, select_photon_candidates = False) # only do reco selection for now
  
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

    PlotTagged(output["michel_score"]["value"], output["michel_score"]["tags"], range = (0, 1), y_scale = "log", bins = 50, x_label = "michel score", ncols = 3)
    DrawCutPosition(output["michel_score"]["cuts"][0], face = "left")
    Plots.Save("michel_score", outDir)

    PlotTagged(output["dxy"]["value"], output["dxy"]["tags"], bins = 50, x_label = "$dxy$ (cm)", y_scale = "log", range = [0, 10])
    DrawCutPosition(output["dxy"]["cuts"][0], arrow_length = 1, face = "left")
    Plots.Save("dxy", outDir)

    PlotTagged(output["dz"]["value"], output["dz"]["tags"], bins = 50, x_label = "$dz$ (cm)", y_scale = "log", range = [0, 10])
    DrawCutPosition(min(output["dz"]["cuts"]), arrow_length = 1, face = "right")
    DrawCutPosition(max(output["dz"]["cuts"]), arrow_length = 1, face = "left")
    Plots.Save("dz", outDir)

    PlotTagged(output["cos_theta"]["value"], output["cos_theta"]["tags"], x_label = "$\cos(\\theta)$", y_scale = "log", bins = 50, range = [0.9, 1])
    DrawCutPosition(output["cos_theta"]["cuts"][0], arrow_length = 0.02)
    Plots.Save("cos_theta", outDir)

    PlotTagged(output["beam_endPos_z"]["value"], output["beam_endPos_z"]["tags"], x_label = "Beam end position z (cm)")
    DrawCutPosition(output["beam_endPos_z"]["cuts"][0], face = "left", arrow_length = 50)
    Plots.Save("beam_endPos_z", outDir)

    PlotTagged(output["median_dEdX"]["value"], output["median_dEdX"]["tags"], y_scale = "log", range = [0, 10], x_label = "median $dE/dX$ (MeV/cm)")
    DrawCutPosition(output["median_dEdX"]["cuts"][0], face = "left", arrow_length = 2)
    Plots.Save("median_dEdX", outDir)

    bar_data = []
    for tag in output["final_tags"]:
        bar_data.extend([tag]*output["final_tags"][tag])
    Plots.PlotBar(bar_data, xlabel = "true particle ID")
    Plots.Save("true_particle_ID", outDir)
    return

def MakePiPlusSelectionPlots(output, outDir : str):

    if "track_score_all" in output:
        PlotTagged(output["track_score_all"]["value"], output["track_score_all"]["tags"], y_scale = "log", x_label = "track score")
        Plots.Save("track_score_all", outDir)

    PlotTagged(output["track_score"]["value"], output["track_score"]["tags"], range = [0, 1], y_scale = "log", bins = 50, ncols = 5, x_label = "track score")
    DrawCutPosition(output["track_score"]["cuts"][0], face = "right")
    Plots.Save("track_score", outDir)


    PlotTagged(output["nHits"]["value"], output["nHits"]["tags"], bins = 50, ncols = 5, range = [0, 500], x_label = "nHits")
    Plots.Save("nHits", outDir)

    Plots.PlotHist2D(ak.ravel(output["completeness"]["value"]), ak.ravel(output["nHits"]["value"]), xlabel = "completeness", ylabel = "nHits", y_range = [0, 500], bins = 50)
    DrawCutPosition(output["nHits"]["cuts"][0], flip = True, arrow_length = 100, arrow_loc = 0.1, color = "red")
    Plots.Save("nHits_vs_completeness", outDir)

    PlotTagged(output["median_dEdX"]["value"], output["median_dEdX"]["tags"], ncols = 3, range = [0, 6], x_label = "median $dEdX$", bins = 50)
    DrawCutPosition(min(output["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "right")
    DrawCutPosition(max(output["median_dEdX"]["cuts"]), arrow_length = 0.5, face = "left")
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
                    elif o == "fit_values":
                        continue
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

    parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "number of batches to split n tuple files into when parallel processing processing data.")
    parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "number of events to process when parallel processing data.")

    parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "number of threads to use when processsing")
    
    # parser.add_argument("-s", "--selection", dest = "selection_type", type = str, choices = ["cheated", "reco"], help = "type of selection to use.", required = True)

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