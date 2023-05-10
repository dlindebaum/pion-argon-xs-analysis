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

# @Processing.log_process
def run(i, file, n_events, start, selected_events, args):
    events = Master.Data(file, nEvents = n_events, start = start) # load data

    # do the beam particle selection (will expand on this later)
    shower_merging.Selection(events, "reco", "reco", veto_daughter_pip = False, select_photon_candidates = False) # only do reco selection for now

    # for now just do the daughter pi+ cuts
    output = {}

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


def MakePlots(output, outDir : str):

    PlotTagged(output["track_score_all"]["value"], output["track_score_all"]["tags"], y_scale = "log", x_label = "track score")
    Plots.Save("track_score_all", outDir)

    PlotTagged(output["track_score"]["value"], output["track_score"]["tags"], y_scale = "log", bins = 50, ncols = 5, x_label = "track score")
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
        for o in output:
            if o not in merged_output:
                merged_output[o] = output[o]
                continue
            else:
                if o == "final_tags":
                    for t in merged_output[o]:
                        merged_output[o][t] = merged_output[o][t] + output[o][t]
                else:
                    merged_output[o]["value"] = ak.concatenate([merged_output[o]["value"], output[o]["value"]])
                    merged_tags = {}
                    for t in merged_output[o]["tags"]:
                        merged_tags[t] = ak.concatenate([merged_output[o]["tags"][t], output[o]["tags"][t]])
                    merged_output[o]["tags"] = merged_tags
                    # we shouldn't need to merge cuts because this should be the same for all events

    rprint("merged_output")
    rprint(merged_output)
    return merged_output


def main(args):
    shower_merging.SetPlotStyle()
    os.makedirs(args.out + "selection_plots/daughter_pi/", exist_ok = True) # only implemented daughter pi+ for now

    output = MergeOutputs(Processing.mutliprocess(run, args.file, args.batches, args.events, vars(args), args.threads)) # run the main analysing method
    MakePlots(output, args.out + "selection_plots/daughter_pi/")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", nargs = "+", help = "NTuple file to study.")
    
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