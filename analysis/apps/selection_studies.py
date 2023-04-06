#!/usr/bin/env python3
"""
Created on: 14/03/2023 15:14

Author: Shyam Bhuller

Description: Applies beam particle selection, PFO selection, produces tables and basic plots.
#? have the capability of storing quantities to hdf5 files (and capability to read them in)?
"""
import argparse
import os
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from rich import print as rprint

from python.analysis import Master, Plots, shower_merging, Processing


def BasicQuantities(events : Master.Data, start_showers_all : ak.Array, to_merge : ak.Array, signal_all : ak.Array, background : ak.Array) -> dict:
    """ Gets quantities to plot, generally are categorised by signal/background events.

    Args:
        events (Master.Data): events to study
        start_showers_all (ak.Array): start shower mask
        to_merge (ak.Array): PFOs to merge mask
        signal_all (ak.Array): signal PFO mask
        background (ak.Array): background PFO mask

    Returns:
        dict: quantities
    """
    data = {
        "n_signal"      : ak.count(signal_all[signal_all], -1),
        "n_background"  : ak.count(background[background], -1),
        "energy_signal" : ak.ravel(events.recoParticles.energy[to_merge][signal_all]),
        "energy_background" : ak.ravel(events.recoParticles.energy[to_merge][background]),
        "nHits_collection_signal" : ak.ravel(events.recoParticles.nHits_collection[to_merge][signal_all]),
        "nHits_collection_background" : ak.ravel(events.recoParticles.nHits_collection[to_merge][background]),
        "nHits_signal" : ak.ravel(events.recoParticles.nHits[to_merge][signal_all]),
        "nHits_background" : ak.ravel(events.recoParticles.nHits[to_merge][background]),
        "cnn_signal" : ak.ravel(events.recoParticles.cnnScore[to_merge][background]),
        "cnn_background" : ak.ravel(events.recoParticles.cnnScore[to_merge][signal_all]),
        "start_shower_purity" : ak.ravel(events.trueParticlesBT.purity[start_showers_all]),
        "start_shower_completeness" : ak.ravel(events.trueParticlesBT.completeness[start_showers_all]),
        "purity_signal" : ak.ravel(events.trueParticlesBT.purity[to_merge][signal_all]),
        "purity_background" : ak.ravel(events.trueParticlesBT.purity[to_merge][background]),
        "completeness_signal" : ak.ravel(events.trueParticlesBT.completeness[to_merge][signal_all]),
        "completeness_background" : ak.ravel(events.trueParticlesBT.completeness[to_merge][background]),
    }
    for d in data:
        print(f"{d} : {repr(data[d])}")
    return data


def BasicTaggedQuantities(events : Master.Data, tags : dict, start_showers_all : ak.Array, signal_all : ak.Array, background : ak.Array) -> dict:
    """ Gets quantities to plot split by the tags specified.

    Args:
        events (Master.Data): events to study
        tags (dict): event tags
        start_showers_all (ak.Array): start shower mask
        signal_all (ak.Array): signal PFO mask
        background (ak.Array): background PFO mask 

    Returns:
        dict: tagged quantities
    """
    data = {
        "n_signal" : ak.count(signal_all[signal_all], -1),
        "n_background" : ak.count(background[background], -1),
        "n_PFO" : [],
        "n_signal_tagged" : [],
        "n_background_tagged" : [],
        "purity" : [],
        "completeness" : [],
        "pdg" : [],
        "mother_pdg" : [],
    }
    for k, t in tags.items():
        data["n_PFO"].append(ak.num(events.recoParticles.number)[t.mask])
        data["n_signal_tagged"].append(ak.ravel(data["n_signal"][t.mask]))
        data["n_background_tagged"].append(ak.ravel(data["n_background"][t.mask]))
        data["completeness"].append(ak.ravel(events.trueParticlesBT.completeness[start_showers_all])[t.mask])
        data["purity"].append(ak.ravel(events.trueParticlesBT.purity[start_showers_all])[t.mask])
        data["pdg"].append(list(np.unique((events.trueParticlesBT.pdg[start_showers_all])[tags[k].mask], return_counts = True)))
        data["mother_pdg"].append(list(np.unique((events.trueParticlesBT.motherPdg[start_showers_all])[tags[k].mask], return_counts = True)))

    for d in data:
        print(f"{d} : {repr(data[d])}")
    return data


def MakePlots(data : dict, out : str):
    """ Makes plots of basic quantities

    Args:
        data (dict): quantities to plot
        out (str): output file directory
    """
    labels = ["background", "signal"]
    
    Plots.PlotHist(ak.ravel(data["n_signal"]), xlabel = "Start shower multiplicity", annotation = args.annotation)
    Plots.Save("multiplicity", out)

    Plots.PlotHistComparison([data["n_background"], data["n_signal"]], xlabel = "Number of PFOs", bins = 20, labels = labels, annotation = args.annotation)
    Plots.Save("nPFO", out)

    nbins =  max(data["n_signal"]) - min(data["n_signal"])
    Plots.PlotHist(data["n_signal"], xlabel="Number of signal PFOs", bins=np.arange(nbins)-0.5, annotation = args.annotation)
    Plots.Save("nPFO_signal", out)

    Plots.PlotHist(data["n_background"], xlabel = "Number of background PFOs", bins = 20, annotation = args.annotation)
    Plots.Save("nPFO_background", out)

    Plots.PlotHistComparison([data["energy_background"], data["energy_signal"]], xlabel = "Energy (MeV)", bins = 20, y_scale = "log", labels = labels, annotation = args.annotation)
    Plots.Save("energy", out)

    Plots.PlotHistComparison([data["nHits_collection_background"], data["nHits_collection_signal"]], xlabel = "Number of collection plane hits", bins = 20, y_scale = "log", labels = labels, annotation = args.annotation)
    Plots.Save("hits_collection", out)

    Plots.PlotHistComparison([data["nHits_background"], data["nHits_signal"]], xlabel = "Number of hits", bins = 20, y_scale = "log", labels = labels, annotation = args.annotation)
    Plots.Save("hits", out)

    Plots.PlotHistComparison([data["cnn_background"], data["cnn_signal"]], xlabel = "CNN score", bins = 20, labels = labels, annotation = args.annotation)
    Plots.Save("cnn", out)

    Plots.PlotHist(data["start_shower_purity"], xlabel = "start shower purity", annotation = args.annotation)
    Plots.Save("ss-purity", out)
    
    Plots.PlotHist(data["start_shower_completeness"], xlabel = "start shower completeness", annotation = args.annotation)
    Plots.Save("ss-completeness", out)

    Plots.PlotHist2D(data["purity_signal"], data["completeness_signal"], bins = 25, xlabel = "purity", ylabel = "completeness", title = "signal")
    Plots.Save("purity_vs_completeness_s", out)

    Plots.PlotHist2D(data["purity_background"], data["completeness_background"], bins = 25, xlabel = "purity", ylabel = "completeness", title = "background")
    Plots.Save("purity_vs_completeness_b", out)
    return


def MakePlotsTagged(data : dict, tags : dict, out : str):
    """ Makes plots of basic quantities.

    Args:
        data (dict): quantities to plot
        tags (dict): event tags
        out (str): output file directory
    """

    n_signal_bins = max(data["n_signal"]) - min(data["n_signal"]) + 2

    labels = []
    colours = []
    for k, t in tags.items():
        labels.append(k)
        colours.append(t.colour)

    Plots.PlotHist(data["n_PFO"].to_list(), bins = 20, xlabel = "number of PFOs", stacked = True, label = labels, color = colours, annotation = args.annotation)
    Plots.Save("tagged_nPFO", out)

    Plots.PlotHist(data["n_signal_tagged"].to_list(), bins = np.arange(n_signal_bins) - 0.5, xlabel = "mutiplicity", stacked = True, label = labels, color = colours, annotation = args.annotation)
    plt.xticks(np.arange(n_signal_bins))
    Plots.Save("tagged_multiplicity", out)

    Plots.PlotHist(data["n_signal_tagged"].to_list(), bins = 20, xlabel = "number of signal PFOs", stacked = True, label = labels, color = colours, annotation = args.annotation)
    Plots.Save("tagged_nPFO_signal", out)

    Plots.PlotHist(data["n_background_tagged"].to_list(), bins = 20, xlabel = "Number of background PFOs", stacked = True, label = labels, color = colours, annotation = args.annotation)
    Plots.Save("tagged_nPFO_background", out)

    Plots.PlotHist(data["completeness"].to_list(), bins = 20, xlabel = "start shower completeness", stacked = True, label = labels, color = colours, annotation = args.annotation)
    Plots.Save("tagged_start_shower_completeness", out)

    Plots.PlotHist(data["purity"].to_list(), bins = 20, xlabel = "start shower purity", stacked = True, label = labels, color = colours, annotation = args.annotation)
    Plots.Save("tagged_start_shower_purity", out)

    Plots.PlotStackedBar(data["pdg"], xlabel = "pdg of intial photon candidates", label_title = "event topolgy", labels = labels, colours = colours, annotation = args.annotation)
    Plots.Save("tagged_pdg", out)

    Plots.PlotStackedBar(data["mother_pdg"], xlabel = "mother pdg of initial photon candidates", label_title = "event topolgy", labels = labels, colours = colours, annotation = args.annotation)
    Plots.Save("tagged_mother_pdg", out)


def MakeInitialTaggingPlots(tags : dict, n_photons : ak.Array, out : str):
    """ Makes truth tag plots, splitting the dataset by the number of pi0 shower candidates.

    Args:
        tags (dict): event tags
        n_photons (ak.Array): pi0 shower candidates mask
        out (str): output file directory
    """
    for s in np.unique(n_photons):
        plt.figure()
        for name, t in tags.items():
            c = shower_merging.CountMask(t.mask[n_photons == s])
            Plots.PlotBar([name]*c, xlabel = "truth tag", color = t.colour, label = t.name_simple, title = f"number of shower candidates: {s}", newFigure = False)
        Plots.Save(f"truth_tags_nPFO_{s}", out)
    return


def run(i, file, n_events, start, args):
    with open(f"out_{i}.log", "w") as sys.stdout, open(f"out_{i}.log", "w") as sys.stderr:
        events = Master.Data(file, nEvents = n_events, start = start) # load data

        #* apply either cheated or reco selection
        match args["selection_type"]:
            case "cheated":
                events_table, pfo_table = shower_merging.Selection(events, args["selection_type"], args["selection_type"]) # apply selection
                start_showers, to_merge = shower_merging.SplitSample(events) # split sample into pi0 showers and PFOs to merge
            case "reco":
                events_table, pfo_table, photon_candidate_table = shower_merging.Selection(events, args["selection_type"], args["selection_type"], False)

                n_photon_candidates = ak.num(events.recoParticles.number[shower_merging.PFOSelection.InitialPi0PhotonSelection(events)]) # get pi0 shower candidates

                tags = shower_merging.GenerateTruthTags(events)

                #* select events which have exactly 2 photon candidates, will deal with > 2 later
                photon_candidates = n_photon_candidates == 2
                events.Filter([photon_candidates], [photon_candidates])

                photon_candidate_tags = shower_merging.GenerateTruthTags()
                for k in tags:
                    mask = ak.Array(tags[k].mask)
                    photon_candidate_tags[k].mask = mask[photon_candidates] # apply photon candidate masks to the tags

                start_showers, to_merge = shower_merging.SplitSampleReco(events)

                # photon_candidate_table.to_latex(args.out + "photon_candidate_selection.tex")

            case _:
                raise Exception(f"event selection type {args['selection_type']} not understood.")


        #* Plots for the candidate events only i.e. where nphoton candidates == 2
        start_showers_all = np.logical_or(*start_showers)
        _, background, signal_all = shower_merging.SignalBackground(events, start_showers, to_merge)

        #* produce data for plotting
        data = BasicQuantities(events, start_showers_all, to_merge, signal_all, background)

        output = {
            "events_table" : events_table,
            "pfo_table" : pfo_table,
            "data" : data
        }
        if args["selection_type"] == "reco":
            tagged_data = BasicTaggedQuantities(events, photon_candidate_tags, start_showers_all, signal_all, background)

            output["photon_candidate_table"] = photon_candidate_table
            output["n_photon_candidates"] = n_photon_candidates
            output["tags"] = tags #TODO also do for cheated selection
            output["photon_candidate_tags"] = photon_candidate_tags #TODO also do for cheated selection
            output["tagged_data"] = tagged_data
    return output

@Master.timer
def main(args):
    #* initial setup
    shower_merging.SetPlotStyle()
    os.makedirs(args.out + "basic_quantities/tagged/", exist_ok = True)

    output = Processing.mutliprocess(run, args.file, args.batches, args.events, vars(args))

    tags = shower_merging.GenerateTruthTags()
    selected_tags = shower_merging.GenerateTruthTags()
    n_photon_candidates = []

    data = {}
    tagged_data = {}

    for o in output:
        all_tags = o["tags"]
        for k, t in all_tags.items():
            if tags[k].mask is None:
                tags[k].mask = t.mask
            else:
                tags[k].mask = ak.concatenate([tags[k].mask, t.mask])

        pi0_candidate_tags = o["photon_candidate_tags"]
        for k, t in pi0_candidate_tags.items():
            if selected_tags[k].mask is None:
                selected_tags[k].mask = t.mask
            else:
                selected_tags[k].mask = ak.concatenate([selected_tags[k].mask, t.mask])

        if "n_photon_candidates" in o.keys():
            n_photon_candidates = ak.concatenate([n_photon_candidates, o["n_photon_candidates"]])

        for k, v in o["data"].items():
            if k in data:
                data[k] = ak.concatenate([data[k], v])
            else:
                data[k] = ak.Array(v)

        for k, v in o["tagged_data"].items():
            if k in tagged_data:
                tagged_data[k] = ak.concatenate([tagged_data[k], v], axis = -1)
            else:
                tagged_data[k] = ak.Array(v)

    def format_bar_plots(d):
        pdg_list = np.unique(ak.ravel(d[:, 0]))
        counts = []
        for pdg in pdg_list:
            to_sum = d[:, 0] == pdg
            s = ak.sum(d[:, 1][to_sum], axis = -1)
            counts.append(s)
        counts = ak.concatenate(ak.unflatten(counts, 1, -1), -1)
        labels = [pdg_list] * len(counts)
        return ak.concatenate([ak.unflatten(labels, 1), ak.unflatten(counts, 1)], 1)

    tagged_data["pdg"] = ak.to_numpy(format_bar_plots(tagged_data["pdg"]))
    tagged_data["mother_pdg"] = ak.to_numpy(format_bar_plots(tagged_data["mother_pdg"]))

    MakeInitialTaggingPlots(tags, n_photon_candidates, args.out + "basic_quantities/tagged/") # plots of the event topology

    MakePlots(data, args.out + "basic_quantities/")

    if args.selection_type == "reco":
        MakePlotsTagged(tagged_data, selected_tags, args.out + "basic_quantities/tagged/")

    rprint(f"plots and tables saved to: {args.out}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", nargs = "+", help = "NTuple file to study.")
    
    parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "number of batches to split n tuple files into when parallel processing processing data.")
    parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "number of events to process when parallel processing data.")

    parser.add_argument("-t", "--used-threads", dest = "use_threads", action = "store_true", help = "sets the number of batches to the number of threads on the machine.")
    
    parser.add_argument("-s", "--selection", dest = "selection_type", type = str, choices = ["cheated", "reco"], help = "type of selection to use.", required = True)

    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save plots")
    parser.add_argument("-a", "--annotation", dest = "annotation", type = str, default = None, help = "annotation to add to plots")

    args = parser.parse_args()

    if args.out is None:
        if len(args.file) == 1:
            args.out = args.file[0].split("/")[-1].split(".")[0] + "/"
        else:
            args.out = "selection_studies/" #? how to make a better name for multiple input files?
    if args.out[-1] != "/": args.out += "/"

    if args.use_threads:
        args.batches = os.cpu_count()

    rprint(vars(args))
    main(args)