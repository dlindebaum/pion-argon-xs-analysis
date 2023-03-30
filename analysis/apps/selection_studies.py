#!/usr/bin/env python3
"""
Created on: 14/03/2023 15:14

Author: Shyam Bhuller

Description: Applies beam particle selection, PFO selection, produces tables and basic plots.
"""
import argparse
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from rich import print

from python.analysis import Master, Plots, shower_merging, EventSelection

from dataclasses import dataclass

@dataclass(slots = True)
class tag:
    name : str = ""
    name_simple : str = ""
    colour : str = ""
    mask : ak.Array = ""


def MakePlots(events : Master.Data, start_showers : ak.Array, to_merge : ak.Array, save : bool, out : str):
    """ Make plots of some basic quantities.

    Args:
        events (Master.Data): events to study
        start_showers (ak.Array): initial photon candidates
        to_merge (ak.Array): remaining PFOs
        save (bool): save plots?
        out (str): ouput file path
    """
    signal, background, signal_all = shower_merging.SignalBackground(events, start_showers, to_merge)

    #* plot number of signal and background per event
    nSignal = ak.count(signal_all[signal_all], -1)
    nBackground = ak.count(background[background], -1)

    print(f"Total number of Signal PFOs :{ak.sum(nSignal)}")
    print(f"Total number of background PFOs :{ak.sum(nBackground)}")

    if save:
        path = out + "basic_quantities/"
        os.makedirs(path, exist_ok = True)
        labels = ["background", "signal"]
        
        Plots.PlotHist(ak.ravel(nSignal), xlabel = "Start shower multiplicity", annotation = args.annotation)
        Plots.Save("multiplicity", path)

        Plots.PlotHistComparison([nBackground, nSignal], xlabel = "Number of PFOs", bins = 20, labels = labels, annotation = args.annotation)
        Plots.Save("nPFO", path)

        #TODO fix these plots for cheated selection
        # Plots.PlotHist2D(ak.ravel(vector.magnitude(events.trueParticles.momentum[events.trueParticles.PrimaryPi0Mask])), nSignal, 50, xlabel = "True $\pi^{0}}$ momentum (GeV)", ylabel = "Number of signal PFO", annotation = args.annotation)
        # Plots.Save("pi0_p_vs_nPFO_signal", path)

        # Plots.PlotHist2D(ak.ravel(vector.magnitude(events.trueParticles.momentum[events.trueParticles.PrimaryPi0Mask])), nBackground, 50, xlabel = "True $\pi^{0}}$ momentum (GeV)", ylabel = "Number of background PFO", annotation = args.annotation)
        # Plots.Save("pi0_p_vs_nPFO_background", path)

        nbins =  max(nSignal) - min(nSignal)
        Plots.PlotHist(nSignal, xlabel="Number of signal PFOs", bins=np.arange(nbins)-0.5, annotation = args.annotation)
        Plots.Save("nPFO_signal", path)

        Plots.PlotHist(nBackground, xlabel = "Number of background PFOs", bins = 20, annotation = args.annotation)
        Plots.Save("nPFO_background", path)

        Plots.PlotHistComparison([ak.ravel(events.recoParticles.energy[to_merge][background]), ak.ravel(events.recoParticles.energy[to_merge][np.logical_or(*signal)])], xlabel = "Energy (MeV)", bins = 20, y_scale = "log", labels = labels, annotation = args.annotation)
        Plots.Save("energy", path)

        Plots.PlotHistComparison([ak.ravel(events.recoParticles.nHits_collection[to_merge][background]), ak.ravel(events.recoParticles.nHits_collection[to_merge][np.logical_or(*signal)])], xlabel = "Number of collection plane hits", bins = 20, y_scale = "log", labels = labels, annotation = args.annotation)
        Plots.Save("hits_collection", path)

        Plots.PlotHistComparison([ak.ravel(events.recoParticles.nHits_collection[to_merge][background]), ak.ravel(events.recoParticles.nHits[to_merge][np.logical_or(*signal)])], xlabel = "Number of hits", bins = 20, y_scale = "log", labels = labels, annotation = args.annotation)
        Plots.Save("hits", path)

        Plots.PlotHistComparison([ak.ravel(events.recoParticles.cnnScore[to_merge][background]), ak.ravel(events.recoParticles.cnnScore[to_merge][np.logical_or(*signal)])], xlabel = "CNN score", bins = 20, labels = labels, annotation = args.annotation)
        Plots.Save("cnn", path)

        purity = events.trueParticlesBT.purity
        completeness = events.trueParticlesBT.completeness

        start_showers_all = np.logical_or(*start_showers)
        
        Plots.PlotHist(ak.ravel(purity[start_showers_all]), xlabel = "start shower purity", annotation = args.annotation)
        Plots.Save("ss-purity", path)
        
        Plots.PlotHist(ak.ravel(completeness[start_showers_all]), xlabel = "start shower completeness", annotation = args.annotation)
        Plots.Save("ss-completeness", path)

        Plots.PlotHistComparison([ak.ravel(purity[to_merge][background]), ak.ravel(purity[to_merge][np.logical_or(*signal)])], labels = labels, xlabel = "purity", annotation = args.annotation)
        Plots.Save("purity", path)
        
        Plots.PlotHistComparison([ak.ravel(completeness[to_merge][background]), ak.ravel(completeness[to_merge][np.logical_or(*signal)])], labels = labels, xlabel = "completeness", annotation = args.annotation)
        Plots.Save("completeness", path)

        Plots.PlotHist2D(ak.ravel(purity), ak.ravel(completeness), xlabel = "purity", ylabel = "completeness")
        Plots.Save("purity_vs_completeness", path)

        Plots.PlotHist2D(ak.ravel(purity[to_merge][np.logical_or(*signal)]), ak.ravel(completeness[to_merge][np.logical_or(*signal)]), bins = 25, xlabel = "purity", ylabel = "completeness", title = "signal")
        Plots.Save("purity_vs_completeness_s", path)

        Plots.PlotHist2D(ak.ravel(purity[to_merge][background]), ak.ravel(completeness[to_merge][background]), bins = 25, xlabel = "purity", ylabel = "completeness", title = "background")
        Plots.Save("purity_vs_completeness_b", path)
    return


def main(args):
    shower_merging.SetPlotStyle()
    events = Master.Data(args.file, nEvents = args.nEvents[0], start = args.nEvents[1])

    if args.save: os.makedirs(args.out, exist_ok = True)

    #* apply either cheated or reco selection
    match args.pfo_selection_type:
        case "cheated":
            events_table, pfo_table = shower_merging.Selection(events, args.beam_selection_type, args.pfo_selection_type)
            start_showers, to_merge = shower_merging.SplitSample(events)
        case "reco":
            events_table, pfo_table, photon_candidate_table = shower_merging.Selection(events, args.beam_selection_type, args.pfo_selection_type, False)

            n_photon_candidates = ak.num(events.recoParticles.number[shower_merging.PFOSelection.InitialPi0PhotonSelection(events)])

            if args.tag:
                tags = {
                    "$\geq 1\pi^{0} + X$"        : tag("$\geq 1\pi^{0} + X$",        "inclusive signal", "#348ABD", EventSelection.generate_truth_tags(events, (1,), 0)),
                    "$1\pi^{0} + 0\pi^{+}$"      : tag("$1\pi^{0} + 0\pi^{+}$",      "exclusive signal", "#8EBA42", EventSelection.generate_truth_tags(events, 1, 0)),
                    "$0\pi^{0} + 0\pi^{+}$"      : tag("$0\pi^{0} + 0\pi^{+}$",      "background",       "#777777", EventSelection.generate_truth_tags(events, 0, (0,))),
                    "$1\pi^{0} + \geq 1\pi^{+}$" : tag("$1\pi^{0} + \geq 1\pi^{+}$", "sideband",         "#E24A33", EventSelection.generate_truth_tags(events, 1, (1,))),
                    "$0\pi^{0} + \geq 1\pi^{+}$" : tag("$0\pi^{0} + \geq 1\pi^{+}$", "sideband",         "#988ED5", EventSelection.generate_truth_tags(events, 0, (1,))),
                }

                MakeInitialTaggingPlots(tags, n_photon_candidates, args.save, args.out)

            photon_candidates = n_photon_candidates == 2
            events.Filter([photon_candidates], [photon_candidates])

            if args.tag:
                for k in tags:
                    tags[k].mask = tags[k].mask[photon_candidates] # apply photon candidate masks to the tags

            start_showers, to_merge = shower_merging.SplitSampleReco(events)

            if args.save:
                photon_candidate_table.to_latex(args.out + "photon_candidate_selection.tex")
        case _:
            raise Exception(f"event selection type {args.pfo_selection_type} not understood.")

    if args.save:
        events_table.to_latex(args.out + "event_selection.tex")
        pfo_table.to_latex(args.out + "pfo_selection.tex")

    if args.tag:
        MakePlotsTagged(events, tags, start_showers, to_merge, args.save, args.out)
    else:
        MakePlots(events, start_showers, to_merge, args.save, args.out)
    return


def MakePlotsTagged(events : Master.Data, tags, start_showers, to_merge, save, out):
    if save:
        path = out + "basic_quantities/tagged/"
        os.makedirs(path, exist_ok = True)

        start_showers_all = np.logical_or(*start_showers)
        signal, background, signal_all = shower_merging.SignalBackground(events, start_showers, to_merge)

        n_signal = ak.count(signal_all[signal_all], -1)
        n_signal_bins =  max(n_signal) - min(n_signal) + 2

        n_background = ak.count(background[background], -1)

        labels = []
        colours = []
        n_PFO = []
        n_signal_tagged = []
        n_background_tagged = []
        purity = []
        completeness = []
        pdg = []
        mother_pdg = []
        for k, t in tags.items():
            labels.append(k)
            colours.append(t.colour)

            n_PFO.append(ak.num(events.recoParticles.number)[t.mask])

            n_signal_tagged.append(ak.ravel(n_signal[t.mask]))
            n_background_tagged.append(ak.ravel(n_background[t.mask]))

            completeness.append(ak.ravel(events.trueParticlesBT.completeness[start_showers_all])[t.mask])
            purity.append(ak.ravel(events.trueParticlesBT.purity[start_showers_all])[t.mask])

            pdg.append(list(np.unique((events.trueParticlesBT.pdg[start_showers_all])[tags[k].mask], return_counts = True)))
            mother_pdg.append(list(np.unique((events.trueParticlesBT.motherPdg[start_showers_all])[tags[k].mask], return_counts = True)))

        Plots.PlotHist(n_PFO, bins = 20, xlabel = "number of PFOs", stacked = True, label = labels, color = colours, annotation = args.annotation)
        Plots.Save("tagged_nPFO", path)

        Plots.PlotHist(n_signal_tagged, bins = np.arange(n_signal_bins) - 0.5, xlabel = "mutiplicity", stacked = True, label = labels, color = colours, annotation = args.annotation)
        plt.xticks(np.arange(n_signal_bins))
        Plots.Save("tagged_multiplicity", path)

        Plots.PlotHist(n_signal_tagged, bins = 20, xlabel = "number of signal PFOs", stacked = True, label = labels, color = colours, annotation = args.annotation)
        Plots.Save("tagged_nPFO_signal", path)

        Plots.PlotHist(n_background_tagged, bins = 20, xlabel = "Number of background PFOs", stacked = True, label = labels, color = colours, annotation = args.annotation)
        Plots.Save("tagged_nPFO_background", path)

        Plots.PlotHist(completeness, bins = 20, xlabel = "start shower completeness", stacked = True, label = labels, color = colours, annotation = args.annotation)
        Plots.Save("tagged_start_shower_completeness", path)

        Plots.PlotHist(purity, bins = 20, xlabel = "start shower purity", stacked = True, label = labels, color = colours, annotation = args.annotation)
        Plots.Save("tagged_start_shower_purity", path)

        Plots.PlotStackedBar(pdg, xlabel = "pdg of intial photon candidates", label_title = "initial $\pi^{0}$ photon candidates.", labels = labels, colours = colours, annotation = args.annotation)
        if save: Plots.Save("tagged_pdg", path)

        Plots.PlotStackedBar(mother_pdg, xlabel = "mother pdg of initial photon candidates", label_title = "initial $\pi^{0}$ photon candidates.", labels = labels, colours = colours, annotation = args.annotation)
        if save: Plots.Save("tagged_mother_pdg", path)



def MakeInitialTaggingPlots(tags, n_photons, save, out):
    if save:
        path = out + "basic_quantities/tagged/"
        os.makedirs(path, exist_ok = True)

    for s in np.unique(n_photons):
        plt.figure()
        for name, t in tags.items():
            c = shower_merging.CountMask(t.mask[n_photons == s])
            Plots.PlotBar([name]*c, xlabel = "truth tag", color = t.colour, label = t.name_simple, title = f"number of shower candidates: {s}", newFigure = False)
        if save: Plots.Save(f"truth_tags_nPFO_{s}", path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", type = str, help = "NTuple file to study.")
    parser.add_argument("-e", "--events", dest = "nEvents", type = int, nargs = 2, default = [-1, 0], help = "number of events to analyse and number to skip (-1 is all)")

    parser.add_argument("-b", "--beam-particle-selection", dest = "beam_selection_type", type = str, choices = ["cheated", "reco"], help = "type of beam particle selection to use.", required = True)
    parser.add_argument("-p", "--pfo-selection", dest = "pfo_selection_type", type = str, choices = ["cheated", "reco"], help = "type of pfo selection to use when selecting photon shower candidates.", required = True)
    parser.add_argument("-t", "--tag", dest = "tag", action = "store_true", help = "generate truth level tags for events.")

    parser.add_argument("-s", "--save", dest = "save", action = "store_true", help = "whether to save the plots")
    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save plots")
    parser.add_argument("-a", "--annotation", dest = "annotation", type = str, default = None, help = "annotation to add to plots")

    args = parser.parse_args()

    if args.out is None:
        args.out = args.file.split("/")[-1].split(".")[0] + "/"

    print(vars(args))
    main(args)