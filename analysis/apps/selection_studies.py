#!/usr/bin/env python3
"""
Created on: 14/03/2023 15:14

Author: Shyam Bhuller

Description: Applies beam particle selection, PFO selection, produces tables and basic plots.
"""
import argparse
import os

import awkward as ak
import numpy as np

from python.analysis import Master, Plots, shower_merging


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

    #* apply either cheated or reco selection
    match args.pfo_selection_type:
        case "cheated":
            events_table, pfo_table = shower_merging.Selection(events, args.beam_selection_type, args.pfo_selection_type)
            start_showers, to_merge = shower_merging.SplitSample(events)
        case "reco":
            events_table, pfo_table, photon_candidate_table = shower_merging.Selection(events, args.beam_selection_type, args.pfo_selection_type)
            start_showers, to_merge = shower_merging.SplitSampleReco(events)

            if args.save:
                photon_candidate_table.to_latex(args.out + "photon_candidate_selection.tex")
        case _:
            raise Exception(f"event selection type {args.pfo_selection_type} not understood.")

    if args.save:
        events_table.to_latex(args.out + "event_selection.tex")
        pfo_table.to_latex(args.out + "pfo_selection.tex")


    MakePlots(events, start_showers, to_merge, args.save, args.out)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", type = str, help = "NTuple file to study.")
    parser.add_argument("-e", "--events", dest = "nEvents", type = int, nargs = 2, default = [-1, 0], help = "number of events to analyse and number to skip (-1 is all)")

    parser.add_argument("-b", "--beam-particle-selection", dest = "beam_selection_type", type = str, choices = ["cheated", "reco"], help = "type of beam particle selection to use.", required = True)
    parser.add_argument("-p", "--pfo-selection", dest = "pfo_selection_type", type = str, choices = ["cheated", "reco"], help = "type of pfo selection to use when selecting photon shower candidates.", required = True)

    parser.add_argument("-s", "--save", dest = "save", action = "store_true", help = "whether to save the plots")
    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save plots")
    parser.add_argument("-a", "--annotation", dest = "annotation", type = str, default = None, help = "annotation to add to plots")

    args = parser.parse_args()

    if args.out is None:
        args.out = args.file.split("/")[-1].split(".")[0] + "/"

    print(vars(args))
    main(args)