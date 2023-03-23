#!/usr/bin/env python3
"""
Created on: 09/08/2022 14:41

Author: Shyam Bhuller

Description: Process both ROOT and csv data for the shower merging analysis with production 4a MC.
TODO break workflows into different apps.
"""
import argparse
import itertools
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from python.analysis import Master, Plots, vector, shower_merging, CutOptimization


def ROOTWorkFlow():
    """ Analysis that can be done when supplied a NTuple file.
        Can calculate:
         - basic quantities
         - geometric quantities for signal/background discrimination
         - pair quantities and shower merging
    """
    events = Master.Data(file, nEvents = args.nEvents[0], start = args.nEvents[1])
    shower_merging.EventSelection(events)
    shower_merging.ValidPFOSelection(events)
    start_showers, to_merge = shower_merging.SplitSample(events, args.matchBy)
    
    #* class to calculate quantities
    q = shower_merging.ShowerMergeQuantities(events, to_merge, args.analysedCuts)

    if args.merge == None:
        signal, background, signal_all = shower_merging.SignalBackground(events, start_showers, to_merge)

        #* plot number of signal and background per event
        nSignal = ak.count(signal_all[signal_all], -1)
        nBackground = ak.count(background[background], -1)

        print(f"Total number of Signal PFOs :{ak.sum(nSignal)}")
        print(f"Total number of background PFOs :{ak.sum(nBackground)}")

        if plotsToMake == "all":
            subDir = "basic_quantities/"
            os.makedirs(outDir+subDir, exist_ok=True)
            labels = ["background", "signal"]
            
            Plots.PlotHist(ak.ravel(nSignal), xlabel="Start shower multiplicity", density=norm, y_scale=scale, annotation=args.dataset)
            if save: Plots.Save("multiplicity", outDir+subDir)

            Plots.PlotHistComparison([nBackground, nSignal], xlabel="Number of PFOs", bins=20, labels=labels, density=norm, y_scale=scale, annotation=args.dataset)
            if save: Plots.Save("nPFO", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(vector.magnitude(events.trueParticles.momentum[events.trueParticles.PrimaryPi0Mask])), nSignal, 50, xlabel = "True $\pi^{0}}$ momentum (GeV)", ylabel="Number of signal PFO", annotation=args.dataset)
            if save: Plots.Save("pi0_p_vs_nPFO_signal", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(vector.magnitude(events.trueParticles.momentum[events.trueParticles.PrimaryPi0Mask])), nBackground, 50, xlabel = "True $\pi^{0}}$ momentum (GeV)", ylabel="Number of background PFO", annotation=args.dataset)
            if save: Plots.Save("pi0_p_vs_nPFO_background", outDir+subDir)

            nbins =  max(nSignal) - min(nSignal)
            Plots.PlotHist(nSignal, xlabel="Number of signal PFOs", bins=np.arange(nbins)-0.5, y_scale=scale, annotation=args.dataset)
            if save: Plots.Save("nPFO_signal", outDir+subDir)

            Plots.PlotHist(nBackground, xlabel="Number of background PFOs", bins=20, y_scale=scale, annotation=args.dataset)
            if save: Plots.Save("nPFO_background", outDir+subDir)

            Plots.PlotHistComparison([ak.ravel(events.recoParticles.energy[to_merge][background]), ak.ravel(events.recoParticles.energy[to_merge][np.logical_or(*signal)])], xlabel="Energy (MeV)", bins=20, labels=labels, density = norm, y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("energy", outDir+subDir)

            Plots.PlotHistComparison([ak.ravel(events.recoParticles.nHits_collection[to_merge][background]), ak.ravel(events.recoParticles.nHits_collection[to_merge][np.logical_or(*signal)])], xlabel="Number of hits", bins=20, labels=labels, density = norm, y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("hits", outDir+subDir)

            Plots.PlotHistComparison([ak.ravel(events.recoParticles.cnnScore[to_merge][background]), ak.ravel(events.recoParticles.cnnScore[to_merge][np.logical_or(*signal)])], xlabel="CNN score", bins=20, labels=labels, density = norm, y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("cnn", outDir+subDir)

            purity = events.trueParticlesBT.purity
            completeness = events.trueParticlesBT.completeness

            start_showers_all = np.logical_or(*start_showers)
            Plots.PlotHist(ak.ravel(purity[start_showers_all]), xlabel="start shower purity", y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("ss-purity", outDir+subDir)
            Plots.PlotHist(ak.ravel(completeness[start_showers_all]), xlabel="start shower completeness", y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("ss-completeness", outDir+subDir)

            Plots.PlotHistComparison([ak.ravel(purity[to_merge][background]), ak.ravel(purity[to_merge][np.logical_or(*signal)])], labels=labels, xlabel="purity", y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("purity", outDir+subDir)
            Plots.PlotHistComparison([ak.ravel(completeness[to_merge][background]), ak.ravel(completeness[to_merge][np.logical_or(*signal)])], labels=labels, xlabel="completeness", y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("completeness", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(purity), ak.ravel(completeness), xlabel="purity", ylabel="completeness")
            if save: Plots.Save("purity_vs_completeness", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(purity[to_merge][np.logical_or(*signal)]), ak.ravel(completeness[to_merge][np.logical_or(*signal)]), bins = 25, xlabel="purity", ylabel="completeness", title = "signal")
            if save: Plots.Save("purity_vs_completeness_s", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(purity[to_merge][background]), ak.ravel(completeness[to_merge][background]), bins = 25, xlabel="purity", ylabel="completeness", title = "background")
            if save: Plots.Save("purity_vs_completeness_b", outDir+subDir)

        #* calculate geometric quantities
        if save is True and plotsToMake is None:
            q.Evaluate(events, start_showers)
            if args.csv is None:
                q.SaveQuantitiesToCSV(signal, background)
            else:
                q.SaveQuantitiesToCSV(signal, background, args.csv)
    else:
        if args.merge == "reco":
            q.bestCut = args.cut_type
            q.to_merge_dir = events.recoParticles.direction
            q.to_merge_pos = events.recoParticles.startPos
            start_showers = shower_merging.ShowerMerging(events, start_showers, q, -1)
            start_showers_all = np.logical_or(*start_showers)

        elif args.merge == "unmerged":
            start_showers_all = np.logical_or(*start_showers)
            #events.Filter([np.logical_or(*start_showers)])

        elif args.merge == "cheat":
            # start_shower_ID = events.trueParticlesBT.number[np.logical_or(*start_showers)]
            # pi0_PFOs = [events.trueParticlesBT.number == start_shower_ID[:, i] for i in range(2)]
            # pi0_PFOs = np.logical_or(*pi0_PFOs)
            # events.Filter([pi0_PFOs])
            events.MergePFOCheat(1)
            start_showers_all = events.trueParticlesBT.mother == ak.flatten(events.trueParticles.number[events.trueParticles.PrimaryPi0Mask])

        else:
            raise Exception("Don't understand the merge type")

        pairs = Master.ShowerPairs(events, shower_pair_mask=start_showers_all)
        pairs.SaveToCSV(args.outDir + args.csv)
        # p = Master.CalculateQuantities(s, True)
        # PairQuantitiesToCSV(p)


def ShowerMergingCriteria(q : shower_merging.ShowerMergeQuantities):
    """ Performs a cut based scan on various criteria that can be used for shower merging

    Args:
        q (ShowerMergeQuantities): quantities to perform cut based scan on
    """
    def Spinner(counter : int, spinner="lines") -> str:
        """ Janky spinner, cause why not?

        Args:
            counter (int): iteration
            spinner (str, optional): type of spinner. Defaults to "lines".

        Returns:
            str: string to print at this interval
        """
        spinners = {
            "lines" : "-\|/",
            "box"   : "⠦⠆⠖⠒⠲⠰⠴⠤",
        }
        return spinners[spinner][counter % len(spinners[spinner])]

    min_val = [] # min range of each variable
    max_val = [] # max range
    for i in range(len(q.selectionVariables)):
        min_val.append(0)
        max_val.append(ak.max(getattr(q, q.selectionVariables[i])))

    min_val = np.array(min_val)
    max_val = np.array(max_val)

    values = np.linspace(min_val+(0.1*max_val), max_val-(0.1*max_val), 3, True) # values that are used to create combinations of cuts to optimize
    metric_labels = ["s", "b", "s/b", "$s\\sqrt{b}$", "purity", "$\\epsilon_{s}$", "$\\epsilon_{b}$", "$\\epsilon$"] # performance metrics to choose cuts #? add purity*efficiency?

    #* create input data strutcure
    counter = 0

    cuts = []
    for v in q.selectionVariables:
        operator = CutOptimization.Operator.GREATER if v == "cnn" else CutOptimization.Operator.LESS
        cuts.append(CutOptimization.Cuts(v, operator, None))

    print("list of cut types:")
    print(cuts)

    if args.csv is None:
        output_path = f"{outDir}analysedCuts.csv"
    else:
        output_path = f"{outDir}{args.csv}"

    #* loop through all combination of values for each parameter and optmize the final cut
    for initial_cuts in itertools.product(*values.T):
        for i in range(len(cuts)):
            cuts[i].value = initial_cuts[i]
        cutOptimization = CutOptimization.OptimizeSingleCut(q, cuts, False)
        c, m = cutOptimization.Optimize(10, CutOptimization.MaxSRootBRatio) # scan over 10 bins and optimize cut by looking for max s/sqrt(b)

        o = [c[i] + m[i] for i in range(len(c))] # combine output
        o = pd.DataFrame(o, columns = q.selectionVariables + metric_labels)
        o.to_csv(output_path, mode = "a", header = not os.path.exists(output_path))

        counter += 1
        end = '\n' if counter == 0 else '\r'
        print(f" {Spinner(counter, 'box')} progess: {counter/(len(values)**len(initial_cuts))*100:.3f}% | {counter} | {len(values)**len(initial_cuts)}", end=end)


def CSVWorkFlow():
    """ Analysis that can be done with the csv files produced by this program
    """
    q = shower_merging.ShowerMergeQuantities(analysedCuts=args.analysedCuts) # can apply cuts to shower quantities
    q.LoadQuantitiesFromCSV(file)
    if args.cut is True:
        ShowerMergingCriteria(q)
        return
    if plotsToMake in ["all", "quantities"]:
        q.PlotQuantities(q.signal, q.background, False)


@Master.timer
def main():
    plt.style.use('ggplot')
    plt.rcParams.update({'patch.linewidth': 1})
    plt.rcParams.update({'font.size': 10})
    if save:
        os.makedirs(outDir, exist_ok = True)
    fileFormat = file.split('.')[-1]
    if fileFormat == "root":
        ROOTWorkFlow()
    if fileFormat == "csv":
        CSVWorkFlow()

if __name__ == "__main__":
    example_usage = """Example Uasge:
    Open a ROOT file and plot basic quantities:
        prod4a_merge_study.py <ROOT file> -s -p <plot type> -d <out directory> -n

    Open a ROOT file and save merge quantities to file:
        prod4a_merge_study.py <ROOT file> -s -d <out directory>

    Open a csv file with merge quantities and plot them:
        prod4a_merge_study.py <merge quantity csv> -p <plot type> -s -d <out directory> -n

    Open a csv file with merge quantities and scan for cut values:
        prod4a_merge_study.py <merge quantity csv> -p <plot type> -s -d <out directory> -c

    Open a ROOT file and csv with list of cuts and merge PFOs based on the cut type:
        prod4a_merge_study.py <ROOT file> --cuts <cuts csv> --cut-type <cut type> -m reco -s -o <output filename> -d <out directory> -a

    Open a ROOT file, merge PFOs based on truth information and save shower pair quantities to file:
        prod4a_merge_study.py <ROOT file> -m cheat -s -o <output filename> -d <out directory>
    """

    parser = argparse.ArgumentParser(description = "Shower merging study for beamMC, plots quantities used to decide which showers to merge.", formatter_class = argparse.RawDescriptionHelpFormatter, epilog = example_usage)
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-e", "--events", dest="nEvents", type=int, nargs=2, default=[-1, 0], help="number of events to analyse and number to skip (-1 is all)")
    parser.add_argument("-n", "--normalize", dest="norm", action="store_true", help="normalise plots to compare shape")
    parser.add_argument("-l" "--log", dest="log", action="store_true", help="plot y axis on log scale")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="prod4a_merge_study/", help="directory to save plots")
    parser.add_argument("-p", "--plots", dest="plotsToMake", type=str, choices=["all", "quantities", "multiplicity", "nPFO", "2D"], help="what plots we want to make")
    parser.add_argument("-c", "--cutScan", dest="cut", action="store_true", help="whether to do a cut based scan")
    parser.add_argument("--start-showers", dest="matchBy", type=str, choices=["angular", "spatial"], default="spatial", help="method to detemine start showers")
    parser.add_argument("--cuts", dest="analysedCuts", default=None, type=str, help="data produced by ShowerMergingCriteria i.e. use the -c option")
    parser.add_argument("-a", "--apply-cuts", dest="applyCuts", action="store_true", help="apply cuts to shower merge quantities")
    parser.add_argument("-m", "--merge", dest="merge", type=str, choices=["unmerged", "reco", "cheat", None], default=None, help="Do shower merging (cuts required)")
    parser.add_argument("--cut-type", dest="cut_type", type=str, choices=["purity", "balanced", "efficiency"], default="balanced", help="type of cut to pick from cut scan.")
    parser.add_argument("-o", "--out-csv", dest="csv", type=str, default=None, help="output csv filename (will default to whatever type of data is produced)")
    parser.add_argument("--annotation", dest="dataset", type=str, help="annotation for plots.")
    args = parser.parse_args() #! run in command line
    print(vars(args))
    
    file = args.file
    save = args.save
    outDir = args.outDir
    plotsToMake = args.plotsToMake
    norm = args.norm

    if plotsToMake is not None:
        print("making directory")
        os.makedirs(outDir, exist_ok=True)
        print(f"made {outDir}")

    if args.csv is not None: args.csv += ".csv"
    
    if args.log is True:
        scale = "log"
    else:
        scale = "linear"

    main()