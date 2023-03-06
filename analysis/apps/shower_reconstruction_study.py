"""
Created on: 04/03/2022 16:01

Author: Shyam Bhuller

Description: Script which will read an ntuple ROOT file of MC and analyse reconstructed shower properties.
"""
import argparse
import os

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from notebooks import merge_study
# custom modules
from python.analysis import Master, Plots


def Plot2DRatio(ind : int, truths : np.array, errors : np.array, labels : str, xlabels : str, ylabels : str, nrows : int, ncols : int, bins : int = 25):
    """ Plot ratio of 2D histograms

    Args:
        ind (int): index of quantity to plot
        truths (np.array): true data
        errors (np.array): fractional error
        labels (str): plot labels
        xlabels (str): x labels
        ylabels (str): y labels
        nrows (int): number of rows
        ncols (int): number of columns
        bins (int, optional): number of bins per axes. Defaults to 25.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4*ncols,4.8*nrows)) # make subplots
    for i in range(len(s_l)):
        x = truths[i][ind]
        y = errors[i][ind]
        if len(e_range[ind]) == 0:
            y_range = [min(y), max(y)]
        else:
            y_range = e_range[ind]

        if len(np.unique(x)) == 1:
            x_range = [min(x)-0.01, max(x)+0.01] # do this to allow plots of data with single bins
        else:
            x_range = [min(x), max(x)]
        if i == 0:
            # first histogram is the denominator for other histograms
            h0, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range ], density=True)
            h0[h0==0] = np.nan
            h0T = h0.T # transpose cause numpy is wierd
            im = axes.flat[i].imshow(np.flip(h0T, 0), extent=[x_range[0], x_range[1], y_range[0], y_range[1]], norm=matplotlib.colors.LogNorm())#, norm=norm, cmap=cmap)
            fig.colorbar(im, ax=axes.flat[i])
        else:
            h, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], range=[x_range, y_range], density=True)
            h = h / h0
            h[h==0] = np.nan
            im = axes.flat[i].imshow(np.flip(h.T, 0), extent=[x_range[0], x_range[1], y_range[0], y_range[1]], norm=matplotlib.colors.LogNorm())#, norm=norm, cmap=cmap)
            fig.colorbar(im, ax=axes.flat[i])
        axes.flat[i].set_aspect("auto")
        axes.flat[i].set_title(labels[i])
    # add common x and y axis labels
    fig.add_subplot(1, 1, 1, frame_on=False)
    # Hiding the axis ticks and tick labels of the bigger plot
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    # Adding the x-axis and y-axis labels for the bigger plot
    plt.tight_layout()
    plt.xlabel(xlabels, labelpad=10, fontsize=14)
    plt.ylabel(ylabels, labelpad=15, fontsize=14)


def Plot2DMulti(x : ak.Array, y : ak.Array, xlabel : str, ylabel : str, name : str, x_range : list = [], y_range : list = []):
    """ Plot generic 2D histograms

    Args:
        x (ak.Array): x data
        y (ak.Array): y data
        xlabel (str): x label
        ylabel (str): y label
        name (str): name of image to save
        x_range (list) : plot range on x axis
        y_range (list) : plot range on y axis
    """
    plt.rcParams["figure.figsize"] = (6.4*2,4.8*3)
    plt.figure()
    for i in range(len(s_l)):
        plt.subplot(3, 2, i+1)
        if i == 0:
            _, edges = Plots.PlotHist2D(x[i], y[i], bins, x_range, y_range, xlabel, ylabel, title=s_l[i], newFigure=False)
        else:
            Plots.PlotHist2D(x[i], y[i], edges, x_range, y_range, xlabel, ylabel, title=s_l[i], newFigure=False)
    if save is True: Plots.Save( name , outDir + "2D/")
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def Plot1D(data : ak.Array, xlabels : list, subDir : str, labels : list, plot_ranges = [[]]*5, legend_loc = ["upper right"]*5, x_scale=["linear"]*5, y_scale=["linear"]*5):
    """ 1D histograms of data for each sample

    Args:
        data (ak.Array): list of samples data to plot
        xlabels (list): x labels
        subDir (str): subdirectiory to save in
        plot_range (list, optional): range to plot. Defaults to [].
    """
    if save is True: os.makedirs(outDir + subDir, exist_ok=True)
    for i in range(len(names)):
        if len(labels) == 1:
            d = data[0][i]
            if len(plot_ranges[i]) == 2:
                d = d[d > plot_ranges[i][0]]
                d = d[d < plot_ranges[i][1]]
            else:
                d = d[d > -999]
            Plots.PlotHist(d, bins, xlabel=xlabels[i], x_scale=x_scale[i], y_scale=y_scale[i], density=True)
        else:
            Plots.PlotHistComparison(data[:, i], plot_ranges[i], bins, xlabel=xlabels[i], histtype="step", labels=labels, x_scale=x_scale[i], y_scale=y_scale[i], density=True)
            plt.legend(loc=legend_loc[i])
        if save is True: Plots.Save( names[i] , outDir + subDir)


def Plot2D(true_data : ak.Array, error_data : ak.Array):
    """ Plots 2D histograms

    Args:
        true_data (ak.Array): true quantities
        error_data (ak.Array): fraction errors
    """
    # plot 2D plots of all quantities
    if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
    if len(s_l) > 1:
        for i in range(len(names)):
            Plot2DRatio(i, true_data, error_data, s_l, t_l[i], e_l[i], figSize2D[0], figSize2D[1])
            if save is True: Plots.Save(names[i], outDir + "2D/")
            if save is True: os.makedirs(outDir + "2D/", exist_ok=True)

        # plot opening angle and invariant mass against pi0 momentum
        #Plot2DMulti(true_data[:, 4], error_data[:, 0], t_l[4], e_l[0], "inv_mass_vs_mom", t_range[4], e_range[0])
        #Plot2DMulti(true_data[:, 4], error_data[:, 1], t_l[4], e_l[1], "angle_vs_mom", t_range[4], e_range[0])
    else:
        for i in range(len(names)):
            Plots.PlotHist2D(true_data[0, i], error_data[0, i], bins, t_range[i], e_range[i], t_l[i], e_l[i])
            if save is True: Plots.Save(names[i], outDir + "2D/")


def SelectSample(events : Master.Data, nDaughters : int, merge : bool = False, backtracked : bool = False, cheatMerging : bool = False):
    """ Applies MC matching and shower merging to events with
        specificed number of objects

    Args:
        events (Master.Data): events to look at
        nDaughters (int): number of objects per event
        merge (bool, optional): should we do shower merging?. Defaults to False.

    Returns:
        Master.Data: selected sample
    """

    valid = Master.Pi0MCMask(events, nDaughters) # get mask of events
    filtered = events.Filter([valid], [valid], True) # filter events with mask

    if backtracked == False:
        matched, unmatched, selection = filtered.MCMatching(applyFilters=False)
        filtered.Filter([selection],[selection]) # apply the selection
    else:
        singleMatchedEvents = filtered.trueParticlesBT.SingleMatch
        filtered.Filter([singleMatchedEvents], [singleMatchedEvents])
        best_match, selection = filtered.MatchByAngleBT()
        filtered.Filter([selection], [selection])

    if merge is True and backtracked is False:
        filtered = merge_study.mergeShower(filtered, matched[selection], unmatched[selection], 1, False)

    if merge is True and backtracked is True:
        if cheatMerging is True:
            filtered, null = filtered.MergePFOCheat()
            filtered.Filter([null], [null])
        else:
            filtered = filtered.MergeShowerBT(best_match[selection])

    # if we don't merge showers, just get the showers matched to MC
    if merge is False and backtracked is False:
        filtered.Filter([matched[selection]])
    if merge is False and backtracked is True:
        filtered.Filter([best_match[selection]])
    return filtered


def PlotFromCSV():
    t = []
    r = []
    e = []
    for f in file:
        data = np.genfromtxt(f, delimiter=',')
        data = data[1:, 1:]

        t.append(np.transpose(data[:, 0:5]))
        r.append(np.transpose(data[:, 5:10]))
        e.append(np.transpose(data[:, 10:15]))
    t = ak.Array(t)
    r = ak.Array(r)
    e = ak.Array(e)

    print(f"number of pi0 decays: {len(t[0][0])}")

    if plotsToMake in ["all", "truth"]: Plot1D(t, t_l, "truth/", s_l, t_range, t_locs) # MC truth for unmerged and merged are identical, so don't plot them
    if plotsToMake in ["all", "reco"]: Plot1D(r, r_l, "reco/", s_l, r_range, r_locs, r_xs, r_ys)
    if plotsToMake in ["all", "error"]: Plot1D(e, e_l, "error/", s_l, e_range, e_locs)
    if plotsToMake in ["all", "2D"]: Plot2D(t, e)


def AnalyseMultipleFiles():
    nPFP = []
    for f in file:
        events = Master.Data(f)
        events.ApplyBeamFilter()

        if ak.count(nPFP) == 0:
            nPFP = ak.count(events.recoParticles.nHits, -1)
        else:
            nPFP = ak.concatenate([nPFP, ak.count(events.recoParticles.nHits, -1)])

        # apply additional selection for beam MC events
        print(f"beamMC : {events.trueParticles.pi0_MC}")
        if events.trueParticles.pi0_MC == False:
            print("apply beam MC filter")
            events = Master.BeamMCFilter(events)

        samples = []
        for i in range(len(s_l)):
            print(f"number of objects: {n_obj[i]}")
            print(f"merge?: {merge[i]}")
            samples.append(SelectSample(events, n_obj[i], merge[i], bt[i], cheat[i]))
        label = ["true", "reco", "error"]
        for k in range(len(s_l)):
            q = Master.CalculateQuantities(samples[k])
            for i in range(len(q)):
                if i == 0:
                    df = pd.concat([ak.to_pandas(q[i][j], anonymous=f"{label[i]} {names[j]}") for j in range(len(names))], axis=1)
                else:
                    df = pd.concat([df, pd.concat([ak.to_pandas(q[i][j], anonymous=f"{label[i]} {names[j]}") for j in range(len(names))], axis=1)], axis=1)
            df.to_csv(f"output_{s_l[k]}.csv", mode="a")

    if save is True: os.makedirs(outDir, exist_ok=True)
    Plots.PlotBar(nPFP[nPFP>0], xlabel="Number of particle flow objects per event")
    if save is True: Plots.Save(outDir + "n_objects")


def AnalyseSingle():
    events = Master.Data(file)
    events.ApplyBeamFilter() # apply beam filter if possible

    # apply additional selection for beam MC events
    print(f"beamMC : {events.trueParticles.pi0_MC}")
    if events.trueParticles.pi0_MC == False:
        print("apply beam MC filter")
        events = Master.BeamMCFilter(events)

    if save is True: os.makedirs(outDir, exist_ok=True)
    n = ak.count(events.recoParticles.nHits, -1)
    Plots.PlotBar(n[n>0], xlabel="Number of particle flow objects per event")
    if save is True: Plots.Save(outDir + "n_objects")

    samples = []
    for i in range(len(s_l)):
        print(f"number of objects: {n_obj[i]}")
        print(f"merge?: {merge[i]}")
        samples.append(SelectSample(events, n_obj[i], merge[i], bt[i], cheat[i]))

    t = []
    r = []
    e = []
    for sample in samples:
        q = Master.CalculateQuantities(sample)
        t.append(q[0])
        r.append(q[1])
        e.append(q[2])
    
    t = ak.Array(t)
    r = ak.Array(r)
    e = ak.Array(e)
    
    if plotsToMake in ["all", "truth"]: Plot1D(t[0:3], t_l, "truth/", s_l[0:3], t_range, t_locs) # MC truth for unmerged and merged are identical, so don't plot them
    if plotsToMake in ["all", "reco"]: Plot1D(r, r_l, "reco/", s_l, r_range, r_locs, r_xs, r_ys)
    if plotsToMake in ["all", "error"]: Plot1D(e, e_l, "error/", s_l, e_range, e_locs)
    if plotsToMake in ["all", "2D"]: Plot2D(t, e)


@Master.timer
def main():
    if isinstance(file, list):

        if all("csv" == f.split('.')[-1] for f in file):
            print("plot data")
            PlotFromCSV()
        elif all("root" == f.split('.')[-1] for f in file):
            print("analyse data")
            AnalyseMultipleFiles()
        else:
            print("bad file list, include either only root files or csv files")
    else:
        AnalyseSingle()

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    # plot labels
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading photon energy (GeV)", "True Sub leading photon energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    e_l = ["Invariant mass fractional error", "Opening angle fractional error", "Leading shower energy fractional error", "Sub leading shower energy fractional error", "$\pi^{0}$ momentum fractional error"]
    r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Sub leading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]
    # plot ranges, order is invariant mass, angle, lead energy, sub energy, pi0 momentum
    #e_range = [[-10, 10], [-10, 10], [], [], [-1, 0]]
    #e_range = [[-1, 5], [-1, 10], [-1, 1], [-1, 2], [-1, 0]] * 5
    e_range = [[]] * 5
    r_range = [[]] * 5
    r_range[0] = [0, 0.5]
    t_range = [[]] * 5
    # legend location, order is invariant mass, angle, lead energy, sub energy, pi0 momentum
    r_locs = ["upper right", "upper right", "upper left", "upper right", "upper left"]
    t_locs = ["upper left", "upper right", "upper right", "upper right", "upper right"]
    e_locs = ["upper right", "upper left", "upper right", "upper right", "upper left"]

    r_xs = ["linear"]*5
    r_ys = ["linear"]*5
    #r_ys = ["linear", "linear", "log", "log", "linear"]
    #r_xs = ["linear", "linear", "linear", "linear", "linear"]

    # n_obj = [2, -2, -2, -2]
    # bt = [True, True, True, True]
    # merge = [True, False, True, True]
    # cheat = [False, False, False, True]
    # s_l = ["2 PFP's", "unmerged", "merged", "cheated merge"]
    # figSize2D = [2, 2]

    n_obj = [-1, -1, -1]
    bt = [True] * 3
    merge = [False, True, True]
    cheat = [False, False, True]
    s_l = ["unmerged", "merged", "cheated"]
    figSize2D = [1, 3]

    # n_obj = [-1]
    # bt = [True]
    # merge = [True]
    # cheat = [True]
    # s_l = [""]
    # figSize2D = [1, 1]

    parser = argparse.ArgumentParser(description="Plot quantities to study shower reconstruction")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="pi0_0p5GeV_100K/shower_merge/", help="directory to save plots")
    parser.add_argument("-p", "--plots", dest="plotsToMake", type=str, choices=["all", "truth", "reco", "error", "2D"], default="all", help="what plots we want to make")
    #args = parser.parse_args("csvfilelist.list -b 20 -p 2D".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    if args.file.split('.')[-1] != "root":
        files = []
        with open(args.file) as filelist:
            file = filelist.read().splitlines() 
    else:
        file = args.file

    bins = args.bins
    save = args.save
    outDir = args.outDir
    plotsToMake = args.plotsToMake
    
    main()