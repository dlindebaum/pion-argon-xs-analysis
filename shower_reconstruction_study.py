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
# custom modules
import Plots
import Master
import merge_study
import matching_study

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
    for i in range(len(names)):
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
    fig.delaxes(axes.flat[-1])
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
        Plots.PlotHistComparison(data[:, i], plot_ranges[i], bins, xlabel=xlabels[i], histtype="step", labels=labels, x_scale=x_scale[i], y_scale=y_scale[i], density=True)
        plt.legend(loc=legend_loc[i])
        if save is True: Plots.Save( names[i] , outDir + subDir)


def Plot2D(true_data : np.array, error_data : np.array):
    """ Plots 2D histograms

    Args:
        true_data (np.array): true quantities
        error_data (np.array): fraction errors
    """
    # plot 2D plots of all quantities
    if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
    for i in range(len(names)):
        Plot2DRatio(i, true_data, error_data, s_l, t_l[i], e_l[i], 3, 2)
        if save is True: Plots.Save(names[i], outDir + "2D/")
        if save is True: os.makedirs(outDir + "2D/", exist_ok=True)

    # plot opening angle and invariant mass against pi0 momentum
    t_awk = ak.Array(true_data)
    e_awk = ak.Array(error_data)
    Plot2DMulti(t_awk[:, 4], e_awk[:, 0], t_l[4], e_l[0], "inv_mass_vs_mom", t_range[4], e_range[0])
    Plot2DMulti(t_awk[:, 4], e_awk[:, 1], t_l[4], e_l[1], "angle_vs_mom", t_range[4], e_range[0])


def SelectSample(events : Master.Data, nDaughters : int, merge : bool = False):
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
    filtered = events.Filter([valid], [valid]) # filter events with mask
    matched, unmatched, selection = filtered.MCMatching(applyFilters=False)
    filtered.Filter([selection],[selection], returnCopy=False) # apply the selection
    if merge is True:
        filtered = merge_study.mergeShower(filtered, matched[selection], unmatched[selection], 1, False)
    else:
        # if we don't merge showers, just get the showers matched to MC
        filtered.Filter([matched[selection]], returnCopy=False)
    return filtered

@Master.timer
def main():
    events = Master.Data(file)
    events.ApplyBeamFilter() # apply beam filter if possible
    
    if save is True: os.makedirs(outDir, exist_ok=True)
    Plots.PlotBar(ak.count(events.recoParticles.nHits, -1), xlabel="Number of particle flow objects per event")
    if save is True: Plots.Save(outDir + "n_objects")
    
    events_2_shower = SelectSample(events, 2, False)
    events_3_shower = SelectSample(events, 3, False)
    events_remaning = SelectSample(events, -3, False)
    events_3_shower_merged = SelectSample(events, 3, True)
    events_remaning_merged = SelectSample(events, -3, True)
    samples = [events_2_shower, events_3_shower, events_remaning, events_3_shower_merged, events_remaning_merged]
    
    t = []
    r = []
    e = []
    for sample in samples:
        q = Master.CalculateQuantities(sample, names)
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

if __name__ == "__main__":
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    # plot labels
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading photon energy (GeV)", "True Sub leading photon energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    e_l = ["Invariant mass fractional error", "Opening angle fractional error", "Leading shower energy fractional error", "Sub leading shower energy fractional error", "$\pi^{0}$ momentum fractional error"]
    r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Sub leading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]
    # plot ranges, order is invariant mass, angle, lead energy, sub energy, pi0 momentum
    #e_range = [[], [], [-10, 10], [-10, 10], [-10, 0]]
    e_range = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 0]] * 5
    r_range = [[]] * 5
    r_range[0] = [0, 0.5]
    t_range = [[]] * 5
    # legend location, order is invariant mass, angle, lead energy, sub energy, pi0 momentum
    r_locs = ["upper right", "upper right", "upper right", "upper right", "upper right"]
    t_locs = ["upper left", "upper right", "upper right", "upper right", "upper right"]
    e_locs = ["upper right", "upper left", "upper right", "upper right", "upper left"]

    r_ys = ["linear", "linear", "log", "log", "linear"] #? something like this?
    r_xs = ["linear", "linear", "linear", "linear", "linear"] #? something like this?

    s_l = ["2 showers", "3 showers unmerged", ">3 showers unmerged", "3 showers merged", ">3 showers merged"]

    parser = argparse.ArgumentParser(description="Plot quantities to study shower reconstruction")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="pi0_0p5GeV_100K/shower_merge/", help="directory to save plots")
    parser.add_argument("-p", "--plots", dest="plotsToMake", type=str, choices=["all", "truth", "reco", "error", "2D"], default="all", help="what plots we want to make")
    #args = parser.parse_args("ROOTFiles/pi0_0p5GeV_100K_5_7_21.root -b 20".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    outDir = args.outDir
    plotsToMake = args.plotsToMake
    main()