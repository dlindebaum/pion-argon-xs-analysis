"""
Created on: 09/08/2022 14:41

Author: Shyam Bhuller

Description: Plot Shower pair quantities produced from ParticleData classes (in csv format)
"""

import os
import argparse
import pandas as pd
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import Plots


def Plot1D(data : ak.Array, xlabels : list, subDir : str, labels : list, plot_ranges = [[]]*5, legend_loc = ["upper right"]*5, x_scale=["linear"]*5, y_scale=["linear"]*5, norm : bool = True, save : bool = False, outDir : str = "", bins : int = 20, annotation : str = None):
    """ 1D histograms of data for each sample

    Args:
        data (ak.Array): list of samples data to plot
        xlabels (list): x labels
        subDir (str): subdirectiory to save in
        plot_range (list, optional): range to plot. Defaults to [].
    """
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    if save is True: os.makedirs(outDir + subDir, exist_ok=True)
    for i in range(len(names)):
        if len(labels) == 1:
            d = data[0][i]
            if len(plot_ranges[i]) == 2:
                d = d[d > plot_ranges[i][0]]
                d = d[d < plot_ranges[i][1]]
            else:
                d = d[d > -999]
            Plots.PlotHist(d, bins, xlabel=xlabels[i], x_scale=x_scale[i], y_scale=y_scale[i], density=True, annotation=annotation)
        else:
            Plots.PlotHistComparison(data[:, i], plot_ranges[i], bins, xlabel=xlabels[i], histtype="step", labels=labels, x_scale=x_scale[i], y_scale=y_scale[i], density=norm, annotation=annotation)
            plt.legend(loc=legend_loc[i])
        if save is True: Plots.Save( names[i] , outDir + subDir)


def main(args):
    #* plot labels
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading photon energy (GeV)", "True Sub leading photon energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    e_l = ["Invariant mass fractional error", "Opening angle fractional error", "Leading shower energy fractional error", "Sub leading shower energy fractional error", "$\pi^{0}$ momentum fractional error"]
    r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Sub leading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]

    #* plot ranges
    e_range = [[-1, 10]] * 5
    #e_range = [[]] * 5
    r_range = [[]] * 5
    r_range[0] = [0, 0.5]
    t_range = [[]] * 5

    #* data to plot
    t = []
    r = []
    e = []

    print(args.files)
    print(args.labels)
    print(args.save)

    for file in args.files:
        print(file)
        data = pd.read_csv(file)
        data = np.transpose(data.values[:, 1:])
        t.append(data[0:5, :])
        r.append(data[5:10, :])
        e.append(data[10:15, :])

    if args.labels is not None:
        labels = args.labels
    else:
        labels = args.files

    if args.log is True:
        scale = ["log"]*5
    else:
        scale = ["linear"]*5
    
    if args.plotsToMake in ["all", "truth"]: Plot1D(ak.Array(t), t_l, "truth/", labels, t_range, save = args.save, outDir=args.outDir, norm=args.norm, y_scale=scale, bins=args.bins, annotation=args.annotation)
    if args.plotsToMake in ["all", "reco"]: Plot1D(ak.Array(r), r_l, "reco/", labels, r_range, save = args.save, outDir=args.outDir, norm=args.norm, y_scale=scale, bins=args.bins, annotation=args.annotation)
    if args.plotsToMake in ["all", "error"]: Plot1D(ak.Array(e), e_l, "error/", labels, e_range, save = args.save, outDir=args.outDir, norm=args.norm, y_scale=scale, bins=args.bins, annotation=args.annotation)


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    parser = argparse.ArgumentParser(description="Plot Shower pair quantities produced from ParticleData classes (in csv format)")
    parser.add_argument(dest="files", nargs="+", help="csv file/s to open.")
    parser.add_argument("-p", "--plots", dest="plotsToMake", type=str, choices=["all", "truth", "reco", "error"], help="what plots we want to make")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="shower_quantities/", help="directory to save plots")
    parser.add_argument("-b", "--bins", dest="bins", type=int, default=20, help="number of bins")
    parser.add_argument("-a", "--annotate", dest="annotation", type=str, default=None, help="annotation to add to plots")
    parser.add_argument("-n", "--normalize", dest="norm", action="store_true", help="normalise plots to compare shape")
    parser.add_argument("-l" "--log", dest="log", action="store_true", help="plot y axis on log scale")
    parser.add_argument("-L", "--labels", dest="labels", nargs="+", help="custom plot labels for each sample")
    main(parser.parse_args())
