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

quantities = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]

def Plot1D(data : ak.Array, xlabels : list, subDir : str, labels : list, plot_ranges = [[]]*5, legend_loc = ["upper right"]*5, x_scale=["linear"]*5, y_scale=["linear"]*5, norm : bool = True, save : bool = False, outDir : str = "", bins : int = 20, annotation : str = None):
    """ 1D histograms of data for each sample

    Args:
        data (ak.Array): list of samples data to plot
        xlabels (list): x labels
        subDir (str): subdirectiory to save in
        plot_range (list, optional): range to plot. Defaults to [].
    """
    if save is True: os.makedirs(outDir + subDir, exist_ok=True)
    for i in range(len(quantities)):
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
        if save is True: Plots.Save( quantities[i] , outDir + subDir)


def PlotQQ(data : ak.Array, labels : list, control : str, axisLabels : list, quantity : str = "inv_mass", save : bool = False, outDir : str = "", subDir : str = "", annotation : str = None):
    """ Make a Quantile Quantile plot

    Args:
        data (ak.Array): samples to plot
        labels (list): plot labels
        control (str): control sample
        quantity (str, optional): quantity to plot, defaults to inv_mass
    """
    if save is True: os.makedirs(outDir + subDir, exist_ok=True)
    c = labels.index(control)
    q = quantities.index(quantity)

    x = data[c][q]
    x = x[x > -999] # exclude null data from plots
    qx = np.percentile(x, range(100))
    Plots.Plot(qx, qx, label="perfect fit", annotation=annotation)

    for i in range(len(data)):
        if i == c: continue
        y = data[i][q]
        qy = np.percentile(y[y > -999], range(100))
        Plots.Plot(qx, qy, f"quantile of {axisLabels[q]}, {control}", f"quantile of {axisLabels[q]} target", label=labels[i], newFigure=False)
    if save is True: Plots.Save(f"qq_{quantity}", outDir + subDir)


def PlotDiff(data : ak.Array, labels : list, control : str, xlabels : list, legend_loc = ["upper right"]*5, x_scale=["linear"]*5, y_scale=["linear"]*5, norm : bool = True, save : bool = False, outDir : str = "", subDir : str = "", annotation : str = None, bins : int = 20):
    if save is True: os.makedirs(outDir + subDir, exist_ok=True)
    c = labels.index(control)
    l = labels[:c] + labels[c+1:]
    for j in range(quantities):
        diff = []
        for i in range(len(data)):
            if i == c: continue
            d = data[i][j] - data[c][j]
            d = d[data[i][j] != -999]
            diff.append(d[data[c][j] != -999])
        Plots.PlotHistComparison(diff, bins = bins, xlabel=f"difference in {xlabels[j]} wrt {control}", labels=l, x_scale=x_scale[j], y_scale=y_scale[j], density=norm)
        plt.legend(loc=legend_loc[j])


def main(args):
    print(f"Files: {args.files}")
    print(f"Labels: {args.labels}")
    print(f"Save: {args.save}")

    if args.labels is not None:
        labels = args.labels
    else:
        labels = args.files

    if args.log is True:
        scale = ["log"]*5
    else:
        scale = ["linear"]*5

    if args.qq[0] is None:
        args.qq[0] = labels[0]
    else:
        if args.qq[0] not in labels:
            raise ValueError(f"control group must be in labels, {labels}")

    if args.qq[1] not in quantities:
        raise ValueError(f"qq quantity mst be in {quantities}")

    if args.diff is not None and args.diff not in labels:
        raise ValueError(f"control sample must be {labels}")

    if args.plotsToMake in ["qq", "diff"] and len(args.files) == 1:
        raise Exception(f"{args.plotsToMake} plots require at least two samples")

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

    for file in args.files:
        print(file)
        data = pd.read_csv(file)
        data = np.transpose(data.values[:, 1:])
        t.append(data[0:5, :])
        r.append(data[5:10, :])
        e.append(data[10:15, :])

    t = ak.Array(t)
    r = ak.Array(r)
    e = ak.Array(e)
    
    if args.plotsToMake in ["all", "hist"]:
        if args.dataType in ["all", "truth"]: Plot1D(t, t_l, "truth/", labels, t_range, save = args.save, outDir=args.outDir, norm=args.norm, y_scale=scale, bins=args.bins, annotation=args.annotation)
        if args.dataType in ["all", "reco"]: Plot1D(r, r_l, "reco/", labels, r_range, save = args.save, outDir=args.outDir, norm=args.norm, y_scale=scale, bins=args.bins, annotation=args.annotation)
        if args.dataType in ["all", "error"]: Plot1D(e, e_l, "error/", labels, e_range, save = args.save, outDir=args.outDir, norm=args.norm, y_scale=scale, bins=args.bins, annotation=args.annotation)
    if args.plotsToMake == "qq" or (args.plotsToMake == "all" and len(args.files > 1)):
        if args.dataType in ["all", "truth"]: PlotQQ(t, labels, args.qq[0], t_l, args.qq[1], args.save, args.outDir, "truth/", args.annotation)
        if args.dataType in ["all", "reco"]: PlotQQ(r, labels, args.qq[0], r_l, args.qq[1], args.save, args.outDir, "reco/", args.annotation)
        if args.dataType in ["all", "error"]: PlotQQ(e, labels, args.qq[0], e_l, args.qq[1], args.save, args.outDir, "error/", args.annotation)
    if args.plotsToMake == "diff" or (args.plotsToMake == "all" and len(args.files > 1)):
        if args.dataType in ["all", "truth"]: PlotDiff(t, labels, args.diff, t_l, save = args.save, outDir=args.outDir, subDir="truth/", norm=args.norm, y_scale=scale, bins=args.bins, annotation=args.annotation)
        if args.dataType in ["all", "reco"]:  PlotDiff(r, labels, args.diff, r_l, save = args.save, outDir=args.outDir, subDir="reco/", norm=args.norm, y_scale=scale, bins=args.bins, annotation=args.annotation)
        if args.dataType in ["all", "error"]: PlotDiff(e, labels, args.diff, e_l, save = args.save, outDir=args.outDir, subDir="error/", norm=args.norm, y_scale=scale, bins=args.bins, annotation=args.annotation)


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    parser = argparse.ArgumentParser(description="Plot Shower pair quantities produced from ParticleData classes (in csv format)")
    parser.add_argument(dest="files", nargs="+", help="csv file/s to open.")
    parser.add_argument("-p", "--plots", dest="plotsToMake", type=str, choices=["all", "truth", "reco", "error", "qq"], help="what plots we want to make")
    parser.add_argument("-t", "--type", dest="dataType", type=str, choices=["all", "reco", "truth", "error"], default="all", help="type of data to plot")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="shower_quantities/", help="directory to save plots")
    parser.add_argument("-b", "--bins", dest="bins", type=int, default=20, help="number of bins")
    parser.add_argument("-a", "--annotate", dest="annotation", type=str, default=None, help="annotation to add to plots")
    parser.add_argument("-n", "--normalize", dest="norm", action="store_true", help="normalise plots to compare shape")
    parser.add_argument("-l" "--log", dest="log", action="store_true", help="plot y axis on log scale")
    parser.add_argument("-L", "--labels", dest="labels", nargs="+", help="custom plot labels for each sample")
    parser.add_argument("--qq", dest="qq", type=str, nargs=2, default=[None, "inv_mass"], help="qq plot settings (control group, quantity)")
    parser.add_argument("--diff", dest="diff", type=str, default=None, help="plot the difference of all samples and the chosen control")
    main(parser.parse_args())
