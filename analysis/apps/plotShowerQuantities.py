#!/usr/bin/env python3
"""
Created on: 09/08/2022 14:41

Author: Shyam Bhuller

Description: Plot Shower pair quantities produced from ParticleData classes (in csv format)
"""
import argparse
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from python.analysis import Plots


class info:
    """ Basically a dicitonary... I don't know why I wrote this
    """
    def __init__(self, mass = None, angle = None, lead_energy = None, sub_energy = None, pi0_mom_mag = None) -> None:
        self.mass = mass
        self.angle = angle
        self.lead_energy = lead_energy
        self.sub_energy = sub_energy
        self.pi0_mom_mag = pi0_mom_mag
        pass
    def __getitem__(self, key : str):
        return getattr(self, key)


def CheckArguement(arg, message : str):
    """ Check if argument exists

    Args:
        arg (): argpare arguement
        message (str): error message
    """
    if not arg: raise argparse.ArgumentError(arg, message)


def Plot1D(data : dict, xlabel : str, outDir : str, name : str, args, plot_ranges = [], legend_loc = "upper right"):
    """ 1D histograms of data for each sample.

    Args:
        data (dict): data to plot
        xlabel (str): plot x label
        outDir (str): output file directory
        name (str): file name
        args (): command line arguements
        plot_ranges (list, optional): x range of plots. Defaults to [].
        legend_loc (str, optional): location of legend. Defaults to "upper right".
    """
    if len(data.keys()) == 1:
        d = data[list(data.keys())[0]]
        if plot_ranges and len(plot_ranges) == 2:
            d = d[d > plot_ranges[0]]
            d = d[d < plot_ranges[1]]
        else:
            d = d[d > -999]
        Plots.PlotHist(d, args.bins, xlabel = xlabel, x_scale = "linear", y_scale = args.log, density = args.norm, annotation = args.annotation)
    else:
        Plots.PlotHistComparison(list(data.values()), plot_ranges, args.bins, xlabel = xlabel, histtype = "step", labels = list(data.keys()), x_scale = "linear", y_scale = args.log, density = args.norm, annotation = args.annotation)
        plt.legend(loc=legend_loc)
    if args.save is True: Plots.Save(name, outDir)


def PlotQQ(data : dict, axisLabels : str, outDir : str, name : str, args):
    """ Make a Quantile Quantile plot

    Args:
        data (dict): data to plot
        xlabel (str): plot x label
        outDir (str): output file directory
        name (str): file name
        args (): command line arguements
    """
    x = data[args.control]
    x = x[x > -999] # exclude null data from plots
    print(x)
    qx = np.percentile(ak.to_numpy(x), range(100))
    Plots.Plot(qx, qx, label="perfect fit", annotation = args.annotation)

    for i in data:
        if i == args.control: continue
        y = ak.to_numpy(data[i])
        qy = np.percentile(y[y > -999], range(100))
        Plots.Plot(qx, qy, f"quantile of {axisLabels}, {args.control}", f"quantile of {axisLabels} target", label = i, newFigure = False)
    if args.save is True: Plots.Save(f"qq_{name}", outDir)


def PlotDiff(data : dict, xlabel : str, outDir : str, name : str, args, legend_loc = "upper right"):
    """ Plot the difference between two samples from the same data source:
        x - y
    Args:
        data (dict): data to plot
        xlabel (str): plot x label
        outDir (str): output file directory
        name (str): file name
        args (): command line arguements
        legend_loc (str, optional): location of legend. Defaults to "upper right".
    """
    c = list(data.keys()).index(args.control)
    l = list(data.keys())[:c] + list(data.keys())[c+1:]
    diff = []
    for i in data:
        if i == args.control: continue
        d = data[i] - data[args.control]
        d = d[(data[i] != -999) & (data[args.control] != -999)]
        diff.append(d)
    Plots.PlotHistComparison(diff, bins = args.bins, xlabel = f"difference in {xlabel} wrt {args.control}", labels = l, x_scale = "linear", y_scale = args.log, density = args.norm, annotation = args.annotation)
    plt.legend(loc = legend_loc)
    if args.save is True: Plots.Save(f"diff_{name}", outDir)


def main(args):
    #* plot styling
    plt.style.use('ggplot')
    plt.rcParams.update({'patch.linewidth': 1})
    plt.rcParams.update({'font.size': 10})

    print(f"Files: {args.files}")
    print(f"Labels: {args.labels}")
    print(f"Save: {args.save}")

    #* plot ranges
    plot_ranges = {
        "reco": info(),
        "true": info(),
        "error": info(
            [-1, 10], 
            [-1, 10], 
            [-1, 10], 
            [-1, 10], 
            [-1, 10]),
        "cheat" : info()
    }

    #* plot labels
    plot_labels = {
        "reco": info(
            "Invariant mass (MeV)",
            "Opening angle (rad)",
            "Leading shower energy (MeV)",
            "Sub leading shower energy (MeV)",
            "$\pi^{0}$ momentum (MeV)"
        ),
        "true": info(
            "True invariant mass (MeV)",
            "True opening angle (rad)",
            "True leading photon energy (MeV)",
            "True sub leading photon energy (MeV)",
            "True $\pi^{0}$ momentum (MeV)"
        ),
        "error": info(
            "Invariant mass fractional error",
            "Opening angle fractional error",
            "Leading shower energy fractional error",
            "Sub leading shower energy fractional error",
            "$\pi^{0}$ momentum fractional error"
        ),
        "cheat" : info(
            "Cheated Invariant mass (MeV)",
            "Cheated Opening angle (rad)",
            "Cheated Leading shower energy (MeV)",
            "Cheated Sub leading shower energy (MeV)",
            "Cheated $\pi^{0}$ momentum (MeV)"
        )
    }

    if args.type == "all":
        args.type = "" # if all types are plotted, regex should search for nothing

    if args.quantity == "all":
        args.quantity = ""

    for q in args.quantity:
        print(q)
        data = {}
        for t in args.type:
            data_type = t.split("/")[-1] # end of keys in the hdf5 file are data frames
            data[data_type] = {}
        for t in args.type:
            data_type = t.split("/")[-1] # end of keys in the hdf5 file are data frames
            sub_label = t.split("/")[-2]
            sub_label = "_" + sub_label if len(sub_label) > 0 else sub_label
            for file, label in zip(args.files, args.labels):
                print(file)
                print(f"finding {t + '/' + q}")
                data[data_type][label + sub_label] = ak.flatten(ak.Array(pd.read_hdf(file, t).filter(regex = q).T.values))
        print(f"{data=}")

        for k, v in data.items():
            out = args.outDir + f"{k}/" # output directory
            if args.save is True: os.makedirs(out, exist_ok = True)
            if "hist" in args.plots:
                Plot1D(v, plot_labels[k][q], out, q, args, plot_ranges[k][q])
            if "diff" in args.plots:
                PlotDiff(v, plot_labels[k][q], out, q, args)
            if "qq" in args.plots:
                PlotQQ(v, plot_labels[k][q], out, q, args)


if __name__ == "__main__":
    
    quantities = ["mass", "angle", "lead_energy", "sub_energy", "pi0_mom_mag"]
    plots = ["diff", "hist", "qq"]

    parser = argparse.ArgumentParser(description = "Plot Shower pair quantities produced from ParticleData classes (in csv format)")
    parser.add_argument(dest = "files", nargs = "+", help = "csv file/s to open.")
    # parser.add_argument("-t", "--type", dest = "type", nargs = "+", type = str, choices = ["all", "reco", "true", "error"], help = "which type of data to plot")
    parser.add_argument("-q", "--quantity", dest = "quantity", nargs = "+", type = str, choices = quantities + ["all"], help = "which quantitiy to plot")
    parser.add_argument("-p", "--plots", dest = "plots", nargs = "+", type = str, choices = plots + ["all"], help = "what plots we want to make")
    parser.add_argument("-t", "--type", dest = "type", nargs = "+", type = str, help = "which type of data to plot, is a hdf5 key")

    parser.add_argument("-c", "--control", dest = "control", type = str, default = None)

    parser.add_argument("-s", "--save", dest = "save", action = "store_true", help = "whether to save the plots")
    parser.add_argument("-d", "--directory", dest = "outDir", type = str, default = "shower_quantities/", help = "directory to save plots")
    
    parser.add_argument("-b", "--bins", dest = "bins", type = int, default = 20, help = "number of bins")
    parser.add_argument("-a", "--annotate", dest = "annotation", type = str, default = None, help = "annotation to add to plots")
    parser.add_argument("-n", "--normalize", dest = "norm", action = "store_true", help = "normalise plots to compare shape")
    parser.add_argument("-l" "--log", dest = "log", action = "store_true", help = "plot y axis on log scale")
    parser.add_argument("-L", "--labels", dest = "labels", nargs = "+", help = "custom plot labels for each sample")
    
    parser.parse_args()
    args = parser.parse_args()

    #* check and format arguements
    # CheckArguement(args.type, "specify the type of data to plot with '-t/--type'")
    CheckArguement(args.quantity, "specify the quantitiy to plot with '-q/--quantity'")
    CheckArguement(args.plots, "specify the type of plot to make '-p/--plots'")

    with pd.HDFStore(args.files[0]) as file: keys = file.keys()

    if args.type is None:
        raise KeyError(f"list of possible keys are {keys}")
    else:
        for t in args.type:
            if t not in keys:
                raise KeyError(f"key '{t}' not found in {args.files}, list of possible keys are {keys}")

    if "all" in args.quantity:
        args.quantity = quantities
    if "all" in args.plots:
        args.plots = plots

    if any(x in args.plots for x in ["qq", "diff"]):
        if len(args.files) == 1:
            raise Exception(f"{args.plots} plots require at least two samples")
        CheckArguement(args.control, "qq and diff plots require a control sample '-c/--control'")

    if args.labels is None:
        args.labels = [f.split("/")[-1].split(".")[0] for f in args.files]

    if args.control and args.control not in args.labels:
            raise ValueError(f"control group must be in labels, {args.labels}")

    if args.log is True:
        args.log = "log"
    else:
        args.log = "linear"

    main(args)
