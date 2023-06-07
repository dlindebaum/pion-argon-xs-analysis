#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:36

Author: Shyam Bhuller

Description: Computes fits for the beam quality selection and stores them in a json file to be used with the full selection.
"""
import argparse
import json
import os

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from rich import print
from python.analysis import Master, BeamParticleSelection, vector, Plots, cross_section
from python.analysis.shower_merging import SetPlotStyle


def Fit_Vector(v : ak.Record, bins : int) -> tuple:
    """ Gaussian fit to each component in the vector.

    Args:
        v (ak.Record): vector
        bins (int): number of bins

    Returns:
        tuple: fit paramaters for each component.
    """
    a = {}
    mu = {}
    sigma = {}
    for i in ["x", "y", "z"]:
        popt, _ = cross_section.Fit_Gaussian(v[i], bins = bins)
        a[i] = popt[0]
        mu[i] = popt[1]
        sigma[i] =popt[2]

    print(mu)
    print(sigma)
    return a, mu, sigma


def plot(value : ak.Array, x_label : str, range : list, mu : float, sigma : float, name : str):
    """ Plot data pluse the gaussian fit made on the data.

    Args:
        value (ak.Array): date to plot
        x_label (str): x label
        range (list): range to plot
        mu (float): mean of fit
        sigma (float): rms of fit
        name (str): plot name
    """
    heights, edges = Plots.PlotHist(value, xlabel = x_label, bins = 50, range = [mu - range, mu + range])
    x = (edges[1:] + edges[:-1]) / 2
    y = cross_section.Gaussian(x, max(heights), mu, sigma)

    x_interp = np.linspace(min(x), max(x), 200)
    y_interp = cross_section.Gaussian(x_interp, max(heights), mu, sigma)

    mse = np.sqrt(np.mean((heights - y)**2))
    plt.errorbar(x, y, mse, fmt = "x",color = "black") # use mse for errors for now, if the fit is poor, then perhaps 1 sigma deviations in the fit or residual
    plt.errorbar(x_interp, y_interp, color = "black")
    plt.ylim(0)
    Plots.Save(name)


def MakePlots(events : Master.Data, fit_values : dict, out : str):
    """ make plots to see how well the gaussian fit performs.

    Args:
        events (Master.Data): data to look at
        fit_values (dict): fit values
    """
    SetPlotStyle()
    plot(events.recoParticles.beam_startPos.x, "x (cm)", 25, fit_values["mu_x"], fit_values["sigma_x"], args.out + "x")
    plot(events.recoParticles.beam_startPos.y, "y (cm)", 25, fit_values["mu_y"], fit_values["sigma_y"], args.out + "y")
    plot(events.recoParticles.beam_startPos.z, "z (cm)", 10, fit_values["mu_z"], fit_values["sigma_z"],args.out + "z")

    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))
    plot(beam_dir.x, "x direction", 1, fit_values["mu_dir_x"], fit_values["sigma_dir_x"], args.out + "dir_x")
    plot(beam_dir.y, "y direction", 1, fit_values["mu_dir_y"], fit_values["sigma_dir_y"], args.out + "dir_y")
    plot(beam_dir.z, "z direction", 1, fit_values["mu_dir_z"], fit_values["sigma_dir_z"], args.out + "dir_z")
    return

def main(args):
    events = Master.Data(args.file, nTuple_type = args.ntuple_type)

    #* apply the following cuts before fitting (following the order in BeamParticleSelection)
    #? allow the option to to the fit without any cuts?
    mask = BeamParticleSelection.CaloSizeCut(events)
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.PiBeamSelection(events, args.sample_type == "data")
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.PandoraTagCut(events)
    events.Filter([mask], [mask])

    #* fit gaussians to the start positions
    a, mu, sigma = Fit_Vector(events.recoParticles.beam_startPos, 100)

    #* fit gaussians to beam directions
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))
    a_dir, mu_dir, sigma_dir = Fit_Vector(beam_dir, 50)

    #* convert to dictionary undestood by the BeamQualityCut function
    fit_values = {
        "mu_x"        : mu["x"],
        "mu_y"        : mu["y"],
        "mu_z"        : mu["z"],
        "sigma_x"     : sigma["x"],
        "sigma_y"     : sigma["y"],
        "sigma_z"     : sigma["z"],
        "mu_dir_x"    : mu_dir["x"],
        "mu_dir_y"    : mu_dir["y"],
        "mu_dir_z"    : mu_dir["z"],
        "sigma_dir_x" : sigma_dir["x"],
        "sigma_dir_y" : sigma_dir["y"],
        "sigma_dir_z" : sigma_dir["z"],
        "a_x"         : a["x"],
        "a_y"         : a["y"],
        "a_z"         : a["z"],
        "a_dir_x"         : a_dir["x"],
        "a_dir_y"         : a_dir["y"],
        "a_dir_z"         : a_dir["z"]
    }
    print(fit_values)

    #* write to json file
    os.makedirs(args.out, exist_ok = True)
    name = args.out + args.file.split("/")[-1].split(".")[0] + "_fit_values.json"
    with open(name, "w") as f:
        json.dump(fit_values, f)

    MakePlots(events, fit_values, args.out)

    print(f"fit values written to {name}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", help = "NTuple file to study.")
    parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {[m.value for m in Master.Ntuple_Type]}.", required = True)
    parser.add_argument("-S", "--sample-type", dest = "sample_type", type = str, choices = ["mc", "data"], help = f"type of sample I am looking at.", required = True)

    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save plots")

    args = parser.parse_args()

    if args.out is None:
        args.out = args.file.split("/")[-1].split(".")[0] + "/"
    if args.out[-1] != "/": args.out += "/"

    print(vars(args))
    main(args)