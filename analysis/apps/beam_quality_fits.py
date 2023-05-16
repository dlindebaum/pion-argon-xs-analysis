#!/usr/bin/env python3
import argparse
import json
import os

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from rich import print
from python.analysis import Master, BeamParticleSelection, vector, Plots
from python.analysis.shower_merging import SetPlotStyle
from scipy.optimize import curve_fit

def gaussian(x : np.array, a : float, x0 : float, sigma : float) -> np.array:
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fit_gaussian(data : ak.Array, bins : int, range : list = None):
    if range is None:
        range = [min(data), max(data)]
    y, bins_edges = np.histogram(np.array(data), bins = bins, range = sorted([np.percentile(data, 10), np.percentile(data, 90)]))
    bin_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
    return curve_fit(gaussian, bin_centers, y, p0 = (0, np.median(data), np.std(data)))


def fit_vector(v : ak.Record, bins : int):
    a = {}
    mu = {}
    sigma = {}
    for i in ["x", "y", "z"]:
        popt, _ = fit_gaussian(v[i], bins = bins)
        a[i] = popt[0]
        mu[i] = popt[1]
        sigma[i] =popt[2]

    print(mu)
    print(sigma)
    return a, mu, sigma


def plot(value, x_label, range, mu, sigma, name):
    heights, edges = Plots.PlotHist(value, xlabel = x_label, bins = 50, range = [mu - range, mu + range])
    x = (edges[1:] + edges[:-1]) / 2
    y = gaussian(x, max(heights), mu, sigma)

    x_interp = np.linspace(min(x), max(x), 200)
    y_interp = gaussian(x_interp, max(heights), mu, sigma)

    mse = np.sqrt(np.mean((heights - y)**2))
    plt.errorbar(x, y, mse, fmt = "x",color = "black") # use mse for errors for now, if the fit is poor, then perhaps 1 sigma deviations in the fit or residual
    plt.errorbar(x_interp, y_interp, color = "black")
    plt.ylim(0)
    Plots.Save(name)
    # y_min = gaussian(x, max(heights), fit_values["mu_x"][0] - fit_values["mu_x"][1]**0.5, fit_values["sigma_x"][0] - fit_values["sigma_x"][1]**0.5)
    # y_max = gaussian(x, max(heights), fit_values["mu_x"][0] + fit_values["mu_x"][1]**0.5, fit_values["sigma_x"][0] + fit_values["sigma_x"][1]**0.5)
    # err = np.sqrt(heights)
    # Plots.Plot(x, y, newFigure = False, marker = "x", xlabel = "x (cm)")
    # plt.errorbar(x, y, np.array(list(zip(3 * abs(y - y_min), 3 * abs(y-y_max)))).T, fmt = "x")


def MakePlots(events : Master.Data, fit_values : dict):
    SetPlotStyle()
    plot(events.recoParticles.beam_startPos.x, "x (cm)", 25, fit_values["mu_x"], fit_values["sigma_x"], "x")
    plot(events.recoParticles.beam_startPos.y, "y (cm)", 25, fit_values["mu_y"], fit_values["sigma_y"], "y")
    plot(events.recoParticles.beam_startPos.z, "z (cm)", 1.5, fit_values["mu_z"], fit_values["sigma_z"], "z")

    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))
    plot(beam_dir.x, "x direction", 1, fit_values["mu_dir_x"], fit_values["sigma_dir_x"], "dir_x")
    plot(beam_dir.y, "y direction", 1, fit_values["mu_dir_y"], fit_values["sigma_dir_y"], "dir_y")
    plot(beam_dir.z, "z direction", 1, fit_values["mu_dir_z"], fit_values["sigma_dir_z"], "dir_z")
    return

def main(args):
    
    events = Master.Data(args.file, nTuple_type = args.ntuple_type)

    #* apply the following cuts before fitting (following the order in BeamParticleSelection)
    mask = BeamParticleSelection.CaloSizeCut(events)
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.PiBeamSelection(events)
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.PandoraTagCut(events)
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.MichelScoreCut(events)
    events.Filter([mask], [mask])

    #* fit gaussians to the start positions
    a, mu, sigma = fit_vector(events.recoParticles.beam_startPos, 100)

    #* fit gaussians to beam directions
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))
    a_dir, mu_dir, sigma_dir = fit_vector(beam_dir, 50)

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

    MakePlots(events, fit_values)

    print(f"fit values written to {name}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", help = "NTuple file to study.")
    parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {Master.Ntuple_Type._member_map_}.", required = True)

    parser.add_argument("-o", "--out", dest = "out", type = str, default = "./", help = "directory to save plots")

    args = parser.parse_args()

    if args.out[-1] != "/": args.out += "/"

    print(vars(args))
    main(args)