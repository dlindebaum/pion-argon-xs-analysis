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
    mu = {}
    mu_err = {}
    sigma = {}
    sigma_err = {}
    for i in ["x", "y", "z"]:
        popt, pcov = cross_section.Fit_Gaussian(v[i], bins = bins)

        mu[i] = popt[1]
        sigma[i] = abs(popt[2])
        mu_err[i] = abs(pcov[1][1])
        sigma_err[i] = abs(pcov[2][2])

    return mu, sigma, mu_err, sigma_err


def plot(value : ak.Array, x_label : str, mu : float, sigma : float, color : str = None, label : str = None, range : list = None):
    """ Plot data pluse the gaussian fit made on the data.

    Args:
        value (ak.Array): date to plot
        x_label (str): x label
        range (list): range to plot
        mu (float): mean of fit
        sigma (float): rms of fit
        name (str): plot name
    """
    y, edges = np.histogram(np.array(value), bins = 50, range = [mu - 5 * sigma, mu + 5 * sigma] if range is None else range)
    x = (edges[1:] + edges[:-1]) / 2
    y_pred = cross_section.Gaussian(x, max(y), mu, sigma)

    x_interp = np.linspace(min(x), max(x), 200)
    y_interp = cross_section.Gaussian(x_interp, max(y), mu, sigma)

    yerr = (y * (1 - (y / np.sum(y))))**0.5

    chisqr = np.sum(((y - y_pred)/ yerr)**2)
    ndf = len(y) - 2

    plt.errorbar(x, y / np.sum(y), yerr / np.sum(y), color = color, fmt = "x", capsize = 3, linestyle = "")
    Plots.Plot(x_interp, y_interp / np.sum(y), xlabel = x_label, ylabel = "Area normalised", color = color, linestyle = "-", label = label + " $\chi^{2}/ndf$ : " + f"{chisqr/ndf:.2f}", newFigure = False)
    plt.ylim(0)
    # Plots.Save(name)

def plot_range(mu : float, sigma : float, tolerance : float = 5):
    return sorted([mu - tolerance * sigma, mu + tolerance * sigma])


def MakePlots(mc_events : Master.Data, mc_fits : dict, data_events : Master.Data, data_fits : dict, out : str):
    SetPlotStyle()

    for i in ["x", "y", "z"]:
        mc_ranges = [] if mc_fits is None else plot_range(mc_fits[f"mu_{i}"], mc_fits[f"sigma_{i}"])
        data_ranges = [] if data_fits is None else plot_range(data_fits[f"mu_{i}"], data_fits[f"sigma_{i}"])

        plot_ranges = mc_ranges + data_ranges
        if mc_events is not None: plot(mc_events.recoParticles.beam_startPos[i], f"Beam start position {i} (cm)", mc_fits[f"mu_{i}"], mc_fits[f"sigma_{i}"], "C0", "MC", range = [min(plot_ranges), max(plot_ranges)])
        if data_events is not None: plot(data_events.recoParticles.beam_startPos[i], f"Beam start position {i} (cm)", data_fits[f"mu_{i}"], data_fits[f"sigma_{i}"], "C1", "Data", range = [min(plot_ranges), max(plot_ranges)])
        Plots.Save(out + i)
    return


def run(file : str, data : bool, ntuple_type : Master.Ntuple_Type, out : str):
    events = Master.Data(file, nTuple_type = ntuple_type)


    mask = BeamParticleSelection.PiBeamSelection(events, data)
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.PandoraTagCut(events)
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.CaloSizeCut(events)
    events.Filter([mask], [mask])
    #* fit gaussians to the start positions
    mu, sigma, mu_err, sigma_err = Fit_Vector(events.recoParticles.beam_startPos, 100)

    #* compute the mean of beam direction components
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))
    mu_dir = {i : ak.mean(beam_dir[i]) for i in ["x", "y", "z"]}
    mu_dir_err = {i : ak.std(beam_dir[i])/np.sqrt(ak.count(beam_dir[i])) for i in ["x", "y", "z"]}

    #* convert to dictionary undestood by the BeamQualityCut function
    fit_values = {
        "mu_x"         : mu["x"],
        "mu_y"         : mu["y"],
        "mu_z"         : mu["z"],
        "sigma_x"      : sigma["x"],
        "sigma_y"      : sigma["y"],
        "sigma_z"      : sigma["z"],
        "mu_dir_x"     : mu_dir["x"],
        "mu_dir_y"     : mu_dir["y"],
        "mu_dir_z"     : mu_dir["z"],
        "mu_err_x"     : mu_err["x"],
        "mu_err_y"     : mu_err["y"],
        "mu_err_z"     : mu_err["z"],
        "sigma_err_x"  : sigma_err["x"],
        "sigma_err_y"  : sigma_err["y"],
        "sigma_err_z"  : sigma_err["z"],
        "mu_dir_err_x" : mu_dir_err["x"],
        "mu_dir_err_y" : mu_dir_err["y"],
        "mu_dir_err_z" : mu_dir_err["z"],
    }
    print(fit_values)

    #* write to json file
    os.makedirs(args.out, exist_ok = True)
    name = out + file.split("/")[-1].split(".")[0] + "_fit_values.json"
    with open(name, "w") as f:
        json.dump(fit_values, f)
    print(f"fit values written to {name}")

    return events, fit_values

@Master.timer
def main(args):
    mc, fit_values_mc = None, None
    data, fit_values_data = None, None
    if args.mc_file is not None:
        mc, fit_values_mc = run(args.mc_file[0], False, args.ntuple_type, args.out)
    if args.data_file is not None:
        data, fit_values_data = run(args.data_file[0], True, args.ntuple_type, args.out)

    MakePlots(mc, fit_values_mc, data, fit_values_data, args.out)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Ntuples(parser, data = True)
    
    cross_section.ApplicationArguments.Output(parser)

    args = parser.parse_args()

    cross_section.ApplicationArguments.ResolveArgs(args)

    print(vars(args))
    main(args)