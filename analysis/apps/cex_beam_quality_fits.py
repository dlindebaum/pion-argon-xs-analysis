#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:36

Author: Shyam Bhuller

Description: Computes fits for the beam quality selection and stores them in a json file to be used with the full selection.
"""
import argparse
import os

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from rich import print
from python.analysis import Master, BeamParticleSelection, vector, Plots, cross_section, Fitting
from python.analysis.shower_merging import SetPlotStyle


def Fit_Vector(v : ak.Record, bins : int) -> tuple[dict, dict, dict, dict]:
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
        data = v[i]
        y, bins_edges = np.histogram(np.array(data[~np.isnan(data)]), bins = bins, range = sorted([np.nanpercentile(data, 10), np.nanpercentile(data, 90)])) # fit only to  data within the 10th and 90th percentile of data to exclude large tails in the distriubtion.
        yerr = np.sqrt(y) # Poisson error

        popt, perr = Fitting.Fit(cross_section.bin_centers(bins_edges), y, yerr, Fitting.gaussian)

        mu[i] = popt[1]
        sigma[i] = abs(popt[2])
        mu_err[i] = abs(perr[1])
        sigma_err[i] = abs(perr[2])

    return mu, sigma, mu_err, sigma_err


def plot(value : ak.Array, x_label : str, mu : float, sigma : float, color : str = None, label : str = None, range : list = None):
    """ Plot data plus the gaussian fit made on the data.

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

    y_pred = Fitting.gaussian.func(x, max(y), mu, sigma)

    x_interp = np.linspace(min(x), max(x), 200)
    y_interp = Fitting.gaussian.func(x_interp, max(y), mu, sigma)

    yerr = (y * (1 - (y / np.sum(y))))**0.5

    chisqr = np.sum(((y - y_pred)/ yerr)**2)
    ndf = len(y) - 2

    Plots.Plot(x, y/np.sum(y), yerr = yerr/np.sum(y), color = color, marker = "x", linestyle = "", capsize = 3, newFigure = False)
    Plots.Plot(x_interp, y_interp / np.sum(y), xlabel = x_label, ylabel = "Counts (area normalised)", color = color, linestyle = "-", label = label + " $\chi^{2}/ndf$ : " + f"{chisqr/ndf:.2f}", newFigure = False)
    plt.ylim(0)


def plot_range(mu : float, sigma : float, tolerance : float = 5) -> list:
    """ Compute range of plot based on properties of the Gaussian.

    Args:
        mu (float): mean
        sigma (float): rms
        tolerance (float, optional): how many standard deviations to extend the range by. Defaults to 5.

    Returns:
        list: plot range
    """
    return sorted([mu - tolerance * sigma, mu + tolerance * sigma])


def MakePlots(mc_events : Master.Data, mc_fits : dict, data_events : Master.Data, data_fits : dict, out : str, truncate : float):
    """ Make plots showing the fits for MC and or Data.

    Args:
        mc_events (Master.Data): mc
        mc_fits (dict): fits made on mc
        data_events (Master.Data): data
        data_fits (dict): fits made on data
        out (str): output directory
    """
    SetPlotStyle()

    data_start_pos, _ = BeamParticleSelection.GetPos(data_events, truncate)
    mc_start_pos, _ = BeamParticleSelection.GetPos(mc_events, truncate)

    with Plots.PlotBook(out + "beam_quality_fits.pdf") as pdf:
        for i in ["x", "y", "z"]:
            plt.figure()
            mc_ranges = [] if mc_fits is None else plot_range(mc_fits[f"mu_{i}"], mc_fits[f"sigma_{i}"])
            data_ranges = [] if data_fits is None else plot_range(data_fits[f"mu_{i}"], data_fits[f"sigma_{i}"])

            plot_ranges = mc_ranges + data_ranges
            if mc_events is not None: plot(mc_start_pos[i], f"Beam start position {i} (cm)", mc_fits[f"mu_{i}"], mc_fits[f"sigma_{i}"], "C0", "MC", range = [min(plot_ranges), max(plot_ranges)])
            if data_events is not None: plot(data_start_pos[i], f"Beam start position {i} (cm)", data_fits[f"mu_{i}"], data_fits[f"sigma_{i}"], "C1", "Data", range = [min(plot_ranges), max(plot_ranges)])
            pdf.Save()
    return


def run(file : str, data : bool, ntuple_type : Master.Ntuple_Type, out : str, tag : str, args : cross_section.argparse.Namespace):
    events = Master.Data(file, nTuple_type = ntuple_type, target_momentum = args.pmom)

    #? should this be made configurable i.e. pass config and apply all selections before the beam quality cuts if it is in the list?

    func_args = "mc_arguments" if data is False else "data_arguments"

    if ("DxyCut" in args.beam_selection["selections"]) or ("DzCut" in args.beam_selection["selections"]) or ("CosThetaCut" in args.beam_selection["selections"]):
        for s in args.beam_selection["selections"]:
            if s in ["DxyCut", "DzCut", "CosThetaCut"]:
                break # only apply cuts before beam quality
            else:
                print(f"{s=}")
                mask = args.beam_selection["selections"][s](events, **args.beam_selection[func_args][s])
                events.Filter([mask], [mask])
    else:
        # do default selections prior to beam quality
        mask = BeamParticleSelection.PiBeamSelection(events, data)
        events.Filter([mask], [mask])

        mask = BeamParticleSelection.PandoraTagCut(events)
        events.Filter([mask], [mask])

        mask = BeamParticleSelection.CaloSizeCut(events)
        events.Filter([mask], [mask])

        mask = BeamParticleSelection.HasFinalStatePFOsCut(events)
        events.Filter([mask], [mask])

    #* fit gaussians to the start positions
    start_pos, end_pos = BeamParticleSelection.GetPos(events, args.beam_quality_truncate)
    mu, sigma, mu_err, sigma_err = Fit_Vector(start_pos, 100)

    #* compute the mean of beam direction components
    beam_dir = vector.normalize(vector.sub(end_pos, start_pos))
    mu_dir = {i : ak.mean(ak.nan_to_num(beam_dir[i])) for i in ["x", "y", "z"]}
    mu_dir_err = {i : ak.std(ak.nan_to_num(beam_dir[i]))/np.sqrt(ak.count(beam_dir[i])) for i in ["x", "y", "z"]}

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
        "truncate"     : args.beam_quality_truncate
    }
    print(fit_values)

    #* write to json file
    os.makedirs(args.out, exist_ok = True)
    name = out + tag + "_beam_quality_fit_values.json"
    Master.SaveConfiguration(fit_values, name)
    print(f"fit values written to {name}")

    return events, fit_values

@Master.timer
def main(args):
    mc, fit_values_mc = None, None
    data, fit_values_data = None, None

    os.makedirs(args.out + "beam_quality/", exist_ok = True)

    if args.mc_file is not None:
        mc, fit_values_mc = run(args.mc_file, False, args.ntuple_type, args.out + "beam_quality/", "mc", args)
    if args.data_file is not None:
        data, fit_values_data = run(args.data_file, True, args.ntuple_type, args.out + "beam_quality/", "data", args)

    MakePlots(mc, fit_values_mc, data, fit_values_data, args.out + "beam_quality/", args.beam_quality_truncate)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Computes Guassian fit paramters needed for the beam quality cuts in the beam particle selection.", formatter_class = argparse.RawDescriptionHelpFormatter)

    # cross_section.ApplicationArguments.Ntuples(parser, data = True)
    cross_section.ApplicationArguments.Config(parser, True)    
    cross_section.ApplicationArguments.Output(parser)

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)
    print(vars(args))
    main(args)