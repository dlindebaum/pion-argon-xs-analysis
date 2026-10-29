"""Beam quality analysis helpers."""
import argparse
import os

import awkward as ak
import numpy as np

from python.analysis import common, fitting, io, plotting


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
        data = np.array(v[i][~np.isnan(v[i])].to_list(), dtype = float)

        y, bins_edges = np.histogram(data, bins = bins, range = sorted([np.nanpercentile(data, 10), np.nanpercentile(data, 90)])) # fit only to data within the 10th and 90th percentile of data to exclude large tails in the distriubtion.
        yerr = np.sqrt(y) # Poisson error

        popt, perr = fitting,fit(common.bin_centers(bins_edges), y, yerr, fitting.gaussian)

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

    y_pred = fitting.gaussian.func(x, max(y), mu, sigma)

    x_interp = np.linspace(min(x), max(x), 200)
    y_interp = fitting.gaussian.func(x_interp, max(y), mu, sigma)

    yerr = (y * (1 - (y / np.sum(y))))**0.5

    chisqr = np.sum(((y - y_pred)/ yerr)**2)
    ndf = len(y) - 2

    plotting.plots.Plot(x, y/np.sum(y), yerr = yerr/np.sum(y), color = color, marker = "x", linestyle = "", capsize = 3, newFigure = False)
    plotting.plots.Plot(x_interp, y_interp / np.sum(y), xlabel = x_label, ylabel = "Counts (area normalised)", color = color, linestyle = "-", label = label + " $\\chi^{2}/ndf$ : " + f"{chisqr/ndf:.2f}", newFigure = False)
    plotting.plots.plt.ylim(0)


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


def MakePlots(output_mc : dict[ak.Array], mc_fits : dict, output_data : dict[ak.Array], data_fits : dict, out : str, truncate : float):
    """ Make plots showing the fits for MC and or Data.

    Args:
        output_mc (dict[ak.Array]): mc
        mc_fits (dict): fits made on mc
        output_data (dict[ak.Array]): data
        data_fits (dict): fits made on data
        out (str): output directory
    """
    plotting.styles.PlotStyler.SetPlotStyle()

    label_name = "Beam" if truncate is None else "Truncated beam"

    with plotting.PlotBook(out + "beam_quality_fits.pdf") as pdf:
        for i in ["x", "y", "z"]:
            plotting.plots.plt.figure()
            mc_ranges = [] if mc_fits is None else plot_range(mc_fits[f"mu_{i}"], mc_fits[f"sigma_{i}"])
            data_ranges = [] if data_fits is None else plot_range(data_fits[f"mu_{i}"], data_fits[f"sigma_{i}"])

            plot_ranges = mc_ranges + data_ranges
            if output_mc is not None: plot(output_mc["start_pos"][i], f"{label_name} start position ${i}$ (cm)", mc_fits[f"mu_{i}"], mc_fits[f"sigma_{i}"], "C0", "MC", range = [min(plot_ranges), max(plot_ranges)])
            if output_data is not None: plot(output_data["start_pos"][i], f"{label_name} start position ${i}$ (cm)", data_fits[f"mu_{i}"], data_fits[f"sigma_{i}"], "C1", "Data", range = [min(plot_ranges), max(plot_ranges)])
            pdf.Save()
    return


def Fits(args : argparse.Namespace, output : dict, out : str, sample_name : str):
    start_pos = output["start_pos"]
    beam_dir = output["beam_dir"]
    mu, sigma, mu_err, sigma_err = Fit_Vector(start_pos, 100)

    mu_dir = {i : ak.mean(ak.nan_to_num(beam_dir[i])) for i in ["x", "y", "z"]}
    mu_dir_err = {i : ak.std(ak.nan_to_num(beam_dir[i]))/np.sqrt(ak.count(beam_dir[i])) for i in ["x", "y", "z"]}

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
    name = out + sample_name + "_beam_quality_fit_values.json"
    io.write_json(fit_values, name)
    print(f"fit values written to {name}")

    return fit_values
