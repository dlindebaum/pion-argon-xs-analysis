#!/usr/bin/env python3
"""
Created on: 13/06/2023 11:19

Author: Shyam Bhuller

Description: Produces plots and fits for the beam scraper study, producing thresholds which dictate what events are beam scrapers.
"""
import argparse
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

from particle import Particle
from rich import print

from python.analysis import (
    Master, cross_section, Plots, Fitting, Processing, EnergyTools)


def run(i : int, file : str, n_events : int, start : int, selected_events, args : dict):
    mc = Master.Data(file, nTuple_type = args["nTuple_type"], target_momentum = args["pmom"])
    for s in args["beam_selection"]["selections"]:
        if s == "BeamScraperCut": break
        mask = args["beam_selection"]["selections"][s](mc, **args["beam_selection"]["mc_arguments"][s])
        mc.Filter([mask], [mask])

    beam_inst_KE = EnergyTools.KE(mc.recoParticles.beam_inst_P, Particle.from_pdgid(211).mass) # get kinetic energy from beam instrumentation
    true_ffKE = mc.trueParticles.beam_KE_front_face

    beam_inst_x = mc.recoParticles.beam_inst_pos.x
    beam_inst_y = mc.recoParticles.beam_inst_pos.y

    delta_KE_upstream = beam_inst_KE - true_ffKE

    return {"beam_inst_KE" : beam_inst_KE, "true_ffKE" : true_ffKE, "beam_inst_x" : beam_inst_x, "beam_inst_y" : beam_inst_y, "delta_KE_upstream" : delta_KE_upstream}


def GetTrueFFKE(KE_tpc : ak.Array, length_to_ff : ak.Array) -> ak.Array:
    """ True Front facing kinetic energy is the kinetic energy of the first particle trajectory point in the tpc,
        plus the energy lost between the first trajectory point in the TPC and trajectory point before.

    Args:
        KE_tpc (ak.Array): kinetic energy at the first trajectory point in the TPC
        length_to_ff (ak.Array): distance from first trajectory point in the TPC and the one before it. idea is to use this as an approximation of the distance to the front face of the TPC

    Returns:
        ak.Array: true kinetic energy at the front face of the TPC
    """
    dEdX = EnergyTools.BetheBloch.meandEdX(KE_tpc, Particle.from_pdgid(211))
    return KE_tpc + dEdX * length_to_ff


def dist_into_tpc(arr : ak.Array, ind : int) -> ak.Array:
    """ Calculates the distance between the trajectory point after ind and ind. Required for akward arrays

    Args:
        arr (ak.Array): trajectory points
        ind (int): index of point which is just before the TPC

    Returns:
        ak.Array: distance between points
    """
    outside_tpc = arr[ind]
    if ind + 1 >= ak.count(arr):
        ff_tpc = outside_tpc
    else:
        ff_tpc = arr[ind + 1]
    return ff_tpc - outside_tpc


def ff_value(arr : ak.Array, ind : int) -> any:
    """ returns a quantity of any trajectory point at the first trajectory point in the TPC.

    Args:
        arr (ak.Array): values
        ind (int): index to return value at

    Returns:
        any: value at ind
    """
    if ind + 1 >= ak.count(arr):
        return arr[ind]
    else:
        return arr[ind + 1]


def GetScraperFits(ke_bins : list, beam_inst_KE : ak.Array, delta_KE_upstream : ak.Array, fit_bins : int, residual_range : list) -> dict:
    """ Fit gaussians to front facing kinetic energies to extract values required to define events as beam scrapers.

    Args:
        ke_bins (list): bins of Kinetic energy to split the sample in.
        beam_inst_KE (ak.Array): beam instrumentation kinetic energy.
        delta_KE_upstream (ak.Array): difference in the front facing KE and beam kinetic energy.
        fit_bins (int): number of bins to histogram the data for the fit.
        residual_range (list): plot range.

    Returns:
        dict: fit parameters
    """
    plot = Plots.MultiPlot(len(ke_bins) - 1)

    scraper_fit = {}
    for i in range(1, len(ke_bins)):
        bin_label = "$KE^{reco}_{inst}$:" + f"[{ke_bins[i-1]},{ke_bins[i]}] (MeV)"
        e = (beam_inst_KE < ke_bins[i]) & (beam_inst_KE > ke_bins[i-1])

        data = delta_KE_upstream[e]

        y, bin_edges = np.histogram(np.array(data[~np.isnan(data)]), bins = fit_bins, range = sorted([np.nanpercentile(data, 10), np.nanpercentile(data, 90)]))
        yerr = np.sqrt(y) # Poisson error

        print(f"{(max(y), np.nanmedian(data), np.nanstd(data))=}")

        next(plot)
        popt, perr, metrics = Fitting.Fit(cross_section.bin_centers(bin_edges), y, yerr, Fitting.gaussian, method = "dogbox", return_chi_sqr = True)#, plot = True, plot_style = "scatter", xlabel = "$\Delta E_{upstream}$ (MeV)", title = bin_label, plot_range = residual_range)
        heights, _ = Plots.PlotHist(np.array(data[~np.isnan(data)]), newFigure = False, bins = fit_bins, range = residual_range, label = "observed")
        x_interp = np.linspace(min(np.array(data[~np.isnan(data)])), max(np.array(data[~np.isnan(data)])), 10 * fit_bins)
        y_interp = Fitting.gaussian.func(x_interp, max(heights), popt[1], popt[2])
        Plots.Plot(x_interp, y_interp, color = "black", label = "fit", title = bin_label, xlabel = "$\Delta E_{upstream}$ (MeV)", newFigure = False)
        plt.axvline(popt[1] + 3 * abs(popt[2]), color = "black", linestyle = "--", label = "$\mu + 3\sigma$")
        plt.xlim(*residual_range)

        main_legend = plt.legend(loc = "upper left")
        main_legend.set_zorder(12)

        #* add fit metrics to the plot in a second legend
        plt.gca().add_artist(main_legend)
        text = ""
        for j in range(len(popt)):
            text += f"\np{j}: ${popt[j]:.2g}\pm${perr[j]:.2g}"
        text += "\n$\chi^{2}/ndf$ : " + f"{metrics[0]/metrics[1]:.2g}, p : " + f"{metrics[2]:.1g}"
        legend = plt.gca().legend(handlelength = 0, labels = [text[1:]], loc = "upper right", title = Fitting.gaussian.__name__)
        legend.set_zorder(12)
        for l in legend.legend_handles:
            l.set_visible(False)

        print(f"{popt=}")
        print(f"{perr=}")
        scraper_fit[(ke_bins[i-1], ke_bins[i])] = {"mu_e_res" : popt[1], "sigma_e_res" : abs(popt[2])}
    return scraper_fit


def BeamScraperPlots(beam_inst_KE_bins : list, output_mc : dict[ak.Array], scraper_fits : dict) -> dict:
    """ Scatter plots of events to show beam scraper events as a function of the XY position at the beam instrumentation.

    Args:
        mc (Master.Data): events to look at.
        beam_inst_KE_bins (list): beam kinetic energy bins.
        beam_inst_KE (ak.Array): beam instrumentation kinetic energy.
        delta_KE_upstream (ak.Array): difference in the front facing KE and beam kinetic energy.
        scraper_fits (dict): parameters to define beam scrapers.

    Returns:
        dict: mean and standard deviations of the positions
    """

    output = {}

    for i in Plots.MultiPlot(len(beam_inst_KE_bins)-1):
        if i == len(beam_inst_KE_bins): continue
        bin_edges = (beam_inst_KE_bins[i], beam_inst_KE_bins[i+1])
        bin_label = "$KE^{reco}_{inst}$:" + f"[{beam_inst_KE_bins[i]},{beam_inst_KE_bins[i+1]}] (MeV)"
        e = (output_mc["beam_inst_KE"] < beam_inst_KE_bins[i+1]) & (output_mc["beam_inst_KE"] > beam_inst_KE_bins[i])
        fit_values = scraper_fits[(beam_inst_KE_bins[i], beam_inst_KE_bins[i+1])]

        is_scraper = output_mc["delta_KE_upstream"][e] > (fit_values["mu_e_res"] + 3 * fit_values["sigma_e_res"])

        Plots.Plot(output_mc["beam_inst_x"][e][~is_scraper], output_mc["beam_inst_y"][e][~is_scraper], newFigure = False, linestyle = "", marker = "o", markersize = 2, color = "C0", alpha = 0.5, label = "non-scraper", rasterized = True)
        Plots.Plot(output_mc["beam_inst_x"][e][is_scraper], output_mc["beam_inst_y"][e][is_scraper], newFigure = False, linestyle = "", marker = "o", markersize = 2, color = "C6", alpha = 0.5, label = "scraper", rasterized = True)

        mu_x = ak.mean(output_mc["beam_inst_x"][e])
        mu_y = ak.mean(output_mc["beam_inst_y"][e])
        print(bin_label)
        print(mu_x, mu_y)
        sigma_x = ak.std(output_mc["beam_inst_x"][e])
        sigma_y = ak.std(output_mc["beam_inst_y"][e])

        output[bin_edges] = {"mu_x_inst" : mu_x, "mu_y_inst" : mu_y, "sigma_x_inst" : sigma_x, "sigma_y_inst" : sigma_y}

        theta = np.linspace(0, 2*np.pi, 100)
        for j, m in enumerate([0.5, 1, 1.5, 2, 3]):
            r = m * (sigma_x**2 + sigma_y**2)**0.5
            x = r*np.cos(theta) + mu_x
            y = r*np.sin(theta) + mu_y

            Plots.Plot(x, y, linestyle = "--", color = f"C{7+j}", alpha = 1, label = f"{m}$r_{{inst}}$", newFigure = False)
        plt.xlabel("$X^{reco}_{inst}$ (cm)")
        plt.ylabel("$Y^{reco}_{inst}$ (cm)")
        plt.title(bin_label)
        plt.axis('scaled')
        plt.legend()
    return output

@Master.timer
def main(args : argparse.Namespace):
    cross_section.PlotStyler.SetPlotStyle(True)
    outdir = args.out + "beam_scraper/"
    os.makedirs(outdir, exist_ok = True)

    output_mc = Processing.ApplicationProcessing(["mc"], outdir, args, run, True)["mc"]

    residual_range = [-300, 300] # range of residual for plots
    bins = 50

    with Plots.PlotBook(outdir + "beam_scraper_fits.pdf") as pdf:
        Plots.Plot(args.beam_scraper_energy_range, args.beam_scraper_energy_range, color = "red")
        Plots.PlotHist2D(output_mc["beam_inst_KE"], output_mc["true_ffKE"], xlabel = "$KE^{reco}_{inst}$ (MeV)", ylabel = "$KE^{true}_{ff}$ (MeV)", x_range = args.beam_scraper_energy_range, y_range = args.beam_scraper_energy_range, newFigure = False)
        pdf.Save()

        Plots.PlotHist2D(output_mc["beam_inst_KE"], output_mc["delta_KE_upstream"], xlabel = "$KE^{reco}_{inst}$ (MeV)", ylabel = "$\Delta E_{upstream}$ (MeV)", x_range = args.beam_scraper_energy_range, y_range = residual_range)
        for i in args.beam_scraper_energy_bins: plt.axvline(i, color = "red")
        pdf.Save()

        scraper_thresholds = GetScraperFits(args.beam_scraper_energy_bins, output_mc["beam_inst_KE"], output_mc["delta_KE_upstream"], bins, residual_range)
        pdf.Save()
        print(scraper_thresholds)

        position_means = BeamScraperPlots(args.beam_scraper_energy_bins, output_mc, scraper_thresholds)
        print(position_means)
        pdf.Save()

    json_dict = {}
    for i, k in enumerate(scraper_thresholds):
        json_dict[str(i)] = {**{"bins" : k}, **scraper_thresholds[k], **position_means[k]}

    name = outdir + "mc_beam_scraper_fit_values.json"
    cross_section.SaveConfiguration(json_dict, name)
    print(f"fit values written to {name}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Calculates parameters required to idenfify beam scrapers and apply the selection.")

    cross_section.ApplicationArguments.Config(parser, required = True)
    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Regen(parser)

    parser.add_argument("--energy_range", dest = "beam_scraper_energy_range", type = float, nargs = 2, help = "energy range to study (MeV).")
    parser.add_argument("--energy_bins", dest = "beam_scraper_energy_bins", type = float, nargs = 5, help = "kinetic energy bin edges (currently allows only 4 bins to be made) (MeV)")

    args = parser.parse_args()
    args = cross_section.ApplicationArguments.ResolveArgs(args)

    print(vars(args))
    main(args)