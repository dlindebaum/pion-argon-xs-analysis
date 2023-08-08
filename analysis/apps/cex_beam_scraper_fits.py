#!/usr/bin/env python3
"""
Created on: 13/06/2023 11:19

Author: Shyam Bhuller

Description: Produces plots and fits for the beam scraper study, producing thresholds which dictate what events are beam scrapers.
"""
import argparse
import json
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from particle import Particle
from rich import print

from python.analysis import Master, cross_section, vector, Plots, Fitting


def GetTrueFFKE(KE_tpc : ak.Array, length_to_ff : ak.Array) -> ak.Array:
    """ True Front facing kinetic energy is the kinetic energy of the first particle trajectory point in the tpc,
        plus the energy lost between the first trajectory point in the TPC and trajectory point before.

    Args:
        KE_tpc (ak.Array): kinetic energy at the first trajectory point in the TPC
        length_to_ff (ak.Array): distance from first trajectory point in the TPC and the one before it. idea is to use this as an approximation of the distance to the front face of the TPC

    Returns:
        ak.Array: true kinetic energy at the front face of the TPC
    """
    dEdX = cross_section.BetheBloch.meandEdX(KE_tpc, Particle.from_pdgid(211))
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
    plt.subplots(2, 2, figsize = [6.4 * 2, 4.8 * 2], num = 1)
    plt.gcf().set_size_inches(6.4 * 2, 4.8 * 2)

    scraper_fit = {}
    for i in range(1, len(ke_bins)):
        bin_label = "$KE_{inst}$:" + f"[{ke_bins[i-1]},{ke_bins[i]}] (MeV)"
        e = (beam_inst_KE < ke_bins[i]) & (beam_inst_KE > ke_bins[i-1])

        data = delta_KE_upstream[e]

        y, bin_edges = np.histogram(np.array(data[~np.isnan(data)]), bins = fit_bins, range = sorted([np.nanpercentile(data, 10), np.nanpercentile(data, 90)]))
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        yerr = np.sqrt(y) # Poisson error

        print(f"{(max(y), np.nanmedian(data), np.nanstd(data))=}")

        popt, perr = Fitting.Fit(bin_centers, y, yerr, Fitting.gaussian, method = "dogbox")
        print(f"{popt=}")
        print(f"{perr=}")

        plt.figure(1)
        plt.subplot(2, 2, i)
        heights, _ = Plots.PlotHist(data, newFigure = False, bins = fit_bins, range = residual_range)

        x_interp = np.linspace(min(data), max(data), 10 * fit_bins)
        y_interp = Fitting.gaussian.func(x_interp, max(heights), popt[1], popt[2])
        Plots.Plot(x_interp, y_interp, color = "black", label = "fit", title = bin_label, xlabel = "$KE_{inst} - KE_{true}$ (MeV)", newFigure = False)
        plt.axvline(popt[1] + 3 * abs(popt[2]), color = "black", linestyle = "--", label = "$\mu{+}+3\sigma$")
        plt.xlim(*residual_range)

        scraper_fit[(ke_bins[i-1], ke_bins[i])] = {"mu" : popt[1], "sigma" : abs(popt[2])}
    return scraper_fit


def BeamScraperPlots(mc: Master.Data, beam_inst_KE_bins : list, beam_inst_KE : ak.Array, delta_KE_upstream : ak.Array, scraper_fits : dict):
    """ Scatter plots of events to show beam scraper events as a function of the XY position at the beam instrumentation.

    Args:
        mc (Master.Data): events to look at.
        beam_inst_KE_bins (list): beam kinetic energy bins.
        beam_inst_KE (ak.Array): beam instrumentation kinetic energy.
        delta_KE_upstream (ak.Array): difference in the front facing KE and beam kinetic energy.
        scraper_fits (dict): _description_
    """

    for i in Plots.MultiPlot(len(beam_inst_KE_bins)-1, sharex = True, sharey = True):
        if i == len(beam_inst_KE_bins): continue
        bin_label = "$KE_{inst}$:" + f"[{beam_inst_KE_bins[i]},{beam_inst_KE_bins[i+1]}] (MeV)"
        e = (beam_inst_KE < beam_inst_KE_bins[i+1]) & (beam_inst_KE > beam_inst_KE_bins[i])
        fit_values = scraper_fits[(beam_inst_KE_bins[i], beam_inst_KE_bins[i+1])]

        is_scraper = delta_KE_upstream[e] > (fit_values["mu"] + 3 * fit_values["sigma"])

        Plots.Plot(mc.recoParticles.beam_inst_pos.x[e][~is_scraper], mc.recoParticles.beam_inst_pos.y[e][~is_scraper], newFigure = False, linestyle = "", marker = "o", markersize = 2, color = "C0", alpha = 0.5, label = "non-scraper")
        Plots.Plot(mc.recoParticles.beam_inst_pos.x[e][is_scraper], mc.recoParticles.beam_inst_pos.y[e][is_scraper], newFigure = False, linestyle = "", marker = "o", markersize = 2, color = "C6", alpha = 0.5, label = "scraper")

        mu_x = ak.mean(mc.recoParticles.beam_inst_pos[e].x)
        mu_y = ak.mean(mc.recoParticles.beam_inst_pos[e].y)
        sigma_x = ak.std(mc.recoParticles.beam_inst_pos[e].x)
        sigma_y = ak.std(mc.recoParticles.beam_inst_pos[e].y)


        theta = np.linspace(0, 2*np.pi, 100)
        for j, m in enumerate([0.5, 1, 1.5, 2, 3]):
            r = m * (sigma_x**2 + sigma_y**2)**0.5
            x = r*np.cos(theta) + mu_x
            y = r*np.sin(theta) + mu_y

            Plots.Plot(x, y, linestyle = "--", color = f"C{7+j}", alpha = 1, label = f"{m}$r$", newFigure = False)
        plt.xlabel("$X_{inst}$ (cm)")
        plt.ylabel("$Y_{inst}$ (cm)")
        plt.title(bin_label)
        plt.axis('scaled')
        plt.legend()


def main(args : argparse.Namespace):
    cross_section.SetPlotStyle(True)

    mc = Master.Data(args.mc_file[0], nTuple_type = args.ntuple_type)
    bq_fit = cross_section.LoadConfiguration(args.mc_beam_quality_fit)
    mask = cross_section.BeamParticleSelection.CreateDefaultSelection(mc, False, bq_fit, return_table = False)
    mc.Filter([mask], [mask]) # apply default beam selection

    beam_inst_KE = cross_section.KE(mc.recoParticles.beam_inst_P, Particle.from_pdgid(211).mass) # get kinetic energy from beam instrumentation

    true_ff_ind = ak.argmax(mc.trueParticles.beam_traj_pos.z >= 0, -1, keepdims = True)
    # not_int_tpc = true_ff_ind == 0

    pitches = ak.ravel(vector.dist(mc.trueParticles.beam_traj_pos[true_ff_ind], mc.trueParticles.beam_traj_pos[true_ff_ind + 1]))
    first_KE = mc.trueParticles.beam_traj_KE[true_ff_ind]

    true_ffKE = ak.ravel(GetTrueFFKE(first_KE, pitches)) # get the true Kinetic energy as the front face of the TPC
    delta_KE_upstream = beam_inst_KE - true_ffKE

    residual_range = [-300, 300] # range of residual for plots
    bins = 50

    os.makedirs(args.out + "beam_scraper/", exist_ok = True)

    with PdfPages(args.out + "beam_scraper/" + "beam_scraper_fits.pdf") as pdf:
        Plots.Plot(args.energy_range, args.energy_range, color = "red")
        Plots.PlotHist2D(beam_inst_KE, true_ffKE, xlabel = "$KE_{inst}$ (MeV)", ylabel = "$KE_{true}$ (MeV)", x_range = args.energy_range, y_range = args.energy_range, newFigure = False)
        pdf.savefig()

        Plots.PlotHist2D(beam_inst_KE, delta_KE_upstream, xlabel = "$KE_{inst}$ (MeV)", ylabel = "$KE_{inst} - KE_{true}$ (MeV)", x_range = args.energy_range, y_range = residual_range)
        for i in args.beam_inst_KE_bins: plt.axvline(i, color = "red")
        pdf.savefig()

        scraper_thresholds = GetScraperFits(args.beam_inst_KE_bins, beam_inst_KE, delta_KE_upstream, bins, residual_range)
        pdf.savefig()
        print(scraper_thresholds)

        BeamScraperPlots(mc, args.beam_inst_KE_bins, beam_inst_KE, delta_KE_upstream, scraper_thresholds)
        pdf.savefig()

    json_dict = {}
    for i, (k, v) in enumerate(scraper_thresholds.items()):
        json_dict[str(i)] = {**{"bins" : k}, **v}

    name = args.out + "beam_scraper/" + "mc_beam_scraper_fit_values.json"
    with open(name, "w") as f:
        json.dump(json_dict, f, indent = 4)
    print(f"fit values written to {name}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Ntuples(parser)
    cross_section.ApplicationArguments.BeamQualityCuts(parser)
    cross_section.ApplicationArguments.Output(parser)

    parser.add_argument("--energy_range", dest = "energy_range", type = float, nargs = 2, help = "energy range to study (MeV).")
    parser.add_argument("--energy_bins", dest = "beam_inst_KE_bins", type = float, nargs = 5, help = "kinetic energy bin edges (currently allows only 4 bins to be made) (MeV)")

    args = parser.parse_args()
    cross_section.ApplicationArguments.ResolveArgs(args)

    print(vars(args))   
    main(args)