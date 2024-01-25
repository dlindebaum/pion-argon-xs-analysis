#!/usr/bin/env python3
"""
Created on: 02/12/2023 13:14

Author: Shyam Bhuller

Description: reweighting parameters for beam momentum.
"""
import os

import numpy as np

from rich import print

from python.analysis import cross_section, Plots, SelectionTools



def ReWeight(sample : dict[cross_section.Data], p_nominal : float, bins : int = 10, p_range : np.array = np.array([0.75, 1.25]), book : Plots.PlotBook = Plots.PlotBook.null):
    p_mc, edges = np.histogram(np.array(sample["mc"].recoParticles.beam_inst_P), bins, range = p_nominal * p_range)
    p_data = np.histogram(np.array(sample["data"].recoParticles.beam_inst_P), bins, range = p_nominal * p_range)[0]

    with Plots.RatioPlot((edges[1:] + edges[:-1]) / 2, p_data, p_mc, np.sqrt(p_data), np.sqrt(p_mc), "$P_{inst}^{reco}$ (MeV)", "Data/MC") as ratio_plot:
        Plots.Plot(ratio_plot.x, ratio_plot.y1, yerr = ratio_plot.y1_err, newFigure = False)
        Plots.Plot(ratio_plot.x, ratio_plot.y2, yerr = ratio_plot.y2_err, newFigure = False, ylabel = "Counts")
    book.Save()

    scale = sum(ratio_plot.y1) / sum(ratio_plot.y2)

    ratio = scale * np.nan_to_num(cross_section.nandiv(ratio_plot.y2, ratio_plot.y1), posinf = 0)
    ratio_err = np.nan_to_num(abs(ratio * np.sqrt(cross_section.nandiv(ratio_plot.y1_err, ratio_plot.y1)**2 + cross_section.nandiv(ratio_plot.y2_err, ratio_plot.y2)**2)))

    # r = "$\\frac{R N_{mc}}{N_{data}}$"
    Plots.Plot(ratio_plot.x, ratio, yerr = ratio_err, xlabel = "$P_{inst}^{reco}$ (MeV)", ylabel = "$r$")
    book.Save()

    results = {}
    for f in [cross_section.Fitting.gaussian, cross_section.Fitting.student_t, cross_section.Fitting.poly2d, cross_section.Fitting.crystal_ball, cross_section.Fitting.double_crystal_ball, cross_section.Fitting.double_gaussian]:
        Plots.plt.figure()
        results[f.__name__] = cross_section.Fitting.Fit(ratio_plot.x[ratio > 0], ratio[ratio > 0], ratio_err[ratio > 0], f, plot = True, xlabel = "$P_{inst}^{reco}$(MeV)", ylabel = "$r$")
        book.Save()
    return results


def RatioWeights(mc : cross_section.Data, func : str, params : list, truncate : int = 10):
    weights = 1/getattr(cross_section.Fitting, func)(np.array(mc.recoParticles.beam_inst_P), *params)
    weights = np.where(weights > truncate, truncate, weights)
    return weights


def ReWeightResults(sample : dict[cross_section.Data], args : cross_section.argparse.Namespace, bins : int, reweight_results : dict, reweight_func : str, book : Plots.PlotBook = Plots.PlotBook.null):
    weights = RatioWeights(sample["mc"], reweight_func, reweight_results[reweight_func][0], 3)

    plot_range = [args.beam_momentum * 0.75, args.beam_momentum * 1.25]

    Plots.PlotHist(weights, range = [0, 3], xlabel = "weights", truncate = True)
    book.Save()

    Plots.PlotTagged(sample["mc"].recoParticles.beam_inst_P, cross_section.Tags.GenerateTrueBeamParticleTags(sample["mc"]), data2 = sample["data"].recoParticles.beam_inst_P, x_range = plot_range, norm = args.norm, data_weights = None, bins = bins, title = "nominal", x_label = "$P_{inst}^{reco}$ (MeV)")
    book.Save()

    Plots.PlotTagged(sample["mc"].recoParticles.beam_inst_P, cross_section.Tags.GenerateTrueBeamParticleTags(sample["mc"]), data2 = sample["data"].recoParticles.beam_inst_P, x_range = plot_range, norm = args.norm, data_weights = weights, bins = bins, title = f"reweighted : {reweight_func}", x_label = "$P_{inst}^{reco}$ (MeV)")
    book.Save()
    return


def SmearingFactors(sample, weights : np.array = None):
    average = np.average(sample.recoParticles.beam_inst_P, weights=weights)
    variance = np.average((sample.recoParticles.beam_inst_P-average)**2, weights=weights)
    std = np.sqrt(variance)
    return average, std

@cross_section.timer
def main(args : cross_section.argparse.Namespace):
    cross_section.SetPlotStyle(extend_colors = True, dpi = 100)

    invert = "HasFinalStatePFOsCut"
    sideband_selection = {}
    for k, sample in args.selection_masks.items():
        masks = {}
        for m in sample["beam"]:
            if m == invert:
                masks[m] = ~sample["beam"][m]
            else:
                masks[m] = sample["beam"][m]
        sideband_selection[k] = SelectionTools.CombineMasks(masks, "and")
    print({k : sum(v) for k, v in sideband_selection.items()})
    os.makedirs(args.out, exist_ok = True)
    os.makedirs(args.out + "plots/", exist_ok = True)

    events = {"mc" : cross_section.Data(args.mc_file, nTuple_type = args.ntuple_type, target_momentum = args.pmom), "data" : cross_section.Data(args.data_file, nTuple_type = args.ntuple_type)}

    analysis_sample = {}
    for s in events:
        mask = SelectionTools.CombineMasks(args.selection_masks[s]["beam"])
        analysis_sample[s] = events[s].Filter([mask], [mask], returnCopy = True)

    sideband_sample = {}
    for s in events:
        sideband_sample[s] = events[s].Filter([sideband_selection[s]], [sideband_selection[s]], returnCopy = True)

    with Plots.PlotBook(args.out + "plots/" + "reweight_fits.pdf", True) as book:
        results = ReWeight(sideband_sample, args.beam_momentum, 20, np.array([0.75, 1.25]), book = book)

    for r in results:
        with Plots.PlotBook(args.out + "plots/" + f"reweight_results_{r}.pdf", True) as book:
            ReWeightResults(sideband_sample, args, 25, results, r, book = book)
            ReWeightResults(analysis_sample, args, 50, results, r, book = book)
            reweight_params = {f"p{i}" : {"value" : results[r][0][i], "error" : results[r][1][i]} for i in range(getattr(cross_section.Fitting, r).n_params)}
            cross_section.SaveConfiguration(reweight_params, args.out + r + ".json")
        Plots.plt.close("all")
    return

if __name__ == "__main__":
    args = cross_section.argparse.ArgumentParser("Calculates reweighting parameters for beam momentum.")
    cross_section.ApplicationArguments.Config(args, "True")
    cross_section.ApplicationArguments.Output(args)

    args = cross_section.ApplicationArguments.ResolveArgs(args.parse_args())
    args.out = args.out + "beam_reweight/"
    print(vars(args))
    main(args)