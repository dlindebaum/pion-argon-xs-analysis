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


def ReWeight(p_MC, p_Data, p_nominal : float, bins : int = 10, p_range : np.array = np.array([0.75, 1.25]), book : Plots.PlotBook = Plots.PlotBook.null):
    p_mc, edges = np.histogram(np.array(p_MC), bins, range = p_nominal * p_range)
    p_data = np.histogram(np.array(p_Data), bins, range = p_nominal * p_range)[0]

    with Plots.RatioPlot((edges[1:] + edges[:-1]) / 2, p_data, p_mc, np.sqrt(p_data), np.sqrt(p_mc), "$P_{inst}^{reco}$ (MeV)", "Data/MC") as ratio_plot:
        Plots.Plot(ratio_plot.x, ratio_plot.y1, yerr = ratio_plot.y1_err, newFigure = False)
        Plots.Plot(ratio_plot.x, ratio_plot.y2, yerr = ratio_plot.y2_err, newFigure = False, ylabel = "Counts")
    book.Save()

    scale = sum(ratio_plot.y1) / sum(ratio_plot.y2)

    ratio = scale * np.nan_to_num(cross_section.nandiv(ratio_plot.y2, ratio_plot.y1), posinf = 0)
    ratio_err = np.nan_to_num(abs(ratio * np.sqrt(cross_section.nandiv(ratio_plot.y1_err, ratio_plot.y1)**2 + cross_section.nandiv(ratio_plot.y2_err, ratio_plot.y2)**2)))

    Plots.Plot(ratio_plot.x, ratio, yerr = ratio_err, xlabel = "$P_{inst}^{reco}$ (MeV)", ylabel = "$r$")
    book.Save()

    results = {}
    for f in [cross_section.Fitting.gaussian, cross_section.Fitting.student_t, cross_section.Fitting.poly2d, cross_section.Fitting.crystal_ball, cross_section.Fitting.double_crystal_ball, cross_section.Fitting.double_gaussian]:
        Plots.plt.figure()
        results[f.__name__] = cross_section.Fitting.Fit(ratio_plot.x[ratio > 0], ratio[ratio > 0], ratio_err[ratio > 0], f, plot = True, xlabel = "$P_{inst}^{reco}$(MeV)", ylabel = "$r$")
        book.Save()
    return results


def ReWeightResults(sideband_mc : dict, sideband_data : dict, args : cross_section.argparse.Namespace, bins : int, reweight_results : dict, reweight_func : str, book : Plots.PlotBook = Plots.PlotBook.null):
    weights = cross_section.RatioWeights(np.array(sideband_mc["p_inst"]), reweight_func, reweight_results[reweight_func][0], args.beam_reweight["strength"])

    plot_range = [args.beam_momentum * 0.75, args.beam_momentum * 1.25]

    Plots.PlotHist(weights, range = [0, 3], xlabel = "weights", truncate = True)
    book.Save()

    Plots.PlotTagged(sideband_mc["p_inst"], sideband_mc["tags"], data2 = sideband_data["p_inst"], x_range = plot_range, norm = args.norm, data_weights = None, bins = bins, x_label = "$P_{inst}^{reco}$ (MeV)", ncols = 1)
    Plots.plt.title("nominal", pad = 15)
    book.Save()

    Plots.PlotTagged(sideband_mc["p_inst"], sideband_mc["tags"], data2 = sideband_data["p_inst"], x_range = plot_range, norm = args.norm, data_weights = weights, bins = bins, x_label = "$P_{inst}^{reco}$ (MeV)", ncols = 1)
    Plots.plt.title(f"reweighted : {reweight_func}", pad = 15)
    book.Save()
    return


def SmearingFactors(sample, weights : np.array = None):
    average = np.average(sample.recoParticles.beam_inst_P, weights=weights)
    variance = np.average((sample.recoParticles.beam_inst_P-average)**2, weights=weights)
    std = np.sqrt(variance)
    return average, std


def run(i : int, file : str, n_events : int, start : int, selected_events, args : dict):

    sample = "data" if args["data"] else "mc"

    selections = args["selection_masks"][sample]

    if "fiducial" in selections:
        if len(selections["fiducial"]) > 0:
            fiducial_mask = SelectionTools.CombineMasks(selections["fiducial"][file])
        else:
            fiducial_mask = None
    else:
        fiducial_mask = None

    invert = ["HasFinalStatePFOsCut"] # invert preselection

    sideband_selection = {}
    for m in selections["beam"][file]:
        if m in invert:
            sideband_selection[m] = ~selections["beam"][file][m]
        else:
            sideband_selection[m] = selections["beam"][file][m]

    table = {}
    mask = None
    for s in sideband_selection:
        if mask is None:
            mask = sideband_selection[s]
        else:
            mask = mask & sideband_selection[s]
        table[s] = sum(mask)

    print(table)
    sideband_selection = SelectionTools.CombineMasks(sideband_selection)

    events = cross_section.Data(file, n_events, start, args["nTuple_type"], args["pmom"])

    mask = SelectionTools.CombineMasks(selections["beam"][file])

    if fiducial_mask is not None:
        masks = [fiducial_mask, mask]
    else:
        masks = [mask]

    analysis_sample = events.Filter(masks, masks, returnCopy = True)

    if fiducial_mask is not None:
        masks = [fiducial_mask, sideband_selection]
    else:
        masks = [sideband_selection]

    sideband_sample = events.Filter(masks, masks, returnCopy = True)

    output = {
        "sideband" : {
            "p_inst" : sideband_sample.recoParticles.beam_inst_P,
            "tags" : cross_section.Tags.GenerateTrueBeamParticleTags(sideband_sample)
        },
        "analysis" : {
            "p_inst" : analysis_sample.recoParticles.beam_inst_P,
            "tags" : cross_section.Tags.GenerateTrueBeamParticleTags(analysis_sample)
        },
        "table" : table
    }

    return output


@cross_section.timer
def main(args : cross_section.argparse.Namespace):
    cross_section.PlotStyler.SetPlotStyle(extend_colors = True, dpi = 100)
    out = args.out + "beam_reweight/"
    os.makedirs(out, exist_ok = True)

    args.batches = None
    args.events = None
    args.threads = 1

    outputs = cross_section.ApplicationProcessing(list(args.ntuple_files.keys()), out, args, run, True)

    for o in outputs:
        for t in outputs[o]["table"]:
            if type(outputs[o]["table"][t]) == list:
                outputs[o]["table"][t] = sum(outputs[o]["table"][t])

    output_mc = outputs["mc"]
    output_data = outputs["data"]

    table_data = cross_section.pd.DataFrame(output_data["table"], index = ["Counts"]).T
    table_mc = cross_section.pd.DataFrame(output_mc["table"], index = ["Counts"]).T

    print(table_data)
    print(table_mc)

    table_data.to_hdf(out + "selection_data.hdf5", key = "df")
    table_mc.to_hdf(out + "selection_mc.hdf5", key = "df")
    os.makedirs(out + "plots/", exist_ok = True)

    with Plots.PlotBook(out + "plots/" + "reweight_fits.pdf", True) as book:
        results = ReWeight(output_mc["sideband"]["p_inst"], output_data["sideband"]["p_inst"], args.beam_momentum, 20, np.array([0.75, 1.25]), book = book)

    for r in results:
        with Plots.PlotBook(out + "plots/" + f"reweight_results_{r}.pdf", True) as book:
            ReWeightResults(output_mc["sideband"], output_data["sideband"], args, 25, results, r, book = book)
            ReWeightResults(output_mc["analysis"], output_data["analysis"], args, 50, results, r, book = book)
            reweight_params = {f"p{i}" : {"value" : results[r][0][i], "error" : results[r][1][i]} for i in range(getattr(cross_section.Fitting, r).n_params)}
            cross_section.SaveConfiguration(reweight_params, out + r + ".json")
        Plots.plt.close("all")

    reweight_params = cross_section.LoadConfiguration(out + "gaussian" + ".json")


    chi2_table = {}
    test_range = [1700, 2300]
    for s, b in zip(["sideband", "analysis"], [25, 50]):
        print(s)
        mc_mom = np.array(output_mc[s]["p_inst"])
        data_mom = np.array(output_data[s]["p_inst"])

        mc_weights = cross_section.RatioWeights(mc_mom, "gaussian", [reweight_params[k]["value"] for k in reweight_params], 100)
        mc_weights_truncated = cross_section.RatioWeights(mc_mom, "gaussian", [reweight_params[k]["value"] for k in reweight_params], args.beam_reweight["strength"])

        mc_counts, bins = np.histogram(mc_mom, b, test_range)
        data_counts, _ = np.histogram(data_mom, bins, test_range)

        mc_weight_counts, bins = np.histogram(mc_mom, bins, test_range, weights = mc_weights)
        mc_weight_counts_truncated, bins = np.histogram(mc_mom, bins, test_range, weights = mc_weights_truncated)

        chi2_table[s] = {}
        for k, v in zip(["unweighted", "weighted", f"weighted $w < {args.beam_reweight['strength']}$"], [mc_counts, mc_weight_counts, mc_weight_counts_truncated]):
            chi2_table[s][k] = (b - 1) * cross_section.weighted_chi_sqr(data_counts.astype(float), v.astype(float), v.astype(float))

    table = cross_section.pd.DataFrame(chi2_table)
    table.to_hdf(out + "chi2_reweight.hdf5", key = "df")
    table.to_latex(out + "chi2_reweight.tex")

    return

if __name__ == "__main__":
    args = cross_section.argparse.ArgumentParser("Calculates reweighting parameters for beam momentum.")
    cross_section.ApplicationArguments.Config(args, True)
    cross_section.ApplicationArguments.Output(args)
    cross_section.ApplicationArguments.Regen(args)

    args = cross_section.ApplicationArguments.ResolveArgs(args.parse_args())
    print(vars(args))
    main(args)