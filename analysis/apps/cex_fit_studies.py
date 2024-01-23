#!/usr/bin/env python3
"""
Created on: 17/01/2024 13:46

Author: Shyam Bhuller

Description: app which performs cross checks for the region fit using toys.
"""
import os

import numpy as np
import pandas as pd

from rich import print
from scipy.interpolate import CubicSpline
from scipy.stats import norm, lognorm


from python.analysis import cross_section, Plots
from apps import cex_toy_generator

region_colours = {
    "absorption"      : "#777777",
    "charge_exchange" : "#8EBA42",
    "single_pion_production" : "#E24A33",
    "pion_production" : "#988ED5",
}


target_map = {
    "_abs" : 'absorption',
    "_quasi" : 'quasielastic',
    "_cex" : 'charge_exchange',
    "_dcex" : 'double_charge_exchange',
    "_pip" : 'pion_production'
}

#? make these json configs?
spline_point_index = {
    "absorption" : [5, 40, 65, 100, 180, 245, 300, 420, 750, 1120, -500, -1],
    "quasielastic" : [5, 100, 200, 270, 330, 520, 650, 860, 1120, -750, -250, -1],
    "charge_exchange" : [5, 100, 200, 270, 330, 650, 860, 1120, -750, -250, -1],
    "double_charge_exchange" : [5, 150, 270, 400, 800, 1120, -450, -1],
    "pion_production" : [5, 400, 600, 850, 1120, -450, -1],
}
# arranged low, high
spline_shape_param_factors = {
    "absorption" : [np.array([1.2, 1, 1, 1, 1.2, 1.1, 1, 0.85, 1, 1.2, 1, 0.8]), np.array([0.8, 1, 1, 1, 0.8, 0.9, 1, 1.15, 1, 0.8, 1, 1.2])],
    "quasielastic" : [np.array([0.8, 0.8, 0.8, 1.15, 1.1, 0.85, 0.8, 1, 1.2, 1.1, 0.9, 0.8]), np.array([1.2, 1.2, 0.8, 0.8, 0.9, 1.2, 1, 0.9, 0.8, 0.9, 1.1, 1.2])],
    "charge_exchange" : [np.array([1.2, 1.2, 1.2, 1.15, 1, 1, 1.2, 1.1, 1, 0.9, 0.8]), np.array([0.8, 0.8, 0.8, 0.85, 1, 1, 0.82, 0.82, 0.9, 1.1, 1.2])],
    "double_charge_exchange" : [np.array([1.2, 1.2, 1.2, 0.85, 1.1, 1.2, 1, 0.8]), np.array([0.8, 0.8, 0.8, 1.15, 0.85, 0.85, 1, 1.2])],
    "pion_production" : [np.array([0.8, 0.8, 1, 1.2, 1.1, 0.9, 0.8]), np.array([1.2, 1.2, 0.8, 0.8, 0.9, 1.1, 1.2])]
}

shape_param_factors = {
    "absorption" : [np.array([1.7, 1, 1, 1, 1, 1.3, 0.9, 1.5]), np.array([0.4, 1, 1, 1, 100, 0.9, 1.2, 1])],
    "quasielastic" : [np.array([1.8, 1.2, 1.2, 1, 1, 1, 1, 0.5]), np.array([0.45, 0.85, 0.8, 1.08, 1.15, 1, 1, 1.5])],
    "charge_exchange" : [np.array([1.5, 1.1, 1, 0.88, 1, 1.1, 1, 1]), np.array([0.7, 0.9, 1, 1.1, 1, 0.95, 1, 1])],
    "double_charge_exchange" : [np.array([0.7, 0.9, 1, 1.15, 1, 0.95, 1, 1]), np.array([1.5, 1.1, 0.95, 0.825, 1, 1.05, 1, 0.95])],
    "pion_production" : [np.array([0, 1, 1, 0.88, 1.005, 0.85, 0.1, 1]), np.array([0, 1, 1, 0.75, 1.01, 1, 0.1, 2])]
}


def OverrideConfig(args):
    for a in ["step", "events", "seed"]:
        if getattr(args, a):
            args.toy_data_config[a] = getattr(args, a)
    return


def CreateConfigNormalisation(scales : dict, data_config : dict) -> dict:
    cfg = {k : v for k, v in data_config.items()}
    cfg["pdf_scale_factors"] = scales
    return cfg


def ModifiedConfigTest(config : dict, energy_slice : cross_section.Slices, model : cross_section.pyhf.Model, toy_template : cross_section.Toy, mean_track_score_bins : np.array = None) -> tuple[dict]:
    toy_alt_pdf = cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(df = cex_toy_generator.main(cex_toy_generator.ResolveConfig(config))))
    
    obs = cross_section.RegionFit.GenerateObservations(toy_alt_pdf, energy_slice, mean_track_score_bins, model, False)
    # fit_result = cross_section.RegionFit.Fit(obs, model, None, par_bounds = [(0, np.inf)]*model.config.npars)
    fit_result = cross_section.RegionFit.Fit(obs, model, None, verbose = False)

    true_process_counts = {}
    for v in toy_alt_pdf.exclusive_process:
        true_process_counts[v] = toy_alt_pdf.NInteract(energy_slice, toy_alt_pdf.exclusive_process[v])

    expected_mus = [sum(toy_alt_pdf.exclusive_process[r]) / sum(toy_template.truth_regions[r]) for r in toy_alt_pdf.exclusive_process]

    return fit_result, true_process_counts, expected_mus


def NormalisationTest(directory : str, data_config : dict, model : cross_section.pyhf.Model, toy_template : cross_section.Toy, energy_slice : cross_section.Slices, mean_track_score_bins : np.ndarray):
    folder = {
        'absorption': "abs",
        'quasielastic': "quasi",
        'charge_exchange': "cex",
        'double_charge_exchange': "dcex",
        'pion_production': "pip"
    }

    for target in folder:
        print(target)

        results = {}
        true_counts = {}
        expected_mus = {}
        scales = {k : 1 for k in folder}
        for i in [0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.5]:
            print(f"normalisation : {i}")
            if i == 1:
                config = data_config
            else:
                scales[target] = i
                config = CreateConfigNormalisation(scales, data_config)
            results[i], true_counts[i], expected_mus[i] = ModifiedConfigTest(config, energy_slice, model, toy_template, mean_track_score_bins)

        cross_section.SaveObject(f"{directory}fit_results_{folder[target]}.dill", {"results" : results, "true_counts" : true_counts, "expected_mus" : expected_mus})
    return


class lognormal_gaussian_exp(cross_section.Fitting.FitFunction):
    n_params = 8

    def __new__(cls, x, p0, p1, p2, p3, p4, p5, p6, p7) -> np.array:
        return cls.func(x, p0, p1, p2, p3, p4, p5, p6, p7)

    def func(x, p0, p1, p2, p3, p4, p5, p6, p7):
        lognormal_component = lognorm.pdf(x, s = p2, scale = p1)
        gaussian_component = norm.pdf(x, loc = p4, scale = p5)
        exponential_component = np.exp(-p7 * x)
        return p0 * lognormal_component + p3 * gaussian_component + p6 * exponential_component # Adjust weights as needed

    def bounds(x, y):
        lims = np.array([
            (0, 1),
            (min(x), max(x)),
            (0.001, np.inf),

            (0, 1),
            (min(x), max(x)),
            (0.001, np.inf),

            (-np.inf, np.inf),
            (-np.inf, np.inf),

        ])
        return (lims[:, 0], lims[:, 1])    


def FitSimulationCurve(xs_sim, sampling_factor : int, process : str, function : cross_section.Fitting.FitFunction, plot : bool = False):
    x = xs_sim.KE[::sampling_factor]/1000
    y = getattr(xs_sim, process)[::sampling_factor]/1000

    if plot is True: Plots.plt.figure()
    results = cross_section.Fitting.Fit(x, y, None, function, method = "dogbox", plot = plot, xlabel = "$KE$(GeV)", ylabel = "$\sigma$ (b)", title = process)
    return results


def CreateShapeParams(xs_sim, process : str, high, low, plot : bool = False):
    xs_fit_results = FitSimulationCurve(xs_sim, 20, process, lognormal_gaussian_exp, False)
    params = np.array(xs_fit_results[0])
    step = (high - low) / 100
    shape_params = []
    for i in [0, 25, 75, 100]:
        shape_params.append((low + (i *step)) * params)
    shape_params.insert(2, params)

    if plot is True:
        xs = getattr(xs_sim, process)
        Plots.Plot(xs_sim.KE, xs, color = "k", xlabel = "KE (MeV)", ylabel = "$\sigma$(mb)")
        Plots.plt.fill_between(xs_sim.KE, 0.8 * xs, 1.2 * xs, alpha = 0.5, color = "k", label = "Geant 4 $\pm$ 20%")

        c = Plots.plt.cm.autumn_r(np.linspace(0, 1, len(shape_params)))
        for i in range(len(shape_params)):
            Plots.Plot(xs_sim.KE, 1000 * lognormal_gaussian_exp(xs_sim.KE/1000, *shape_params[i]), color = c[i], linestyle = (0, (5, 6)), newFigure = False, label = i)
    return shape_params


def CreateShapeParamsSpline(xs_sim, process, indices, shape_factors, plot):
    xs = getattr(xs_sim, process)
    Plots.Plot(xs_sim.KE, xs, color = "k", xlabel = "KE (MeV)", ylabel = "$\sigma$(mb)")
    Plots.plt.fill_between(xs_sim.KE, 0.8 * xs, 1.2 * xs, alpha = 0.5, color = "k", label = "Geant 4 $\pm$ 20%")

    x = xs_sim.KE
    x_sample = x[indices[process]]
    y_sample = xs[indices[process]]

    if plot is True:
        c = Plots.plt.cm.autumn_r(np.linspace(0, 1, 5))
        step = (shape_factors[process][1] - shape_factors[process][0]) / 4
        splines = []
        for i in range(5):
            spline = CubicSpline(x_sample, (shape_factors[process][0] + (i * step))* y_sample)
            splines.append(spline)
            Plots.Plot(xs_sim.KE, spline(xs_sim.KE), linestyle = (0, (5, 6)), color = c[i], label = i, newFigure = False)
    return splines


def CountsFractionalError(results, true_counts, model):
    fe_err = []
    fe = []

    for s in results:
        true_counts_arr = np.array(list(true_counts[s].values()))
        post_fit_pred = cross_section.cabinetry.model_utils.prediction(model, fit_results = results[s], label = "post-fit")

        if any([c["name"] == "mean_track_score" for c in post_fit_pred.model.spec["channels"]]):
            KE_int_prediction = cross_section.RegionFit.SliceModelPrediction(post_fit_pred, slice(-1), "KE_int_postfit") # exclude the channel which is the mean track score
        else:
            KE_int_prediction = cross_section.RegionFit.SliceModelPrediction(post_fit_pred, slice(0, len(post_fit_pred.model_yields)), "KE_int_postfit")

        true_counts_pred_res_err = cross_section.quadsum(np.array(KE_int_prediction.total_stdev_model_bins)[:, :-1], 0)
        fe_err.append(cross_section.nandiv(true_counts_pred_res_err, true_counts_arr))

        pred_counts = np.sum(np.array(KE_int_prediction.model_yields), 0)
        fe.append(np.array([cross_section.nandiv(pred_counts[i] - v, v) for i, v in enumerate(true_counts_arr)]))
    fe_err = np.swapaxes(np.array(fe_err), 0, 1)
    fe = np.swapaxes(np.array(fe), 0, 1)
    return fe, fe_err


def CreateConfigShapeTest(data_config, modified_PDFs : dict):
    cfg = {k : v for k, v in data_config.items()}
    cfg["modified_PDFs"] = modified_PDFs
    return cfg


def CreateModPDFDict(KE : np.array, name : str, xs : np.array) -> dict[np.array]:
    return {
        "KE" : KE,
        name : xs
    }


def ShapeTest(directory, data_config, method, shape_param_factors, spline_shape_param_factors, xs_sim, model, toy_template, mean_track_score_bins, energy_slices):
    folder = {
        'absorption': "abs",
        'quasielastic': "quasi",
        'charge_exchange': "cex",
        'double_charge_exchange': "dcex",
        'pion_production': "pip"
    }
    for target in folder:
        print(target)
        results = {}
        true_counts = {}
        expected_mus = {}

        # with Plots.PlotBook(f"{directory}fit_results_{folder[target]}.pdf", False) as pdf:
        if method == "function":
            shape_params = CreateShapeParams(xs_sim, target, shape_param_factors[target][0], shape_param_factors[target][1])
        elif method == "spline":
            shape_params = CreateShapeParamsSpline(xs_sim, target, spline_point_index, spline_shape_param_factors)
        else:
            raise Exception(f"{method} not a valid type")

        # pdf.Save()
        # PlotHistShapeTest(target, data_config, shape_params, method, pdf)
        for i, p in enumerate(shape_params):
            print(f"shape: {i}")
            if i == 2:
                config = CreateConfigShapeTest(data_config, modified_PDFs = None)
            elif method == "function":
                config = CreateConfigShapeTest(data_config, modified_PDFs = CreateModPDFDict(xs_sim.KE, target, 1000 * lognormal_gaussian_exp(xs_sim.KE/1000, *p)))
            elif method == "spline":
                config = CreateConfigShapeTest(data_config, modified_PDFs = CreateModPDFDict(xs_sim.KE, target, p(xs_sim.KE)))
            else:
                raise Exception(f"{method} not a valid type")
            results[i], true_counts[i], expected_mus[i] = ModifiedConfigTest(config, energy_slices, model, toy_template, mean_track_score_bins)
            # print(config)
            # clear_output()
            # print(target)

        cross_section.SaveObject(f"{directory}fit_results_{folder[target]}.dill", {"results" : results, "true_counts" : true_counts, "expected_mus" : expected_mus}) # keep results for future reference
        # PlotCrossCheckResults(f"{target} shape", model, toy_template, results, true_counts, energy_overflow, pdf)
        Plots.plt.close("all")
    return


def PlotHistShapeTest(target, data_config : dict, shape_params : dict, type : str, xs_sim : cross_section.GeantCrossSections, energy_overflow, energy_slice, book : Plots.PlotBook = Plots.PlotBook.null):
    if type == "function":
        config_high = CreateConfigShapeTest(**data_config, modified_PDFs = CreateModPDFDict(xs_sim.KE, target, 1000 * lognormal_gaussian_exp(xs_sim.KE/1000, *shape_params[-1])))
        config_low = CreateConfigShapeTest(**data_config, modified_PDFs = CreateModPDFDict(xs_sim.KE, target, 1000 * lognormal_gaussian_exp(xs_sim.KE/1000, *shape_params[0])))
    elif type == "spline":
        config_high = CreateConfigShapeTest(**data_config, modified_PDFs = CreateModPDFDict(xs_sim.KE, target, shape_params[1](xs_sim.KE)))
        config_low = CreateConfigShapeTest(**data_config, modified_PDFs = CreateModPDFDict(xs_sim.KE, target, shape_params[0](xs_sim.KE)))
    else:
        raise Exception("not a valid type")
    toys = {
    "nominal" : cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(df = cex_toy_generator.main(cex_toy_generator.ResolveConfig(data_config)))),
    "high" : cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(df = cex_toy_generator.main(cex_toy_generator.ResolveConfig(config_high)))),
    "low" : cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(df = cex_toy_generator.main(cex_toy_generator.ResolveConfig(config_low))))
    }

    def ratio_err(a, b):
        return abs((a/b) * np.sqrt((np.sqrt(a)/a)**2 + (np.sqrt(b)/b)**2))

    def RatioPlot(data):
        for _, i in Plots.IterMultiPlot(data):
            Plots.Plot(energy_overflow, data[i]["high"] / data[i]["nominal"], yerr = ratio_err(data[i]["high"], data[i]["nominal"]), color = colours["high"], label = "high", title = i, xlabel = "$N_{int}$ (MeV)", ylabel = "ratio", marker = "o", newFigure = False, linestyle = "")
            Plots.Plot(energy_overflow, data[i]["low"] / data[i]["nominal"], yerr = ratio_err(data[i]["low"], data[i]["nominal"]), color = colours["low"], label = "low", title = i, xlabel = "$N_{int}$ (MeV)", ylabel = "ratio", marker = "o", newFigure = False, linestyle = "")
            Plots.plt.axhline(1, color = "k")
            Plots.plt.ylim(0, 2)
        book.Save()
        return

    colours = {"nominal" : "k", "high" : "C6", "low" : "C0"}

    n_interact_process = {}
    for _, i in Plots.IterMultiPlot(toys["nominal"].exclusive_process):
        tmp = {}
        for k, v in toys.items():
            tmp[k] = v.NInteract(energy_slice, v.exclusive_process[i])
            Plots.Plot(energy_overflow, tmp[k] / sum(tmp[k]), yerr = np.sqrt(tmp[k]) / sum(tmp[k]), label = k, title = i, color = colours[k], xlabel = "$N_{int}$ (MeV)", ylabel = "fractional counts", style = "step", newFigure = False)
        n_interact_process[i] = tmp
    book.Save()

    RatioPlot(n_interact_process)

    n_interact_region = {}
    for _, i in Plots.IterMultiPlot(toys["nominal"].regions):
        tmp = {}
        for k, v in toys.items():
            tmp[k] = v.NInteract(energy_slice, v.regions[i])
            Plots.Plot(energy_overflow, tmp[k] / sum(tmp[k]), yerr = np.sqrt(tmp[k]) / sum(tmp[k]), label = k, color = colours[k], title = f"reco region : {i}", xlabel = "$N_{int}$ (MeV)", ylabel = "fractional counts", style = "step", newFigure = False)
        n_interact_region[i] = tmp
    book.Save()

    RatioPlot(n_interact_region)
    return


def PlotCrossCheckResults(xlabel, model : cross_section.pyhf.Model, toy_template : cross_section.Toy, results, true_counts, energy_overflow : np.ndarray, pdf : Plots.PlotBook = Plots.PlotBook.null):
    true_counts_all = {}
    for t in true_counts:
        true_counts_all[t] = {k : np.sum(v) for k, v in true_counts[t].items()}

    scale_factors = {k : sum(true_counts_all[k].values()) / sum(toy_template.df.total_inelastic) for k in true_counts_all}
    x = list(range(len(results)))

    mu = []
    mu_err = []
    for k in results:
        mu.append(results[k].bestfit[0:4] / scale_factors[k])
        mu_err.append(results[k].uncertainty[0:4] / scale_factors[k])
    mu = np.array(mu)
    mu_err = np.array(mu_err)

    process_map = {0 : "abs", 1 : "cex", 2 : "spip", 3 : "pip"}

    # Plot the fit value for each scale factor 
    Plots.plt.figure()
    for i in range(4):
        Plots.Plot(x, mu[:, i], yerr = mu_err[:, i], newFigure = False, label = f"$\mu_{{{process_map[i]}}}$", marker = "o", ylabel = "fit value", color = list(region_colours.values())[i], linestyle = "")
    Plots.plt.xticks(ticks = x, labels = results.keys())
    Plots.plt.xlabel(xlabel)
    pdf.Save()

    # same as above, in separate plots
    for i in Plots.MultiPlot(4):
        Plots.Plot(x, mu[:, i], yerr = mu_err[:, i], newFigure = False, title = f"$\mu_{{{process_map[i]}}}$", marker = "o", xlabel = xlabel, ylabel = "fit value", color = list(region_colours.values())[i], linestyle = "")
        Plots.plt.xticks(ticks = x, labels = results.keys())
    pdf.Save()

    true_counts_all = pd.DataFrame(true_counts_all)
    fe, fe_err = CountsFractionalError(results, true_counts, model)
    tc_arr = np.swapaxes(np.array([np.array(list(v.values())) for v in true_counts.values()]), 0, 1)
    fractional_error = np.nansum(fe * tc_arr, 2) / np.sum(tc_arr, 2)
    fractional_error_unc = cross_section.nanquadsum(fe_err * tc_arr, 2) / np.sum(tc_arr, 2)

    norms = list(true_counts.keys())
    proc = list(list(true_counts.values())[0].keys())

    fractional_error = pd.DataFrame(fractional_error, columns = norms, index = proc)
    fractional_error_unc = pd.DataFrame(fractional_error_unc, columns = norms, index = proc)

    prefit_counts = [model.main_model.expected_data(np.array(model.config.suggested_init()) * scale_factors[k], return_by_sample = True) for k in scale_factors]

    if "mean_track_score" in model.config.channels:
        prefit_counts = [i[:, :-model.config.channel_nbins["mean_track_score"]] for i in prefit_counts]
    prefit_counts = np.array([np.sum(i, 1) for i in prefit_counts]).T


    # plot true process residual
    for n, i in Plots.IterMultiPlot(true_counts_all.index):
        Plots.Plot(x, true_counts_all.loc[i] * fractional_error.loc[i], yerr = true_counts_all.loc[i] * fractional_error_unc.loc[i], title = f"$N_{{{process_map[n]}}}^{{pred}}$", xlabel = xlabel, ylabel = "residual", linestyle = "", marker = "o", color = list(region_colours.values())[n], newFigure = False)
        Plots.plt.axhline(0, color = "black", linestyle = "--")
        Plots.plt.xticks(ticks = x, labels = results.keys())
    pdf.Save()

    # plot true process fractional error
    for n, i in Plots.IterMultiPlot(true_counts_all.index):
        Plots.Plot(x, fractional_error.loc[i], yerr = fractional_error_unc.loc[i], title = f"measured $N_{{{process_map[n]}}}$", xlabel = xlabel, ylabel = "fractional error", linestyle = "", marker = "o", color = list(region_colours.values())[n], newFigure = False)
        Plots.plt.axhline(0, color = "black", linestyle = "--")
        Plots.plt.xticks(ticks = x, labels = results.keys())
    pdf.Save()

    # plot true process fractional error
    for n, i in Plots.IterMultiPlot(true_counts_all.index):
        Plots.Plot(x, fractional_error.loc[i], yerr = fractional_error_unc.loc[i], title = f"$N_{{{process_map[n]}}}$", xlabel = xlabel, ylabel = "fractional error", linestyle = "", marker = "o", color = list(region_colours.values())[n], label = "measured", newFigure = False)
        Plots.Plot(x, (prefit_counts[n] - true_counts_all.loc[i]) / true_counts_all.loc[i], title = f"$N_{{{process_map[n]}}}$", linestyle = "", marker = "o", color = "k", label = "prefit", newFigure = False)
        Plots.plt.axhline(0, color = "black", linestyle = "--")
        Plots.plt.xticks(ticks = x, labels = results.keys())
    pdf.Save()

    # plot true process fractional error
    Plots.plt.figure()
    for n, i in enumerate(true_counts_all.index):
        Plots.Plot(x, fractional_error.loc[i], yerr = fractional_error_unc.loc[i], label = f"${process_map[n]}$", xlabel = xlabel, ylabel = "fractional error", linestyle = "", marker = "o", color = list(region_colours.values())[n], newFigure = False)
        Plots.plt.axhline(0, color = "black", linestyle = "--")
        Plots.plt.ylim(-1, 1)
    Plots.plt.xticks(ticks = x, labels = results.keys())
    pdf.Save()

    Plots.plt.figure()
    for i, s in Plots.IterMultiPlot(proc):
        for l, y, e in zip(results, fe[i], fe_err[i]):
            Plots.Plot(energy_overflow, y * true_counts[l][s], yerr = e * true_counts[l][s], label = l, ylabel = "residual", newFigure = False)
        Plots.plt.axhline(0, color = "black", linestyle = "--")
        Plots.plt.xlabel("$KE$ (MeV)")
        Plots.plt.title(f"$N_{{{process_map[i]}}}$")
        Plots.plt.legend(title = xlabel)
    pdf.Save()

    Plots.plt.figure()
    for i, s in Plots.IterMultiPlot(proc):
        for l, y, e in zip(results, fe[i], fe_err[i]):
            Plots.Plot(energy_overflow, y, yerr = e, label = l, ylabel = "fractional error", newFigure = False)
        Plots.plt.axhline(0, color = "black", linestyle = "--")
        Plots.plt.xlabel("$KE$ (MeV)")
        Plots.plt.title(f"$N_{{{process_map[i]}}}$")
        Plots.plt.legend(title = xlabel)
    pdf.Save()
    return


def BSTrueCounts(fit_results : dict, signal_process : str) -> tuple[np.ndarray]:
    b_tc = []
    s_tc = []
    for r in fit_results["results"]:
        b_tc.append(sum([v for k, v in fit_results["true_counts"][r].items() if k != signal_process]))
        s_tc.append(fit_results["true_counts"][r][signal_process])
    return np.array(s_tc), np.array(b_tc)


def BSCounts(model : cross_section.pyhf.Model, toy_template, fit_results, signal_process : str) -> tuple[tuple[np.ndarray, np.ndarray]]:
    b_c = []
    b_c_err = []
    s_c = []
    s_c_err = []

    for r in fit_results["results"]:
        bkg, bkg_var = cross_section.RegionFit.EstimateBackground(fit_results["results"][r], model, toy_template, signal_process)

        b_c.append(bkg)
        b_c_err.append(np.sqrt(bkg_var))

        n_total = sum([v for v in fit_results["true_counts"][r].values()])
        s_c.append(n_total - bkg)
        s_c_err.append(np.sqrt(n_total + bkg_var))
    return (np.array(s_c), np.array(s_c_err)), (np.array(b_c), np.array(b_c_err))


def BSFractionalError(s_c, s_c_err, b_c, b_c_err, s_tc, b_tc):
    s_r = s_c - s_tc
    b_r = b_c - b_tc

    s_fe = cross_section.nandiv(s_r, s_tc)
    s_fe_err = cross_section.nandiv(s_c_err, s_tc)
    b_fe = cross_section.nandiv(b_r, b_tc)
    b_fe_err = cross_section.nandiv(b_c_err, b_tc)

    return (s_fe, s_fe_err), (b_fe, b_fe_err)


def BSFractionalErrorTotal(s_c, s_c_err, b_c, b_c_err, s_tc, b_tc):
    s_r = s_c - s_tc
    b_r = b_c - b_tc

    s_r_total = np.sum(s_r, 1)
    b_r_total = np.sum(b_r, 1)

    s_fe_total = s_r_total / np.sum(s_tc, 1)
    b_fe_total = b_r_total / np.sum(b_tc, 1)

    s_fe_err_total = cross_section.quadsum(s_c_err, 1) / np.sum(s_tc, 1)
    b_fe_err_total = cross_section.quadsum(b_c_err, 1) / np.sum(b_tc, 1)    
    return (s_fe_total, s_fe_err_total), (b_fe_total, b_fe_err_total)


def BSPerformanceCheck(directory : str, signal_process : str, model : cross_section.pyhf.Model, template : cross_section.Toy, energy_overflow : np.ndarray, ylims : tuple = None, book : Plots.PlotBook = Plots.PlotBook.null):
    results_files = [i for i in cross_section.os.listdir(directory) if "dill" in i]

    s_fe = [None]*len(results_files)
    s_fe_err = [None]*len(results_files)
    b_fe = [None]*len(results_files)
    b_fe_err = [None]*len(results_files)

    s_fe_total = [None]*len(results_files)
    b_fe_total = [None]*len(results_files)
    s_fe_err_total = [None]*len(results_files)
    b_fe_err_total = [None]*len(results_files)

    for i, f in enumerate(results_files):
        fit_results = cross_section.LoadObject(directory + f)
        target = [target_map[k] for k in target_map if k in f][0]

        (s_c, s_c_err), (b_c, b_c_err) = BSCounts(model, template, fit_results, signal_process)
        s_tc, b_tc = BSTrueCounts(fit_results, signal_process)

        (s_fe[i], s_fe_err[i]), (b_fe[i], b_fe_err[i]) = BSFractionalError(s_c, s_c_err, b_c, b_c_err, s_tc, b_tc)
        (s_fe_total[i], s_fe_err_total[i]), (b_fe_total[i], b_fe_err_total[i]) = BSFractionalErrorTotal(s_c, s_c_err, b_c, b_c_err, s_tc, b_tc)

    fig_b = Plots.plt.figure(1)
    fig_s = Plots.plt.figure(2)
    for i, f in enumerate(results_files):
        fit_results = cross_section.LoadObject(directory + f)
        target = [target_map[k] for k in target_map if k in f][0]

        x = range(len(list(fit_results["results"].keys())))

        Plots.plt.figure(fig_b)
        Plots.Plot(x, b_fe_total[i], yerr = b_fe_err_total[i], marker = "o", xlabel = "normalisation", ylabel = "fractional error", title = "total background counts", label = cross_section.remove_(target), newFigure = False)
        Plots.plt.xticks(ticks = x, labels = fit_results["results"].keys())
        Plots.plt.tight_layout()
        if ylims:
            Plots.plt.ylim(ylims)

        Plots.plt.figure(fig_s)
        Plots.Plot(x, s_fe_total[i], yerr = s_fe_err_total[i], marker = "o", xlabel = "normalisation", ylabel = "fractional error", title = "total background subtracted counts", label = cross_section.remove_(target), newFigure = False)
        Plots.plt.xticks(ticks = x, labels = fit_results["results"].keys())
        Plots.plt.tight_layout()
        if ylims:
            Plots.plt.ylim(ylims)

    Plots.plt.figure(fig_b)
    book.Save()
    Plots.plt.figure(fig_s)
    book.Save()

    for i, f in enumerate(results_files):
        fit_results = cross_section.LoadObject(directory + f)
        target = [target_map[k] for k in target_map if k in f][0]

        labels = list(fit_results["results"].keys())

        Plots.plt.figure()
        for j, l in enumerate(labels):
            Plots.Plot(energy_overflow, s_fe[i][j], yerr = s_fe_err[i][j], label = l, xlabel = "$KE$ (MeV)", ylabel = "fractional error", title = "background subtracted counts", newFigure = False)
        Plots.plt.legend(title = f"{target} normalisation")
        book.Save()

        Plots.plt.figure()
        for j, l in enumerate(labels):
            Plots.Plot(energy_overflow, b_fe[i][j], yerr = b_fe_err[i][j], label = l, xlabel = "$KE$ (MeV)", ylabel = "fractional error", title = "background counts", newFigure = False)
        Plots.plt.legend(title = f"{target} normalisation")
        book.Save()
    return


def BackgroundSubtractionSummary(directory, model, signal_process, toy_template, test):
    results_files = [i for i in cross_section.os.listdir(directory) if "dill" in i]
    results = {[target_map[t] for t in target_map if t in f][0] : cross_section.LoadObject(directory + f) for f in results_files}

    s_fe_total_max = {}
    b_fe_total_max = {}

    s_fe_max = {}
    b_fe_max = {}
    for r in results:
        v = list(results[r]["results"].keys())

        (s_c, s_c_err), (b_c, b_c_err) = BSCounts(model, toy_template, results[r], signal_process)
        s_tc, b_tc = BSTrueCounts(results[r], "charge_exchange")
        (s_fe_total, s_fe_total_err), (b_fe_total, b_fe_total_err) = BSFractionalErrorTotal(s_c, s_c_err, b_c, b_c_err, s_tc, b_tc)
        (s_fe, s_fe_err), (b_fe, b_fe_err) = BSFractionalError(s_c, s_c_err, b_c, b_c_err, s_tc, b_tc)

        if test == "normalisation":
            ind = [v.index(0.8), v.index(1.2)]
            s_fe = s_fe[ind]
            b_fe = b_fe[ind]
            s_fe_err = s_fe_err[ind]
            b_fe_err = b_fe_err[ind]
            s_fe_total = s_fe_total[ind]
            b_fe_total = b_fe_total[ind]
            s_fe_total_err = s_fe_total_err[ind]
            b_fe_total_err = b_fe_total_err[ind]

        s_fe_total_max[r] = (max(abs(s_fe_total)), s_fe_total_err[np.argmax(abs(s_fe_total))])
        b_fe_total_max[r] = (max(abs(b_fe_total)), b_fe_total_err[np.argmax(abs(b_fe_total))])
        s_fe_max[r] = (np.max(abs(s_fe), 0), np.where(np.argmax(abs(s_fe), 0) == 0, s_fe_err[0], s_fe_err[1]))
        b_fe_max[r] = (np.max(abs(b_fe), 0), np.where(np.argmax(abs(b_fe), 0) == 0, b_fe_err[0], b_fe_err[1]))
    return s_fe_max, b_fe_max, s_fe_total_max, b_fe_total_max


def PredictedCountsSummary(directory : str, model : cross_section.pyhf.Model, test : str):
    results_files = [i for i in cross_section.os.listdir(directory) if "dill" in i]
    results = {[target_map[t] for t in target_map if t in f][0] : cross_section.LoadObject(directory + f) for f in results_files}

    n_fe_total_max = {}
    n_fe_max = {}
    for r in results:
        v = list(results[r]["results"].keys())

        fe, fe_err = CountsFractionalError(results[r]["results"], results[r]["true_counts"], model)
        tc_arr = np.swapaxes(np.array([np.array(list(v.values())) for v in results[r]["true_counts"].values()]), 0, 1)
        fractional_error = np.nansum(fe * tc_arr, 2) / np.sum(tc_arr, 2)
        fractional_error_unc = cross_section.nanquadsum(fe_err * tc_arr, 2) / np.sum(tc_arr, 2)

        if test == "normalisation":
            ind = [v.index(0.8), v.index(1.2)]
            fractional_error = fractional_error[:, ind]
            fractional_error_unc = fractional_error_unc[:, ind]
            fe = fe[:, ind]
            fe_err = fe_err[:, ind]

        n_fe_total_max[r] = (
            np.max(abs(fractional_error), 1),
            np.where(np.argmax(abs(fractional_error), 1) == 0, fractional_error_unc[:, 0], fractional_error_unc[:, 1])
            )

        fe = np.nan_to_num(fe, posinf = 0, nan = 0, neginf = 0)
        fe_err = np.nan_to_num(fe_err, posinf = 0, nan = 0, neginf = 0)        
        n_fe_max[r] = (
            np.max(abs(fe), 1),
            np.where(np.argmax(abs(fe), 1) == 0, fe_err[:, 0], fe_err[:, 1])
        )
    return n_fe_max, n_fe_total_max


def CreateSummaryTables(total_summaries, row_names):
    table_fe = pd.DataFrame({k : v[0] for k, v in total_summaries.items()}, index = row_names)
    table_err = pd.DataFrame({k : v[1] for k, v in total_summaries.items()}, index = row_names)

    table_str = {}
    for r, i, j in zip(row_names, table_fe.values, table_err.values):
        row = {}
        for c, v, e in zip(table_fe.columns, i, j):
            e_s = f"{100 * e:.1g}"
            v_s = str(round(100 * v, len(e_s.split(".")[-1])))
            row[cross_section.remove_(c)] = f"{v_s} $\pm$ {e_s}"
        table_str[cross_section.remove_(r) + "(\%)"] = row
    return table_fe, table_err, pd.DataFrame(table_str)


def SaveSummaryTables(directory : str, tables : tuple[pd.DataFrame], name : str):
    names = ["fe", "unc", "fmt"]
    for n, t in zip(names, tables):
        t.style.to_latex(f"{directory}table_{name}_{n}.tex")
    return


def Summary(directory : str, test : str, signal_process : str, model : cross_section.pyhf.Model, energy_overflow : np.ndarray, template : cross_section.Toy, book : Plots.PlotBook = Plots.PlotBook.null):
    n_fe_max, n_fe_total_max = PredictedCountsSummary(directory, model, test)

    indices = ["absorption", "charge_exchange", "single_pion_production", "pion_production"]
    xlabel = "$KE$ (MeV)"

    tables_n = CreateSummaryTables(n_fe_total_max, indices)
    SaveSummaryTables(directory, tables_n, "processes")
    print(tables_n[2])

    for i, p in Plots.IterMultiPlot(indices):
        for j, t in enumerate(n_fe_max):
            y = n_fe_max[t][0]
            err = n_fe_max[t][1]
            Plots.Plot(energy_overflow, y[i], yerr = err[i], color = f"C{2*j}", label = cross_section.remove_(t), ylabel = "fractional error in fitted counts", xlabel = xlabel, title = f"process : {cross_section.remove_(p)}", newFigure = False)
            Plots.plt.legend(title = f"{test}s test")
    book.Save()

    s_fe_max, b_fe_max, s_fe_total_max, b_fe_total_max = BackgroundSubtractionSummary(directory, model, signal_process, template, test)

    tables_s = CreateSummaryTables(s_fe_total_max, ["background subtracted counts"])
    SaveSummaryTables(directory, tables_s, "signal")
    print(tables_s[2])

    tables_b = CreateSummaryTables(b_fe_total_max, ["background counts"])
    SaveSummaryTables(directory, tables_b, "background")
    print(tables_b[2])

    Plots.plt.figure()
    for i, r in enumerate(s_fe_max):
        Plots.Plot(energy_overflow, s_fe_max[r][0], yerr = s_fe_max[r][1], color = f"C{2*i}", label = cross_section.remove_(r), xlabel = xlabel, ylabel = "fractional error", title = "background subtracted counts", newFigure = False)
        Plots.plt.legend(title = f"{test} test")
    book.Save()
    Plots.plt.figure()
    for i, r in enumerate(b_fe_max):
        Plots.Plot(energy_overflow, b_fe_max[r][0], yerr = b_fe_max[r][1], color = f"C{2*i}", label = cross_section.remove_(r), xlabel = xlabel, ylabel = "fractional error", title = "background counts", newFigure = False)
        Plots.plt.legend(title = f"{test} test")
    book.Save()
    return

@cross_section.timer
def main(args : cross_section.argparse.Namespace):
    cross_section.SetPlotStyle(extend_colors = True)
    args.template = cross_section.Toy(file = args.template)

    energy_slices = cross_section.Slices(args.slice_width, min(args.energy_range), max(args.energy_range), True)

    energy_overflow = np.insert(energy_slices.pos, 0, energy_slices.max_pos + energy_slices.width) # xlabels for KE, which has an overflow bin (max energy + energy bin width) 
    energy_bins = np.sort(energy_overflow)

    mean_track_score_bins = np.linspace(0, 1, 21, True)

    models = {"normal" : cross_section.RegionFit.CreateModel(cross_section.AnalysisInput.CreateAnalysisInputToy(args.template), energy_slices, None, False, None, False)}
    models["track_score"], templates_energy, tempalates_mean_track_score = cross_section.RegionFit.CreateModel(cross_section.AnalysisInput.CreateAnalysisInputToy(args.template), energy_slices, mean_track_score_bins, True, None, False, False)

    os.makedirs(args.out, exist_ok = True)

    if args.toy_data_config:
        print("Running tests")

        for m in models:
            os.makedirs(args.out + f"normalisation_test_{m}/", exist_ok = True)
            NormalisationTest(
                args.out + f"normalisation_test_{m}/",
                args.toy_data_config, models[m],
                args.template,
                energy_slices,
                mean_track_score_bins if m == "track_score" else None)

            xs_sim = cross_section.GeantCrossSections(energy_range = [0, max(args.energy_range)])

            os.makedirs(args.out + f"shape_test_{m}/", exist_ok = True)
            ShapeTest(
                args.out + f"shape_test_{m}/",
                args.toy_data_config,
                args.shape_gen,
                shape_param_factors,
                spline_shape_param_factors, 
                xs_sim,
                models[m],
                args.template,
                mean_track_score_bins if m == "track_score" else None,
                energy_slices)


    if args.workdir:
        print("Making test results")

        test = ["shape", "normalisation"] # use when shape test is done 
        for t in test:
            for m in models:
                directory = args.workdir + f"{t}_test_{m}/"
                results_files = [i for i in cross_section.os.listdir(directory) if "dill" in i]

                for f in results_files:
                    fit_result = cross_section.LoadObject(directory+f)
                    target = [target_map[k] for k in target_map if k in f][0]
                    with Plots.PlotBook(directory+f.split(".")[0]+".pdf", True) as pdf:
                        PlotCrossCheckResults(f"{target} {t}", models[m], args.template, fit_result["results"], fit_result["true_counts"], energy_overflow, pdf)
                    Plots.plt.close("all")
            
                with Plots.PlotBook(f"{directory}background_sub_fractional_err.pdf", True) as book:
                    BSPerformanceCheck(directory, args.signal_process, models[m], args.template, energy_overflow, [-0.04, 0.04], book)
                Plots.plt.close("all")

                with Plots.PlotBook(f"{directory}summary_plots.pdf", True) as book:
                    Summary(directory, t, args.signal_process, models[m], energy_overflow, args.template, book)
                Plots.plt.close("all")

    return


if __name__ == "__main__":
    parser = cross_section.argparse.ArgumentParser("app which performs cross checks for the region fit using toys.")

    parser.add_argument("--template", "-t", dest = "template", type = str, help = "toy template hdf5 file", required = True)

    parser.add_argument("--toy_data_config", "-c", dest = "toy_data_config", type = str, help = "json config for toy data", required = False)

    parser.add_argument("--shape_gen", "-g", dest = "shape_gen", type = str, choices = ["spline", "function"], help = "method used to generate different cross section shapes for shape test", required = True)

    parser.add_argument("--workdir", "-w", dest = "workdir", type = str, help = "work directory which contains output from this application, use this to remake plots without running all the tests again", required = False)
    parser.add_argument("--signal_process", dest = "signal_process", type = str, help = "signal process for background subtraction")

    parser.add_argument("--energy_range", "-e", dest = "energy_range", type = int, nargs = 2, help = "enegy range for slices", required = True)
    parser.add_argument("--slice_width", "-W", dest = "slice_width", type = float, help = "energy slice width (MeV)", required = True)

    parser.add_argument("--step", "-s", dest = "step", type = float, help = "step size for toy, if provided will override toy data config.")
    parser.add_argument("--events", "-n", dest = "events", type = float, help = "events for toy, if provided will override toy data config.")
    parser.add_argument("--seed", dest = "seed", type = int, help = "seed for toy, if provided will override toy data config.")
    cross_section.ApplicationArguments.Output(parser, "region_fit_studies")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())

    if (not args.toy_data_config) and (not args.workdir):
        raise Exception("--toy_data_config or --workdirs or both must be supplied")

    args.energy_range = sorted(args.energy_range)
    if args.events: args.events = int(args.events)

    if args.toy_data_config:
        args.toy_data_config = cross_section.LoadConfiguration(args.toy_data_config)
        OverrideConfig(args)

    if args.workdir:
        if args.workdir[-1] != "/": args.workdir = args.workdir + "/"
        if not args.signal_process:
            raise Exception("--signal_process must be supplied if --workdirs is supplied")

    print(vars(args))
    main(args)