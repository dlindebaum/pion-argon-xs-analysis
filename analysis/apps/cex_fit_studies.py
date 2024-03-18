#!/usr/bin/env python3
"""
Created on: 17/01/2024 13:46

Author: Shyam Bhuller

Description: app which performs cross checks for the region fit using toys.
"""
import os

import itertools
import numpy as np
import pandas as pd

from rich import print, rule
from scipy.ndimage import gaussian_filter1d


from python.analysis import cross_section, Plots
from python.analysis.Master import DictToHDF5, ReadHDF5
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
    "_pip" : 'pion_production',
    "_spip" : "single_pion_production"
}

folder = {
    'absorption': "abs",
    'charge_exchange': "cex",
    'single_pion_production': "spip",
    'pion_production': "pip"
}

process_map = {0 : "abs", 1 : "cex", 2 : "spip", 3 : "pip"}


def OverrideConfig(args):
    for a in ["step", "events", "seed"]:
        if getattr(args, a):
            args.toy_data_config[a] = getattr(args, a)
    return


def CreateConfigNormalisation(scales : dict, data_config : dict) -> dict:
    cfg = {k : v for k, v in data_config.items()}
    cfg["pdf_scale_factors"] = scales
    return cfg


def ModifiedConfigTest(config : dict, energy_slice : cross_section.Slices, model : cross_section.pyhf.Model, toy_template : cross_section.AnalysisInput, mean_track_score_bins : np.array = None, single_bin : np.array = False) -> tuple[dict]:
    toy_alt_pdf = cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(df = cex_toy_generator.run(config)))
    
    obs = cross_section.RegionFit.GenerateObservations(toy_alt_pdf, energy_slice, mean_track_score_bins, model, False, single_bin)
    fit_result = cross_section.RegionFit.Fit(obs, model, None, verbose = False, tolerance = 0.1) # relax the tolerance to allow fit to converge for a large parameter space

    true_process_counts = {}
    for v in toy_alt_pdf.exclusive_process:
        true_process_counts[v] = toy_alt_pdf.NInteract(energy_slice, toy_alt_pdf.exclusive_process[v])

    expected_mus = [sum(toy_alt_pdf.exclusive_process[r]) / sum(toy_template.exclusive_process[r]) for r in toy_alt_pdf.exclusive_process]

    return fit_result, true_process_counts, expected_mus


def NormalisationTest(directory : str, data_config : dict, model : cross_section.pyhf.Model, toy_template : cross_section.AnalysisInput, energy_slice : cross_section.Slices, mean_track_score_bins : np.ndarray, single_bin : bool):
    for target in folder:
        print(target)

        results = {}
        true_counts = {}
        expected_mus = {}
    
        scales = {k : 1 for k in ['absorption', 'quasielastic', 'charge_exchange', 'double_charge_exchange', 'pion_production']}
        for i in [0.8, 0.9, 1, 1.1, 1.2]:
            print(rule.Rule(f"process : {target} | normalisation : {i}"))
            if i == 1:
                config = data_config
            else:
                if target == "single_pion_production":
                    # treat these as the same process
                    scales["quasielastic"] = i
                    scales["double_charge_exchange"] = i
                else:
                    scales[target] = i
                config = CreateConfigNormalisation(scales, data_config)
            results[i], true_counts[i], expected_mus[i] = ModifiedConfigTest(config, energy_slice, model, toy_template, mean_track_score_bins, single_bin)

        cross_section.SaveObject(f"{directory}fit_results_{folder[target]}.dill", {"results" : results, "true_counts" : true_counts, "expected_mus" : expected_mus})
    return


def FitSimulationCurve(xs_sim, sampling_factor : int, process : str, function : cross_section.Fitting.FitFunction, plot : bool = False):
    x = xs_sim.KE[::sampling_factor]/1000
    y = getattr(xs_sim, process)[::sampling_factor]/1000

    if plot is True: Plots.plt.figure()
    results = cross_section.Fitting.Fit(x, y, None, function, method = "dogbox", plot = plot, xlabel = "$KE$(GeV)", ylabel = "$\sigma$ (b)", title = process)
    return results


def CountsFractionalError(results, true_counts, model, single_bin = False):
    fe_err = []
    fe = []

    mean_track_score = any([c["name"] == "mean_track_score" for c in model.spec["channels"]])

    for s in results:
        true_counts_arr = np.array(list(true_counts[s].values()))
        post_fit_pred = cross_section.cabinetry.model_utils.prediction(model, fit_results = results[s], label = "post-fit")

        if mean_track_score:
            KE_int_prediction = cross_section.RegionFit.SliceModelPrediction(post_fit_pred, slice(-1), "KE_int_postfit") # exclude the channel which is the mean track score
        else:
            KE_int_prediction = cross_section.RegionFit.SliceModelPrediction(post_fit_pred, slice(0, len(post_fit_pred.model_yields)), "KE_int_postfit")

        pred_counts_err = cross_section.quadsum(np.array(KE_int_prediction.total_stdev_model_bins)[:, :-1], 0)
    
        if single_bin is True:
            t = np.sum(true_counts_arr, 1)
            pred_counts_err = pred_counts_err.flatten()
        else:
            t = true_counts_arr

        fe_err.append(cross_section.nandiv(pred_counts_err, t))

        pred_counts = np.sum(np.array(KE_int_prediction.model_yields), 0)
        if single_bin:
            pred_counts = pred_counts.flatten()

        fe.append(np.array([cross_section.nandiv(pred_counts[i] - v, v) for i, v in enumerate(t)]))
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


def SmoothStep(x : np.ndarray, high : float, low : float, split : float, smooth_amount : float):
    step = np.where(x >= split, high, low)
    if smooth_amount == 0:
        return step
    else:
        return gaussian_filter1d(step, smooth_amount)


def ShapeTestNew(directory : str, data_config : dict, model : cross_section.pyhf.Model, toy_template : cross_section.AnalysisInput, mean_track_score_bins : np.array, xs_sim : cross_section.GeantCrossSections, energy_slices : cross_section.Slices, single_bin):
    n = [0.8, 1, 1.2]
    alpha = [500, 1000]
    x0 = [500, 1000, 1500]
    perms = []
    for perm in itertools.product(n, n, alpha, x0):
        if perm[0] == perm[1]:
            continue
        else:
            perms.append(perm)


    perms.append("null")

    print(f"number of tests per process : {len(perms)=}")

    for target in folder:
        print(target)

        results = {}
        true_counts = {}
        expected_mus = {}
        for e, i in enumerate(perms):
            print(rule.Rule(f"process: {target} | iteration : {e} | params : {i}"))
            if i == "null":
                config = data_config
            else:
                if target == "single_pion_production":
                    xs_quasi = getattr(xs_sim, "quasielastic") * SmoothStep(xs_sim.KE, i[0], i[1], i[3], i[2])
                    xs_dcex = getattr(xs_sim, "double_charge_exchange") * SmoothStep(xs_sim.KE, i[0], i[1], i[3], i[2])
                    mod_pdf = {**CreateModPDFDict(xs_sim.KE, "quasielastic", xs_quasi), **CreateModPDFDict(xs_sim.KE, "double_charge_exchange", xs_dcex)}
                    config = CreateConfigShapeTest(data_config, mod_pdf)
                else:
                    xs = getattr(xs_sim, target) * SmoothStep(xs_sim.KE, i[0], i[1], i[3], i[2])
                    config = CreateConfigShapeTest(data_config, CreateModPDFDict(xs_sim.KE, target, xs))
            results[i], true_counts[i], expected_mus[i] = ModifiedConfigTest(config, energy_slices, model, toy_template, mean_track_score_bins, single_bin)
        cross_section.SaveObject(f"{directory}fit_results_{folder[target]}.dill", {"results" : results, "true_counts" : true_counts, "expected_mus" : expected_mus})
    return


def PullStudy(template : cross_section.AnalysisInput, model : cross_section.pyhf.Model, energy_slices : cross_section.Slices, mean_track_score_bins : np.ndarray, data_config : dict, n : int, single_bin) -> dict:
    out = {"expected" : None, "scale" : pd.Series(len(template) / data_config["events"]), "bestfit" : None, "uncertainty" : None}

    cfg = {k : v for k, v in data_config.items()}
    cfg["seed"] = None # ensure we generate random toys

    template_fractions = {s : (sum(template.exclusive_process[s]) / len(template)) for s in template.exclusive_process}

    expected = []
    bestfit = []
    uncertainty = []

    for i in range(n):
        print(rule.Rule(f"iteration : {i} | total : {n}"))
        toy_alt_pdf = cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(df = cex_toy_generator.run(cfg)))

        expected.append({s : (sum(toy_alt_pdf.exclusive_process[s]) / len(toy_alt_pdf.exclusive_process[s])) / template_fractions[s] for s in toy_alt_pdf.exclusive_process})

        result = cross_section.RegionFit.Fit(cross_section.RegionFit.GenerateObservations(toy_alt_pdf, energy_slices, mean_track_score_bins, model, single_bin = single_bin), model, None, [(0, np.inf)]*model.config.npars, False, tolerance = 0.1)

        bestfit.append({list(toy_alt_pdf.exclusive_process.keys())[j] : result.bestfit[j] for j in range(len(template.exclusive_process))})
        uncertainty.append({list(toy_alt_pdf.exclusive_process.keys())[j] : result.uncertainty[j] for j in range(len(template.exclusive_process))})

    out["expected"] = pd.DataFrame(expected)
    out["bestfit"] = pd.DataFrame(bestfit)
    out["uncertainty"] = pd.DataFrame(uncertainty)
    return out


def PlotShapeExamples(energy_slices : cross_section.Slices, book : Plots.PlotBook = Plots.PlotBook.null):
    norms = [0.8, 1.2]
    split = 1000
    smooth_amount = 500
    xs_sim = cross_section.GeantCrossSections(energy_range = [0, energy_slices.max_pos + energy_slices.width])

    def plot_curves(curves : dict, geant : bool, label : str, ylabel : str):
        Plots.plt.figure()
        for k, v in curves.items():
            Plots.Plot(xs_sim.KE, v, "KE (MeV)", ylabel = ylabel, label = f"{label} : {k}", newFigure = False)
        if geant:
            xs_sim.Plot(proc, label = "Geant4", color = "k")
            Plots.plt.fill_between(xs_sim.KE, norms[0] * y, norms[1] * y, color = "k", alpha = 0.5)
        return

    plot_curves({i : SmoothStep(xs_sim.KE, norms[1], norms[0], split, i) for i in [0, 100, 250, 500, 1000]}, False, "$\\alpha$", "$\mathcal{N}(E)$")
    book.Save()

    for proc in list(folder.keys()):
        if proc == "single_pion_production":
            y = xs_sim.quasielastic + xs_sim.double_charge_exchange
        else:
            y = getattr(xs_sim, proc)

        plot_curves({i : y * SmoothStep(xs_sim.KE, norms[1], norms[0], split, i) for i in [0, 100, 250, 500, 1000]}, True, "$\\alpha$", "")
        book.Save()

        plot_curves({i : y * SmoothStep(xs_sim.KE, norms[1], norms[0], i, smooth_amount) for i in [100, 500, 1000, 1500, 1900]}, True, "$x_{0}$", "")
        book.Save()

        Plots.plt.figure()
        for i, j in itertools.combinations([0.8, 1, 1.2], 2):
            smooth_factors = SmoothStep(xs_sim.KE, j, i, split, smooth_amount)
            Plots.Plot(xs_sim.KE, y*smooth_factors, label = f"$n_{{-}} = {i}, n_{{+}} = {j},$", newFigure = False)
            Plots.Plot(xs_sim.KE, y*(smooth_factors[::-1]), label = f"$n_{{-}} = {j}, n_{{+}} = {i},$", newFigure = False)
        xs_sim.Plot(proc, label = "nominal", color = "k")
        Plots.plt.fill_between(xs_sim.KE, norms[0] * y, norms[1] * y, color = "k", alpha = 0.5)
        book.Save()
    return

@cross_section.timer
def PlotCrossCheckResults(xlabel, model : cross_section.pyhf.Model, template_counts : int, results, true_counts, energy_overflow : np.ndarray, pdf : Plots.PlotBook = Plots.PlotBook.null, single_bin : bool = False):
    data, data_energy = ProcessResults(template_counts, results, true_counts, model, single_bin)
    x = list(range(len(results)))

    # Plot the fit value for each scale factor 
    Plots.plt.figure()
    for i in range(4):
        Plots.Plot(x, data[f"mu_{i}"], yerr = data[f"mu_err_{i}"], newFigure = False, label = f"$\mu_{{{process_map[i]}}}$", marker = "o", ylabel = "fit value", color = list(region_colours.values())[i], linestyle = "")
    Plots.plt.xticks(ticks = x, labels = results.keys())
    Plots.plt.xlabel(xlabel)
    pdf.Save()

    # same as above, in separate plots
    for i in Plots.MultiPlot(4):
        Plots.Plot(x, data[f"mu_{i}"], yerr = data[f"mu_err_{i}"], newFigure = False, title = f"$\mu_{{{process_map[i]}}}$", marker = "o", xlabel = xlabel, ylabel = "fit value", color = list(region_colours.values())[i], linestyle = "")
        Plots.plt.xticks(ticks = x, labels = results.keys())
    pdf.Save()

    proc = list(list(true_counts.values())[0].keys())

    # plot true process residual
    for n in Plots.MultiPlot(4):
        Plots.Plot(x, data[f"true_counts_{n}"] * data[f"fe_total_{n}"], yerr = data[f"true_counts_{n}"] * data[f"fe_err_total_{n}"], title = f"$N_{{{process_map[n]}}}^{{pred}}$", xlabel = xlabel, ylabel = "residual", linestyle = "", marker = "o", color = list(region_colours.values())[n], newFigure = False)
        Plots.plt.axhline(0, color = "black", linestyle = "--")
        Plots.plt.xticks(ticks = x, labels = results.keys())
    pdf.Save()

    # plot true process fractional error
    Plots.plt.figure()
    for n in Plots.MultiPlot(4):
        Plots.Plot(x, data[f"fe_total_{n}"], yerr = data[f"fe_err_total_{n}"], label = f"${process_map[n]}$", xlabel = xlabel, ylabel = "fractional error", linestyle = "", marker = "o", color = list(region_colours.values())[n], newFigure = False)
        Plots.plt.axhline(0, color = "black", linestyle = "--")
        Plots.plt.ylim(1.5 * np.min(data.filter(regex = "fe_total_").values), 1.5 * np.max(data.filter(regex = "fe_total_").values))
        Plots.plt.xticks(ticks = x, labels = results.keys())
    pdf.Save()

    # plot true process fractional error
    Plots.plt.figure()
    for n in range(4):
        Plots.Plot(x, data[f"fe_total_{n}"], yerr = data[f"fe_err_total_{n}"], label = f"${process_map[n]}$", xlabel = xlabel, ylabel = "fractional error", linestyle = "", marker = "o", color = list(region_colours.values())[n], newFigure = False)
        Plots.plt.axhline(0, color = "black", linestyle = "--")
        Plots.plt.ylim(1.5 * np.min(data.filter(regex = "fe_total_").values), 1.5 * np.max(data.filter(regex = "fe_total_").values))
    Plots.plt.xticks(ticks = x, labels = results.keys())
    pdf.Save()

    if single_bin is False:
        Plots.plt.figure()
        for i, s in Plots.IterMultiPlot(proc):
            for l, y, e in zip(results, data_energy["fe"][i], data_energy["fe_err"][i]):
                Plots.Plot(energy_overflow, y * true_counts[l][s], yerr = e * true_counts[l][s], label = l, ylabel = "residual", newFigure = False)
            Plots.plt.axhline(0, color = "black", linestyle = "--")
            Plots.plt.xlabel("$KE$ (MeV)")
            Plots.plt.title(f"$N_{{{process_map[i]}}}$")
            Plots.plt.legend(title = xlabel)
        pdf.Save()

        Plots.plt.figure()
        for i, s in Plots.IterMultiPlot(proc):
            for l, y, e in zip(results, data_energy["fe"][i], data_energy["fe_err"][i]):
                Plots.Plot(energy_overflow, y, yerr = e, label = l, ylabel = "fractional error", newFigure = False)
            Plots.plt.axhline(0, color = "black", linestyle = "--")
            Plots.plt.xlabel("$KE$ (MeV)")
            Plots.plt.title(f"$N_{{{process_map[i]}}}$")
            Plots.plt.legend(title = xlabel)
        pdf.Save()
    return

@cross_section.timer
def ProcessResults(template_counts : int, results : dict, true_counts : dict, model : cross_section.pyhf.Model, single_bin : bool):
    true_counts_all = {}
    for t in true_counts:
        true_counts_all[t] = {f"true_counts_{i}" : np.sum(v) for i, v in enumerate(true_counts[t].values())}

    scale_factors = {k : sum(true_counts_all[k].values()) / template_counts for k in true_counts_all}

    mu = {}
    mu_err = {}
    for k in results:
        mu[k] = (results[k].bestfit[0:4] / scale_factors[k])
        mu_err[k] = (results[k].uncertainty[0:4] / scale_factors[k])

    fe, fe_err = CountsFractionalError(results, true_counts, model, single_bin)
    tc_arr = np.swapaxes(np.array([np.array(list(v.values())) for v in true_counts.values()]), 0, 1)

    if single_bin is True:
        tc_arr = np.sum(tc_arr, 2)
        #* single bin gives total fe already
        fe_total = fe 
        fe_err_total = fe_err
    else:
        fe_total = np.nansum(fe * tc_arr, 2) / np.sum(tc_arr, 2)
        fe_err_total = cross_section.nanquadsum(fe_err * tc_arr, 2) / np.sum(tc_arr, 2)

    data = pd.concat([
        pd.DataFrame(scale_factors, index = ['scale_factors']),
        pd.DataFrame(true_counts_all),
        pd.DataFrame(mu, index = [f"mu_{i}" for i in range(4)]),
        pd.DataFrame(mu_err, index = [f"mu_err_{i}" for i in range(4)]),
        pd.DataFrame(fe_total, index = [f"fe_total_{i}" for i in range(4)], columns = list(scale_factors.keys())),
        pd.DataFrame(fe_err_total, index = [f"fe_err_total_{i}" for i in range(4)], columns = list(scale_factors.keys()))
        ])
    data_energy = {"values" : list(true_counts.keys()), "fe" : fe, "fe_err" : fe_err} # can't store this in pandas dataframes
    return data.T, data_energy

@cross_section.timer
def ProcessResultsEnergy(results, true_counts, model):
    true_counts_all = {}
    for t in true_counts:
        true_counts_all[t] = {k : np.sum(v) for k, v in true_counts[t].items()}

    true_counts_all = pd.DataFrame(true_counts_all)
    fe, fe_err = CountsFractionalError(results, true_counts, model)
    return list(true_counts.keys()), fe, fe_err

@cross_section.timer
def PlotDataShapeTestEnergy(data : tuple, energy_overflow : np.ndarray, book : Plots.PlotBook.null):

    def remove_zero(a):
        return a[a != 0]

    for d in range(data["fe"].shape[0]):
        y = data["fe"][d] # second index is process
        y_err = data["fe_err"][d]
        x = data["values"]
        for i, j in enumerate(x):
            if j == "null":
                x[i] = (1, 1, 0, 0)

        x = np.array(x)
        unique_values = [np.unique(x[:, i]) for i in range(x.shape[1])]
        Plots.plt.subplots(len(remove_zero(unique_values[2])), len(remove_zero(unique_values[3])), figsize = [4.8 * len(remove_zero(unique_values[3])), 4.8 * len(remove_zero(unique_values[2]))], sharex = True, sharey = True)
        for p, ((ia, a), (ib, b)) in enumerate(itertools.product(enumerate(remove_zero(unique_values[2])), enumerate(remove_zero(unique_values[3])))):
            Plots.plt.subplot(len(remove_zero(unique_values[2])), len(remove_zero(unique_values[3])), p+1)
            
            for i, j in enumerate(x):
                if (j[2] == a) and (j[3] == b):
                    Plots.Plot(energy_overflow, y[i], yerr = y_err[i], label = f"{x[i][0]}, {x[i][1]}", newFigure = False)
            Plots.plt.fill_between(energy_overflow, 0, linestyle = "--", color = "k")

            if ia == len(remove_zero(unique_values[2])) - 1:
                Plots.plt.xlabel("$N_{int}$(MeV)" + f"\n\n $x0$ : {b}")
            if ib == 0:
                Plots.plt.ylabel(f"$\\alpha$ : {a}\n\n" + "fractional error")
        Plots.plt.suptitle(f"$N_{{{process_map[d]}}}$", size = 20)
        Plots.plt.tight_layout()
        book.Save()
    return

def PlotDataShapeTest(data, key, label, vmin = None, vmax = None):
    def remove_zero(a):
        return a[a != 0]

    y = data[key]
    if vmin is None: vmin = min(y)
    if vmax is None: vmax = max(y)
    x = list(data.index)
    for i, j in enumerate(x):
        if j == "null":
            x[i] = (1, 1, 0, 0)

    x = np.array(x)
    unique_values = [np.unique(x[:, i]) for i in range(x.shape[1])]
    grid = np.meshgrid(unique_values[0], unique_values[1]) #* 0 = high, 1 = low

    fig, axes = Plots.plt.subplots(len(remove_zero(unique_values[2])), len(remove_zero(unique_values[3])), figsize = [4.8 * len(remove_zero(unique_values[3])), 4.8 * len(remove_zero(unique_values[2]))], sharex = True, sharey = True)

    for p, ((ia, a), (ib, b)) in enumerate(itertools.product(enumerate(remove_zero(unique_values[2])), enumerate(remove_zero(unique_values[3])))):
        Plots.plt.subplot(len(remove_zero(unique_values[2])), len(remove_zero(unique_values[3])), p+1)
        im = np.zeros(grid[0].shape)
        for i, j in zip(x, y):
            if i[2] not in [0, a] : continue
            if i[3] not in [0, b] : continue
            loc = np.argwhere((grid[0] == i[0]) & (grid[1] == i[1]))[0]
            im[loc[0], loc[1]] = j

        im_c = Plots.plt.imshow(np.where(im == 0, np.nan, im), cmap = "plasma", vmin = vmin, vmax = vmax)

        Plots.plt.grid(False)
        if ia == len(remove_zero(unique_values[2])) - 1:
            Plots.plt.xlabel("$n_{-}$" + f"\n\n $x0$ : {b}")
        if ib == 0:
            Plots.plt.ylabel(f"$\\alpha$ : {a}\n\n" + "$n_{+}$")


        Plots.plt.yticks(ticks = range(len(unique_values[0])), labels = unique_values[0])
        Plots.plt.xticks(ticks = range(len(unique_values[1])), labels = unique_values[1])

        for (i, j), z in np.ndenumerate(im):
            if z != 0:
                Plots.plt.gca().text(j, i, f"{z:.2f}", ha='center', va='center', fontsize = 12, color = "limegreen", weight = "bold")
        Plots.plt.grid(False)


    Plots.plt.subplots_adjust(wspace = 0, hspace = 0)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
    fig.suptitle(label, size = 20)
    fig.colorbar(im_c, label = label, cax = cbar_ax)
    fig.tight_layout()

@cross_section.timer
def PlotCrossCheckResultsShape(results : dict, template_counts : int, model : cross_section.pyhf.Model, energy_overflow : np.ndarray, single_bin : bool, book : Plots.PlotBook.null):

    processed_data, processed_data_energy = ProcessResults(template_counts, results["results"], results["true_counts"], model, single_bin)

    for p in process_map:
        PlotDataShapeTest(processed_data, f"mu_{p}", f"$\mu_{{{process_map[p]}}}$")
        book.Save()
    if single_bin is False:
        PlotDataShapeTestEnergy(processed_data_energy, energy_overflow, book)

    return


@cross_section.timer
def PredictedCountsSummary(template_counts : float, directory : str, model : cross_section.pyhf.Model, test : str, single_bin : bool):
    results_files = [i for i in cross_section.os.listdir(directory) if "dill" in i]
    results = {[target_map[t] for t in target_map if t in f][0] : cross_section.LoadObject(directory + f) for f in results_files}

    n_fe_total_max = {}
    n_fe_max = {}
    for r in results:
        v = list(results[r]["results"].keys())

        pr, pr_energy = ProcessResults(template_counts, results[r]["results"], results[r]["true_counts"], model, single_bin)
        fractional_error = pr.filter(regex = "fe_total")
        fractional_error_unc = pr.filter(regex = "fe_err_total")
        fe = pr_energy["fe"]
        fe_err = pr_energy["fe_err"]

        n_fe_total_max[r] = (
            np.max(abs(fractional_error.values), 0),
            np.where(np.argmax(abs(fractional_error.values), 0) == 0, fractional_error_unc.values[0, :], fractional_error_unc.values[1, :])
            )

        print(f"{n_fe_total_max[r]=}")

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


def Summary(directory : str, test : str, signal_process : str, model : cross_section.pyhf.Model, energy_overflow : np.ndarray, template : cross_section.Toy, template_counts : float, single_bin : bool, book : Plots.PlotBook = Plots.PlotBook.null, ymax = None):
    n_fe_max, n_fe_total_max = PredictedCountsSummary(template_counts, directory, model, test, single_bin)

    indices = ["absorption", "charge_exchange", "single_pion_production", "pion_production"]
    xlabel = "$KE$ (MeV)"

    tables_n = CreateSummaryTables(n_fe_max, indices)
    SaveSummaryTables(directory, tables_n, "processes")
    print(tables_n[2])

    if ymax is None:
        ymax = np.max([n_fe_max[t][0] for t in n_fe_max])

    if single_bin is False:
        for i, p in Plots.IterMultiPlot(indices):
            for j, t in enumerate(n_fe_max):
                y = n_fe_max[t][0]
                err = n_fe_max[t][1]
                Plots.Plot(energy_overflow, y[i], yerr = err[i], color = f"C{j}", label = cross_section.remove_(t), ylabel = "fractional error in fitted counts", xlabel = xlabel, title = f"process : {cross_section.remove_(p)}", newFigure = False)
            Plots.plt.legend(title = f"{test} test")
            Plots.plt.ylim(0, 1.1 * ymax)
        book.Save()

        for i, t in Plots.IterMultiPlot(n_fe_max):
            for j, p in enumerate(indices):
                y = n_fe_max[t][0]
                err = n_fe_max[t][1]
                Plots.Plot(energy_overflow, y[j], yerr = err[j], xlabel = xlabel, ylabel = "$|f_{bs}|$", label = cross_section.remove_(p), title = f"{test} test : {cross_section.remove_(t)}", newFigure = False)
            Plots.plt.legend(title = "process")
        book.Save()
    return ymax


def PlotTemplates(templates_energy : np.ndarray, tempalates_mean_track_score : np.ndarray, energy_slices : cross_section.Slices, mean_track_score_bins : np.ndarray, template : cross_section.AnalysisInput, book : Plots.PlotBook = Plots.PlotBook.null):
    tags = cross_section.Tags.ExclusiveProcessTags(template.exclusive_process)
    for j, c in Plots.IterMultiPlot(templates_energy):
        for i, s in enumerate(c):
            Plots.Plot(energy_slices.pos_overflow, s/np.sum(templates_energy), color = tags.number[i].colour, label = f"$\lambda_{{{process_map[j]},{process_map[i]}}}$", xlabel = f"$\lambda_{{{process_map[j]},s}}$ (MeV)", ylabel = "normalised counts", style = "step", newFigure = False)
        Plots.plt.title(f"region : {process_map[j]}")
        Plots.plt.legend(title = "process")
    book.Save()

    if tempalates_mean_track_score is not None:
        Plots.plt.figure()
        for i, s in enumerate(tempalates_mean_track_score):
            Plots.Plot(cross_section.bin_centers(mean_track_score_bins), s/np.sum(tempalates_mean_track_score), color = tags.number[i].colour, label = f"$\lambda_{{t,{process_map[i]}}}$", xlabel = f"$\lambda_{{t,s}}$", ylabel = "normalised counts", style = "step", newFigure = False)
        Plots.plt.legend(loc = "upper left")
        book.Save()
        book.close()
    return


def PlotTotalChannel(templates_energy : np.ndarray, tempalates_mean_track_score : np.ndarray, energy_slices : cross_section.Slices, mean_track_score_bins : np.ndarray, book : Plots.PlotBook = Plots.PlotBook.null):
    for j, c in Plots.IterMultiPlot(templates_energy):
        Plots.Plot(energy_slices.pos_overflow, sum(c), xlabel = f"$n_{{{j}}}$ (MeV)", ylabel = "counts", style = "bar", newFigure = False)
    book.Save()

    if tempalates_mean_track_score is not None:
        Plots.Plot(cross_section.bin_centers(mean_track_score_bins), sum(tempalates_mean_track_score), xlabel = f"$n_{{ts}}$", ylabel = "counts", style = "bar")
        book.Save()
    book.close()
    return


@cross_section.timer
def main(args : cross_section.argparse.Namespace):
    cross_section.SetPlotStyle(extend_colors = True, dark = True)
    args.template = cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(file = args.template))

    mean_track_score_bins = np.linspace(0, 1, 21, True)

    models = {}
    if args.fit["mean_track_score"] is True:
        models["track_score"], templates_energy, tempalates_mean_track_score = cross_section.RegionFit.CreateModel(args.template, args.energy_slices, mean_track_score_bins, True, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])
    else:
        models["normal"], templates_energy, tempalates_mean_track_score = cross_section.RegionFit.CreateModel(args.template, args.energy_slices, None, True, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])

    os.makedirs(args.out, exist_ok = True)

    if args.toy_data_config:
        print("Running tests")

        for m in models:

            ts_bins = mean_track_score_bins if m == "track_score" else None

            if "normalisation" not in args.skip:
                os.makedirs(args.out + f"normalisation_test_{m}/", exist_ok = True)
                NormalisationTest(
                    args.out + f"normalisation_test_{m}/",
                    args.toy_data_config, models[m],
                    args.template,
                    args.energy_slices,
                    ts_bins,
                    args.fit["single_bin"])

            if "shape" not in args.skip:
                xs_sim = cross_section.GeantCrossSections(energy_range = [0, max(args.energy_slices.pos) + args.energy_slices.width])

                os.makedirs(args.out + f"shape_test_{m}/", exist_ok = True)
                ShapeTestNew(
                    args.out + f"shape_test_{m}/",
                    args.toy_data_config,
                    models[m],
                    args.template,
                    ts_bins,
                    xs_sim,
                    args.energy_slices,
                    args.fit["single_bin"])
            if "pulls" not in args.skip:
                pull_results = PullStudy(args.template, models[m], args.energy_slices, mean_track_score_bins if m == "track_score" else None, args.toy_data_config, 100, args.fit["single_bin"])
                os.makedirs(args.out + f"pull_test_{m}/", exist_ok = True)
                DictToHDF5(pull_results, args.out + f"pull_test_{m}/" + "pull_results.hdf5")


    if args.workdir:
        print("Making test results")

        if args.fit["single_bin"] is False:
            with Plots.PlotBook(args.workdir + "templates") as book:
                PlotTemplates(templates_energy, tempalates_mean_track_score, args.energy_slices, mean_track_score_bins, args.template, book)

            with Plots.PlotBook(args.workdir + "observation_exmaple") as book:
                PlotTotalChannel(templates_energy, tempalates_mean_track_score, args.energy_slices, mean_track_score_bins, book)

        with Plots.PlotBook(args.workdir + "xs_curves") as book:
            PlotShapeExamples(args.energy_slices, book)

        label_map = {"absorption" : "abs", "charge_exchange" : "cex", "single_pion_production" : "spip", "pion_production" : "pip"}

        test = ["shape", "normalisation", "pulls"] 

        template_counts = sum(args.template.inclusive_process)

        ymax = {"shape" : 1.5, "normalisation" : 0.6}

        for t in test:
            if t in args.skip: continue
            for m in models:
                if t == "pulls":
                    with Plots.PlotBook(args.workdir + f"pull_test_{m}/pulls.pdf", True) as book:
                        pull_results = ReadHDF5(args.workdir + f"pull_test_{m}/" + "pull_results.hdf5")

                        pulls = (pull_results["bestfit"] - (pull_results["expected"] / pull_results["scale"][0])) / pull_results["uncertainty"]

                        xlabel = "$\\theta$"

                        for _, k in Plots.IterMultiPlot(pulls.columns):
                            mean = np.mean(pulls[k])
                            std = np.std(pulls[k])
                            sem = std / np.sqrt(len(pulls[k]))
                            Plots.PlotHist(pulls[k], bins = 10, title = f"$\mu_{{{label_map[k]}}}$ | mean : {mean:.3g} $\pm$ {sem:.1g} | std.dev : {std:.3g} ", xlabel = xlabel, newFigure = False)
                        book.Save()

                else:
                    directory = args.workdir + f"{t}_test_{m}/"
                    results_files = [i for i in cross_section.os.listdir(directory) if "dill" in i]

                    for f in results_files:
                        fit_result = cross_section.LoadObject(directory+f)
                        target = [target_map[k] for k in target_map if k in f][0]
                        with Plots.PlotBook(directory+f.split(".")[0]+".pdf", True) as pdf:
                            if (t == "shape"):
                                PlotCrossCheckResultsShape(fit_result, template_counts, models[m], args.energy_slices.pos_overflow, args.fit["single_bin"], pdf)
                            else:
                                PlotCrossCheckResults(f"{cross_section.remove_(target)} {t}", models[m], template_counts, fit_result["results"], fit_result["true_counts"], args.energy_slices.pos_overflow, pdf, args.fit["single_bin"])
                        Plots.plt.close("all")
                
                    with Plots.PlotBook(f"{directory}summary_plots.pdf", True) as book:
                        Summary(directory, t, args.signal_process, models[m], args.energy_slices.pos_overflow, args.template, template_counts, args.fit["single_bin"], book, ymax = ymax[t])
                    Plots.plt.close("all")

    return


if __name__ == "__main__":
    parser = cross_section.argparse.ArgumentParser("app which performs cross checks for the region fit using toys.")
    
    cross_section.ApplicationArguments.Config(parser, True)

    parser.add_argument("--template", "-t", dest = "template", type = str, help = "toy template hdf5 file", required = True)

    parser.add_argument("--toy_data_config", "-d", dest = "toy_data_config", type = str, help = "json config for toy data", required = False)

    parser.add_argument("--workdir", "-w", dest = "workdir", type = str, help = "work directory which contains output from this application, use this to remake plots without running all the tests again", required = False)
    parser.add_argument("--signal_process", dest = "signal_process", type = str, help = "signal process for background subtraction")

    parser.add_argument("--step", "-s", dest = "step", type = float, help = "step size for toy, if provided will override toy data config.")
    parser.add_argument("--events", "-n", dest = "events", type = float, help = "events for toy, if provided will override toy data config.")
    parser.add_argument("--seed", dest = "seed", type = int, help = "seed for toy, if provided will override toy data config.")

    parser.add_argument("--skip", dest = "skip", type = str, choices = ["normalisation", "shape", "pulls"], nargs = "+", default = [], help = "test to skip")

    cross_section.ApplicationArguments.Output(parser, "region_fit_studies")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())

    if (not args.toy_data_config) and (not args.workdir):
        raise Exception("--toy_data_config or --workdirs or both must be supplied")

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