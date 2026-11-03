#!/usr/bin/env python3
"""
Created on: 17/01/2024 13:46

Author: Shyam Bhuller

Description: app which performs cross checks for the region fit using toys.
"""
import argparse
import os

import cabinetry
import itertools
import numpy as np
import pandas as pd
import pyhf

from rich import print, rule
from scipy.ndimage import gaussian_filter1d

from python.analysis import application
from python.analysis import common
from python.analysis import cross_section
from python.analysis import fitting
from python.analysis import io
from python.analysis import plotting
from python.analysis import slices
from python.analysis import tags
from python.analysis import toy
from python.legacy import RegionFit
from apps import xs_toy_generator

from python.analysis.plotting import plots

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


def override_config(args):
    """Apply the supplied step, event-count, and seed overrides to the toy configuration.

        Args:
            args (Any): Parsed arguments containing optional step, event-count, and seed overrides and the toy_data_config dictionary.
    """
    for a in ["step", "events", "seed"]:
        if getattr(args, a):
            args.toy_data_config[a] = getattr(args, a)
    return


def create_config_normalisation(scales : dict, data_config : dict) -> dict:
    """Copy a toy configuration and set its process PDF scale factors.

        Args:
            scales (dict): Process-to-scale-factor mapping stored under pdf_scale_factors.
            data_config (dict): Base toy-generation configuration to copy.

        Returns:
            dict: Shallow copy of data_config with pdf_scale_factors set to scales.
    """
    cfg = {k : v for k, v in data_config.items()}
    cfg["pdf_scale_factors"] = scales
    return cfg


def modified_config_test(config : dict, energy_slice : slices.Slices, model : pyhf.Model, toy_template : cross_section.analysis_input.AnalysisInput, mean_track_score_bins : np.array = None, single_bin : np.array = False) -> tuple[dict]:
    """Generate configured toy data, fit it, and return fit results, true counts, and expected process scales.

        Args:
            config (dict): Toy-generation configuration used to create the alternative sample.
            energy_slice (slices.Slices): Energy slicing used to build observations and count interactions.
            model (pyhf.Model): Statistical model fitted to the generated observations.
            toy_template (cross_section.analysis_input.AnalysisInput): Reference sample used to calculate expected process scale factors.
            mean_track_score_bins (np.array): Optional bin edges for the mean-track-score channel.
            single_bin (np.array): Whether to fit one combined bin instead of energy-resolved bins.

        Returns:
            tuple[dict]: Fit result, true interaction counts by process, and expected process scales.
    """
    toy_alt_pdf = cross_section.analysis_input.AnalysisInput.create_analysis_input_toy(toy.Toy(df = xs_toy_generator.run(config)))
    
    obs = RegionFit.GenerateObservations(toy_alt_pdf, energy_slice, mean_track_score_bins, model, False, single_bin)
    fit_result = RegionFit.Fit(obs, model, None, verbose = False, tolerance = 0.1) # relax the tolerance to allow fit to converge for a large parameter space

    true_process_counts = {}
    for v in toy_alt_pdf.exclusive_process:
        true_process_counts[v] = toy_alt_pdf.n_interact(energy_slice, toy_alt_pdf.exclusive_process[v])

    expected_mus = [sum(toy_alt_pdf.exclusive_process[r]) / sum(toy_template.exclusive_process[r]) for r in toy_alt_pdf.exclusive_process]

    return fit_result, true_process_counts, expected_mus


def normalisation_test(directory : str, data_config : dict, model : pyhf.Model, toy_template : cross_section.analysis_input.AnalysisInput, energy_slice : slices.Slices, mean_track_score_bins : np.ndarray, single_bin : bool):
    """Repeat toy fits at several process normalizations and save the results for each target process.

        Args:
            directory (str): Output directory prefix for the saved fit-result files.
            data_config (dict): Base configuration for generating toy samples.
            model (pyhf.Model): Statistical model fitted to each generated sample.
            toy_template (cross_section.analysis_input.AnalysisInput): Reference sample for expected process scale factors.
            energy_slice (slices.Slices): Energy slicing used to build fit observations.
            mean_track_score_bins (np.ndarray): Bin edges for the optional mean-track-score channel.
            single_bin (bool): Whether to fit one combined bin instead of energy-resolved bins.
    """
    for target in folder:
        print(target)

        results = {}
        true_counts = {}
        expected_mus = {}
    
        scales = {k : 1 for k in ['absorption', 'quasielastic', 'charge_exchange', 'double_charge_exchange', 'pion_production']}
        for i in [0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.5]:
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
                config = create_config_normalisation(scales, data_config)
            results[i], true_counts[i], expected_mus[i] = modified_config_test(config, energy_slice, model, toy_template, mean_track_score_bins, single_bin)

        io.write_dill(f"{directory}fit_results_{folder[target]}.dill", {"results" : results, "true_counts" : true_counts, "expected_mus" : expected_mus})
    return


def fit_simulation_curve(xs_sim, sampling_factor : int, process : str, function : fitting.FitFunction, plot : bool = False):
    """Fit a sampled simulation cross-section curve as a function of kinetic energy.

        Args:
            xs_sim (Any): Simulated cross-section object providing kinetic energy and the selected process values.
            sampling_factor (int): Stride used to subsample the simulation points.
            process (str): Cross-section attribute name to fit.
            function (fitting.FitFunction): Parametric model fitted to the sampled curve.
            plot (bool): Whether to draw the fitted curve and data.

        Returns:
            tuple: Optimized fit parameters and their estimated errors from fitting.fit.
    """
    x = xs_sim.KE[::sampling_factor]/1000
    y = getattr(xs_sim, process)[::sampling_factor]/1000

    if plot is True: plots.plt.figure()
    results = fitting.fit(x, y, None, function, method = "dogbox", plot = plot, xlabel = "$KE$(GeV)", ylabel = "$\\sigma$ (b)", title = process)
    return results


def counts_fractional_error(results, true_counts, model, single_bin = False):
    """Compare post-fit predicted counts with true counts and return fractional differences and uncertainties.

        Args:
            results (Any): Fit results keyed by test configuration.
            true_counts (Any): True interaction counts keyed by configuration and process.
            model (Any): Fitted pyhf model used to obtain post-fit predictions.
            single_bin (Any): Whether to combine all bins when comparing predicted and true counts.

        Returns:
            tuple: Fractional count differences and their propagated uncertainties, arranged by process, configuration, and bin.
    """
    fe_err = []
    fe = []

    mean_track_score = any([c["name"] == "mean_track_score" for c in model.spec["channels"]])

    for s in results:
        true_counts_arr = np.array(list(true_counts[s].values()))
        post_fit_pred = cabinetry.model_utils.prediction(model, fit_results = results[s], label = "post-fit")

        if mean_track_score:
            KE_int_prediction = RegionFit.SliceModelPrediction(post_fit_pred, slice(-1), "KE_int_postfit") # exclude the channel which is the mean track score
        else:
            KE_int_prediction = RegionFit.SliceModelPrediction(post_fit_pred, slice(0, len(post_fit_pred.model_yields)), "KE_int_postfit")

        pred_counts_err = common.quadsum(np.array(KE_int_prediction.total_stdev_model_bins)[:, :-1], 0)
    
        if single_bin is True:
            t = np.sum(true_counts_arr, 1)
            pred_counts_err = pred_counts_err.flatten()
        else:
            t = true_counts_arr

        fe_err.append(common.nandiv(pred_counts_err, t))

        pred_counts = np.sum(np.array(KE_int_prediction.model_yields), 0)
        if single_bin:
            pred_counts = pred_counts.flatten()

        fe.append(np.array([common.nandiv(pred_counts[i] - v, v) for i, v in enumerate(t)]))
    fe_err = np.swapaxes(np.array(fe_err), 0, 1)
    fe = np.swapaxes(np.array(fe), 0, 1)
    return fe, fe_err


def create_config_shape_test(data_config, modified_PDFs : dict):
    """Copy a toy configuration and attach the process PDFs modified for a shape test.

        Args:
            data_config (Any): Base toy-generation configuration to copy.
            modified_PDFs (dict): Process-to-modified-cross-section mapping inserted into the configuration.

        Returns:
            dict: Shallow copy of data_config with modified_PDFs set to the supplied mapping.
    """
    cfg = {k : v for k, v in data_config.items()}
    cfg["modified_PDFs"] = modified_PDFs
    return cfg


def smooth_step(x : np.ndarray, high : float, low : float, split : float, smooth_amount : float) -> np.ndarray:
    """ Generate points for a smoothstep function.

        Args:
            x (np.ndarray): x values.
            high (float): High value of the smoothstep.
            low (float): Low value of the smoothstep.
            split (float): Where the step occurs in x.
            smooth_amount (float): Amount to smooth the step by 0 results in a step function.

        Returns:
            np.ndarray: y values of smooth step function.
    """
    step = np.where(x >= split, high, low)
    if smooth_amount == 0:
        return step
    else:
        return gaussian_filter1d(step, smooth_amount)


def create_modified_xs(xs_sim : cross_section.geant.GeantCrossSections, process : str, high : float, low : float, split : float, smooth_amount : float) -> dict:
    """ Create a modified cross section curve based on a smoothstep function.

        Args:
            xs_sim (cross_section.geant.GeantCrossSections): GEANT cross sections.
            process (str): Processes cross section to modify.
            high (float): High value of the smoothstep.
            low (float): Low value of the smoothstep.
            split (float): Where the step occurs in x.
            smooth_amount (float): Amount to smooth the step by 0 results in a step function.

        Returns:
            dict: dictionary with KE and modified cross section.
    """
    if process == "single_pion_production":
        xs_quasi = xs_sim.quasielastic * smooth_step(xs_sim.KE, high, low, split, smooth_amount)
        xs_dcex = xs_sim.double_charge_exchange * smooth_step(xs_sim.KE, high, low, split, smooth_amount)
        return {**{"KE" : xs_sim.KE, "quasielastic" : xs_quasi}, **{"KE" : xs_sim.KE, "double_charge_exchange" : xs_dcex}}
    else:
        xs = getattr(xs_sim, process) * smooth_step(xs_sim.KE, high, low, split, smooth_amount)
        return {"KE" : xs_sim.KE, process : xs}


def shape_test_new(directory : str, data_config : dict, model : pyhf.Model, toy_template : cross_section.analysis_input.AnalysisInput, mean_track_score_bins : np.array, xs_sim : cross_section.geant.GeantCrossSections, energy_slices : slices.Slices, single_bin):
    """Run toy fits for smooth-step cross-section variations and save results by target process.

        Args:
            directory (str): Output directory prefix for the saved fit-result files.
            data_config (dict): Base configuration for generating toy samples.
            model (pyhf.Model): Statistical model fitted to each generated sample.
            toy_template (cross_section.analysis_input.AnalysisInput): Reference sample for expected process scale factors.
            mean_track_score_bins (np.array): Bin edges for the optional mean-track-score channel.
            xs_sim (cross_section.geant.GeantCrossSections): Simulated cross sections modified in the shape variations.
            energy_slices (slices.Slices): Energy slicing used to build fit observations.
            single_bin (Any): Whether to fit one combined bin instead of energy-resolved bins.
    """
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
                config = create_config_shape_test(data_config, create_modified_xs(xs_sim, target, i[0], i[1], i[3], i[2]))

            results[i], true_counts[i], expected_mus[i] = modified_config_test(config, energy_slices, model, toy_template, mean_track_score_bins, single_bin)
        io.write_dill(f"{directory}fit_results_{folder[target]}.dill", {"results" : results, "true_counts" : true_counts, "expected_mus" : expected_mus})
    return


def pull_study(template : cross_section.analysis_input.AnalysisInput, model : pyhf.Model, energy_slices : slices.Slices, mean_track_score_bins : np.ndarray, data_config : dict, n : int, single_bin) -> dict:
    """Generate n toy samples, fit each sample, and collect expected scales, best fits, and uncertainties.

        Args:
            template (cross_section.analysis_input.AnalysisInput): Reference sample used to calculate expected process fractions.
            model (pyhf.Model): Statistical model fitted to each generated toy sample.
            energy_slices (slices.Slices): Energy slicing used to build fit observations.
            mean_track_score_bins (np.ndarray): Bin edges for the optional mean-track-score channel.
            data_config (dict): Toy-generation configuration, copied with its seed cleared for random samples.
            n (int): Number of toy samples to generate and fit.
            single_bin (Any): Whether to fit one combined bin instead of energy-resolved bins.

        Returns:
            dict: DataFrames of expected scales, best-fit values, and uncertainties, plus the template/data scale Series.
    """
    out = {"expected" : None, "scale" : pd.Series(len(template) / data_config["events"]), "bestfit" : None, "uncertainty" : None}

    cfg = {k : v for k, v in data_config.items()}
    cfg["seed"] = None # ensure we generate random toys

    template_fractions = {s : (sum(template.exclusive_process[s]) / len(template)) for s in template.exclusive_process}

    expected = []
    bestfit = []
    uncertainty = []

    for i in range(n):
        print(rule.Rule(f"iteration : {i} | total : {n}"))
        toy_alt_pdf = cross_section.analysis_input.AnalysisInput.create_analysis_input_toy(toy.Toy(df = xs_toy_generator.run(cfg)))

        expected.append({s : (sum(toy_alt_pdf.exclusive_process[s]) / len(toy_alt_pdf.exclusive_process[s])) / template_fractions[s] for s in toy_alt_pdf.exclusive_process})

        result = RegionFit.Fit(RegionFit.GenerateObservations(toy_alt_pdf, energy_slices, mean_track_score_bins, model, single_bin = single_bin), model, None, [(0, np.inf)]*model.config.npars, False, tolerance = 0.1)

        bestfit.append({list(toy_alt_pdf.exclusive_process.keys())[j] : result.bestfit[j] for j in range(len(template.exclusive_process))})
        uncertainty.append({list(toy_alt_pdf.exclusive_process.keys())[j] : result.uncertainty[j] for j in range(len(template.exclusive_process))})

    out["expected"] = pd.DataFrame(expected)
    out["bestfit"] = pd.DataFrame(bestfit)
    out["uncertainty"] = pd.DataFrame(uncertainty)
    return out


def pull_study_fast(toys : toy.Toy, n_template : int, n_data : int, args, energy_slices : slices.Slices, mean_track_score_bins : np.ndarray, n : int) -> dict:
    """Run repeated fits using resampled template/data rows from a shared toy sample.

        Args:
            toys (toy.Toy): Shared toy sample from which template and data rows are resampled.
            n_template (int): Number of rows sampled to construct each fit template.
            n_data (int): Number of rows sampled to construct each pseudo-data sample.
            args (Any): Study arguments providing energy slices and fit configuration options.
            energy_slices (slices.Slices): Energy slicing used to build fit observations.
            mean_track_score_bins (np.ndarray): Bin edges for the optional mean-track-score channel.
            n (int): Number of repeated fits to perform.

        Returns:
            dict: DataFrames of expected scales, scale factors, best-fit values, and uncertainties from the repeated fits.
    """
    out = {"expected" : None, "scale" : None, "bestfit" : None, "uncertainty" : None}

    @common.timer
    def Pull():
        """Resample template and data rows, build and fit a model, and collect scale estimates.

            Returns:
                dict: Expected process scales, fitted values, uncertainties, and template/data scale factors for one resampled fit.
        """
        output = {"expected" : None, "bestfit" : None, "uncertainty" : None}
        print(rule.Rule(f"iteration : {i} | total : {n}"))
        template = cross_section.analysis_input.AnalysisInput.create_analysis_input_toy(toy.Toy(df = toys.df.iloc[np.random.choice(len(toys.df), int(n_template))].reset_index()))
        data = cross_section.analysis_input.AnalysisInput.create_analysis_input_toy(toy.Toy(df = toys.df.iloc[np.random.choice(len(toys.df), int(n_data))].reset_index()))

        if mean_track_score_bins:
            model = RegionFit.CreateModel(template, args.energy_slices, mean_track_score_bins, False, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])
        else:
            model = RegionFit.CreateModel(template, args.energy_slices, None, False, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])

        template_fractions = {s : (sum(template.exclusive_process[s]) / len(template)) for s in template.exclusive_process}

        output["expected"] = {s : (sum(data.exclusive_process[s]) / len(data.exclusive_process[s])) / template_fractions[s] for s in data.exclusive_process}

        result = RegionFit.Fit(RegionFit.GenerateObservations(data, energy_slices, mean_track_score_bins, model, single_bin = args.fit["single_bin"]), model, None, [(0, np.inf)]*model.config.npars, False, tolerance = 0.1)

        output["bestfit"] = {list(data.exclusive_process.keys())[j] : result.bestfit[j] for j in range(len(template.exclusive_process))}
        output["uncertainty"] = {list(data.exclusive_process.keys())[j] : result.uncertainty[j] for j in range(len(template.exclusive_process))}
        output["scale"] = {k : sum(v) / sum(data.exclusive_process[k]) for k,v in template.exclusive_process.items()}
        return output

    expected = []
    bestfit = []
    uncertainty = []
    scales = []
    for i in range(n):
        while True:
            try:
                o = Pull()
                expected.append(o["expected"])
                bestfit.append(o["bestfit"])
                uncertainty.append(o["uncertainty"])
                scales.append(o["scale"])
                break
            except ValueError:
                print("minimum was invalid, trying again")

    out["expected"] = pd.DataFrame(expected)
    out["bestfit"] = pd.DataFrame(bestfit)
    out["uncertainty"] = pd.DataFrame(uncertainty)
    out["scale"] = pd.DataFrame(scales)
    return out

def plot_shape_examples(energy_slices : slices.Slices, book : plotting.PlotBook = plotting.PlotBook.null()):
    """Plot example cross-section curves with normalization and smooth-step shape variations.

        Args:
            energy_slices (slices.Slices): Energy slicing that sets the range of simulated kinetic energies.
            book (plotting.PlotBook): PlotBook used to save each generated figure.
    """
    norms = [0.8, 1.2]
    split = 1000
    smooth_amount = 500
    xs_sim = cross_section.geant.GeantCrossSections(energy_range = [0, energy_slices.max + energy_slices.width])

    def plot_curves(curves : dict, geant : bool, label : str, ylabel : str):
        """Plot each supplied curve and optionally overlay the GEANT cross section and variation band.

            Args:
                curves (dict): Mapping from curve labels to values on the simulated kinetic-energy grid.
                geant (bool): Whether to overlay the nominal GEANT curve and its normalization band.
                label (str): Label describing the varied parameter shown in the legend.
                ylabel (str): Vertical-axis label for the plotted curves.
        """
        plots.plt.figure()
        for k, v in curves.items():
            plots.plot(xs_sim.KE, v, "KE (MeV)", ylabel = ylabel, label = f"{label} : {k}", new_figure = False)
        if geant:
            xs_sim.plot(proc, label = "Geant4", color = "k")
            plots.plt.fill_between(xs_sim.KE, norms[0] * y, norms[1] * y, color = "k", alpha = 0.5)
        return

    plot_curves({i : smooth_step(xs_sim.KE, norms[1], norms[0], split, i) for i in [0, 100, 250, 500, 1000]}, False, "$\\alpha$", "$\\mathcal{N}(E)$")
    book.save()

    for proc in list(folder.keys()):
        if proc == "single_pion_production":
            y = xs_sim.quasielastic + xs_sim.double_charge_exchange
        else:
            y = getattr(xs_sim, proc)

        plot_curves({i : y * smooth_step(xs_sim.KE, norms[1], norms[0], split, i) for i in [0, 100, 250, 500, 1000]}, True, "$\\alpha$", "")
        book.save()

        plot_curves({i : y * smooth_step(xs_sim.KE, norms[1], norms[0], i, smooth_amount) for i in [100, 500, 1000, 1500, 1900]}, True, "$x_{0}$", "")
        book.save()

        plots.plt.figure()
        for i, j in itertools.combinations([0.8, 1, 1.2], 2):
            smooth_factors = smooth_step(xs_sim.KE, j, i, split, smooth_amount)
            plots.plot(xs_sim.KE, y*smooth_factors, label = f"$n_{{-}} = {i}, n_{{+}} = {j},$", new_figure = False)
            plots.plot(xs_sim.KE, y*(smooth_factors[::-1]), label = f"$n_{{-}} = {j}, n_{{+}} = {i},$", new_figure = False)
        xs_sim.plot(proc, label = "nominal", color = "k")
        plots.plt.fill_between(xs_sim.KE, norms[0] * y, norms[1] * y, color = "k", alpha = 0.5)
        book.save()
    return

@common.timer
def plot_cross_check_results(xlabel : str, data, data_energy, energy_overflow : np.ndarray, pdf : plotting.PlotBook = plotting.PlotBook.null(), single_bin : bool = False, ylim = None):
    """Plot fitted process scales, count residuals, and fractional errors across test settings and energy.

        Args:
            xlabel (str): Horizontal-axis label describing the tested setting.
            data (Any): Process-indexed fit summary table containing fitted scales, true counts, and fractional errors.
            data_energy (Any): Energy-binned fractional errors and uncertainties keyed by process and test setting.
            energy_overflow (np.ndarray): Energy-bin coordinates used for the energy-resolved plots.
            pdf (plotting.PlotBook): PlotBook used to save the generated figures.
            single_bin (bool): Whether the fit uses a single combined bin, suppressing energy-resolved plots.
            ylim (Any): Optional lower and upper limits for fractional-error plot axes.
    """
    x = data.index.values

    xt = np.arange(min(x), max(x)+0.1, 0.1)

    # Plot the fit value for each scale factor 
    plots.plt.figure()
    for i in range(4):
        plots.plot(x, data[f"mu_{i}"], yerr = data[f"mu_err_{i}"], new_figure = False, label = f"${process_map[i]}$", marker = "o", ylabel = "$\\mu^{fit}_{s}$", color = list(region_colours.values())[i], linestyle = "")
    plots.plt.legend(title = "$s$")
    plots.plt.xticks(ticks = xt, labels = np.round(xt, 1))
    plots.plt.xlabel(xlabel)
    pdf.save()

    # same as above, in separate plots
    for i in plots.multi_plot(4):
        plots.plot(x, data[f"mu_{i}"], yerr = data[f"mu_err_{i}"], new_figure = False, marker = "o", xlabel = xlabel, ylabel = f"$\\mu^{{fit}}_{{{process_map[i]}}}$", color = list(region_colours.values())[i], linestyle = "")
        plots.plt.xticks(ticks = xt, labels = np.round(xt, 1))
    pdf.save()

    # plot true process residual
    for n in plots.multi_plot(4):
        plots.plot(x, data[f"true_counts_{n}"] * data[f"fe_total_{n}"], yerr = data[f"true_counts_{n}"] * data[f"fe_err_total_{n}"], title = f"$N_{{{process_map[n]}}}^{{pred}}$", xlabel = xlabel, ylabel = "residual", linestyle = "", marker = "o", color = list(region_colours.values())[n], new_figure = False)
        plots.plt.axhline(0, color = "black", linestyle = "--")
        plots.plt.xticks(ticks = xt, labels = np.round(xt, 1))
    pdf.save()

    # plot true process fractional error
    plots.plt.figure()
    for n in plots.multi_plot(4):
        plots.plot(x, data[f"fe_total_{n}"], yerr = data[f"fe_err_total_{n}"], xlabel = xlabel, ylabel = f"$f_{{{process_map[n]}}}$", linestyle = "", marker = "o", color = list(region_colours.values())[n], new_figure = False)
        plots.plt.axhline(0, color = "black", linestyle = "--")
        plots.plt.ylim(1.5 * np.min(data.filter(regex = "fe_total_").values), 1.5 * np.max(data.filter(regex = "fe_total_").values))
        plots.plt.xticks(ticks = xt, labels = np.round(xt, 1))
        if ylim: plots.plt.ylim(ylim)
    pdf.save()

    # plot true process fractional error
    plots.plt.figure()
    for n in range(4):
        plots.plot(x, data[f"fe_total_{n}"], yerr = data[f"fe_err_total_{n}"], label = f"${process_map[n]}$", xlabel = xlabel, ylabel = "$f_{s}$", linestyle = "", marker = "o", color = list(region_colours.values())[n], new_figure = False)
        plots.plt.axhline(0, color = "black", linestyle = "--")
        plots.plt.ylim(1.5 * np.min(data.filter(regex = "fe_total_").values), 1.5 * np.max(data.filter(regex = "fe_total_").values))
    plots.plt.legend(title = "$s$")
    plots.plt.xticks(ticks = xt, labels = np.round(xt, 1))
    if ylim: plots.plt.ylim(ylim)
    pdf.save()

    plots.plt.figure()
    for i in plots.multi_plot(4):
        plots.plot(data[f"mu_exp_{i}"], data[f"mu_{i}"], yerr = data[f"mu_err_{i}"], linestyle = "", marker = "o", xlabel = f"$\\mu^{{exp}}_{process_map[i]}$", ylabel = "$\\mu^{fit}_{s}$", label = f"${process_map[i]}$", color = list(region_colours.values())[i], new_figure = False)
        yl = plots.plt.gca().get_ylim()
        xl = plots.plt.gca().get_xlim()

        plots.plt.axline([0, 0], [1, 1], linestyle = "--", color = "k", label = "$y = x$")
        plots.plt.xlim(xl)
        plots.plt.ylim(yl)
        plots.plt.legend(title = "$s$")

    pdf.save()

    if single_bin is False:
        plots.plt.figure()
        for i, s in plots.iter_multi_plot(process_map):
            for l, y, e in zip(x, data_energy["fe"][i], data_energy["fe_err"][i]):
                plots.plot(energy_overflow, data[f"true_counts_{s}"] * y, yerr = data[f"true_counts_{s}"] * e, label = l, ylabel = "residual", new_figure = False)
            plots.plt.axhline(0, color = "black", linestyle = "--")
            plots.plt.xlabel("$KE$ (MeV)")
            plots.plt.title(f"$N_{{{process_map[i]}}}$")
            plots.plt.legend(title = xlabel)
            if ylim: plots.plt.ylim(ylim)
        pdf.save()

        plots.plt.figure()
        for i, s in plots.iter_multi_plot(process_map):
            for l, y, e in zip(x, data_energy["fe"][i], data_energy["fe_err"][i]):
                plots.plot(energy_overflow, y, yerr = e, label = l, ylabel = "$f$", new_figure = False)
            plots.plt.axhline(0, color = "black", linestyle = "--")
            plots.plt.xlabel("$KE$ (MeV)")
            plots.plt.title(f"$N_{{{process_map[i]}}}$")
            plots.plt.legend(title = xlabel)
            if ylim: plots.plt.ylim(ylim)
        pdf.save()
    return


def process_results(template_counts : int, results : dict, true_counts : dict, expected_mu : dict, model : pyhf.Model, single_bin : bool):
    """Normalize fit results by event-count scales and calculate total and energy-binned fractional count errors.

        Args:
            template_counts (int): Number of entries in the reference template used to normalize process counts.
            results (dict): Fit results keyed by test configuration.
            true_counts (dict): True interaction counts keyed by configuration and process.
            expected_mu (dict): Expected process scale factors keyed by configuration.
            model (pyhf.Model): Fitted pyhf model used to calculate post-fit count differences.
            single_bin (bool): Whether to combine all bins when calculating total fractional errors.

        Returns:
            tuple: Process-indexed total summary DataFrame and energy-binned error dictionary.
    """
    true_counts_all = {}
    for t in true_counts:
        true_counts_all[t] = {f"true_counts_{i}" : np.sum(v) for i, v in enumerate(true_counts[t].values())}

    scale_factors = {k : sum(true_counts_all[k].values()) / template_counts for k in true_counts_all}

    mu = {}
    mu_err = {}
    mu_exp = {}
    for k in results:
        mu[k] = results[k].bestfit[0:4] / scale_factors[k]
        mu_err[k] = results[k].uncertainty[0:4] / scale_factors[k]
        mu_exp[k] = expected_mu[k] / scale_factors[k]

    fe, fe_err = counts_fractional_error(results, true_counts, model, single_bin)
    tc_arr = np.swapaxes(np.array([np.array(list(v.values())) for v in true_counts.values()]), 0, 1)

    if single_bin is True:
        tc_arr = np.sum(tc_arr, 2)
        #* single bin gives total fe already
        fe_total = fe 
        fe_err_total = fe_err
    else:
        fe_total = np.nansum(fe * tc_arr, 2) / np.sum(tc_arr, 2)
        fe_err_total = common.nanquadsum(fe_err * tc_arr, 2) / np.sum(tc_arr, 2)

    data = pd.concat([
        pd.DataFrame(scale_factors, index = ['scale_factors']),
        pd.DataFrame(true_counts_all),
        pd.DataFrame(mu, index = [f"mu_{i}" for i in range(4)]),
        pd.DataFrame(mu_err, index = [f"mu_err_{i}" for i in range(4)]),
        pd.DataFrame(mu_exp, index = [f"mu_exp_{i}" for i in range(4)], columns = list(scale_factors.keys())),
        pd.DataFrame(fe_total, index = [f"fe_total_{i}" for i in range(4)], columns = list(scale_factors.keys())),
        pd.DataFrame(fe_err_total, index = [f"fe_err_total_{i}" for i in range(4)], columns = list(scale_factors.keys())),
        ])
    data_energy = {"values" : list(true_counts.keys()), "fe" : fe, "fe_err" : fe_err} # can't store this in pandas dataframes
    return data.T, data_energy


def mean_processed_results(total, energy):
    """Average total result tables and energy-binned fractional-error arrays across studies.

        Args:
            total (Any): Sequence of total-summary DataFrames to average.
            energy (Any): Sequence of energy-binned fractional-error dictionaries to average.

        Returns:
            tuple: Mean total-summary DataFrame and mean energy-binned errors with their shared settings.
    """
    total = sum(total) / len(total)

    energy_reshaped = {}
    for res in energy:
        for k, v in res.items():
            if k not in energy_reshaped:
                energy_reshaped[k] = []
            energy_reshaped[k].append(v)

    energy_mean = {"values" : energy_reshaped["values"][0]}
    energy_mean["fe"] = np.mean(energy_reshaped["fe"], 0)
    energy_mean["fe_err"] = np.mean(energy_reshaped["fe_err"], 0)
    return total, energy_mean


@common.timer
def plot_data_shape_test_energy(data : tuple, energy_overflow : np.ndarray, book : plotting.PlotBook):

    """Plot energy-binned fractional count errors for the shape-test configurations.

        Args:
            data (tuple): Dictionary containing shape-test settings, energy-binned fractional errors, and uncertainties.
            energy_overflow (np.ndarray): Energy-bin coordinates used for the horizontal axis.
            book (plotting.PlotBook): PlotBook used to save each process figure.
    """
    def remove_zero(a):
        """Return the elements of a that are not zero.

            Args:
                a (Any): Array of varied parameter values from which zeros are excluded.

            Returns:
                np.ndarray: Elements of a that are not zero.
        """
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
        plots.plt.subplots(len(remove_zero(unique_values[2])), len(remove_zero(unique_values[3])), figsize = [4.8 * len(remove_zero(unique_values[3])), 4.8 * len(remove_zero(unique_values[2]))], sharex = True, sharey = True)
        for p, ((ia, a), (ib, b)) in enumerate(itertools.product(enumerate(remove_zero(unique_values[2])), enumerate(remove_zero(unique_values[3])))):
            plots.plt.subplot(len(remove_zero(unique_values[2])), len(remove_zero(unique_values[3])), p+1)
            
            for i, j in enumerate(x):
                if (j[2] == a) and (j[3] == b):
                    plots.plot(energy_overflow, y[i], yerr = y_err[i], label = f"{x[i][0]}, {x[i][1]}", new_figure = False)
            plots.plt.fill_between(energy_overflow, 0, linestyle = "--", color = "k")

            if ia == len(remove_zero(unique_values[2])) - 1:
                plots.plt.xlabel("$N_{int}$(MeV)" + f"\n\n $x0$ : {b}")
            if ib == 0:
                plots.plt.ylabel(f"$\\alpha$ : {a}\n\n" + "fractional error")
        plots.plt.suptitle(f"$N_{{{process_map[d]}}}$", size = 20)
        plots.plt.tight_layout()
        book.save()
    return

def plot_data_shape_test(data, key, label, vmin = None, vmax = None):
    """Render a selected shape-test result as heatmaps over its varied parameters.

        Args:
            data (Any): DataFrame indexed by shape-test settings and containing the requested result column.
            key (Any): Column name to render as the heatmap value.
            label (Any): Label used for the figure and colorbar.
            vmin (Any): Optional minimum value for the heatmap color scale.
            vmax (Any): Optional maximum value for the heatmap color scale.
    """
    def remove_zero(a):
        """ Remove zero.

            Args:
                a (Any): Array of varied parameter values from which zeros are excluded.

            Returns:
                np.ndarray: Elements of a that are not zero.
        """
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

    fig, axes = plots.plt.subplots(len(remove_zero(unique_values[2])), len(remove_zero(unique_values[3])), figsize = [4.8 * len(remove_zero(unique_values[3])), 4.8 * len(remove_zero(unique_values[2]))], sharex = True, sharey = True)

    for p, ((ia, a), (ib, b)) in enumerate(itertools.product(enumerate(remove_zero(unique_values[2])), enumerate(remove_zero(unique_values[3])))):
        plots.plt.subplot(len(remove_zero(unique_values[2])), len(remove_zero(unique_values[3])), p+1)
        im = np.zeros(grid[0].shape)
        for i, j in zip(x, y):
            if i[2] not in [0, a] : continue
            if i[3] not in [0, b] : continue
            loc = np.argwhere((grid[0] == i[0]) & (grid[1] == i[1]))[0]
            im[loc[0], loc[1]] = j

        im_c = plots.plt.imshow(np.where(im == 0, np.nan, im), cmap = "plasma", vmin = vmin, vmax = vmax)

        plots.plt.grid(False)
        if ia == len(remove_zero(unique_values[2])) - 1:
            plots.plt.xlabel("$n_{-}$" + f"\n\n $x0$ : {b}")
        if ib == 0:
            plots.plt.ylabel(f"$\\alpha$ : {a}\n\n" + "$n_{+}$")


        plots.plt.yticks(ticks = range(len(unique_values[0])), labels = unique_values[0])
        plots.plt.xticks(ticks = range(len(unique_values[1])), labels = unique_values[1])

        for (i, j), z in np.ndenumerate(im):
            if z != 0:
                plots.plt.gca().text(j, i, f"{z:.2f}", ha='center', va='center', fontsize = 12, color = "limegreen", weight = "bold")
        plots.plt.grid(False)


    plots.plt.subplots_adjust(wspace = 0, hspace = 0)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
    fig.suptitle(label, size = 20)
    fig.colorbar(im_c, label = label, cax = cbar_ax)
    fig.tight_layout()

@common.timer
def plot_cross_check_results_shape(results_total, results_bins, energy_overflow : np.ndarray, single_bin : bool, book : plotting.PlotBook):

    """Plot fitted process-scale heatmaps and, for multi-bin studies, energy-dependent errors.

        Args:
            results_total (Any): Process-indexed total result tables for each shape-test setting.
            results_bins (Any): Process-indexed energy-binned fractional errors and uncertainties.
            energy_overflow (np.ndarray): Energy-bin coordinates used for energy-resolved plots.
            single_bin (bool): Whether the fit uses one combined bin, suppressing energy-resolved plots.
            book (plotting.PlotBook): PlotBook used to save the generated figures.
    """
    for p in process_map:
        plot_data_shape_test(results_total, f"mu_{p}", f"$\\mu_{{{process_map[p]}}}$")
        book.save()
    if single_bin is False:
        plot_data_shape_test_energy(results_bins, energy_overflow, book)

    return


@common.timer
def predicted_counts_summary(template_counts : float, model : pyhf.Model, test_name : str, model_name : str, workdirs : str, single_bin : bool):
    """Load processed fit results and find the largest absolute fractional errors and their uncertainties.

        Args:
            template_counts (float): Number of entries in the reference template used to normalize process counts.
            model (pyhf.Model): Model used to compute post-fit predictions.
            test_name (str): Test-name component of the result directory and file names.
            model_name (str): Model-name component of the result directory and file names.
            workdirs (str): Directories containing saved results for the test and model.
            single_bin (bool): Whether the fit used a single combined bin.

        Returns:
            tuple: Energy-binned maxima with uncertainties and total fractional-error maxima with uncertainties.
    """
    pr, pr_energy = calculate_results_from_file(workdirs, test_name, model_name, template_counts, model, single_bin)    

    n_fe_total_max = {}
    n_fe_max = {}
    for r in pr:
        fractional_error = pr[r].filter(regex = "fe_total")
        fractional_error_unc = pr[r].filter(regex = "fe_err_total")
        fe = pr_energy[r]["fe"]
        fe_err = pr_energy[r]["fe_err"]

        n_fe_total_max[r] = (
            np.max(abs(fractional_error.values), 0),
            np.where(np.argmax(abs(fractional_error.values), 0) == 0, fractional_error_unc.values[0, :], fractional_error_unc.values[1, :])
            )

        fe = np.nan_to_num(fe, posinf = 0, nan = 0, neginf = 0)
        fe_err = np.nan_to_num(fe_err, posinf = 0, nan = 0, neginf = 0)        
        n_fe_max[r] = (
            np.max(abs(fe), 1),
            np.where(np.argmax(abs(fe), 1) == 0, fe_err[:, 0], fe_err[:, 1])
        )
    return n_fe_max, n_fe_total_max

def create_summary_tables(total_summaries, row_names):
    """Build numeric and formatted tables of per-process fractional errors and uncertainties.

        Args:
            total_summaries (Any): Mapping from process to arrays of fractional-error maxima and corresponding uncertainties.
            row_names (Any): Process names used as the output table rows.

        Returns:
            tuple: Numeric fractional-error table, uncertainty table, and formatted percentage table.
    """
    proc_tags = tags.exclusive_process_tags({k : None for k in dict.fromkeys(list(row_names) + list(total_summaries.keys()))})

    table_fe = pd.DataFrame({k : v[0] for k, v in total_summaries.items()}, index = row_names)
    table_err = pd.DataFrame({k : v[1] for k, v in total_summaries.items()}, index = row_names)

    table_str = {}
    for r, i, j in zip(row_names, table_fe.values, table_err.values):
        row = {}
        for c, v, e in zip(table_fe.columns, i, j):
            e_s = f"{100 * e:.1g}"
            v_s = str(round(100 * v, len(e_s.split(".")[-1])))
            row[proc_tags[c].name_simple] = f"{v_s} $\\pm$ {e_s}"
        table_str[proc_tags[r].name_simple + "(\\%)"] = row
    return table_fe, table_err, pd.DataFrame(table_str)


def save_summary_tables(directory : str, tables : tuple[pd.DataFrame], name : str):
    """Write the numeric, uncertainty, and formatted summary tables as LaTeX files.

        Args:
            directory (str): Directory prefix where the LaTeX table files are written.
            tables (tuple[pd.DataFrame]): Fractional-error, uncertainty, and formatted tables, in that order.
            name (str): Filename prefix shared by the three saved tables.
    """
    names = ["fe", "unc", "fmt"]
    for n, t in zip(names, tables):
        t.style.to_latex(f"{directory}table_{name}_{n}.tex")
    return


def summary(directory : str, test_name : str, model_name : str, model : pyhf.Model, energy_overflow : np.ndarray, template_counts : float, single_bin : bool, book : plotting.PlotBook = plotting.PlotBook.null(), ymax = None):

    """Create fractional-error summary tables and plots, returning the selected plot y-axis maximum.

        Args:
            directory (str): Directory prefix for the saved summary tables.
            test_name (str): Test-name component used to load study results and label plots.
            model_name (str): Model-name component used to load study results.
            model (pyhf.Model): Fitted model used to calculate predicted-count summaries.
            energy_overflow (np.ndarray): Energy-bin coordinates used for energy-resolved plots.
            template_counts (float): Number of entries in the reference template used to normalize process counts.
            single_bin (bool): Whether the fit uses a single combined bin, suppressing energy-resolved plots.
            book (plotting.PlotBook): PlotBook used to save the generated plots.
            ymax (Any): Optional maximum fractional error for the summary plot axes.

        Returns:
            Any: Plot y-axis maximum, supplied by the caller or derived from the largest energy-binned fractional error.
    """
    n_fe_max, n_fe_total_max = predicted_counts_summary(template_counts, model, test_name, model_name, args.workdir, single_bin)

    indices = ["absorption", "charge_exchange", "single_pion_production", "pion_production"]
    xlabel = "$KE$ (MeV)"

    tables_n = create_summary_tables(n_fe_max, indices)
    save_summary_tables(directory, tables_n, "processes")
    print(tables_n[2])

    if ymax is None:
        ymax = np.max([n_fe_max[t][0] for t in n_fe_max])

    if single_bin is False:
        for i, p in plots.iter_multi_plot(indices):
            for j, t in enumerate(n_fe_max):
                y = n_fe_max[t][0]
                err = n_fe_max[t][1]
                plots.plot(energy_overflow, y[i], yerr = err[i], color = f"C{j}", label = common.remove_(t), ylabel = "fractional error in fitted counts", xlabel = xlabel, title = f"process : {common.remove_(p)}", new_figure = False)
            plots.plt.legend(title = f"{test_name} test")
            plots.plt.ylim(0, 1.1 * ymax)
        book.save()

        for i, t in plots.iter_multi_plot(n_fe_max):
            for j, p in enumerate(indices):
                y = n_fe_max[t][0]
                err = n_fe_max[t][1]
                plots.plot(energy_overflow, y[j], yerr = err[j], xlabel = xlabel, ylabel = "$|f_{bs}|$", label = common.remove_(p), title = f"{test_name} test : {common.remove_(t)}", new_figure = False)
            plots.plt.legend(title = "process")
        book.save()
    return ymax


def plot_templates(templates_energy : np.ndarray, tempalates_mean_track_score : np.ndarray, energy_slices : slices.Slices, mean_track_score_bins : np.ndarray, template : cross_section.analysis_input.AnalysisInput, book : plotting.PlotBook = plotting.PlotBook.null()):
    """ Plot templates.

        Args:
            templates_energy (np.ndarray): Template count distributions over energy bins, separated by region and process.
            tempalates_mean_track_score (np.ndarray): Optional template count distributions over mean-track-score bins.
            energy_slices (slices.Slices): Energy slicing that supplies the energy-bin coordinates.
            mean_track_score_bins (np.ndarray): Edges for the optional mean-track-score histogram.
            template (cross_section.analysis_input.AnalysisInput): Template sample supplying exclusive-process tags and colors.
            book (plotting.PlotBook): PlotBook used to save the template figures.
    """
    process_tags = tags.exclusive_process_tags(template.exclusive_process)
    for j, c in plots.iter_multi_plot(templates_energy):
        for i, s in enumerate(c):
            plots.plot(energy_slices.pos_overflow, s/np.sum(templates_energy), color = process_tags.number[i].colour, label = f"$\\lambda_{{{process_map[j]},{process_map[i]}}}$", xlabel = f"$\\lambda_{{{process_map[j]},s}}$ (MeV)", ylabel = "normalised counts", style = "step", new_figure = False)
        plots.plt.title(f"region : {process_map[j]}")
        plots.plt.legend(title = "process")
    book.save()

    if tempalates_mean_track_score is not None:
        plots.plt.figure()
        for i, s in enumerate(tempalates_mean_track_score):
            plots.plot(common.bin_centers(mean_track_score_bins), s/np.sum(tempalates_mean_track_score), color = process_tags.number[i].colour, label = f"$\\lambda_{{t,{process_map[i]}}}$", xlabel = f"$\\lambda_{{t,s}}$", ylabel = "normalised counts", style = "step", new_figure = False)
        plots.plt.legend(loc = "upper left")
        book.save()
        book.close()
    return


def plot_total_channel(templates_energy : np.ndarray, tempalates_mean_track_score : np.ndarray, energy_slices : slices.Slices, mean_track_score_bins : np.ndarray, book : plotting.PlotBook = plotting.PlotBook.null()):
    """Plot summed template counts by energy slice and, when available, mean-track-score bin.

        Args:
            templates_energy (np.ndarray): Template count distributions over energy bins, separated by region and process.
            tempalates_mean_track_score (np.ndarray): Optional template count distributions over mean-track-score bins.
            energy_slices (slices.Slices): Energy slicing that supplies the energy-bin coordinates.
            mean_track_score_bins (np.ndarray): Edges for the optional mean-track-score histogram.
            book (plotting.PlotBook): PlotBook used to save the total-channel figures.
    """
    for j, c in plots.iter_multi_plot(templates_energy):
        plots.plot(energy_slices.pos_overflow, sum(c), xlabel = f"$n_{{{j}}}$ (MeV)", ylabel = "Counts", style = "bar", new_figure = False)
    book.save()

    if tempalates_mean_track_score is not None:
        plots.plot(common.bin_centers(mean_track_score_bins), sum(tempalates_mean_track_score), xlabel = f"$n_{{ts}}$", ylabel = "Counts", style = "bar")
        book.save()
    book.close()
    return


def calculate_results_from_file(workdirs, test_name, model_name, template_counts, model, single_bin):
    """Load and process saved fit-study results, averaging across workdirs and aligning nominal process rows.

        Args:
            workdirs (Any): Directories containing saved results for the test and model.
            test_name (Any): Test-name component of the result directory and file names.
            model_name (Any): Model-name component of the result directory and file names.
            template_counts (Any): Number of reference-template entries used to normalize process counts.
            model (Any): Fitted model used to calculate post-fit count differences.
            single_bin (Any): Whether to combine all bins when calculating total fractional errors.

        Returns:
            tuple: Process-indexed total result tables and energy-binned error dictionaries.
    """
    results_files = []
    for d in workdirs:
        directory = d + f"{test_name}_test_{model_name}/"
        rf = {}
        for i in os.listdir(directory):
            if "dill" in i:
                for k in target_map:
                    if k in i:
                        rf[target_map[k]] = i
        results_files.append(rf)

    results_total = {k : [] for k in results_files[0]}
    results_bins = {k : [] for k in results_files[0]}
    for d, files in zip(workdirs, results_files):
        for f in files:
            fit_result = io.read_dill(d + f"{test_name}_test_{model_name}/" + files[f])
            r = process_results(template_counts, fit_result["results"], fit_result["true_counts"], fit_result["expected_mus"], model, single_bin)
            results_total[f].append(r[0])
            results_bins[f].append(r[1])

    for k in list(results_total.keys()):
        if len(workdirs) > 1:
            results_total[k], results_bins[k] = mean_processed_results(results_total[k], results_bins[k])
        else:
            results_total[k], results_bins[k] = results_total[k][0], results_bins[k][0]

    nominal = results_total['charge_exchange'].loc[1]
    for p in results_total:
        results_total[p].loc[1] = nominal
    return results_total, results_bins



@common.timer
def main(args : argparse.Namespace):
    """ Handle main.

        Args:
            args (argparse.Namespace): Input value for args.
    """
    plotting.styles.PlotStyler.set_plot_style(extend_colors = True, dark = True, font_scale = 1.4)
    toys = toy.Toy(file = args.template)
    args.template = cross_section.analysis_input.AnalysisInput.create_analysis_input_toy(toys)

    mean_track_score_bins = np.linspace(0, 1, 21, True)

    models = {}
    if args.fit["mean_track_score"] is True:
        models["track_score"], templates_energy, tempalates_mean_track_score = RegionFit.CreateModel(args.template, args.energy_slices, mean_track_score_bins, True, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])
    else:
        models["normal"], templates_energy, tempalates_mean_track_score = RegionFit.CreateModel(args.template, args.energy_slices, None, True, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])


    if args.toy_data_config:
        print("Running tests")
        if args.out is None:
            args.out = "region_fit_studies/"
        os.makedirs(args.out, exist_ok = True)
        for m in models:

            ts_bins = mean_track_score_bins if m == "track_score" else None

            if "normalisation" not in args.skip:
                os.makedirs(args.out + f"normalisation_test_{m}/", exist_ok = True)
                normalisation_test(
                    args.out + f"normalisation_test_{m}/",
                    args.toy_data_config, models[m],
                    args.template,
                    args.energy_slices,
                    ts_bins,
                    args.fit["single_bin"])

            if "shape" not in args.skip:
                xs_sim = cross_section.geant.GeantCrossSections(energy_range = [0, max(args.energy_slices.pos) + args.energy_slices.width])

                os.makedirs(args.out + f"shape_test_{m}/", exist_ok = True)
                shape_test_new(
                    args.out + f"shape_test_{m}/",
                    args.toy_data_config,
                    models[m],
                    args.template,
                    ts_bins,
                    xs_sim,
                    args.energy_slices,
                    args.fit["single_bin"])
            if "pulls" not in args.skip:
                pull_results = pull_study_fast(toys, int(5E6), int(1E6), args, args.energy_slices, mean_track_score_bins if m == "track_score" else None, 500)
                os.makedirs(args.out + f"pull_test_{m}/", exist_ok = True)
                io.write_hdf5(pull_results, args.out + f"pull_test_{m}/" + "pull_results.hdf5")


    if args.workdir:
        print("Making test results")

        if (args.out is None):
            if len(args.workdir) == 1:
                outdir = args.workdir[0]
            else:
                outdir = "fit_studies_results_merged/"
        else:
            outdir = args.out

        if args.fit["single_bin"] is False:
            with plotting.PlotBook(outdir + "templates") as book:
                plot_templates(templates_energy, tempalates_mean_track_score, args.energy_slices, mean_track_score_bins, args.template, book)

            with plotting.PlotBook(outdir + "observation_exmaple") as book:
                plot_total_channel(templates_energy, tempalates_mean_track_score, args.energy_slices, mean_track_score_bins, book)

            with plotting.PlotBook(outdir + "xs_curves") as book:
                plot_shape_examples(args.energy_slices, book)

        if (hasattr(args.template, "exclusive_process")) and (type(args.template.exclusive_process) == dict):
            data_tags = tags.exclusive_process_tags({k : None for k in args.template.exclusive_process.keys()})
        else:
            data_tags = tags.exclusive_process_tags({k : None for k in args.template.regions.keys()})

        test = ["shape", "normalisation", "pulls"] 

        template_counts = sum(args.template.inclusive_process)

        ymax = {"shape" : 1.5, "normalisation" : 0.6}

        for t in test:
            if t in args.skip: continue
            for m in models:
                if t == "pulls":
                    with plotting.PlotBook(outdir + f"pull_test_{m}/pulls.pdf", True) as book:
                        pull_results = io.read_hdf5(outdir + f"pull_test_{m}/" + "pull_results.hdf5")

                        pulls = (pull_results["bestfit"] - (pull_results["expected"] / pull_results["scale"])) / pull_results["uncertainty"]
                        res = (pull_results["scale"] * pull_results["bestfit"]) - pull_results["expected"]
                        fe = ((pull_results["scale"] * pull_results["bestfit"]) - pull_results["expected"]) / pull_results["expected"]
                        fe_err = (pull_results["scale"] * pull_results["uncertainty"]) / pull_results["expected"]

                        mean_res = {}
                        mean_fe = {}
                        mean_fe_err = {}
                        means = {}
                        stds = {}

                        for c in pulls.columns:
                            plots.plt.figure()
                            y, edges = np.histogram(pulls[c], bins = 10)
                            x = (edges[1:] + edges[:-1]) / 2
                            popt, perr = fitting.fit(x, y, np.sqrt(y), fitting.gaussian, plot = True, plot_style = "hist", method = "trf", xlabel = "$\\theta$", ylabel = "Number of entries")
                            book.save()

                        for _, k in plots.iter_multi_plot(pulls.columns):
                            mean_res[folder[k]] = f"{np.mean(res[k]):.3g}"
                            mean_fe[folder[k]] = f"{np.mean(fe[k]):.3g}"
                            mean_fe_err[folder[k]] = f"{np.mean(fe_err[k]):.3g}"
                            mean = np.mean(pulls[k])
                            std = np.std(pulls[k])
                            sem = std / np.sqrt(len(pulls[k]))
                            p = len(f"{sem:.1g}") - 2
                            print(folder[k])
                            means[folder[k]] = f"{mean:.{p}g} $\\pm$ {sem:.1g}"
                            stds[folder[k]] = f"{std:.3g}"
                            # plots.plot_hist(pulls[k], bins = 10, title = f"$\\mu_{{{data_tags[k].name_simple}}}$ | mean : {mean:.3g} $\\pm$ {sem:.1g} | std.dev : {std:.3g} ", xlabel = "$\\theta$", new_figure = False)
                            plots.plot_hist(pulls[k], bins = 10, xlabel = "$\\theta$", new_figure = False)

                        book.save()
                        for k in pulls.columns:
                            mean = np.mean(pulls[k])
                            std = np.std(pulls[k])
                            sem = std / np.sqrt(len(pulls[k]))
                            plots.plt.figure(figsize = (6.4, 5))
                            # plots.plot_hist(pulls[k], bins = 10, title = f"$\\mu_{{{data_tags[k].name_simple}}}$ | mean : {mean:.3g} $\\pm$ {sem:.1g} | std.dev : {std:.3g} ", xlabel = "$\\theta$", new_figure = True)
                            plots.plot_hist(pulls[k], bins = 10, xlabel = "$\\theta$", new_figure = False)
                            # plots.plt.ylabel(plots.plt.gca().yaxis.get_label_text(), loc = "bottom")
                            book.save()

                        table = pd.DataFrame({"mean" : means, "st.dev" : stds, "mean residual" : mean_res, "mean fractional error" : mean_fe, "mean_fractional_error_error" : mean_fe_err}).T
                        table.style.to_latex(outdir + f"pull_test_{m}/pulls.tex")


                else:

                    results_total, results_bins = calculate_results_from_file(args.workdir, t, m, template_counts, models[m], args.fit["single_bin"])

                    for f in results_total:
                        os.makedirs(outdir + f"{t}_test_{m}/", exist_ok = True)
                        with plotting.PlotBook(outdir + f"{t}_test_{m}/" + f + ".pdf", True) as pdf:
                            if (t == "shape"):
                                plot_cross_check_results_shape(results_total[f], results_bins[f], args.energy_slices.pos_overflow, args.fit["single_bin"], pdf)
                            else:
                                plot_cross_check_results(f"$\\mathcal{{N}}_{{{data_tags[f].name_simple}}}$", results_total[f], results_bins[f], args.energy_slices.pos_overflow, pdf, args.fit["single_bin"], [-0.02, 0.06])
                        plots.plt.close("all")
                
                    with plotting.PlotBook(outdir + f"{t}_test_{m}/summary_plots.pdf", True) as book:
                        summary(outdir + f"{t}_test_{m}/", t, m, models[m], args.energy_slices.pos_overflow, template_counts, args.fit["single_bin"], book, ymax = ymax[t])
                    plots.plt.close("all")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("app which performs cross checks for the region fit using toys.")
    
    application.ApplicationArguments.config(parser, True)

    parser.add_argument("--template", "-t", dest = "template", type = str, help = "toy template hdf5 file", required = True)

    parser.add_argument("--toy_data_config", "-d", dest = "toy_data_config", type = str, help = "json config for toy data", required = False)

    parser.add_argument("--workdir", "-w", dest = "workdir", nargs = "+", type = str, help = "work directories which contains output from this application, use this to remake plots without running all the tests again", required = False)
    parser.add_argument("--signal_process", dest = "signal_process", type = str, help = "signal process for background subtraction")

    parser.add_argument("--step", "-s", dest = "step", type = float, help = "step size for toy, if provided will override toy data config.")
    parser.add_argument("--events", "-n", dest = "events", type = float, help = "events for toy, if provided will override toy data config.")
    parser.add_argument("--seed", dest = "seed", type = int, help = "seed for toy, if provided will override toy data config.")

    parser.add_argument("--skip", dest = "skip", type = str, choices = ["normalisation", "shape", "pulls"], nargs = "+", default = [], help = "test to skip")

    application.ApplicationArguments.output(parser, None)

    args = application.ApplicationArguments.resolve_args(parser.parse_args(), False)

    if (not args.toy_data_config) and (not args.workdir):
        raise Exception("--toy_data_config or --workdir or both must be supplied")

    if args.events: args.events = int(args.events)

    if args.toy_data_config:
        args.toy_data_config = io.read_json(args.toy_data_config)
        override_config(args)

    if args.workdir:
        for i in range(len(args.workdir)):
            if args.workdir[i][-1] != "/": args.workdir[i] = args.workdir[i] + "/"
        if not args.signal_process:
            raise Exception("--signal_process must be supplied if --workdirs is supplied")

    print(vars(args))
    main(args)