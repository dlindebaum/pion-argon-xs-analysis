#!/usr/bin/env python3
"""
Created on: 13/11/2023 21:54

Author: Shyam Bhuller

Description: Runs cross section measurement.
"""
import os
import argparse

import numpy as np
import awkward as ak
import pandas as pd
import cabinetry
import pickle
import warnings
from scipy.stats import truncnorm

from particle import Particle
from rich import print

import matplotlib.pyplot as plt

from pyunfold.callbacks import SplineRegularizer
from python.analysis import (
    cross_section, Plots,
    EnergyTools, Slicing, AnalysisInputs, SelectionTools)
from python.gnn import Fitter

known_gnn_theory_procs = {
    "true_proton_multiplicity": lambda evts: ak.sum(
        evts.trueParticles.pdg[..., 1:] == 2212, axis=-1),
    "reco_proton_multiplicity": lambda evts: ak.sum(
        evts.trueParticlesBT.pdg == 2212, axis=-1),
    "true_neutron_multiplicity": lambda evts: ak.sum(
        trueParticles.pdg[..., 1:] == 2112, axis=-1)}
known_gnn_theory_sizes = {
    "true_proton_multiplicity": 0.2,
    "reco_proton_multiplicity": 0.2,
    "true_neutron_multiplicity": 0.2}
known_upstream_methods = [
    # "function",
    "binned", "binned_init_reduced"]
    # "binned_multi_dim_reduced"]

label_map = {"toy" : "toy", "pdsp" : "ProtoDUNE SP"}

process_labels = {"absorption": "abs", "charge_exchange" : "cex", "single_pion_production" : "spip", "pion_production" : "pip"}

# def PlotKEs(
#         mc : cross_section.AnalysisInput,
#         data : cross_section.AnalysisInput,
#         args : argparse.Namespace,
#         book : Plots.PlotBook):
#     def PlotDataMCTruth(mc, data, truth, bins, xlabel, x_range, norm, weights, mc_label = "MC reco", data_label = "Data reco", truth_label = "MC truth"):
#         Plots.PlotHistDataMC(data, mc, bins = bins, x_range = x_range, norm = norm, xlabel = xlabel, mc_labels = mc_label, data_label = data_label, mc_weights = weights)

#         y, edges = np.histogram(np.array(truth), bins, range = x_range, weights = weights)
#         Plots.Plot(cross_section.bin_centers(edges), args.norm * y, label = truth_label, style = "step", color = "C2", newFigure = False)

#         h, l = Plots.plt.gca().get_legend_handles_labels()
#         Plots.plt.legend(h + [Plots.matplotlib.patches.Rectangle((0,0), 0, 0, fill = False, edgecolor='none', visible=False)], l + [f"norm: {args.norm:.3g}"], labelspacing = 0.25,  columnspacing = 0.25, ncols = 2)
#         return

#     PlotDataMCTruth(mc.KE_init_reco, data.KE_init_reco, mc.KE_init_true, 50, "$KE_{init}$ (MeV)", args.KE_init_range, args.norm, np.array(mc.weights))
#     book.Save()
#     PlotDataMCTruth(mc.KE_int_reco, data.KE_int_reco, mc.KE_int_true, 50, "$KE_{int}$ (MeV)", args.KE_int_range, args.norm, np.array(mc.weights))
#     book.Save()

#     return

# def CreateInitParams(
#         model : cross_section.pyhf.Model,
#         analysis_input : AnalysisInputs.AnalysisInput,
#         energy_slices : Slicing.Slices,
#         mean_track_score_bins : np.array) -> np.array:
#     """ Create initial parameters for the region fit, using the proportion of reco regions and template to get a rough estimate of the process rates.

#     Args:
#         model (cross_section.pyhf.Model): fit model
#         analysis_input (cross_section.AnalysisInput): analysis input
#         energy_slices (Slicing.Slices): energy slices
#         mean_track_score_bins (np.array): mean track score bins

#     Returns:
#         np.array[float]: initial parameter values
#     """
#     prefit_pred = cross_section.cabinetry.model_utils.prediction(model)
#     template_KE = [np.sum(prefit_pred.model_yields[i], 0) for i in range(len(prefit_pred.model_yields))][:-1]
#     input_data = cross_section.RegionFit.CreateObservedInputData(analysis_input, energy_slices, mean_track_score_bins)

#     init = model.config.suggested_init()
#     mu_init = [np.sum(input_data[i]) / np.sum(template_KE[i]) for i in range(len(template_KE))]
#     poi = [i for i in model.config.parameters if "mu_" in i]
#     poi_ind =  [model.config.par_slice(i).start for i in poi]
#     for i, v in zip(poi_ind, mu_init):
#         init[i] = v
#     return init


# def RegionFit(fit_input : cross_section.AnalysisInput, energy_slice : Slicing.Slices, mean_track_score_bins : np.array, template_input : cross_section.AnalysisInput | cross_section.pyhf.Model, suggest_init : bool = False, template_weights : np.array = None, return_fit_results : bool = False, mc_stat_unc : bool = False, single_bin : bool = False, fix_np : bool = False) -> cross_section.cabinetry.model_utils.ModelPrediction | cross_section.FitResults:
#     """ Fit model to analysis input to predict the normalaisations of each process.

#     Args:
#         fit_input (cross_section.AnalysisInput): observed data
#         energy_slice (Slicing.Slices): energy slices
#         mean_track_score_bins (np.array): mean track score bins
#         template_input (cross_section.AnalysisInput | cross_section.pyhf.Model): template sample or existing model
#         suggest_init (bool, optional): estimate normalisations ans use these as the initial values for the fit. Defaults to False.
#         template_weights (np.array, optional): weights for the mean track score. Defaults to None.
#         return_fit_results (bool, optional): return the raw fit results as  well as the prediction. Defaults to False.

#     Returns:
#         cross_section.cabinetry.model_utils.ModelPrediction | cross_section.FitResults: model prediction and or the raw fit result.
#     """
#     if type(template_input) == cross_section.AnalysisInput:
#         model = cross_section.RegionFit.CreateModel(template_input, energy_slice, mean_track_score_bins, False, template_weights, mc_stat_unc, True, single_bin)
#     else:
#         model = template_input

#     observed = cross_section.RegionFit.GenerateObservations(fit_input, energy_slice, mean_track_score_bins, model, single_bin = single_bin)

#     if suggest_init is True:
#         init_params = CreateInitParams(model, fit_input, energy_slice, mean_track_score_bins)
#     else:
#         init_params = None

#     result = cross_section.RegionFit.Fit(observed, model, init_params, [[0, np.inf]]*model.config.npars, verbose = False)

#     # redo fit, but fix the NPs to their already determined values
#     if (mc_stat_unc and fix_np) is True:
#         fix = ["mu" not in i for i in model.config.par_order]

#         index = [j for j, i in enumerate(model.config.par_order) if "mu" not in i]

#         if init_params is None:
#             init_params = np.ones_like(model.config.par_order, dtype = float)
#         init_params[index] = result.bestfit[index]
#         result = cross_section.RegionFit.Fit(observed, model, fix_pars = list(fix), init_params = list(init_params), par_bounds = [[0, np.inf]]*model.config.npars, verbose = False)

#     if return_fit_results is True:
#         return cross_section.cabinetry.model_utils.prediction(model, fit_results = result), result
#     else:
#         return cross_section.cabinetry.model_utils.prediction(model, fit_results = result)


# def BkgSubAllRegion(data : cross_section.AnalysisInput, energy_slices : Slicing.Slices, bkg, bkg_err):
#     N_int = data.NInteract(energy_slices, np.ones_like(data.outside_tpc_reco, dtype = bool))
#     N_int_ex = N_int - np.sum(bkg, 0)
#     N_int_ex_err = np.sqrt(N_int + np.sum(bkg_err**2, 0))
#     return N_int_ex, N_int_ex_err


# def BkgSubRegions(data : cross_section.AnalysisInput, energy_slices : Slicing.Slices, bkg, bkg_err):
#     processes = list(data.regions.keys())
    
#     N_int_regions = {k : data.NInteract(energy_slices, v, None, True) for k, v in data.regions.items()}
#     N_int_ex = {}
#     N_int_ex_err = {}

#     for p in processes:
#         N_int_ex[p] = N_int_regions[p] - np.sum(bkg[p], 0)
#         # N_int_ex_err[p] = cross_section.quadsum([np.sqrt(N_int_regions[p]), cross_section.quadsum(bkg_err[p], 0)], 0)
#         N_int_ex_err[p] = cross_section.quadsum(bkg_err[p], 0)
#     return N_int_ex, N_int_ex_err


# def BkgSingleBin(N_bkg_s : np.ndarray, N_bkg_err_s : np.ndarray, template : cross_section.AnalysisInput, templates_energy : list[np.ndarray], signal_process : str, mc_stat : bool):
#     bkg_mask = signal_process != np.array(template.region_labels)
#     labels = np.array(template.region_labels)[bkg_mask]

#     N_MC_cbs = templates_energy
#     N_MC = np.sum(N_MC_cbs)
#     N_MC_s = np.sum(np.sum(N_MC_cbs, 0), 1)
#     lambda_cbs = N_MC_cbs/N_MC
#     rel_scales = N_MC/N_MC_s

#     rel_lambda_bs = (rel_scales * np.sum(lambda_cbs, 0).T).T[bkg_mask]

#     N_bkg = (N_bkg_s * rel_lambda_bs)

#     N_bkg_err_fit_var = (N_bkg_err_s * rel_lambda_bs)**2

#     N_bkg_err_template_var = ((N_bkg_s.T)**2/N_MC_s[bkg_mask]).T * rel_lambda_bs * (1 + rel_lambda_bs)

#     if mc_stat:
#         N_bkg_var = N_bkg_err_fit_var + N_bkg_err_template_var
#     else:
#         N_bkg_var = N_bkg_err_fit_var

#     return N_bkg, np.sqrt(N_bkg_var), labels


# def PlotBkgRegions(energy_slices : Slicing.Slices, data : cross_section.AnalysisInput, bkg : dict[np.ndarray], bkg_err : dict[np.ndarray], bkg_label : dict[list[str]], book : Plots.PlotBook):

#     N_int_regions = {k : data.NInteract(energy_slices, v, None, True) for k, v in data.regions.items()}

#     tags = cross_section.Tags.ExclusiveProcessTags(None)

#     for r in N_int_regions:
#         cross_section.PlotXSHists(energy_slices, N_int_regions[r], xlabel = "$KE$ (MeV)", label = f"$N^{{Data}}_{{{tags[r].name_simple}}}$", title = cross_section.remove_(r).capitalize(), color = "k")
#         for b, be, bl in zip(bkg[r], bkg_err[r], bkg_label[r]):
#             cross_section.PlotXSHists(energy_slices, b, be, label = f"$N^{{bkg}}_{{{tags[bl].name_simple}}}$", newFigure = False, color = tags[bl].colour)
#         Plots.plt.legend(fontsize = 12, ncols = 1, loc = "upper left")
#         book.Save()
#     return


# def BackgroundSubtraction(data : cross_section.AnalysisInput, process : str, energy_slice : Slicing.Slices, postfit_pred : cross_section.cabinetry.model_utils.ModelPrediction = None, single_bin : bool = False, regions : bool = False, template : cross_section.AnalysisInput = None, mc_stat : bool = True, book : Plots.PlotBook = Plots.PlotBook.null) -> tuple[np.ndarray]:
#     """ Background subtraction using the fit if a fit result is specified.

#     Args:
#         data (cross_section.AnalysisInput): observed data
#         process (str): signal process
#         energy_slice (Slicing.Slices): energy slices
#         fit_results (cross_section.cabinetry.model_utils.ModelPrediction, optional): fit predictions. Defaults to None.
#         book (Plots.PlotBook, optional): plot book. Defaults to Plots.PlotBook.null.

#     Returns:
#         tuple[np.array]: true histograms (if data is mc), reco histograms postfit, error in reco hitograms postfit
#     """
#     if data.KE_init_true is not None:
#         histograms_true_obs = data.CreateHistograms(energy_slice, process, False, False)
#     else:
#         histograms_true_obs = None
#     histograms_reco_obs = data.CreateHistograms(energy_slice, process, True, False)
#     histograms_reco_obs_err = {k : np.sqrt(v) for k, v in histograms_reco_obs.items()}
    
#     templates_energy = cross_section.RegionFit.CreateKEIntTemplates(template, energy_slice, False, False, True)

#     if postfit_pred is not None:
#         if regions:
#             bkg, bkg_err = cross_section.RegionFit.EstimateBackgroundInRegions(postfit_pred, data)
#             if single_bin:
#                 bkg_b = {}
#                 bkg_err_b = {}
#                 bkg_label = {}
#                 for p in bkg:
#                     bkg_b[p], bkg_err_b[p], bkg_label[p] = BkgSingleBin(bkg[p], bkg_err[p], template, templates_energy, p, mc_stat)
#                 bkg = bkg_b
#                 bkg_err = bkg_err_b
#             KE_int_fit, KE_int_fit_err = BkgSubRegions(data, energy_slice, bkg, bkg_err)
#             if book is not None:
#                 PlotBkgRegions(energy_slice, data, bkg, bkg_err, bkg_label, book)
#         else:
#             print(f"signal: {process}")
#             bkg, bkg_err = cross_section.RegionFit.EstimateBackgroundAllRegions(postfit_pred, template, process)
#             if single_bin:
#                 bkg, bkg_err, bkg_label = BkgSingleBin(bkg, bkg_err, template, templates_energy, process, mc_stat)
#             KE_int_fit, KE_int_fit_err = BkgSubAllRegion(data, energy_slice, bkg, bkg_err)

#         if book is not None:
#             if data.exclusive_process is not None:
#                 energy_bins = np.sort(np.insert(energy_slice.pos, 0, energy_slice.max_pos + energy_slice.width))

#                 if regions:
#                     for i in KE_int_fit:
#                         actual = {l : data.NInteract(energy_slice, data.exclusive_process[l], mask = data.regions[i], weights = data.weights) for l in data.region_labels}
#                         actual_sig = actual[i]
#                         actual_bkg = sum(np.array(list(actual.values()))[i != np.array(data.region_labels)])
#                         cross_section.RegionFit.PlotPrefitPostFit(actual_sig, np.sqrt(actual_sig), KE_int_fit[i], KE_int_fit_err[i], energy_bins, f"$N^{{reco}}_{{int,{process_labels[i]}}}$ (MeV)")
#                         book.Save()
#                         cross_section.RegionFit.PlotPrefitPostFit(actual_bkg, np.sqrt(actual_bkg), np.sum(bkg[i], 0), np.sum(bkg_err[i], 0), energy_bins, "$N^{reco}_{int,bkg}$ (MeV)")
#                         book.Save()

#                 else:
#                     actual = {l : data.NInteract(energy_slice, data.exclusive_process[l], weights = data.weights) for l in data.region_labels}
#                     actual_sig = actual[process]
#                     actual_bkg = sum(np.array(list(actual.values()))[process != np.array(data.region_labels)])
#                     cross_section.RegionFit.PlotPrefitPostFit(actual_sig, np.sqrt(actual_sig), KE_int_fit, KE_int_fit_err, energy_bins, f"$N^{{reco}}_{{int,{process_labels[process]}}}$ (MeV)")
#                     book.Save()
#                     cross_section.RegionFit.PlotPrefitPostFit(actual_bkg, np.sqrt(actual_bkg), np.sum(bkg, 0), np.sum(bkg_err, 0), energy_bins, "$N^{reco}_{int,bkg}$ (MeV)")
#                     book.Save()

#         if regions:
#             histograms_reco_obs["int_ex"] = {k : np.where(v < 0, 0, v) for k, v in KE_int_fit.items()}
#         else:
#             histograms_reco_obs["int_ex"] = np.where(KE_int_fit < 0, 0, KE_int_fit)

#         histograms_reco_obs_err["int_ex"] = KE_int_fit_err

#     return histograms_true_obs, histograms_reco_obs, histograms_reco_obs_err


# def PlotDataBkgSub(hist_data : dict[np.ndarray], hist_data_err : dict[np.ndarray], mc : cross_section.AnalysisInput, regions : bool, signal_process : str, energy_slices : Slicing.Slices, scale : float, sample_name : str, data_label = "data", mc_label = "mc", book : Plots.PlotBook = Plots.PlotBook.null):
#     labels = {"init" : "$N_{init}$", "int" : "$N_{int}$", "int_ex" : "$N_{int, ex}$", "inc" : "$N_{inc}$"}

#     histograms_mc_reco = mc.CreateHistograms(energy_slices, signal_process, True, False)

#     if regions:
#         histograms_mc_reco.pop("int_ex")
#         N_ex_MC = {l : mc.NInteract(energy_slices, mc.exclusive_process[l], mask = mc.regions[l], weights = mc.weights) for l in mc.regions}
#         for i in histograms_mc_reco:
#             cross_section.PlotXSHists(energy_slices, histograms_mc_reco[i], None, True, scale, label = mc_label, color = "C1", title = labels[i])
#             cross_section.PlotXSHists(energy_slices, hist_data[i], hist_data_err[i], True, newFigure = False, label = data_label, color = "k")
#             Plots.plt.legend(loc = "upper left")
#             book.Save()
#         for i in N_ex_MC:
#             cross_section.PlotXSHists(energy_slices, N_ex_MC[i], title = f"$N_{{int, {process_labels[i]}}}$", label = mc_label, color = "C1", scale = scale)
#             cross_section.PlotXSHists(energy_slices, hist_data["int_ex"][i], hist_data_err["int_ex"][i], newFigure = False, label = data_label, color = "k")
#             Plots.plt.legend(loc = "upper left")
#             book.Save()
#     else:
#         for i in hist_data:
#             cross_section.PlotXSHists(energy_slices, histograms_mc_reco[i], scale = scale, title = labels[i], label = mc_label, color = "C1")
#             cross_section.PlotXSHists(energy_slices, hist_data[i], hist_data_err[i], newFigure = False, label = data_label, color = "k")
#             book.Save()

#         cross_section.PlotXSHists(energy_slices, histograms_mc_reco["int_ex"], scale = scale, title = labels["int_ex"]  + ":" + sample_name, label = mc_label, color = "C1")
#         cross_section.PlotXSHists(energy_slices,hist_data["int_ex"], hist_data_err["int_ex"], newFigure = False, label = f"{data_label}, background subtracted", color = "k")
#         Plots.plt.legend(loc = "upper left")
#         book.Save()

#     return

def project_unfolding_matrix(unfolding_mat, reco_multi_dim, reco_binner, true_binner):
    reco_init_int_slicer = reco_binner.slice_by_hist(slice(None), slice(None), 0)
    truth_init_int_slicer = true_binner.slice_by_hist(slice(None), slice(None), 0)
    shape_uf_mat = unfolding_mat[truth_init_int_slicer, ...]
    shape_uf_mat = shape_uf_mat[..., reco_init_int_slicer]
    reco_2d = reco_multi_dim[reco_init_int_slicer]
    # reco_int = reco_2d.sum(axis=0, keepdims=True)
    # int_norm = reco_2d/reco_int
    # norm_uf_mat = shape_uf_mat * int_norm[np.newaxis, np.newaxis, ...]
    # int_uf_mat = norm_uf_mat.sum(axis=(0,2))
    # # Estimating variance as Np/N iwth var(Np) = Np(1-p) and var(N) = N
    # var_p = int_uf_mat
    # # Alternate: Estimate error as pure binomial
    # # var(p_i) = var(Np_i)/N^2 = p_i(1-p_i)/N
    # # var_p = int_uf_mat * (1-int_uf_mat)
    # int_err_mat = np.nan_to_num(np.sqrt(var_p/reco_int), nan=0)

    # Interation-interaction migration given by Np with
    #   N as number of elements in original bin,
    #   and p migration probability.
    # Interaction matrix is then this / N (with N as orignial interactions)
    inte_migration = np.sum(
        shape_uf_mat * reco_2d[np.newaxis, np.newaxis, ...],
        axis=(0,2))
    n_inte = reco_2d.sum(axis=0)
    int_uf_mat = inte_migration/(n_inte[np.newaxis, :])
    # For error, assume N~Poisson, Np~Multinomial
    # Var in (Np)/N is then p(1-p)/N + p^2/N
    inte_ps = shape_uf_mat.sum(axis=(0,2))
    int_var_mat = inte_ps/(n_inte[np.newaxis, :])
    return int_uf_mat, np.sqrt(int_var_mat)
    # # Sum over reco_2d not count_migrations, since some counts
    # #   will have migrated to the impure bin and disappeared
    # alt_uf_mat = count_migrations/(reco_2d.sum(axis=0)[np.newaxis, :])
    # alt_err_mat = np.sqrt(shape_uf_mat.sum(axis=(0,2))/(reco_2d.sum(axis=0)[np.newaxis, :]))
    # # alt_uf_mat = np.sum(
    # #     shape_uf_mat * reco_2d[np.newaxis, np.newaxis, ...],
    # #     axis=(0,2)) / (reco_2d.sum(axis=0)[..., np.newaxis])
    # # alt_err_mat = np.sum(
    # #     shape_uf_mat * reco_2d[np.newaxis, np.newaxis, ...],
    # #     axis=(0,2)) / ((reco_2d.sum(axis=0)**2)[..., np.newaxis])
    # print(int_uf_mat)
    # print(alt_uf_mat)
    # print(int_err_mat/int_uf_mat)
    # print(alt_err_mat/alt_uf_mat)
    # print(alt_err_mat/alt_uf_mat >= int_err_mat/int_uf_mat)

    # return int_uf_mat, int_err_mat
    # # int_norm = reco_2d/reco_2d.sum(axis=0, keepdims=True)
    # # int_uf_mat = shape_uf_mat * int_norm[np.newaxis, np.newaxis, ...]
    # true_counts = shape_uf_mat * reco_2d[np.newaxis, np.newaxis, ...]
    # int_norm_uf_mat = true_counts/true_counts.sum(axis=2, keepdims=True)
    # int_uf_mat = int_norm_uf_mat.sum(axis=(0,2))
    # # Estimating error as pure binomial
    # # var(p_i) = var(Np_i)/N^2 = p_i(1-p_i)/N
    # int_counts = reco_2d.sum(axis=0)
    # var_p = int_uf_mat * (1-int_uf_mat)
    # int_err_mat = np.sqrt(var_p / (int_counts[np.newaxis, :]))
    # return int_uf_mat, int_err_mat
    # # return int_uf_mat.sum(axis=(0,2))

def extra_inte_purity_from_uf_mat(unfolding_mat, reco_multi_dim, reco_binner):
    uf_counts = unfolding_mat * reco_multi_dim[np.newaxis, :]
    mdim_purity_counts = uf_counts[-1, :]
    reco_init_int_slicer = reco_binner.slice_by_hist(slice(None), slice(None), 0)
    shape_purity_counts = mdim_purity_counts[reco_init_int_slicer]
    reco_2d = reco_multi_dim[reco_init_int_slicer]
    return shape_purity_counts/reco_2d.sum(axis=0, keepdims=True)

def project_unfolding_matrix_no_purity(unfolding_mat, reco_multi_dim, reco_binner, true_binner):
    no_purity_fracs = 1-unfolding_mat[-1, :]
    no_purity_mat = unfolding_mat/(no_purity_fracs[np.newaxis, :])
    reco_init_int_slicer = reco_binner.slice_by_hist(slice(None), slice(None), 0)
    truth_init_int_slicer = true_binner.slice_by_hist(slice(None), slice(None), 0)
    shape_uf_mat = no_purity_mat[truth_init_int_slicer, ...]
    shape_uf_mat = shape_uf_mat[..., reco_init_int_slicer]
    no_purity_reco = reco_multi_dim * no_purity_fracs
    reco_2d = no_purity_reco[reco_init_int_slicer]
    int_norm = reco_2d/reco_2d.sum(axis=0, keepdims=True)
    int_uf_mat = shape_uf_mat * int_norm[np.newaxis, np.newaxis, ...]
    return int_uf_mat.sum(axis=(0,2))

def calc_proc_purity_estimation(est_impure_frac, args, true_binner):
    proc_effs = args.mc_efficiencies["process"]
                                        # [:-1] removes purity bin
    pure_counts = {k: list(p["truth_pure_count"].values())[-1][:-1]
                   for k, p in proc_effs.items()}
    all_counts = {k: list(p["truth_all_count"].values())[-1][:-1]
                   for k, p in proc_effs.items()}
    impure_counts = {k: all_counts[k] - pure_counts[k]
                     for k in pure_counts.keys()}
    tot_all = sum(all_counts.values())
    tot_impure = sum(impure_counts.values())
    impure_frac = tot_impure/tot_all
    reweight_impure_frac = est_impure_frac/impure_frac
    est_impure_counts = {k: v * reweight_impure_frac
                         for k, v in impure_counts.items()}
    slicer = true_binner.slice_all()
    sum_ax = (0,2) if true_binner.has_tpc else 0
    return {k: v[slicer].sum(axis=sum_ax)
            for k, v in est_impure_counts.items()}

def calc_proc_efficiency_estimation(est_impure_frac, args, true_binner):
    proc_effs = args.mc_efficiencies["process"]
    init_counts = {k: p["truth_pure_count"]["FiducialStart"]
                   for k, p in proc_effs.items()}
    end_counts = {k: list(p["truth_pure_count"].values())[-1]
                  for k, p in proc_effs.items()}
    slicer = true_binner.slice_all()
    sum_ax = (0,2) if true_binner.has_tpc else 0
    init_count_shaped = {k: v[slicer].sum(axis=sum_ax)
                         for k, v in init_counts.items()}
    end_count_shaped = {k: v[slicer].sum(axis=sum_ax)
                         for k, v in end_counts.items()}
    return {k: init_count_shaped[k]/end_count_shaped[k]
            for k in init_count_shaped.keys()}

def unfold_interactions(int_yields, int_errs, int_uf_mat, int_err_mat):
    # Shape to the number of processes
    shape_uf_mat = int_uf_mat[..., np.newaxis]
    contribs = shape_uf_mat * int_yields
    truth_yields = np.sum(contribs, axis=1)
    # To avoid reducing the uncertainty fraction,
    #   we estimate the new variance as the mean of the indiviudial variances
    int_var = int_errs**2
    weighted_var = np.sum(shape_uf_mat * int_var, axis=1)
    truth_errs = np.sqrt(weighted_var)
    return truth_yields, truth_errs
    yield_frac_vars = (int_errs/int_yields)**2
    # print("Yield frac errs:")
    # print(np.sqrt(yield_frac_vars))
    uf_frac_vars = np.nan_to_num(int_err_mat[..., np.newaxis]/shape_uf_mat, nan=0)**2
    # print("UF frac errs:")
    # print(np.sqrt(uf_frac_vars))
    # print(yield_frac_vars)
    # print(uf_frac_vars)
    multi_errs = (contribs**2) * (yield_frac_vars+uf_frac_vars)
    truth_errs = np.sqrt(np.sum(multi_errs, axis=1))
    # print("Truth frac errs:")
    # print(truth_errs/truth_yields)
    # print("Raws")
    # print(truth_errs)
    # print(np.sqrt(np.sum((contribs**2) * (yield_frac_vars), axis=1)))
    # print(np.sqrt(np.sum((contribs**2) * (uf_frac_vars), axis=1)))
    # truth_errs = np.sqrt(np.sum(
    #     contribs * ()
    #     (shape_uf_mat**2) * (int_errs**2),
    #     axis=1))
    return truth_yields, truth_errs

def yields_to_fracs(yields, errs):
    yield_sums = yields.sum(axis=-1, keepdims=True)
    fracs = yields/yield_sums
    error_terms = (np.eye(3)[np.newaxis, :, :]
                   - fracs[..., np.newaxis])**2
    frac_errs = np.sqrt(np.sum(
        errs[:, np.newaxis, :]**2 * error_terms,
        axis=-1))/yield_sums
    return fracs, frac_errs

def gen_efficiency_correction_func(args, has_purity=True):
    """Returns a function with signature:
    corrector(unfolded_bins, unfolded_errs)
    Which returns the efficiency corrected counts, and associated error."""
    # use_pre = pre_counts[:-1-int(has_purity)]
    # use_post = post_counts[:-1-int(has_purity)]
    mc_true_eff = args.mc_efficiencies["truth"]
    sel_efficiency = list(mc_true_eff["efficiency"].values())[-1]
    pre_counts = mc_true_eff["count"]["FiducialTruth"]
    post_counts = list(mc_true_eff["count"].values())[-1]
    assert (list(mc_true_eff["efficiency"].keys())[-1]
            ==  list(mc_true_eff["count"].keys())[-1])
    multiplier_mask = np.logical_and(pre_counts != 0, post_counts != 0)
    multi_factor = 1/sel_efficiency
    zeros_mask = sel_efficiency == 0
    final_mc_count = np.sum(post_counts)
    zeros_mc_factor = pre_counts/final_mc_count
    # Where pre_count is 0, don't change anything

    # Initial overestimated as
    # var(Np/N) = p(1-p)/N + p^2/N = p/N => var(p)/p^2 = 1/Np
    # Correct method
    # var(p) = p(1-p)/N => var(p)/p^2 = (1-p)/Np which is smaller
    multi_frac_vars = (1/post_counts) - (1/pre_counts)
    # multi_frac_vars = (1-sel_efficiency)/post_counts
    # Treat additions as Poisson
    add_self_term = zeros_mc_factor / final_mc_count
    add_final_term = (zeros_mc_factor**2) / final_mc_count
    def corrector(unfolded_bins, unfolded_errs):
        data_zeros = unfolded_bins == 0
        new_mul_mask = np.logical_and(multiplier_mask,
                                      np.logical_not(data_zeros))
        new_zero_mask = np.logical_or(zeros_mask, data_zeros)
        updated_bins = unfolded_bins.copy()
        tot_count = np.sum(unfolded_bins)
        updated_errs = unfolded_errs.copy()
        uf_vars = unfolded_errs**2
        sum_var = np.sum(uf_vars)
        updated_bins[new_mul_mask] = (updated_bins[new_mul_mask]
                                      *multi_factor[new_mul_mask])
        updated_errs[new_mul_mask] = updated_bins[new_mul_mask] * np.sqrt(
            multi_frac_vars[new_mul_mask]
            + ((unfolded_errs/unfolded_bins)**2)[new_mul_mask])
        new_zero_factor = zeros_mc_factor[new_zero_mask]
        updated_bins[new_zero_mask] = (
            updated_bins[new_zero_mask] + tot_count*new_zero_factor)
        updated_errs[new_zero_mask] = (
            uf_vars[new_zero_mask] * ((1 + new_zero_factor)**2)
            + (sum_var - uf_vars[new_zero_mask]) * (new_zero_factor**2)
            + add_self_term[new_zero_mask] * (tot_count**2)
            + add_final_term[new_zero_mask] * (tot_count**2))
        if has_purity:
            updated_bins[-1] = unfolded_bins[-1]
        return updated_bins, updated_errs
    return corrector

def get_frac_errs(err_multiplier, multi_dim_hist, multi_binner, axis=0):
    if isinstance(err_multiplier, np.ndarray):
        assert err_multiplier.size == multi_binner.n_bins_init, (
            "Error count does not match bin count")
        bin_inds = np.arange(multi_binner.n_bins_init)
        base_errs = np.ones_like(multi_binner.slice_all())
        shape = [1] * len(base_errs.shape)
        shape[axis] = -1
        shape = tuple(shape)
        shape_errs = np.reshape(err_multiplier, shape) * base_errs
        err = shape_errs[multi_binner.corr_hist_to_flat_multi_dim_slicer()]
    else:
        err = np.full_like(multi_dim_hist, err_multiplier)
    return err

def indices_to_weights_only(
        mdim_up_down, weights, mdim_binner):
    bins_1d = np.arange(mdim_binner.n_bins)
    weights_1d = []
    for up, down in mdim_up_down:
        up_1d = np.histogram(up, bins=bins_1d, weights=weights)[0]
        down_1d = np.histogram(down, bins=bins_1d, weights=weights)[0]
        weights_1d.append((up_1d, down_1d))
    return weights_1d

def indices_to_weights_and_resp(
        mdim_up_down, mdim_other, weights,
        mdim_reco, mdim_true, is_reco):
    if is_reco:
        bins_1d = mdim_reco.multi_bin_edges
        which_ind = 0
    else:
        bins_1d = mdim_true.multi_bin_edges
        which_ind=1
    bins_2d = [mdim_reco.multi_bin_edges, mdim_true.multi_bin_edges]
    data_2d = [None, None]
    data_2d[1-which_ind] = mdim_other
    weights_1d = []
    weights_2d = []
    for up, down in mdim_up_down:
        up_1d = np.histogram(up, bins=bins_1d, weights=weights)[0]
        down_1d = np.histogram(down, bins=bins_1d, weights=weights)[0]
        weights_1d.append((up_1d, down_1d))
        data_2d[which_ind] = up
        up_2d = np.histogram2d(
            data_2d[0], data_2d[1], bins=bins_2d, weights=weights)[0]
        data_2d[which_ind] = down
        down_2d = np.histogram2d(
            data_2d[0], data_2d[1], bins=bins_2d, weights=weights)[0]
        weights_2d.append((up_2d.flatten(), down_2d.flatten()))
    return weights_1d, weights_2d

def get_indexing(reco_mbins, true_mbins):
    reco_bins = np.arange(reco_mbins.n_bins)
    true_bins = np.arange(true_mbins.n_bins)
    resp_x = (reco_bins[:, np.newaxis]
              * np.ones_like(true_bins)[np.newaxis, :]).flatten()
    resp_y = (np.ones_like(reco_bins)[:, np.newaxis]
              * true_bins[np.newaxis, :]).flatten()
    return reco_bins, true_bins, (resp_x, resp_y)

def calc_inte_sys_weighting(
        frac_err, multi_hist_inds, multi_binner,
        init_es, end_es, in_tpc):
    if isinstance(frac_err, tuple):
        up_err = get_frac_errs(
            1+frac_err[0], multi_hist_inds, multi_binner, axis=1)
        down_err = get_frac_errs(
            1-frac_err[1], multi_hist_inds, multi_binner, axis=1)
    else:
        up_err = get_frac_errs(
            1+frac_err, multi_hist_inds, multi_binner, axis=1)
        down_err = get_frac_errs(
            1-frac_err, multi_hist_inds, multi_binner, axis=1)
    e_diff = init_es - end_es
    up_down = []
    resps = []
    for i in range(multi_binner.n_bins - int(multi_binner.has_beam)):
        sel_mask = multi_hist_inds == i
        if np.sum(sel_mask) == 0:
            continue
        # Assumes Bethe Bloch is flat
        new_ends_up = ak.where(
            sel_mask, init_es - (e_diff*up_err[i]), end_es)
        new_ends_down = ak.where(
            sel_mask, init_es - (e_diff*down_err[i]), end_es)
        up = multi_binner.energies_to_multi_dim_inds(
            init_es, new_ends_up, in_tpc=in_tpc)
        down = multi_binner.energies_to_multi_dim_inds(
            init_es, new_ends_down, in_tpc=in_tpc)
        up_down.append((up, down))
    return up_down

def calc_init_sys_weighting(
        frac_err, multi_hist_inds, multi_binner,
        init_es, end_es, in_tpc):
    if isinstance(frac_err, tuple):
        up_err = get_frac_errs(
            1+frac_err[0], multi_hist_inds, multi_binner, axis=1)
        down_err = get_frac_errs(
            1-frac_err[1], multi_hist_inds, multi_binner, axis=1)
    else:
        up_err = get_frac_errs(
            1+frac_err, multi_hist_inds, multi_binner, axis=1)
        down_err = get_frac_errs(
            1-frac_err, multi_hist_inds, multi_binner, axis=1)
    e_diff = init_es - end_es
    up_down = []
    for i in range(multi_binner.n_bins - int(multi_binner.has_beam)):
        sel_mask = multi_hist_inds == i
        if np.sum(sel_mask) == 0:
            continue
        new_init_up = ak.where(sel_mask, init_es*up_err[i], init_es)
        new_init_down = ak.where(sel_mask, init_es*down_err[i], init_es)
        # Assumes Bethe Bloch is flat
        up = multi_binner.energies_to_multi_dim_inds(
            new_init_up, new_init_up-e_diff, in_tpc=in_tpc)
        down = multi_binner.energies_to_multi_dim_inds(
            new_init_down, new_init_down-e_diff, in_tpc=in_tpc)
        up_down.append((up, down))
    return up_down

def calc_inte_sys_params(
        frac_err, multi_hist_inds, multi_binner,
        init_es, end_es, in_tpc):
    if isinstance(frac_err, tuple):
        up_err = get_frac_errs(
            1+frac_err[0], multi_hist_inds, multi_binner, axis=1)
        down_err = get_frac_errs(
            1-frac_err[1], multi_hist_inds, multi_binner, axis=1)
    else:
        up_err = get_frac_errs(
            1+frac_err, multi_hist_inds, multi_binner, axis=1)
        down_err = get_frac_errs(
            1-frac_err, multi_hist_inds, multi_binner, axis=1)
    up_down = []
    for i in range(multi_binner.n_bins - int(multi_binner.has_beam)):
        sel_mask = multi_hist_inds == i
        sel_ends = end_es[sel_mask]
        sel_inits = init_es[sel_mask]
        sel_tpc = in_tpc[sel_mask]
        e_diff = sel_inits - sel_ends
        # Assumes Bethe Bloch is flat
        up = multi_binner.energies_to_multi_dim_inds(
            sel_inits, sel_inits - (e_diff*up_err[i]), in_tpc=sel_tpc)
        down = multi_binner.energies_to_multi_dim_inds(
            sel_inits, sel_inits - (e_diff*down_err[i]), in_tpc=sel_tpc)
        up_down.append((up, down))
    return up_down

def calc_init_sys_params(
        frac_err, multi_hist_inds, multi_binner,
        init_es, end_es, in_tpc,
        norm_by_count=False, weights=None):
    if isinstance(frac_err, tuple):
        up_err = get_frac_errs(
            1+frac_err[0], multi_hist_inds, multi_binner, axis=0)
        down_err = get_frac_errs(
            1-frac_err[1], multi_hist_inds, multi_binner, axis=0)
    else:
        up_err = get_frac_errs(
            1+frac_err, multi_hist_inds, multi_binner, axis=0)
        down_err = get_frac_errs(
            1-frac_err, multi_hist_inds, multi_binner, axis=0)
    if norm_by_count:
        bin_counts = np.histogram(
            multi_hist_inds, bins=multi_binner.multi_bin_edges,
            weights=weights)[0]
        if multi_binner.has_beam:
            bin_counts = bin_counts[:-1]
        up_err = up_err / np.sqrt(bin_counts)
        down_err = down_err / np.sqrt(bin_counts)
    up_down = []
    for i in range(multi_binner.n_bins - int(multi_binner.has_beam)):
        sel_mask = multi_hist_inds == i
        sel_ends = end_es[sel_mask]
        sel_inits = init_es[sel_mask]
        sel_tpc = in_tpc[sel_mask]
        e_diff = sel_inits - sel_ends
        # Assumes Bethe Bloch is flat
        init_up = sel_inits*(up_err[i])
        up = multi_binner.energies_to_multi_dim_inds(
            init_up, init_up - e_diff, in_tpc=sel_tpc)
        init_down = sel_inits*(down_err[i])
        down = multi_binner.energies_to_multi_dim_inds(
            init_down, init_down - e_diff, in_tpc=sel_tpc)
        up_down.append((up, down))
    return up_down

def split_info_by_other_ind(
        info,
        splitting_inds,
        splitting_multi_binner):
    split_info = []
    for i in range(splitting_multi_binner.n_bins):
        sel_mask = splitting_inds == i
        sel_info = info[sel_mask]
        split_info.append(sel_info)
    return split_info

def plot_multi_dim_hist(hist, binner, plt_conf, pdf=None, title=None):
    plt_conf.setup_figure(title=None, size="half")#title)
    plt.step(binner.multi_bin_edges[:-1], hist,
             **plt_conf.gen_kwargs(where="mid"))
    plt_conf.format_axis(xlabel="Multi. dim bin index", ylabel="Count")
    if pdf is not None:
        pdf.Save()
    return plt_conf.end_plot()

def plot_multi_dim_comparision(
        hist_1, binner_1, label_1,
        hist_2, binner_2, label_2,
        plt_conf, pdf=None, title=None, norm=True):
    plt_conf.setup_figure(title=None, size="half")#title)
    if norm:
        slice_1 = slice(0, -1) if binner_1.has_beam else slice(None)
        slice_2 = slice(0, -1) if binner_2.has_beam else slice(None)
        hist_1_norm = np.sum(hist_1[slice_1])
        hist_2_norm = np.sum(hist_2[slice_2])
        ylab = "Density"#"Valid density"
    else:
        hist_1_norm = 1
        hist_2_norm = 1
        ylab = "Count"
    plt.step(binner_1.multi_bin_edges[:-1], hist_1/hist_1_norm,
             **plt_conf.gen_kwargs(label=label_1, where="mid"))
    plt.step(binner_2.multi_bin_edges[:-1], hist_2/hist_2_norm,
             **plt_conf.gen_kwargs(label=label_2, where="mid", color="C2"))
    plt_conf.format_axis(xlabel=r"$B^{ijk}$ flattened index", ylabel=ylab)
    if pdf is not None:
        pdf.Save()
    return plt_conf.end_plot()

def plot_unfolding_matrix(
        uf_mat, plt_conf, pdf=None, title="Unfolding matrix"):
    x_uf_mat = np.arange(uf_mat.shape[0])[:, np.newaxis]
    y_uf_mat = np.arange(uf_mat.shape[1])[np.newaxis, :]
    x_uf_flat = (x_uf_mat * np.ones_like(y_uf_mat)).flatten()
    y_uf_flat = (y_uf_mat * np.ones_like(x_uf_mat)).flatten()
    bin_edges_uf = [np.arange(uf_mat.shape[0]+1), np.arange(uf_mat.shape[1]+1)]
    uf_weights_flat = uf_mat.flatten()
    uf_weights_flat[uf_weights_flat == 0] = np.nan

    plt_conf.setup_figure(title=None, size="half")#title)
    plt.hist2d(
        x_uf_flat, y_uf_flat, bins=bin_edges_uf, weights=uf_weights_flat,
        norm="linear", vmax=1, vmin=0)
    plt.colorbar()
    plt_conf.format_axis(xlabel="Truth bin index", ylabel="Reco bin index", ylog=False)
    if pdf is not None:
        pdf.Save()
    return plt_conf.end_plot()

def get_finite_clipped_edges(overflow_edges):
    finite_min = np.min(np.nan_to_num(
        overflow_edges, neginf = np.max(overflow_edges)))
    finite_max = np.max(np.nan_to_num(
        overflow_edges, posinf = np.min(overflow_edges)))
    ave_width = ((finite_max - finite_min)
                 /(np.sum(np.isfinite(overflow_edges)) - 1))
    return np.clip(overflow_edges, finite_min-ave_width, finite_max+ave_width)

def plot_fit_results(
        values, errs, bin_edges, plt_conf, title=None,
        ylab="Fit result", proc_labels=None, log=True, pdf=None):
    n_procs = values.shape[-1]
    if proc_labels is not None:
        assert len(proc_labels) == n_procs
    else:
        proc_labels = list(range(n_procs))
    clip_edges = get_finite_clipped_edges(bin_edges)
    e_centres = 0.5 * (clip_edges[1:] + clip_edges[:-1])
    proc_offsets = (np.arange(n_procs) - (n_procs/2)) * 5
    err_offsets = np.array([proc_offsets, -proc_offsets])[:, np.newaxis, :]
    x_errs = (0.5*np.abs(clip_edges[1:]-clip_edges[:-1]))[np.newaxis, :,  np.newaxis] + err_offsets
    plt_conf.setup_figure(title=None, size="half")#title)
    for i, lab in enumerate(proc_labels):
        plt.errorbar(
            e_centres + proc_offsets[i],
            values[..., i],
            xerr = x_errs[..., i],
            yerr=errs[..., i],
            **plt_conf.gen_kwargs(index=i, label=lab, ls=""))
    ylim = 1 if log else None
    plt_conf.format_axis(xlabel="Interaction energy / MeV", ylabel=ylab, ylog=log, ylim=ylim)
    if pdf is not None:
        pdf.Save()
    return plt_conf.end_plot()

def plot_xs(xs, xs_err, bins, plt_conf, proc="total", pdf=None, stat_err=None):
    ave_width = (np.max(bins) - np.min(bins))/(bins.size - 1)
    min_e_x = np.min(bins) - (0.5*ave_width)
    max_e_x = np.max(bins) + (0.5*ave_width)
    g4_xs = cross_section.GeantCrossSections(
        file="/users/wx21978/projects/pion-phys/pi0-analysis/analysis/data/g4_xs.root",
        energy_range = [min_e_x, max_e_x])
    interp_dict = {
        "total": g4_xs.GetInterpolatedCurve("total_inelastic"),
        "abs.": g4_xs.GetInterpolatedCurve("absorption"),
        "cex.": g4_xs.GetInterpolatedCurve("charge_exchange"),
        "pion": lambda e: (g4_xs.GetInterpolatedCurve("total_inelastic")(e)
                           - g4_xs.GetInterpolatedCurve("absorption")(e)
                           - g4_xs.GetInterpolatedCurve("charge_exchange")(e))}
    interp = interp_dict[proc.lower()]
    # Assume xs is approximately linear (seems valid!)
    # Expected result is the the xs of the central value
    bin_centres = 0.5*(bins[:-1]+bins[1:])
    expected = interp(bin_centres)
    chi2 = np.sum(((xs - expected)**2)/(xs_err**2))
    dof = len(xs) - 2 # assume cross section is linear
    # plt_conf.setup_figure(title=r"$\chi^2$: "+f"{chi2}\n "+r"$\chi^2$"+f"/{dof}: {chi2/dof}")
    plt_conf.setup_figure(title=None, size="half")#r"$\chi^2$: "+f"{chi2}")
    xs_x = np.linspace(min_e_x, max_e_x, 50)
    plt.plot(xs_x, interp(xs_x), **plt_conf.gen_kwargs(label=f"Geant4 {proc}"))
    lab = "Stat. + sys." if stat_err is not None else "Calculated XS"
    plt.errorbar(
        bin_centres,
        xs,
        xerr = 0.5*np.abs(bins[1:]-bins[:-1]),
        yerr=xs_err,
        **plt_conf.gen_kwargs(index=0, label=lab, ls="", capsize=4, capthick=2))
    if stat_err is not None:
        plt.errorbar(
            bin_centres,
            xs,
            xerr = 0.5*np.abs(bins[1:]-bins[:-1]),
            yerr=stat_err,
            **plt_conf.gen_kwargs(index=1, label="Stat. only",
                                  ls="", capsize=2))
    plt_conf.format_axis(xlabel="Energy / MeV", ylabel="Cross section / mb", ylog=False)
    if pdf is not None:
        pdf.Save()
    return plt_conf.end_plot()

def GNNFit(
        data_input : AnalysisInputs.AnalysisInputGNN,
        energy_slice : Slicing.Slices,
        template_input : AnalysisInputs.AnalysisInputGNN,
        template_weights : np.array = None,
        return_fits : bool = False,
        impure_template : bool = False) \
        -> tuple:
    """ Fit model to analysis input to predict the normalaisations of each process.

    Args:
        fit_input (cross_section.AnalysisInput): observed data
        energy_slice (Slicing.Slices): energy slices
        template_input (cross_section.AnalysisInput | cross_section.pyhf.Model): template sample or existing model
        suggest_init (bool, optional): estimate normalisations ans use these as the initial values for the fit. Defaults to False.
        template_weights (np.array, optional): weights for the mean track score. Defaults to None.
        return_fit_results (bool, optional): return the raw fit results as  well as the prediction. Defaults to False.

    Returns:
        cross_section.cabinetry.model_utils.ModelPrediction | cross_section.FitResults: model prediction and or the raw fit result.
    """
    generators = []
    init_preds = []
    proc_labs = list(template_input.classification_info.keys())
    if impure_template:
        pure_lab=["Impure"]
        temp_procs = template_input.exclusive_process
        impure_mask = np.logical_not(template_input.inclusive_process)
        temp_procs[impure_mask] = len(proc_labs)
    else:
        pure_lab = []
        temp_procs = template_input.exclusive_process
    # n_data = []
    # n_temps = []
    data_int_inds = data_input.FetchInteractionInds(energy_slice)
    temp_int_inds = template_input.FetchInteractionInds(energy_slice)
    for i in range(-1, energy_slice.n_slices+1):
        temp_mask = temp_int_inds == i
        # n_temps.append(np.sum(temp_mask))
        data_mask = data_int_inds == i
        # n_data.append(np.sum(data_mask))
        generators.append(Fitter.DistGenCorr(
            template_input.gnn_scores[temp_mask],
            temp_procs[temp_mask],
            data_input.gnn_scores[data_mask],
            bins=12, fix_bin_range=(0,1),
            labels=proc_labs + pure_lab))
        init_preds.append(np.histogram( # In case of not predicting something
            data_input.gnn_preds[data_mask],
            bins=np.arange(data_input.gnn_scores.shape[1] + 1))[0])
    
    fit_outs = [Fitter.generator_fit(g, init_preds=p, printout=False)
                for g, p in zip(generators, init_preds)]
    yields = np.array([list(f.values) for f in fit_outs])
    uncs = np.array([list(f.errors) for f in fit_outs])
    if return_fits:
        return yields, uncs, fit_outs
    else:
        return yields, uncs

def gen_extra_dim_weighter(
        extra_dim_bin_edges, expect_weights, weight_std,
        n_scores=3, has_truth=True):
    extra_dim_shape = tuple(n-1 for n in extra_dim_bin_edges.shape)
    extra_shape = extra_dim_shape + (n_scores+int(has_truth))*(1,)
    mean = np.reshape(np.ones(extra_dim_shape) * expect_weights, extra_shape)
    std = np.reshape(np.ones(extra_dim_shape) * weight_std, extra_shape)
    # Guassian, but negative values disallowed
    trunc_norm = truncnorm(loc=mean, scale=std, a=-mean/std, b=np.inf)
    def weighting_sampler(counts):
        """
        counts with shape (n_extra_bins,) + 3*(12,)
        for case of 12 GNN score bins.
        """
        return counts * trunc_norm.rvs()
    return weighting_sampler

def GNNWeightingSys(
        data_input : AnalysisInputs.AnalysisInputGNN,
        energy_slice : Slicing.Slices,
        template_input : AnalysisInputs.AnalysisInputGNN,
        extra_info, extra_binning,
        extra_deviation : float,
        n_pulls : int = 1000,
        template_weights : np.array = None,
        impure_template : bool = False) -> tuple:
    """ Fit model to analysis input to predict the normalaisations of each process.

    Args:
        fit_input (cross_section.AnalysisInput): observed data
        energy_slice (Slicing.Slices): energy slices
        template_input (cross_section.AnalysisInput | cross_section.pyhf.Model): template sample or existing model
        suggest_init (bool, optional): estimate normalisations ans use these as the initial values for the fit. Defaults to False.
        template_weights (np.array, optional): weights for the mean track score. Defaults to None.
        return_fit_results (bool, optional): return the raw fit results as  well as the prediction. Defaults to False.

    Returns:
        cross_section.cabinetry.model_utils.ModelPrediction | cross_section.FitResults: model prediction and or the raw fit result.
    """
    generators = []
    init_preds = []
    proc_labs = list(template_input.classification_info.keys())
    if impure_template:
        pure_lab=["Impure"]
        temp_procs = template_input.exclusive_process
        impure_mask = np.logical_not(template_input.inclusive_process)
        temp_procs[impure_mask] = len(proc_labs)
        n_scores = len(proc_labs) + 1
    else:
        pure_lab = []
        temp_procs = template_input.exclusive_process
        n_scores = len(proc_labs)
    data_int_inds = data_input.FetchInteractionInds(energy_slice)
    temp_int_inds = template_input.FetchInteractionInds(energy_slice)
    extra_weight_sampler = gen_extra_dim_weighter(
        extra_binning, 1, extra_deviation, n_scores=n_scores, has_truth=True)
    for i in range(-1, energy_slice.n_slices+1):
        temp_mask = temp_int_inds == i
        data_mask = data_int_inds == i
        this_gen = Fitter.DistGenCorr(
            template_input.gnn_scores[temp_mask],
            temp_procs[temp_mask],
            data_input.gnn_scores[data_mask],
            template_extra=extra_info[temp_mask],
            extra_bins=extra_binning,
            bins=12, fix_bin_range=(0,1),
            labels=proc_labs+pure_lab)
        this_gen.set_template_sample_params(weighting_func = extra_weight_sampler)
        generators.append(this_gen)
        init_preds.append(np.histogram( # In case of not predicting something
            data_input.gnn_preds[data_mask],
            bins=np.arange(data_input.gnn_scores.shape[1] + 1))[0])
    full_gen = Fitter.DistGenCorr(
            template_input.gnn_scores,
            temp_procs,
            data_input.gnn_scores,
            template_extra=extra_info,
            extra_bins=extra_binning,
            bins=12, fix_bin_range=(0,1),
            labels=proc_labs+pure_lab)
    full_gen.set_template_sample_params(weighting_func = extra_weight_sampler)
    print("All events GNN model std:")
    print(np.std(Fitter.generate_pulls(n_pulls, full_gen)[-2, :], axis=-1))
    fit_centres = np.array([Fitter.generate_pulls(n_pulls, g, init_preds=p)[-2, :]
                for g, p in zip(generators, init_preds)])
    return np.std(fit_centres, axis=-1)

def apply_proc_fracs(
        tot_xs, tot_xs_err,
        proc_fracs, proc_frac_errs,
        has_overflow=True):
    slicer = slice(1, -1) if has_overflow else slice(None)
    proc_xs = tot_xs.to_numpy()[:, np.newaxis] * proc_fracs[slicer]
    proc_xs_errs = proc_xs * np.sqrt(
        (tot_xs_err/tot_xs).to_numpy()[:, np.newaxis]**2
        + (proc_frac_errs/proc_fracs)[slicer]**2)
    return proc_xs, proc_xs_errs

def append_systematics(
    unfolding_dict, args, reco_mbins, true_mbins,
    data_ai, truth_ai):
    reco_inds, true_inds, resp_inds = get_indexing(reco_mbins, true_mbins)
    unfolding_dict.update({
        "reco_weight_inds": reco_inds,
        "true_weight_inds": true_inds,
        "resp_weight_inds": resp_inds})

    init_e_systs = ["beam_momentum", "upstream_energy"]
    inte_e_systs = ["track_length"]
    data_init = data_ai.KE_init_reco
    data_end = data_ai.KE_int_reco
    data_in_tpc = np.logical_not(data_ai.outside_tpc_reco)
    truth_init_reco = truth_ai.KE_init_reco
    truth_end_reco = truth_ai.KE_int_reco
    truth_in_tpc_reco = np.logical_not(truth_ai.outside_tpc_reco)
    truth_init_true = truth_ai.KE_init_true
    truth_end_true = truth_ai.KE_int_true
    truth_in_tpc_true = np.logical_not(truth_ai.outside_tpc_true)
    
    if "track_length" in args.systematics.keys():
        data_track_len_inds = calc_inte_sys_weighting(
            args.systematics["track_length"],
            unfolding_dict["data_multi_dim"], reco_mbins,
            data_init, data_end, data_in_tpc)
        data_track_len_sys = indices_to_weights_only(
            data_track_len_inds,
            unfolding_dict["reco_weights"], reco_mbins)
        del data_track_len_inds
        meas_track_len_inds = calc_inte_sys_weighting(
            args.systematics["track_length"],
            unfolding_dict["truth_multi_dim_reco"], reco_mbins,
            truth_init_reco, truth_end_reco, truth_in_tpc_reco)
        meas_track_len_sys, resp_track_len_sys = indices_to_weights_and_resp(
            meas_track_len_inds, unfolding_dict["truth_multi_dim_reco"],
            unfolding_dict["truth_weights"], reco_mbins, true_mbins, True)
        del meas_track_len_inds
        unfolding_dict.update({
            "sys_data_track_length": data_track_len_sys,
            "sys_meas_track_length": meas_track_len_sys})
            # "sys_resp_track_length": resp_track_len_sys})
        # data_track_len_sys = calc_inte_sys_params(
        #     args.systematics["track_length"],
        #     unfolding_dict["data_multi_dim"], reco_mbins,
        #     data_init, data_end, data_in_tpc)
        # truth_track_len_sys = calc_inte_sys_params(
        #     args.systematics["track_length"],
        #     unfolding_dict["truth_multi_dim_reco"], reco_mbins,
        #     truth_init_reco, truth_end_reco, truth_in_tpc_reco)
        # unfolding_dict.update({
        #     "sys_data_track_length": data_track_len_sys,
        #     "sys_meas_track_length": truth_track_len_sys})

    if "beam_momentum" in args.systematics.keys():
        data_beam_mom_inds = calc_init_sys_weighting(
            args.systematics["beam_momentum"],
            unfolding_dict["data_multi_dim"], reco_mbins,
            data_init, data_end, data_in_tpc)
        data_beam_mom_sys = indices_to_weights_only(
            data_beam_mom_inds,
            unfolding_dict["reco_weights"], reco_mbins)
        del data_beam_mom_inds
        unfolding_dict.update({
            "sys_data_beam_mom": data_beam_mom_sys})
        # data_beam_mom_sys = calc_init_sys_params(
        #     args.systematics["beam_momentum"],
        #     unfolding_dict["data_multi_dim"], reco_mbins,
        #     data_init, data_end, data_in_tpc)
        # # # MC doesn't include error in the upstream momentum
        # # truth_beam_mom_sys = calc_init_sys_params(
        # #     args.systematics["beam_momentum"],
        # #     unfolding_dict["truth_multi_dim_reco"], reco_mbins,
        # #     truth_init_reco, truth_end_reco, truth_in_tpc_reco)
        # unfolding_dict.update({
        #     # "sys_meas_beam_mom": truth_beam_mom_sys,
        #     "sys_data_beam_mom": data_beam_mom_sys})

    if "upstream_energy" in args.systematics.keys():
        # First determine the errors to use
        finite_edges = get_finite_clipped_edges(
            args.energy_slices.bin_edges_with_overflow)
        median_energies = 0.5*(finite_edges[:-1] + finite_edges[1:])
        # extra_kwargs = {}
        if args.systematics["upstream_energy"] == "function":
            raise NotImplementedError("Not yet dealt with different up and down errs")
            high_corr_params = {
                f"p{i}" : (args.upstream_loss_correction_params["value"][f"p{i}"]
                           + args.upstream_loss_correction_params["error"][f"p{i}"])
                for i in range(args.upstream_loss_response.n_params)}
            low_corr_params = {
                f"p{i}" : (args.upstream_loss_correction_params["value"][f"p{i}"]
                           - args.upstream_loss_correction_params["error"][f"p{i}"])
                for i in range(args.upstream_loss_response.n_params)}
            high_frac_errs = args_c["upstream_loss_response"].func(
                median_energies, **high_corr_params)
            low_frac_errs = args_c["upstream_loss_response"].func(
                median_energies, **low_corr_params)
        else:
            bin_stds = np.array(
                args.upstream_loss_correction_params["ana_bin_stds"])
            if args.systematics["upstream_energy"] == "binned":
                upstream_frac_errs = bin_stds/median_energies
            elif args.systematics["upstream_energy"] == "binned_init_reduced":
                upstream_frac_errs = (
                    (bin_stds/median_energies)
                    / np.sqrt(np.array(
                        args.upstream_loss_correction_params["ana_bin_counts"])))
            elif args.systematics["upstream_energy"] == "binned_multi_dim_reduced":
                raise NotImplementedError("Not yet dealt with weights")
                # # Need to do weights differently for truth and data (since they have different weights!)
                # upstream_frac_errs = bin_stds/median_energies
                # extra_kwargs.update({"norm_by_count":True,
                #                      "weights":template.weights})
            else:
                raise ValueError("Unknown upstream method "
                                 +args.systematics["upstream_energy"])
        # Apply the errors
        data_upstream_inds = calc_inte_sys_weighting(
            upstream_frac_errs,
            unfolding_dict["data_multi_dim"], reco_mbins,
            data_init, data_end, data_in_tpc)
        data_upstream_sys = indices_to_weights_only(
            data_upstream_inds,
            unfolding_dict["reco_weights"], reco_mbins)
        del data_upstream_inds
        meas_upstream_inds = calc_inte_sys_weighting(
            upstream_frac_errs,
            unfolding_dict["truth_multi_dim_reco"], reco_mbins,
            truth_init_reco, truth_end_reco, truth_in_tpc_reco)
        meas_upstream_sys, resp_upstream_sys = indices_to_weights_and_resp(
            meas_upstream_inds, unfolding_dict["truth_multi_dim_reco"],
            unfolding_dict["truth_weights"], reco_mbins, true_mbins, True)
        del meas_upstream_inds
        unfolding_dict.update({
            "sys_data_upstream_corr": data_upstream_sys,
            "sys_meas_upstream_corr": meas_upstream_sys})
            # "sys_resp_upstream_corr": resp_upstream_sys})
        # data_upstream_sys = calc_init_sys_params(
        #     upstream_frac_errs,
        #     unfolding_dict["data_multi_dim"], reco_mbins,
        #     data_init, data_end, data_in_tpc)
        # truth_upstream_sys = calc_init_sys_params(
        #     upstream_frac_errs,
        #     unfolding_dict["truth_multi_dim_reco"], reco_mbins,
        #     truth_init_reco, truth_end_reco, truth_in_tpc_reco)
        # unfolding_dict.update({
        #     "sys_data_upstream_corr": data_upstream_sys,
        #     "sys_meas_upstream_corr": truth_upstream_sys})

    if "purity" in args.systematics.keys():
        assert true_mbins.has_beam, "Must have beam data for purity"
        purity_up = 1 + args.systematics["purity"]
        purity_down = 1 - args.systematics["purity"]
        impure_mask = (
            unfolding_dict["truth_multi_dim_true"] == (true_mbins.n_bins-1))
        purity_weights_up = ak.where(
            impure_mask,
            unfolding_dict["truth_weights"] * purity_up,
            unfolding_dict["truth_weights"])
        purity_weights_down = ak.where(
            impure_mask,
            unfolding_dict["truth_weights"] * purity_up,
            unfolding_dict["truth_weights"])
        purity_bins = [reco_mbins.multi_bin_edges, true_mbins.multi_bin_edges]
        purity_resp_up = np.histogram2d(
            unfolding_dict["truth_multi_dim_reco"],
            unfolding_dict["truth_multi_dim_true"],
            bins=purity_bins, weights=purity_weights_up)[0].flatten()
        purity_resp_down = np.histogram2d(
            unfolding_dict["truth_multi_dim_reco"],
            unfolding_dict["truth_multi_dim_true"],
            bins=purity_bins, weights=purity_weights_down)[0].flatten()
        unfolding_dict.update({"sys_resp_purity":
                               [(purity_resp_up, purity_resp_down)]})

    # meas_split_true = split_info_by_other_ind(
    #     unfolding_dict["truth_multi_dim_reco"],
    #     unfolding_dict["truth_multi_dim_true"],
    #     true_mbins)
    # true_split_meas = split_info_by_other_ind(
    #     unfolding_dict["truth_multi_dim_true"],
    #     unfolding_dict["truth_multi_dim_reco"],
    #     reco_mbins)
    # data_weights_split_reco = split_info_by_other_ind(
    #     unfolding_dict["reco_weights"],
    #     unfolding_dict["data_multi_dim"],
    #     reco_mbins)
    # weights_split_true = split_info_by_other_ind(
    #     unfolding_dict["truth_weights"],
    #     unfolding_dict["truth_multi_dim_true"],
    #     true_mbins)
    # weights_split_meas = split_info_by_other_ind(
    #     unfolding_dict["truth_weights"],
    #     unfolding_dict["truth_multi_dim_reco"],
    #     reco_mbins)
    # unfolding_dict.update({"meas_resp_by_true": meas_split_true,
    #                        "true_resp_by_meas": true_split_meas,
    #                        "data_weights_split_reco": data_weights_split_reco,
    #                        "truth_weights_split_true": weights_split_true,
    #                        "truth_weights_split_meas": weights_split_meas})
    return unfolding_dict
    

def Analyse(args : argparse.Namespace, plot : bool = False):
    if args.toy_template:
        raise NotImplementedError("Can't do GNN with Toys")
    assert args.pdsp, "Must have PDSP type data"
    template_ai = AnalysisInputs.FromFile(args.analysis_input["mc"])
    uf_temp_ai = AnalysisInputs.FromFile(args.analysis_input["mc_with_train"])
    data_ai = AnalysisInputs.FromFile(args.analysis_input["data"])
    # cheated_ai = AnalysisInputs.FromFile(args.analysis_input["mc_cheated"])

    print(f"Analysing PDSP")

    outdir = args.out + f"gnn_analysis/"
    if plot is True:
        os.makedirs(outdir, exist_ok = True)

    # scale = args.norm

    # unfolding_args = args.unfolding

    reco_mdim_inds, reco_mbins = data_ai.CreateHistogramInds(
        args.energy_slices, True)
    true_mdim_inds_truth, true_mbins = uf_temp_ai.CreateHistogramInds(
        args.energy_slices, False, uf_temp_ai.inclusive_process)
    true_mdim_inds_reco, _ = uf_temp_ai.CreateHistogramInds(
        args.energy_slices, True)
    reco_mdim_hist = np.histogram(
        reco_mdim_inds, bins=reco_mbins.multi_bin_edges)[0]
    if data_ai.weights is None:
        data_weights = np.ones_like(reco_mdim_inds)
    else:
        warnings.warn("Data probably shouldn't be weighted...")
        data_weights = data_ai.weights.to_numpy()
    if uf_temp_ai.weights is None:
        warnings.warn("Unweighted template, missing bema reweight?")
        true_weights = np.ones_like(true_mdim_inds_reco)
    else:
        true_weights = uf_temp_ai.weights.to_numpy()
    unfolding_inputs = {
        "reco_bin_edges": reco_mbins.multi_bin_edges,
        "true_bin_edges": true_mbins.multi_bin_edges,
        "reco_weights": data_weights,
        "truth_weights": true_weights,
        "data_multi_dim": reco_mdim_inds,
        "truth_multi_dim_reco": true_mdim_inds_reco,
        "truth_multi_dim_true": true_mdim_inds_truth}
    unfolding_inputs = append_systematics(
        unfolding_inputs, args, reco_mbins, true_mbins, data_ai, uf_temp_ai)
    os.makedirs(args.out + "unfolding_pickles/", exist_ok = True)
    pre_unfold_path = args.out + "unfolding_pickles/unfold_inputs.pkl"
    with open(pre_unfold_path, "wb") as f:
        pickle.dump(unfolding_inputs, f)
    post_unfold_path = args.out + "unfolding_pickles/unfold_outputs.pkl"

    do_uf = True
    if do_uf:
        print("Running RooUnfold script in ROOT environment")
        # TODO configurable
        os.system(f"bash /users/wx21978/projects/pion-phys/pi0-analysis/analysis/scripts/run_unfold_in_root.sh {pre_unfold_path} {post_unfold_path}")

        print("Returned to original python environment")
    with open(post_unfold_path, "rb") as f:
        unfolding_results = pickle.load(f)
    print("Loaded RooUnfold outputs")

    eff_corr_func = gen_efficiency_correction_func(args, has_purity=True)
    corr_hist, corr_errs = eff_corr_func(unfolding_results["unfolded_hist"],
                                         unfolding_results["unfolded_errs"])
    _, corr_errs_stat = eff_corr_func(unfolding_results["unfolded_hist"],
                                         unfolding_results["unfolded_stat_errs"])
    tot_xs, tot_xs_err = Slicing.EnergySliceFiducial.TotalCrossSection(
        corr_hist, true_mbins, corr_errs)
    _, tot_xs_stat_err = Slicing.EnergySliceFiducial.TotalCrossSection(
        corr_hist, true_mbins, corr_errs_stat)

    int_uf_mat, int_err_mat = project_unfolding_matrix(
        unfolding_results["unfolding_matrix"], reco_mdim_hist,
        reco_mbins, true_mbins)
    int_fits, int_errs = GNNFit(
        data_ai,
        args.energy_slices,
        template_ai,
        template_weights = template_ai.weights,
        return_fits = False)
    #     impure_template=True)
    # print(f"Impure counts: {int_fits[-1]}, error: {int_errs[-1]}")
    # int_fits = int_fits[:-1]
    # int_errs = int_errs[:-1]
    if "GNN_model" in args.systematics.keys():
    # if True:
        extra_model_param = template_ai.gnn_model_sys_param
        assert extra_model_param is not None, "Missing extra model parameter"
        if np.min(extra_model_param) == 0 and np.max(extra_model_param) < 30:
            extra_bins = np.arange(np.max(extra_model_param)+2) - 0.5
        else:
            extra_bins = 30
        gnn_model_sys = GNNWeightingSys(
            data_ai,
            args.energy_slices,
            template_ai,
            extra_model_param, extra_bins,
            known_gnn_theory_sizes[args.systematics["GNN_model"]],
            n_pulls = 1000,
            template_weights = template_ai.weights)
    else:
        gnn_model_sys = np.zeros_like(int_errs)
    print(int_fits)
    print(int_errs)
    print(gnn_model_sys),
    print(gnn_model_sys/int_fits)
    comb_int_err = np.sqrt((int_errs**2) + (gnn_model_sys**2))
    # with GNN model sys
    uf_int_yield, uf_int_yield_err = unfold_interactions(
        int_fits, comb_int_err, int_uf_mat, int_err_mat)
    uf_int_frac, uf_int_err = yields_to_fracs(
        uf_int_yield, uf_int_yield_err)
    # Stat only
    _, uf_int_yield_stat_err = unfold_interactions(
        int_fits, int_errs, int_uf_mat, int_err_mat)
    _, uf_int_stat_err = yields_to_fracs(
        uf_int_yield, uf_int_yield_stat_err)
    # Stat. + sys.
    proc_xs, proc_xs_errs = apply_proc_fracs(
        tot_xs, tot_xs_err, uf_int_frac, uf_int_err)
    proc_xs_list = [proc_xs[:, i] for i in range(len(data_ai.classification_info))]
    proc_xs_err_list = [proc_xs_errs[:, i] for i in range(len(data_ai.classification_info))]
    # Stat. only
    _, proc_xs_stat_errs = apply_proc_fracs(
        tot_xs, tot_xs_stat_err, uf_int_frac, uf_int_stat_err)
    proc_xs_stat_err_list = [proc_xs_stat_errs[:, i] for i in range(len(data_ai.classification_info))]

    xs = {
        "total": tot_xs,
        "total_err": tot_xs_err,
        "total_stat_err": tot_xs_stat_err}
    for i, proc in enumerate(list(data_ai.classification_info.keys())):
        xs.update({proc : proc_xs_list[i],
                   proc+"_err" : proc_xs_err_list[i],
                   proc+"_stat_err" : proc_xs_stat_err_list[i]})

    # Plotting
    plt_conf = Plots.PlotConfig()
    plt_conf.SHOW_PLOT = True
    plt_conf.SAVE_FOLDER = None
    with Plots.PlotBook(outdir + "unfolding_plots.pdf", plot) as book:
        plot_multi_dim_hist(
            reco_mdim_hist, reco_mbins, plt_conf, pdf=book,
            title="Data counts pre-unfolding")

        true_mdim_hist_truth = np.histogram(
            true_mdim_inds_truth, bins=true_mbins.multi_bin_edges)[0]
        true_mdim_hist_reco = np.histogram(
            true_mdim_inds_reco, bins=reco_mbins.multi_bin_edges)[0]
        plot_multi_dim_hist(
            reco_mdim_hist, reco_mbins, plt_conf, pdf=book,
            title="Data counts pre-unfolding")
        plot_multi_dim_comparision(
            true_mdim_hist_reco, reco_mbins, "Sim. reco.",
            true_mdim_hist_truth, true_mbins, "Sim. truth",
            plt_conf, pdf=book, title="Sim. reco vs. truth, counts", norm=False)
        plot_multi_dim_comparision(
            true_mdim_hist_reco, reco_mbins, "MC reco",
            true_mdim_hist_truth, true_mbins, "MC truth",
            plt_conf, pdf=book, title="MC reco vs. truth", norm=True)
        plot_multi_dim_comparision(
            reco_mdim_hist, reco_mbins, "Data",
            true_mdim_hist_reco, reco_mbins, "MC reco",
            plt_conf, pdf=book, title="Data vs. MC reco", norm=True)
        
        plot_multi_dim_hist(
            unfolding_results["unfolded_hist"], true_mbins, plt_conf,
            pdf=book, title="Unfolded data")
        plot_multi_dim_comparision(
            unfolding_results["unfolded_hist"], true_mbins, "Unfolded data",
            true_mdim_hist_truth, true_mbins, "MC truth",
            plt_conf, pdf=book, title="Unfolded data vs. MC truth", norm=True)
        plot_multi_dim_comparision(
            reco_mdim_hist, reco_mbins, "Pre-unfold data",
            unfolding_results["unfolded_hist"], true_mbins, "Unfolded data",
            plt_conf, pdf=book, title="Raw vs. unfolded data, counts", norm=False)
        plot_multi_dim_comparision(
            reco_mdim_hist, reco_mbins, "Pre-unfold data",
            unfolding_results["unfolded_hist"], true_mbins, "Unfolded data",
            plt_conf, pdf=book, title="Raw vs. unfolded data", norm=True)

        plot_unfolding_matrix(
            unfolding_results["unfolding_matrix"], plt_conf, pdf=book)
        plot_unfolding_matrix(int_uf_mat, plt_conf, pdf=book,
                              title="Interaction unfolding matrix")
        
        corr_mc, _ = eff_corr_func(true_mdim_hist_truth,
                                   np.sqrt(true_mdim_hist_truth))
        plot_multi_dim_hist(
            corr_hist, true_mbins, plt_conf,
            pdf=book, title="Efficiency corrected data unfolded")
        plot_multi_dim_comparision(
            corr_hist, true_mbins, "Eff. corrected data",
            unfolding_results["unfolded_hist"], true_mbins, "Unfolded data",
            plt_conf, pdf=book, title="Eff. corr. vs. unfolded data", norm=True)
        plot_multi_dim_comparision(
            corr_hist, true_mbins, "Eff. corrected data",
            unfolding_results["unfolded_hist"], true_mbins, "Unfolded data",
            plt_conf, pdf=book, norm=False,
            title="Eff. corr. vs. unfolded data, counts")
        plot_multi_dim_comparision(
            corr_hist, true_mbins, "Eff. corrected data",
            corr_mc, true_mbins, "Eff. corr. sim. truth",
            plt_conf, pdf=book, title="Eff. corr. data vs. MC truth")
    
    proc_labs = list(template_ai.classification_info.keys())
    with Plots.PlotBook(outdir + "gnn_fit_plots.pdf", plot) as book:
        edges = args.energy_slices.bin_edges_with_overflow
        plot_fit_results(
            int_fits, int_errs, edges, plt_conf, title="Pre-unfold yields",
            ylab="Fit yield", proc_labels=proc_labs, pdf=book, log=True)
        plot_fit_results(
            uf_int_yield, uf_int_yield_err, edges, plt_conf,
            title="Unfolded yields", ylab="Fit yield",
            proc_labels=proc_labs, pdf=book, log=True)
        pre_uf_fracs, pre_uf_frac_errs = yields_to_fracs(
            int_fits, int_errs)
        plot_fit_results(
            pre_uf_fracs, pre_uf_frac_errs, edges, plt_conf,
            title="Pre-unfold fracs.", ylab=r"$\hat{f_p}$ from fit",
            proc_labels=proc_labs, pdf=book, log=False)
        plot_fit_results(
            uf_int_frac, uf_int_err, edges, plt_conf,
            title="Unfolded fracs.", ylab=r"$\hat{f_p}$ from fit",
            proc_labels=proc_labs, pdf=book, log=False)

    with Plots.PlotBook(outdir + "cross_section_plots.pdf", plot) as book:
        procs_to_plot = ["total"] + proc_labs
        for p in procs_to_plot:
            plot_xs(xs[p], xs[p+"_err"], args.energy_slices.bin_edges, plt_conf,
                    proc=p, pdf=book, stat_err=xs[p+"_stat_err"])

    # Plots to make:
    # 1 Multi dimensions histogram inputs
    # 1 Multi-dim histogram unfolding output
    # 1 Unfolding response matrix
    # 1 Comparison with input
    # 1 Effeciency corrected output
    # 1 Comparision with input/unfolding
    # 1 Interaction projection matrix
    # 1 Raw fit outputs
    # 1 Unfolded fit outputs
    # Total cross section compared with GEANT 4
    # Per process cross section compared with GEANT 4

    # Then add GNN Fit uncertainty calculation (see gnn_model_dependence notebook)


    # TODO think of seom plot here...
    # print(f"{fit_values.bestfit=}")
    # if plot:
    #     indices = [f"$\mu_{{{i}}}$" for i in ["abs", "cex", "spip", "pip"]]
    #     table = cross_section.pd.DataFrame({"fit value" : fit_values.bestfit[0:4] / scale, "uncertainty" : fit_values.uncertainty[0:4] / scale}, index = indices).T
    #     table.to_hdf(outdir + "fit_results_POI.hdf5", key = "df")
    #     FitParamTables(table).style.hide(axis = "index").to_latex(outdir + "fit_results_POI.tex")
    #     if len(fit_values.bestfit) > 4:
    #         indices = [f"$\\alpha_{{{i}}}$" for i in ["abs", "cex", "spip", "pip"]]
    #         table = cross_section.pd.DataFrame({"fit value" : fit_values.bestfit[4:], "uncertainty" : fit_values.uncertainty[4:]}, index = indices).T
    #         table.to_hdf(outdir + "fit_results_NP.hdf5", key = "df")
    #         FitParamTables(table).style.hide(axis = "index").to_latex(outdir + "fit_results_NP.tex")

    # if args.all is True:
    #     process = {i : None for i in templates[k].exclusive_process}
    # elif args.fit["regions"] is True:
    #     process = {"all" : None}
    # else:
    #     process = {args.signal_process : None}
    
    # for p in process:
    #     with Plots.PlotBook(outdir + f"plots_{p}.pdf", plot) as book, cross_section.PlotStyler(extend_colors = False, dark = True).Update(font_scale = 1.1):
    #         if p != "all":
    #             xs_true = Slicing.EnergySlice.CrossSection(
    #                 true_hists["int_"+p],
    #                 true_hists["int"][1:args.energy_slices.max_num+2],
    #                 true_hists["inc"][1:args.energy_slices.max_num+2],
    #                 EnergyTools.BetheBloch.meandEdX(
    #                     args.energy_slices.pos_bins[1:],
    #                     Particle.from_pdgid(211)),
    #                     args.energy_slices.width)

    #         if k == "toy":
    #             data_label = "toy data"
    #             mc_label = "toy template"
    #         elif k == "pdsp":
    #             data_label = "Data"
    #             mc_label = "MC (scaled to Data)"

    #         PlotDataBkgSub(histograms_reco_obs, histograms_reco_obs_err, templates[k], args.fit["regions"], p if p != "all" else "charge_exchange", args.energy_slices, scale, None, data_label, mc_label, book)
    #         if args.fit["regions"]:
    #             histograms_reco_obs = {**histograms_reco_obs, **histograms_reco_obs["int_ex"]}
    #             histograms_reco_obs.pop("int_ex")
    #             histograms_reco_obs_err = {**histograms_reco_obs_err, **histograms_reco_obs_err["int_ex"]}
    #             histograms_reco_obs_err.pop("int_ex")

    #         unfolding_stat = args.unfolding["mc_stat_unc"]

    #         unfolding_result = Unfolding(histograms_reco_obs, histograms_reco_obs_err, templates[k], dict(unfolding_args) if unfolding_args is not None else unfolding_args, p if p != "all" else "charge_exchange", scale, args.energy_slices, args.fit["regions"], unfolding_stat, mc_cheat, book)

    #         process[p] = XSUnfold(unfolding_result, args.energy_slices, unfolding_stat, True, regions = args.fit["regions"])
    #         if args.fit["regions"] is False:
    #             cross_section.PlotXSComparison({f"{label_map[k]} Data reco" : process[p], f"{label_map[k]} MC truth" : xs_true}, args.energy_slices, p, {f"{label_map[k]} Data reco" : "C0", f"{label_map[k]} MC truth" : "C1"}, simulation_label = "Geant4 v10.6")
    #             book.Save()
    #     Plots.plt.close("all")
    # if args.fit["regions"] is True:
    #     process = {i : j for i, j in process["all"].items()}
    #     with Plots.PlotBook(outdir + "results_all_regions.pdf", plot) as book:
    #         for i in process:
    #             if k == "pdsp":
    #                 true_hists = mc_cheat.CreateHistograms(args.energy_slices, i, False)
    #             else:
    #                 true_hists = templates[k].CreateHistograms(args.energy_slices, i, False, ~templates[k].inclusive_process)
    #             xs_true = Slicing.EnergySlice.CrossSection(
    #                 true_hists["int_ex"][1:-1],
    #                 true_hists["int"][1:-1],
    #                 true_hists["inc"][1:-1],
    #                 EnergyTools.BetheBloch.meandEdX(
    #                     args.energy_slices.pos_bins[1:-1],
    #                     Particle.from_pdgid(211)),
    #                     args.energy_slices.width)
    #             cross_section.PlotXSComparison({f"{label_map[k]} Data reco" : process[i], f"{label_map[k]} MC truth" : xs_true}, args.energy_slices, i, {f"{label_map[k]} Data reco" : "C0", f"{label_map[k]} MC truth" : "C1"}, simulation_label = "Geant4 v10.6")
    #             book.Save()
    #     Plots.plt.close("all")
    # print(f"{process=}")
    # xs[k] = process

    return xs

def main(args):
    cross_section.PlotStyler.SetPlotStyle(extend_colors = False, dark = True)
    args.out = args.out + "measurement/"

    xs = Analyse(args, True)

    cross_section.SaveObject(args.out + "xs.dill", xs)

    # with Plots.PlotBook(args.out + "results.pdf") as book:
    #     colours = {f"{label_map[k]} Data" : f"C{i}" for i, k in enumerate(xs.keys())}
    #     for p in list(xs.values())[0]:
    #         data = {f"{label_map[k]} Data" : xs[k][p] for k in xs.keys()}
    #         cross_section.PlotXSComparison(data, args.energy_slices, p, colours, simulation_label = "Geant4 v10.6")
    #         book.Save()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Computes the upstream energy loss for beam particles after beam particle selection, then writes the fitted parameters to file.",
        formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Config(parser, required = True)
    cross_section.ApplicationArguments.Output(parser)

    parser.add_argument("--toy_data", dest = "toy_data", type = str, help = "toy data, proivde a hdf5 toy file or toy config")
    parser.add_argument("--toy_template", dest = "toy_template", type = str, help = "toy template, proivde a hdf5 toy file or toy config")
    parser.add_argument("--pdsp", dest = "pdsp", action = "store_true", help = "run the analysis with the PDSP samples")
    parser.add_argument("--all", dest = "all", action = "store_true", help = "measure all exclusive cross sections.")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())

    if args.toy_data and (not args.toy_template):
        raise Exception("if toy data is provided toy template must also be provided")
    if (not args.toy_data) and args.toy_template:
        raise Exception("if toy template is provided toy data must also be provided")

    print(vars(args))
    main(args)