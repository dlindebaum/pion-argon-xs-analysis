#!/usr/bin/env python3
"""
Created on: 06/10/2023 00:05

Author: Shyam Bhuller

Description: Analyses MC ntuples in order to determine parameters used to emulate selection efficiency and detector effects for the toy model.
"""

import argparse
import json
import os

import awkward as ak
import numpy as np
import pandas as pd
import scipy.stats as stats

from apps import cex_analyse
from python.analysis import cross_section, Master, Plots, Tags

from rich import print


def SaveCorrectionParams(params : any, file : str):
    """ Saves an objects to a json file. params should be an object serialisable by json (no numpy arrays etc.).

    Args:
        params (any): correction parameters.
        file (str): file path.
    """
    with open(file, "w") as f:
        json.dump(params, f)


def ComputeQuantities(mc : Master.Data, args : argparse.Namespace) -> dict[dict, dict]:
    """ Compute Quantities used for the cross section measurement.

    Args:
        mc (Master.Data): mc events.
        args (argparse.Namespace): application arguments.

    Returns:
        dict[dict, dict]: dictionary of quantities, one for reco and truth.
    """
    reco_upstream_loss = cross_section.UpstreamEnergyLoss(cross_section.KE(mc.recoParticles.beam_inst_P, cross_section.Particle.from_pdgid(211).mass), args.upstream_loss_correction_params["value"])
    
    reco_KE_ff = cross_section.KE(mc.recoParticles.beam_inst_P, cross_section.Particle.from_pdgid(211).mass) - reco_upstream_loss
    reco_KE_int = reco_KE_ff - cross_section.RecoDepositedEnergy(mc, reco_KE_ff, "bb")
    reco_track_length = mc.recoParticles.beam_track_length

    true_KE_ff = mc.trueParticles.beam_KE_front_face
    true_KE_int = mc.trueParticles.beam_traj_KE[:, -2]
    true_track_length = mc.trueParticles.beam_track_length

    return {
        "reco" : {"KE_init" : reco_KE_ff, "KE_int" : reco_KE_int, "z_int" : reco_track_length},
        "true" : {"KE_init" : true_KE_ff, "KE_int" : true_KE_int, "z_int" : true_track_length}
    }


def ResolutionStudy(plot_book : Plots.PlotBook, reco_quantity : ak.Array, true_quantity : ak.Array, mask : ak.Array = None, label = "quantity(units)", plot_range = None, residual_range = None, fit_functions : list[cross_section.Fitting.FitFunction] = [cross_section.Fitting.gaussian, cross_section.Fitting.student_t]) -> dict:
    """ Study of residuals of cross section inputs, used to smear toy observables. Done by fitting a curve to the residual, returning the fit parameters.

    Args:
        plot_book (Plots.PlotBook): object to save plots to pdf file
        reco_quantity (ak.Array): reco quantity
        true_quantity (ak.Array): truth quantity
        mask (ak.Array, optional): mask to apply to quantity. Defaults to None.
        label (str, optional): plot label. Defaults to "quantity(units)".
        plot_range (_type_, optional): plot range. Defaults to None.
        residual_range (_type_, optional): range of residual to fit to. Defaults to None.
        fit_functions (list[cross_section.Fitting.FitFunction], optional): fit functions to try fit to the residuals. Defaults to [cross_section.Fitting.gaussian, cross_section.Fitting.student_t].

    Returns:
        dict: fit parameters for each fit functions. 
    """
    residual = reco_quantity - true_quantity
    if mask is not None:
        residual = residual[mask]
    Plots.PlotHistComparison([reco_quantity, true_quantity], labels = ["reco", "true"], xlabel = label, x_range = plot_range)
    plot_book.Save()
    Plots.PlotHist(residual, xlabel = "residual " + label, range = residual_range)
    plot_book.Save()
    Plots.PlotHist(residual, xlabel = "residual " + label, range = residual_range, y_scale = "log")
    plot_book.Save()

    counts, edges = np.histogram(np.array(residual), 100, range = residual_range)
    centers = (edges[1:] + edges[:-1]) / 2

    params = {}
    errors = {}
    for f in fit_functions:
        Plots.plt.figure()
        params[f.__name__], errors[f.__name__] = cross_section.Fitting.Fit(centers, counts, np.sqrt(counts), f, plot = True, xlabel = label, ylabel = "counts")
        plot_book.Save()
    params_formatted = {p : {"function" : p, "values" : {f"p{i}" : params[p][i] for i in range(len(params[p]))}} for p in params}

    params_formatted = {}
    for p in params:
        params_formatted[p] = {
            "function" : p,
            "values" : {f"p{i}" : params[p][i] for i in range(len(params[p]))},
            "errors" : {f"p{i}" : errors[p][i] for i in range(len(errors[p]))},
            "range" : residual_range
            }
    return params_formatted


def PlotCorrelationMatrix(counts : np.array = None, true_labels = None, reco_labels = None, title : str = None, newFigure : bool = True):
    """ Plots Correlation matrix of two sets of regions.

    Args:
        true_regions (dict): true regions
        reco_regions (dict): reco regions
        title (str, optional): title. Defaults to None.
    """
    fractions = counts / np.sum(counts, axis = 1)[:, np.newaxis]

    # fractions, counts = ComputeFractions(true_regions, reco_regions, return_counts = True)
    if newFigure: Plots.plt.figure()
    Plots.plt.imshow(counts/np.max(counts, axis = 0), cmap = "cool", origin = "lower")
    Plots.plt.colorbar(label = "counts (column normalised)")

    true_counts = np.sum(counts, axis = 1)
    reco_counts = np.sum(counts, axis = 0)

    true_counts = [f"{true_labels[t].replace('_', ' ')}\n({true_counts[t]})" for t in range(len(true_labels))]
    reco_counts = [f"{reco_labels[r].replace('_', ' ')}\n({reco_counts[r]})" for r in range(len(reco_labels))]


    Plots.plt.gca().set_xticks(np.arange(len(reco_counts)), labels=reco_counts)
    Plots.plt.gca().set_yticks(np.arange(len(true_counts)), labels=true_counts)

    Plots.plt.xlabel("reco region")
    Plots.plt.ylabel("true process")
    Plots.plt.xticks(rotation = 30)
    Plots.plt.yticks(rotation = 30)

    if title is not None:
        Plots.plt.title(title + "| Key: (counts, fraction(%))")
    else:
        Plots.plt.title("Key: (counts, fraction(%))")

    for (i, j), z in np.ndenumerate(counts):
        Plots.plt.gca().text(j, i, f"{z},\n{fractions[i][j]*100:.2g}%", ha='center', va='center', fontsize = 8)
    Plots.plt.grid(False)
    Plots.plt.tight_layout()



@Master.timer
def Smearing(cross_section_quantities : dict, true_pion_mask : ak.Array, args : argparse.Namespace, ranges : dict, labels : dict):
    """ Smearing study, fits functions to residuals of cross section quantities and saves the fit parameters of the best fit to file.
        (Best is double crystal ball).

    Args:
        cross_section_quantities (dict): cross section quantities
        true_pion_mask (ak.Array): true inelastic pion mask
        args (argparse.Namespace): application arguments
        ranges (dict): plot ranges
        labels (dict): plot x labels
    """
    selected_quantities = {}
    for k, v in cross_section_quantities.items():
        selected_quantities[k] = {i : j[true_pion_mask] for i, j in v.items()}

    os.makedirs(args.out + "smearing/", exist_ok = True)
    
    trial_functions = [cross_section.Fitting.gaussian, cross_section.Fitting.student_t, cross_section.Fitting.crystal_ball, cross_section.Fitting.double_crystal_ball]

    pdf = Plots.PlotBook(args.out + "smearing/smearing_study")
    track_length_params =  ResolutionStudy(pdf, selected_quantities["reco"]["z_int"], selected_quantities["true"]["z_int"], selected_quantities["reco"]["z_int"] != 0, labels["z_int"], ranges["z_int"], [-25, 25], trial_functions)
    print(track_length_params)
    KE_ff_params = ResolutionStudy(pdf, selected_quantities["reco"]["KE_init"], selected_quantities["true"]["KE_init"], selected_quantities["reco"]["KE_init"] != 0, labels["KE_init"], ranges["KE_init"], [-250, 250], trial_functions)
    print(KE_ff_params)
    KE_int_params = ResolutionStudy(pdf, selected_quantities["reco"]["KE_int"], selected_quantities["true"]["KE_int"], selected_quantities["reco"]["KE_int"] != 0, labels["KE_int"], ranges["KE_int"], [-250, 250], trial_functions)
    print(KE_int_params)
    pdf.close()

    SaveCorrectionParams(track_length_params["double_crystal_ball"], args.out + "smearing/track_length_resolution.json")
    SaveCorrectionParams(KE_ff_params["double_crystal_ball"], args.out + "smearing/KE_ff_resolution.json")
    SaveCorrectionParams(KE_int_params["double_crystal_ball"], args.out + "smearing/KE_int_resolution.json")
    return


def GetTotalPionInelMasks(mc : Master.Data) -> ak.Array:
    """ Returns the mask which selects true inelastic pi+.

    Args:
        mc (Master.Data): mc events.

    Returns:
        ak.Array: mask.
    """
    particle_tags = Tags.GenerateTrueBeamParticleTags(mc)
    return particle_tags["$\\pi^{+}$:inel"].mask


def GetTotalPionCounts(pion_mask : ak.Array, quantities : dict[str, ak.Array], bins : dict[str, np.array], ranges : dict[str, list]) -> np.array:
    """ Counts number of pi+ in bins of each quantity.

    Args:
        pion_mask (ak.Array): pion mask.
        quantities (dict[str, ak.Array]): quantities.
        bins (dict[str, np.array]): bins for each quantity.
        ranges (dict[str, list]): ranges for each quantity.

    Returns:
        np.array: _description_
    """
    counts = {q : np.histogram(np.array(quantities[q][pion_mask]), bins[q], range = ranges[q])[0] for q in quantities}
    return counts


def Efficiency(selected_count : np.array, total_count : np.array) -> tuple[np.array, np.array]:
    """ Calcualtes selection efficiency and binomial error.

    Args:
        selected_count (np.array): number of selected events
        total_count (np.array): number of total events

    Returns:
        tuple[np.array, np.array]: efficiency, error
    """
    p = selected_count / total_count
    p = np.nan_to_num(p)
    error = (p * (1 - p) / total_count)**0.5
    return p, error

@Master.timer
def BeamSelectionEfficiency(quantities : dict, pion_inel_mask : ak.Array, args : argparse.Namespace, ranges : dict, labels : dict, bins : dict):
    """ Study which looks at the beam selection efficiency as a function of each cross section quantity, then saves the per bin efficiencies to file to use for the toy simulation.

    Args:
        quantities (dict): cross section quantities
        pion_inel_mask (ak.Array): true inelastic pion mask without any selection
        args (argparse.Namespace): application arguments
        ranges (dict): plot ranges
        labels (dict): plot xlabels 
        bins (dict): plot bins
    """
    initial_counts_true = GetTotalPionCounts(pion_inel_mask, quantities, bins, ranges)
    
    selection = args.selection_masks["mc"]["beam"]

    selected_pion_inel_mask = ak.Array(pion_inel_mask)
    selected_quantities = {i : ak.Array(quantities[i]) for i in quantities}

    for s in selection:
        mask = selection[s]
        selected_pion_inel_mask = selected_pion_inel_mask[mask]
        for q in selected_quantities:
            selected_quantities[q]= selected_quantities[q][mask]

    selected_counts_true = GetTotalPionCounts(selected_pion_inel_mask, selected_quantities, bins, ranges)

    os.makedirs(args.out + "pi_beam_efficiency/", exist_ok = True)
    pdf = Plots.PlotBook(args.out + "pi_beam_efficiency/efficiency_study")

    x = {i : (bins[i][1:] + bins[i][:-1])/2 for i in initial_counts_true}
    e = {i : Efficiency(selected_counts_true[i], initial_counts_true[i]) for i in initial_counts_true}

    for _, i in Plots.IterMultiPlot(initial_counts_true):
        Plots.Plot(x[i], e[i][0], yerr = e[i][1], xlabel = "true" + labels[i], ylabel = "beam $\pi^{+}$:inel selection efficiency", newFigure = False)
    pdf.Save()
    pdf.close()

    selection_efficiency_info ={
        "bins" : pd.DataFrame(bins),
        "efficiency" : pd.DataFrame({i : e[i][0] for i in e}),
        "error" : pd.DataFrame({i : e[i][1] for i in e})
    }

    with pd.HDFStore(args.out + "pi_beam_efficiency/beam_selection_efficiencies_true.hdf5") as hdf:
        for i in selection_efficiency_info:
            hdf.put(i, selection_efficiency_info[i])
    return

@Master.timer
def RecoRegionSelection(mc : Master.Data, args : argparse.Namespace):
    """ Study which computes a correlation matrix ofthe event faction for reco regions and true regions.
        Saved to file to be used in toy simulation.

    Args:
        mc (Master.Data): mc events.
        args (argparse.Namespace): application arguments.
    """
    reco_regions, true_regions = cex_analyse.RegionSelection(mc, args, True)

    os.makedirs(args.out + "reco_regions/", exist_ok = True)
    pdf = Plots.PlotBook(args.out + "reco_regions/reco_regions_study")
    counts = cross_section.Toy.ComputeCounts(true_regions, reco_regions)
    PlotCorrelationMatrix(counts, list(true_regions.keys()), list(reco_regions.keys()))
    pdf.Save()
    pdf.close()


    fractions_df = counts / np.sum(counts, axis = 1)[:, np.newaxis]
    fractions_df = pd.DataFrame(np.array(fractions_df).T, columns = true_regions, index = reco_regions)# columns are the true regions, so index over those to get the fractions
    fractions_df.to_hdf(args.out + "reco_regions/reco_region_fractions.hdf5", "df")
    return

@Master.timer
def MeanTrackScoreKDE(mc : Master.Data, args : argparse.Namespace):
    mc_copy = mc.Filter(returnCopy = True)

    for s, m in args.selection_masks["mc"]["beam"].items():
        print(s)
        mc_copy.Filter([m], [m])
    mc_copy.Filter([args.selection_masks["mc"]['null_pfo']['ValidPFOSelection']])

    has_pfo = cross_section.BeamParticleSelection.HasFinalStatePFOsCut(mc_copy) #! add as preselection
    mc_copy.Filter([has_pfo], [has_pfo])

    mean_track_score = ak.fill_none(ak.mean(mc_copy.recoParticles.trackScore, axis = -1), -0.05)

    true_processes = cross_section.EventSelection.create_regions_new(mc_copy.trueParticles.nPi0, mc_copy.trueParticles.nPiPlus + mc_copy.trueParticles.nPiMinus)

    kdes = {}
    for k, v in true_processes.items():
        kdes[k] = stats.gaussian_kde(mean_track_score[v])
        kdes[k].set_bandwidth(0.2)

    os.makedirs(args.out + "meanTrackScoreKDE/", exist_ok = True)
    cross_section.SaveSelection(args.out + "meanTrackScoreKDE/kdes.dill", kdes)
    return

@Master.timer
def main(args : argparse.Namespace):
    cross_section.SetPlotStyle(True, 100)

    ranges = {
        "KE_init" : [0, 1250],
        "KE_int" : [0, 1250],
        "z_int" : [0, 500]
    }
    bins = {r : np.linspace(min(ranges[r]), max(ranges[r]), 50) for r in ranges}
    labels = {
        "KE_init" : "$KE^{reco}_{ff}$ (MeV)",
        "KE_int" : "$KE^{reco}_{int}$ (MeV)",
        "z_int" : "$l^{reco}$ (cm)"
    }

    mc = Master.Data(args.mc_file, -1, nTuple_type = args.ntuple_type)

    pion_inel_mask = GetTotalPionInelMasks(mc)
    cross_section_quantities = ComputeQuantities(mc, args)
    Smearing(cross_section_quantities, mc.trueParticles.pdg[:, 0] == 211, args, ranges, labels)

    BeamSelectionEfficiency(cross_section_quantities["true"], pion_inel_mask, args, ranges, labels, bins)

    RecoRegionSelection(mc, args)

    MeanTrackScoreKDE(mc, args)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Analyses MC ntuples in order to determine parameters used to emulate selection efficiency and detector effects for the toy model.")

    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Config(parser)
    args = parser.parse_args()
    args = cross_section.ApplicationArguments.ResolveArgs(args)

    args.out += "toy_parameters/"

    print(vars(args))
    main(args)