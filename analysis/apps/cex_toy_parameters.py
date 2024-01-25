#!/usr/bin/env python3
"""
Created on: 06/10/2023 00:05

Author: Shyam Bhuller

Description: Analyses MC ntuples in order to determine parameters used to emulate selection efficiency and detector effects for the toy model.
"""

import argparse
import os

import awkward as ak
import numpy as np
import pandas as pd
import scipy.stats as stats

from alive_progress import alive_bar

from python.analysis import cross_section, Master, Plots, Tags, SelectionTools

from apps.cex_analysis_input import RegionSelection

from rich import print


@Master.timer
def ComputeQuantities(mc : Master.Data, args : argparse.Namespace) -> dict[dict, dict]:
    """ Compute Quantities used for the cross section measurement.

    Args:
        mc (Master.Data): mc events.
        args (argparse.Namespace): application arguments.

    Returns:
        dict[dict, dict]: dictionary of quantities, one for reco and truth.
    """
    with alive_bar(title = "computng reco quantities") as bar:
        reco_upstream_loss = cross_section.UpstreamEnergyLoss(cross_section.KE(mc.recoParticles.beam_inst_P, cross_section.Particle.from_pdgid(211).mass), args.upstream_loss_correction_params["value"])
        
        reco_KE_ff = cross_section.KE(mc.recoParticles.beam_inst_P, cross_section.Particle.from_pdgid(211).mass) - reco_upstream_loss
        reco_KE_int = reco_KE_ff - cross_section.RecoDepositedEnergy(mc, reco_KE_ff, "bb")
        reco_track_length = mc.recoParticles.beam_track_length

    with alive_bar(title = "computng true quantities") as bar:
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
    Plots.PlotHist(residual, xlabel = label, range = residual_range)
    plot_book.Save()
    Plots.PlotHist(residual, xlabel = label, range = residual_range, y_scale = "log")
    plot_book.Save()

    counts, edges = np.histogram(np.array(residual), 100, range = residual_range)
    centers = (edges[1:] + edges[:-1]) / 2

    params = {}
    errors = {}
    for f in fit_functions:
        Plots.plt.figure()
        params[f.__name__], errors[f.__name__] = cross_section.Fitting.Fit(centers, counts, np.sqrt(counts), f, plot = True, xlabel = label, ylabel = "counts", plot_style = "hist")
        plot_book.Save()
        Plots.plt.close()
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


def PlotCorrelationMatrix(counts : np.array = None, true_labels = None, reco_labels = None, title : str = None, newFigure : bool = True, cmap : str = "cool"):
    """ Plots Correlation matrix of two sets of regions.

    Args:
        true_regions (dict): true regions
        reco_regions (dict): reco regions
        title (str, optional): title. Defaults to None.
    """
    fractions = counts / np.sum(counts, axis = 1)[:, np.newaxis]

    # fractions, counts = ComputeFractions(true_regions, reco_regions, return_counts = True)
    if newFigure: Plots.plt.figure()
    Plots.plt.imshow(counts/np.max(counts, axis = 0), cmap = cmap, origin = "lower")
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
def Smearing(cross_section_quantities : dict, true_pion_mask : ak.Array, args : argparse.Namespace, labels : dict):
    """ Smearing study, fits functions to residuals of cross section quantities and saves the fit parameters of the best fit to file.
        (Best is double crystal ball).

    Args:
        cross_section_quantities (dict): cross section quantities
        true_pion_mask (ak.Array): true inelastic pion mask
        args (argparse.Namespace): application arguments
        labels (dict): plot x labels
    """
    selected_quantities = {}
    for k, v in cross_section_quantities.items():
        selected_quantities[k] = {i : j[true_pion_mask] for i, j in v.items()}

    os.makedirs(args.out + "smearing/", exist_ok = True)
    
    trial_functions = [cross_section.Fitting.gaussian, cross_section.Fitting.student_t, cross_section.Fitting.crystal_ball, cross_section.Fitting.double_crystal_ball]

    params = {}
    with Plots.PlotBook(args.out + "smearing/smearing_study") as pdf:
        for k in labels:
            params[k] = ResolutionStudy(pdf, selected_quantities["reco"][k], selected_quantities["true"][k], selected_quantities["reco"][k] != 0, labels[k], args.toy_parameters["plot_ranges"][k], args.toy_parameters["smearing_residual_ranges"][k], trial_functions)

    for q in labels:
        out = args.out + f"smearing/{q}/"
        os.makedirs(out, exist_ok = True)
        for k in params[q]:
            Master.SaveConfiguration(params[q][k], out + f"{k}.json")
    return

@Master.timer
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

    mask = SelectionTools.CombineMasks(selection)

    selected_pion_inel_mask = selected_pion_inel_mask[mask]
    for q in selected_quantities:
        selected_quantities[q] = selected_quantities[q][mask]

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

    Master.DictToHDF5(selection_efficiency_info, args.out + "pi_beam_efficiency/beam_selection_efficiencies_true.hdf5")
    return

@Master.timer
def RecoRegionSelection(mc : Master.Data, args : argparse.Namespace):
    """ Study which computes a correlation matrix ofthe event faction for reco regions and true regions.
        Saved to file to be used in toy simulation.

    Args:
        mc (Master.Data): mc events.
        args (argparse.Namespace): application arguments.
    """
    reco_regions, true_regions = RegionSelection(mc, args, True)

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
    """ Derives mean track score kernels from mc.

    Args:
        mc (Master.Data): mc sample
        args (argparse.Namespace): application arguments.
    """
    mc_copy = mc.Filter(returnCopy = True)

    mask = SelectionTools.CombineMasks(args.selection_masks["mc"]["beam"])
    mc_copy.Filter([mask], [mask])
    mc_copy.Filter([args.selection_masks["mc"]['null_pfo']['ValidPFOSelection']])

    has_pfo = cross_section.BeamParticleSelection.HasFinalStatePFOsCut(mc_copy) #! add as preselection
    mc_copy.Filter([has_pfo], [has_pfo])

    mean_track_score = ak.fill_none(ak.mean(mc_copy.recoParticles.track_score, axis = -1), -0.05)

    true_processes = cross_section.EventSelection.create_regions_new(mc_copy.trueParticles.nPi0, mc_copy.trueParticles.nPiPlus + mc_copy.trueParticles.nPiMinus)

    kdes = {}
    for k, v in true_processes.items():
        kdes[k] = stats.gaussian_kde(mean_track_score[v])
        kdes[k].set_bandwidth(0.2)

    os.makedirs(args.out + "meanTrackScoreKDE/", exist_ok = True)
    cross_section.SaveObject(args.out + "meanTrackScoreKDE/kdes.dill", kdes)
    return


def FitBeamProfile(KE_init : np.array, func : cross_section.Fitting.FitFunction, KE_range : list, bins : int, book : Plots.PlotBook = Plots.PlotBook.null) -> dict:
    """ Fit the MC beam profile for use increating KE init.

    Args:
        KE_init (np.array): initial kinetic energy.
        func (cross_section.Fitting.FitFunction): fit function
        KE_range (list): kinetic energy range
        bins (int): bin numbers
        book (Plots.PlotBook, optional): plot book. Defaults to Plots.PlotBook.null.

    Returns:
        dict: _description_
    """
    x = np.linspace(min(KE_range), max(KE_range), bins)
    y = np.histogram(KE_init, bins = x)[0]

    Plots.plt.figure()
    print(f"{book.open=}")
    params = cross_section.Fitting.Fit((x[1:] + x[:-1])/2, y, np.sqrt(y), func, plot = book.is_open, plot_style = "hist", xlabel = "$KE^{true}_{ff}$ (MeV)")[0]
    book.Save()
    return {
        "function" : func.__name__,
        "parameters" : {f"p{i}" : params[i] for i in range(len(params))},
        "min" : min(KE_range),
        "max" : max(KE_range)
        }

@Master.timer
def BeamProfileStudy(quantities : dict, args : argparse.Namespace, true_beam_mask : ak.Array, func : cross_section.Fitting.FitFunction, KE_range : list):
    """ Try to fit various fit functions to the MC beam profile.

    Args:
        quantities (dict): analysis quanitites.
        args (argparse.Namespace): application arguments.
        true_beam_mask (ak.Array): mask which selects true beam pions.
        func (cross_section.Fitting.FitFunction): fit function to try.
        KE_range (list): kinetic energy range.
    """
    out = args.out + "beam_profile/"
    os.makedirs(out, exist_ok = True)
    with Plots.PlotBook(out + "beam_profile.pdf") as book:
        beam_profile = FitBeamProfile(quantities["true"]["KE_init"][true_beam_mask], func, KE_range, 50, book)
        Master.SaveConfiguration(beam_profile, out + "beam_profile.json")
    return

@Master.timer
def main(args : argparse.Namespace):
    cross_section.SetPlotStyle(True)

    bins = {r : np.linspace(min(args.toy_parameters["plot_ranges"][r]), max(args.toy_parameters["plot_ranges"][r]), 50) for r in args.toy_parameters["plot_ranges"]}
    labels = {
        "KE_init" : "$KE^{res, MC}_{ff}$ (MeV)",
        "KE_int" : "$KE^{res, MC}_{int}$ (MeV)",
        "z_int" : "$l^{res, MC}$ (cm)"
    }

    with alive_bar(title = "load mc") as bar:
        mc = Master.Data(args.mc_file, -1, nTuple_type = args.ntuple_type, target_momentum = args.pmom)

    with alive_bar(title = "create mask") as bar:
        true_pion_mask = mc.trueParticles.pdg[:, 0] == 211

    with alive_bar(title = "pion inel mask") as bar:
        pion_inel_mask = GetTotalPionInelMasks(mc)
    
    cross_section_quantities = ComputeQuantities(mc, args)
    print(cross_section_quantities)

    BeamProfileStudy(cross_section_quantities, args, true_pion_mask, args.toy_parameters["beam_profile"], args.toy_parameters["plot_ranges"]["KE_init"])

    Smearing(cross_section_quantities, true_pion_mask, args, labels)

    BeamSelectionEfficiency(cross_section_quantities["true"], pion_inel_mask, args, args.toy_parameters["plot_ranges"], labels, bins)

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