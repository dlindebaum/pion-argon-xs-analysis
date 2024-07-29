#!/usr/bin/env python3
"""
Created on: 23/05/2023 12:58

Author: Shyam Bhuller

Description: Applies beam particle selection, photon shower candidate selection and writes out shower energies.
"""
import argparse
import os
import warnings

import awkward as ak
import numpy as np
import pandas as pd
from rich import print

from scipy.optimize import curve_fit

from python.analysis import Master, Processing, cross_section, EventSelection, Tags, SelectionTools, Plots, Fitting

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # hide pesky pandas warnings (performance is actually ok)


#! replace with selection tools equivalent
def CreatePFOMasks(masks : dict[ak.Array]) -> ak.Array:
    """ combine a dicitonary of masks into a single mask.

    Args:
        masks (dict[ak.Array]): masks.

    Returns:
        ak.Array: combined mask.
    """
    mask = None
    for m in masks:
        if mask is None:
            mask = masks[m]
        else:
            mask = mask & masks[m]
    return mask


def run(i, file, n_events, start, selected_events, args):
    output = {}

    events = Master.Data(file, n_events, start, args["nTuple_type"], args["pmom"])

    for s, a in zip(args["beam_selection"]["selections"].values(), args["beam_selection"]["mc_arguments"].values()):
        mask = s(events, **a)
        events.Filter([mask], [mask])
    photon_masks = {}
    if args["valid_pfo_selection"] is True:
        for k, s, a in zip(args["photon_selection"]["selections"].keys(), args["photon_selection"]["selections"].values(), args["photon_selection"]["mc_arguments"].values()):
            photon_masks[k] = s(events, **a)
    photon_mask = SelectionTools.CombineMasks(photon_masks)
    events.Filter([photon_mask])
    print("making pairs")
    pairs = EventSelection.NPhotonCandidateSelection(events, photon_mask[photon_mask], 2)

    shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photon_mask[photon_mask] & pairs)

    params = ["angle", "sub_energy", "lead_energy", "mass"]

    for p in params:
        for t in ["reco", "true"]:
            output[f"shower_pairs_{t}_{p}"] = ak.flatten(getattr(shower_pairs, f"{t}_{p}")[pairs])

    params = {
        "reco" : ["shower_direction", "shower_start_pos", "shower_length", "n_hits", "n_hits_collection", "shower_energy"],
        "true" : ["direction", "shower_start_pos", "energy"]
    }

    for (k, param), particleData in zip(params.items(), {"reco" : events.recoParticles, "true" : events.trueParticlesBT}.values()):
        for p in param:
            if hasattr(particleData, p):
                v = getattr(particleData, p)
                # if v is None: continue
                if v is not None:
                    if hasattr(v, "x"):
                        for i in ["x", "y", "z"]:
                            output[f"{k}_{p}_{i}"] = ak.flatten(v[i])
                    else:
                        output[f"{k}_{p}"] = ak.flatten(v)

    output["true_mother"] = ak.flatten(events.trueParticlesBT.motherPdg)
    pfo_tags = Tags.GenerateTrueParticleTagsPi0Shower(events)
    output["pi0_photon"] = ak.flatten(pfo_tags["$\\gamma$:beam $\\pi^0$"].mask | pfo_tags["$\\gamma$:other $\\pi^0$"].mask)

    pi0_tags = Tags.GeneratePi0Tags(events, photon_mask[photon_mask] & pairs)
    for t in pi0_tags:
        pi0_tags[t].mask = pi0_tags[t].mask[pairs]
    output["pi0_tags"] = pi0_tags
    
    fs_tags = EventSelection.GenerateTrueFinalStateTags(events)
    for t in fs_tags:
        fs_tags[t].mask = fs_tags[t].mask[pairs]
    output["final_state_tags"] = fs_tags

    return output


def PhotonSelection(df : pd.DataFrame, book : Plots.PlotBook = Plots.PlotBook.null):
    pi0_mother = df.true_mother == 111
    counts = {"pi0_daughter" : len(pi0_mother[df.pi0_photon]), "other" : len(df.pi0_photon) - len(df.pi0_photon[df.pi0_photon])}
    print(counts)
    print({c : counts[c] / len(df) for c in counts})

    Plots.PlotHist([df[pi0_mother].fractional_error, df[~pi0_mother].fractional_error], stacked = False, histtype = "step", range = [-1, 1], xlabel = "Shower energy fractional error", label = ["$\pi^{0}$ daughter", "other"])
    book.Save()

    df = df[pi0_mother]
    Plots.PlotHist2DImshowMarginal(df.reco_shower_energy, df.fractional_error, bins = 100, xlabel = "Reco shower energy (MeV)", ylabel = "Shower energy fractional error", cmap = "plasma", x_range = [0, 1500], y_range = [-1, 1], norm = False)
    book.Save()
    return df


def LinearFit(x, m):
    return m * x


def binned_dataframe(df : pd.DataFrame, bins : list, energy_range : list) -> list:
    """ split a dataframe into a list of dataframes based on reco energy bins

    Args:
        bins (list): bin edges

    Returns:
        list: list of binned data frames
    """
    binned_data = []
    for i in range(1, len(bins)):
        data = df[(df.reco_shower_energy < bins[i]) & (df.reco_shower_energy > bins[i-1])]
        binned_data.append(data[data.true_energy < max(energy_range)])
    return binned_data


def linear_fit(df : pd.DataFrame, bins : np.ndarray, energy_range : list, book : Plots.PlotBook = Plots.PlotBook.null) -> float:
    """ perform linear fit of true energy vs reco energy, equivalent to the correction done in the microboone simulation paper.

    Args:
        bins (list): reco energy bins

    Returns:
        float: gradient of linear fit (correction)
    """
    x = (bins[1:] + bins[:-1]) / 2
    y = np.array([d.true_energy.mean() for d in binned_dataframe(df, bins, energy_range)])

    popt, pcov = curve_fit(LinearFit, x, y)

    print(popt, pcov**0.5)
    perr = np.array([pcov[i][i] for i in range(len(popt))])**0.5

    Plots.Plot(x, y, marker = "x", linestyle = "")
    Plots.Plot(x, LinearFit(x, *popt), newFigure = False, label = "fit")
    Plots.plt.fill_between(x, LinearFit(x, *(popt + perr)), LinearFit(x, *(popt - perr)), color = "C3", alpha = 0.5)
    Plots.Plot(x, x, newFigure = False, label = "$y = x$", xlabel = "Reco shower energy (MeV)", ylabel = "True shower energy (MeV)")
    Plots.plt.legend()


    Plots.PlotHist2D(df.true_energy, df.reco_shower_energy, x_range = [0, 1500], y_range = [0, 1500], cmap = "summer")
    Plots.Plot(x, LinearFit(x, *popt), newFigure = False, label = "fit", color = "C0")
    Plots.Plot(x, x, newFigure = False, color = "black", label = "$y = x$")
    Plots.Plot(x, y, marker = "x", linestyle = "", ylabel = "True shower energy (MeV)", xlabel = "Reco shower energy (MeV)", newFigure = False, color = "C0")
    book.Save()
    return popt[0]


def LinearFitPerformance(df : pd.DataFrame, linear_correction : float, book : Plots.PlotBook = Plots.PlotBook.null):
    """ Performance plots and metrics of linear correction.

    Args:
        linear_correction (float): correction from fit
    """
    corrected_energy = cross_section.EnergyCorrection.LinearCorrection(df.reco_shower_energy, linear_correction)

    fe = (df.reco_shower_energy / df.true_energy) - 1
    fec = (corrected_energy / df.true_energy) - 1

    print(f"correction factor : {linear_correction}")
    print(f"mean shower energy fractional error: {np.mean(fe)} +- {np.std(fe)}")
    print(f"mean shower energy fractional error after correction: {np.mean(np.mean(fec))} +- {np.std(fec)}")

    Plots.PlotHistComparison([df.reco_shower_energy, corrected_energy], labels = ["uncorrected", "corrected"], x_range = [0, 2000], xlabel = "Shower energy (MeV)")
    Plots.PlotHistComparison([df.reco_shower_energy - df.true_energy, corrected_energy - df.true_energy], labels = ["uncorrected", "corrected"], x_range=[-500, 500], xlabel = "Shower energy residual (MeV)")
    book.Save()

    Plots.PlotHist2DComparison([df.true_energy, df.true_energy], [fe, fec], [0, 2000], [-1, 1], bins = 50, cmap = "Accent", xlabels = ["True shower energy (MeV)"]*2, ylabels = ["Fractional error"]*2, titles = ["uncorrected", "corrected"])
    book.Save()
    return


def IterBinnedDF(df : pd.DataFrame, variable : str, v_range: list, reco_bins : list) -> pd.DataFrame:
    for i in range(1, len(reco_bins)):
        binned_data = df[(df.reco_shower_energy > reco_bins[i-1]) & (df.reco_shower_energy < reco_bins[i])]
        ranged_data = binned_data[(binned_data[variable] > min(v_range)) & (binned_data[variable] < max(v_range))]
        yield ranged_data[variable]


def calculate_mean(df : pd.DataFrame, variable : str, v_range: list, reco_bins : list) -> np.ndarray:
    means = []
    for column in IterBinnedDF(df, variable, v_range, reco_bins):
        means.append(column.mean())
    return np.array(means)


def calculate_sem(df : pd.DataFrame, variable : str, v_range: list, reco_bins : list) -> np.ndarray:
    sem = []
    for column in IterBinnedDF(df, variable, v_range, reco_bins):
        sem.append(column.std() / np.sqrt(len(column)))
    return np.array(sem)


def create_bins_df(value : pd.Series, n_entries, v_range : list = None) -> np.ndarray:
    sorted_value = value.sort_values()
    n_bins = len(sorted_value) // n_entries

    bins = []
    for i in range(n_bins + 1):
        mi = sorted_value.values[i * n_entries]
        bins.append(mi)
    if v_range:
        bins[0] = min(v_range)
        bins[-1] = max(v_range)
    return np.array(bins)


def CalculateCentralValues(df : pd.date_range, bins : np.ndarray, book : Plots.PlotBook) -> dict:
    central_values = {}

    with cross_section.PlotStyler().Update(font_scale = 1.3):
        for k, v in {"student_t" : Fitting.student_t, "gaussian" : Fitting.gaussian}.items():
            central_values[k] = Fitting.ExtractCentralValues_df(df, "reco_shower_energy", "fractional_error", [-1, 1], [v], bins, 20, bin_label = "$E^{reco}$", bin_units = "(MeV)")
            book.Save()
        # central_values["gaussian"] = Fitting.ExtractCentralValues_df(df, "reco_shower_energy", "fractional_error", [-1, 1], [Fitting.gaussian], bins, 20)
        # book.Save()
        central_values["mean"] = [calculate_mean(df, "fractional_error", [-1, 1], bins), calculate_sem(df, "fractional_error", [-1, 1], bins)]
        book.Save()
    return central_values


def ResponseFits(central_values : dict, bins : np.ndarray, book : Plots.PlotBook) -> dict:
    x = (bins[1:] + bins[:-1]) / 2
    response_params = {}

    for _, cv in Plots.IterMultiPlot(central_values):
        print(cv)
        print(central_values[cv][1])
        popt, perr = Fitting.Fit(x, central_values[cv][0], central_values[cv][1], cross_section.EnergyCorrection.ResponseFit, method = "lm", plot = True, xlabel = "Reco shower energy (MeV)", ylabel = "Fractional error", maxfev = int(1E6))
        Plots.plt.ylim(-1, 1)
        Plots.plt.title(cv)
        response_params[cv] = {"value" : popt, "error" : perr}

    print(response_params)
    book.Save()


    for cv in central_values:
        Plots.plt.figure()
        popt, _ = Fitting.Fit(x, central_values[cv][0], central_values[cv][1], cross_section.EnergyCorrection.ResponseFit, method = "lm", plot = True, xlabel = "Reco shower energy (MeV)", ylabel = "Fractional error", maxfev = int(1E6))
        Plots.plt.ylim(-1, 1)
        book.Save()
    return response_params


def MethodComparison(df : pd.DataFrame, linear_correction : float, response_params : dict, bins : np.ndarray, energy_range : list, book : Plots.PlotBook):
    energies = {"uncorrected" : df.reco_shower_energy, "linear" : df.reco_shower_energy / linear_correction}

    for p in response_params:
        energies[p] = cross_section.EnergyCorrection.ResponseCorrection(df.reco_shower_energy, *response_params[p]["value"])

    energies = {"uncorrected" : energies["uncorrected"], "gaussian" : energies["gaussian"]}

    Plots.PlotHistComparison(list(energies.values()), labels = list(energies.keys()), x_range = energy_range, xlabel = "Shower energy (MeV)")

    fe = {i : (energies[i]/df.true_energy) - 1 for i in energies}

    Plots.PlotHist2DComparison([df.true_energy]*len(fe), list(fe.values()), energy_range, [-1, 1], bins = 50, cmap = "Accent", xlabels = ["True shower energy (MeV)"]*len(fe), ylabels = ["Fractional error"]*len(fe), titles = list(fe.keys()))
    book.Save()

    Plots.PlotHist2DComparison([df.reco_shower_energy]*len(fe), list(fe.values()), energy_range, [-1, 1], bins = 50, cmap = "Accent", xlabels = ["Uncorrected reco shower energy (MeV)"]*len(fe), ylabels = ["Fractional error"]*len(fe), titles = list(fe.keys()))
    book.Save()

    for _, (l, e) in Plots.IterMultiPlot(energies.items()):
        Plots.PlotHist2D(e, df.true_energy, bins = 100, title = l, xlabel = "Reco shower energy (MeV)", ylabel = "True shower energy (MeV)", x_range = energy_range, y_range = energy_range, cmap = "summer", newFigure = False)
        Plots.plt.plot(energy_range, energy_range)
    book.Save()

    tab = {}
    for l, f in fe.items():
        v = f[(f > -1) & (f < 1)]
        tab[l] = {"$\mu$" : v.mean(), "$\sigma$" : v.std()}
    tab = pd.DataFrame(tab)

    for i, f in Plots.IterMultiPlot(fe):
        binned_data = [fe[f][(df.reco_shower_energy > bins[i + 0]) & (df.reco_shower_energy < bins[i + 1])] for i in range(len(bins)-1)]

        mean = [d.mean() for d in binned_data]
        sem = [d.std()/(len(d)**0.5) for d in binned_data]
        Plots.Plot((bins[1:] + bins[:-1]) / 2, mean, yerr = sem, marker = "x", capsize = 3, newFigure = False)
        Plots.PlotHist2D(df.reco_shower_energy, fe[f], bins = [bins, 50], x_range = energy_range, y_range = [-1, 1], newFigure = False, xlabel = "Uncorrected reco energy (MeV)", ylabel = "Fractional error", title = f)
    book.Save()

    Plots.plt.figure()
    for i, f in enumerate(fe):
        binned_data = [fe[f][(df.reco_shower_energy > bins[i + 0]) & (df.reco_shower_energy < bins[i + 1])] for i in range(len(bins)-1)]
        mean = [d.mean() for d in binned_data]
        sem = [d.std()/(len(d)**0.5) for d in binned_data]
        Plots.Plot((bins[1:] + bins[:-1]) / 2, mean, yerr = sem, marker = "x", capsize = 3, label = f, newFigure = False)
        Plots.plt.xlabel("Uncorrected reco energy (MeV)")
        Plots.plt.ylabel("Fractional error")
        Plots.plt.legend()
    book.Save()
    return tab

@Master.timer
def main(args):
    cross_section.PlotStyler.SetPlotStyle(False)
    out = args.out + "shower_energy_correction/"

    if (not os.path.isfile(out + "photon_energies.hdf5")) or args.regen:
        output = cross_section.RunProcess(args.ntuple_files["mc"], False, args, run)

        output_photons = pd.DataFrame({i : output[i] for i in output if "shower_pairs" not in i and "tags" not in i})
        output_pairs = pd.DataFrame({i : output[i] for i in output if "shower_pairs" in i and "tags" not in i})
        output_tags = pd.DataFrame({i : output[i] for i in output if "tags" in i})

        print(output_photons)
        print(output_pairs)
        print(output_tags)

        os.makedirs(out, exist_ok = True)
        output_photons.to_hdf(out + "photon_energies.hdf5", "all_photons")
        output_pairs.to_hdf(out + "photon_energies.hdf5", "photon_pairs")
        output_tags.to_hdf(out + "photon_energies.hdf5", "tags")

    df = cross_section.ReadHDF5(out + "photon_energies.hdf5")["all_photons"]
    df["residual"] = df.reco_shower_energy - df.true_energy
    df["fractional_error"] = (df.reco_shower_energy / df.true_energy) - 1

    with Plots.PlotBook(out + "plots.pdf") as book:
        #* initial plots
        PhotonSelection(df, book)

        #* linear correction
        energy_range = args.shower_correction["energy_range"]
        bins = np.linspace(min(energy_range), max(energy_range), 11)
        linear_correction = linear_fit(df, bins, energy_range, book)
        LinearFitPerformance(df, linear_correction, book)

        #* response correction
        bins = np.array(create_bins_df(df.reco_shower_energy, int(len(df.reco_shower_energy)//11), energy_range), dtype = int)

        central_values = CalculateCentralValues(df, bins, book)
        response_params = ResponseFits(central_values, bins, book)

        tab = MethodComparison(df, linear_correction, response_params, bins, energy_range, book)
    
    tab.T.style.format(precision = 3).to_latex(out + "table.tex")

    params = {p :
        {
            "value" : {f"p{i}" : response_params[p]["value"][i] for i in range(len(response_params[p]["value"]))},
            "error" : {f"p{i}" : response_params[p]["error"][i] for i in range(len(response_params[p]["error"]))}
        }
    for p in response_params}

    for name, p in params.items():
        sf = [len(f'{p["error"][f"p{i}"]:.1g}') - 1 for i in range(len(p["value"]))]
        table = pd.DataFrame({f"$p_{{{i}}}$" : f'{p["value"][f"p{i}"]:.{sf[i]}f} $\pm$ {p["error"][f"p{i}"]:.1g}' for i in range(len(p["value"]))}, index = [0])
        table.style.hide(axis = "index").to_latex(out + name + ".tex")
        cross_section.SaveConfiguration(p, out + name + ".json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection and saves properties of photon shower candidate PFOs to hdf5 file (MC only)", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Regen(parser)

    args = parser.parse_args()
    args = cross_section.ApplicationArguments.ResolveArgs(args)

    print(vars(args))
    main(args)