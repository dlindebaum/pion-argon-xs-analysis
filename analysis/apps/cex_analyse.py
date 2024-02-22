#!/usr/bin/env python3
"""
Created on: 13/11/2023 21:54

Author: Shyam Bhuller

Description: Runs cross section measurement.
"""
import os

import numpy as np

from rich import print

from apps import cex_toy_generator, cex_analysis_input, cex_toy_parameters
from python.analysis import cross_section, Plots


def CreateInitParams(model : cross_section.pyhf.Model, analysis_input : cross_section.AnalysisInput, energy_slices : cross_section.Slices, mean_track_score_bins : np.array) -> np.array:
    """ Create initial parameters for the region fit, using the proportion of reco regions and template to get a rough estimate of the process rates.

    Args:
        model (cross_section.pyhf.Model): fit model
        analysis_input (cross_section.AnalysisInput): analysis input
        energy_slices (cross_section.Slices): energy slices
        mean_track_score_bins (np.array): mean track score bins

    Returns:
        np.array[float]: initial parameter values
    """
    prefit_pred = cross_section.cabinetry.model_utils.prediction(model)
    template_KE = [np.sum(prefit_pred.model_yields[i], 0) for i in range(len(prefit_pred.model_yields))][:-1]
    input_data = cross_section.RegionFit.CreateObservedInputData(analysis_input, energy_slices, mean_track_score_bins)

    init = model.config.suggested_init()
    mu_init = [np.sum(input_data[i]) / np.sum(template_KE[i]) for i in range(len(template_KE))]
    poi = [i for i in model.config.parameters if "mu_" in i]
    poi_ind =  [model.config.par_slice(i).start for i in poi]
    for i, v in zip(poi_ind, mu_init):
        init[i] = v
    return init


def RegionFit(fit_input : cross_section.AnalysisInput, energy_slice : cross_section.Slices, mean_track_score_bins : np.array, template_input : cross_section.AnalysisInput | cross_section.pyhf.Model, suggest_init : bool = False, template_weights : np.array = None, return_fit_results : bool = False, mc_stat_unc : bool = False) -> cross_section.cabinetry.model_utils.ModelPrediction | cross_section.FitResults:
    """ Fit model to analysis input to predict the normalaisations of each process.

    Args:
        fit_input (cross_section.AnalysisInput): observed data
        energy_slice (cross_section.Slices): energy slices
        mean_track_score_bins (np.array): mean track score bins
        template_input (cross_section.AnalysisInput | cross_section.pyhf.Model): template sample or existing model
        suggest_init (bool, optional): estimate normalisations ans use these as the initial values for the fit. Defaults to False.
        template_weights (np.array, optional): weights for the mean track score. Defaults to None.
        return_fit_results (bool, optional): return the raw fit results as  well as the prediction. Defaults to False.

    Returns:
        cross_section.cabinetry.model_utils.ModelPrediction | cross_section.FitResults: model prediction and or the raw fit result.
    """
    if type(template_input) == cross_section.AnalysisInput:
        model = cross_section.RegionFit.CreateModel(template_input, energy_slice, mean_track_score_bins, False, template_weights, mc_stat_unc, True)
    else:
        model = template_input

    observed = cross_section.RegionFit.GenerateObservations(fit_input, energy_slice, mean_track_score_bins, model)

    if suggest_init is True:
        init_params = CreateInitParams(model, fit_input, energy_slice, mean_track_score_bins)
    else:
        init_params = None

    result = cross_section.RegionFit.Fit(observed, model, init_params, [[0, np.inf]]*model.config.npars, verbose = False)
    # result = cross_section.RegionFit.Fit(observed, model, init_params, verbose = False)
    if return_fit_results is True:
        return cross_section.cabinetry.model_utils.prediction(model, fit_results = result), result
    else:
        return cross_section.cabinetry.model_utils.prediction(model, fit_results = result)


def BackgroundSubtraction(data : cross_section.AnalysisInput, process : str, energy_slice : cross_section.Slices, postfit_pred : cross_section.cabinetry.model_utils.ModelPrediction = None, book : Plots.PlotBook = Plots.PlotBook.null) -> tuple[np.array]:
    """ Background subtraction using the fit if a fit result is specified.

    Args:
        data (cross_section.AnalysisInput): observed data
        process (str): signal process
        energy_slice (cross_section.Slices): energy slices
        postfit_pred (cross_section.cabinetry.model_utils.ModelPrediction, optional): fit predictions. Defaults to None.
        book (Plots.PlotBook, optional): plot book. Defaults to Plots.PlotBook.null.

    Returns:
        tuple[np.array]: true histograms (if data is mc), reco histograms postfit, error in reco hitograms postfit
    """
    if data.KE_init_true is not None:
        histograms_true_obs = data.CreateHistograms(energy_slice, process, False, False)
    else:
        histograms_true_obs = None
    histograms_reco_obs = data.CreateHistograms(energy_slice, process, True, False)
    histograms_reco_obs_err = {k : np.sqrt(v) for k, v in histograms_reco_obs.items()}
    
    if postfit_pred is not None:
        print(f"signal: {process}")
        n = cross_section.RegionFit.CreateObservedInputData(data, energy_slice, None)
        N = sum(n)


        if any([c["name"] == "mean_track_score" for c in postfit_pred.model.spec["channels"]]):
            KE_int_prediction = cross_section.RegionFit.SliceModelPrediction(postfit_pred, slice(-1), "KE_int_postfit") # exclude the channel which is the mean track score
        else:
            KE_int_prediction = cross_section.RegionFit.SliceModelPrediction(postfit_pred, slice(0, len(postfit_pred.model_yields)), "KE_int_postfit")

        L = np.sum(KE_int_prediction.model_yields, 0)

        L_err = KE_int_prediction.total_stdev_model_bins[:, :-1] # last entry in the array is the total error for the whole channel (but we want the total error in each process)
        L_err = np.sqrt(np.sum(L_err **2, 0)) # quadrature sum across all bins

        labels = list(data.regions.keys()) #! make property of AnalysisInput dataclass
        L_var_bkg = sum(L_err[process != np.array(labels)]**2)
        L_bkg = sum(L[process != np.array(labels)])


        KE_int_fit = N - L_bkg
        KE_int_fit_err = np.sqrt(N + L_var_bkg)

        if book is not None:
            if data.exclusive_process is not None:
                actual = {l : data.NInteract(energy_slice, data.exclusive_process[l], weights = data.weights) for l in labels}
                actual_sig = actual[process]
                actual_bkg = sum(np.array(list(actual.values()))[process != np.array(labels)])

                energy_bins = np.sort(np.insert(energy_slice.pos, 0, energy_slice.max_pos + energy_slice.width))

                cross_section.RegionFit.PlotPrefitPostFit(actual_sig, np.sqrt(actual_sig), KE_int_fit, KE_int_fit_err, energy_bins, "$N^{reco}_{int,ex}$ (MeV)")
                book.Save()
                cross_section.RegionFit.PlotPrefitPostFit(actual_bkg, np.sqrt(actual_bkg), L_bkg, np.sqrt(L_var_bkg), energy_bins, "$N^{reco}_{int,bkg}$ (MeV)")
                book.Save()

        histograms_reco_obs["int_ex"] = np.where(KE_int_fit < 0, 0, KE_int_fit)
        histograms_reco_obs_err["int_ex"] = KE_int_fit_err

    return histograms_true_obs, histograms_reco_obs, histograms_reco_obs_err


def PlotDataBkgSub(data : cross_section.AnalysisInput, mc : cross_section.AnalysisInput, fit_results : cross_section.FitResults, signal_process : str, energy_slices : cross_section.Slices, scale : float, sample_name : str, data_label = "data", mc_label = "mc", book : Plots.PlotBook = Plots.PlotBook.null):
    labels = {"init" : "$N_{init}$", "int" : "$N_{int}$", "int_ex" : "$N_{int, ex}$", "inc" : "$N_{inc}$"}
    _, histograms_data, histograms_data_err = BackgroundSubtraction(data, signal_process, energy_slices, fit_results)

    histograms_mc_reco = mc.CreateHistograms(energy_slices, signal_process, True, False)

    for _, i in Plots.IterMultiPlot(histograms_data):
        Plots.Plot(energy_slices.pos_overflow, scale * histograms_mc_reco[i], yerr = np.sqrt(scale * histograms_mc_reco[i]), xlabel = labels[i] + " (MeV)", newFigure = False, style = "step", label = mc_label, color = "C6")
        Plots.Plot(energy_slices.pos_overflow, histograms_data[i], yerr = histograms_data_err[i], newFigure = False, style = "step", label = data_label, color = "k")
        Plots.plt.legend(loc = "upper left")
    Plots.plt.suptitle(sample_name)
    Plots.plt.tight_layout()
    book.Save()

    Plots.Plot(energy_slices.pos_overflow, scale * histograms_mc_reco["int_ex"], yerr = np.sqrt(scale * histograms_mc_reco["int_ex"]), xlabel = labels["int_ex"] + " (MeV)", style = "step", label = mc_label, color = "C6", title = sample_name)
    Plots.Plot(energy_slices.pos_overflow, histograms_data["int_ex"], yerr = histograms_data_err["int_ex"], newFigure = False, style = "step", label = f"{data_label}, background subtracted", color = "k")
    Plots.plt.legend(loc = "upper left")
    book.Save()

    return


def SelectionEfficiency(true_hists_selected, true_hists):
    return {k : cex_toy_parameters.Efficiency(true_hists_selected[k], true_hists[k]) for k in true_hists}


def EfficiencyErrStat(eff, err, val, val_eff, norm, true):
    eff_err = np.where(eff[0] == 0,
        np.sqrt(norm * true),
        val_eff * np.sqrt(cross_section.nandiv(err, val)**2) #+ cross_section.nandiv(eff[1], eff[0])**2
        )
    return np.nan_to_num(eff_err)


def EfficiencyErrSys(eff, err, val, val_eff, norm, true):
    eff_err = np.where(eff[0] == 0,
        0,
        val_eff * np.sqrt(cross_section.nandiv(err, val)**2) #+ cross_section.nandiv(eff[1], eff[0])**2
        )
    return np.nan_to_num(eff_err)


def ApplyEfficiency(energy_bins, efficiencies, unfolding_result, true, norm : float = 1, book : Plots.PlotBook = Plots.PlotBook.null):
    labels = {"init" : "$N_{init}$", "int" : "$N_{int}$", "int_ex" : "$N_{int, ex}$", "inc" : "$N_{inc}$"}
    hist_unfolded_efficiency_corrected = {}

    for k in unfolding_result:
        unfolded_eff = np.where(efficiencies[k][0] == 0, norm * true[k], cross_section.nandiv(unfolding_result[k]["unfolded"], efficiencies[k][0]))

        unfolded_eff_err_stat = EfficiencyErrStat(efficiencies[k], unfolding_result[k]["stat_err"], unfolding_result[k]["unfolded"], unfolded_eff, norm, true[k])
        unfolded_eff_err_sys = EfficiencyErrSys(efficiencies[k], unfolding_result[k]["sys_err"], unfolding_result[k]["unfolded"], unfolded_eff, norm, true[k])
        hist_unfolded_efficiency_corrected[k] = {"unfolded" : unfolded_eff, "stat_err" : unfolded_eff_err_stat, "sys_err" : unfolded_eff_err_sys}

    if book is not None:
        for _, k in Plots.IterMultiPlot(unfolding_result):
            Plots.Plot(energy_bins[::-1], hist_unfolded_efficiency_corrected[k]["unfolded"], yerr = hist_unfolded_efficiency_corrected[k]["stat_err"], style = "step", color = "C4", label = "unfolded, efficiency corrected", newFigure = False, xlabel = labels[k], ylabel  ="Counts")
            Plots.Plot(energy_bins[::-1], norm * true[k], style = "step", color = "C0", label = "true", newFigure = False)
        book.Save()
    return hist_unfolded_efficiency_corrected


def Unfolding(reco_hists : dict, reco_hists_err : dict, mc : cross_section.AnalysisInput, unfolding_args : dict, signal_process, norm, energy_slices, mc_cheat : cross_section.AnalysisInput = None, book : Plots.PlotBook = Plots.PlotBook.null):
    labels = {"init" : "$N_{init}$", "int" : "$N_{int}$", "int_ex" : "$N_{int, ex}$", "inc" : "$N_{inc}$"}

    true_hists_selected = mc.CreateHistograms(energy_slices, signal_process, False, ~mc.inclusive_process)

    if mc_cheat is not None:
        true_hists = mc_cheat.CreateHistograms(energy_slices, signal_process, False, ~mc_cheat.inclusive_process)
        efficiencies = SelectionEfficiency(true_hists_selected, true_hists)

        if book is not None:
            for _, (k, v) in Plots.IterMultiPlot(efficiencies.items()):
                Plots.Plot(energy_slices.pos_overflow, v[0], yerr = v[1], ylabel = "efficiency", xlabel = f"$N_{{{k}}}$", marker = "x", newFigure = False)
                Plots.plt.ylim(0, 1)
            book.Save()
    else:
        efficiencies = None


    if unfolding_args is None:
        print("using default options for unfolding")
        unfolding_args = {"ts_stop" : 0.01, "max_iter" : 100, "ts" : "ks", "method" : 1}

    if unfolding_args["method"] == 1:
        resp = cross_section.Unfold.CalculateResponseMatrices(mc, signal_process, energy_slices, book, None)
        priors = {k : v for k, v in true_hists_selected.items()}
    if unfolding_args["method"] == 2:
        resp = cross_section.Unfold.CalculateResponseMatrices(mc_cheat, signal_process, energy_slices, book, {k : v[0] for k, v in efficiencies.items()})
        priors = {k : v for k, v in true_hists.items()}


    unfolding_args["priors"] = priors
    unfolding_args["response_matrices"] = resp

    result = cross_section.Unfold.Unfold(reco_hists, reco_hists_err, verbose = True, **{k : v for k, v in unfolding_args.items() if k != "method"})

    n_incident_unfolded = cross_section.EnergySlice.NIncident(result["init"]["unfolded"], result["int"]["unfolded"])
    n_incident_unfolded_stat_err = np.sqrt(result["int"]["stat_err"]**2 + np.cumsum(result["init"]["stat_err"]**2 + result["int"]["stat_err"]**2))
    n_incident_unfolded_sys_err = np.sqrt(result["int"]["sys_err"]**2 + np.cumsum(result["init"]["sys_err"]**2 + result["int"]["sys_err"]**2))

    result["inc"] = {"unfolded" : n_incident_unfolded, "stat_err" : n_incident_unfolded_stat_err, "sys_err" : n_incident_unfolded_sys_err}

    if book is not None:
        for k in result:
            if k == "inc" : continue
            cross_section.Unfold.PlotUnfoldingResults(reco_hists[k], norm * true_hists_selected[k], result[k], energy_slices, labels[k], book)

        Plots.Plot(energy_slices.pos_bins[::-1], reco_hists["inc"], style = "step", label = "Data reco", color = "C6")
        Plots.Plot(energy_slices.pos_bins[::-1], norm * true_hists_selected["inc"], style = "step", label = "MC true", color = "C0", newFigure = False)
        Plots.Plot(energy_slices.pos_bins[::-1], result["inc"]["unfolded"], yerr = result["inc"]["stat_err"], style = "step", label = "Data unfolded", xlabel = "$N_{inc}$ (MeV)", color = "C4", newFigure = False)
        book.Save()

    if (unfolding_args["method"] == 1) and (efficiencies is not None):
        return ApplyEfficiency(energy_slices.pos_bins, efficiencies, result, true_hists, norm, book)
    else:
        return result


def XSUnfold(unfolded_result, energy_slices, sys : bool = False, stat = True):
    total_err = {}

    for r in unfolded_result:
        errs = []
        if sys is True:
            errs.append(unfolded_result[r]["sys_err"][1:]) # MC stat error from template used in fit and unfolding
        if stat is True:
            errs.append(unfolded_result[r]["stat_err"][1:]) # statistical uncertainties from fit and unfolding

        if len(errs) > 1:
            total_err[r] = cross_section.quadsum(errs, 0)
        elif len(errs) == 1:
            total_err[r] = errs[0]
        else:
            total_err[r] = None # statistical uncertianties from histograms only

    return cross_section.EnergySlice.CrossSection(
        unfolded_result["int_ex"]["unfolded"][1:],
        unfolded_result["int"]["unfolded"][1:],
        unfolded_result["inc"]["unfolded"][1:],
        cross_section.BetheBloch.meandEdX(energy_slices.pos_bins[1:], cross_section.Particle.from_pdgid(211)),
        energy_slices.width,
        total_err["int_ex"],
        total_err["int"],
        total_err["inc"]
    )

def LoadToy(file):
    if file.split(".")[-1] == "hdf5":
        toy = cross_section.Toy(file = file)
    elif file.split(".")[-1] == "json":
        toy = cross_section.Toy(df = cex_toy_generator.run(cross_section.LoadConfiguration(file)))
    else:
        raise Exception("toy file format not recognised")
    return cross_section.AnalysisInput.CreateAnalysisInputToy(toy)


def main(args):
    cross_section.SetPlotStyle(extend_colors = False, dark = True)
    args.out = args.out + "measurement/"

    samples = {}
    templates = {}
    if args.toy_template:
        print("loading toy template")
        templates["toy"] = LoadToy(args.toy_template)
        print("loading toy data")
        samples["toy"] = LoadToy(args.toy_data)
    if args.pdsp:
        print("loading Data and MC")
        if args.analysis_input:
            templates["pdsp"] = cross_section.AnalysisInput.FromFile(args.analysis_input["mc"])
            samples["pdsp"] = cross_section.AnalysisInput.FromFile(args.analysis_input["data"])
            templates["mc_cheated"] = cross_section.AnalysisInput.FromFile(args.analysis_input["mc_cheated"])
        else:
            templates["pdsp"] = cex_analysis_input.CreateAnalysisInput(cross_section.Data(args.mc_file, nTuple_type = args.ntuple_type, target_momentum = args.pmom), args, True)
            templates["mc_cheated"] = cex_analysis_input.CreateAnalysisInputMCTrueBeam(cross_section.Data(args.mc_file, nTuple_type = args.ntuple_type, target_momentum = args.pmom), args)
            samples["pdsp"] = cex_analysis_input.CreateAnalysisInput(cross_section.Data(args.data_file, nTuple_type = args.ntuple_type), args, False)

    label_map = {"toy" : "toy", "pdsp" : "ProtoDUNE SP"}

    if args.fit["mean_track_score"] == True:
        mean_track_score_bins = np.linspace(0, 1, 21, True) #TODO make configurable
    else:
        mean_track_score_bins = None
    xs = {}
    for k, v in samples.items():
        print(f"analysing {k}")

        outdir = args.out + f"{k}/"
        os.makedirs(outdir, exist_ok = True)

        if k == "toy":
            scale = len(samples[k].KE_init_reco) / len(templates[k].KE_init_reco)
        else:
            scale = args.norm

        if k == "toy":
            unfolding_args = None # for the toy use default unfolding as this will be more optimal
        else:
            unfolding_args = args.unfolding

        mc_cheat = None if k == "toy" else templates["mc_cheated"]

        #* should some pre-requisit plots be made?

        region_fit_result, fit_values = RegionFit(v, args.energy_slices, mean_track_score_bins, templates[k], return_fit_results = True, mc_stat_unc = args.fit["mc_stat_unc"])

        # scale = len(templates[k].KE_int_reco) / len(samples[k].KE_int_reco)
        indices = [f"$\mu_{{{i}}}$" for i in ["abs", "cex", "spip", "pip"]]
        print(f"{fit_values.bestfit=}")
        table = cross_section.pd.DataFrame({"fit value" : fit_values.bestfit[0:4] / scale, "uncertainty" : fit_values.uncertainty[0:4] / scale}, index = indices).T
        table.style.to_latex(outdir + "fit_results.tex")

        if args.all is True:
            process = {i : None for i in templates[k].exclusive_process}
        else:
            process = {args.signal_process : None}

        for p in process:
            with Plots.PlotBook(outdir + f"plots_{p}.pdf") as book:
                if k == "pdsp":
                    true_hists = mc_cheat.CreateHistograms(args.energy_slices, p, False, ~mc_cheat.inclusive_process)
                else:
                    true_hists = templates[k].CreateHistograms(args.energy_slices, p, False, ~templates[k].inclusive_process)

                xs_true = cross_section.EnergySlice.CrossSection(true_hists["int_ex"][1:], true_hists["int"][1:], true_hists["inc"][1:], cross_section.BetheBloch.meandEdX(args.energy_slices.pos_bins[1:], cross_section.Particle.from_pdgid(211)), args.energy_slices.width)

                _, histograms_reco_obs, histograms_reco_obs_err = BackgroundSubtraction(v, p, args.energy_slices, region_fit_result, book) #? make separate background subtraction function?
                
                if k == "toy":
                    data_label = "toy data"
                    mc_label = "toy template"
                elif k == "pdsp":
                    data_label = "Data"
                    mc_label = "MC cheated (scaled to Data)"

                PlotDataBkgSub(samples[k], templates[k], region_fit_result, p, args.energy_slices, scale, None, data_label, mc_label, book)

                unfolding_result = Unfolding(histograms_reco_obs, histograms_reco_obs_err, templates[k], unfolding_args, p, scale, args.energy_slices, mc_cheat, book)

                process[p] = XSUnfold(unfolding_result, args.energy_slices)

                cross_section.PlotXSComparison({f"{label_map[k]} Data reco" : process[p], f"{label_map[k]} MC truth" : xs_true}, args.energy_slices, p, {f"{label_map[k]} Data reco" : "C0", f"{label_map[k]} MC truth" : "C1"}, simulation_label = "Geant4 v10.6")
                book.Save()
            Plots.plt.close("all")
        xs[k] = process
    with Plots.PlotBook(args.out + "results.pdf") as book:
        colours = {f"{label_map[k]} Data" : f"C{i}" for i, k in enumerate(xs.keys())}
        for p in list(xs.values())[0]:
            data = {f"{label_map[k]} Data" : xs[k][p] for k in xs.keys()}
            cross_section.PlotXSComparison(data, args.energy_slices, p, colours, simulation_label = "Geant4 v10.6")
            book.Save()
    return


if __name__ == "__main__":
    parser = cross_section.argparse.ArgumentParser(description = "Computes the upstream energy loss for beam particles after beam particle selection, then writes the fitted parameters to file.", formatter_class = cross_section.argparse.RawDescriptionHelpFormatter)

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