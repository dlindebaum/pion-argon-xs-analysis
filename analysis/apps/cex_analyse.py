#!/usr/bin/env python3
"""
Created on: 13/11/2023 21:54

Author: Shyam Bhuller

Description: Runs cross section measurement
"""
import os

import numpy as np

from rich import print

from apps import cex_toy_generator
from python.analysis import cross_section, SelectionTools, Plots


def BeamPionSelection(events : cross_section.Data, args : cross_section.argparse.Namespace, is_mc : bool) -> cross_section.Data:
    """ Apply beam pion selection to ntuples.

    Args:
        events (cross_section.Data): analysis ntuple
        args (cross_section.argparse.Namespace): analysis configuration
        is_mc (bool): is the ntuple mc or data?

    Returns:
        cross_section.Data: selected events.
    """
    events_copy = events.Filter(returnCopy = True)
    if is_mc:
        selection_args = "mc_arguments"
        sample = "mc"
    else:
        selection_args = "data_arguments"
        sample = "data"

    if "selection_masks" in args:
        mask = SelectionTools.CombineMasks(args.selection_masks[sample]["beam"])
        events_copy.Filter([mask], [mask])
    else:
        for s in args.beam_selection["selections"]:
            mask = args.beam_selection["selections"][s](events_copy, **args.beam_selection[selection_args][s])
            events_copy.Filter([mask], [mask])
            print(events_copy.cutTable.get_table())

    if hasattr(args, "valid_pfo_selection"):
        if args.valid_pfo_selection is True:
            events_copy.Filter([args.selection_masks[sample]['null_pfo']['ValidPFOSelection']]) # apply PFO preselection here
    return events_copy

@cross_section.timer
def RegionSelection(events : cross_section.Data, args : cross_section.argparse.Namespace, is_mc : bool) -> dict[np.array]:
    """ Get reco and true regions (if possible) for ntuple.

    Args:
        events (Master.Data): events after beam pion selection
        args (argparse.Namespace): application arguements
        is_mc (bool): if ntuple is MC or Data.

    Returns:
        tuple[dict, dict]: regions
    """
    if is_mc:
        key = "mc"
    else:
        key = "data"

    n_pi =  SelectionTools.GetPFOCounts(args.selection_masks[key]["pi"])
    n_pi0 = SelectionTools.GetPFOCounts(args.selection_masks[key]["pi0"])
    reco_regions = cross_section.EventSelection.create_regions_new(n_pi0, n_pi)

    if is_mc:
        events_copy = events.Filter(returnCopy = True)
        
        n_pi_true = events_copy.trueParticles.nPiMinus + events_copy.trueParticles.nPiPlus
        n_pi0_true = events_copy.trueParticles.nPi0

        is_pip = events_copy.trueParticles.pdg[:, 0] == 211

        mask = None
        for m in args.selection_masks["mc"]["beam"].values():
            if mask is None:
                mask = m
            else:
                mask = mask & m
        n_pi_true = n_pi_true[mask]
        n_pi0_true = n_pi0_true[mask]
        is_pip = is_pip[mask]
        true_regions = cross_section.EventSelection.create_regions_new(n_pi0_true, n_pi_true)
        for k in true_regions:
            true_regions[k] = true_regions[k] & (is_pip)
        for k in reco_regions:
            reco_regions[k] = reco_regions[k] & (is_pip)
        return reco_regions, true_regions
    else:
        return reco_regions


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


def RegionFit(fit_input : cross_section.AnalysisInput, energy_slice : cross_section.Slices, mean_track_score_bins : np.array, template_input : cross_section.AnalysisInput | cross_section.pyhf.Model, suggest_init : bool = False, template_weights : np.array = None, return_fit_results : bool = False) -> cross_section.cabinetry.model_utils.ModelPrediction | cross_section.FitResults:
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
        model = cross_section.RegionFit.CreateModel(template_input, energy_slice, mean_track_score_bins, False, template_weights, False, True)
    else:
        model = template_input

    observed = cross_section.RegionFit.GenerateObservations(fit_input, energy_slice, mean_track_score_bins, model)

    if suggest_init is True:
        init_params = CreateInitParams(model, fit_input, energy_slice, mean_track_score_bins)
    else:
        init_params = None

    result = cross_section.RegionFit.Fit(observed, model, init_params, [[0, np.inf]]*model.config.npars, verbose = False)
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
        print("using KE_int,ex from region fit")
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


        if data.exclusive_process is not None:
            actual = {l : data.NInteract(energy_slice, data.exclusive_process[l], weights = data.weights) for l in labels}
            actual_sig = actual[process]
            actual_bkg = sum(np.array(list(actual.values()))[process != np.array(labels)])

            energy_bins = np.sort(np.insert(energy_slice.pos, 0, energy_slice.max_pos + energy_slice.width))

            cross_section.RegionFit.PlotPrefitPostFit(actual_sig, np.sqrt(actual_sig), KE_int_fit, KE_int_fit_err, energy_bins)
            book.Save()
            cross_section.RegionFit.PlotPrefitPostFit(actual_bkg, np.sqrt(actual_bkg), L_bkg, np.sqrt(L_var_bkg), energy_bins)
            book.Save()

        histograms_reco_obs["int_ex"] = np.where(KE_int_fit < 0, 0, KE_int_fit)
        histograms_reco_obs_err["int_ex"] = KE_int_fit_err

    return histograms_true_obs, histograms_reco_obs, histograms_reco_obs_err


def Unfolding(hist_reco : dict[np.array], hist_reco_err : dict[np.array], energy_slices : cross_section.Slices, template : cross_section.AnalysisInput, signal_process : str, book : Plots.PlotBook = None, unfolding_args : dict = None) -> dict[dict]:
    """ Unfold post fit reco histograms

    Args:
        hist_reco (dict[np.array]): reco hitograms
        hist_reco_err (dict[np.array]): error in reco histograms
        energy_slices (cross_section.Slices): energy slices
        template (cross_section.Toy): template to create response matrices
        signal_process (str): signal process
        book (Plots.PlotBook, optional): plot book. Defaults to None.

    Returns:
        dict[dict]: unolfing results
    """

    if "efficiencies" in unfolding_args:
        eff = unfolding_args["efficiencies"]
    else:
        eff = None

    response_matrices = cross_section.Unfold.CalculateResponseMatrices(template, signal_process, energy_slices, book, efficiencies = eff)

    result = cross_section.Unfold.Unfold(hist_reco, hist_reco_err, response_matrices, **unfolding_args)
    n_incident_unfolded = cross_section.EnergySlice.NIncident(result["init"]["unfolded"], result["int"]["unfolded"])
    n_incident_unfolded_err = np.sqrt(result["int"]["stat_err"]**2 + np.cumsum(result["init"]["stat_err"]**2 + result["int"]["stat_err"]**2))

    result["inc"] = {"unfolded" : n_incident_unfolded, "stat_err" : n_incident_unfolded_err}

    return result


def XSUnfold(unfolded_result, energy_slices, energy_bins):
    return cross_section.EnergySlice.CrossSection(
        unfolded_result["int_ex"]["unfolded"][1:],
        unfolded_result["int"]["unfolded"][1:],
        unfolded_result["inc"]["unfolded"][1:],
        cross_section.BetheBloch.meandEdX(energy_bins[1:], cross_section.Particle.from_pdgid(211)),
        energy_slices.width,
        np.sqrt(unfolded_result["int_ex"]["stat_err"][1:]**2 + unfolded_result["int_ex"]["stat_err"][1:]**2),
        np.sqrt(unfolded_result["int"]["stat_err"][1:]**2 + unfolded_result["int"]["stat_err"][1:]**2),
        np.sqrt(unfolded_result["inc"]["stat_err"][1:]**2 + unfolded_result["inc"]["stat_err"][1:]**2)
    )


def CreateAnalysisInput(sample : cross_section.Toy | cross_section.Data, args : cross_section.argparse.Namespace, is_mc : bool) -> cross_section.AnalysisInput:
    """ Create analysis input from either toy or ntuple sample

    Args:
        sample (cross_section.Toy | cross_section.Data): sample
        args (cross_section.argparse.Namespace): analysis configurations
        is_mc (bool): is the sample mc?

    Returns:
        cross_section.AnalysisInput: analysis input.
    """
    if type(sample) == cross_section.Toy:
        ai = cross_section.AnalysisInput.CreateAnalysisInputToy(sample)
    elif type(sample) == cross_section.Data:
        sample_selected = BeamPionSelection(sample, args, is_mc)
        if is_mc:
            reco_regions, true_regions = RegionSelection(sample, args, True)
            reweight_params = args.beam_reweight_params
        else:
            reco_regions = RegionSelection(sample, args, False)
            true_regions = None
            reweight_params = None
        ai = cross_section.AnalysisInput.CreateAnalysisInputNtuple(sample_selected, args.upstream_loss_correction_params["value"], reco_regions, true_regions, reweight_params)
    else:
        raise Exception(f"object type {type(sample)} not a valid sample")
    return ai


def main(args):
    cross_section.SetPlotStyle(extend_colors = True)
    samples = {}
    if args.toy:
        print(f"analyse toy: {args.toy}")
        if args.toy.split(".")[-1] == "hdf5":
            samples["Toy"] = cross_section.Toy(file = args.toy)
        elif args.toy.split(".")[-1] == "json":
            samples["Toy"] = cross_section.Toy(df = cex_toy_generator.main(cex_toy_generator.ResolveConfig(cross_section.LoadConfiguration(args.toy))))
        else:
            raise Exception("toy file format not recognised")
    elif args.mc:
        print(f"analyse MC: {args.mc_file}")
        samples["MC"] = cross_section.Data(args.mc_file, nTuple_type = args.ntuple_type)
    elif args.data:
        print(f"analyse Data: {args.data_file}")
        # samples["Data"] = cross_section.Data(args.data_file, nTuple_type = args.ntuple_type) #! not yet
    else:
        raise Exception("--toy, --mc and or --data must be specified")

    energy_slices = cross_section.Slices(50, 0, 1050, True) #TODO make configurable
    mean_track_score_bins = np.linspace(0, 1, 21, True) #TODO make configurable
    energy_bins = np.sort(np.insert(energy_slices.pos, 0, energy_slices.max_pos + energy_slices.width)) # for plotting
    xs = {}
    for k, v in samples.items():
        print(f"analysing {k}")

        outdir = args.out + f"{k}/"
        os.makedirs(outdir, exist_ok = True)

        is_mc = False if k == "Data" else True
        analysis_input = CreateAnalysisInput(v, args, is_mc) # is mc not required for toy
        template_input = cross_section.AnalysisInput.CreateAnalysisInputToy(args.toy_template) #! need to consolidate option for different template types

        with Plots.PlotBook(outdir + "plots.pdf") as book:
            region_fit_result = RegionFit(analysis_input, energy_slices, mean_track_score_bins, template_input)

            histograms_true_obs, histograms_reco_obs, histograms_reco_obs_err = BackgroundSubtraction(analysis_input, args.signal_process, energy_slices, region_fit_result, book) #? make separate background subtraction function?

            unfolding_args = {"efficiencies" : None, "priors" : histograms_true_obs, "regularizers" : None, "ts_stop" : 0.01, "max_iter" : 3, "ts" : "ks"} # for toy only, make configurable

            unfolding_result = Unfolding(histograms_reco_obs, histograms_reco_obs_err, energy_slices, template_input, args.signal_process, book, unfolding_args = unfolding_args)

            labels = {"init" : "$N_{init}$", "int" : "$N_{int}$", "int_ex" : "$N_{int, ex}$"}
            for i in unfolding_result:
                if i == "inc": continue
                cross_section.Unfold.PlotUnfoldingResults(histograms_reco_obs[i], histograms_true_obs[i], unfolding_result[i], energy_bins, labels[i], book)
            Plots.Plot(energy_bins[::-1], histograms_reco_obs["inc"], style = "step", label = "reco", color = "C6")
            Plots.Plot(energy_bins[::-1], histograms_true_obs["inc"], style = "step", label = "true", color = "C0", newFigure = False)
            Plots.Plot(energy_bins[::-1], unfolding_result["inc"]["unfolded"], yerr = unfolding_result["inc"]["stat_err"], style = "step", label = "unfolded", xlabel = "$N_{inc}$ (MeV)", color = "C4", newFigure = False)
            book.Save()

            xs[k] = XSUnfold(unfolding_result, energy_slices, energy_bins)

    with Plots.PlotBook(args.out + "results.pdf") as book:
        cross_section.PlotXSComparison(xs, energy_slices, args.signal_process, {list(xs.keys())[0] : "C6"})
        book.Save()
    return


if __name__ == "__main__":
    parser = cross_section.argparse.ArgumentParser(description = "Computes the upstream energy loss for beam particles after beam particle selection, then writes the fitted parameters to file.", formatter_class = cross_section.argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Config(parser, required = True)
    cross_section.ApplicationArguments.Output(parser)

    parser.add_argument("--toy", dest = "toy", type = str, help = "use toy, proivde a hdf5 toy file or toy config")
    parser.add_argument("--mc", dest = "mc", action = "store_true", help = "use mc")
    parser.add_argument("--data", dest = "data", action = "store_true", help = "use data")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())
    print("parsed config, loading toy template")
    args.toy_template = cross_section.Toy(file = args.toy_template)
    args.out = args.out + "analysis/"
    print(vars(args))
    main(args)