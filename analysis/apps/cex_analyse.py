#!/usr/bin/env python3
"""
Created on: 13/11/2023 21:54

Author: Shyam Bhuller

Description: Runs main CEX analysis (post event selection)
"""
import os

import numpy as np

from rich import print
from scipy.interpolate import interp1d

from apps import cex_toy_generator
from python.analysis import cross_section, SelectionTools, Plots


def BeamPionSelection(events : cross_section.Master.Data, args : cross_section.argparse.Namespace, is_mc : bool) -> cross_section.Master.Data:
    events_copy = events.Filter(returnCopy = True)
    if is_mc:
        selection_args = "mc_arguments"
        sample = "mc"
    else:
        selection_args = "data_arguments"
        sample = "data"
    if "selection_masks" in args:
        for m in args.selection_masks[sample]["beam"].values():
            events_copy.Filter([m], [m])
    else:
        for s in args.beam_selection["selections"]:
            mask = args.beam_selection["selections"][s](events_copy, **args.beam_selection[selection_args][s])
            events_copy.Filter([mask], [mask])
            print(events_copy.cutTable.get_table())

    events_copy.Filter([args.selection_masks[sample]['null_pfo']['ValidPFOSelection']]) # apply PFO preselection here
    return events_copy

@cross_section.Master.timer
def RegionSelection(events : cross_section.Master.Data, args : cross_section.argparse.Namespace, is_mc : bool) -> dict[np.array]:
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

        for m in args.selection_masks["mc"]["beam"].values():
            n_pi_true = n_pi_true[m]
            n_pi0_true = n_pi0_true[m]
            is_pip = is_pip[m]
        true_regions = cross_section.EventSelection.create_regions_new(n_pi0_true, n_pi_true)
        for k in true_regions:
            true_regions[k] = true_regions[k] & (is_pip)
        for k in reco_regions:
            reco_regions[k] = reco_regions[k] & (is_pip)
        return reco_regions, true_regions
    else:
        return reco_regions


def CreateInitParams(model : cross_section.pyhf.Model, analysis_input : cross_section.AnalysisInput, energy_slices : cross_section.Slices, mean_track_score_bins : np.array):
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


def RegionFit(fit_input : cross_section.AnalysisInput, energy_slice : cross_section.Slices, mean_track_score_bins : np.array, template_input : cross_section.AnalysisInput, suggest_init : bool = False, template_weights : np.array = None) -> cross_section.cabinetry.model_utils.ModelPrediction:

    model = cross_section.RegionFit.CreateModel(template_input, energy_slice, mean_track_score_bins, False, template_weights)

    observed = cross_section.RegionFit.GenerateObservations(fit_input, energy_slice, mean_track_score_bins, model)

    if suggest_init is True:
        init_params = CreateInitParams(model, fit_input, energy_slice, mean_track_score_bins)
    else:
        init_params = None

    result = cross_section.RegionFit.Fit(observed, model, init_params, verbose = True)
    return cross_section.cabinetry.model_utils.prediction(model, fit_results = result)


def BackgroundSubtraction(data : cross_section.AnalysisInput, process : str, energy_slice : cross_section.Slices, postfit_pred : cross_section.cabinetry.model_utils.ModelPrediction = None, book : Plots.PlotBook = None) -> tuple[np.array]:
    histograms_true_obs = cross_section.Unfold.CreateHistograms(data, energy_slice, process, False, False)
    histograms_reco_obs = cross_section.Unfold.CreateHistograms(data, energy_slice, process, True, False)
    histograms_reco_obs_err = {k : np.sqrt(v) for k, v in histograms_reco_obs.items()}
    
    if postfit_pred is not None:
        print("using KE_int,ex from region fit")
        print(f"signal: {process}")
        n = cross_section.RegionFit.CreateObservedInputData(data, energy_slice, None)
        N = sum(n)

        KE_int_prediction = cross_section.RegionFit.SliceModelPrediction(postfit_pred, slice(-1), "KE_int_postfit") # exclude the channel which is the mean track score

        L = np.sum(KE_int_prediction.model_yields, 0)

        L_err = KE_int_prediction.total_stdev_model_bins[:, :-1] # last entry in the array is the total error for the whole channel (but we want the total error in each process)
        L_err = np.sqrt(np.sum(L_err **2, 0)) # quadrature sum across all bins

        labels = list(data.exclusive_process.keys()) #! make property of AnalysisInput dataclass
        L_var_bkg = sum(L_err[process != np.array(labels)]**2)
        L_bkg = sum(L[process != np.array(labels)])


        KE_int_fit = N - L_bkg
        KE_int_fit_err = np.sqrt(N + L_var_bkg)


        actual = {l : data.NInteract(energy_slice, data.exclusive_process[l]) for l in labels}
        actual_sig = actual[process]
        actual_bkg = sum(np.array(list(actual.values()))[process != np.array(labels)])

        energy_bins = np.sort(np.insert(energy_slice.pos, 0, energy_slice.max_pos + energy_slice.width))
        cross_section.RegionFit.PlotPrefitPostFit(actual_sig, np.sqrt(actual_sig), KE_int_fit, KE_int_fit_err, energy_bins)
        if book is not None: book.Save()
        cross_section.RegionFit.PlotPrefitPostFit(actual_bkg, np.sqrt(actual_bkg), L_bkg, np.sqrt(L_var_bkg), energy_bins)
        if book is not None: book.Save()

        histograms_reco_obs["int_ex"] = np.where(KE_int_fit < 0, 0, KE_int_fit)
        histograms_reco_obs_err["int_ex"] = KE_int_fit_err

    return histograms_true_obs, histograms_reco_obs, histograms_reco_obs_err



def Unfolding(hist_reco : dict[np.array], hist_reco_err : dict[np.array], energy_slices : cross_section.Slices, toy_template : cross_section.Toy, fit_results : cross_section.cabinetry.model_utils.ModelPrediction, signal_process : str, book : Plots.PlotBook = None):
    response_matrices = cross_section.Unfold.CalculateResponseMatrices(toy_template, signal_process, energy_slices, book)
    return cross_section.Unfold.Unfold(hist_reco, hist_reco_err, response_matrices, ts_stop = 1E-2, ts = "bf", max_iter = 100)


def CreateAnalysisInput(sample : cross_section.Toy | cross_section.Master.Data, args : cross_section.argparse.Namespace, is_mc : bool, beam_selection : bool = False) -> cross_section.AnalysisInput:
    if type(sample) == cross_section.Toy:
        ai = cross_section.AnalysisInput.CreateAnalysisInputToy(sample, beam_selection)
    elif type(sample) == cross_section.Master.Data:
        sample_selected = BeamPionSelection(sample, args, is_mc)
        if is_mc:
            reco_regions, true_regions = RegionSelection(sample, args, True)
        else:
            reco_regions = RegionSelection(sample, args, False)
            true_regions = None
        ai = cross_section.AnalysisInput.CreateAnalysisInputNtuple(sample_selected, args.upstream_loss_correction_params["value"], reco_regions, true_regions)
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
        samples["MC"] = cross_section.Master.Data(args.mc_file, nTuple_type = args.ntuple_type)
    elif args.data:
        print(f"analyse Data: {args.data_file}")
        # samples["Data"] = cross_section.Master.Data(args.data_file, nTuple_type = args.ntuple_type) #! not yet
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
            
            unfolding_result = Unfolding(histograms_reco_obs, histograms_reco_obs_err, energy_slices, args.toy_template, region_fit_result, args.signal_process, book)

            labels = {"init" : "$N_{init}$", "int" : "$N_{int}$", "int_ex" : "$N_{int, ex}$"}
            for i in unfolding_result:
                cross_section.Unfold.PlotUnfoldingResults(histograms_reco_obs[i], histograms_true_obs[i], unfolding_result[i], energy_bins, labels[i], book)

            #* integrate into unfolding results
            n_incident_unfolded = cross_section.EnergySlice.NIncident(unfolding_result["init"]["unfolded"], unfolding_result["int"]["unfolded"])
            n_incident_unfolded_err = np.sqrt(unfolding_result["int"]["stat_err"]**2 + np.cumsum(unfolding_result["init"]["stat_err"]**2 + unfolding_result["int"]["stat_err"]**2))

            Plots.Plot(energy_bins[::-1], histograms_reco_obs["inc"], style = "step", label = "reco", color = "C6")
            Plots.Plot(energy_bins[::-1], histograms_true_obs["inc"], style = "step", label = "true", color = "C0", newFigure = False)
            Plots.Plot(energy_bins[::-1], n_incident_unfolded, yerr = n_incident_unfolded_err, style = "step", label = "unfolded", xlabel = "$N_{inc}$ (MeV)", color = "C4", newFigure = False)
            book.Save()

            slice_dEdX = cross_section.EnergySlice.Slice_dEdX(energy_slices, cross_section.Particle.from_pdgid(211))
            xs[k] = cross_section.EnergySlice.CrossSection(unfolding_result["int_ex"]["unfolded"][1:], unfolding_result["int"]["unfolded"][1:], n_incident_unfolded[1:], slice_dEdX, energy_slices.width, unfolding_result["int_ex"]["stat_err"][1:], unfolding_result["int"]["stat_err"][1:], n_incident_unfolded_err[1:])

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