from __future__ import annotations

import numpy as np
import pyhf
import cabinetry

from cabinetry.fit.results_containers import FitResults
from python.analysis import Plots
from python.analysis.Utils import quadsum

from python.analysis.cross_section import AnalysisInput, Slices


def Model(n_channels: int, KE_int_templates: np.ndarray, mean_track_score_templates: np.ndarray = None, mc_stat_unc: bool = False) -> pyhf.Model:
    """ Create a pyhf model from per-channel templates.

        Args:
            n_channels (int): Number of region channels to create.
            KE_int_templates (np.ndarray): Interaction-energy templates, indexed by channel and sample.
            mean_track_score_templates (np.ndarray): Optional mean-track-score templates.
            mc_stat_unc (bool): Whether to add statistical-error modifiers to samples.

        Returns:
            pyhf.Model: Model containing the region channels and optional score channel.
    """
    def channel(channel_name: str, samples: np.ndarray, mc_stat_unc: bool):
        """ Build a pyhf channel specification from its sample templates.

            Args:
                channel_name (str): Name assigned to the channel.
                samples (np.ndarray): Per-sample bin counts for the channel.
                mc_stat_unc (bool): Whether to attach statistical-error modifiers.

            Returns:
                dict: Pyhf channel specification.
        """
        ch = {
            "name": channel_name,
            "samples": [
                {
                    "name": f"sample_{i}",
                    "data": s.tolist(),
                    "modifiers": [
                        {"name": f"mu_{i}", "type": "normfactor", "data": None},
                    ],
                }
                for i, s in enumerate(samples)
            ],
        }
        if mc_stat_unc is True:
            for i in range(len(samples)):
                ch["samples"][i]["modifiers"].append({
                    "name": f"{channel_name}_stat_err",
                    "type": "staterror",
                    "data": np.sqrt(np.sum(samples, 0)).astype(int).tolist(),
                })
        return ch

    spec = {"channels": [channel(f"channel_{n}", KE_int_templates[n], mc_stat_unc) for n in range(n_channels)]}

    if mean_track_score_templates is not None:
        spec["channels"] += [channel("mean_track_score", mean_track_score_templates, mc_stat_unc)]

    model = pyhf.Model(spec, poi_name="mu_0")
    return model


def PrintModelSpecs(model: pyhf.Model):
    """ Print the main configuration details of a pyhf model.

        Args:
            model (pyhf.Model): Model whose channels, samples, modifiers, and parameters are printed.
    """
    print(f"  channels: {model.config.channels}")
    print(f"     nbins: {model.config.channel_nbins}")
    print(f"   samples: {model.config.samples}")
    print(f" modifiers: {model.config.modifiers}")
    print(f"parameters: {model.config.parameters}")
    print(f"  nauxdata: {model.config.nauxdata}")
    print(f"   auxdata: {model.config.auxdata}")


def GenerateObservations(fit_input: AnalysisInput, energy_slices: Slices, mean_track_score_bins: np.ndarray, model: pyhf.Model, verbose: bool = True, single_bin: bool = False) -> np.ndarray:
    """ Create the observation vector expected by the fit model.

        Args:
            fit_input (AnalysisInput): Input containing observed event data.
            energy_slices (Slices): Energy slices used to form interaction counts.
            mean_track_score_bins (np.ndarray): Bin edges for the optional score distribution.
            model (pyhf.Model): Model whose auxiliary data is appended to the observations.
            verbose (bool): Whether to print model initialization and likelihood details.
            single_bin (bool): Whether to combine each interaction distribution into one bin.

        Returns:
            np.ndarray: Concatenated observation counts and model auxiliary data.
    """
    data = CreateObservedInputData(fit_input, energy_slices, mean_track_score_bins, single_bin)
    if verbose is True:
        print(f"{model.config.suggested_init()=}")
    observations = np.concatenate(data + [model.config.auxdata])
    if verbose is True:
        print(f"{model.logpdf(pars=model.config.suggested_init(), data=observations)=}")
    return observations


def Fit(observations, model: pyhf.Model, init_params: list[float] = None, par_bounds: list[tuple] = None, verbose: bool = True, tolerance: float = 1E-2, fix_pars: list[bool] = None) -> FitResults:
    """ Fit the region model to the supplied observations.

        Args:
            observations (np.ndarray): Observed counts and auxiliary data passed to the fit.
            model (pyhf.Model): Model to fit.
            init_params (list[float]): Optional initial parameter values.
            par_bounds (list[tuple]): Optional lower and upper bounds for parameters.
            verbose (bool): Whether to print fit inputs and results.
            tolerance (float): Convergence tolerance passed to cabinetry.
            fix_pars (list[bool]): Optional flags indicating parameters to fix.

        Returns:
            FitResults: Fit result, including best-fit parameters and uncertainties.
    """
    pyhf.set_backend(backend="numpy", custom_optimizer="minuit")
    if verbose is True:
        print(f"{init_params=}")
    result = cabinetry.fit.fit(model, observations, init_pars=init_params, custom_fit=True, tolerance=tolerance, par_bounds=par_bounds, fix_pars=fix_pars)

    poi_ind = [model.config.par_slice(i).start for i in model.config.par_names if "mu" in i]
    if verbose is True:
        print(f"{poi_ind=}")
    parameter = [i for i in model.config.par_names if "mu" in i]
    bestfit = result.bestfit[poi_ind]
    uncertainty = result.uncertainty[poi_ind]

    if verbose is True:
        print(f"{parameter=}")
        print(f"{bestfit=}")
        print(f"{uncertainty=}")
        print(f"{result=}")
    return result


def GetPredictedCorrelationMatrix(model: pyhf.Model, mu: np.ndarray) -> np.ndarray:
    """ Compute integer sample yields by channel after applying normalisations.

        Args:
            model (pyhf.Model): Model whose channel templates provide the nominal yields.
            mu (np.ndarray): Sample normalization factors applied to each channel.

        Returns:
            np.ndarray: Integer yield matrix with samples along rows and channels along columns.
    """
    counts_matrix = []
    for channel in model.spec["channels"]:
        counts = []
        for sample in channel["samples"]:
            counts.append(sum(sample["data"]))
        counts_matrix.append(counts * mu)
    counts_matrix = np.array(counts_matrix).T
    counts_matrix = np.array(counts_matrix, dtype=int)
    return counts_matrix


def CreateKEIntTemplates(analysis_input: AnalysisInput, energy_slices: Slices, single_bin: bool = False, pad: bool = False, reco: bool = True) -> list[np.ndarray]:
    """ Create interaction-energy templates for the model samples.

        Args:
            analysis_input (AnalysisInput): Analysis input providing process and region masks.
            energy_slices (Slices): Energy slices used to count interactions.
            single_bin (bool): Whether to sum each template into one bin.
            pad (bool): Whether to add a small offset to each bin.
            reco (bool): Whether to use reconstructed rather than true quantities.

        Returns:
            list[np.ndarray]: Templates organized by region and exclusive process.
    """
    model_input_data = []
    for c in analysis_input.regions:
        tmp = []
        for s in analysis_input.exclusive_process:
            n_int = analysis_input.n_interact(energy_slices, analysis_input.exclusive_process[s], analysis_input.regions[c], reco, analysis_input.weights) + 1E-10 * int(pad)
            if single_bin:
                tmp.append(np.array([sum(n_int)]))
            else:
                tmp.append(n_int)
        model_input_data.append(tmp)
    return model_input_data


def Createmean_track_scoreTemplates(analysis_input: AnalysisInput, bins: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """ Create mean-track-score histograms for each exclusive process.

        Args:
            analysis_input (AnalysisInput): Input providing score values and process masks.
            bins (np.ndarray): Histogram bin edges.
            weights (np.ndarray): Optional event weights applied to each process.

        Returns:
            np.ndarray: Histogram counts, with one row per exclusive process.
    """
    templates = []
    for t in analysis_input.exclusive_process:
        mask = analysis_input.exclusive_process[t]
        templates.append(np.histogram(analysis_input.mean_track_score[mask], bins, weights=weights[mask] if weights is not None else weights)[0])
    return np.array(templates)


def CreateModel(template: AnalysisInput, energy_slice: Slices, mean_track_score_bins: np.ndarray, return_templates: bool = False, weights: np.ndarray = None, mc_stat_unc: bool = True, pad: bool = True, single_bin: bool = False) -> pyhf.Model:
    """ Create a fit model and its templates from an analysis input.

        Args:
            template (AnalysisInput): Input used to build the model templates.
            energy_slice (Slices): Energy slices used for the interaction templates.
            mean_track_score_bins (np.ndarray): Optional score histogram bin edges.
            return_templates (bool): Whether to return templates alongside the model.
            weights (np.ndarray): Optional weights for score histograms.
            mc_stat_unc (bool): Whether to include Monte Carlo statistical uncertainties.
            pad (bool): Whether to add a small offset to interaction-template bins.
            single_bin (bool): Whether to combine each interaction template into one bin.

        Returns:
            pyhf.Model: Model alone, or the model and both template collections when requested.
    """
    templates_energy = CreateKEIntTemplates(template, energy_slice, single_bin, pad)
    if mean_track_score_bins is not None:
        templates_mean_track_score = Createmean_track_scoreTemplates(template, mean_track_score_bins, weights)
    else:
        templates_mean_track_score = None
    model = Model(len(template.regions), templates_energy, templates_mean_track_score, mc_stat_unc=mc_stat_unc)
    PrintModelSpecs(model)
    if return_templates is True:
        return model, templates_energy, templates_mean_track_score
    return model


def CreateObservedInputData(fit_input: AnalysisInput, slices: Slices, mean_track_score_bins: np.ndarray = None, single_bin: bool = False) -> np.ndarray:
    """ Create observed interaction and optional score distributions.

        Args:
            fit_input (AnalysisInput): Input containing observed data and region masks.
            slices (Slices): Energy slices used to form interaction counts.
            mean_track_score_bins (np.ndarray): Optional score histogram bin edges.
            single_bin (bool): Whether to sum interaction counts into one bin per region.

        Returns:
            np.ndarray: Observed distributions in region order, optionally followed by score counts.
    """
    observed_binned = []
    if fit_input.inclusive_process is None:
        mask = np.ones_like(fit_input.KE_int_reco, dtype=bool)
    else:
        mask = fit_input.inclusive_process
    for v in fit_input.regions.values():
        n_int = fit_input.n_interact(slices, v & mask, reco=True)
        if single_bin:
            n_int = [sum(n_int)]
        observed_binned.append(n_int)
    if mean_track_score_bins is not None:
        observed_binned.append(np.histogram(fit_input.mean_track_score[fit_input.inclusive_process], mean_track_score_bins)[0])
    return observed_binned


def SliceModelPrediction(prediction: cabinetry.model_utils.ModelPrediction, slice_: slice, label: str) -> cabinetry.model_utils.ModelPrediction:
    """ Create a model prediction containing a selected slice of bins.

        Args:
            prediction (cabinetry.model_utils.ModelPrediction): Prediction to slice.
            slice_ (slice): Bin selection applied to the prediction arrays.
            label (str): Label assigned to the returned prediction.

        Returns:
            cabinetry.model_utils.ModelPrediction: Prediction containing the selected yield and uncertainty bins.
    """
    return cabinetry.model_utils.ModelPrediction(prediction.model, np.array(prediction.model_yields[slice_]), np.array(prediction.total_stdev_model_bins[slice_]), np.array(prediction.total_stdev_model_channels[slice_]), label)


def PlotPrefitPostFit(prefit : np.ndarray, prefit_err: np.ndarray, postfit: np.ndarray, postfit_err: np.ndarray, energy_bins: np.ndarray, xlabel="$KE_{int}$ (MeV)"):
    """ Plot pre-fit and post-fit yields with their ratio.

        Args:
            prefit (np.ndarray): Pre-fit yields.
            prefit_err (np.ndarray): Uncertainties on the pre-fit yields.
            postfit (np.ndarray): Post-fit yields.
            postfit_err (np.ndarray): Uncertainties on the post-fit yields.
            energy_bins (np.ndarray): Energy bin edges.
            xlabel (str): Label for the x-axis.
    """
    with Plots.ratio_plot(energy_bins[::-1], postfit, prefit, postfit_err, prefit_err, xlabel, "fit/ true") as ratio_plot:
        Plots.plot(ratio_plot.x, ratio_plot.y2, yerr=ratio_plot.y2_err, color="C0", label="true", style="step", new_figure=False)
        Plots.plot(ratio_plot.x, ratio_plot.y1, yerr=ratio_plot.y1_err, color="C6", label="fit", style="step", ylabel="Counts", new_figure=False)


def EstimateCounts(postfit_pred: cabinetry.model_utils.ModelPrediction) -> tuple[np.ndarray, np.ndarray]:
    """ Extract post-fit yields and bin uncertainties for the interaction channels.

        Args:
            postfit_pred (cabinetry.model_utils.ModelPrediction): Post-fit model prediction.

        Returns:
            tuple[np.ndarray, np.ndarray]: Matrix and corresponding bin-uncertainty matrix.
    """
    if any([c["name"] == "mean_track_score" for c in postfit_pred.model.spec["channels"]]):
        KE_int_prediction = SliceModelPrediction(postfit_pred, slice(-1), "KE_int_postfit")
    else:
        KE_int_prediction = SliceModelPrediction(postfit_pred, slice(0, len(postfit_pred.model_yields)), "KE_int_postfit")

    L = KE_int_prediction.model_yields
    L_err = KE_int_prediction.total_stdev_model_bins[:, :-1]

    return L, L_err


def EstimateBackgroundAllRegions(postfit_pred: cabinetry.model_utils.ModelPrediction, template: AnalysisInput, signal_process: str) -> tuple[np.ndarray, np.ndarray]:
    """ Sum predicted background yields across all regions.

        Args:
            postfit_pred (cabinetry.model_utils.ModelPrediction): Post-fit model prediction.
            template (AnalysisInput): Input whose region labels identify the signal region.
            signal_process (str): Region label excluded as signal; other regions are treated as background.

        Returns:
            tuple[np.ndarray, np.ndarray]: Yields and uncertainties summed across regions.
    """
    N, N_err = EstimateCounts(postfit_pred)
    N = np.sum(N, 0)
    N_err = quadsum(N_err, 0)

    labels = list(template.regions.keys())
    N_bkg_err = N_err[signal_process != np.array(labels)]
    N_bkg = N[signal_process != np.array(labels)]

    return N_bkg, N_bkg_err


def EstimateBackgroundInRegions(postfit_pred: cabinetry.model_utils.ModelPrediction, data: AnalysisInput) -> tuple[dict, dict]:
    """ Estimate the background yields in each region by excluding its matching signal bin.

        Args:
            postfit_pred (cabinetry.model_utils.ModelPrediction): Post-fit model prediction.
            data (AnalysisInput): Input whose region labels define the signal-process names.

        Returns:
            tuple[dict, dict]: Of background yields and uncertainties keyed by region label.
    """
    N, N_err = EstimateCounts(postfit_pred)
    processes = list(data.regions.keys())

    bkg_in_region = {}
    bkg_in_region_err = {}

    for p in processes:
        signal = p
        signal_index = processes.index(signal)

        signal_region = N[signal_index]
        signal_region_err = N_err[signal_index]

        bkg_in_region[signal] = np.array([signal_region[i] for i in range(len(signal_region)) if i != signal_index])
        bkg_in_region_err[signal] = np.array([signal_region_err[i] for i in range(len(signal_region_err)) if i != signal_index])

    return bkg_in_region, bkg_in_region_err
