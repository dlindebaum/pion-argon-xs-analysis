from __future__ import annotations

import numpy as np
import pyhf
import cabinetry

from cabinetry.fit.results_containers import FitResults
from python.analysis import Plots
from python.analysis.Utils import quadsum

from python.analysis.cross_section import AnalysisInput, Slices


def Model(n_channels: int, KE_int_templates: np.ndarray, mean_track_score_templates: np.ndarray = None, mc_stat_unc: bool = False) -> pyhf.Model:
    """ Create pyhf model for region fit.

    Args:
        n_channels (int): number of channels in the model.
        KE_int_templates (np.ndarray): Templates for the KE_int distributions for each channel.
        mean_track_score_templates (np.ndarray, optional): Templates for the mean track score distributions for each channel. Defaults to None.
        mc_stat_unc (bool, optional): Whether to include Monte Carlo statistical uncertainties. Defaults to False.

    Returns:
        pyhf.Model: Pyhf model.
    """
    def channel(channel_name: str, samples: np.ndarray, mc_stat_unc: bool):
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
    """ Print Model specifications.

    Args:
        model (pyhf.Model): Pyhd model.
    """
    print(f"  channels: {model.config.channels}")
    print(f"     nbins: {model.config.channel_nbins}")
    print(f"   samples: {model.config.samples}")
    print(f" modifiers: {model.config.modifiers}")
    print(f"parameters: {model.config.parameters}")
    print(f"  nauxdata: {model.config.nauxdata}")
    print(f"   auxdata: {model.config.auxdata}")


def GenerateObservations(fit_input: AnalysisInput, energy_slices: Slices, mean_track_score_bins: np.ndarray, model: pyhf.Model, verbose: bool = True, single_bin: bool = False) -> np.ndarray:
    """ Create Observable distribution to fit with.

    Args:
        fit_input (AnalysisInput): AnalysisInput object containing the data to fit.
        energy_slices (Slices): energy slices to use for the fit.
        mean_track_score_bins (np.ndarray): Bins for the mean track score distribution.
        model (pyhf.Model): Pyhf model.
        verbose (bool, optional): Optionally print verbose output. Defaults to True.
        single_bin (bool, optional): Perform a single bin fit. Defaults to False.

    Returns:
        np.ndarray: Bins of the obsevations.
    """
    data = CreateObservedInputData(fit_input, energy_slices, mean_track_score_bins, single_bin)
    if verbose is True:
        print(f"{model.config.suggested_init()=}")
    observations = np.concatenate(data + [model.config.auxdata])
    if verbose is True:
        print(f"{model.logpdf(pars=model.config.suggested_init(), data=observations)=}")
    return observations


def Fit(observations, model: pyhf.Model, init_params: list[float] = None, par_bounds: list[tuple] = None, verbose: bool = True, tolerance: float = 1E-2, fix_pars: list[bool] = None) -> FitResults:
    """ Perform region fit.

    Args:
        observations (_type_): Observations to fit with.
        model (pyhf.Model): Pyhf model.
        init_params (list[float], optional): Initial parameters. Defaults to None.
        par_bounds (list[tuple], optional): Parameter bounds. Defaults to None.
        verbose (bool, optional): Optionally print verbose output. Defaults to True.
        tolerance (float, optional): Tolerance for fit convergence (difference between log likelihood of the current and previous iteration). Defaults to 1E-2.
        fix_pars (list[bool], optional): Boolean values to indicate if the parameter value should be fixed. Defaults to None.

    Returns:
        FitResults: Results of the fit.
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
    """ Compute  the correlation matrix for the smaples and channels using normalisations.

    Args:
        model (pyhf.Model): Pyhf model.
        mu (np.ndarray): Mean normalisation values.

    Returns:
        np.ndarray: Correlation matrix.
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
    """ Create KE template distributions for defining the model samples.

    Args:
        analysis_input (AnalysisInput): MC analysis input file.
        energy_slices (Slices): Energy slices.
        single_bin (bool, optional): Whether to create single bin templates. Defaults to False.
        pad (bool, optional): Pad the counts to ensure there are no zero entries. Defaults to False.
        reco (bool, optional): Whether to use reconstructed data. Defaults to True.

    Returns:
        list[np.ndarray]: List of KE templates for each region and sample.
    """
    model_input_data = []
    for c in analysis_input.regions:
        tmp = []
        for s in analysis_input.exclusive_process:
            n_int = analysis_input.NInteract(energy_slices, analysis_input.exclusive_process[s], analysis_input.regions[c], reco, analysis_input.weights) + 1E-10 * int(pad)
            if single_bin:
                tmp.append(np.array([sum(n_int)]))
            else:
                tmp.append(n_int)
        model_input_data.append(tmp)
    return model_input_data


def CreateMeanTrackScoreTemplates(analysis_input: AnalysisInput, bins: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """ Create mean track score template distributions.

    Args:
        analysis_input (AnalysisInput): _description_
        bins (np.ndarray): _description_
        weights (np.ndarray, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    templates = []
    for t in analysis_input.exclusive_process:
        mask = analysis_input.exclusive_process[t]
        templates.append(np.histogram(analysis_input.mean_track_score[mask], bins, weights=weights[mask] if weights is not None else weights)[0])
    return np.array(templates)


def CreateModel(template: AnalysisInput, energy_slice: Slices, mean_track_score_bins: np.ndarray, return_templates: bool = False, weights: np.ndarray = None, mc_stat_unc: bool = True, pad: bool = True, single_bin: bool = False) -> pyhf.Model:
    """ Create fit model from an MC analysis input.

    Args:
        template (AnalysisInput): MC analysis input file.
        energy_slice (Slices): Energy slices.
        mean_track_score_bins (np.ndarray): Mean track score bins.
        return_templates (bool, optional): Return templates along with the pyhf model. Defaults to False.
        weights (np.ndarray, optional): Weights for the MC events. Defaults to None.
        mc_stat_unc (bool, optional): MC statistical uncertainties. Defaults to True.
        pad (bool, optional): Pad the counts to ensure there are no zero entries. Defaults to True.
        single_bin (bool, optional): Create a single bin model. Defaults to False.

    Returns:
        pyhf.Model: Pyhf model and optionally template distributions.
    """
    templates_energy = CreateKEIntTemplates(template, energy_slice, single_bin, pad)
    if mean_track_score_bins is not None:
        templates_mean_track_score = CreateMeanTrackScoreTemplates(template, mean_track_score_bins, weights)
    else:
        templates_mean_track_score = None
    model = Model(len(template.regions), templates_energy, templates_mean_track_score, mc_stat_unc=mc_stat_unc)
    PrintModelSpecs(model)
    if return_templates is True:
        return model, templates_energy, templates_mean_track_score
    return model


def CreateObservedInputData(fit_input: AnalysisInput, slices: Slices, mean_track_score_bins: np.ndarray = None, single_bin: bool = False) -> np.ndarray:
    """ Create observed distributions from an analysis input.

    Args:
        fit_input (AnalysisInput): Analysis input.
        slices (Slices): energy slices.
        mean_track_score_bins (np.ndarray, optional): mean track score bins. Defaults to None.
        single_bin (bool, optional): if the fit is single binned. Defaults to False.

    Returns:
        np.ndarray: observable distributions to fit with.
    """
    observed_binned = []
    if fit_input.inclusive_process is None:
        mask = np.ones_like(fit_input.KE_int_reco, dtype=bool)
    else:
        mask = fit_input.inclusive_process
    for v in fit_input.regions.values():
        n_int = fit_input.NInteract(slices, v & mask, reco=True)
        if single_bin:
            n_int = [sum(n_int)]
        observed_binned.append(n_int)
    if mean_track_score_bins is not None:
        observed_binned.append(np.histogram(fit_input.mean_track_score[fit_input.inclusive_process], mean_track_score_bins)[0])
    return observed_binned


def SliceModelPrediction(prediction: cabinetry.model_utils.ModelPrediction, slice_: slice, label: str) -> cabinetry.model_utils.ModelPrediction:
    """ Calculate the model prediction of the yields.

    Args:
        prediction (cabinetry.model_utils.ModelPrediction): Model prediction base.
        slice_ (slice): slice of bins to get.
        label (str): model prediction label.

    Returns:
        cabinetry.model_utils.ModelPrediction: Model prediction.
    """
    return cabinetry.model_utils.ModelPrediction(prediction.model, np.array(prediction.model_yields[slice_]), np.array(prediction.total_stdev_model_bins[slice_]), np.array(prediction.total_stdev_model_channels[slice_]), label)


def PlotPrefitPostFit(prefit : np.ndarray, prefit_err: np.ndarray, postfit: np.ndarray, postfit_err: np.ndarray, energy_bins: np.ndarray, xlabel="$KE_{int}$ (MeV)"):
    """ Plot prefit and post fit distributions.

    Args:
        prefit (np.ndarray): Prefit data.
        prefit_err (np.ndarray): Prefit errors.
        postfit (np.ndarray): Postfit data.
        postfit_err (np.ndarray): Postfit errors.
        energy_bins (np.ndarray): Energy bins.
        xlabel (str, optional): X-axis label. Defaults to "$KE_{int}$ (MeV)".
    """
    with Plots.RatioPlot(energy_bins[::-1], postfit, prefit, postfit_err, prefit_err, xlabel, "fit/ true") as ratio_plot:
        Plots.Plot(ratio_plot.x, ratio_plot.y2, yerr=ratio_plot.y2_err, color="C0", label="true", style="step", newFigure=False)
        Plots.Plot(ratio_plot.x, ratio_plot.y1, yerr=ratio_plot.y1_err, color="C6", label="fit", style="step", ylabel="Counts", newFigure=False)


def EstimateCounts(postfit_pred: cabinetry.model_utils.ModelPrediction) -> tuple[np.ndarray, np.ndarray]:
    """ Estimate the predicted yield and error for each bin.

    Args:
        postfit_pred (cabinetry.model_utils.ModelPrediction): Postfit model prediction.

    Returns:
        tuple[np.ndarray, np.ndarray]: Predicted yields and errors.
    """
    if any([c["name"] == "mean_track_score" for c in postfit_pred.model.spec["channels"]]):
        KE_int_prediction = SliceModelPrediction(postfit_pred, slice(-1), "KE_int_postfit")
    else:
        KE_int_prediction = SliceModelPrediction(postfit_pred, slice(0, len(postfit_pred.model_yields)), "KE_int_postfit")

    L = KE_int_prediction.model_yields
    L_err = KE_int_prediction.total_stdev_model_bins[:, :-1]

    return L, L_err


def EstimateBackgroundAllRegions(postfit_pred: cabinetry.model_utils.ModelPrediction, template: AnalysisInput, signal_process: str) -> tuple[np.ndarray, np.ndarray]:
    """ Estimate backgorund counts for across all regions.

    Args:
        postfit_pred (cabinetry.model_utils.ModelPrediction): Post fit prediction.
        template (AnalysisInput): model template analysis input.
        signal_process (str): signal process (others are assumed background to this).

    Returns:
        tuple[np.ndarray, np.ndarray]: Predicted background yields and errors.
    """
    N, N_err = EstimateCounts(postfit_pred)
    N = np.sum(N, 0)
    N_err = quadsum(N_err, 0)

    labels = list(template.regions.keys())
    N_bkg_err = N_err[signal_process != np.array(labels)]
    N_bkg = N[signal_process != np.array(labels)]

    return N_bkg, N_bkg_err


def EstimateBackgroundInRegions(postfit_pred: cabinetry.model_utils.ModelPrediction, data: AnalysisInput) -> tuple[dict, dict]:
    """ Estimate background counts for each region. The region label is considered the signal process.

    Args:
        postfit_pred (cabinetry.model_utils.ModelPrediction): Postfit model prediction.
        data (AnalysisInput): Data analysis input.

    Returns:
        tuple[dict, dict]: Predicted background yields and errors.
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
