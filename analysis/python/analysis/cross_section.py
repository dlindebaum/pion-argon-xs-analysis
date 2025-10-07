"""
Created on: 02/06/2023 10:37

Author: Shyam Bhuller

Description: Library for code used in the cross section analysis. Refer to the README to see which apps correspond to the cross section analysis.
"""
import argparse
import copy
import glob
import os

from collections import namedtuple
from dataclasses import dataclass

import awkward as ak
import cabinetry
import numpy as np
import pandas as pd
import pyhf
import uproot

from cabinetry.fit.results_containers import FitResults
from particle import Particle
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import chi2

from python.analysis import (
    BeamParticleSelection, PFOSelection, EventSelection, SelectionTools, EnergyTools,
    Fitting, Plots, vector, Tags, RegionIdentification, Processing, Slicing)
from python.analysis.Master import (
    LoadConfiguration, LoadObject, SaveObject, SaveConfiguration,
    ReadHDF5, Data, Ntuple_Type, timer, IO)
from python.analysis.Utils import *
from python.analysis.AnalysisInputs import AnalysisInput, AnalysisInputGNN
from apps.cex_gnn_analyse import known_gnn_theory_procs, known_upstream_methods

GEANT_XS = os.environ["PYTHONPATH"] + "/data/g4_xs_pi_KE_100.root"

### OVERRIDE UNFOLDING API TO RETURN COVARIANCE MATRIX ###

from pyunfold.callbacks import setup_callbacks_regularizer, Logger
from pyunfold.mix import Mixer
from pyunfold.teststat import get_ts
from pyunfold.priors import setup_prior
from pyunfold.utils import cast_to_array

def iterative_unfold(data=None, data_err=None, response=None,
                     response_err=None, efficiencies=None,
                     efficiencies_err=None, prior=None, ts='ks',
                     ts_stopping=0.01, max_iter=100, cov_type='multinomial',
                     return_iterations=False, callbacks=None):
    """Performs iterative unfolding. Custom method which returns the covariance matrix.

    Parameters
    ----------
    data : array_like
        Input observed data distribution.
    data_err : array_like
        Uncertainties of the input observed data distribution. Must be the
        same shape as ``data``.
    response : array_like
        Response matrix.
    response_err : array_like
        Uncertainties of response matrix. Must be the same shape as
        ``response``.
    efficiencies : array_like
        Detection efficiencies for the cause distribution.
    efficiencies_err : array_like
        Uncertainties of detection efficiencies. Must be the same shape as
        ``efficiencies``.
    prior : array_like, optional
        Prior distribution to use in unfolding. If None, then a uniform
        (or flat) prior will be used. If array_like, then must have the same
        shape as ``efficiencies`` (default is None).
    ts : {'ks', 'chi2', 'bf', 'rmd'}
        Test statistic to use for stopping condition (default is 'ks').
        For more information about the available test statistics, see the
        `Test Statistics API documentation <api.rst#test-statistics>`__.
    ts_stopping : float, optional
        Test statistic stopping condition. At each unfolding iteration, the
        test statistic is computed between the current and previous iteration.
        Once the test statistic drops below ts_stopping, the unfolding
        procedure is stopped (default is 0.01).
    max_iter : int, optional
        Maximum number of iterations to allow (default is 100).
    cov_type : {'multinomial', 'poisson'}
        Whether to use the Multinomial or Poisson form for the covariance
        matrix (default is 'multinomial').
    return_iterations : bool, optional
        Whether to return unfolded distributions for each iteration
        (default is False).
    callbacks : list, optional
        List of ``pyunfold.callbacks.Callback`` instances to be applied during
        unfolding (default is None, which means no Callbacks are applied).

    Returns
    -------
    unfolded_result : dict
        Returned if ``return_iterations`` is False (default). Dictionary
        containing the final unfolded distribution, associated uncertainties,
        and test statistic information.

        The returned ``dict`` has the following keys:

            unfolded
                Final unfolded cause distribution
            stat_err
                Statistical uncertainties on the unfolded cause distribution
            sys_err
                Systematic uncertainties on the unfolded cause distribution
                associated with limited statistics in the response matrix
            ts_iter
                Final test statistic value
            ts_stopping
                Test statistic stopping criterion
            num_iterations
                Number of unfolding iterations
            unfolding_matrix
                Unfolding matrix

    unfolding_iters : pandas.DataFrame
        Returned if ``return_iterations`` is True. DataFrame containing the
        unfolded distribution, associated uncertainties, test statistic
        information, etc. at each iteration.
    """
    # Validate user input
    inputs = {'data': data,
              'data_err': data_err,
              'response': response,
              'response_err': response_err,
              'efficiencies': efficiencies,
              'efficiencies_err': efficiencies_err
              }
    for name in inputs:
        if inputs[name] is None:
            raise ValueError('The input for {} must not be None.'.format(name))
        elif np.amin(inputs[name]) < 0:
            raise ValueError('The items in {} must be non-negative.'.format(name))

    data, data_err = cast_to_array(data, data_err)
    response, response_err = cast_to_array(response, response_err)
    efficiencies, efficiencies_err = cast_to_array(efficiencies,
                                                   efficiencies_err)

    num_causes = len(efficiencies)

    # Setup prior
    prior = setup_prior(prior=prior, num_causes=num_causes)

    # Define first prior counts distribution
    n_c = np.sum(data) * prior

    # Setup Mixer
    mixer = Mixer(data=data,
                  data_err=data_err,
                  efficiencies=efficiencies,
                  efficiencies_err=efficiencies_err,
                  response=response,
                  response_err=response_err,
                  cov_type=cov_type)

    # Setup test statistic
    ts_obj = get_ts(ts)
    ts_func = ts_obj(tol=ts_stopping,
                     num_causes=num_causes,
                     TestRange=[0, 1e2],
                     verbose=False)

    unfolding_iters = _unfold_custom(prior=n_c,
                                     mixer=mixer,
                                     ts_func=ts_func,
                                     max_iter=max_iter,
                                     callbacks=callbacks)

    if return_iterations:
        return unfolding_iters
    else:
        unfolded_result = dict(unfolding_iters.iloc[-1])
        return unfolded_result


def _unfold_custom(prior=None, mixer=None, ts_func=None, max_iter=100,
            callbacks=None):
    """Perform iterative unfolding. Custom version of the method which just returns the covariance method in addition to the regular output

    Parameters
    ----------
    prior : array_like
        Initial cause distribution.
    mixer : pyunfold.Mix.Mixer
        Mixer to perform the unfolding.
    ts_func : pyunfold.Utils.TestStat
        Test statistic object.
    max_iter : int, optional
        Maximum allowed number of iterations to perform.
    callbacks : list, optional
        List of ``pyunfold.callbacks.Callback`` instances to be applied during
        unfolding (default is None, which means no Callbacks are applied).

    Returns
    -------
    unfolding_iters : pandas.DataFrame
        DataFrame containing the unfolded result for each iteration.
        Each row in unfolding_result corresponds to an iteration.
    """
    # Set up callbacks, regularizer Callbacks are treated separately
    callbacks, regularizer = setup_callbacks_regularizer(callbacks)
    callbacks.on_unfolding_begin()

    current_n_c = prior.copy()
    iteration = 0
    unfolding_iters = []
    while not ts_func.pass_tol() and iteration < max_iter:
        callbacks.on_iteration_begin(iteration=iteration)

        # Perform unfolding for this iteration
        unfolded_n_c = mixer.smear(current_n_c)
        iteration += 1
        status = {'unfolded': unfolded_n_c,
                  'stat_err': mixer.get_stat_err(),
                  'sys_err': mixer.get_MC_err(),
                  'num_iterations': iteration,
                  'unfolding_matrix': mixer.Mij,
                  'covariance_matrix': mixer.get_cov()}

        if regularizer:
            # Will want the nonregularized distribution for the final iteration
            unfolded_nonregularized = status['unfolded'].copy()
            regularizer.on_iteration_end(iteration=iteration, status=status)

        ts_iter = ts_func.calc(status['unfolded'], current_n_c)
        status['ts_iter'] = ts_iter
        status['ts_stopping'] = ts_func.tol

        callbacks.on_iteration_end(iteration=iteration, status=status)
        unfolding_iters.append(status)

        # Updated current distribution for next iteration of unfolding
        current_n_c = status['unfolded'].copy()

    # Convert unfolding_iters list of dictionaries to a pandas DataFrame
    unfolding_iters = pd.DataFrame.from_records(unfolding_iters)

    # Replace final folded iteration with un-regularized distribution
    if regularizer:
        last_iteration_index = unfolding_iters.index[-1]
        unfolding_iters.at[last_iteration_index, 'unfolded'] = unfolded_nonregularized

    callbacks.on_unfolding_end(status=status)

    return unfolding_iters


# required_parset = pyhf.modifiers.staterror.required_parset
# def to_poisson(func):
#     def wrapper(*args, **kwargs):
#       result = required_parset(*args, **kwargs)
#       result['paramset_type'] = 'constrained_by_poisson'
#       result['factors'] = result.pop('sigmas')
#       return result
#     return wrapper

# pyhf.modifiers.staterror.required_parset = to_poisson(pyhf.modifiers.staterror.required_parset)

class PlotStyler:
    def __init__(self, extend_colors : bool = False, custom_colors : list = None, dpi : int = 100, dark : bool = False, font_scale : float = 1, font_style : str = "sans"):
        self.args = locals()
        self.args.pop("self")
        PlotStyler.SetPlotStyle(**self.args)

    class __updater__:
        def __init__(self, parent : "PlotStyler", extend_colors : bool = None, custom_colors : list = None, dpi : int = None, dark : bool = None, font_scale : float = None, font_style : str = None):
            self.args = locals()
            self.args.pop("self")
            self.args.pop("parent")
            self.parent = parent

            for k in self.args:
                if self.args[k] is None:
                    self.args[k] = self.parent.args[k]

            pass
        def __enter__(self):
            PlotStyler.SetPlotStyle(**self.args)
            pass
        def __exit__(self, type, value, traceback):
            PlotStyler.SetPlotStyle(**self.parent.args)
            pass

    def Update(self, extend_colors : bool = None, custom_colors : list = None, dpi : int = None, dark : bool = None, font_scale : float = None, font_style : str = None):
        return self.__updater__(self, extend_colors, custom_colors, dpi, dark, font_scale, font_style)

    @staticmethod
    def SetPlotStyle(extend_colors : bool = False, custom_colors : list = None, dpi : int = 300, dark : bool = False, font_scale : float = 1, font_style : str = "sans"):
        Plots.plt.style.use("default") # first load the default to reset any previous changes made by other styles
        Plots.plt.style.use("~/pi0-analysis/analysis/config/thesis_plotstyle.mplstyle")
        # Plots.plt.style.use('seaborn-v0_8-paper')
        # Plots.plt.rcParams.update({'patch.linewidth': 1})
        # Plots.plt.rcParams.update({'font.size': font_scale * 10})
        # # Plots.plt.rcParams.update({"axes.titlecolor" : "#555555"})
        # Plots.plt.rcParams.update({"axes.titlesize" : font_scale * 12})
        # Plots.plt.rcParams['figure.dpi'] = dpi
        # Plots.plt.rcParams['legend.fontsize'] = "small"
        # # Plots.plt.rcParams["font.family"] = font_style

        Plots.plt.rc('text.latex', preamble=r"\\usepackage{amsmath}")
        if custom_colors:
            Plots.plt.rcParams.update({"axes.prop_cycle" : Plots.plt.cycler("color", custom_colors)})
        if dark:
            l_2 = [
            Plots.matplotlib.cm.get_cmap("tab20c").colors[0],
            Plots.matplotlib.cm.get_cmap("tab20c").colors[8],
            Plots.matplotlib.cm.get_cmap("tab20b").colors[13],
            Plots.matplotlib.cm.get_cmap("tab20b").colors[0],
            Plots.matplotlib.cm.get_cmap("tab20b").colors[17],
            Plots.matplotlib.cm.get_cmap("tab20b").colors[4],
            Plots.matplotlib.cm.get_cmap("tab20c").colors[12],
            Plots.matplotlib.cm.get_cmap("tab20c").colors[16],
            ]
            Plots.plt.rcParams.update({"axes.prop_cycle" : Plots.plt.cycler("color", l_2)})
        if extend_colors:
            Plots.plt.rcParams.update({"axes.prop_cycle" : Plots.plt.cycler("color", Plots.matplotlib.cm.get_cmap("tab20").colors)})
        return

def CountInRegions(true_regions : dict, reco_regions : dict, selection_efficincy : np.ndarray = None) -> np.ndarray:
    """ Computes the counts of each combination of reco and true regions.

    Args:
        true_regions (dict): true region masks
        reco_regions (dict): reco region masks
        return_counts (bool, optional): return matrix of counts. Defaults to False.

    Returns:
        np.ndarray: counts.
    """
    counts = []
    for t in true_regions:
        true_counts = []
        for r in reco_regions:
            mask = true_regions[t] & reco_regions[r]
            if selection_efficincy is not None: mask = mask & selection_efficincy
            true_counts.append(ak.sum(mask)) # true counts for each reco region
        counts.append(true_counts)
    return counts

def RatioWeights(beam_inst_P : np.ndarray, func : str, params : list, truncate : int = 10):
    weights = 1/getattr(Fitting, func)(beam_inst_P, *params)
    weights = np.where(weights > truncate, truncate, weights)
    return weights

def PlotXSHists(energy_slices, hist_counts : np.ndarray, hist_counts_err : np.ndarray = None, overflow : bool = True, scale : float = 1, xlabel : str = "$KE$ (MeV)", ylabel : str = "Counts", label : str = None, color : str = None, newFigure : bool = True, title : str = None):
    if hist_counts_err is None:
        hist_counts_err = np.sqrt(hist_counts)

    if overflow is False:
        s = slice(1, -1)
    else:
        s = slice(0, len(energy_slices.pos_overflow))
    x = energy_slices.pos_overflow - energy_slices.width/2
    x = x[s]

    Plots.Plot(x, scale * hist_counts[s], yerr = scale * hist_counts_err[s], xlabel = xlabel, newFigure = newFigure, style = "step", label = label, color = color, ylabel = ylabel, title = title)
    return

def HypTestXS(cv, error, process, energy_slice, file = GEANT_XS):
    xs_sim = GeantCrossSections(file, energy_range = [energy_slice.min_pos - energy_slice.width, energy_slice.max_pos])
    sim_curve_interp = xs_sim.GetInterpolatedCurve(process)
    x = energy_slice.pos[:-1] - energy_slice.width/2

    w_chi_sqr = weighted_chi_sqr(cv, sim_curve_interp(x), error)

    p = chi2.sf((len(x)-1) * w_chi_sqr, len(x) - 1)
    return {"w_chi2" : w_chi_sqr, "p" : p}

def PlotXSComparison(xs : dict[np.ndarray], energy_slice, process : str = None, colors : dict[str] = None, xs_sim_color : str = "k", title : str = None, simulation_label : str = "simulation", chi2 : bool = True, newFigure : bool = True, cv_only : bool = False, marker_size : float = 6):
    ave_slice_width = (np.max(energy_slice.bin_edges) - np.min(energy_slice.bin_edges))/energy_slice.n_slices
    xs_sim = GeantCrossSections(
        energy_range = [energy_slice.min_pos - ave_slice_width,
                        energy_slice.max_pos + ave_slice_width])

    if colors is None:
        colors = {k : f"C{i}" for i, k in enumerate(xs)}

    sim_curve_interp = xs_sim.GetInterpolatedCurve(process)
    x = energy_slice.pos[:-1] - energy_slice.width/2

    if newFigure is True: Plots.plt.figure()
    chi_sqrs = {}
    for k, v in xs.items():
        w_chi_sqr = weighted_chi_sqr(v[0], sim_curve_interp(x), v[1])
        chi_sqrs[k] = w_chi_sqr
        if (chi2 is True) and (cv_only is False):
            chi2_l = ", $\chi^{2}/ndf$ = " + f"{w_chi_sqr:.3g}"
        else:
            chi2_l = ""
        Plots.Plot(x, v[0], xerr = energy_slice.width / 2  if cv_only is False else None, yerr = v[1] if cv_only is False else None, label = k + chi2_l, color = colors[k], linestyle = "", marker = "x", newFigure = False, markersize = marker_size, capsize = marker_size/2)
    
    if process == "single_pion_production":
        Plots.Plot(xs_sim.KE, sim_curve_interp(xs_sim.KE), label = simulation_label, title = "Single pion production" if title is None else title.capitalize(), newFigure = False, xlabel = "$KE (MeV)$", ylabel = "$\sigma (mb)$", color = xs_sim_color)
    else:
        xs_sim.Plot(process, label = simulation_label, color = xs_sim_color, title = title.capitalize() if type(title) is str else title)

    Plots.plt.ylim(0)
    if max(Plots.plt.gca().get_ylim()) > np.nanmax(sim_curve_interp(xs_sim.KE).astype(float)) * 2:
        Plots.plt.ylim(0, max(sim_curve_interp(xs_sim.KE)) * 2)
    Plots.plt.xlim(energy_slice.min_pos - (0.2 * energy_slice.width), energy_slice.max_pos + (0.2 * energy_slice.width))
    return chi_sqrs


class ApplicationArguments:
    @staticmethod
    def Processing(parser : argparse.ArgumentParser):
        parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "Number of batches to split n tuple files into when parallel processing processing data.")
        parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "Number of events to process when parallel processing data.")
        parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "Number of threads to use when processsing.")

    @staticmethod
    def Regen(parser : argparse.ArgumentParser):
        parser.add_argument("-r", "--regen", dest = "regen", action = "store_true", help = "Regenerate any stored data.")

    @staticmethod
    def Output(parser : argparse.ArgumentParser, default : str = None):
        parser.add_argument("-o", "--out", dest = "out", type = str, default = default, help = "Directory to save output files.")
        return

    @staticmethod
    def Plots(parser : argparse.ArgumentParser):
        parser.add_argument("--nbins", dest = "nbins", type = int, default = 50, help = "Number of bins to make for histogram plots.")
        parser.add_argument("-a", "--annotation", dest = "annotation", type = str, default = None, help = "Annotation to add to plots.")
        return

    @staticmethod
    def Config(parser : argparse.ArgumentParser, required : bool = False):
        parser.add_argument("-c", "--config", dest = "config", type = str, default = None, required = required, help = "Analysis configuration file, if supplied will override command line arguments.")

    @staticmethod
    def ResolveArgs(args : argparse.Namespace, override_out : bool = True) -> argparse.Namespace:
        """ Parses command line arguements.

        Args:
            args (argparse.Namespace): arguements to parse

        Returns:
            argparse.Namespace: parsed arguements
        """
        if hasattr(args, "config"):
            args_copy = argparse.Namespace()
            for a, v in args._get_kwargs():
                setattr(args_copy, a, v)
            args = ApplicationArguments.ResolveConfig(LoadConfiguration(args.config))
            for a, v in args_copy._get_kwargs():
                if a not in args:
                    setattr(args, a, v)
        else:
            if hasattr(args, "data_file") and hasattr(args, "data_beam_quality_fit"):
                if args.data_file is not None and args.data_beam_quality_fit is None:
                    raise Exception("beam quality fit values for data are required")

        if hasattr(args, "out") and (override_out is True):
            if args.out is None:
                filename = None
                if hasattr(args, "mc_file"):
                    filename = args.mc_file
                elif hasattr(args, "data_file"):
                    filename = args.data_file
                elif hasattr(args, "file"):
                    filename = args.file
                else:
                    filename = ""

                if type(filename) == list:
                    if len(filename) == 1:
                        args.out = filename[0].split("/")[-1].split(".")[0] + "/"
                    else:
                        args.out = "output/" #? how to make a better name for multiple input files?
                else:
                    args.out = filename.split("/")[-1].split(".")[0] + "/"
            if args.out[-1] != "/": args.out += "/"

        return args

    @staticmethod
    def __CreateSelection(value : dict, module) -> dict:
        """ Creates a dictionary of selection functions and their argumenets as specified in the analysise configuration file. 

        Args:
            value (dict): dicionary describing a particular selection, key value is the function name, value is a list of function arguements.
            module (module): which python module does this selection belong to.

        Returns:
            _type_: _description_
        """
        selection = {"selections" : {}, "arguments" : {}}
        for func, opt in value.items():
            if opt["enable"] is True:
                selection["selections"][func] = getattr(module, func)
                copy = opt.copy()
                copy.pop("enable")
                selection["arguments"][func] = copy
        return selection

    @staticmethod
    def ResolveConfig(config : dict) -> argparse.Namespace:
        """ Reads analysis configuration file and unpacks/serializes relavent objects.

        Args:
            config (dict): file path

        Returns:
            argparse.Namespace: unpacked configuration
        """
        args = argparse.Namespace()
        for head, value in config.items():
            if head == "NTUPLE_FILES":
                args.ntuple_files = value
                args.has_data = "data" in value.keys()
            elif head == "REGION_IDENTIFICATION":
                args.gnn_do_predict = value["type"] in ["gnn"]
                args.sample_only = value["type"] in ["sample_only"]
                # args.gnn_event_by_event = value["type"] == "gnn_per_event"
                args.region_identification = RegionIdentification.regions[value["type"]]
            elif head == "BEAM_PARTICLE_SELECTION":
                args.beam_selection = ApplicationArguments.__CreateSelection(value, BeamParticleSelection)
            elif head == "HAS_FINAL_STATE_PFO_SELECTION":
                args.has_final_state_pfo_selection = value["enable"]
            elif head == "VALID_PFO_SELECTION":
                args.valid_pfo_selection = value["enable"]
            elif head == "FINAL_STATE_PIPLUS_SELECTION":
                args.piplus_selection = ApplicationArguments.__CreateSelection(value, PFOSelection)
            elif head == "FINAL_STATE_PHOTON_SELECTION":
                args.photon_selection = ApplicationArguments.__CreateSelection(value, PFOSelection)
            elif head == "FINAL_STATE_LOOSE_PION_SELECTION":
                args.loose_pion_selection = ApplicationArguments.__CreateSelection(value, PFOSelection)
            elif head == "FINAL_STATE_LOOSE_PHOTON_SELECTION":
                args.loose_photon_selection = ApplicationArguments.__CreateSelection(value, PFOSelection)
            elif head == "FINAL_STATE_PI0_SELECTION":
                args.pi0_selection = ApplicationArguments.__CreateSelection(value, EventSelection)
            elif head == "BEAM_QUALITY_FITS":
                if "mc" in value:
                    args.mc_beam_quality_fit = LoadConfiguration(value["mc"]) # generally expected to have MC at a minimum
                if "data" in value:
                    args.data_beam_quality_fit = LoadConfiguration(value["data"])
                args.beam_quality_truncate = value["truncate"]
            elif head == "BEAM_SCRAPER_FITS":
                args.beam_scraper_energy_range = value["energy_range"]
                args.beam_scraper_energy_bins = value["energy_bins"]
                if "mc" in value:
                    args.mc_beam_scraper_fit = LoadConfiguration(value["mc"])
            elif head == "ENERGY_CORRECTION":
                args.shower_correction = {}
                for k, v in value.items():
                    args.shower_correction[k] = v
            elif head == "UPSTREAM_ENERGY_LOSS":
                args.upstream_loss_cv_function = value["cv_function"]
                args.upstream_loss_response = getattr(Fitting, value["response"])
                args.upstream_loss_bins = value["bins"]
                if "correction_params" in value:
                    if value["correction_params"] is not None:
                        args.upstream_loss_correction_params = LoadConfiguration(value["correction_params"])
            elif head == "BEAM_REWEIGHT":
                args.beam_reweight = {}
                if value["params"] is not None:
                    args.beam_reweight["params"] = LoadConfiguration(value["params"])
                args.beam_reweight["strength"] = value["strength"]
            elif head == "FIDUCIAL_VOLUME":
                args.fiducial_volume = {
                    "start": min(value),
                    "end": max(value),
                    "list": [min(value), max(value)]}
            elif head == "BEAM_SELECTION_MASKS":
                args.beam_selection_masks = {}
                for k, v in value.items():
                    args.beam_selection_masks[k] = {i : LoadObject(j) for i, j in v.items()}
            elif head == "REGION_SELECTION_MASKS":
                args.region_selection_masks = {}
                for k, v in value.items():
                    args.region_selection_masks[k] = {i : LoadObject(j) for i, j in v.items()}
            elif head == "MC_EFFICIENCIES":
                args.mc_efficiencies = {i : LoadObject(j) for i, j in value.items()}
            elif head == "TOY_PARAMETERS":
                args.toy_parameters = {}
                for k, v in value.items():
                    if k == "beam_profile": 
                        args.toy_parameters[k] = getattr(Fitting, v)
                    else:
                        args.toy_parameters[k] = v
            elif head == "FIT":
                args.fit = {}
                for k, v in value.items():
                    args.fit[k] = v
            elif head == "ESLICE":
                if ("edges" in value.keys()) and (value["edges"] is not None):
                    args.energy_slices = Slicing.Slices(value["edges"])
                elif value["width"] is not None:
                    args.energy_slices = Slicing.Slices(
                        value["width"],
                        # min - width to allocate an underflow bin
                        #   (not used in the measurement)
                        value["min"] - value["width"],
                        value["max"], reversed = True)
            elif head == "SYSTEMATICS":
                sys_dict = {}
                standard_syst = ["track_length", "beam_momentum",
                                 "GNN_theory", "beam_reweight",
                                 "purity"]
                other_syst = ["GNN_model", "upstream_energy"]
                for k in value.keys():
                    if k not in (standard_syst + other_syst):
                        raise NotImplementedError(f"Unknown systematic {k}")
                for sys in standard_syst:
                    if sys in value.keys():
                        if value[sys] is not None:
                            sys_dict.update({sys: value[sys]})
                if "GNN_model" in value.keys():
                    if value["GNN_model"] is not None:
                        # known_gnn_theory_procs imported from apps.cex_gnn_analyse
                        if value["GNN_model"] not in known_gnn_theory_procs.keys():
                            raise NotImplementedError(f"Unknown process {value['GNN_model']}")
                        sys_dict.update({"GNN_model": value["GNN_model"]})
                if "upstream_energy" in value.keys():
                    if value["upstream_energy"] is not None:
                        # known_upstream_methods imported from apps.cex_gnn_analyse
                        if value["upstream_energy"] not in known_upstream_methods:
                            raise NotImplementedError(f"Unknown sys method {value['upstream_energy']}")
                        sys_dict.update({"upstream_energy": value["upstream_energy"]})
                args.systematics = sys_dict
            elif head == "ANALYSIS_INPUTS":
                args.analysis_input = {k : v for k, v in value.items()}
            elif head == "UNFOLDING":
                if "purity_bin" in value.keys():
                    args.uf_purity_bin = value.pop("purity_bin")
                else:
                    args.uf_purity_bin = False
                args.unfolding = {k : v for k, v in value.items()}
            elif head == "GNN_MODEL":
                # args.gnn_model_path = str(path)
                args.gnn_model_path = value["model_path"]
                args.gnn_region_labels = value["region_labels"]
            elif head == "GNN_PREDICTIONS":
                args.gnn_results = {}
                for info, save_map in value.items():
                    args.gnn_results[info] = {
                        which_pred : LoadObject(path)
                            for which_pred, path in save_map.items()}
            else:
                setattr(args, head, value) # allow for generic configurations in the json file
        args.multi_dim_bins = Slicing.MultiDimBins(
            args.energy_slices.bin_edges_with_overflow, True, args.uf_purity_bin)
        if hasattr(args, "beam_selection"):
            ApplicationArguments.DataMCSelectionArgs(args)
        if hasattr(args, "pi0_selection"):
            ApplicationArguments.AddEnergyCorrection(args)
        if hasattr(args, "beam_selection"):
            if "PiBeamSelection" in args.beam_selection["mc_arguments"]:
                args.beam_selection["data_arguments"]["PiBeamSelection"]["use_beam_inst"] = (
                    args.beam_selection["data_arguments"]["PiBeamSelection"].pop("use_beam_inst_data"))
                args.beam_selection["mc_arguments"]["PiBeamSelection"]["use_beam_inst"] = (
                    args.beam_selection["mc_arguments"]["PiBeamSelection"].pop("use_beam_inst_mc"))
                del args.beam_selection["data_arguments"]["PiBeamSelection"]["use_beam_inst_mc"]
                del args.beam_selection["mc_arguments"]["PiBeamSelection"]["use_beam_inst_data"]
            if hasattr(args, "fiducial_volume"):
                fiducial_args = {
                    "cut": args.fiducial_volume["start"],
                    "op": ">"}
                args.beam_selection["selections"].update({
                    "FiducialStart": BeamParticleSelection.FiducialStart})
                args.beam_selection["data_arguments"].update(
                    {"FiducialStart": fiducial_args})
                args.beam_selection["mc_arguments"].update(
                    {"FiducialStart": fiducial_args})

        return args


    @staticmethod
    def AddEnergyCorrection(args):
        if hasattr(args, "shower_correction") and (args.shower_correction["correction_params"] != None):
            method = EnergyTools.EnergyCorrection.shower_energy_correction[args.shower_correction["correction"]]
            params = LoadConfiguration(args.shower_correction["correction_params"])
        else:
            method = None
            params = None
            args.shower_correction["correction"] = None
            args.shower_correction["correction_params"] = None
        args.pi0_selection["mc_arguments"]["Pi0MassSelection"]["correction"] = method
        args.pi0_selection["mc_arguments"]["Pi0MassSelection"]["correction_params"] = params
        args.pi0_selection["data_arguments"]["Pi0MassSelection"]["correction"] = method
        args.pi0_selection["data_arguments"]["Pi0MassSelection"]["correction_params"] = params

    @staticmethod
    def DataMCSelectionArgs(args : argparse.Namespace):
        for a in vars(args):
            if ("selection" in a) and (type(getattr(args, a)) == dict):
                if "arguments" in getattr(args, a): 
                    getattr(args, a)["mc_arguments"] = copy.deepcopy(getattr(args, a)["arguments"])
                    getattr(args, a)["data_arguments"] = copy.deepcopy(getattr(args, a)["arguments"])
                    getattr(args, a).pop("arguments")

        for i, s in args.beam_selection["selections"].items():
            if s in [BeamParticleSelection.BeamQualityCut, BeamParticleSelection.DxyCut, BeamParticleSelection.DzCut, BeamParticleSelection.CosThetaCut]:
                if hasattr(args, "mc_beam_quality_fit"): 
                    args.beam_selection["mc_arguments"][i]["fits"] = args.mc_beam_quality_fit
                if hasattr(args, "data_beam_quality_fit"): 
                    args.beam_selection["data_arguments"][i]["fits"] = args.data_beam_quality_fit
            elif s == BeamParticleSelection.BeamScraperCut:
                if hasattr(args, "mc_beam_scraper_fit"): 
                    args.beam_selection["mc_arguments"][i]["fits"] = args.mc_beam_scraper_fit
                    args.beam_selection["data_arguments"][i]["fits"] = args.mc_beam_scraper_fit
            else:
                continue
        return args


class GeantCrossSections:
    """ Object for accessing Geant 4 cross sections from the root file generated with Geant4Reweight tools.
    """
    labels = {"abs_KE;1" : "absorption", "inel_KE;1" : "quasielastic", "cex_KE;1" : "charge_exchange", "dcex_KE;1" : "double_charge_exchange", "prod_KE;1" : "pion_production", "total_inel_KE;1" : "total_inelastic"}

    def __init__(self, file : str = GEANT_XS, energy_range : list = None, n_cascades : int = None) -> None:
        with uproot.open(file) as ufile: # open root file
            self.KE = ufile["abs_KE;1"].all_members["fX"] # load kinetic energy from one channel (shared for all cross section channels)

            if energy_range:
                self.KE = self.KE[(self.KE <= max(energy_range)) & (self.KE >= min(energy_range))]

            for k in ufile.keys():
                if "KE" in k:
                    g = ufile[k]
                    if energy_range:
                        mask = (g.all_members["fX"] <= max(energy_range)) & (g.all_members["fX"] >= min(energy_range))
                        xs = g.all_members["fY"][mask]
                    else:
                        xs = g.all_members["fY"]
                    s = "_frac" if "frac" in k else "" 
                    setattr(self, self.labels[k.replace("_frac", "")] + s, xs[0:len(self.KE)]) # assign class variables for each cross section channel

            self.exclusive_processes = list(self.labels.values())
            self.exclusive_processes.remove("total_inelastic")
            self.n_cascades = n_cascades
        pass


    def Stat_Error(self, xs : str) -> np.ndarray:
        """ Statisitical error of the simulation, done using binomial uncertainties. Only works if n_cascades is known.

        Args:
            xs (str): cross section process

        Returns:
            np.ndarray: statistical error
        """
        if (self.n_cascades is None) or (not hasattr(self, xs + "_frac")):
            return 0 * getattr(self, xs)
        else:
            return getattr(self, xs) * np.sqrt(getattr(self, xs + "_frac") / self.n_cascades)


    def __PlotAll(self, title : str = None):
        """ Plot all cross section channels.
        """
        for k in self.labels.values():
            Plots.Plot(self.KE, getattr(self, k), label = remove_(k), newFigure = False, xlabel = "$KE$ (MeV)", ylabel = "$\sigma$ (mb)", title = title)
            # Plots.plt.fill_between(self.KE, getattr(self, k) - self.Stat_Error(k), getattr(self, k) + self.Stat_Error(k), color = Plots.plt.gca()._get_lines.get_next_color())


    def Plot(self, xs : str, color : str = None, label : str = None, title : str = None):
        """ Plot cross sections. To be used in conjunction with other plots for comparisons.

        Args:
            xs (str): cross section channel to plot, if given all, will plot all cross section channels
            color (str, optional): colour of single plot. Defaults to None.
            label (str, optional): label of plot, if None, the channel name is used. Defaults to None.
            title (str, optional): title of plot, set to the channel name if label is provided. Defaults to None.
        """
        if xs == "all":
            self.__PlotAll(title = title)
        else:
            if label is None:
                label = remove_(xs)
            else:
                if title is None:
                    title = remove_(xs).capitalize()
            if xs == "single_pion_production":
                y = self.quasielastic + self.double_charge_exchange
            else:
                y = getattr(self, xs)
            Plots.Plot(self.KE, y, label = label, title = title, newFigure = False, xlabel = "$KE$ (MeV)", ylabel = "$\sigma$  (mb)", color = color)
            # Plots.plt.fill_between(self.KE, getattr(self, xs) - self.Stat_Error(xs), getattr(self, xs) + self.Stat_Error(xs), color = Plots.plt.gca()._get_lines.get_next_color())


    def GetInterpolatedCurve(self, process : str) -> interp1d:
        """ returns interpolated cross section curve as function of KE.

        Args:
            process (str): cross section process

        Returns:
            interp1d: _description_
        """
        if process == "single_pion_production":
            sigma = self.quasielastic + self.double_charge_exchange
        else:
            sigma = getattr(self, process)
        return interp1d(self.KE, sigma, fill_value = "extrapolate")


class Toy:
    def __init__(self, file : str = None, df : str = None) -> None:
        if file is not None:
            self.df = ReadHDF5(file)
        elif df is not None:
            self.df = df
        else:
            return
    
        self.exclusive_processes = np.unique(self.df.exclusive_process)
        self.exclusive_processes = self.exclusive_processes[self.exclusive_processes != ""]

    @staticmethod
    def GetRegion(toy : pd.DataFrame, region : str) -> pd.DataFrame:
        """ get region/process masks from data frame.

        Args:
            toy (pd.DataFrame): Toy
            region (str): region regex, either "truth_region_" or "reco_region_"

        Returns:
            pd.DataFrame: dataframe of masks
        """
        regions = toy.filter(regex = region)
        new_col_names = {}
        for i in regions:
            new_col_names[i] = i.split(region)[1]
        regions = regions.rename(columns = new_col_names)
        return regions


    def GetCorrelationMatrix(self) -> np.ndarray:
        """ Compute the confusion matrix for the reco/truth regions.

        Args:
            toy (pd.DataFrame): Toy

        Returns:
            np.ndarray: confusion matrix
        """
        reco_regions = Toy.GetRegion(self.df, "reco_regions_")
        true_regions = Toy.GetRegion(self.df, "truth_regions_")
        return CountInRegions(true_regions, reco_regions)


    def SetProperty(self, name : str, value : any):
        hidden_name = f"_{type(self).__name__}__{name}"
        if not hasattr(self, hidden_name):
            setattr(self, hidden_name, value)
        return getattr(self, hidden_name)


    def GetRegionNames(self, name : str) -> list[str]:
        """ Get names of each region as labelled in the dataframe column.

        Args:
            name (str): column regex

        Returns:
            list[str]: names
        """
        labels = self.df.filter(regex = name).columns
        return [s.split(name)[-1] for s in labels]

    @property
    def outside_tpc(self):
        return self.SetProperty("outside_tpc", (self.df.z_int < 0) | (self.df.z_int > 700))

    @property
    def outside_tpc_smeared(self):
        return self.SetProperty("outside_tpc_smeared", (self.df.z_int_smeared < 0) | (self.df.z_int_smeared > 700))

    @property
    def truth_regions(self):
        return self.SetProperty("truth_regions", self.GetRegion(self.df, "truth_regions_"))

    @property
    def reco_regions(self):
        return self.SetProperty("reco_regions", self.GetRegion(self.df, "reco_regions_"))

    @property
    def reco_region_labels(self):
        return self.GetRegionNames("reco_regions_")

    @property
    def truth_region_labels(self):
        return self.GetRegionNames("truth_regions_")

    @staticmethod
    def PlotObservablesInRegions(observable : pd.Series, reco_regions : pd.DataFrame, true_regions : pd.DataFrame, label : str, norm : bool = False, stacked : bool = False, histtype = "step"):
        """ Plot an observable from the toy in each region for each process.

        Args:
            observable (pd.Series): observable to plot
            reco_regions (pd.DataFrame): reco regions
            true_regions (pd.DataFrame): true regions
            label (str): x label
            norm (bool, optional): normalise plots. Defaults to False.
            stacked (bool, optional): stack histograms. Defaults to False.
            histtype (str, optional): histogram style. Defaults to "step".
        """
        for _, r in Plots.IterMultiPlot(reco_regions.columns):
            tmp_regions = {t : true_regions[t].values & reco_regions[r].values & (observable > 0) for t in true_regions.columns} # filter the reco events for this region only
            Plots.PlotTagged(observable, Tags.ExclusiveProcessTags(tmp_regions), bins = 50, newFigure = False, title = f"reco region : {r}", reverse_sort = False, stacked = stacked, histtype = histtype, x_label = label, ncols = 1, norm = norm)
        return


    def NInteract(
            self,
            energy_slice : Slicing.Slices,
            process : np.ndarray,
            mask : np.ndarray = None,
            weights : np.ndarray = None) -> np.ndarray:
        """ Exclusive interaction histogram using energy slice method.

        Args:
            energy_slice (Slicing.Slices): energy slices
            process (np.ndarray): exclusive process mask
            mask (np.ndarray, optional): additional mask to apply. Defaults to None.
            weights (np.ndarray, optional): event weights. Defaults to None.

        Returns:
            np.ndarray: exclusive interaction histogram
        """
        if mask is None: mask = np.ones(len(self.df), dtype = bool)
        w = weights if weights is None else weights[mask]
        n_interact = Slicing.EnergySlice.CountingExperiment(
            self.df.KE_int_smeared[mask].values,
            self.df.KE_init_smeared[mask].values,
            self.outside_tpc_smeared[mask].values,
            process[mask].values,
            energy_slice,
            interact_only = True,
            weights = w)
        return n_interact


class RegionFit:

    @staticmethod    
    def Model(n_channels : int, KE_int_templates : np.ndarray, mean_track_score_templates : np.ndarray = None, mc_stat_unc : bool = False) -> pyhf.Model:
        def channel(channel_name : str, samples : np.ndarray, mc_stat_unc : bool):
            ch = {
                "name": channel_name,
                "samples":[
                    {
                        "name" : f"sample_{i}",
                        "data" : s.tolist(),
                        "modifiers" : [
                            {'name': f"mu_{i}", 'type': 'normfactor', 'data': None},
                            # {'name': "normsys", 'type': "normsys", 'data' : {'lo' : 0.8, 'hi' : 1.2}}
                            ]
                    }
                for i, s in enumerate(samples)
                ]
            }
            if mc_stat_unc == True:
                for i in range(len(samples)):
                    ch["samples"][i]["modifiers"].append({"name" : f"{channel_name}_stat_err", "type" : "staterror", "data" : np.sqrt(np.sum(samples, 0)).astype(int).tolist()})
                    # ch["samples"][i]["modifiers"].append({"name" : f"{channel_name}_stat_err", "type" : "shapesys", "data" : (quadsum(np.sqrt(samples), 0)).astype(int).tolist()})
                    # ch["samples"][i]["modifiers"].append({'name': f"{channel_name}_sample_{i}_pois_err", 'type': 'shapesys', 'data': np.array(samples[i]).astype(int).tolist()})
            return ch

        spec = {"channels" : [channel(f"channel_{n}", KE_int_templates[n], mc_stat_unc) for n in range(n_channels)]}

        if mean_track_score_templates is not None:
            spec["channels"] += [channel("mean_track_score", mean_track_score_templates, mc_stat_unc)]
        
        model = pyhf.Model(spec, poi_name = "mu_0")
        return model

    @staticmethod
    def PrintModelSpecs(model : pyhf.Model):
        print(f"  channels: {model.config.channels}")
        print(f"     nbins: {model.config.channel_nbins}")
        print(f"   samples: {model.config.samples}")
        print(f" modifiers: {model.config.modifiers}")
        print(f"parameters: {model.config.parameters}")
        print(f"  nauxdata: {model.config.nauxdata}")
        print(f"   auxdata: {model.config.auxdata}")

    @staticmethod
    def GenerateObservations(
            fit_input : AnalysisInput,
            energy_slices : Slicing.Slices,
            mean_track_score_bins : np.ndarray,
            model : pyhf.Model,
            verbose : bool = True,
            single_bin : bool = False) -> np.ndarray:
        data = RegionFit.CreateObservedInputData(fit_input, energy_slices, mean_track_score_bins, single_bin)
        if verbose is True: print(f"{model.config.suggested_init()=}")
        observations = np.concatenate(data + [model.config.auxdata])
        if verbose is True: print(f"{model.logpdf(pars=model.config.suggested_init(), data=observations)=}")
        return observations

    @staticmethod
    def Fit(observations, model : pyhf.Model, init_params : list[float] = None, par_bounds : list[tuple] = None, verbose : bool = True, tolerance : float = 1E-2, fix_pars : list[bool] = None) -> FitResults:
        pyhf.set_backend(backend = "numpy", custom_optimizer = "minuit")
        if verbose is True: print(f"{init_params=}")
        result = cabinetry.fit.fit(model, observations, init_pars = init_params, custom_fit = True, tolerance = tolerance, par_bounds = par_bounds, fix_pars = fix_pars)

        poi_ind = [model.config.par_slice(i).start for i in model.config.par_names if "mu" in i]
        if verbose is True: print(f"{poi_ind=}")
        parameter = [i for i in model.config.par_names if "mu" in i]
        bestfit = result.bestfit[poi_ind]
        uncertainty = result.uncertainty[poi_ind]

        if verbose is True: print(f"{parameter=}")
        if verbose is True: print(f"{bestfit=}")
        if verbose is True: print(f"{uncertainty=}")
        if verbose is True: print(f"{result=}")
        return result

    @staticmethod
    def GetPredictedCorrelationMatrix(model : pyhf.Model, mu : np.ndarray) -> np.ndarray:
        counts_matrix = []
        for channel in model.spec["channels"]:
            counts = []
            for sample in channel["samples"]:
                counts.append(sum(sample["data"]))
            counts_matrix.append(counts * mu)
        counts_matrix = np.array(counts_matrix).T
        counts_matrix = np.array(counts_matrix, dtype = int)
        return counts_matrix

    @staticmethod
    def CreateKEIntTemplates(
            analysis_input : AnalysisInput,
            energy_slices : Slicing.Slices,
            single_bin : bool = False,
            pad : bool = False,
            reco : bool = True) -> list[np.ndarray]:
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

    @staticmethod
    def CreateMeanTrackScoreTemplates(analysis_input : AnalysisInput, bins : np.ndarray, weights : np.ndarray = None) -> np.ndarray:
        templates = []
        for t in analysis_input.exclusive_process:
            mask = analysis_input.exclusive_process[t]
            templates.append(np.histogram(analysis_input.mean_track_score[mask], bins, weights = weights[mask] if weights is not None else weights)[0])
        return np.array(templates)

    @staticmethod
    def CreateModel(
            template : AnalysisInput,
            energy_slice : Slicing.Slices,
            mean_track_score_bins : np.ndarray,
            return_templates : bool = False,
            weights : np.ndarray = None,
            mc_stat_unc : bool = True,
            pad : bool = True,
            single_bin : bool = False) -> pyhf.Model:
        templates_energy = RegionFit.CreateKEIntTemplates(template, energy_slice, single_bin, pad)
        if mean_track_score_bins is not None:
            templates_mean_track_score = RegionFit.CreateMeanTrackScoreTemplates(template, mean_track_score_bins, weights)
        else:
            templates_mean_track_score = None
        model = RegionFit.Model(len(template.regions), templates_energy, templates_mean_track_score, mc_stat_unc = mc_stat_unc)
        RegionFit.PrintModelSpecs(model)
        if return_templates is True:
            return model, templates_energy, templates_mean_track_score
        else:
            return model

    @staticmethod
    def CreateObservedInputData(
            fit_input : AnalysisInput,
            slices : Slicing.Slices,
            mean_track_score_bins : np.ndarray = None,
            single_bin : bool = False) -> np.ndarray:
        observed_binned = []
        if fit_input.inclusive_process is None:
            mask = np.ones_like(fit_input.KE_int_reco, dtype = bool)
        else:
            mask = fit_input.inclusive_process
        for v in fit_input.regions.values():
            n_int = fit_input.NInteract(slices, v & mask, reco = True)
            if single_bin:
                n_int = [sum(n_int)]
            observed_binned.append(n_int)
        if mean_track_score_bins is not None:
            observed_binned.append(np.histogram(fit_input.mean_track_score[fit_input.inclusive_process], mean_track_score_bins)[0])
        return observed_binned

    @staticmethod
    def SliceModelPrediction(prediction : cabinetry.model_utils.ModelPrediction, slice : slice, label : str) -> cabinetry.model_utils.ModelPrediction:
        return cabinetry.model_utils.ModelPrediction(prediction.model, np.array(prediction.model_yields[slice]), np.array(prediction.total_stdev_model_bins[slice]), np.array(prediction.total_stdev_model_channels[slice]), label)

    @staticmethod
    def PlotPrefitPostFit(prefit, prefit_err, postfit, postfit_err, energy_bins, xlabel = "$KE_{int}$ (MeV)"):
        with Plots.RatioPlot(energy_bins[::-1], postfit, prefit, postfit_err, prefit_err, xlabel, "fit/ true") as ratio_plot:
            Plots.Plot(ratio_plot.x, ratio_plot.y2, yerr = ratio_plot.y2_err, color = "C0", label = "true", style = "step", newFigure = False)
            Plots.Plot(ratio_plot.x, ratio_plot.y1, yerr = ratio_plot.y1_err, color = "C6", label = "fit", style = "step", ylabel = "Counts", newFigure = False)

    @staticmethod
    def EstimateCounts(postfit_pred : cabinetry.model_utils.ModelPrediction) -> tuple[np.ndarray, np.ndarray]:
        if any([c["name"] == "mean_track_score" for c in postfit_pred.model.spec["channels"]]):
            KE_int_prediction = RegionFit.SliceModelPrediction(postfit_pred, slice(-1), "KE_int_postfit") # exclude the channel which is the mean track score
        else:
            KE_int_prediction = RegionFit.SliceModelPrediction(postfit_pred, slice(0, len(postfit_pred.model_yields)), "KE_int_postfit")

        L = KE_int_prediction.model_yields

        L_err = KE_int_prediction.total_stdev_model_bins[:, :-1] # last entry in the array is the total error for the whole channel (but we want the total error in each process)

        return L, L_err

    @staticmethod
    def EstimateBackgroundAllRegions(postfit_pred : cabinetry.model_utils.ModelPrediction, template : AnalysisInput, signal_process : str) -> tuple[np.ndarray, np.ndarray]:
        N, N_err = RegionFit.EstimateCounts(postfit_pred)
        N = np.sum(N, 0)
        N_err = quadsum(N_err, 0)

        labels = list(template.regions.keys()) #! make property of AnalysisInput dataclass
        N_bkg_err = N_err[signal_process != np.array(labels)]
        N_bkg = N[signal_process != np.array(labels)]

        return N_bkg, N_bkg_err

    @staticmethod
    def EstimateBackgroundInRegions(postfit_pred : cabinetry.model_utils.ModelPrediction, data : AnalysisInput) -> tuple[dict, dict]:
        N, N_err = RegionFit.EstimateCounts(postfit_pred)
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


class Unfold:
    @staticmethod
    def CorrelationMarix(observed : np.ndarray, true : np.ndarray, bins : np.ndarray, remove_overflow : bool = True) -> np.ndarray:
        """ Caclulate Correlation matrix of observed and true parameters.

        Args:
            observed (np.ndarray): observed data (reco).
            true (np.ndarray): true data (truth).
            bins (np.ndarray): bins.
            remove_overflow (bool, optional): remove the first bins which are interpreted as overflow. Defaults to True.

        Returns:
            np.ndarray: Correlation matrix.
        """
        corr = np.histogram2d(np.array(observed), np.array(true), bins = bins)[0]
        if remove_overflow is True:
            corr = corr[1:, 1:]
        return corr

    @staticmethod
    def ResponseMatrix(observed : np.ndarray, true : np.ndarray, bins : np.ndarray, efficiencies : np.ndarray = None, remove_overflow : bool = False) -> tuple[np.ndarray, np.ndarray]:
        """ Caclulate Correlation matrix of observed and true parameters.

        Args:
            observed (np.ndarray): observed data (reco).
            true (np.ndarray): true data (truth).
            bins (np.ndarray): bins.
            efficiencies (np.ndarray, optional): selection efficiency. Defaults to None.
            remove_overflow (bool, optional): remove the first bins which are interpreted as overflow. Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray]: response matrix and the statistical error in the response matrix
        """
        if efficiencies is None:
            efficiencies = np.ones(len(bins) - 1 - int(remove_overflow))

        response_hist = Unfold.CorrelationMarix(observed, true, bins, remove_overflow)
        response_hist_err = np.sqrt(response_hist)

        column_sums = response_hist.sum(axis=0)
        if remove_overflow is True:
            normalization_factor = nandiv(efficiencies[1:], column_sums)
        else:
            normalization_factor = nandiv(efficiencies, column_sums)
        response = response_hist * normalization_factor
        response_err = response_hist_err * normalization_factor
        
        response = np.nan_to_num(response)
        response_err = np.nan_to_num(response_err)

        return response, response_err

    @staticmethod #? move to Plots?
    def PlotMatrix(
            matrix : np.ndarray,
            energy_slices : Slicing.Slices,
            title : str = None,
            c_label : str = None,
            text : bool = False,
            text_colour = "k",
            cmap = "plasma"):
        """ Plot numpy matrix.

        Args:
            matrix (np.ndarray): matrix
            title (str, optional): plot title. Defaults to None.
            c_label (str, optional): colourbar label. Defaults to None.
        """
        x = energy_slices.pos_overflow - energy_slices.width/2
        #* cause = true, effect = reco
        Plots.plt.figure()
        Plots.plt.imshow(np.flip(matrix), origin = "lower", cmap = cmap, vmin = np.nanmin(matrix), vmax = np.nanmax(matrix))
        Plots.plt.xlabel("True $KE$ (MeV)")
        Plots.plt.ylabel("Reco $KE$ (MeV)")
        Plots.plt.grid(False)
        Plots.plt.colorbar(label = c_label)
        Plots.plt.title(title, pad = 10)
        Plots.plt.xticks(np.linspace(0, len(x) - 1, len(x)), np.array(x[::-1], dtype = int), rotation = 30)
        Plots.plt.yticks(np.linspace(0, len(x) - 1, len(x)), np.array(x[::-1], dtype = int), rotation = 30)
        Plots.plt.tight_layout(pad = 1)

        if text:
            for (i, j), z in np.ndenumerate(np.flip(matrix)):
                Plots.plt.gca().text(j, i, f"{z:.1g}", ha='center', va='center', fontsize = 10, color = text_colour)
        return

    @staticmethod
    def CalculateResponseMatrices(
            template : AnalysisInput,
            process : str,
            energy_slice : Slicing.Slices,
            regions : bool = False,
            book : Plots.PlotBook = None,
            efficiencies : dict[np.ndarray] = None) -> dict[np.ndarray]:
        """ Calculate response matrix of energy histograms from analysis input.

        Args:
            template (AnalysisInput): template analysis input
            process (str): exclusive process
            energy_slice (Slicing.Slices): energy slices
            book (Plots.PlotBook, optional): plot book. Defaults to None.
            efficiencies (dict[np.ndarray], optional): selection efficiencies. Defaults to None.

        Returns:
            dict[np.ndarray]: response matrices with errors for each histogram
        """
        slice_bins = np.arange(-1 - 0.5, energy_slice.max_num + 1.5)

        outside_tpc_mask = template.outside_tpc_reco | template.outside_tpc_true

        true_slices = Slicing.EnergySlice.SliceNumbers(
            template.KE_int_true, template.KE_init_true,
            outside_tpc_mask, energy_slice)
        reco_slices = Slicing.EnergySlice.SliceNumbers(
            template.KE_int_reco, template.KE_init_reco,
            outside_tpc_mask, energy_slice)

        if regions:
            channel = {i : (template.exclusive_process[i])[~outside_tpc_mask] for i in template.regions}
        else:
            channel = template.exclusive_process[process][~outside_tpc_mask]

        slice_pairs = {
            "init" : [reco_slices[0], true_slices[0]],
            "int" : [reco_slices[1], true_slices[1]]
        }

        if regions:
            for i in channel:
                slice_pairs[i] = [reco_slices[1][channel[i]], true_slices[1][channel[i]]]
        else:
            slice_pairs["int_ex"] = [reco_slices[1][channel], true_slices[1][channel]]

        corr = {}
        resp = {}

        labels = {"init" : "$N_{init}$", "int" : "$N_{int}$", "int_ex" : "$N_{int, ex}$", "absorption": "$N_{int,abs}$", "charge_exchange" : "$N_{int,cex}$", "single_pion_production" : "$N_{int,spip}$", "pion_production" : "$N_{int,pip}$"}


        for k, v in slice_pairs.items():
            corr[k] = Unfold.CorrelationMarix(*v, bins = slice_bins, remove_overflow = False)
            resp[k] = Unfold.ResponseMatrix(*v, bins = slice_bins, efficiencies = None if efficiencies is None else efficiencies[k][0], remove_overflow = False)

        if book is not None:
            for k in resp:
                Unfold.PlotMatrix(corr[k], energy_slice, title = f"Response marix: {labels[k]}", c_label = "Counts")
                book.Save()
                Unfold.PlotMatrix(resp[k][0], energy_slice, title = f"Normalised response matrix: {labels[k]}", c_label = "$P(E_{i}|C_{j})$")
                book.Save()

        return resp

    @staticmethod
    def Unfold(observed : dict[np.ndarray], observed_err : dict[np.ndarray], response_matrices : dict[np.ndarray], priors : dict[np.ndarray] = None, ts_stop = 0.01, max_iter = 100, ts = "ks", regularizers : dict[UnivariateSpline] = None, verbose : bool = False, efficiencies : dict[np.ndarray] = None, covariance : str = "multinomial") -> dict[dict]:
        """ Run iterative bayesian unfolding for each histogram.

        Args:
            observed (dict[np.ndarray]): observed data
            observed_err (dict[np.ndarray]): observed data error
            response_matrices (dict[np.ndarray]): repsonse matrices
            priors (dict[np.ndarray], optional): pior distributions. Defaults to None.
            ts_stop (float, optional): tolerance of test statistic. Defaults to 0.01.
            max_iter (int, optional): maximum number of iterations. Defaults to 100.
            ts (str, optional): test statistic type. Defaults to "ks".
            regularizers (dict[UnivariateSpline], optional): splines to regularise the priors. Defaults to None.
            verbose (bool, optional): verbose printout. Defaults to False.
            efficiencies (dict[np.ndarray], optional): selection efficiencies. Defaults to None.

        Returns:
            dict: unfolding results for each histogram.
        """
        def make_cb(key):
            cb = []
            if verbose: cb.append(Logger())
            if regularizers is not None:
                cb = cb + [regularizers[key]]
            return cb
        results = {}

        for k, v in response_matrices.items():
            if verbose: print(k)
            n = observed[k]
            n_e = observed_err[k]


            cb = make_cb(k)

            if efficiencies is not None:
                efficiency = efficiencies[k][0]
                efficiency_err = efficiencies[k][1]
            else:
                efficiency = np.ones_like(n, dtype = float) #! for the toy, assume perfect selection efficiency, so 1 +- 0
                efficiency_err = np.zeros_like(n, dtype = float) #? not exactly zero, make this very small so the systematic uncertainty from the response matrix can still be calculated?

            if priors is None:
                p = n/sum(n)
            else:
                p = priors[k] / sum(priors[k])

            results[k] = iterative_unfold(n, n_e, v[0], v[1], efficiency, efficiency_err, callbacks = cb, prior = p, ts_stopping = ts_stop, max_iter = max_iter, ts = ts, cov_type = covariance)
        return results

    @staticmethod
    def PlotUnfoldingResults(
            obs : np.ndarray,
            obs_err : np.ndarray,
            true : np.ndarray,
            results : dict,
            energy_slices : Slicing.Slices,
            title : str,
            book : Plots.PlotBook = Plots.PlotBook.null):
        """ Plot unfolded histogram in comparison to observed and true.

        Args:
            obs (np.ndarray): observation
            true (np.ndarray): truth
            results (dict): unfolding results
            energy_bins (np.ndarray): energy bins
            label (str): x label (units of MeV are automatically applied)
            book (Plots.PlotBook, optional): plot book. Defaults to Plots.PlotBook.null.
        """
        if "num_iterations" in results:
            label = f"Data unfolded, {results['num_iterations']} iterations"
        else:
            label = "Data unfolded"

        PlotXSHists(energy_slices, obs, obs_err, True, 1/sum(obs), label = "Data reco", ylabel = "Fractional counts", color = "k")
        PlotXSHists(energy_slices, true, None, True, 1/sum(true), label = "MC true (initial prior)", ylabel = "Fractional counts", color = "C1", newFigure = False)
        PlotXSHists(energy_slices, results["unfolded"], results["stat_err"], True, 1 / sum(results["unfolded"]), label =  label, color = "C4", ylabel = "Fractional counts", newFigure = False, title = title)
        Plots.plt.legend(loc = "upper left")
        book.Save() 
        if "unfolding_matrix" in results:
            Unfold.PlotMatrix(results["unfolding_matrix"], energy_slices, title = "Unfolded matrix: " + label, c_label = "$P(C_{j}|E_{i})$", text = True, text_colour = "red")
            book.Save()

            Unfold.PlotMatrix((results["unfolding_matrix"].T * obs).T, energy_slices, "Migrations : " + label, c_label = "Counts")
            book.Save()

        if "covariance_matrix" in results:
            Unfold.PlotMatrix(results["covariance_matrix"], energy_slices, title = "Covariance matrix: " + label, c_label = "Counts")
            book.Save()

            Unfold.PlotMatrix(np.corrcoef(results["covariance_matrix"]), energy_slices, title = "Correlation matrix: " + label, c_label = "Correlation coefficient", text = True, text_colour = "k", cmap = "coolwarm")
            book.Save()

        return