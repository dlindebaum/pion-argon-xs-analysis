# import sys
# import os
# from importlib import reload
# import copy
import warnings
import numpy as np
from iminuit import cost, Minuit
# import tensorflow as tf
# import tensorflow_gnn as tfgnn
# from tensorflow_gnn.models import gat_v2
# from tensorflow import keras
# import sklearn.ensemble as ensemble
# import sklearn.impute as impute
# import awkward as ak
# import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.colors as plt_colours
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
from python.analysis import Plots
# from python.analysis import EventSelection, Plots, vector, PairSelection, Master, PFOSelection, cross_section, CutOptimization
# from python.analysis import SelectionEvaluation as seval
from python.gnn import Models
# from apps import photon_pairs
import scipy.stats as scistats
import scipy.optimize as opt
# import time
# import timeit

class FitterBase():
    def plot_templates(self):
        fig, axes = self.plt_cfg.setup_figure(2, 2, figsize=(20, 12), title="Templates")
        for i, temp in enumerate(self.labels):
            ax = axes[1-(i//2), i%2]
            ax.set_title(f"{temp} template")
            for j, lab in enumerate(self.labels):
                lw = 3 + 2*(temp == lab)
                ax.hist(
                    self.bins[lab][:-1],
                    **self.plt_cfg.gen_kwargs(
                        type="hist", index=j, bins=self.bins[lab],
                        weights=self.templates[lab][i], label=lab, lw=lw))
            self.plt_cfg.format_axis(ax, xlabel="GNN score", ylabel = "Density")
        return self.plt_cfg.end_plot()

class FitterUncorr(FitterBase):
    def __init__(
            self,
            template_pred,
            template_truth,
            n_bins=5,
            labels=["Abs.", "CEx.", "1 pi", "Multi."],
            plot_config=None,
            data_mins=None,
            data_maxs=None,
            eps=1e-5):
        self.labels = labels
        if plot_config is None:
            self.plt_cfg = Plots.PlotConfig()
            self.plt_cfg.SHOW_PLOT = True
            self.plt_cfg.SAVE_FOLDER = None
        else:
            self.plt_cfg = plot_config
        self.templates = {}
        template_inds = np.argmax(template_pred, axis=1)
        region_masks = Models._get_region_masks(
            template_inds, template_truth, 4)
        true_region_masks = np.apply_along_axis(np.any, 0, region_masks)
        self.bins = {}
        mins = np.min(template_pred, axis=0)
        maxs = np.max(template_pred, axis=0)
        if data_mins is not None:
            mins = np.min([mins, data_mins], axis=0)
        if data_maxs is not None:
            maxs = np.max([maxs, data_maxs], axis=0)
        for i, lab in enumerate(self.labels):
            self.bins[lab] = np.linspace(mins[i], maxs[i] + eps, n_bins+1)
            these_hists = []
            for t, _ in enumerate(self.labels):
                these_hists.append(
                    np.histogram(
                        template_pred[:, i][true_region_masks[t]],
                        bins=self.bins[lab])[0])
            self.templates[lab] = these_hists
        return

    def fit_to_data(
            self,
            data_preds, data_truth=None,
            printout=True, pull_out=False):
        show_truth = data_truth is not None
        if show_truth:
            true_data_counts = np.unique(data_truth, return_counts=True)[1]
        init_preds = np.unique(
            np.argmax(data_preds, axis=1), return_counts=True)[1]
        funcs = []
        for i, lab in enumerate(self.labels):
            hist = np.histogram(data_preds[:, i], bins=self.bins[lab])[0]
            this_func = cost.Template(
                hist, #n
                self.bins[lab], # xe = bin edge locations
                self.templates[lab], # templates
                name=self.labels)
            funcs.append(this_func)
        comb_cost = sum(funcs)
        fitter = Minuit(comb_cost, *tuple(init_preds))
        fitter.limits = (0, data_preds.shape[0])
        fitter.migrad()
        fitter.hesse()
        if printout:
            print(f"Initial count prediction:\t{init_preds}")
            if show_truth:
                print(f"True data counts:\t\t{true_data_counts}")
            print(f"Fitted data counts:\t\t{np.array(fitter.values)}")
            print(f"Fitted data errors:\t\t{np.array(fitter.errors)}")
        if pull_out:
            return np.array([
                true_data_counts,
                np.array(fitter.values),
                np.array(fitter.errors)])[..., np.newaxis]
        else:
            return fitter
    
class FitterCorr(FitterBase):
    def __init__(
            self,
            template_pred,
            template_truth,
            bins=3,
            labels=["Abs.", "CEx.", "1 pi", "Multi."],
            plot_config=None,
            data_mins=None,
            data_maxs=None,
            weights=None):
        self.labels = labels
        if plot_config is None:
            self.plt_cfg = Plots.PlotConfig()
            self.plt_cfg.SHOW_PLOT = True
            self.plt_cfg.SAVE_FOLDER = None
        else:
            self.plt_cfg = plot_config
        self.templates = {}
        template_inds = np.argmax(template_pred, axis=1)
        region_masks = Models._get_region_masks(
            template_inds, template_truth, 4)
        true_region_masks = np.apply_along_axis(np.any, 0, region_masks)
        self.bins = np.histogramdd(template_pred, bins=bins)[1]
        if weights is not None:
            weights = np.repeat(weights[:, np.newaxis], len(labels), axis=1)
        for i, lab in enumerate(self.labels):
            if weights is not None:
                w_arg = weights[true_region_masks[i], :]
            else:
                w_arg = None
            self.templates[lab] = np.histogramdd(
                template_pred[true_region_masks[i], :],
                bins=self.bins, weights=w_arg)[0]
        return

    def fit_to_data(
            self,
            data_preds, data_truth=None,
            printout=True, pull_out=False):
        show_truth = data_truth is not None
        if show_truth:
            true_data_counts = np.unique(data_truth, return_counts=True)[1]
        init_preds = np.unique(np.argmax(data_preds, axis=1), return_counts=True)[1]
        cost_func = cost.Template(
            np.histogramdd(data_preds, bins=self.bins)[0], #n
            self.bins, # xe = bin edge locations
            list(self.templates.values()), # templates
            name=self.labels)
        fitter = Minuit(cost_func, *tuple(init_preds))
        fitter.limits = (0, data_preds.shape[0])
        fitter.migrad()
        fitter.hesse()
        if printout:
            print(f"Initial count prediction:\t{init_preds}")
            if show_truth:
                print(f"True data counts:\t\t{true_data_counts}")
            print(f"Fitted data counts:\t\t{np.array(fitter.values)}")
            print(f"Fitted data errors:\t\t{np.array(fitter.errors)}")
        if pull_out:
            return np.array([
                true_data_counts,
                np.array(fitter.values),
                np.array(fitter.errors)])[..., np.newaxis]
        else:
            return fitter
        
    def plot_templates(self):
        fig, axes = self.plt_cfg.setup_figure(2, 2, figsize=(20, 12), title="Templates")
        for i, temp in enumerate(self.labels):
            ax = axes[1-(i//2), i%2]
            ax.set_title(f"{temp} template")
            for j, lab in enumerate(self.labels):
                lw = 3 + 2*(temp == lab)
                sum_axes = tuple(k for k in range(len(self.labels)) if k != j)
                ax.hist(
                    self.bins[j][:-1],
                    **self.plt_cfg.gen_kwargs(
                        type="hist", index=j, bins=self.bins[j],
                        weights=self.templates[temp].sum(axis=sum_axes), label=lab, lw=lw))
            self.plt_cfg.format_axis(ax, xlabel="GNN score", ylabel = "Density")
        return self.plt_cfg.end_plot()

class DummyDist():
    def __init__(self, data):
        self.data = data
        return
    def rvs(self):
        return self.data

class ReferenceDummy():
    def __init__(self, generator, which, multiplier):
        self.gen = generator
        self.multiplier = multiplier
        if which =="data":
            self.which_attr = "data_hist"
        elif which == "template":
            self.which_attr = "templates"
        else:
            raise ValueError(f"Unknown information: {which}")
    def rvs(self):
        return self.multiplier * getattr(self.gen, self.which_attr)

class ReferenceBinom():
    def __init__(self, generator, which, p):
        self.gen = generator
        self.p = p
        if which =="data":
            self.which_attr = "data_hist"
        elif which == "template":
            self.which_attr = "templates"
        else:
            raise ValueError(f"Unknown information: {which}")
    def rvs(self):
        return scistats.binom.rvs(getattr(self.gen, self.which_attr), self.p)

class DummyMixer():
    def __init__(self, template, data):
        self.template = template
        self.data = data
    def rvs(self):
        return self.template, self.data

class BinomMixer():
    def __init__(self, joint, template_frac):
        self.joint = joint
        self.t_sampler = scistats.binom(self.joint, template_frac)
    def rvs(self):
        sample = self.t_sampler.rvs()
        return sample, joint-sample
    
class AllSmearer():
    def __init__(self, grads, axes, n_iters=1, no_loss=True):
        self.no_loss = no_loss
        self.ax_funcs = [self._create_axis_applier(g) for g in grads]
        self.axes = axes
        self.iters = n_iters

    def _create_axis_applier(self, ax_grad):
        deficit = np.clip(1 - np.abs(ax_grad), 0., 2.)
        if self.no_loss:
            # Remove the deficit if end gradient leaves the bins
            deficit[0] -= max(-1, min(ax_grad[0], 0))
            deficit[-1] += min(1, max(ax_grad[-1], 0))
        pos_move = np.clip(ax_grad, 0., 1.)[:-1]
        neg_move = np.abs(np.clip(ax_grad, -1., 0.)[1:])
        def apply_grad_along_axis(values):
            res = values * deficit
            res[:-1] += values[1:] * neg_move
            res[1:] += values[:-1] * pos_move
            return res
        return apply_grad_along_axis

    def __call__(self, values):
        res = values
        for _ in range(self.iters):
            for ind, ax in enumerate(self.axes):
                res = np.apply_along_axis(self.ax_funcs[ind], ax, res)
        return res

class AllSmearerUncorr(AllSmearer):
    def __init__(
            self,
            grads, axes,
            n_iters=1, no_loss=True,
            regions=["Abs.", "CEx.", "1 pi", "Multi."]):
        super().__init__(grads, axes, n_iters=n_iters, no_loss=no_loss)
        self.regions = regions
    
    def __call__(self, values):
        res = values
        for _ in range(self.iters):
            for ind, ax in enumerate(self.axes):
                lab = self.regions[ax]
                res[lab] = np.apply_along_axis(
                    self.ax_funcs[ind], -1, res[lab])
        return res

class RegionSmearer(AllSmearer):
    def __init__(self, grads, regions, axes, n_iters=1, no_loss=True):
        """
        Create a RegionSmearer instance which suffles bin values around
        as specified by grads, given the region and score axis.

        Parameters
        ----------
        grads : list [ list [ array-like ] ]
            See notes for details on formatting.
        regions : list [ int ]
            See notes for details on formatting.
        axes : list [ list [ int ] ]
            See notes for details on formatting.
        n_iters : int, optional
            Number of iterations of gradient application. Does not
            attenuate the gradients. Default is 1.
        no_loss : bool, optional
            If True, enforce that the end bins cannot lose events, i.e.
            the lowest/highest bin cannot have a negative/positive
            gradient respectively. Default is True

        Notes
        -----
        `grads` and `axes` should both be a list of lists, of matching
        lengths.
        The outer axis corresponds to the region in which the smearing
        should be applied. This is referenced by `regions`, and as such
        must have the same length.
        The second layer of lists can vary in length, as long as
        `grads` and `axes` agree. This corresponds to the score
        distribution over which the smearing, defined by the
        corresponding value in axes.
        """
        self.no_loss = no_loss
        self.ax_funcs = [[self._create_axis_applier(g) for g in axs]
                         for axs in grads]
        self.regions = regions
        self.axes = axes
        self.iters = n_iters

    def __call__(self, values):
        res = values
        for _ in range(self.iters):
            for r_ind, reg in enumerate(self.regions):
                for a_ind, ax in enumerate(self.axes[r_ind]):
                    res[reg] = np.apply_along_axis(
                        self.ax_funcs[r_ind][a_ind], ax-1, res[reg])
        #         -1 to account for missing first axis /\
        return res

class RegionSmearerUncorr(RegionSmearer):
    def __init__(
            self,
            grads, axes,
            n_iters=1, no_loss=True,
            regions=["Abs.", "CEx.", "1 pi", "Multi."]):
        super().__init__(grads, axes, n_iters=n_iters, no_loss=no_loss)
    
    def __call__(self, values):
        res = values
        for _ in range(self.iters):
            for r_ind, reg in enumerate(self.regions):
                for a_ind, ax in enumerate(self.axes[r_ind]):
                    res[reg] = np.apply_along_axis(
                        self.ax_funcs[r_ind][a_ind], -1, res[reg])
        #         -1 to account for missing first axis /\
        return res

class DistGenCorr():
    """
    Class to hold template and data GNN score information, and generate
    samples.

    Attributes
    ----------
    all_smear : class, class variable
        Class to use for smearing over all regions.
    region_smear : class, class variable
        Class to use for region-based smearing.
    correlated : bool
        Indicates if the distributions are correlated.
    labels : list of str
        Labels for the different regions.
    n_scores : int
        Number of scores (regions).
    bin_edges : list of ndarray
        Edges of the GNN score distribution bins.
    n_bins : int
        Number of bins (equal for all score channels).
    templates : np.ndarray
        Histogram of GNN scores for templates.
    has_data_truth : bool
        Indicates if data truth is provided.
    data_hist : np.ndarray
        Histogram of GNN scores for data.
    has_extra_dim : bool
        Indicates if instance contains an additional
    _mixed : bool
        Indicates if the files have been mixed.

    Methods
    -------
    __repr__():
        Returns a string representation of the class.
    _zero_safe_gamma(scales):
        Generates psuedo-gamma-distributed random variables where 0 is
        a valid parameter return a Dirac delta function.
    _make_region_hists(preds, truth):
        Generates the data/template histograms.
    _get_param_case(value):
        Determines the format of the supplied parameter w.r.t. stored
        information.
    _get_grads_case(value):
        Determine the format of the smearing gradients w.r.t. stored
        information.
    _reshape_to_bins(arr):
        Reshapes the array to allow the array to be broadcast onto the
        stored information along the region axis.
    gen_binned_from_funcs(funcs, base_value=1., op="mul"):
        Generates weights over GNN scores from a list of functions.
    join_region_bin_values(region, bins):
        Join a 1D array accross regions with an array over GNN score
        bins into a single array.
    bin_func_decorator(limits=None, min_x=0, max_x=None):
        Decorates a function to format output to bin weights.
    _shape_to_info(param, is_data, extra_dim=None):
        Shapes the parameter to work as a drawing or weighting param.
    _generate_counter_extra(which_info, p_draw=1., extra_dim=None, di stribute_counts=True, reference=False):
    Generates a counter with extra dimensions.
    generate_counter(which_info, p_draw=1., distribute_counts=True, reference=False):
    Generates a counter for pulling counts from a histogram.
    _generate_weighter_extra(which_info, expect_weights, extra_dim=None, distribute_weights=False):
    Generates a weighter with extra dimensions.
    generate_weighter(which_info, expect_weights, distribute_weights=False):
    Generates a weighter for weighting bin counts.
    _drop_none_and_get_axes(lst):
    Drops None values and gets the corresponding axes.
    _match_axis_list(vals, axis):
    Matches values to the corresponding axes.
    _convert_axis_array_to_lists(vals, axis, extra_dim):
    Converts an axis array to lists.
    _convert_list_axis_to_lists(vals, axis, extra_dim):
    Converts a list of axis arrays to lists.
    _convert_truth_array_to_lists(vals, axis, extra_dim):
    Converts a truth array to lists.
    _convert_list_truth_to_lists(vals, axis, extra_dim):
    Converts a list of truth arrays to lists.
    _get_extra_dim(is_data, case):
        Gets the extra dimension based on the data and case.
    generate_smearer(which_info, bin_values, axis=None, is_gradient=True, iterations=1, no_loss=True):
        Generates a smearer for the GNN scores.
    mix_files(template_frac=0.5, distribute=True):
        Shuffles around which sample a given event occurs in.
    restore_file_mixing():
        Restores mixed files to the intially defined separation.
    set_data_sample_params(count_drawer=None, weighting_func=None, smearer=None):
        Sets the parameters for data sample generation.
    sample_data(return_truth=False):
        Return a sample taken from the data.
    set_template_sample_params(count_drawer=None, weighting_func=None, smearer=None):
        Sets the parameters for template sample generation.
    _arr_to_list(arr):
        Convert the outer dimension of an array into a list.
    sample_template(return_truth=False):
        Return a sample taken from the data, formatted to work for
        MINUIT.
    sample_template_like_data(return_truth=False):
        Return a sample taken from the data, formatted identically to a
        data sample.
    """

    all_smear = AllSmearer
    region_smear = RegionSmearer
    def __init__(
            self,
            template_preds, template_truth,
            data_preds, data_truth=None,
            template_extra=None, data_extra=None,
            bins=6, fix_bin_range=None, extra_bins=5,
            labels=["Abs.", "CEx.", "1 pi", "Multi."]):
        self.correlated = True
        self.labels = labels
        self.n_scores = len(self.labels)
        if fix_bin_range is not None:
            bins = ([np.linspace(fix_bin_range[0], fix_bin_range[1], bins+1)]
                    * self.n_scores)
        self.bin_edges = np.histogramdd(
            np.concatenate([template_preds, data_preds], axis=0),
            bins=bins)[1]
        self.n_bins = self.bin_edges[0].size - 1
        self._mixed=False
        self.has_data_truth = data_truth is not None
        self.has_extra_dim = ((template_extra is not None)
                              or (data_extra is not None))
        if self.has_extra_dim:
            extra_info = np.concatenate(
                [info for info in [template_extra, data_extra]
                 if info is not None])
            self.extra_bin_edges = np.histogram(
                extra_info, bins=extra_bins)[1]
            self.extra_n_bins = self.extra_bin_edges.size - 1
            if template_extra is not None:
                temp_extra_n_bins = self.extra_n_bins
                digitized_temp = np.digitize(
                    template_extra, self.extra_bin_edges[1:])
            else:
                temp_extra_n_bins = 1
                digitized_temp = np.zeros(template_preds.shape[0])
            if data_extra is not None:
                data_extra_n_bins = self.extra_n_bins
                digitized_data = np.digitize(
                    data_extra, self.extra_bin_edges[1:])
            else:
                data_extra_n_bins = 1
                digitized_data = np.zeros(data_preds.shape[0])
            self.templates = self._make_region_hists_extra(
                template_preds, template_truth,
                digitized_temp, temp_extra_n_bins)
            self.data_hist = self._make_region_hists_extra(
                data_preds, data_truth,#=None if not self.has_data_truth
                digitized_data, data_extra_n_bins)
        else:
            self.templates = self._make_region_hists(
                template_preds, template_truth)
            if self.has_data_truth:
                self.data_hist = self._make_region_hists(data_preds, data_truth)
            else:
                self.data_hist = np.histogramdd(
                    data_preds, bins=self.bin_edges)[0].astype(int)
        self._calc_sampling_axes()
        self.set_template_sample_params()
        self.set_data_sample_params()
        return

    def __repr__(self):
        return f"DistGenCorr(bins={self.n_bins})"

    def _zero_safe_gamma(self, scales):
        mask = scales > 0.
        res = np.zeros_like(scales, dtype=float)
        res[mask] = scistats.gamma.rvs(scales[mask])
        return res

    def _make_region_hists(self, preds, truth):
        hists = []
        for i, _ in enumerate(self.labels):
            hists.append(np.histogramdd(
                preds[truth == i, :],
                bins=self.bin_edges)[0].astype(int))
        return np.array(hists)

    def _make_region_hists_extra(
            self,
            preds, truth,
            extra_digitized,
            n_extra_bins):
        extra_dimmed = []
        for extra_ind in range(self.extra_n_bins):
            extra_mask = extra_digitized == extra_ind
            if truth is not None:
                filt_pred = preds[extra_mask]
                filt_true = truth[extra_mask]
                extra_dimmed.append(
                    self._make_region_hists(filt_pred, filt_true))
            else:
                extra_dimmed.append(
                    np.histogramdd(
                        preds[extra_mask],
                        bins=self.bin_edges)[0].astype(int))
        return np.array(extra_dimmed)

    def _get_param_case(self, value):
        is_arr = isinstance(value, np.ndarray)
        if is_arr:
            has_truth = ((len(value.shape) == 1)
                         or (len(value.shape) == (1 + self.n_scores)))
        else:
            has_truth = False
        true_shape = (self.n_scores,)
        correlate_factor = self.n_scores if self.correlated else 1
        bins_shape = (1,) + (self.n_bins,) * correlate_factor
        both_shape = ((self.n_scores,)
                      + (self.n_bins,) * correlate_factor)
        if not is_arr:
            if (value is None) or (value == 1.):
                cas = "identity"
            else:
                cas = "number"
        else:
            v_shape = value.shape
            if v_shape == true_shape:
                cas = "truth_array"
            elif v_shape == bins_shape:
                cas = "bins_array"
            elif v_shape == both_shape:
                cas = "joint_array"
            else:
                raise ValueError(
                    "Array type inputs must have one of the following "
                    + f"shapes: {true_shape}, {bins_shape}, {both_shape}. "
                    + f"Found: {v_shape}")
            if np.all(value == np.ones_like(value)):
                cas = "identity"
        return cas, has_truth

    def _get_grads_case(self, value):
        """
        Looks for one of 5 allow gradient situations:
        1. A single 1D array with one value per 1D binning
        2. A single 2D array with the front axis refering the truth
            region, the second with one value per 1D binning
        3. A list of arrays like (1.), for each GNN score
        4. A list of arrays like (2.) for each GNN score
        5. A list of lists like (3.) for each truth region

        These are returned as:
        1. "axis_array"
        2. "truth_array"
        3. "list_axis"
        4. "list_truth"
        5. "truth_list_axis"
        """
        is_arr = isinstance(value, np.ndarray)
        is_list = isinstance(value, list)
        true_arr_shape = (self.n_scores, len(self.bin_edges[0]) - 1)
        ax_arr_shape = (len(self.bin_edges[0]) - 1,)
        def check_list_arrs(v):
            if isinstance(v, list):
                return "list"
            elif v is None:
                return "empty"
            elif isinstance(v, np.ndarray):
                if v.shape == true_arr_shape:
                    return "truth"
                elif v.shape == ax_arr_shape:
                    return "axis"
                else:
                    return "bad"
            else:
                return "bad"
        def check_whole_list(l):
            if l is None:
                return "empty"
            if len(l) != self.n_scores:
                return "bad"
            types = set([check_list_arrs(v) for v in l])
            if not (len(types) == 1
                    or ((len(types) == 2) and ("empty" in types))):
                return "bad"
            end = ""
            if "list" in types:
                types = set([check_whole_list(v) for v in l])
                if not (len(types) == 1
                        or ((len(types) == 2) and ("empty" in types))):
                    return "bad"
                end = "_list"
            if "truth" in types:
                return "truth" + end
            elif "axis" in types:
                return "axis" + end
            elif "empty" in types:
                return "empty" + end
            else:
                return "bad"
        if isinstance(value, np.ndarray):
            if value.shape == true_arr_shape:
                cas = "truth_array"
                has_truth = True
            elif value.shape == ax_arr_shape:
                cas = "axis_array"
                has_truth = False
            else:
                cas = None
        elif isinstance(value, list):
            list_type = check_whole_list(value)
            if list_type == "truth":
                cas = "list_truth"
                has_truth = True
            elif list_type == "axis":
                cas = "list_axis"
                has_truth = False
            elif list_type == "axis_list":
                cas = "truth_list_axis"
                has_truth = True
            else:
                cas = None
        else:
            cas = None
        if cas is None:
            raise ValueError(
                "Unrecognised input type. Following types allowed:\n"
                + f"Numpy array, shape {true_arr_shape} or {ax_arr_shape}.\n"
                + f"List: [{self.n_scores} arrays, shape {ax_arr_shape}], "
                + f"or a list of {self.n_scores} of the previous lists")
            if np.all(value == np.ones_like(value)):
                cas = "identity"
        return cas, has_truth
    
    def _reshape_to_bins(self, arr):
        return np.reshape(arr, arr.shape + (1,)*self.n_scores)
    
    def gen_binned_from_funcs(self, funcs, base_value=1., op="mul"):
        x = np.arange(self.n_bins)
        if len(funcs) != self.n_scores:
            raise ValueError("Must past a list of one function (or None)"
                             + f" per GNN score ({self.n_scores})")
        # Case checking should already ensure the right shape
        weights = np.full((self.n_bins,)*self.n_scores, base_value)
        for i, func in enumerate(funcs):
            if func is None:
                this_weight = np.ones(self.n_bins)
            else:
                this_weight = func(x)
            shape = tuple(n_bins if j == i else 1
                          for j in range(self.n_scores))
            this_weight = np.reshape(this_weight, shape)
            if op == "mul":
                weights = weights * this_weight
            elif op == "add":
                weights = weights + this_weight
            else:
                raise ValueError(
                    f'Unknown operation: {op}. Should be "mul" or "add".')
        return weights
    
    def join_region_bin_values(self, region, bins):
        if isinstance(region, np.ndarray):
            return self._reshape_to_bins(region) * bins[np.newaxis, ...]
        else:
            return region * bins

    def bin_func_decorator(
            self,
            limits=None,
            min_x=0, max_x=None):
        """
        Formats a function which returns a weighting as a function of
        bin index.

        `min_x` and `max_x` specify where along the supplied function
        the bin indicies are taken from. If not supplied, these are
        integers [0, n_bins).

        Specifying limits as a tuple of a low and upper limit forces
        the region of interest to be limited to the range supplied.
        Note this will cause an error if the derivative of the function
        is 0 over the range of interest.

        Parameters
        ----------
        limits : tuple
            Tuple of two values which are used to scale the range of
            the returned function.
        min_x : float, optional
            x value corresponding to bin 0. Default is 0.
        max_x : float, optional
            x value corresponding to the final bin. If not set, simply
            count 1 for each bin. Default is None.
        
        Returns
        -------
        callable
            Decorator for a bin weighting function.
        """
        if max_x is None:
            def new_x(bin_inds):
                return bin_inds + min_x
        else:
            def new_x(bin_inds):
                return np.linspace(min_x, max_x, bin_inds.size)
        if limits is None:
            def decorator(func):
                def wrapped_func(bin_inds):
                    x = new_x(bin_inds)
                    return func(x)
                return wrapped_func
        else:
            def decorator(func):
                def wrapped_func(bin_inds):
                    x = new_x(bin_inds)
                    res = func(x)
                    return min(limits) + (res - np.min(res))*np.abs(
                        (limits[1] - limits[0])/(np.max(res)-np.min(res)))
                return wrapped_func
        return decorator

    def _shape_to_info(self, param, is_data, extra_dim=None):
        case, truth = self._get_param_case(param)
        if truth and is_data and not self.has_data_truth:
            raise ValueError("This instance does not contain data truth.")
        if case == "identity":
            shaped = 1
        elif case == "truth_array":
            shaped = self._reshape_to_bins(param)
        else: # "number", "joint_array", or "bins_array"
            shaped = param
        if extra_dim is not None:
            extra_dim = np.reshape(
                extra_dim,        # Don't include truth dim if
                (extra_dim.shape  # \/  is_data and no data_truth
                 + (1,)*(int(not (is_data and (not self.has_data_truth)))
                         +self.n_scores)))
            return extra_dim * shaped
        else:
            return shaped

    def generate_counter(
            self,
            which_info,
            p_draw=1.,
            extra_dim_factors=None,
            distribute_counts=True,
            reference=False
        ):
        """
        Define the method of pulling counts from a correlated histogram
        of binned GNN scores.

        This results should be passed to `set_data_sample_params` or
        `set_template_sample_params` as the `count_drawer`.

        Parameters
        ----------
        which_info : {"data", "template"}
            Which information to base the sampler from.

        p_used : float or np.ndarray, optional
            Binomial probability that any given event is included. If
            passed as an array with 4 elements, each element defines
            the selection probability for events in the corresponding
            true region. If not passed, all elements are selected.
            Default is None.
        
        extra_dim_factors : np.ndarray, optional
            Factors to be applied to the extra dimension of information
            added during the creation of the instance. The draw
            probabilities generated the standard way are multiplied by
            the corresponding factor in this array for each bin in the
            extra dimension. If None, each extra dimension bin is
            treated equally. Default is None.

        distribute_counts : bool, optional
            If true, counts are binomially distributed with N as the
            raw data/template counts in that bin, and p as the
            corresponding value set in p_draw. Default is True.
        
        reference : bool, optional
            If True, reference the in this instance each time when
            generating the samples. This is slightly slower, but allows
            for mixing which information forms the templates/data
            between samples. If False, a snapshot of the state at the
            time of this function call is used. Default is False.
        
        Returns
        -------
        scipy.stats.rv_discrete or DummyDist
            Object with the `rvs` method which generates a random
            sample of the information as specified by the arguments of
            this function.
        """
        if which_info == "data":
            raw = self.data_hist
            is_data = True
        elif which_info == "template":
            raw = self.templates
            is_data = False
        else:
            raise ValueError(f'Unknown info type {which_info}. '
                             + 'Must be one of "data", "template".')
        if (extra_dim_factors is not None) and (not self.has_extra_dim):
            raise AttributeError(
                "Cannot use extra dimension information in an "
                + "instance with self.has_extra_dim==False")
        # p_case, p_truth = self._get_param_case(p_draw)
        # if p_truth and is_data and not self.has_data_truth:
        #     raise ValueError("This instance does not contain data truth.")
        # if p_case == "identity":
        #     count_ps = 1
        #     distribute_counts = False
        #     if distribute_counts:
        #         warnings.warn("Not distributing counts with p_draw = 1.")
        # elif p_case == "truth_array":
        #     count_ps = self._reshape_to_bins(p_draw)
        # else: # "number", "joint_array", or "bins_array"
        #     count_ps = p_draw
        count_ps = np.clip(
            self._shape_to_info(
                p_draw, is_data, extra_dim=extra_dim_factors),
            0., 1.)
        if distribute_counts:
            if reference:
                return ReferenceBinom(self, which_info, count_ps)
            else:
                return scistats.binom(raw, count_ps)
        else:
            if reference:
                return ReferenceDummy(self, which_info, count_ps)
            else:
                return DummyDist(raw * count_ps)
    
    def _generate_weighter_extra(
            self,
            which_info,
            expect_weights,
            extra_dim=None,
            distribute_weights=False):
        """
        Back-end version of weight generation which includes the
        ability to cosider an extra dimension.
        """
        if which_info == "data":
            is_data = True
        elif which_info == "template":
            is_data = False
        else:
            raise ValueError(f'Unknown info type {which_info}. '
                             + 'Must be one of "data", "template".')
        weights = self._shape_to_info(expect_weights, is_data,
                                      extra_dim=extra_dim)
        if distribute_weights:
            return lambda counts: self._zero_safe_gamma(
                counts * weights)
        else:
            return lambda counts: counts * weights

    def generate_weighter(
            self,
            which_info,
            expect_weights,
            distribute_weights=False,
            extra_dim_factors=None):
        """
        Define the method of weighting bin counts from a correlated
        histogram of binned GNN scores.

        This results should be passed to `set_data_sample_params` or
        `set_template_sample_params` as the `weighting_func`.

        Parameters
        ----------
        which_info : {"data", "template"}
            Which information to base the sampler from. Note: only used
            for validating the weight shape.

        region_weight_expect : float or np.ndarray, optional
            Weight applied to each event, if an array containing four
            elements indicating the weight given to each sample, given
            it is from the corresponding true region. Default is 1.

        distribute_weights : bool, optional
            If True, weights are distributed around the expected region
            weight by a chi-squared distribution with 3 d.o.f. Default
            is False.
        
        Returns
        -------
        callable
            Function which takes an array of counts and returns
            counts weighted as specified by the arguments of this
            generator function.
        """
        return self._generate_weighter_extra(
            which_info,
            expect_weights,
            extra_dim=extra_dim_factors,
            distribute_weights=distribute_weights)

    def _drop_none_and_get_axes(self, lst):
        if len(lst) != self.n_scores:
            raise Exception("Can't match bin_values to scores")
        vals = []
        axes = []
        for i, v in enumerate(lst):
            if v is not None:
                axes.append(i)
                vals.append(v)
        return vals, axes

    def _match_axis_list(self, vals, axis):
        if axis is None:
            return list(range(self.n_scores))
        elif isinstance(axis, int):
            return [axis]
        elif len(vals) != self.n_scores:
            if len(vals) == len(axis):
                return axis
            else:
                raise Exception("Cannot match bin_values to axes")
        else:
            return list(range(self.n_scores))

    def _convert_axis_array_to_lists(self, vals, axis, extra_dim):
        if axis is None:
            grads = [vals] * self.n_scores
            axes = [a + extra_dim for a in range(self.n_scores)]
        else:
            try:
                grads = [vals] * len(axis)
                axes = [a + extra_dim for a in axis]
            except TypeError:
                grads = [vals]
                axes = [axis + extra_dim]
        return grads, axes

    def _convert_list_axis_to_lists(self, vals, axis, extra_dim):
        axes = self._match_axis_list(vals, axis)
        grads = []
        new_axes = []
        for ind, ax in enumerate(axes):
            g, a = self._convert_axis_array_to_lists(
                vals[ind], ax, extra_dim)
            grads += g
            new_axes += a
        return grads, new_axes
    
    def _convert_truth_array_to_lists(self, vals, axis, extra_dim):
        r_axes = list(range(self.n_scores))
        s_axes = []
        grads = []
        for v in vals:
            gs, axs = self._convert_axis_array_to_lists(
                v, axis, extra_dim)
            s_axes.append(axs)
            grads.append(gs)
        return grads, r_axes, s_axes

    def _convert_list_truth_to_lists(self, vals, axis, extra_dim):
        axes = self._match_axis_list(vals, axis)
        r_axes = list(range(self.n_scores))
        s_axes = [[]]*self.n_scores
        grads = [[]]*self.n_scores
        for ind, ax in enumerate(axes):
            gs, _, axs = self._convert_truth_array_to_lists(
                vals[ind], ax, extra_dim)
            for r in r_axes:
                grads[r] += gs[r]
                s_axes[r] += axs[r]
        return grads, r_axes, s_axes

    def _get_extra_dim(self, is_data, case):
        return int((not is_data) or (self.has_data_truth))

    def generate_smearer(
            self,
            which_info,
            bin_values,
            axis=None,
            is_gradient=True,
            iterations=1,
            no_loss=True):
        if self.has_extra_dim:
            raise NotImplementedError(
                "Smearing not available with extra dimensions")
        if which_info == "data":
            is_data = True
        elif which_info == "template":
            is_data = False
        else:
            raise ValueError(f'Unknown info type {which_info}. '
                             + 'Must be one of "data", "template".')
        vals_case, vals_truth = self._get_grads_case(bin_values)
        if vals_truth and is_data and not self.has_data_truth:
            raise ValueError("This instance does not contain data truth.")
        extra_dim = self._get_extra_dim(is_data, vals_case)
        if vals_case == "axis_array":
            # Single array of gradients, applied axes spcified by axis
            if not is_gradient:
                bin_values = np.gradient(bin_values)
            grads, axes = self._convert_axis_array_to_lists(
                bin_values, axis, extra_dim)
            # Reference the class, since corr vs. uncorr have
            #   different smearers
            return type(self).all_smear(
                grads, axes, no_loss=no_loss, n_iters=iterations)
        elif vals_case == "list_axis":
            # List of single arrays to be apply over each score
            new_vals, new_axs = self._drop_none_and_get_axes(bin_values)
            if not is_gradient:
                new_vals = [np.gradient(v) for v in new_vals]
            grads, axes = self._convert_list_axis_to_lists(
                new_vals, new_axs, extra_dim)
            return type(self).all_smear(
                grads, axes, no_loss=no_loss, n_iters=iterations)
        elif vals_case == "truth_list_axis":
            # List of lists, smears for each truth region for each axis
            if axis is not None:
                warnings.warn("Ignoring axis for list [list] type values")
            vals, r_axes = self._drop_none_and_get_axes(bin_values)
            # r_axes: list of regions for which gradients exist
            s_axes = []
            grads = []
            for i, s_vals in enumerate(vals):
                s_val, s_ax = self._drop_none_and_get_axes(s_vals)
                if not is_gradient:
                    s_val = [np.gradient(v) for v in s_val]
                gs, axs = self._convert_list_axis_to_lists(
                    s_val, s_ax, extra_dim)
                grads.append(gs)
                s_axes.append(axs)
            return type(self).region_smear(
                grads, r_axes, s_axes, no_loss=no_loss, n_iters=iterations)
        elif vals_case == "truth_array":
            # An array with an axis-0 entry for each truth region
            if not is_gradient:
                bin_values = np.gradient(bin_values, axis=1)
            grads, r_axes, s_axes = self._convert_truth_array_to_lists(
                bin_values, axis, extra_dim)
            return type(self).region_smear(
                grads, r_axes, s_axes, no_loss=no_loss, n_iters=iterations)
        elif vals_case == "list_truth":
            # A list of array for each score, array specify true region
            new_vals, new_axs = self._drop_none_and_get_axes(bin_values)
            if not is_gradient:
                bin_values = np.gradient(bin_values, axis=1)
            grads, r_axes, s_axes = self._convert_list_truth_to_lists(
                new_vals, new_axs, extra_dim)
            return type(self).region_smear(
                grads, r_axes, s_axes, no_loss=no_loss, n_iters=iterations)  

    def mix_files(self, template_frac=0.5, distribute=True):
        if not self.has_data_truth:
            raise Exception("Cannot mix without data truth")
        self._mixed = True
        self._original_data = np.copy(self.data_hist)
        self._original_template = np.copy(self.templates)
        joint_info = self.data_hist + self.templates
        if distribute:
            self.templates = scistats.binom.rvs(joint_info, template_frac)
            self.data_hist = joint_info - self.templates
        else:
            self.templates = np.astype(joint_info * template_frac, int)
            self.data_hist = np.astype(joint_info - self.templates, int)
        return

    def restore_file_mixing(self):
        if self._mixed:
            self.data_hist = np.copy(self._original_data)
            self.templates = np.copy(self._original_template)
            self._mixed=False
        return

    def set_data_sample_params(
            self,
            count_drawer=None,
            weighting_func=None,
            smearer=None):
        """
        Define the method of generating data which the is used when
        calling `sample_data`.

        Parameters
        ----------
        count_drawer : scipy.stats.rv_discrete-like, optional
            If passed, must be an instance with a `rvs` method. This
            will be used to draw counts from the data bins. Note there
            is no checking the data source. Ensure you generate this
            instance using the `generate_counter` method with
            `which_info` as "data". If None, simply use the stored data
            counts. Default is None.

        weighting_func : callable, optional
            If passed, must be an callable function which takes a set
            of counts in bins and returns a weighted count in the bins.
            The `generate_weighter` method will generate such
            functions. If None, use a function which returns the
            inputs. Default is None.

        distribute_weights : bool, optional
            If true, weights are distributed around the expected region
            weight by a chi-squared distribution with 3 d.o.f. Default
            is False.
        """
        if count_drawer is None:
            self.data_counter = DummyDist(self.data_hist)
        else:
            self.data_counter = count_drawer
        if weighting_func is None:
            self.data_weighter = lambda counts: counts
        else:
            self.data_weighter = weighting_func
        if smearer is None:
            self.data_smearer = lambda counts: counts
        else:
            if self.has_extra_dim:
                raise NotImplementedError(
                    "Smearing not available with extra dimensions")
            self.data_smearer = smearer
        return
        
    def _calc_sampling_axes(self):
        self.ex_dim_sum_ax = int(self.has_extra_dim)*(0,)
        self.ds_sum_ax = int(self.has_data_truth)*(int(self.has_extra_dim),)
        self.s_truth_sum_axs = tuple(range(
            1+int(self.has_extra_dim),
            1+int(self.has_extra_dim)+len(self.labels)))
        self.s_extra_sum_axs = tuple(range(
            1+int(self.has_extra_dim),
            1+int(self.has_extra_dim)+len(self.labels)))
        return

    def sample_data(
            self,
            return_truth=False,
            return_extra=False, split_extra=False):
        sample = self.data_smearer(
            self.data_weighter(
                self.data_counter.rvs()))
        res = sample.sum(
            axis=(int(not split_extra)*self.ex_dim_sum_ax
                  + self.ds_sum_ax))
        if not (return_truth or return_extra):
            return res
        rets = ()
        if return_truth:
            if not self.has_data_truth:
                raise Exception(
                    "Requested truth, but no data truth in instance.")
            rets += (sample.sum(axis=(self.ex_dim_sum_ax
                                      + self.s_truth_sum_axs)),)
        if return_extra:
            rets += (sample.sum(axis=self.s_extra_sum_axs),)
        return (res, *rets)

    def set_template_sample_params(
            self,
            count_drawer=None,
            weighting_func=None,
            smearer=None):
        """
        Define the method of generating data which the is used when
        calling `sample_template`.

        Parameters
        ----------
        count_drawer : scipy.stats.rv_discrete-like, optional
            If passed, must be an instance with a `rvs` method. This
            will be used to draw counts from the data bins. Note there
            is no checking the data source. Ensure you generate this
            instance using the `generate_counter` method with
            `which_info` as "template". If None, simply use the stored
            template counts. Default is None.

        weighting_func : callable, optional
            If passed, must be an callable function which takes a set
            of counts in bins and returns a weighted count in the bins.
            The `generate_weighter` method will generate such
            functions. If None, use a function which returns the
            inputs. Default is None.
        """
        if count_drawer is None:
            self.temp_counter = DummyDist(self.templates)
        else:
            self.temp_counter = count_drawer
        if weighting_func is None:
            self.temp_weighter = lambda counts: counts
        else:
            self.temp_weighter = weighting_func
        if smearer is None:
            self.temp_smearer = lambda counts: counts
        else:
            if self.has_extra_dim:
                raise NotImplementedError(
                    "Smearing not available with extra dimensions")
            self.temp_smearer = smearer
        return
    
    def _arr_to_list(self, arr):
        return [arr[i] for i in range(arr.shape[0])]

    def sample_template(
            self,
            return_truth=False,
            return_extra=False, split_extra=False):
        sample = self.temp_smearer(
            self.temp_weighter(
                self.temp_counter.rvs()))
        if split_extra:
            if not self.has_extra_dim:
                raise ValueError("Cannot split extras if instance "
                                 + "doesn't contain extra dimension")
            res = self._arr_to_list(
                np.swapaxes(sample, 0, 1))
        else:
            res = self._arr_to_list(sample.sum(axis=self.ex_dim_sum_ax))
        if not (return_truth or return_extra):
            return res
        # This should be editied. Want at most 1 extra array.
        # If return_truth only, then this is (self.n_scores,)
        # If return_extra only, then this is (self.extra_n_bins,)
        # If both, then this is (self.n_scores, self.extra_n_bins)
        rets = ()
        if return_truth:
            rets += (sample.sum(axis=(self.ex_dim_sum_ax
                                      + self.s_truth_sum_axs)),)
        if return_extra:
            rets += (sample.sum(axis=self.s_extra_sum_axs),)
        return (res, *rets)
    
    def sample_template_like_data(
            self,
            return_truth=False,
            return_extra=False, split_extra=False):
        sample = self.temp_smearer(
            self.temp_weighter(
                self.temp_counter.rvs()))
        res = sample.sum(axis=int(not split_extra)*(0,) + (int(self.has_extra_dim),))
        if not (return_truth or return_extra):
            return res
        rets = ()
        if return_truth:
            rets += (sample.sum(axis=self.s_truth_sum_axs),)
        if return_extra:
            rets += (sample.sum(axis=self.s_extra_sum_axs),)
        return (res, *rets)

class DistGenUncorr(DistGenCorr):
    all_smear = AllSmearerUncorr
    region_smear = RegionSmearerUncorr

    def __init__(
            self,
            template_preds, template_truth,
            data_preds, data_truth=None,
            bins=10, fix_bin_range=None,
            labels=["Abs.", "CEx.", "1 pi", "Multi."]):
        super().__init__(
            template_preds, template_truth,
            data_preds, data_truth=data_truth,
            bins=bins, fix_bin_range=fix_bin_range,
            labels=labels)
        self.correlated = False
        return
    
    def __repr__(self):
        return f"DistGenUncorr(bins={self.n_bins})"
    
    def _get_extra_dim(self, is_data, case):
        all_smear = case in ["axis_array", "list_axis"]
        return int(((not all_smear)
                    and ((not is_data) or (self.has_data_truth))))

    def _other_axes(self, ax):
        return tuple(a for a in range(self.n_scores) if a != ax)

    def sample_data(self, return_truth=False):
        sample = self.data_weighter(self.data_counter.rvs())
        if return_truth:
            if not self.has_data_truth:
                raise Exception(
                    "Requested truth, but no data truth in instance.")
            truth = sample.sum(
                axis=tuple(range(1, 1+len(self.labels))))
            sample = sample.sum(axis=0)
        elif self.has_data_truth:
            sample = sample.sum(axis=0)
        dists = {lab: sample.sum(axis=self._other_axes(ax))
                 for ax, lab in enumerate(self.labels)}
        dists = self.data_smearer(dists)
        if return_truth:
            return dists, truth
        return dists

    def sample_template(self, return_truth=False):
        sample = self.temp_weighter(self.temp_counter.rvs())
        if return_truth:
            truth = sample.sum(
                axis=tuple(range(1, 1+len(self.labels))))
        temps = {lab: np.array(
                [s.sum(axis=self._other_axes(ax)) for s in sample])
            for ax, lab in enumerate(self.labels)}
        temps = self.temp_smearer(temps)
        if return_truth:
            return temps, truth
        return temps

class DistGenCorrExtra(DistGenCorr):
    all_smear = AllSmearer
    region_smear = RegionSmearer
    def __init__(
            self,
            template_preds, template_truth,
            data_preds, data_truth=None,
            template_extra=None, data_extra=None,
            bins=6, extra_bins=5, labels=["Abs.", "CEx.", "1 pi", "Multi."]):
        self.correlated = True
        self.labels = labels
        self.n_scores = len(self.labels)
        self.bin_edges = np.histogramdd(
            np.concatenate([template_preds, data_preds], axis=0),
            bins=bins)[1]
        self.n_bins = self.bin_edges[0].size - 1
        self.extra_bin_edges = np.histogram(
            np.concatenate([template_extra, data_extra], axis=0),
            bins=extra_bins)[1]
        self.extra_n_bins = self.extra_bin_edges[0].size - 1
        digitized_temp = np.digitize(template_extra, self.extra_bin_edges[:-1])
        self.templates = self._make_region_hists(
            template_preds, template_truth)
        self.has_data_truth = data_truth is not None
        if self.has_data_truth:
            self.data_hist = self._make_region_hists(data_preds, data_truth)
        else:
            self.data_hist = np.histogramdd(data_preds, bins=self.bin_edges)[0]
        self.set_template_sample_params()
        self.set_data_sample_params()
        self._mixed=False
        return

    def _make_region_hists_extra(self, preds, truth, extra_digitized):
        extra_dimmed = []
        for extra_ind in range(self.extra_n_bins):
            extra_mask = extra_digitized == extra_ind
            filt_pred = preds[extra_mask]
            filt_true = truth[extra_mask]
            extra_dimmed.append(
                self._make_region_hists(filt_pred, filt_true))
        return hists
        # hists = np.array(extra_dimmed)
        # return np.swapaxes(hists, 0, -1)
        
    def generate_counter(
            self,
            which_info,
            p_draw=1.,
            extra_dim_factors=None,
            distribute_counts=True,
            reference=False
        ):
        """
        Define the method of pulling counts from a correlated histogram
        of binned GNN scores.

        This results should be passed to `set_data_sample_params` or
        `set_template_sample_params` as the `count_drawer`.

        Parameters
        ----------
        which_info : {"data", "template"}
            Which information to base the sampler from.

        p_used : float or np.ndarray, optional
            Binomial probability that any given event is included. If
            passed as an array with 4 elements, each element defines
            the selection probability for events in the corresponding
            true region. If not passed, all elements are selected.
            Default is None.
        
        extra_dim_factors : np.ndarray, optional
            Factors to be applied to the extra dimension of information
            added during the creation of the instance. The draw
            probabilities generated the standard way are multiplied by
            the corresponding factor in this array for each bin in the
            extra dimension. If None, each extra dimension bin is
            treated equally. Default is None.

        distribute_counts : bool, optional
            If true, counts are binomially distributed with N as the
            raw data/template counts in that bin, and p as the
            corresponding value set in p_draw. Default is True.
        
        reference : bool, optional
            If True, reference the in this instance each time when
            generating the samples. This is slightly slower, but allows
            for mixing which information forms the templates/data
            between samples. If False, a snapshot of the state at the
            time of this function call is used. Default is False.
        
        Returns
        -------
        scipy.stats.rv_discrete or DummyDist
            Object with the `rvs` method which generates a random
            sample of the information as specified by the arguments of
            this function.
        """
        return self._generate_counter_extra(
            which_info,
            p_draw=p_draw,
            extra_dim=extra_dim_factors,
            distribute_counts=distribute_counts,
            reference=reference)

    def generate_weighter(
            self,
            which_info,
            expect_weights,
            extra_dim_factors=None,
            distribute_weights=False):
        """
        Define the method of weighting bin counts from a correlated
        histogram of binned GNN scores.

        This results should be passed to `set_data_sample_params` or
        `set_template_sample_params` as the `weighting_func`.

        Parameters
        ----------
        which_info : {"data", "template"}
            Which information to base the sampler from. Note: only used
            for validating the weight shape.

        region_weight_expect : float or np.ndarray, optional
            Weight applied to each event, if an array containing four
            elements indicating the weight given to each sample, given
            it is from the corresponding true region. Default is 1.
        
        extra_dim_factors : np.ndarray, optional
            Factors to be applied to the extra dimension of information
            added during the creation of the instance. The weights
            generated the standard way are multiplied by the
            corresponding factor in this array for each bin in the
            extra dimension. If None, each extra dimension bin is
            treated equally. Default is None.

        distribute_weights : bool, optional
            If True, weights are distributed around the expected region
            weight by a chi-squared distribution with 3 d.o.f. Default
            is False.
        
        Returns
        -------
        callable
            Function which takes an array of counts and returns
            counts weighted as specified by the arguments of this
            generator function.
        """
        return self._generate_weighter_extra(
            which_info,
            expect_weights,
            extra_dim=extra_dim_factors,
            distribute_weights=distribute_weights)
    
    def generate_smearer(self, *args, **kwargs):
        raise NotImplementedError(
            "Smearing not implemented for extra dimension type generators.")
    
    def set_data_sample_params(
            self,
            count_drawer=None,
            weighting_func=None):
        return super().set_data_sample_params(
            count_drawer=count_drawer,
            weighting_func=weighting_func,
            smearer=None)
    
    def sample_data(self, return_truth=False, return_extra=False):
        sample = self.data_weighter(self.data_counter.rvs())
        if return_truth:
            if not self.has_data_truth:
                raise Exception(
                    "Requested truth, but no data truth in instance.")
            truth_counts = sample.sum(
                axis=tuple(range(1, 1+len(self.labels))))
            return sample.sum(axis=0), truth_counts
        elif self.has_data_truth:
            return sample.sum(axis=0)
        else:
            return sample

    def set_template_sample_params(
            self,
            count_drawer=None,
            weighting_func=None):
        return super().set_template_sample_params(
            count_drawer=count_drawer,
            weighting_func=weighting_func,
            smearer=None)
    
    def sample_template(self, return_truth=False):
        sample = self.temp_smearer(
            self.temp_weighter(
                self.temp_counter.rvs()))
        if return_truth:
            truth_counts = sample.sum(
                axis=tuple(range(1, 1+len(self.labels))))
            return self._arr_to_list(sample), truth_counts
        else:
            return self._arr_to_list(sample)

def generator_fit(
        generator,
        init_preds=None,
        printout=True,
        pull_out=False,
        mix_template_frac=None,
        **kwargs):
    if pull_out and not generator.has_data_truth:
        raise Exception("generator does not contain truth information")
    if mix_template_frac is not None:
        generator.mix_files(template_frac=mix_template_frac, distribute=True)
    if generator.has_data_truth:
        d_hist, d_truth = generator.sample_data(return_truth=True)
    else:
        d_hist = generator.sample_data(return_truth=False)
    templates = generator.sample_template(return_truth=False)
    n_scores = generator.n_scores
    if generator.correlated:
        n_data = d_hist.sum()
        cost_func = cost.Template(
            d_hist, #n
            generator.bin_edges, # xe = bin edge locations
            templates, # templates
            name=generator.labels)
    else:
        n_data = d_hist[generator.labels[0]].sum()
        score_costs = []
        for i, lab in enumerate(generator.labels):
            score_costs.append(cost.Template(
                d_hist[lab], #n
                generator.bin_edges[i], # xe = bin edge locations
                templates[lab], # templates
                name=generator.labels,
                **kwargs))
        cost_func = sum(score_costs)
    if init_preds is None:
        init_preds = np.full(n_scores, n_data/n_scores)
    fitter = Minuit(cost_func, *tuple(init_preds))
    fitter.limits = (0, n_data)
    fitter.migrad()
    fitter.hesse()
    if printout:
        print(f"Initial count prediction:\t{init_preds}"
              + "\t(using uniform prior from generator)")
        if generator.has_data_truth:
            print(f"True data counts:\t\t{d_truth}")
        print(f"Fitted data counts:\t\t{np.array(fitter.values)}")
        print(f"Fitted data errors:\t\t{np.array(fitter.errors)}")
    if pull_out:
        return np.array([
            d_truth,
            np.array(fitter.values),
            np.array(fitter.errors)])[..., np.newaxis]
    else:
        return fitter

def generate_pulls(n, generator, **kwargs):
    fit_vals = []
    for _ in range(n):
        fit_vals.append(
            generator_fit(
                generator, printout=False,
                pull_out=True, **kwargs))
    return np.concatenate(fit_vals, axis=-1)

def add_ending(save_name, ending):
    if save_name is not None:
        return save_name + "_" + ending + ".png"
    else:
        return None

def make_pulls(
        pull_results,
        plot_config=None,
        return_errors=False,
        save_name=None,
        labels=["Abs.", "CEx.", "1 pi", "Multi."]):
    # Dumb function, shouldn't be so much repeat in return_errors part
    test_stat = (pull_results[1] - pull_results[0]) / pull_results[2]
    if plot_config is not None:
        fig, axes = plot_config.setup_figure(2, 2, figsize=(20, 12))
    pull_fits = []
    for i, lab in enumerate(labels):
        mu, sig = scistats.norm.fit(test_stat[i])
        pull_fits.append([mu, sig])
        if plot_config is not None:
            ax = axes[1-(i//2), i%2]
            ax.set_title(f"{lab} pulls")
            _, bins, _ = ax.hist(test_stat[i], **plot_config.gen_kwargs(
                type="hist", bins="auto", density=True, label="Data"))
            ax.plot(
                bins, scistats.norm.pdf(bins, mu, sig),
                **plot_config.gen_kwargs(
                    index=1,
                    label=f"Gaussian fit:\nmu={mu:.2f}, sigma={sig:.2f}"))
            plot_config.format_axis(
                ax, xlabel="Normalised error", ylabel = "Density")
            ax.legend()
    if plot_config is not None:
        plot_config.end_plot(add_ending(save_name, "pulls"))
    if return_errors:
        if plot_config is not None:
            fig, axes = plot_config.setup_figure(2, 2, figsize=(20, 12))
        fit_errs = []
        for i, lab in enumerate(labels):
            frac_errs = (pull_results[1][i]/pull_results[0][i])-1
            mu_unc, sig_unc = scistats.norm.fit(pull_results[2][i])
            mu_frac, sig_frac = scistats.norm.fit(frac_errs)
            mu_unc_f, sig_unc_f = scistats.norm.fit(
                pull_results[2][i]/pull_results[0][i])
            fit_errs.append([
                [mu_unc, sig_unc],
                [mu_frac, sig_frac],
                [mu_unc_f, sig_unc_f]])
            if plot_config is not None:
                ax = axes[1-(i//2), i%2]
                ax.set_title(f"{lab} errors")
                _, bins, _ = ax.hist(frac_errs, **plot_config.gen_kwargs(
                    type="hist", bins="doane", density=True, label="Data"))
                ax.plot(
                    bins, scistats.norm.pdf(bins, mu_frac, sig_frac),
                    **plot_config.gen_kwargs(
                        index=1,
                        label=(f"Gaussian fit:\nmu={mu_frac:.2f}, "
                               + f"sigma={sig_frac:.2f}")))
                plot_config.format_axis(
                    ax, xlabel="Fractional error", ylabel = "Density")
                ax.legend()
        if plot_config is not None:
            plot_config.end_plot(add_ending(save_name, "frac_err"))
            fig, axes = plot_config.setup_figure(2, 2, figsize=(20, 12))
        for i, lab in enumerate(labels):
            mu_unc, sig_unc = fit_errs[i][0]
            if plot_config is not None:
                ax = axes[1-(i//2), i%2]
                ax.set_title(f"{lab} errors")
                _, bins, _ = ax.hist(pull_results[2][i], **plot_config.gen_kwargs(
                    type="hist", bins="doane", density=True, label="Data"))
                ax.plot(
                    bins, scistats.norm.pdf(bins, mu_unc, sig_unc),
                    **plot_config.gen_kwargs(
                        index=1,
                        label=(f"Gaussian fit:\nmu={mu_unc:.2f}, "
                               + f"sigma={sig_unc:.2f}")))
                plot_config.format_axis(
                    ax, xlabel="MINUIT fit uncertainty", ylabel = "Density")
                ax.legend()
        if plot_config is not None:
            plot_config.end_plot(add_ending(save_name, "fit_uncert"))
        return pull_fits, fit_errs
    return pull_fits

def pull_study(
        n, generator,
        plot_config=None,
        return_errors=False,
        save_name=None,
        **kwargs):
    pulls = generate_pulls(n, generator, **kwargs)
    if np.any(pulls[1, :, :] > 1):
        print(f"Found {np.sum(np.any(pulls[1, :, :] < 1, axis=0))}"
              + " samples with a region prediction <1.")
        good_mask = np.all(pulls[1, :, :] > 1, axis=0)
        if np.any(good_mask):
            print("Plotting full predictions:")
            make_pulls(
                pulls, plot_config=plot_config, labels=generator.labels,
                return_errors=return_errors, save_name=save_name)
            print("Plotting prediction with no 0s:")
            pulls = pulls[:, :, good_mask]
        else:
            print("No good fits found, so using all fits.")
    return make_pulls(
        pulls, plot_config=plot_config, labels=generator.labels,
        return_errors=return_errors, save_name=save_name)

def _sqrt_func(x, a, b, c):
    return a * np.sqrt(x-b) + c

def plot_and_fit_range(
        x, x_lab, n_pulls,
        fits, errs=None,
        fit_sqrt=True, main_ax=None,
        plot_config=None, title=None,
        save_name=None, labels=["Abs.", "CEx.", "1 pi", "Multi."]):
    if plot_config is None:
        plot_config = Plots.PlotConfig()
    fig, axes = plot_config.setup_figure()
    if title is not None:
        fig.suptitle(title)
    for i, line in enumerate(labels):
        plt.plot(x, fits[:, i, 0],
                **plot_config.gen_kwargs(index=i, label=f"{line}"))
        lab = f"Pull \u03C3/{np.sqrt(n_pulls):.1f}" if i==0 else None
        plt.fill_between(
            x,
            fits[:, i, 0]-fits[:, i, 1]/np.sqrt(n_pulls),
            fits[:, i, 0]+fits[:, i, 1]/np.sqrt(n_pulls),
            **plot_config.gen_kwargs(index=i, fc=f"C{i}",
            alpha=0.2, label=lab))
        if fit_sqrt:
            fit_params = opt.curve_fit(
                _sqrt_func, x, fits[:, i, 0],
                p0=np.array([-0.01, 0., 0.]),
                bounds=(np.array([-100, -100, -10]),
                        np.array([100, np.min(x), 10])),
                method="trf", max_nfev=10000)[0]
            plt.plot(x, _sqrt_func(x, *fit_params),
                        **plot_config.gen_kwargs(
                            index=i, ls="--", c=f"C{i}"))
    plot_config.format_axis(xlabel=x_lab, ylabel = "Mean pull")
    plot_config.end_plot(add_ending(save_name, "pull_variation"))
    if errs is not None:
        fig, axes = plot_config.setup_figure()
        if title is not None:
            fig.suptitle(title)
        for i, line in enumerate(labels):
            plt.plot(x, errs[:, i, 0, 0],
                    **plot_config.gen_kwargs(index=i, label=f"{line}"))
            plt.fill_between(
                x,
                errs[:, i, 0, 0]-errs[:, i, 0, 1],
                errs[:, i, 0, 0]+errs[:, i, 0, 1],
                **plot_config.gen_kwargs(index=i, fc=f"C{i}", alpha=0.2))
        plot_config.format_axis(xlabel=x_lab, ylabel = "Uncertainties (yield)")
        plot_config.end_plot(add_ending(save_name, "yield_uncert_var"))
        fig, axes = plot_config.setup_figure()
        if title is not None:
            fig.suptitle(title)
        for i, line in enumerate(labels):
            plt.plot(x, errs[:, i, 2, 0],
                    **plot_config.gen_kwargs(index=i, label=f"{line}"))
            plt.fill_between(
                x,
                errs[:, i, 2, 0]-errs[:, i, 2, 1],
                errs[:, i, 2, 0]+errs[:, i, 2, 1],
                **plot_config.gen_kwargs(index=i, fc=f"C{i}", alpha=0.2))
        plot_config.format_axis(xlabel=x_lab, ylabel = "Uncertainties (fraction of true count)")
        plot_config.end_plot(add_ending(save_name, "frac_uncert_var"))
        fig, axes = plot_config.setup_figure()
        if title is not None:
            fig.suptitle(title)
        for i, line in enumerate(labels):
            plt.plot(x, errs[:, i, 1, 0],
                    **plot_config.gen_kwargs(index=i, label=f"{line}"))
            plt.fill_between(
                x,
                errs[:, i, 1, 0]-errs[:, i, 1, 1],
                errs[:, i, 1, 0]+errs[:, i, 1, 1],
                **plot_config.gen_kwargs(index=i, fc=f"C{i}", alpha=0.2))
        plot_config.format_axis(xlabel=x_lab, ylabel = "Fractional errors")
        plot_config.end_plot(add_ending(save_name, "frac_err_var"))
        fig, axes = plot_config.setup_figure()
        if title is not None:
            fig.suptitle(title)
        for i, line in enumerate(labels):
            plt.plot(x, errs[:, i, 1, 0],
                    **plot_config.gen_kwargs(index=i, label=f"{line}"))
            lab = f"Frac err. \u03C3/{np.sqrt(n_pulls):.1f}" if i==0 else None
            plt.fill_between(
                x,
                errs[:, i, 1, 0]-errs[:, i, 1, 1]/np.sqrt(n_pulls),
                errs[:, i, 1, 0]+errs[:, i, 1, 1]/np.sqrt(n_pulls),
                **plot_config.gen_kwargs(index=i, fc=f"C{i}",
                alpha=0.2, label=lab))
        plot_config.format_axis(xlabel=x_lab, ylabel = "Fractional errors")
        plot_config.end_plot(add_ending(save_name, "frac_err_var_mean"))
    return

def robustness_test(
        generator,
        gen_update,
        changing_params,
        param_name,
        title=None,
        n_pulls=1000,
        plot_config=None,
        fit_sqrt=False,
        plot_missing=True,
        plot_pulls=False,
        save_name=None,
        **kwargs):
    if plot_config is None:
        plot_config = Plots.PlotConfig()
    pull_conf = plot_config if plot_pulls else None
    bad_fracs = []
    fits = []
    errs = []
    for val in changing_params:
        generator = gen_update(generator, val)
        print(f"{param_name} = {val}")
        pulls = generate_pulls(n_pulls, generator, **kwargs)
        bad_pull_mask = np.any(pulls[1, :, :] < 1, axis=0)
        bad_fracs.append(np.sum(bad_pull_mask)/n_pulls)
        fit, err = make_pulls(
            pulls[:, :, np.logical_not(bad_pull_mask)],
            plot_config=pull_conf, return_errors=True,
            save_name=save_name, labels=generator.labels)
        fits.append(fit)
        errs.append(err)
    fits = np.array(fits)
    errs = np.array(errs)
    plot_and_fit_range(
        changing_params, param_name, n_pulls, fits, errs,
        title=title, fit_sqrt=fit_sqrt,
        plot_config=plot_config, save_name=save_name,
        labels=generator.labels)
    if plot_missing:
        plot_config.setup_figure(title=title)
        plt.plot(
            changing_params, np.array(bad_fracs),
            **plot_config.gen_kwargs())
        plot_config.format_axis(
            xlabel=param_name, ylabel = "Frac. fits with 0 preds")
        plot_config.end_plot(add_ending(save_name, "bad_fits"))
    return

def make_rand_sample_update(
        which_info, reference=False,
        distribute_counts=True):
    if which_info =="template":
        def func(generator, p_temp):
            sampler = generator.generate_counter(
                "template", p_draw=p_temp,
                distribute_counts=distribute_counts,
                reference=reference)
            generator.set_template_sample_params(
                count_drawer=sampler)
            return generator
    elif which_info =="data":
        def func(generator, p_temp):
            sampler = generator.generate_counter(
                "data", p_draw=p_temp,
                distribute_counts=distribute_counts,
                reference=reference)
            generator.set_data_sample_params(
                count_drawer=sampler)
            return generator
    else:
        raise Exception(f"Unnkown information {which_info}")
    return func

def make_rand_weight_update(which_info, distribute_weights=False):
    if which_info =="template":
        def func(generator, weight):
            weighter = generator.generate_weighter(
                "template", expect_weights=weight,
                distribute_weights=distribute_weights)
            generator.set_template_sample_params(
                weighting_func=weighter)
            return generator
    elif which_info =="data":
        def func(generator, weight):
            weighter = generator.generate_weighter(
                "data", expect_weights=weight,
                distribute_weights=distribute_weights)
            generator.set_data_sample_params(
                weighting_func=weighter)
            return generator
    else:
        raise Exception(f"Unnkown information {which_info}")
    return func

def make_rand_inverse_update(
        which_info, reference=False,
        distribute_counts=True,
        distribute_weights=False):
    if which_info =="template":
        def func(generator, p_temp):
            sampler = generator.generate_counter(
                "template", p_draw=p_temp,
                distribute_counts=distribute_counts,
                reference=reference)
            weighter = generator.generate_weighter(
                "template", expect_weights=1/p_temp,
                distribute_weights=distribute_weights)
            generator.set_template_sample_params(
                count_drawer=sampler,weighting_func=weighter)
            return generator
    elif which_info =="data":
        def func(generator, p_temp):
            sampler = generator.generate_counter(
                "data", p_draw=p_temp,
                distribute_counts=distribute_counts,
                reference=reference)
            weighter = generator.generate_weighter(
                "data", expect_weights=1/p_temp,
                distribute_weights=distribute_weights)
            generator.set_data_sample_params(
                count_drawer=sampler, weighting_func=weighter)
            return generator
    else:
        raise Exception(f"Unnkown information {which_info}")
    return func

def gen_process_updater(
        baseline_val, process_index, which_info,
        n_regions=4, which_process="sample", reference=True,
        distribute_counts=True, distribute_weights=False):
    '''Which process: "sample", "weight", "inverse"'''
    if which_info not in  ["template", "data"]:
        raise Exception(f"Unknown information {which_info}")
    baseline = np.full(n_regions, baseline_val)
    process = np.zeros(n_regions)
    process[process_index]= 1.
    if which_process == "sample":
        func = make_rand_sample_update(
            which_info, reference=reference,
            distribute_counts=distribute_counts)
    elif which_process == "weight":
        func = make_rand_weight_update(
            which_info, distribute_weights=distribute_weights)
    elif which_process == "inverse":
        func = make_rand_inverse_update(
            which_info, reference=reference,
            distribute_counts=distribute_counts,
            distribute_weights=distribute_weights)
    else:
        raise ValueError('which_process must be one of "sample", "weight", '
                         f'or "inverse". Received {which_process}')
    def sample_process(generator, p_temp):
        return func(generator, baseline*(1 + process*(p_temp-1)))
    return sample_process

def gen_exclusive_extra_dims_updated(
        bins_per_test=1, reference=True):
    def apply_test(generator, bin_ind):
        data_bins = np.zeros(generator.extra_n_bins)
        data_bins[:bins_per_test] = 1.
        data_bins = np.roll(data_bins, bin_ind)
        temp_bins = np.ones(generator.extra_n_bins) - data_bins
        t_sampler = generator.generate_counter(
            "template", extra_dim_factors = data_bins,
            distribute_counts=False, reference=reference)
        d_sampler = generator.generate_counter(
            "data", extra_dim_factors = temp_bins,
            distribute_counts=False, reference=reference)
        generator.set_template_sample_params(count_drawer=t_sampler)
        generator.set_data_sample_params(count_drawer=d_sampler)
        return generator
    return apply_test

# def gen_data_process(baseline_val, process_index, which_process="sample"):
#     '''Which process: "sample", "weight", "inverse"'''
#     baseline = np.full(4, baseline_val)
#     process = np.zeros(4)
#     process[process_index]= 1.
#     if which_process == "sample":
#         func = temp_rand_sample_update
#     elif which_process == "weight":
#         func = temp_fixed_weight_update
#     elif which_process == "inverse":
#         func = temp_rand_sample_inverse_weight_update
#     else:
#         raise ValueError('which_process must be one of "sample", "weight", '
#                          f'or "inverse". Received {which_process}')
#     def temp_sample_process(generator, p_temp):
#         return func(generator, baseline*(1 + process*(p_temp-1)))
#     return temp_sample_process

# class DataGenUncorr():
#     def __init__(
#             self,
#             template_preds, tempalte_truth,
#             data_preds, data_truth=None,
#             n_bins=10, labels=["Abs.", "CEx.", "1 pi", "Multi."]):
#         # Handling data
#         self.bin_edges = {}
#         self.n_temp = template_preds.shape[0]
#         self.n_data = data_preds.shape[0]
#         self.data_locs = {}
#         data_inds = np.arange(data_preds.shape[0])
#         self.template_locs = {}
#         template_inds = np.argmax(template_pred, axis=1)
#         for i, lab in enumerate(labels):
#             # Setup bin edges for the investigate GNN output (lab)
#             self.bin_edges[lab] = np.histogram(
#                 np.concatenate(template_preds[:, i], data_preds[:, i]),
#                 bins=n_bins)[1]
#             # Get the locations each data bin pulls from
#             self.data_locs[lab] = [
#                 data_inds[m] for m in
#                 self._get_reference_locations(
#                     data_preds[:, i], self.bin_edges[lab])]
#             # Get the location each template bin pulls from
#             #   (for each true region)
#             template_masks = self._get_reference_locations(
#                 template_preds[:, i], self.bin_edges[lab])
#             these_temps = {}
#             for j, temp in enumerate(labels):
#                 these_temps[temp] = [
#                     template_inds[np.logical_and(
#                         m, true_region_masks[j])]
#                     for m in template_masks]
#         # Handling truth
#         self.has_data_truth = data_truth is not None
#         if self.has_data_truth:
#             self.data_truth_locs = {}
#         self.temp_truth_locs = {}
#         for i, true in enumerate(labels):
#             if self.has_data_truth:
#                 self.data_truth_locs[true] = data_inds[data_truth == i]
#             self.temp_truth_locs[true] = template_inds[template_truth == i]
            
#     def _get_reference_masks(self, data, bin_edges):
#         # Final bin catches all upper values
#         which_bins = np.digitize(data, bin_edges[:-1])
#         return [which_bins == i for i in range(bin_edges.size - 1)]
    
#     def template_drawing(
#             self,
#             p_used=None,
#             region_weight_expect=np.array([1., 1., 1., 1.]),
#             distribute_weights=False):
#         """
#         Define the method of generating templates which the is used
#         when calling `get_templates`.

#         Parameters
#         ----------
#         p_used : int or np.ndarry, optional
#             Binomial probability that any given event is included. If
#             passed as an array with 4 elements, each element defines
#             the selection probability for events in teh corresponding
#             region. If not passed, all elements are selected. Default
#             is None.

#         region_weight_expect : np.ndarray, optional
#             Array with 4 elements indicating the weight given to each
#             sample, given it is from the corresponding region. Default
#             is `np.array([1., 1., 1., 1.])`.

#         distribute_weights : bool, optional
#             If true, weights are distributed around the expected region
#             weight by a chi-squared distribution with 3 d.o.f. Default
#             is False.
#         """
#         if region_weight_expect is not None:
#             weights = scistats.chi2(
#                 np.full(self.n_temps)[np.newaxis, :],
#                 scale=region_weights[:, np.newaxis])
            
#     def get_templates(self):
#         pass

#     def get_data(self):
#         pass

# class FitterBaseOld():
#     def __init__(
#             self,
#             model,
#             template_path, template_schema,
#             data_path, data_schema,
#             bins=3):
#         template_preds, template_truth = Models.get_predicitions(
#             model, template_schema, template_path)
#         data_preds, data_truth = Models.get_predicitions(
#             model, data_schema, data_path)
        
#         self.n_dims = template_preds.shape[-1]
#         self.bins = bins
#         eps = 1e-5
#         bin_edges = np.zeros((self.bins + 1, self.n_dims))
#         for i in range(self.n_dims):
#             bin_min = min(np.min(template_preds[:, i]),
#                           np.min(data_preds[:, i]))
#             bin_max = max(np.max(template_preds[:, i]),
#                           np.max(data_preds[:, i]))
#             bin_edges[:, i] = np.linspace(bin_min, bin_max+eps,
#                                           num=self.bins+1)
#         self.bin_edges = bin_edges

#         self.template_hists = self.gen_template_hists(template_preds,
#                                                       template_truth)
#         self.template_probs = self.convert_hists_to_probs(self.template_hists)
#         self.data = self.get_data_predictions(data_preds, data_truth)
#         self.n_data = np.sum(self.data)
#         self.init_pred = self.gen_initial_pred(data_preds)
#         return
    
#     def gen_template_hists(self, template_preds, template_truth):
#         template_hists = np.zeros((self.bins,)*self.n_dims + (self.n_dims,),
#                                   dtype=np.int32)
#         pred_index = np.argmax(template_preds, axis=1)
#         region_masks = Models._get_region_masks(
#             pred_index, template_truth, self.n_dims)
#         true_reg_masks = np.apply_along_axis(np.any, 0, region_masks)
#         for i in range(self.n_dims):
#             template_hists[..., i] = self.get_multinomial_hist(
#                 template_preds[true_reg_masks[i]])
#         return template_hists

#     def get_data_predictions(self, data_preds, data_truth=None):
#         self.has_data_truth = data_truth is not None
#         if self.has_data_truth:
#             pred_index = np.argmax(data_preds, axis=1)
#             region_masks = Models._get_region_masks(
#                 pred_index, data_truth, self.n_dims)
#             true_reg_masks = np.apply_along_axis(np.any, 0, region_masks)
#             self.true_data_counts = np.sum(true_reg_masks, axis=-1)
#         data_hist = self.get_multinomial_hist(data_preds)
#         return data_hist

#     def gen_initial_pred(self, data_preds):
#         pred_index = np.where(
#             data_preds == np.max(data_preds, axis=1)[:, np.newaxis])[1]
#         return np.histogram(pred_index, bins=np.arange(self.n_dims+1))[0]

#     def get_multinomial_hist(self, preds):
#         return np.histogramdd(
#             preds, bins=[self.bin_edges[:, i] for i in range(self.n_dims)])[0]
#         # bin_pos = np.zeros(preds.shape[0])
#         # for i in range(self.n_dims):
#         #     bin_pos += (self.bins ** i) * np.digitize(
#         #         preds[:, i], self.bin_edges[:, i])
#         # hist_flat = np.histogram(
#         #     bin_pos,
#         #     np.arange((self.bins ** self.n_dims)+1))[0]
#         # result = np.zeros((self.bins,)*self.n_dims, dtype=np.int32)
#         # for i, count in enumerate(hist_flat):
#         #     ind_list = [i%self.bins]
#         #     ind = ((i//dim) % self.bins for dim in range(self.n_dims))
#         #     result[ind] = count
#         # return np.reshape(hist_flat, (self.bins,)*self.n_dims)
    
#     def convert_hists_to_probs(self, hists):
#         # Current method for dealing with 0-count bins
#         # Worth eploring other methods
#         adjusted_hists = hists + 1
#         return adjusted_hists/np.sum(adjusted_hists,
#                                      axis=tuple(range(self.n_dims)),
#                                      keepdims=True)
    
#     def convert_probs_to_hist(self, x, templates):
#         pred_hist = np.zeros_like(self.data)
#         for i in range(self.n_dims):
#             pred_hist += (x[i]/x.sum()) * templates[..., i]
#         return pred_hist

#     def reshape_forward(self, arr):
#         new_shape = (self.bins ** self.n_dims,) + arr.shape[self.n_dims:]
#         return np.reshape(arr, new_shape)

#     def reshape_backwards(self, arr):
#         new_shape = (self.bins,)*self.n_dims + arr.shape[1:]
#         return np.reshape(arr, new_shape)

#     def get_likelihood_bad(self, pred_hist):
#         L = scistats.multinomial.pmf(self.reshape_forward(self.data),
#                                      self.n_data,
#                                      self.reshape_forward(pred_hist))
#         return - 2 * np.log(L).sum()

#     def get_likelihood(self, data, counts):
#         pass

#     def bad_fit_func(self, x):
#         expected_hist = self.convert_probs_to_hist(x, self.template_hists)
#         expected_probs = self.convert_hists_to_probs(expected_hist)
#         return self.get_likelihood_bad(expected_probs)
    
#     def bad_fit_func_constraint(self):
#         # Constraint ensures we predict the proportion
#         #   of each event (i.e. sum(x) = 1)
#         def eq_func(x):
#             return x.sum() - 1.
#         return {
#             "type":"eq",
#             "fun": eq_func}
    
#     def fit(self):
#         which_fit_func = self.bad_fit_func
#         which_constraints = self.bad_fit_func_constraint()
#         print("Initial prediction (GNN classification counts):")
#         print(self.init_pred)
#         self.fit_result = opt.minimize(
#             which_fit_func, self.init_pred/self.n_data,
#             bounds=opt.Bounds(0, 1), constraints=which_constraints)
#         print("\nFit results")
#         print(self.fit_result.x)
#         print("\nFitted counts")
#         print(self.fit_result.x * self.n_data)
#         if self.has_data_truth:
#             print("\nTrue counts:")
#             print(self.true_data_counts)
#         print("\nFitted likelihood:")
#         print(which_fit_func(self.fit_result.x))
#         if self.has_data_truth:
#             print("Likelihood from true counts:")
#             print(which_fit_func(
#                 self.true_data_counts/self.n_data))
#         return self.fit_result

# class MultinomiallFitter(FitterBase):
#     def gen_template_hists(self, template_preds, template_truth):
#         template_hists = np.zeros((self.bins,)*self.n_dims + (self.n_dims,),
#                                   dtype=np.int32)
#         pred_index = np.argmax(template_preds, axis=1)
#         region_masks = Models._get_region_masks(
#             pred_index, template_truth, self.n_dims)
#         true_reg_masks = np.apply_along_axis(np.any, 0, region_masks)
#         for i in range(self.n_dims):
#             template_hists[..., i] = self.get_multinomial_hist(
#                 template_preds[true_reg_masks[i]])
#         return template_hists

#     def get_data_predictions(self, data_preds, data_truth=None):
#         self.has_data_truth = data_truth is not None
#         if self.has_data_truth:
#             pred_index = np.argmax(data_preds, axis=1)
#             region_masks = Models._get_region_masks(
#                 pred_index, data_truth, self.n_dims)
#             true_reg_masks = np.apply_along_axis(np.any, 0, region_masks)
#             self.true_data_counts = np.sum(true_reg_masks, axis=-1)
#         data_hist = self.get_multinomial_hist(data_preds)
#         return data_hist

#     def gen_initial_pred(self, data_preds):
#         pred_index = np.where(
#             data_preds == np.max(data_preds, axis=1)[:, np.newaxis])[1]
#         return np.histogram(pred_index, bins=np.arange(self.n_dims+1))[0]

#     def get_multinomial_hist(self, preds):
#         return np.histogramdd(
#             preds, bins=[self.bin_edges[:, i] for i in range(self.n_dims)])[0]
#         # bin_pos = np.zeros(preds.shape[0])
#         # for i in range(self.n_dims):
#         #     bin_pos += (self.bins ** i) * np.digitize(
#         #         preds[:, i], self.bin_edges[:, i])
#         # hist_flat = np.histogram(
#         #     bin_pos,
#         #     np.arange((self.bins ** self.n_dims)+1))[0]
#         # result = np.zeros((self.bins,)*self.n_dims, dtype=np.int32)
#         # for i, count in enumerate(hist_flat):
#         #     ind_list = [i%self.bins]
#         #     ind = ((i//dim) % self.bins for dim in range(self.n_dims))
#         #     result[ind] = count
#         # return np.reshape(hist_flat, (self.bins,)*self.n_dims)
    
#     def convert_hists_to_probs(self, hists):
#         # Current method for dealing with 0-count bins
#         # Worth eploring other methods
#         adjusted_hists = hists + 1
#         return adjusted_hists/np.sum(adjusted_hists,
#                                      axis=tuple(range(self.n_dims)),
#                                      keepdims=True)
    
#     def convert_probs_to_hist(self, x, templates):
#         pred_hist = np.zeros_like(self.data)
#         for i in range(self.n_dims):
#             pred_hist += (x[i]/x.sum()) * templates[..., i]
#         return pred_hist

#     def reshape_forward(self, arr):
#         new_shape = (self.bins ** self.n_dims,) + arr.shape[self.n_dims:]
#         return np.reshape(arr, new_shape)

#     def reshape_backwards(self, arr):
#         new_shape = (self.bins,)*self.n_dims + arr.shape[1:]
#         return np.reshape(arr, new_shape)

#     def get_likelihood_bad(self, pred_hist):
#         L = scistats.multinomial.pmf(self.reshape_forward(self.data),
#                                      self.n_data,
#                                      self.reshape_forward(pred_hist))
#         return - 2 * np.log(L).sum()

#     def get_likelihood(self, data, counts):
#         pass

#     def bad_fit_func(self, x):
#         expected_hist = self.convert_probs_to_hist(x, self.template_hists)
#         expected_probs = self.convert_hists_to_probs(expected_hist)
#         return self.get_likelihood_bad(expected_probs)
    
#     def bad_fit_func_constraint(self):
#         # Constraint ensures we predict the proportion
#         #   of each event (i.e. sum(x) = 1)
#         def eq_func(x):
#             return x.sum() - 1.
#         return {
#             "type":"eq",
#             "fun": eq_func}
    
#     def fit(self):
#         which_fit_func = self.bad_fit_func
#         which_constraints = self.bad_fit_func_constraint()
#         print("Initial prediction (GNN classification counts):")
#         print(self.init_pred)
#         self.fit_result = opt.minimize(
#             which_fit_func, self.init_pred/self.n_data,
#             bounds=opt.Bounds(0, 1), constraints=which_constraints)
#         print("\nFit results")
#         print(self.fit_result.x)
#         print("\nFitted counts")
#         print(self.fit_result.x * self.n_data)
#         if self.has_data_truth:
#             print("\nTrue counts:")
#             print(self.true_data_counts)
#         print("\nFitted likelihood:")
#         print(which_fit_func(self.fit_result.x))
#         if self.has_data_truth:
#             print("Likelihood from true counts:")
#             print(which_fit_func(
#                 self.true_data_counts/self.n_data))
#         return self.fit_result

# n_bins=10
# classification_labels=["Abs.", "CEx.", "1 pi", "Multi."]
# predictions, truth_index = Models.get_predicitions(
#     loaded_model, which_path_params["schema_path"], which_path_params["test_path"])
# # pred_index = np.where(
# #     predictions == np.max(predictions, axis=1)[:, np.newaxis])[1]
# pred_index = np.argmax(predictions, axis=1)
# labels = Models._parse_classification_labels(classification_labels,
#                                       pred_index, truth_index)
# n_events = predictions.shape[0]
# n_regions = predictions.shape[-1]
# region_masks = Models._get_region_masks(pred_index, truth_index, n_regions)
# # bins = np.linspace(np.min(predictions), np.max(predictions)+1e-2, 31)
# # true_counts = region_masks.sum(axis=(0, -1))
# # print("True counts:")
# # print(true_counts)

# classification_labels=["Abs.", "CEx.", "1 pi", "Multi."]
# val_predictions, val_truth_index = Models.get_predicitions(
#     loaded_model, which_path_params["schema_path"], which_path_params["val_path"])
# val_pred_index = np.where(
#     val_predictions == np.max(val_predictions, axis=1)[:, np.newaxis])[1]
# val_region_masks = Models._get_region_masks(val_pred_index, val_truth_index, n_regions)


# bins = np.linspace(min(np.min(predictions), np.min(val_predictions)), max(np.max(predictions), np.max(val_predictions))+1e-2, n_bins+1)
# true_counts = val_region_masks.sum(axis=(0, -1))
# print("True counts:")
# print(true_counts)
# def get_hists(preds, bins, n_regions=n_regions, density=False):
#     result = np.zeros((bins.size-1, n_regions))
#     for i in range(n_regions):
#         result[:, i] = np.histogram(preds[:, i], bins, density=density)[0]
#     if density:
#         result *= (bins[1] - bins[0])
#         result = np.clip(result, 1/preds.shape[0], None)
#     return result

# hists = {"full": get_hists(val_predictions, bins)}
# for true_i in range(n_regions):
#     all_reco_mask = region_masks[0, true_i]
#     for i in range(1, n_regions):
#         all_reco_mask = np.logical_or(all_reco_mask, region_masks[i, true_i])
#     hists[classification_labels[true_i]] = get_hists(predictions[all_reco_mask], bins, density=True)

# def get_likelihood(true_counts, expected_rates):
#     L = poisson.pmf(true_counts, expected_rates)
#     return - 2 * np.log(L).sum()

# def make_expected_hist(x):
#     absorb = x[0]
#     cex = x[1]
#     sing = x[2]
#     multi = x[3]
#     return  (absorb * hists["Abs."]
#              + cex * hists["CEx."]
#              + sing * hists["1 pi"]
#              + multi * hists["Multi."])
        
# def fit(x):
#     # expected_hist = np.clip(make_expected_hist(x), sys.float_info.min, None)
#     expected_hist = make_expected_hist(x)
#     return get_likelihood(hists["full"], expected_hist)

# init_pred = val_region_masks.sum(axis=(1, -1))
# print("\nInitial prediction (GNN classification counts):")
# print(init_pred)
# result = opt.minimize(fit, init_pred, bounds = [(0, n_events)] * 4)
# print("\nFitted counts")
# print(result.x)

# print("\nFitted likelihood:")
# print(fit(result.x))
# print("Lieklihood from true counts:")
# print(fit(true_counts))

# plt_conf.setup_figure()
# plt.hist(np.repeat(bins[:-1, np.newaxis], n_regions, axis=1), bins=bins, weights=hists["full"], lw=3, ls = "--", histtype="step", label="True dists")
# plt.hist(np.repeat(bins[:-1, np.newaxis], n_regions, axis=1), bins=bins, weights=make_expected_hist(result.x), lw=4, histtype="step", label="Fitted dists")
# plt_conf.format_axis(xlabel="Score", ylabel = "Count")
# plt_conf.end_plot()
