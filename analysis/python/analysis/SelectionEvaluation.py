# Code for generating plots indicating performance of some selection
# Dennis Lindebaum
# 16.11.23

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from python.analysis import Master, Plots, PairSelection
import itertools

class SimpleCutSelector():
    def __init__(self, **cuts):
        self.known_properties = [key for key in cuts.keys()]
        
        self.applied_cuts = cuts

        self.cuts_dict = {}

        for i, cut in enumerate(cuts.values()):
            self._gen_cut(cut, self.known_properties[i])
        return

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            return self.metric(*args)
        elif len(args) == len(self.known_properties):
            return self.explicit_args(*args)
        elif len(args) != 0:
            raise TypeError(f"{self.__name__}() takes 1 or {len(self.known_properties)} positional arguments")
        else:
            for prop in kwargs.keys():
                if prop not in self.known_properties:
                    raise TypeError(f"{self.__name__}() got an unexpected keyword argument '{prop}'")
            return self.explicit_args(**kwargs)

    def _gen_cut(self, limit, prop_name):
        if limit is None:
            return
        elif isinstance(limit, tuple):
            if len(limit) == 2:
                def cut(prop):
                    return np.logical_and(prop > limit[0], prop < limit[1])
            else:
                def cut(prop):
                    return prop > limit[0]
        else:
            def cut(prop):
                return np.abs(prop) < limit
        self.cuts_dict.update({prop_name:cut})
        return

    def explicit_args(self, **args):
        props_dict = {}
        for i, prop in enumerate(args.values()):
            props_dict.update({self.known_properties[i]:prop})
        return self.metric(props_dict)

    def metric(self, properties_dict):
        passing_pairs = []
        for prop in self.cuts_dict.keys():
            if properties_dict[prop] is not None:
                passing_pairs += [ self.cuts_dict[prop]( properties_dict[prop] ) ]

        # If no cuts specified, return 1s
        if len(passing_pairs) == 0:
            return ak.ones_like(properties_dict[self.known_properties[0]])
        
        # Return 1s for values passing the cut(s), 0s for others
        elif len(passing_pairs) == 1:
            return ak.values_astype(passing_pairs[0], float)
        else:
            current_passing = passing_pairs[0]
            for i in range(1, len(passing_pairs)):
                current_passing = np.logical_and(current_passing, passing_pairs[i])
            return ak.values_astype(current_passing, float) 


class SelectorEvaluator():
    """
    Class to store properties and cuts an generate
    plots indicating cut performance.

    ...
    
    Attributes
    ----------
    properties : dict {str: ak.Array}
        A dictionary containing the properties which
        may be cut on, with an label.
    prop_labels : list
        The list of keys of the `properties` dictionary.
    truth_mask : ak.Array
        A mask which is true for "signal" PFOs, and
        False for backgrounds
    pfo_name : str
        Name of the signal type PFOs to appear in text.
    metrics : dict {str: function}
        A dictionary containing cuts which to select
        signal PFOs.
    
    INCOMPLETE

    Methods
    -------
    get_metric_values
        Returns the values of PFOs under a specified metric.
    get_threshold
        Returns to cut threshold used by a specified metric.
    get_passing
        Returns an boolean mask of PFOs which exceed a
        threshold under a specified metric.
    
    
    """
    def __init__(self, truth, props_dict, pfo_name="signal"):
        """
        Creates a SelectorEvaluator instance.

        Parameters
        ----------
        truth : ak.Array
            Boolean array containing True for signal type PFOs
        props_dict : dict {str: ak.Array}
            Dictionary containing a property label, with an
            array holding the values of that property for each
            PFO.
        """
        self.properties = props_dict
        self.prop_labels = [key for key in self.properties.keys()]
        self.truth_mask = truth

        self.pfo_name = pfo_name

        self.metrics = {}
        self.thresholds = {}
        self.rankings = {}

        self.default_threshold = 0.5

        self.plt_cfg = Plots.PlotConfig()
        self.plt_cfg.BEST_BINS_MULTIPLIER = 2
        self.plt_cfg.SHOW_PLOT = True
        self.plt_cfg.SAVE_FOLDER = None
        self.plotting_style = "density_rejected"

        return
    
    def add_metric(self, name, function, threshold=None):
        """
        Adds the metric generation `function` stored as `name`.

        The function recieves a dictionary containing the
        properties specified as `props_dict` in `__init__`.
        """
        self.metrics.update({name:function})

        if threshold is None:
            threshold = self.default_threshold
        self.thresholds.update({name:threshold})
        return
    
    def add_selector_metric(self, name, **cuts):
        """
        Adds a selector called `name` with the supplied cuts.

        Each supplied cut must be of the form:
        `{property: cut}`
        With `property` being a string indexing an item in
        `self.properties`, and `cut` being one of the following:
            None            : no cut applied
            number          : require `property < number`
            (number,)        : require `property > number`
            (num1, num2)    : require `num1 < property < num2`
        """
        full_cuts = {p : None for p in self.prop_labels}
        full_cuts.update(cuts)
        self.metrics.update({name:SimpleCutSelector(**full_cuts)})

        self.thresholds.update({name:0.5})
        return

    def get_metric_values(self, metric):
        """
        Return an array values under a specfied metric that
        each PFO has.
        
        Parameters
        ----------
        metric : str or function
            Either a string indexing the metric to use (in
            `self.metrics`), or a function containing the
            metric calculation.

        Returns
        -------
        ak.Array
            Array containing the values of each PFO under
            the specified metric.
        """
        if isinstance(metric, str):
            return self.metrics[metric](self.properties)
        return metric(self.properties)

    def get_threshold(self, metric):
        """
        Get the threshold using in a given metric cut
        
        Parameters
        ----------
        metric : str
            Metric for which to get the threshold

        Returns
        -------
        float
            Threshold applied by the metric. If not specified,
            this will be `self.default_threshold`.
        """
        if isinstance(metric, str):
            return self.thresholds[metric]
        return self.default_threshold

    def get_passing(self, metric, threshold=None):
        """
        Creates a boolean mask indicating which PFOs
        pass the specified metric.
        Parameters
        ----------
        metric : str
            Name of the metric as stored in the
            `self.metrics` dictionary.
        threshold : float, optional
            Threshold that must be exceeded by the metric
            for a PFO to pass. If unspecified, the threshold
            assiciated with the metric is used.

        Returns
        -------
        ak.Array
            Boolean mask of PFOs whose metric values
            exceed the metric's threshold.
        """
        if threshold is None:
            threshold = self.get_threshold(metric)
        return self.get_metric_values(metric) > threshold

    def wrap_metric_arguments(self, metric_func, kwarg_metrics=False):
        if kwarg_metrics:
            def dict_metric(parameters):
                return metric_func(**parameters)
        else:
            def dict_metric(parameters):
                return metric_func(parameters[prop] for prop in self.prop_labels)
        return dict_metric
    
    def _confirm_passing(self, vals):
        """Makes sure there exist some values, else returns a non-zero weight"""
        if len(vals) > 0:
            return vals, None
        zero = np.array([0.])
        return zero, zero

    def _plot_hist(self, data, bin_gen, label, index, ax=None, **kwargs):
        """Plot a histogram in the standard form"""
        data_checked, weights = self._confirm_passing(data)
        if weights is not None:
            kwargs.update({"weights":weights})
        if ax is None:
            return plt.hist(
                data_checked,
                **self.plt_cfg.gen_kwargs(
                    type="hist", label=label,
                    index=index, bins=bin_gen(data_checked), **kwargs))
        return ax.hist(
            data_checked,
            **self.plt_cfg.gen_kwargs(
                type="hist", label=label,
                index=index, bins=bin_gen(data_checked), **kwargs))


    def get_selection_stats(self, metric_name, threshold=None, return_text=False, print_text=False):
        passing_pfos = self.get_passing(metric_name, threshold=threshold)

        no_events = passing_pfos.ndim == 1

        num_events = ak.num(passing_pfos, axis=0)

        false_counts = ak.sum( ak.values_astype(passing_pfos[ np.logical_not(self.truth_mask) ], int) )

        if not no_events:
            true_sig_count_per_event = ak.sum( ak.values_astype(passing_pfos[ self.truth_mask ], int), axis=1 )
            true_events_found = ak.sum( ak.values_astype(true_sig_count_per_event > 0, int) )
            excess_events_found = ak.sum( np.maximum(true_sig_count_per_event - 1, 0) )
        
            truth_event_count = ak.sum( ak.values_astype(ak.sum( ak.values_astype(self.truth_mask, int), axis=1 ) > 0, int) )
        else:
            truth_event_count = ak.sum( ak.values_astype(self.truth_mask, int) )
            true_events_found = ak.sum( ak.values_astype(passing_pfos[ self.truth_mask ], int))

        if print_text or return_text:
            summary_text = ""
            if no_events:
                summary_text += f"{num_events} PFOs searched with {truth_event_count} true {self.pfo_name} PFOs.\n"
                summary_text += f"{false_counts} false triggers on non {self.pfo_name} PFOs, leading to a ratio of \n"
                summary_text += f"{false_counts/true_events_found:.2f} false PFOs per true PFO found.\n"
                summary_text += f"{true_events_found} true PFOs found, for an efficiency of {100*true_events_found/truth_event_count:.2f}%.\n"
                summary_text += "Event-by-event stats is not avaiable with these inputs."
            else:
                summary_text += f"{num_events} events searched with {true_events_found} events containing {self.pfo_name}s.\n"
                summary_text += f"{false_counts} false triggers on non {self.pfo_name} PFOs, leading to an average \n"
                summary_text += f"rate of {(false_counts)/num_events:.2f} false triggers per event.\n"
                summary_text += f"{true_events_found} events with true pi0 triggering found, for an efficiency of {100*true_events_found/truth_event_count:.2f}%.\n"
                summary_text += f"{excess_events_found} excess truth events were found for and average of {excess_events_found/truth_event_count:.2f}\n"
                summary_text += f"excess events per truth pi0 trigger."
        if print_text:
            print(summary_text)
        if return_text:
            return summary_text

        return true_events_found/truth_event_count, false_counts/num_events

    def plot_metric_distribution(
        self,
        metric_name,
        y_scaling="linear",
        override=None,
        equal_bins=True,
        ax=None,
        **kwargs 
    ):
        if override == "display":
            y_scaling = "linear"
            equal_bins = True
            kwargs.update({"density":True})

        values = self.get_metric_values(metric_name)

        signal = ak.ravel(values[self.truth_mask])
        background = ak.ravel(values[np.logical_not(self.truth_mask)])

        primary_data = self._determine_primary_data(signal, background)
        bins_gen = self.check_equal_binning(equal_bins, primary_data)
        
        if ax is None:
            self.plt_cfg.setup_figure()
        self._plot_hist(signal,     bins_gen, "Signal",     0, ax=ax, **kwargs)
        self._plot_hist(background, bins_gen, "Background", 1, ax=ax, **kwargs)
        #     plt.hist(signal, **self.plt_cfg.gen_kwargs(type="hist", label="Signal",  index=0, bins=bins_gen(signal), **kwargs))
        #     plt.hist(background,  **self.plt_cfg.gen_kwargs(type="hist", label="Background",  index=1, bins=bins_gen(background), **kwargs))
        # else:
        #     ax.hist(signal, **self.plt_cfg.gen_kwargs(type="hist", label="Signal",  index=0, bins=bins_gen(signal), **kwargs))
        #     ax.hist(background,  **self.plt_cfg.gen_kwargs(type="hist", label="Background",  index=1, bins=bins_gen(background), **kwargs))
        if isinstance(metric_name, str):
            title = metric_name.title()
        else:
            title = "Metric"
        self.plt_cfg.format_axis(ax, xlabel=title, ylabel="Count", ylog=y_scaling=='log')
        if ax is None:
            self.plt_cfg.end_plot()
        else:
            ax.legend()
        return

    def plot_param_distributions(
        self,
        metric_name,
        property,
        y_scaling="linear",
        quartile=100,
        equal_bins=True,
        plot_rejected=False,
        ax=None,
        override=None,
        threshold=None,
        **kwargs 
    ):
        if override == "display":
            quartile = 95
            y_scaling = "log"
            equal_bins = True
            plot_rejected = False
            kwargs.update({"density":True})

        passing_events = self.get_passing(metric_name, threshold=threshold)

        passing_sig  = np.logical_and(self.truth_mask, passing_events)
        passing_bkg  = np.logical_and( np.logical_not(self.truth_mask), passing_events)

        prop = self.properties[property]

        passing_sig  = ak.ravel(prop[passing_sig])
        passing_bkg  = ak.ravel(prop[passing_bkg])

        if plot_rejected:
            rejected_sig = np.logical_and(self.truth_mask, np.logical_not(passing_events))
            rejected_bkg = np.logical_not( np.logical_or(self.truth_mask, passing_events) )
        
            rejected_sig = ak.ravel(prop[rejected_sig])
            rejected_bkg = ak.ravel(prop[rejected_bkg])

        primary_data = ak.ravel(prop)
        quart_data = np.sort(ak.ravel(prop))
        format_ax_kwargs = {}
        if quartile != 100:
            count_to_keep = len(quart_data) * quartile/100
            quartile_range = self._gen_quartile_range(quart_data, count_to_keep)
            # if quart_data[0] < 0:
            #     quartile_range = (
            #         quart_data[max(int(np.round(full_count * (100 - quartile)/200)), 0)],
            #         quart_data[min(int(np.round(full_count * (100 + quartile)/200)) + 1, full_count)])
            # else:
            #     quartile_range = (0, quart_data[min(int(np.round(full_count * quartile/100)) + 1, full_count)])
            
            kwargs.update({"range":quartile_range})
            format_ax_kwargs.update({"xlim":tuple(map(lambda x: x*1.08, quartile_range))})

            primary_data = quart_data[np.logical_and(quart_data >= quartile_range[0], quart_data < quartile_range[1])]
            def apply_lims(data):
                return np.clip(data, quartile_range[0], quartile_range[1])
            passing_sig = apply_lims(passing_sig)
            passing_bkg = apply_lims(passing_bkg)
            if plot_rejected:
                rejected_sig = apply_lims(rejected_sig)
                rejected_bkg = apply_lims(rejected_bkg)
        
        bins_gen = self.check_equal_binning(equal_bins, primary_data)

            # passing_sig = apply_lims(passing_sig)
            # passing_bkg = apply_lims(passing_bkg)
            # if plot_rejected:
            #     rejected_sig = apply_lims(rejected_sig)
            #     rejected_bkg = apply_lims(rejected_bkg)

        if ax is None:
            self.plt_cfg.setup_figure()
        self._plot_hist(passing_sig, bins_gen, "Passing sig", 0, ax=ax, **kwargs)
        self._plot_hist(passing_bkg, bins_gen, "Passing bkg", 1, ax=ax, **kwargs)
        if plot_rejected:
            self._plot_hist(rejected_sig, bins_gen, "Rejected sig", 2, ax=ax, ls="--", **kwargs)
            self._plot_hist(rejected_bkg, bins_gen, "Rejected bkg", 3, ax=ax, ls="--", **kwargs)
        #     plt.hist(passing_sig,  **self.plt_cfg.gen_kwargs(type="hist", label="Passing sig",  index=0, bins=bins_gen(passing_sig), **kwargs))
        #     plt.hist(passing_bkg,  **self.plt_cfg.gen_kwargs(type="hist", label="Passing bkg",  index=1, bins=bins_gen(passing_bkg), **kwargs))
        #     if plot_rejected:
        #         plt.hist(rejected_sig, **self.plt_cfg.gen_kwargs(type="hist", label="Rejected sig", index=2, ls='--', bins=bins_gen(rejected_sig), **kwargs))
        #         plt.hist(rejected_bkg, **self.plt_cfg.gen_kwargs(type="hist", label="Rejected bkg", index=3, ls='--', bins=bins_gen(rejected_bkg), **kwargs))
        # else:
        #     ax.hist(passing_sig,  **self.plt_cfg.gen_kwargs(type="hist", label="Passing sig",  index=0, bins=bins_gen(passing_sig), **kwargs))
        #     ax.hist(passing_bkg,  **self.plt_cfg.gen_kwargs(type="hist", label="Passing bkg",  index=1, bins=bins_gen(passing_bkg), **kwargs))
        #     if plot_rejected:
        #         ax.hist(rejected_sig, **self.plt_cfg.gen_kwargs(type="hist", label="Rejected sig", index=2, ls='--', bins=bins_gen(rejected_sig), **kwargs))
        #         ax.hist(rejected_bkg, **self.plt_cfg.gen_kwargs(type="hist", label="Rejected bkg", index=3, ls='--', bins=bins_gen(rejected_bkg), **kwargs))
        self.plt_cfg.format_axis(ax, xlabel=property.title(), ylabel="Count", ylog=y_scaling=='log', **format_ax_kwargs)
        if ax is None:
            self.plt_cfg.end_plot()
        else:
            ax.legend()
        return
    
    def plot_selection_comparison(
        self,
        metric_name_pre,
        metric_name_post,
        selection_text="",
        y_scaling='linear',
        override=None,
        equal_bins=False,
        ax=None,
        threshold_pre=None,
        threshold_post=None,
        **kwargs
    ):
        if override == "display":
            y_scaling = "log"
            equal_bins = True

        passing_events_pre = self.get_passing(metric_name_pre, threshold=threshold_pre)
        passing_events_post = self.get_passing(metric_name_post, threshold=threshold_post)

        passing_sig_pre   = np.logical_and(self.truth_mask, passing_events_pre)
        passing_bkg_pre   = np.logical_and(np.logical_not(self.truth_mask), passing_events_pre)
        passing_sig_post  = np.logical_and(self.truth_mask, passing_events_post)
        passing_bkg_post  = np.logical_and(np.logical_not(self.truth_mask), passing_events_post)
        
        data_pre  = self.get_metric_values(metric_name_pre)
        data_post = self.get_metric_values(metric_name_post)

        passing_sig_pre = ak.ravel(data_pre[passing_sig_pre])
        passing_bkg_pre = ak.ravel(data_pre[passing_bkg_pre])
        passing_sig_post = ak.ravel(data_post[passing_sig_post])
        passing_bkg_post = ak.ravel(data_post[passing_bkg_post])

        primary_data = self._determine_primary_data(passing_events_pre, passing_events_post)
        bins_gen = self.check_equal_binning(equal_bins, primary_data)

        if ax is None:
            self.plt_cfg.setup_figure()
        self._plot_hist(passing_sig_pre,  bins_gen, "Sig pre " +selection_text, 0, ax=ax, **kwargs)
        self._plot_hist(passing_sig_post, bins_gen, "Sig post "+selection_text, 1, ax=ax, **kwargs)
        self._plot_hist(passing_bkg_pre,  bins_gen, "Bkg pre " +selection_text, 2, ax=ax, ls="--", **kwargs)
        self._plot_hist(passing_bkg_post, bins_gen, "Bkg post "+selection_text, 3, ax=ax, ls="--", **kwargs)
        #     plt.hist(passing_sig_pre,  **self.plt_cfg.gen_kwargs(type="hist", label="Sig pre "+selection_text,  index=0, bins=bins_gen(passing_sig_pre), **kwargs))
        #     plt.hist(passing_sig_post, **self.plt_cfg.gen_kwargs(type="hist", label="Sig post "+selection_text, index=1, bins=bins_gen(passing_sig_post), **kwargs))
        #     plt.hist(passing_bkg_pre,  **self.plt_cfg.gen_kwargs(type="hist", label="Bkg pre "+selection_text,  index=2, ls='--', bins=bins_gen(passing_bkg_pre), **kwargs))
        #     plt.hist(passing_bkg_post, **self.plt_cfg.gen_kwargs(type="hist", label="Bkg post "+selection_text, index=3, ls='--', bins=bins_gen(passing_bkg_post), **kwargs))
        # else:
        #     ax.hist(passing_sig_pre,  **self.plt_cfg.gen_kwargs(type="hist", label="Sig pre "+selection_text,  index=0, bins=bins_gen(passing_sig_pre), **kwargs))
        #     ax.hist(passing_sig_post, **self.plt_cfg.gen_kwargs(type="hist", label="Sig post "+selection_text, index=1, bins=bins_gen(passing_sig_post), **kwargs))
        #     ax.hist(passing_bkg_pre,  **self.plt_cfg.gen_kwargs(type="hist", label="Bkg pre "+selection_text,  index=2, ls='--', bins=bins_gen(passing_bkg_pre), **kwargs))
        #     ax.hist(passing_bkg_post, **self.plt_cfg.gen_kwargs(type="hist", label="Bkg post "+selection_text, index=3, ls='--', bins=bins_gen(passing_bkg_post), **kwargs))
        self.plt_cfg.format_axis(ax, xlabel="Passing metric scores", ylabel="Count", ylog=y_scaling=='log')
        if ax is None:
            self.plt_cfg.end_plot()
        else:
            ax.legend()
        
        return

    def plot_param_correlation(
        self,
        param1,
        param2,
        quartile=100,
        y_scaling = "linear",
        override=None,
        ax=None,
        **kwargs 
    ):
        if override == "display":
            y_scaling = "log"
            quartile = 95
        prop1 = ak.ravel(self.properties[param1])
        prop2 = ak.ravel(self.properties[param2])
        # p1sorting = np.argsort(prop1)
        # quart_data1 = prop1[p1sorting]
        # p2sorting = np.argsort(prop2)
        # quart_data2 = prop2[p2sorting]
        if quartile != 100:
            quart_data1 = np.sort(prop1)
            quart_data2 = np.sort(prop2)
            full_count = len(quart_data1)
            target_count = int(np.round(full_count * quartile/100))
            quartile_range1 = self._gen_quartile_range(quart_data1, target_count)
            quartile_range2 = self._gen_quartile_range(quart_data2, target_count)
            test_bins1 = self.plt_cfg.get_bins(
                prop1[np.logical_and(prop1 >= quartile_range1[0],
                                     prop1 < quartile_range1[1])])
            test_bins2 = self.plt_cfg.get_bins(
                prop2[np.logical_and(prop2 >= quartile_range2[0],
                                     prop2 < quartile_range2[1])])
            n_bins = min(test_bins1, test_bins2)
            bin_width1 = (quartile_range1[1] - quartile_range1[0])/n_bins
            bin_width2 = (quartile_range2[1] - quartile_range2[0])/n_bins
            prop1 = np.clip(prop1,
                            quartile_range1[0] - bin_width1,
                            quartile_range1[1] + bin_width1)
            prop2 = np.clip(prop2,
                            quartile_range2[0] - bin_width2,
                            quartile_range2[1] + bin_width2)
            n_bins += 2
        else:
            n_bins = min(self.plt_cfg.get_bins(prop1),
                         self.plt_cfg.get_bins(prop2))
            # prop1, prop2 = self._double_apply_quartile(prop1, prop2, quartile)
            # full_count = len(quart_data1)
            # target_count = int(np.round(full_count * quartile/100))
            # print(target_count)
            # print(full_count)
            # print((quart_data1[0], quart_data1[-1]))
            # print((quart_data2[0], quart_data2[-1]))
            # print()
            # current_count = 0
            # iter_count = 0
            # p1keep = full_count * quartile/100
            # p2keep = full_count * quartile/100
            # quartile_range1 = self._gen_quartile_range(quart_data1, p1keep)
            # quartile_range2 = self._gen_quartile_range(quart_data2, p2keep)
            # p1mask = np.logical_and(prop1 >= quartile_range1[0], prop1 < quartile_range1[1])
            # p2mask = np.logical_and(prop2 >= quartile_range2[0], prop1 < quartile_range2[1])
            # p1inds = itertools.roundrobin(quart_data1[-np.sum(prop1 >= quartile_range1[1]):],
            #                               quart_data1[np.sum(prop1 < quartile_range1[0])-1::-1])
            # p2inds = itertools.roundrobin(quart_data2[-np.sum(prop2 >= quartile_range2[1]):],
            #                               quart_data2[np.sum(prop2 < quartile_range2[0])-1::-1])
            # tot_inds = np.unique(itertools.roundrobin(p1inds, p2inds))
            # new_inds = tot_inds[target_count-full_count:]
            # p1keep = np.sum(p1mask[new_inds])
            # p2keep = np.sum(p2mask[new_inds])
            # quartile_range1
            # while (target_count != current_count) and (iter_count < 200):
            #     iter_count += 1
            #     quartile_range1 = self._gen_quartile_range(quart_data1, p1keep)
            #     quartile_range2 = self._gen_quartile_range(quart_data2, p2keep)
            #     print(quartile_range1)
            #     print(quartile_range2)
            #     p1mask = np.logical_and(prop1 >= quartile_range1[0], prop1 < quartile_range1[1])
            #     p2mask = np.logical_and(prop2 >= quartile_range2[0], prop1 < quartile_range2[1])
            #     joint_mask = np.logical_or(p1mask, p2mask)
            #     print(np.sum(p1mask))
            #     print(np.sum(p2mask))
            #     print(np.sum(joint_mask))
            #     current_count = np.sum(p1mask) + np.sum(p2mask) - np.sum(joint_mask)
            #     print(current_count)
            #     # p2change = int(np.round((current_count - target_count)/2))
            #     # p1change = current_count - target_count - p2change
            #     print(p1keep)
            #     print(p2keep)
            #     print((target_count - current_count)/2)
            #     p1keep -= (current_count - target_count)/2
            #     p2keep -= (current_count - target_count)/2
            # if iter_count >= 200:
            #     raise Exception("Couldn't apply quartile range after 200 iterations")
            # if quart_data1[0] < 0:
                #     quartile_range1 = (
                #         quart_data1[max(int(np.round(full_count * (100 - quartile)/200)), 0)],
                #         quart_data1[min(int(np.round(full_count * (100 + quartile)/200)) + 1, full_count)])
                # else:
                #     quartile_range1 = (0, quart_data1[min(int(np.round(full_count * quartile/100)) + 1, full_count)])
                # if quart_data2[0] < 0:
                #     quartile_range2 = (
                #         quart_data2[max(int(np.round(full_count * (100 - quartile)/200)), 0)],
                #         quart_data2[min(int(np.round(full_count * (100 + quartile)/200)) + 1, full_count)])
                # else:
                #     quartile_range2 = (0, quart_data1[min(int(np.round(full_count * quartile/100)) + 1, full_count)])

        # bins1 = self.plt_cfg.get_bins(prop1)
        # bins2 = self.plt_cfg.get_bins(prop2)
        if ax is None:
            self.plt_cfg.setup_figure()
            h = plt.hist2d(prop1, prop2, bins=n_bins, norm=y_scaling)
        else:
            h = ax.hist2d(prop1, prop2, bins=n_bins, norm=y_scaling)
        plt.colorbar(h[-1], ax=ax)
        self.plt_cfg.format_axis(ax, xlabel=param1.title(), ylabel=param2.title())
        if ax is None:
            self.plt_cfg.end_plot()
        return

    # @staticmethod
    # def roundrobin(*iterables):
    #     "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    #     # Taken from itertools 'recipes'
    #     # Recipe credited to George Sakkis
    #     num_active = len(iterables)
    #     nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    #     while num_active:
    #         try:
    #             for next in nexts:
    #                 yield next()
    #         except StopIteration:
    #             # Remove the iterator we just exhausted from the cycle.
    #             num_active -= 1
    #             nexts = itertools.cycle(itertools.islice(nexts, num_active))

    # def _order_passing_inds(self, sorted_inds, low_count, high_count):
    #     if low_count != 0:
    #         low_slice = slice(low_count-1, None, -1)
    #     else:
    #         low_slice = slice(None, 0)
    #     if high_count != 0:
    #         high_slice = slice(-high_count)
    #     else:
    #         high_slice = slice(-1, -1)
    #     return self.roundrobin(sorted_inds[high_slice], sorted_inds[low_slice])

    # def _double_apply_quartile(self, prop1, prop2, quartile):
    #     """
    #     Attmepts to apply some quartile to two properties simultaneously
    #     Works, but isn't the cleanest
    #     """
    #     p1sorting = np.argsort(prop1)
    #     quart_data1 = prop1[p1sorting]
    #     p2sorting = np.argsort(prop2)
    #     quart_data2 = prop2[p2sorting]
    #     full_count = len(quart_data1)
    #     target_count = int(np.round(full_count * quartile/100))
    #     p1keep = full_count * quartile/100
    #     p2keep = full_count * quartile/100
    #     quartile_range1 = self._gen_quartile_range(quart_data1, p1keep)
    #     quartile_range2 = self._gen_quartile_range(quart_data2, p2keep)
    #     p1mask = np.logical_and(prop1 >= quartile_range1[0],
    #                             prop1 < quartile_range1[1])
    #     p2mask = np.logical_and(prop2 >= quartile_range2[0],
    #                             prop1 < quartile_range2[1])
    #     p1ind_generator = self._order_passing_inds(p1sorting,
    #                                       np.sum(prop1 < quartile_range1[0]),
    #                                       np.sum(prop1 >= quartile_range1[1]))
    #     p2ind_generator = self._order_passing_inds(p2sorting,
    #                                       np.sum(prop2 < quartile_range2[0]),
    #                                       np.sum(prop2 >= quartile_range2[1]))
    #     # p1inds = self.roundrobin(p1sorting[-np.sum(prop1 >= quartile_range1[1]):],
    #     #                               p1sorting[np.sum(prop1 < quartile_range1[0])-1::-1])
    #     # p2inds = self.roundrobin(p2sorting[-np.sum(prop2 >= quartile_range2[1]):],
    #     #                               p2sorting[np.sum(prop2 < quartile_range2[0])-1::-1])
    #     comb_inds = np.fromiter(self.roundrobin(p1ind_generator, p2ind_generator), int)
    #     print(p1sorting)
    #     print(comb_inds)
    #     comb_inds = comb_inds[np.sort(np.unique(comb_inds, return_index=True)[1])]
    #     new_inds = comb_inds[target_count-full_count:]
    #     print(comb_inds)
    #     print(new_inds)
    #     p1keep = np.sum(p1mask[new_inds])
    #     p2keep = np.sum(p2mask[new_inds])
    #     quartile_range1 = self._gen_quartile_range(quart_data1, p1keep)
    #     quartile_range2 = self._gen_quartile_range(quart_data2, p2keep)
    #     p1mask = np.logical_and(prop1 >= quartile_range1[0], prop1 < quartile_range1[1])
    #     p2mask = np.logical_and(prop2 >= quartile_range2[0], prop1 < quartile_range2[1])
    #     mask = np.logical_and(p1mask, p2mask)
    #     return prop1[mask], prop2[mask]

    def set_style(self, style):
        self.plotting_style = style
        return

    def format_style(self, style, plotting_function):
        kwargs = {}
        if style is None:
            style = self.plotting_style
        styles = style.split("_")

        if "display" in styles:
            kwargs.update({"override":"display"})
        else:
            if "binning" in styles:
                kwargs.update({"equal_bins":False})
            if "log" in styles:
                kwargs.update({"y_scaling":"log"})
            if ("density" in styles) and (plotting_function in [self.plot_param_distributions, self.plot_metric_distribution]):
                kwargs.update({"density":True})
            if ("rejected" in styles) and (plotting_function in [self.plot_param_distributions]):
                kwargs.update({"plot_rejected":True})
            if ("quartile" in styles) and (plotting_function in [self.plot_param_distributions, self.plot_param_correlation]):
                kwargs.update({"quartile":90})
            
        return kwargs

    def _gen_quartile_range(self, sorted_data, num_to_keep):
        """
        Find the upper and lower limts to keep `num_to_keep` values in
        `sorted_data`.
        """
        full_count = len(sorted_data)
        # Don't count invalid values as part of the data quartiles
        # quart_data = sorted_data[sorted_data != -999.]
        if sorted_data[sorted_data != -999.][0] < 0:
            return (
                sorted_data[max(int(np.round(full_count/2 - num_to_keep/2)), 0)],
                sorted_data[min(int(np.round(full_count/2 + num_to_keep/2)), full_count)-1])
        else:
            return (0, sorted_data[min(int(np.round(num_to_keep)), full_count)-1])

    def _determine_primary_data(self, *data):
        """Returns the first ting in data which has values (length > 0)"""
        top_val = 1.
        for d in data:
            if len(d) > 1:
                return d
            elif (len(d) == 1) and (top_val != 1.):
                top_val = d[0]
        # If no good values, set to [0., 1.] for the sake of binning
        return np.array([0., top_val])

    def check_equal_binning(self, equal_binning, primary_data):
        # Override in case there are too many bins (causes large computational delay)
        primary_count = self.plt_cfg.get_bins(primary_data)
        if (primary_count is not None) and (primary_count > 120):
            if equal_binning:
                bins = np.linspace(min(primary_data), max(primary_data), 121)
                def bins_gen(data):
                    return self.plt_cfg.expand_bins(bins, data)
                return bins_gen
            def bins_gen(data):
                return np.linspace(min(data), max(data), 121)
            return bins_gen
        # Standard case if bins look good
        if equal_binning:
            bins = self.plt_cfg.get_bins(primary_data, array=True)
            def bins_gen(data):
                return self.plt_cfg.expand_bins(bins, data)
        else:
            def bins_gen(data):
                return self.plt_cfg.get_bins(data)
        return bins_gen
    
    def remove_property_from_metric(self, prop_to_remove, metric_name):
        def removal_argument_handler(properties_dict):
            # N.B. dictionaries are mutable, so we need to copy it
            prop_removed_dict = properties_dict.copy()
            prop_removed_dict.update({prop_to_remove:None})
            return self.metrics[metric_name](prop_removed_dict)
        return removal_argument_handler
    
    def _get_num_subfigs(self, offset=2):
        n_props = len(self.properties) + offset #Default 2: 1 for summary text
        n_cols = int(np.sqrt(n_props)) #            +1 for first plot
        n_rows = int(np.ceil(n_props/n_cols))
        return n_rows, n_cols

    def make_metric_summary(self, metric_name, style=None):
        metric_kwargs = self.format_style(style, self.plot_metric_distribution)
        param_kwargs = self.format_style(style, self.plot_param_distributions)

        n_rows, n_cols = self._get_num_subfigs()
        _, axes = self.plt_cfg.setup_figure(n_rows, n_cols, figsize=(self.plt_cfg.FIG_SIZE[0]*n_cols, self.plt_cfg.FIG_SIZE[1]*n_rows))
        
        self.plot_metric_distribution(metric_name, ax=axes[0,0], **metric_kwargs)
        base_eff, base_fpr = self.get_selection_stats(metric_name)
        axes[0,0].text(0.55, 0.8, f"{base_eff*100:.2f}% eff., {base_fpr:.2f} false tags/event", fontsize=20, transform=axes[0,0].transAxes)
        
        for index, property in enumerate(self.properties):
            plot_index = index+1
            self.plot_param_distributions(metric_name, property, ax=axes[plot_index//n_cols, plot_index%n_cols], **param_kwargs)
        
        axes[-1,-1].text(0.1, 0.4, self.get_selection_stats(metric_name, return_text=True), fontsize=20)

        self.plt_cfg.end_plot()
        return

    def make_n_minus_1_summary(self, metric_name, prop_to_exclude, style=None):
        comparision_kwargs = self.format_style(style, self.plot_selection_comparison)
        param_kwargs = self.format_style(style, self.plot_param_distributions)

        if isinstance(metric_name, str):
            n_minus_one = f"{metric_name}_n-1_{prop_to_exclude}"
            self.add_metric(n_minus_one, self.remove_property_from_metric(prop_to_exclude, metric_name))
        else:
            n_minus_one = self.remove_property_from_metric(prop_to_exclude, metric_name)

        n_rows, n_cols = self._get_num_subfigs()

        _, axes = self.plt_cfg.setup_figure(n_rows, n_cols, figsize=(self.plt_cfg.FIG_SIZE[0]*n_cols, self.plt_cfg.FIG_SIZE[1]*n_rows))
        
        self.plot_selection_comparison(metric_name, n_minus_one, selection_text=f"{prop_to_exclude} removal", ax=axes[0,0], **comparision_kwargs)
        
        for index, property in enumerate(self.properties):
            plot_index = index+1
            self.plot_param_distributions(n_minus_one, property, ax=axes[plot_index//n_cols, plot_index%n_cols], **param_kwargs)
        
        axes[-1,-1].text(0.1, 0.4, self.get_selection_stats(n_minus_one, return_text=True), fontsize=20)

        self.plt_cfg.end_plot()
        return
    
    def make_n_minus_1_per_property(self, metric_name, style=None, show_thresh=True):
        metric_kwargs = self.format_style(style, self.plot_metric_distribution)
        param_kwargs = self.format_style(style, self.plot_param_distributions)

        n_rows, n_cols = self._get_num_subfigs()

        _, axes = self.plt_cfg.setup_figure(n_rows, n_cols, figsize=(self.plt_cfg.FIG_SIZE[0]*n_cols, self.plt_cfg.FIG_SIZE[1]*n_rows))
        
        self.plot_metric_distribution(metric_name, ax=axes[0,0], **metric_kwargs)
        base_eff, base_fpr = self.get_selection_stats(metric_name)
        axes[0,0].text(0.55, 0.8, f"{base_eff*100:.2f}% eff., {base_fpr:.2f} false tags/event", fontsize=20, transform=axes[0,0].transAxes)

        for index, property in enumerate(self.properties):
            n_minus_one = self.remove_property_from_metric(property, metric_name)

            plot_index = index+1
            curr_ax = axes[plot_index//n_cols, plot_index%n_cols]
            self.plot_param_distributions(n_minus_one, property, ax=curr_ax, **param_kwargs)
            if show_thresh and isinstance(self.metrics[metric_name], SimpleCutSelector):
                threshs = self.metrics[metric_name].applied_cuts[property]
                if threshs is not None:
                    if not isinstance(threshs, tuple):
                        threshs = (-threshs, threshs)
                    curr_ax.vlines(np.array(threshs), 0, 2, **self.plt_cfg.gen_kwargs(type="line"))

            efficiency, fpr = self.get_selection_stats(n_minus_one)
            curr_ax.text(0.55, 0.8, f"{efficiency*100:.2f}% eff., {fpr:.2f} false tags/event", fontsize=20, transform=curr_ax.transAxes)
        
        axes[-1,-1].text(0.1, 0.4, self.get_selection_stats(metric_name, return_text=True), fontsize=20)

        self.plt_cfg.end_plot()
        return

    def make_property_correlations(self, prop, style=None):
        """
        Plot the correlations of `prop` with every property in the
        selector.

        Takes a property, `prop`, which is either a string referencing
        an internal property, and produces a grid of plots showing the
        correlation between the supplied property and each internal
        property in the form of a 2D histogram.

        Parameters
        ----------
        main_prop : str
            Property to test against, reference one of
            `self.properties`.
        style : str, optional
            Style overriding parameters to use. Default is None.
        """
        kwargs = self.format_style(style, self.plot_param_correlation)

        n_rows, n_cols = self._get_num_subfigs(offset=0)
        _, axes = self.plt_cfg.setup_figure(n_rows, n_cols, figsize=(self.plt_cfg.FIG_SIZE[0]*n_cols, self.plt_cfg.FIG_SIZE[1]*n_rows))
        
        for index, property in enumerate(self.properties):
            plot_index = index
            self.plot_param_correlation(property, prop, ax=axes[plot_index//n_cols, plot_index%n_cols], **kwargs)

        self.plt_cfg.end_plot()
        return


class Pi0SelectorEvaluator(SelectorEvaluator):
    def __init__(self, events, pair_coordinates, signal_counts):
        pairs = Master.ShowerPairs(events, pair_coordinates)
        properties = {
            "mass"          : pairs.reco_mass,
            "momentum"      : pairs.reco_pi0_mom,
            "energy"        : pairs.reco_energy,
            "approach"      : pairs.reco_closest_approach,
            "separation"    : pairs.reco_separation,
            "impact"        : PairSelection.paired_beam_impact(pairs),
            "angle"         : pairs.reco_angle
        }

        self.sig_counts = signal_counts
        truth_mask = self.sig_counts == 2
        super().__init__(truth_mask, properties, pfo_name="pi0")

    def get_selection_stats(self, metric_name, threshold=None, return_text=False, print_text=False):
        passing_pairs = self.get_passing(metric_name, threshold=threshold)

        num_events = ak.num(passing_pairs, axis=0)

        false_0_counts = ak.sum( ak.values_astype(passing_pairs[ self.sig_counts == 0 ], int) )
        false_1_counts = ak.sum( ak.values_astype(passing_pairs[ self.sig_counts == 1 ], int) )

        true_sig_count_per_event = ak.sum( ak.values_astype(passing_pairs[ self.sig_counts == 2 ], int), axis=1 )
        true_pi0_events_found = ak.sum( ak.values_astype(true_sig_count_per_event > 0, int) )
        excess_pi0_events_found = ak.sum( np.maximum(true_sig_count_per_event - 1, 0) )
        
        truth_pi0_event_count = ak.sum( ak.values_astype(ak.sum( ak.values_astype(self.sig_counts == 2, int), axis=1 ) > 0, int) )

        if print_text or return_text:
            summary_text = f""
            summary_text += f"{num_events} searched with {truth_pi0_event_count} events containing pi0s.\n"
            summary_text += f"{false_0_counts} false triggers on pairs with no pi0 photons and {false_1_counts} false\n"
            summary_text += f"triggers on pairs with 1 pi0 photon for a total of {false_0_counts + false_1_counts} false triggers,\n"
            summary_text += f"at an average rate of {(false_0_counts + false_1_counts)/num_events:.2f} false triggers per event.\n"
            summary_text += f"{true_pi0_events_found} events with true pi0 triggering found, for an efficiency of {100*true_pi0_events_found/truth_pi0_event_count:.2f}%.\n"
            summary_text += f"{excess_pi0_events_found} excess truth events were found for and average of {excess_pi0_events_found/truth_pi0_event_count:.2f}\n"
            summary_text += f"excess events per truth pi0 trigger."
        if print_text:
            print(summary_text)
        if return_text:
            return summary_text

        return true_pi0_events_found/truth_pi0_event_count, (false_0_counts+false_1_counts)/num_events