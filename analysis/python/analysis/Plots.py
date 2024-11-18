"""
Created on Fri Mar 26 12:16:29 2021

Author: Shyam Bhuller

Description: A script conatining boiler plate code for creating plots with matplotlib.
"""
import copy as copy_lib
import math
import warnings

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.stats import iqr

from python.analysis import vector, Tags, Utils
from python.analysis.SelectionTools import np_to_ak_indicies


class PlotConfig():

    def __init__(self):
        self.PLT_STYLE = "fivethirtyeight"
        self.FIG_SIZE = (14, 10)
        self.FIG_FACECOLOR = 'white'
        self.AXIS_FACECOLOR = 'white'
        self.BINS = 'best'                # Number of bins
        # Allows tweaking the number of bins produced by the "best" algorithm
        self.BEST_BINS_MULTIPLIER = 1
        self.GRID_COLOR = 'gray'                # Color of plot grid
        self.GRID_ALPHA = 0.25                  # Transparency of the plot grid
        self.LEGEND_COLOR = 'white'
        self.MINOR_GRID_X = False
        self.MINOR_GRID_Y = False
        self.SPINE_COLOR = 'black'
        self.TICK_SIZE = 18
        self.TICK_SIZE_MINOR = 14
        self.TICK_LENGTH = 14
        self.TICK_WIDTH = 4
        self.MINOR_TICK_SPACING_X = "auto"
        self.MINOR_TICK_SPACING_Y = "auto"
        self.LEGEND_SIZE = 18
        self.LEGEND_SIZE_DOUBLE = 11
        self.TITLE_SIZE = 28
        self.TITLE_SIZE_DOUBLE = 22
        self.TITLE = None
        self.AX_ALPHA = 0.5
        self.TEXT_SIZE = 18
        self.DPI = 300

        self.SAVE_FOLDER = None                  # Folder to save the plot
        self.SHOW_PLOT = False                 # Show the plot?
        # Count to get number of individual hits per APA. "sum" to get total of hits per APA (Summed Hit Size)
        self.N_HITS_WHICH = "count"
        self.LABEL_SIZE = 24
        self.LOGSCALE = False
        self.LINEWIDTH = 2

        self.HIST_COLOR1 = "tab:blue"
        self.HIST_COLOR2 = "tab:orange"
        self.HIST_COLOR3 = "tab:green"
        self.HIST_COLOR4 = "tab:purple"
        self.HIST_DENSITY = False                 # Normalise the count of the histograms
        self.HIST_TYPE = "step"

        self.HIST_COLOURS = {
            0: self.HIST_COLOR1,
            1: self.HIST_COLOR2,
            2: self.HIST_COLOR3,
            3: self.HIST_COLOR4
        }

    def __str__(self):
        return "\n".join(["{} = {}".format(str(var), vars(self)[var]) for var in vars(self)])

    def copy(self):
        return copy_lib.deepcopy(self)

    def setup_figure(self, sub_rows=1, sub_cols=1, title=None, **kwargs):
        """
        Create a figure and axes instance with the supplied parameters
        and title.

        Parameters
        ----------
        sub_rows : int, optional
            Number of rows to have in the figure. Default is 1.
        sub_cols : int, optional
            Number of columns to have in the figure. Default is 1.
        title : str or None, optional
            Overall title to display over the figure. Default is None.
        **kwargs
            All additional keyword arguments are passed to the
            pyplot.subplots call.

        Returns
        -------
        fig : pyplot.Figure
            Created figure.
        axs : array of Axes
            An array containing an array of the axes in the figure of
            shape `(sub_rows, sub_cols)`. This will be an array
            containing one axis if `sub_rows = sub_cols = 1`.
        """
        # TODO add a scaling function to adjust all test sizes etc,
        # base on either single scalling, or input dimensions
        plt.style.use(self.PLT_STYLE)

        fig_kwargs = {
            "figsize": self.FIG_SIZE,
            "facecolor": self.FIG_FACECOLOR
        }
        fig_kwargs.update(kwargs)

        fig, axs = plt.subplots(sub_rows, sub_cols, **fig_kwargs)

        if title is not None:
            plt.title(title, fontdict={"fontsize": self.TITLE_SIZE})
        elif self.TITLE is not None:
            plt.title(self.TITLE, fontdict={"fontsize": self.TITLE_SIZE})

        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for ax in axs.flatten():
            ax.patch.set_alpha(self.AX_ALPHA)

            ax.tick_params(axis='both', which='major',
                           labelsize=self.TICK_SIZE)
            ax.tick_params(axis='both', direction="out", length=self.TICK_LENGTH,
                           width=self.TICK_WIDTH, grid_color=self.GRID_COLOR, grid_alpha=self.GRID_ALPHA)

            if self.MINOR_TICK_SPACING_X == "auto":
                ax.xaxis.set_minor_locator(AutoMinorLocator())
            else:
                ax.xaxis.set_minor_locator(
                    MultipleLocator(self.MINOR_TICK_SPACING_X))
            if self.MINOR_TICK_SPACING_Y == "auto":
                ax.yaxis.set_minor_locator(AutoMinorLocator())
            else:
                ax.yaxis.set_minor_locator(
                    MultipleLocator(self.MINOR_TICK_SPACING_X))
            ax.tick_params(axis="both", which='minor',
                           labelsize=self.TICK_SIZE-2, length=self.TICK_LENGTH-2)

            ax.xaxis.grid(self.MINOR_GRID_X, which='minor')
            ax.yaxis.grid(self.MINOR_GRID_Y, which='minor')

            if self.LOGSCALE:
                ax.set_yscale("log")

        return fig, axs

    def gen_kwargs(self, type="plot", index=0, **kwargs):
        """
        Generates keyword arguments to format a plot with the style
        of this `PlotConfig` instance.

        The results of this function are intended to be unpacked and
        passed to a `plt.plot` (or similar) call.

        Parameters
        ----------
        type : {"plot", "line", "hist"}, optional
            Type of plot being formatted. "plot" is most generally
            applicable. Default is plot.
        index : int, optional
            Index of the line on the axis. Controls the colour. Default
            is 0.
        **kwargs
            All additional keyword arguments are returned by this
            function, and can overwrite generated parameters.

        Returns
        -------
        fig : pyplot.Figure
            Created figure.
        axs : array of Axes
            An array containing an array of the axes in the figure of
            shape `(sub_rows, sub_cols)`. This will be an array
            containing one axis if `sub_rows = sub_cols = 1`.
        """
        final_kwargs = {
            "linewidth": self.LINEWIDTH
        }

        if type == "hist":
            final_kwargs.update({
                "bins": self.BINS,
                "density": self.HIST_DENSITY,
                "histtype": self.HIST_TYPE
            })
            if index <= max(self.HIST_COLOURS.keys()):
                final_kwargs.update({"color": self.HIST_COLOURS[index]})
        elif type == "line":
            final_kwargs.pop("linewidth")
            final_kwargs.update({
                "linewidth": self.LINEWIDTH,
                "colors": "red"
            })

        final_kwargs.update(kwargs)

        return final_kwargs

    def format_axis(self, *ax, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=True, xlog=None, ylog=None):
        """
        Formats the supplied axis with the style of this `PlotConfig`
        instance.

        Parameters:
        *ax : pyplot.Axes or array of Axes
            Axes to be formatted. If not supplied, the current axis
            will be obtained via `plt.gca()`.
        xlabel : str or None, optional
            Label to be displayed on the x axis. If None, no label is
            printed. Default is None.
        ylabel : str or None, optional
            Label to be displayed on the y axis. If None, no label is
            printed. Default is None.
        xlim : tuple or None, optional
            Range to be displayed along the x axis in form (lower
            limit, upper limit). If None, the default pyplot limits are
            used. Default is None.
        ylim : tuple or None, optional
            Range to be displayed along the y axis in form (lower
            limit, upper limit). If None, the default pyplot limits are
            used. Default is None.
        legend : bool, optional
            Whether or not to display a legend if plot labels have been
            passed. Default is True.
        xlog : bool or None, optional
            Option to override the default x axis log scale of this
            `PlotConfig` instance via True or False. If None, the
            default settings of the `PlotConfig` instance is used.
            Default is None.
        ylog : bool or None, optional
            Option to override the default y axis log scale of this
            `PlotConfig` instance via True or False. If None, the
            default settings of the `PlotConfig` instance is used.
            Default is None.
        """
        if len(ax) == 0 or (len(ax) == 1 and ax[0] is None):
            # Overrides for log scales needs to be before xlim call
            if xlog is not None:
                if xlog:
                    plt.gca().set_xscale("log")
                else:
                    plt.gca().set_xscale("linear")
            if ylog is not None:
                if ylog:
                    plt.gca().set_yscale("log")
                else:
                    plt.gca().set_yscale("linear")

            plt.xlabel(xlabel, size=self.LABEL_SIZE)
            plt.ylabel(ylabel, size=self.LABEL_SIZE)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.gca().set_facecolor(self.AXIS_FACECOLOR)
            for spine in plt.gca().spines.values():
                spine.set_edgecolor(self.SPINE_COLOR)
            if legend:
                plt.legend(prop={'size': self.LEGEND_SIZE},
                           facecolor=self.LEGEND_COLOR)

        else:
            for a in ax:
                # Overrides for log scales, needs to be before xlim call
                if xlog is not None:
                    if xlog:
                        a.set_xscale("log")
                    else:
                        a.set_xscale("linear")
                if ylog is not None:
                    if ylog:
                        a.set_yscale("log")
                    else:
                        a.set_yscale("linear")

                a.set_xlabel(xlabel, size=self.LABEL_SIZE)
                a.set_ylabel(ylabel, size=self.LABEL_SIZE)
                a.set_xlim(xlim)
                a.set_ylim(ylim)
                a.set_facecolor(self.AXIS_FACECOLOR)
                for spine in a.spines.values():
                    spine.set_edgecolor(self.SPINE_COLOR)
                if legend and len(plt.gca().get_legend_handles_labels()[1])>0:
                    a.legend(prop={'size': self.LEGEND_SIZE},
                             facecolor=self.LEGEND_COLOR)

        return

    def end_plot(self, save_name=None):
        """
        Ends a plot by saving it if `save_name` is specified and
        displaying the plot if the `PlotConfig` instance has
        `SHOW_PLOT=True`.
        The figure will only be closed if the current figure was saved
        or shown to avoid accidental plot removal.

        Parameters
        ----------
        save_name : str or None, optional
            Name to save the plot by. If the name contains a "/", the
            save_name will be assumed to be a full path and saved
            accordingly (N.B. designed for bash style file names),
            otherwise, the plot will be saved under the
            `PlotConfig.SAVE_FOLDER + save_name`. If None, the file
            will not be saved. Default is None.
        """
        # Need to get this to work with multiple figures open
        plt.tight_layout()

        if save_name is not None:
            if "/" in save_name:
                plt.savefig(save_name, bbox_inches='tight',
                            facecolor=self.FIG_FACECOLOR, dpi=self.DPI)
            elif self.SAVE_FOLDER is not None:
                if self.SAVE_FOLDER[-1] != "/":
                    self.SAVE_FOLDER += "/"
                plt.savefig(self.SAVE_FOLDER + save_name, bbox_inches='tight',
                            facecolor=self.FIG_FACECOLOR, dpi=self.DPI)
            else:
                warnings.warn(
                    "Plot has been given a file name to save, but not path to folder in the PlotConfig")

        if self.SHOW_PLOT:
            plt.show()

        if self.SHOW_PLOT or (save_name is not None):
            plt.close()
        return

    def get_bins(self, data, array=False):
        """
        Returns the bins to use for a given set of data as defined by
        `BINS` of the `PlotConfig` instance.

        Parameters
        ----------
        data : array-like
            Data to be binned
        array : bool, optional
            Whether to return an array of bin edges or a number of
            bins. Default is False.

        Returns
        bins : int or np.ndarray
            Returns the number of bins to divide the full range of data
            by if `array` is `False`, else returns an array of bin
            edges.
        """
        if len(np.unique(data)) != 1:
            if self.BINS == "best":
                num_bins, _ = self._get_best_bins(data)
                num_bins = int(num_bins * self.BEST_BINS_MULTIPLIER)
            else:
                num_bins = self.BINS
        elif array:
            return np.linspace(min(data), max(data) + 0.1, 2)
        else:
            return None

        if array:
            return np.linspace(min(data), max(data), num_bins+1)
        return num_bins

    def _get_best_bins(self, data):
        # Test to deal with case where the InterQuartile Range
        #   is wholly contained in one data point
        iqrange = iqr(data)
        if iqrange == 0.:
            iqrange = np.max(data) - np.min(data)

        # Freedman–Diaconis rule for optimal bin width
        bin_width = 2*iqrange/(len(data)**(1/3))
        bins = int(round((np.max(data) - np.min(data))/bin_width))
        return bins, bin_width

    @staticmethod
    def expand_bins(bins, data):
        """
        Given an array of bin edges, expands the bins by extending the
        lower (upper) bin edges by the factors of width of the lower
        (upper) bin until the new set of `data` is contined within the
        new bins.

        Parameters
        ----------
        bins : np.ndarray
            Previous set of bin edges.
        data : array-like
            New data to encapsulate in the binning

        Returns
        -------
        new_bins : np.ndarray
            New bin edges which encapsulate the full range of `data` by
            adding additional bins below/above the existing edges in
            `bins` 
        """
        if min(data) < bins[0]:
            low_bin_width = bins[1] - bins[0]
            bins = np.concatenate((np.arange(
                bins[0] - low_bin_width, min(data) - low_bin_width, -low_bin_width)[::-1], bins))
        if max(data) >= bins[-1]:
            high_bin_width = bins[-1] - bins[-2]
            # TODO nicer way to do this?
            bins = np.concatenate((bins, np.arange(
                bins[-1] + high_bin_width, max(data) + high_bin_width*(1+1e-6), high_bin_width)))
        return bins


class HistogramBatchPlotter():
    def __init__(self, plot_config=None):
        if plot_config is None:
            self.plt_cfg = PlotConfig()
        else:
            self.plt_cfg = plot_config

        self.bins = {}
        self.binned_data = {}
        return

    def _extend_bins(self, new_data, hist_name=None):
        """
        We let the initial plot construct whatever the standard bins are.

        This function takes the previous bins and expands them with bins
        of the same width as used from the initial batch.
        """
        bins = self.bins[hist_name]
        data = self.binned_data[hist_name]

        curr_min = bins[0]
        curr_max = bins[-1]

        # We separate out the low and high bin widths in case the user has supplied
        #   custom bins, but these bins to not cover the entire range of data.
        bin_widths_low = bins[1] - curr_min
        bin_widths_high = curr_max - bins[-2]

        new_min = np.min(new_data)
        new_max = np.max(new_data)

        if curr_min > new_min:
            new_bins_count = math.ceil((curr_min-new_min)/bin_widths_low)
            low_bins = np.linspace(
                curr_min - new_bins_count*bin_widths_low, curr_min - bin_widths_low, new_bins_count)

            empty_bin_data = np.zeros(new_bins_count, dtype=data.dtype)

            bins = np.concatenate((low_bins, bins))
            data = np.concatenate((empty_bin_data, data))

        if curr_max < new_max:
            new_bins_count = math.ceil((new_max - curr_max)/bin_widths_high)
            high_bins = np.linspace(
                curr_max + bin_widths_high, curr_max + new_bins_count*bin_widths_high, new_bins_count)

            empty_bin_data = np.zeros(new_bins_count, dtype=data.dtype)

            bins = np.concatenate((bins, high_bins))
            data = np.concatenate((data, empty_bin_data))

        self.bins[hist_name] = bins
        self.binned_data[hist_name] = data
        return

    def add_batch(self, data, hist_name=None, bins=None, range=None, weights=None):
        """
        Adds the supplied data to the histogram of the given name.
        If no plot name is given, it will be assumed all data
        relates to only one plot.

        Parameters
        ----------
        data
        hist_name
        """
        new_plot = hist_name not in self.binned_data.keys()

        if range is not None:
            data = data[np.logical_and(data >= range[0], data < range[1])]

        if new_plot:
            if bins is None:
                bins = self.plt_cfg.get_bins(data)
            counts, hist_bins = np.histogram(
                data, bins=bins, range=range, weights=weights)

            self.bins.update({hist_name: hist_bins})
            self.binned_data.update({hist_name: counts})

        else:
            if range is None:
                self._extend_bins(data, hist_name)
            counts, hist_bins = np.histogram(
                data, bins=self.bins[hist_name], range=range, weights=weights)

            if (hist_bins != self.bins[hist_name]).all():
                raise AssertionError(
                    "Bins from histogram do not match sorted bins.")
            new_binned_data = self.binned_data[hist_name] + counts
            self.binned_data.update({hist_name: new_binned_data})

        return self.binned_data[hist_name], self.bins[hist_name]

    def plot_hist(self, *ax, hist_name=None, plot_config=None, **plot_kwargs):
        """
        Make a histogram plot of stored data.

        Parameters
        ----------
        *ax : pyplot.Axis
            Axis to plot the hist on. If not specified, a new figure is
            created.
        hist_name : str or None, optional
            Identifier of the data to be plotted. Default is None.
        plot_config : PlotConfig or None, optional
            Plot configuration to use to produce the plot. If None, the
            `plt_cfg` in this `HistogramBatchPlotter` is used. Default
            is None.
        **plot_kwargs
            All additional keyword arguments are passed to the `hist()`
            call.
        """
        if plot_config is None:
            plot_config = self.plt_cfg
        plot_config_kwargs = {
            "type": "hist",
            "bins": self.bins[hist_name],
            "weights": self.binned_data[hist_name]
        }
        plot_config_kwargs.update(plot_kwargs)

        if len(ax) == 1:
            return ax[0].hist(
                self.bins[hist_name][:-1],
                **plot_config.gen_kwargs(
                    **plot_config_kwargs
                )
            )
        else:
            return plt.hist(
                self.bins[hist_name][:-1],
                **plot_config.gen_kwargs(
                    **plot_config_kwargs
                )
            )

    def make_hist_figure(self, hist_name=None, save_name=None, xlabel=None, ylabel=None, title=None):
        """
        Produces a new figure using the histogram(s) data from
        `hist_name`.

        Parameters
        ----------
        hist_name : list, str, or None, optional
            Identifier of the data to be plotted. If a list is passed,
            all supplied data will be plotted on the same axis.
            Default is None.
        save_name : str or None, optional
        xlabel
        ylabel
        title
        """
        if not isinstance(hist_name, list):
            hist_name = [hist_name]
        use_labels = len(hist_name) != 1

        fig, axes = self.plt_cfg.setup_figure(title=title)

        for h_name in hist_name:
            self.plot_hist(axes[0], hist_name=h_name,
                           normed=self.plt_cfg.HIST_DENSITY, labelled=use_labels)

        self.plt_cfg.format_axis(axes[0], xlabel=xlabel, ylabel=ylabel)
        self.plt_cfg.end_plot(save_name=save_name)
        return fig, axes


class PairHistsBatchPlotter(HistogramBatchPlotter):
    def __init__(
        self,
        prop_name, units,
        plot_config=None, unique_save_id="",
        inc_stacked=False, inc_norm=True, inc_log=True,
        # bin_size = None,
        bins=100, range=None,
        **kwargs
    ):
        """
        Creates a PairHistsBatchPlotter instance to store and plot data as
        histograms from batched data.

        `plot_config` contains the baseline PlotConfig instance to use for plotting,
        which will be tweaked by the options below to produce the corresponding
        plots.

        Whether to produce stacked, normalised, and log scale plots is controlled
        by `inc_stacked`, `inc_norm`, and `inc_log` respectively.

        `bins` and `range` can be supplied as lists of values (that could be passed
        to a pyplot function). If a list is passed, the options supplied will be
        iterated through, and a plot will be produced for each value given.
        `bin_size`, `bins`, and `range` must all either have same length, or not be
        a list.

        Parameters
        ----------
        prop_name : str
            Name of the property plotted for title and saved file.
        units : str
            Unit the property is measured.
        plot_config : PlotConfig, optional
            Base plot configuration to use. If not passed, a new instance
            will be created.
        unique_save_id : str, optional
            Extra string to add into the save name to avoid
            overwriting. Default is ''.
        inc_stacked : bool, optional
            Whether to include a stacked histogram plot. Default is False.
        inc_norm : bool, optional
            Whether to include a normalised histogram. Default is True.
        inc_log : bool, optional
            Whether to include logarithmic scale histograms. Will
            also produce a logarithmic normalised plot if `inc_norm`
            is True. Default is True.
        bins : int, np.ndarray, or list, optional
            Number of bins if ``int``, or bin edges if ``ndarray``.
            Default is 100. Note if using multiple bins, these must be
            passed as a list, not an array.
        range : float, None, tuple, or list, optional
            Range over which to produce the plot. Float values will be
            converted to tuples with a ower bound of 0. Default is None.
        weights : ak.Array, np.ndarray, or None, optional
            Weights to used for each value of the `property`. Default
            is None.

        Other Parameters
        ----------------
        **kwargs
            Any additional keyword arguments to be passed to the ``plt.hist``
            calls. The same arguments will be passed to all plots.
        """
        # `bin_size` CURRENTLY NOT USED: We don't enforce the bins for each signal type to have the same widths
        # bin_size : str, list, or None, optional
        # The size of the bins. If None, this will be automatically
        # calculated, but can be manually specified, i.e. if using
        # custom/variable bin widths. Default is None.

        super().__init__(plot_config=plot_config)

        # This sets up the types of plot we are making
        self.includes_stacked = inc_stacked
        self.includes_normed = inc_norm
        self.includes_log_scale = inc_log
        self.includes_normed_log_scale = inc_norm and inc_log

        self._plot_types = {"basic": self.plt_cfg.copy()}
        self._plot_types["basic"].LOGSCALE = False
        self._plot_types["basic"].HIST_TYPE = "step"
        self._plot_types["basic"].HIST_DENSITY = False

        if self.includes_normed:
            self._plot_types.update({"norm": self._plot_types["basic"].copy()})
            self._plot_types["norm"].HIST_DENSITY = True
        if self.includes_log_scale:
            self._plot_types.update({"log": self._plot_types["basic"].copy()})
            self._plot_types["log"].LOGSCALE = True
        if self.includes_normed_log_scale:
            self._plot_types.update(
                {"norm_log": self._plot_types["norm"].copy()})
            self._plot_types["norm_log"].LOGSCALE = True

        # Stacked plots are held separated, since they need a different plotting method
        if self.includes_stacked:
            self._stacked_config = self._plot_types["basic"].copy()
            self._stacked_config = "bar"

        # This sets up the bins and ranges we are using:
        if not isinstance(range, list):
            range = [range]
        # Need our ranges to be tuples
        for i, r in enumerate(range):
            if not (isinstance(r, tuple) or r is None):
                range[i] = (0, r)

        if not isinstance(bins, list):
            bins = [bins]

        # Check the lists supplied are compatible: Either they must both be the same,
        #   or one of them must have a length of one.
        if len(range) != len(bins) and len(range) != 1 and len(bins) != 1:
            raise ValueError(
                f"Bins length {len(bins)} and ranges length {len(range)} incompatible")

        # Get the larger number of supplied values
        num_repeats = max(len(range), len(bins))
        # Tile the smaller one until it has the length of the larger one
        range = range * (num_repeats // len(range))
        bins = bins * (num_repeats // len(bins))

        # Put the values in the list of parameters to run over
        self._range_bins_list = [
            {"range": range[i], "bins": bins[i]} for i in np.arange(num_repeats)]

        # CURRENTLY NOT USED: see comment at start of __init__
        # if not isinstance(bin_size, list):
        #     bin_size = [bin_size]
        # if len(bin_size) != 1 and len(bin_size) != len(bins):
        #     raise ValueError(f"Bins length {len(bins)} and bin sizes length {len(bin_size)} incompatible")

        # Additional plotting parameters
        self._plot_kwargs = kwargs

        # Naming
        self.image_save_base = "paired_" + \
            prop_name.replace(" ", "_") + unique_save_id + "_hist"
        # self.units = units
        self.plot_x_label = prop_name.title() + "/" + units  # self.units

        # Internal namse for the plots
        if self.includes_stacked:
            # N.B. flipped because we want the smaller one to appear first
            self._stacked_names = ["stacked_2", "stacked_1", "stacked_0"]
        self._standard_names = ["standard_0", "standard_1", "standard_2"]

        return

    def add_batch(self, data, sig_count, weights=None):
        """
        Adds the `data` from a batch to the histograms, given the `sig_count`s.

        Parameters
        ----------
        data : ak.Array
        Data from a batch to be added.

        sig_count : ak.Array
        Number of signal particles present in each point in `data`.

        weights : ak.Array, optional
        Weightings of `data` points.
        """
        if self.includes_stacked:  # Need to keep track of the raw weights if producing a stacked plot
            raw_weights = weights
        if weights is None:
            weights = {2: None, 1: None, 0: None}
        else:
            weights = {2: ak.ravel(weights[sig_count == 2]), 1: ak.ravel(
                weights[sig_count == 1]), 0: ak.ravel(weights[sig_count == 0])}

        # For some reason, we need to convert the akward array to numpy, or numpy will complain
        #   "to_rectilinear argument must be iterable" when we try to use a tuple for the range...
        # Define this ahead of time since it can be used by both stacked and normal histograms
        two_signal = ak.ravel(data[sig_count == 2]).to_numpy()

        for index, params in enumerate(self._range_bins_list):
            if self.includes_stacked:
                if raw_weights is not None:
                    all_weights = ak.ravel(raw_weights)
                    no_bkg_weights = ak.ravel(raw_weights[sig_count != 0])
                else:
                    all_weights = None
                    no_bkg_weights = None

                super().add_batch(two_signal,
                                  hist_name=self._stacked_names[0] + f"_ind_{index}", weights=weights[2], **params)
                super().add_batch(ak.ravel(data[sig_count != 0]).to_numpy(
                ), hist_name=self._stacked_names[1] + f"_ind_{index}", weights=no_bkg_weights, **params)
                super().add_batch(ak.ravel(data).to_numpy(),
                                  hist_name=self._stacked_names[2] + f"_ind_{index}", weights=all_weights, **params)

            # Only need to do this once, since the changes between logs etc. are cosmetic
            super().add_batch(ak.ravel(data[sig_count == 0]).to_numpy(
            ), hist_name=self._standard_names[0] + f"_ind_{index}", weights=weights[0], **params)
            super().add_batch(ak.ravel(data[sig_count == 1]).to_numpy(
            ), hist_name=self._standard_names[1] + f"_ind_{index}", weights=weights[1], **params)
            super().add_batch(two_signal,
                              hist_name=self._standard_names[2] + f"_ind_{index}", weights=weights[2], **params)

        return

    def make_figures(self):
        """
        Plots all the stored data into figures, as defined by the parameters
        from initialisation.
        """
        for params_index, params in enumerate(self._range_bins_list):
            curr_range = params["range"]
            if curr_range is None:
                save_path_end = "_full"
            elif isinstance(curr_range, tuple):
                save_path_end = f"_{curr_range[0]}-{curr_range[1]}"
            else:
                save_path_end = f"<{curr_range}"

            # CURRENTLY NOT USED: we don't enforce each signal count to have the same bin widths.
            # To use this, need to get the bin sizes from self.bins[hist_name][1] - self.bins[hist_name][0]
            # if bin_size is None:
            #     if isinstance(bins[i], int):
            #         bin_size = f"{hist_range/bins[i]:.2g}" + units
            #     else:
            #         bin_size = f"{hist_range/len(bins[i]):.2g}" + units

            if self.includes_stacked:
                _, stack_axes = self._stacked_config.setup_figure()
                for plot_index, stack_name in enumerate(self._stacked_names):
                    self.plot_hist(
                        stack_axes[0],
                        index=plot_index,
                        hist_name=stack_name + f"_ind_{params_index}",
                        label=f"{stack_name[-1]} signal PFOs",
                        plot_config=self._stacked_config,
                        range=curr_range,
                        **self._plot_kwargs
                    )
                self._stacked_config.format_axis(
                    stack_axes[0], xlabel=self.plot_x_label, ylabel="Count")
                self._stacked_config.end_plot(
                    save_name=self.image_save_base + save_path_end + "_stacked.png")

            for plot_type in self._plot_types.keys():
                ylabel = "Density" if "norm" in plot_type else "Count"

                _, axes = self._plot_types[plot_type].setup_figure()
                for plot_index, name in enumerate(self._standard_names):
                    self.plot_hist(
                        axes[0],
                        index=plot_index,
                        hist_name=name + f"_ind_{params_index}",
                        label=f"{name[-1]} signal PFOs",
                        plot_config=self._plot_types[plot_type],
                        range=curr_range,
                        **self._plot_kwargs
                    )
                self._plot_types[plot_type].format_axis(
                    axes[0], xlabel=self.plot_x_label, ylabel=ylabel)
                self._plot_types[plot_type].end_plot(
                    save_name=self.image_save_base + save_path_end + "_" + plot_type + ".png")

        return


def _adjust_text_colour(value, colour, norm, offset=0.2, max_reduction=0.7):
    scaling = 1-np.round(norm(value)+offset) * max_reduction
    return tuple(val * scaling for val in colour)


class PlotBook:
    def __init__(self, name : str, open : bool = True, watermark : str = None) -> None:
        self.name = name
        self.watermark = watermark
        if ".pdf" not in self.name: self.name += ".pdf" 
        if open: self.open()
        self.is_open = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        self.is_open = False

    def PlotWatermark(self):
        plt.text(0.5, 0.5, self.watermark, transform=plt.gca().transAxes, fontsize=38, color='gray', alpha=0.5, ha='center', va='center', rotation=30, zorder = np.inf)
        return

    def Save(self):
        global preliminary
        if hasattr(self, "pdf"):
            try:
                if self.watermark is not None:
                    self.PlotWatermark()
                self.pdf.savefig(bbox_inches='tight')
            except AttributeError:
                pass

    def open(self):
        if not hasattr(self, "pdf"):
            self.pdf = PdfPages(self.name)
            print(f"pdf {self.name} has been opened")
        else:
            warnings.warn("pdf has already been opened")
        return

    def close(self):
        if hasattr(self, "pdf"):
            self.pdf.close()
            delattr(self, "pdf")
            print(f"pdf {self.name} has been closed")
        else:
            warnings.warn("pdf has not been opened.")
        return

    @classmethod
    @property
    def null(cls):
        return cls(name = "", open = False)

def Save(name: str = "plot", directory: str = "", dpi = 300):
    """ Saves the last created plot to file. Run after one the functions below.

    Args:
        name (str, optional): Name of plot. Defaults to "plot".
        directory (str, optional): directory to save plot in.
    """
    plt.savefig(directory + name + ".png", dpi = dpi, bbox_inches='tight')
    plt.close()


def ClipJagged(array, min, max):
    """ Clips a jagged (awkward) array, if the type of aray is an awkward array, returns as an awkward array, otherwise it returns a list of nested arrays

    Args:
        array: array

    Returns:
        clipped array
    """
    orig_type = type(array)
    if orig_type != ak.Array:
        array = ak.Array(array)

    array = ak.where(array < min, min, array)
    array = ak.where(array > max, max, array)

    if orig_type != ak.Array:
        array = list(array)
    return array


def FigureDimensions(x : int, orientation : str = "horizontal") -> tuple[int]:
    """ Compute dimensions for a multiplot which makes the grid as "square" as possible.

    Args:
        x (int): number of plots in multiplot
        orientation (str, optional): which axis of the grid is longer. Defaults to "horizontal".

    Returns:
        tuple[int]: length of each grid axes
    """
    nearest_square = int(np.ceil(x**0.5)) # get the nearest square number, always round up to ensure there is enough space in the grid to contain all the plots

    if x < 4: # the special case where the the smallest axis is 1
        dim = (1, x)
    elif (nearest_square - 1) * nearest_square >= x: # check if we can fit the plots in a smaller grid than a square to reduce whitespace
        dim = ((nearest_square - 1), nearest_square)
    else:
        dim = (nearest_square, nearest_square)
    
    if orientation == "vertical": # reverse orientation if needed
        dim = dim[::-1]
    return dim


def MultiPlot(n : int, xlim : tuple = None, ylim : tuple = None, orientation = "horizontal"):
    """ Generator for subplots.

    Args:
        n (int): number of plots

    Yields:
        Iterator[int]: ith plot
    """
    dim = FigureDimensions(n, orientation)
    plt.subplots(figsize = [6.4 * dim[1], 4.8 * dim[0]])
    for i in range(n):
        plt.subplot(dim[0], dim[1], i + 1)
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        yield i


def IterMultiPlot(x, xlim : tuple = None, ylim : tuple = None, threshold = 100):
    """ Generator for subplots but also iterates with the data to plot to reduce boilerplate.

    Args:
        x (_type_): data set to plot, must be multidimensional and contains less than 100 samples i.e. plots to make

    Raises:
        Exception: if the length of x is too large

    Yields:
        tuple: ith plot and sample to plot.
    """
    if len(x) > threshold:
        raise Exception("Too many plots specified, did you pass the correct shape data?")
    for i, j in zip(MultiPlot(len(x), xlim, ylim), x):
        yield i, j


def Plot(x, y, xlabel: str = None, ylabel: str = None, title: str = None, label: str = "", marker: str = "", linestyle: str = "-", markersize : float = 6, alpha : float = 1, newFigure: bool = True, x_scale : str = "linear", y_scale : str = "linear", annotation: str = None, color : str = None, xerr = None, yerr = None, capsize : float = 3, zorder : int = None, style : str = "scatter", rasterized : bool = False):
    """ Make scatter plot.
    """
    if newFigure is True:
        plt.figure()

    if style == "bar":
        width = min(x[1:] - x[:-1]) # until I figure out how to do varaible widths
        plt.bar(x, y, width, xerr = xerr, yerr = yerr, linestyle = linestyle, label = label, color = color, alpha = alpha, capsize = capsize, zorder = zorder)
    elif style == "step":
        if (xerr is not None): warnings.warn("x error bars are not supported with style 'step'")
        
        y = np.array([i for _, i in sorted(zip(x, y))])
        if yerr is not None:
            yerr = np.array([i for _, i in sorted(zip(x, yerr))])
        x = sorted(x)

        width = abs(x[0] - x[1]) if len(x) > 1 else 0
        edges = []
        for i in x:
            edges.append(i + width/2)
        edges.insert(0, edges[0] - width)
        
        if color is None: color = next(plt.gca()._get_lines.prop_cycler)["color"] # cause apparently stairs suck

        plt.stairs(y, edges, linestyle = linestyle, edgecolor = color, color = color, alpha = alpha, zorder = zorder, label = label)

        if yerr is not None:
            plt.stairs(y+yerr, edges, baseline=y-yerr, fill = True, alpha = 0.25, color = color)
    elif style == "scatter":
        plt.errorbar(x, y, yerr, xerr, marker = marker, linestyle = linestyle, label = label, color = color, markersize = markersize, alpha = alpha, capsize = capsize, zorder = zorder, rasterized = rasterized)
    else:
        raise Exception(f"{style} not a valid style")


    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if title is not None: plt.title(title)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    if label != "":
        plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()


def PlotComparison(x, y, labels: list, xlabel: str = "", ylabel: str = "", title: str = "", marker: str = "", linestyle: str = "-", markersize : float = 6, alpha : float = 1, newFigure: bool = True, x_scale : str = "linear", y_scale : str = "linear", annotation: str = None, xerr = None, yerr = None, capsize : float = 3):
    """ Make multiple scatter plots in the same axes.
    """
    if newFigure is True: plt.figure()
    for i in range(len(labels)):
        Plot(x[i], y[i], xlabel, ylabel, title, labels[i], marker, linestyle, markersize, alpha, False, x_scale, y_scale, None, None, xerr, yerr, capsize)
    plt.tight_layout()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')


def PlotHist(data, bins = 100, xlabel : str = "", title : str = "", label = None, alpha : int = 1, histtype : str = "bar", sf : int = 2, density : bool = False, x_scale : str = "linear", y_scale : str = "linear", newFigure : bool = True, annotation : str = None, stacked : bool = False, color = None, range : list = None, truncate : bool = False, weights : ak.Array = None):
    """ Plot 1D histograms.
    Returns:
        np.arrays : bin heights and edges
    """
    if newFigure is True: plt.figure()
    if truncate == True:
        if range is None:
            raise Exception("if truncate is true, range must be provided")
        data = ClipJagged(data, min(range), max(range))

    height, edges, _ = plt.hist(data, bins, label = label, alpha = alpha, density = density, histtype = histtype, stacked = stacked, color = color, range = range if range and len(range) == 2 else None, weights = weights)
    binWidth = round((edges[-1] - edges[0]) / len(edges), sf)
    # TODO: make ylabel a parameter
    if density == False:
        yl = "Number of entries (bin width=" + str(binWidth) + ")"
    else:
        yl = "Normalized number of entries (bin width=" + str(binWidth) + ")"
    if xlabel is not None: plt.xlabel(xlabel)
    plt.ylabel(yl)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.title(title)
    if label is not None: plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    return height, edges


def PlotHist2DMarginal(data_x, data_y, bins: int = 100, x_range: list = None, y_range: list = None, z_range: list = [None, None], xlabel: str = "", ylabel: str = "", title: str = "", label: str = "", x_scale: str = "linear", y_scale: str = "linear", annotation: str = None, cmap : str = "viridis", norm : bool = True, whitespace : float = 0.0):
    plt.subplots(2, 2, figsize = (6.4, 4.8 * 1.2), gridspec_kw={"height_ratios" : [1, 5], "width_ratios" : [4, 1], "wspace" : whitespace, "hspace" : whitespace} , sharex = False, sharey = False) # set to that the ratio plot is 1/5th the default plot height
    plt.subplot(2, 2, 2).set_visible(False) # top right

    plt.subplot(2, 2, 3) # bottom left (main plot)

    not_nan = (~np.isnan(data_x)) & (~np.isnan(data_y))

    h, (x_e, y_e) = PlotHist2D(data_x[not_nan], data_y[not_nan], bins, x_range, y_range, z_range, xlabel, ylabel, title, label, x_scale, y_scale, False, annotation, cmap, norm, False)

    plt.subplot(2, 2, 1) # top right (x projection)
    ny, _, _ = plt.hist(x_e[:-1], bins = x_e, weights = np.sum(h, 1), density = True)
    plt.xticks(ticks = x_e, labels = [])
    plt.locator_params(axis='x', nbins=4)
    plt.xlim(min(x_e), max(x_e))
    # plt.ylabel("fraction")
    plt.locator_params(axis='y', nbins=2)
    plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(True)

    ty = [max(ny)/2, max(ny)]
    plt.yticks(ty, [f"{i:.1g}" for i in ty], fontsize = "small")

    plt.subplot(2, 2, 4) # bottom right (y projection)
    nx, _, _ = plt.hist(y_e[:-1], bins = y_e, weights = np.sum(h, 0), density = True, orientation="horizontal")
    plt.yticks(ticks = y_e, labels = [])
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(min(y_e), max(y_e))
    # plt.xlabel("fraction")
    plt.locator_params(axis='x', nbins=2)
    plt.gca().xaxis.get_major_ticks()[0].label1.set_visible(True)

    tx = [max(nx)/2, max(nx)]
    plt.xticks(tx, [f"{i:.1g}" for i in tx], fontsize = "small")

    plt.colorbar(ax = plt.gca())
    plt.subplot(2, 2, 3) # switch back to main plot at the end
    plt.tight_layout()
    return


def PlotHist2D(data_x, data_y, bins: int = 100, x_range: list = None, y_range: list = None, z_range: list = [None, None], xlabel: str = "", ylabel: str = "", title: str = "", label: str = "", x_scale: str = "linear", y_scale: str = "linear", newFigure: bool = True, annotation: str = None, cmap : str = "viridis", norm : bool = True, colorbar : bool = True):
    """ Plot 2D histograms.

    Returns:
        np.arrays : bin heights and edges
    """
    if newFigure is True:
        plt.figure()

    # plot data with a logarithmic color scale
    if norm is True:
        norm_scale = matplotlib.colors.LogNorm()
    else:
        norm_scale = None
    height, xedges, yedges, _ = plt.hist2d(np.array(data_x), np.array(data_y), bins, range = [x_range, y_range], norm = norm_scale, label = label, vmin = z_range[0], vmax = z_range[1], cmap = cmap)
    if colorbar: plt.colorbar()

    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.title(title)
    if label != "":
        plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    return height, [xedges, yedges]


def PlotHistComparison(datas, x_range: list = [], bins: int = 100, xlabel: str = "", title: str = "", labels: list = [], alpha: int = 1, histtype: str = "step", x_scale: str = "linear", y_scale: str = "linear", sf: int = 2, density: bool = True, annotation: str = None, newFigure: bool = True, colours : list = None, weights : list = None):
    """ Plots multiple histograms on one plot

    Args:
        datas (any): list of data sets to plot
        x_range (list, optional): plot range for all data. Defaults to [].
    """
    if newFigure is True:
        plt.figure()
    if colours is None:
        colours = [None]*len(labels)
    if weights is None:
        weights = [None]*len(labels)
    for i in range(len(labels)):
        data = datas[i]
        weight = weights[i]
        if x_range and len(x_range) == 2:
            mask = (data > min(x_range)) & (data < max(x_range))
            data = data[mask]
            if weight is not None: weight = weight[mask]
        if i == 0:
            _, edges = PlotHist(
                data, bins, xlabel, title, labels[i], alpha, histtype, sf, density, color = colours[i], range = x_range, newFigure=False, weights = weight)
        else:
            PlotHist(data, edges, xlabel, title, labels[i], alpha, histtype, sf, density, color = colours[i], range = x_range, newFigure=False, weights = weight)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.tight_layout()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')


def PlotHist2DComparison(x : list, y : list, x_range : list, y_range : list, xlabels = None, ylabels = None, titles = None, bins = 50, cmap = "plasma", func = None, orientation = "horizontal"):
    """ Plots multiple 2D histograms with a shared colour bar.
    """
    if xlabels is None: xlabels = [""]*len(x)
    if ylabels is None: ylabels = [""]*len(y)
    if titles is None: titles = [""] * len(x)

    dim = FigureDimensions(len(x), orientation)
    fig_size = (6.4 * dim[1], 4.8 * dim[0])
    ranges = [x_range, y_range]

    vmax = 0
    for xs, ys in zip(x, y):
        h, _, _ = np.histogram2d(xs, ys, bins, range = ranges)
        vmax = max(vmax, np.max(h))

    fig = plt.figure(figsize = fig_size)
    for i in range(len(x)):
        plt.subplot(*dim, i + 1)
        if func is None:
            _, _, _, im = plt.hist2d(x[i], y[i], bins, range = ranges, cmin = 1, vmin = 0, vmax = vmax, cmap = cmap)
        else:
            _, _, _, im = func(x = x[i], y = y[i], bins = bins, ranges = ranges, cmin = 1, vmin = 0, vmax = vmax, cmap = cmap)
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])
        plt.title(titles[i])
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.subplots_adjust(right=0.84)
    fig.colorbar(im, cax = cbar_ax)


def PlotHistDataMC(data : ak.Array, mc : ak.Array, bins : int = 100, x_range : list = None, stacked = False, data_label : str = "Data", mc_labels = "MC", xlabel : str = None, title : str = None, yscale : str = "linear", legend_loc : str = "best", ncols : int = 2, norm : bool = False, sf : int = 2, colour : str = None, alpha : float = None, truncate : bool = False, mc_weights : np.array = None):
    """ Make Data MC histograms as seen in most typical particle physics analyses (but looks better).

    Args:
        data (ak.Array): data sample
        mc (ak.Array): MC sample
    """
    plt.subplots(2, 1, figsize = (6.4, 4.8 * 1.2), gridspec_kw={"height_ratios" : [5, 1]} , sharex = True) # set to that the ratio plot is 1/5th the default plot height

    is_tagged = hasattr(mc_labels, "__iter__") & (type(mc_labels) != str) 
    if norm == False:
        scale = 1
    elif norm is True:
        if mc_weights is None:
            scale = ak.count(data) / ak.count(mc) # scale MC to data if requested (number of MC entries == number of data entries).
        else:
            scale = ak.count(data) / ak.sum(mc_weights)
    elif norm > 0:
        scale = norm
    else:
        raise Exception("not a valid value for the normalisation")

    if truncate == True:
        data = np.clip(data, min(x_range), max(x_range))
        if is_tagged:
            mc = [np.clip(m, min(x_range), max(x_range)) for m in mc]
        else:
            mc = np.clip(mc, min(x_range), max(x_range))

    plt.subplot(211) # MC histogram
    if x_range is None: x_range = [ak.min([mc, data]), ak.max([mc, data])]

    if is_tagged:
        h_mc = []
        sum_mc = []
        for i, m in enumerate(mc):
            sum_mc.append(int(ak.count(m) * scale))
            h, edges = np.histogram(np.array(m), bins, range = x_range, weights = mc_weights[i] if mc_weights else None)
            h_mc.append(h * scale)
    else:
        sum_mc = int(ak.count(mc) * scale)
        h_mc, edges = np.histogram(np.array(mc), bins, range = x_range, weights = mc_weights)
        h_mc = h_mc * scale

    ind = np.argsort(sum_mc)[::-1]
    if stacked == "ascending":
        ind = ind[::-1]
    centres = (edges[:-1] + edges[1:]) / 2

    if is_tagged:
        for i in range(len(mc_labels)):
            mc_labels[i] = mc_labels[i] + f" ({sum_mc[i]})"
    else:
        mc_labels = mc_labels + f" ({sum_mc})"

    if is_tagged:
        if stacked:
            plt.hist([edges[:-1]]*len(h_mc), edges, weights = np.array(h_mc)[ind].T, range = x_range, stacked = True, label = np.array(mc_labels)[ind], color = np.array(colour)[ind], alpha = alpha)
        else:
            for m in ind:
                plt.hist(edges[:-1], edges, weights = h_mc[m], range = x_range, stacked = False, label = mc_labels[m], color = colour[m], alpha = alpha)
        plt.errorbar(centres, np.sum(h_mc, 0), abs(np.sum(h_mc, 0))**0.5, c = "black", label = "MC total" + f" ({int(ak.count(mc) * scale)})", marker = "x", capsize = 3, linestyle = "")
    else:
        plt.hist(edges[:-1], edges, weights = h_mc, range = x_range, stacked = False, label = mc_labels, color = colour, alpha = alpha)

    plt.yscale(yscale)
    plt.title(title)

    binWidth = round((edges[-1] - edges[0]) / len(edges), sf)
    if norm == False:
        yl = "Number of entries (bin width=" + str(binWidth) + ")"
    else:
        yl = "Normalised number of entries (bin width=" + str(binWidth) + ")"
    plt.ylabel(yl)

    h_data, edges = np.histogram(np.array(data), bins = edges, range = x_range) # bin the data in terms of MC
    data_err = np.sqrt(h_data) # poisson error in each bin
    plt.errorbar(centres, h_data, data_err, marker = "o", c = "black", capsize = 3, linestyle = "", label = data_label + f" ({ak.count(data)})")

    plt.legend(loc = legend_loc, ncols = ncols, labelspacing = 0.25)

    if norm:
        h, l = plt.gca().get_legend_handles_labels()
        plt.legend(h + [matplotlib.patches.Rectangle((0,0), 0, 0, fill = False, edgecolor='none', visible=False)], l + [f"norm: {scale:.3g}"], loc = legend_loc, ncols = ncols, labelspacing = 0.25,  columnspacing = 0.25)

    plt.tick_params("x", labelbottom = False) # hide x axes tick labels

    if is_tagged:
        h_mc = np.sum(h_mc, axis = 0)
    mc_error = np.sqrt(abs(h_mc)) # weights can cause the counts to be negative

    plt.subplot(212) # ratio plot
    ratio = Utils.nandiv(h_data, h_mc) # data / MC
    ratio_err = abs(ratio * np.sqrt(Utils.nandiv(data_err, h_data)**2 + Utils.nandiv(mc_error, h_mc)**2))
    ratio[ratio == np.inf] = -1 # if the ratio is undefined, set it to -1
    plt.errorbar(centres, ratio, ratio_err, c = "black", marker = "o", capsize = 3, linestyle = "")
    plt.ylabel("Data/MC")

    ticks = [0, 0.5, 1, 1.5, 2] # hardcode the yaxis to have 5 ticks
    plt.yticks(ticks, np.char.mod("%.2f", ticks))
    plt.ylim(0, 2)

    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.subplot(211) # do this to set the current figure to the main plot


def PlotHist2DImshow(x : ak.Array, y : ak.Array, bins : int = 100, x_range : list = None, y_range : list = None, c_range : list = None, xlabel : str = None, ylabel : str = None, title : str = None, c_scale : str = "linear", norm : str = None, newFigure : bool = True, colorbar : bool = True, cmap : str = "viridis"):
    """ Same as Hist2D, but uses numpy.histogram2d and imshow to allow more options for plot normalisation,
        but is harder to use in subplots.
    """
    h, x_e, y_e = np.histogram2d(np.array(x), np.array(y), bins = bins, range = [x_range, y_range])

    if norm == "row":
        h_norm = Utils.nandiv(h, np.nanmax(h, 0)) # normalised by row
    elif norm == "column":
        h_norm = Utils.nandiv(h, np.nanmax(h, axis = 1)[:, np.newaxis]) # normalised by column
    else:
        h_norm = h # don't normalise

    if c_range is None: c_range = [np.nanmin(h_norm), np.nanmax(h_norm)]

    if c_scale == "log":
        cnorm = matplotlib.colors.LogNorm(vmin = max(min(c_range), 1) , vmax = max(c_range))
    else:
        cnorm = matplotlib.colors.Normalize(vmin = min(c_range), vmax = max(c_range))

    if newFigure is True: plt.figure()

    plt.imshow(h_norm.T, origin = "lower", extent=[min(x_e), max(x_e), min(y_e), max(y_e)], norm = cnorm, aspect = "auto", cmap = cmap)
    plt.grid(False)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    plt.title(title)
    if colorbar: plt.colorbar()
    return h, (x_e, y_e)


def PlotHist2DImshowMarginal(data_x, data_y, bins: int = 100, x_range: list = None, y_range: list = None, xlabel: str = "", ylabel: str = "", title: str = "", cmap : str = "viridis", norm : bool = True, whitespace : float = 0.0, c_range : list = None, c_scale : str = "linear"):
    plt.subplots(2, 2, figsize = (6.4, 4.8 * 1.2), gridspec_kw={"height_ratios" : [1, 5], "width_ratios" : [4, 1], "wspace" : whitespace, "hspace" : whitespace} , sharex = False, sharey = False) # set to that the ratio plot is 1/5th the default plot height
    plt.subplot(2, 2, 2).set_visible(False) # top right

    plt.subplot(2, 2, 3) # bottom left (main plot)
    h, (x_e, y_e) = PlotHist2DImshow(data_x, data_y, bins, x_range, y_range, c_range, xlabel, ylabel, title, c_scale, norm, False, False, cmap)

    plt.subplot(2, 2, 1) # top right (x projection)
    plt.hist(x_e[:-1], bins = x_e, weights = np.sum(h, 1), density = True)
    plt.xticks(ticks = x_e, labels = [])
    plt.locator_params(axis='both', nbins=4)
    plt.xlim(min(x_e), max(x_e))
    plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(False)

    plt.subplot(2, 2, 4) # bottom right (y projection)
    plt.hist(y_e[:-1], bins = y_e, weights = np.sum(h, 0), density = True, orientation="horizontal")
    plt.yticks(ticks = y_e, labels = [])
    plt.locator_params(axis='both', nbins=4)
    plt.ylim(min(y_e), max(y_e))
    plt.gca().xaxis.get_major_ticks()[0].label1.set_visible(False)

    plt.colorbar(ax = plt.gca())
    plt.tight_layout()
    plt.subplot(2, 2, 3) # switch back to main plot at the end
    return


def PlotConfusionMatrix(counts : np.ndarray, x_tick_labels : list[str] = None, y_tick_labels : list[str] = None, title : str = None, newFigure : bool = True, cmap : str = "cool", x_label : str = None, y_label : str = None):
    """ Plots confusion matrix

    Args:
        counts (np.ndarray, optional): confusion matrix of counts.
        x_tick_labels (list[str], optional): labels for categories in the x axis. Defaults to None.
        y_tick_labels (list[str], optional): labels for categories in the y axis. Defaults to None.
        title (str, optional): plot title. Defaults to None.
        newFigure (bool, optional): create plot in new figure. Defaults to True.
        cmap (str, optional): colour map. Defaults to "cool".
        x_label (str, optional): x label. Defaults to None.
        y_label (str, optional): y label. Defaults to None.
    """
    fractions = counts / np.sum(counts, axis = 1)[:, np.newaxis]
    if newFigure: plt.figure()
    c_norm = counts/np.sum(counts, axis = 0)
    plt.imshow(c_norm, cmap = cmap, origin = "lower", vmin=0., vmax=1.)
    plt.colorbar(label = "Column normalised counts", shrink = 0.8)

    y_counts = np.sum(counts, axis = 1)
    x_counts = np.sum(counts, axis = 0)

    if x_tick_labels is None:
        x_tick_labels = [f"{i}" for i in range(np.array(counts).shape[0])]
    if y_tick_labels is None:
        y_tick_labels = [f"{i}" for i in range(np.array(counts).shape[1])]

    x_counts = [f"{x_tick_labels[r].replace('_', ' ')}\n({x_counts[r]})" for r in range(len(x_tick_labels))]
    y_counts = [f"{y_tick_labels[t].replace('_', ' ')}\n({y_counts[t]})" for t in range(len(y_tick_labels))]


    plt.gca().set_xticks(np.arange(len(x_counts)), labels=x_counts)
    plt.gca().set_yticks(np.arange(len(y_counts)), labels=y_counts)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation = 30)
    plt.yticks(rotation = 30)

    if title is not None:
        plt.title(title + "| Key: (counts, efficiency(%), purity(%))")
    else:
        plt.title("Key: (counts, efficiency(%), purity(%))")

    for (i, j), z in np.ndenumerate(counts):
        plt.gca().text(j, i, f"{z},\n{fractions[i][j]*100:.2g}%,\n{c_norm[i][j]*100:.2g}%", ha='center', va='center', fontsize = 8)
    plt.grid(False)
    plt.tight_layout()


def DrawMultiCutPosition(value : float | list[float], arrow_loc : float = 0.8, arrow_length : float = 0.2, face : str | list[str] = "right", flip : bool = False, color = "black", annotate : bool = False):
    if type(value) == list and type(face) == list:
        for i in range(max(2, len(value))):
            DrawCutPosition(value[i], arrow_loc, arrow_length, face[i], flip, color, annotate)
    else:
        DrawCutPosition(value, arrow_loc, arrow_length, face, flip, color, annotate)
    return


def DrawCutPosition(value : float, arrow_loc : float = 0.8, arrow_length : float = 0.2, face : str = "right", flip : bool = False, color = "black", annotate : bool = False):
    """ Illustrates a cut on a plot. Direction of the arrow indidcates which portion of the plot passes the cut.

    Args:
        value (float): value of the cut
        arrow_loc (float, optional): where along the line to place the arrow. Defaults to 0.8.
        arrow_length (float, optional): length of the arrow, must be in units of the cut. Defaults to 0.2.
        face (str, optional): which way the arrow faces, options are ["left", "right", "<"(left), ">"(right)] . Defaults to "right".
        flip (bool, optional): flip the arrow to the y axis. Defaults to False.
        color (str, optional): colour of the line and arrow. Defaults to "black".
    """

    if face in ["right", ">"]:
        face_factor = 1
    elif face in ["left", "<"]:
        face_factor = -1
    else:
        raise Exception("face must be left or right")

    xy0 = (value - face_factor * (value/1500), arrow_loc)
    xy1 = (value - (value/1500) + face_factor * arrow_length, arrow_loc)
    transform = ("data", "axes fraction")

    if flip:
        xy0 = tuple(reversed(xy0))
        xy1 = tuple(reversed(xy1))
        transform = tuple(reversed(transform))

        plt.axhline(value, color = color)
    else:
        plt.axvline(value, color = color)

    if annotate: plt.annotate(f"{value:.3g}", xy = xy0, xycoords = transform)
    plt.annotate("", xy = xy1, xytext = xy0, arrowprops=dict(facecolor = color, edgecolor = color, arrowstyle = "->"), xycoords= transform)


def PlotTagged(data : np.array, tags : Tags.Tags, bins = 100, x_range : list = None, y_scale : str = "linear", x_label : str = "", loc : str = "best", ncols : int = 2, data2 : np.array = None, norm : bool = False, title : str = "", newFigure : bool = True, stacked : bool = True, alpha : float = None, truncate : bool = False, histtype : str = "stepfilled", reverse_sort : bool = False, data_weights : np.array = None):
    """ Makes a stacked histogram and splits the sample based on tags.

    Args:
        data (np.array): data to plot
        tags (shower_merging.Tags): tags for the data.
        bins (int, optional): number of bins. Defaults to 100.
        range (list, optional): plot range. Defaults to None.
        y_scale (str, optional): y axis scale. Defaults to "linear".
        x_label (str, optional): x label. Defaults to "".
        loc (str, optional): legend location. Defaults to "best".
        ncols (int, optional): number of columns in legend. Defaults to 2.
        data2 (np.array): second sample to plot. if specified it will make a data MC plot (data is MC, data2 is Data).
    """
    split_data = [ak.ravel(data[tags[t].mask]) for t in tags]

    sorted_index = np.argsort(ak.num(split_data))[::-1]
    if reverse_sort:
        sorted_index = sorted_index[::-1]
    split_data = [split_data[i] for i in sorted_index]

    if data_weights is not None:
        split_weights = [ak.ravel(data_weights[tags[t].mask]) for t in tags]
        split_weights = [np.array(split_weights[i]) for i in sorted_index]
    else:
        split_weights = None

    sorted_tags = Tags.Tags()
    for i in sorted_index:
        sorted_tags[tags.number[i].name] = tags.number[i]

    colours = sorted_tags.colour.values
    if ak.any(ak.is_none(colours)):
        print("some tags do not have colours, will override them for the default ones")
        for i in range(len(sorted_tags)):
            colours[i] = "C" + str(i)

    if data2 is None:
        PlotHist(split_data, stacked = stacked, label = [Utils.remove_(i) for i in sorted_tags.name.values], bins = bins, y_scale = y_scale, xlabel = x_label, range = x_range, color = colours, density = bool(norm), title = title, newFigure = newFigure, alpha = alpha, truncate = truncate, histtype = histtype, weights = split_weights)
        plt.legend(loc = loc, ncols = ncols, labelspacing = 0.25,  columnspacing = 0.25)
    else:
        PlotHistDataMC(ak.ravel(data2), split_data, bins, x_range, stacked, "Data", sorted_tags.name.values, x_label, title, y_scale, loc, ncols, norm, colour = colours, alpha = alpha, truncate = truncate, mc_weights = split_weights)


def UniqueData(data):
    """ Formats data to be plotted as a bar plot based on unique values and how often they occur.
    """
    unique, counts = np.unique(data, return_counts=True)
    counts = list(counts)
    unique_labels = []
    for i in range(len(unique)):
        if str(unique[i]) != "[]":
            unique_labels.append(str(unique[i]))
        else:
            counts.pop(i)
    return unique_labels, counts


def PlotBar(data, width: float = 0.4, xlabel: str = "", title: str = "", label: str = "", alpha: float = 1, newFigure: bool = True, annotation: str = None, bar_labels : bool = True, color = None):
    """ Plot a bar graph or unique items in data.
    """
    if newFigure is True:
        plt.figure()

    unique, counts = UniqueData(data)
    bar = plt.bar(unique, counts, width, label=label, alpha=alpha, color = color)
    if bar_labels: plt.bar_label(bar, counts)
    plt.ylabel("Counts")
    if xlabel is not None: plt.xlabel(xlabel)
    plt.title(title)
    if label != "":
        plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    return unique, counts


def PlotBarComparision(data_1, data_2, width: float = 0.4, xlabel: str = "", title: str = "", label_1: str = "", label_2: str = "", newFigure: bool = True, annotation: str = None, ylabel : str = None, fraction : bool = False, barlabel : bool = True):
    """ Plot two bar plots of the same data type side-by-side.
    """
    if newFigure is True:
        plt.figure()

    unique_1, counts_1 = UniqueData(data_1)
    unique_2, counts_2 = UniqueData(data_2)

    m = None
    if(len(unique_2) > len(unique_1)):
        m = unique_2
        unique_2 = unique_1
        unique_1 = m

        m = counts_2
        counts_2 = counts_1
        counts_1 = m

        m = label_2
        label_2 = label_1
        label_1 = m

    missing = [i for i in unique_2 if i not in unique_1]
    loc = [unique_2.index(i) for i in missing]
    for i in loc:
        counts_1.insert(i, 0)
        unique_1.insert(i, unique_2[i])

    missing = [i for i in unique_1 if i not in unique_2]
    loc = [unique_1.index(i) for i in missing]
    for i in loc:
        counts_2.insert(i, 0)

    if fraction is True:
        y_1 = counts_1 / np.sum(counts_1)
        y_2 = counts_2 / np.sum(counts_2)
        yl = "Fractional counts"
    else:
        y_1 = counts_1
        y_2 = counts_2
        yl = "Counts"

    x = np.arange(len(unique_1))

    bar_1 = plt.bar(x - (width/2), y_1, width, label = label_1)
    bar_2 = plt.bar(x + (width/2), y_2, width, label = label_2)

    if barlabel:
        plt.bar_label(bar_1, np.char.mod('%.3f', y_1))
        plt.bar_label(bar_2, np.char.mod('%.3f', y_2))

    plt.xticks(x, unique_1)
    plt.xlabel(xlabel)
    plt.ylabel(yl if ylabel is None else ylabel)
    plt.title(title)
    plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    return [unique_1, counts_1], [unique_2, counts_2]


def PlotStackedBar(bars, labels, xlabel : str = None, colours : list = None, alpha : float = 1, label_title : str = None, width : float = 0.8, annotation : str = None):
    """ Plot stacked bar chart.

    Args:
        bars : bar plot data for each sample, is a list of a list of two numpy arrays, first is the labels, second is the counts.
        labels : sample labels
    """
    bar = []
    for i in range(len(bars)):
        bar.extend(bars[i][0])
    bar = np.unique(bar)

    # pad empty bars and sort
    fixed_bars = list(bars) # copy function input to avoid mutating arguement
    for i in range(len(fixed_bars)):
        for j in range(len(bar)):
            if(bar[j] not in fixed_bars[i][0]):
                fixed_bars[i][0] = np.append(fixed_bars[i][0], bar[j])
                fixed_bars[i][1] = np.append(fixed_bars[i][1], 0)
        fixed_bars[i] = [j[fixed_bars[i][0].argsort()] for j in fixed_bars[i]]
        fixed_bars[i][0] = [str(i) for i in fixed_bars[i][0]] # convert bar values to string for better plotting

    # stack counts
    for i in range(len(fixed_bars)):
        if i == 0: continue
        fixed_bars[i][1] += fixed_bars[i-1][1]

    plt.figure()
    if colours == None:
        for b in reversed(fixed_bars):
            bar = plt.bar(b[0], b[1], alpha = alpha, width = width)
    else:
        for b, c in zip(reversed(fixed_bars), reversed(colours)):
            bar = plt.bar(b[0], b[1], color = c, alpha = alpha, width = width)

    plt.legend(labels = list(reversed(labels)), title = label_title)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.tight_layout()

    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')


def PlotTags(tags : Tags.Tags, xlabel : str = "name", fraction : bool = True, newFigure : bool = True, display_values : bool = False):
    if newFigure: plt.figure()    
    counts = [ak.sum(m) for m in tags.mask.values]
    yl = "Counts"
    if fraction is True:
        counts = counts / np.sum(counts)
        yl = "Fractional counts"
    bar = plt.bar(tags.name.values, counts, color = tags.colour.values)
    plt.xlabel(xlabel)
    plt.ylabel(yl)
    if display_values: plt.bar_label(bar)
    plt.xticks(rotation = 30)


class RatioPlot():
    def __init__(self, x = None, y1 = None, y2 = None, y1_err = None, y2_err = None, xlabel = "x", ylabel = "y2/y1") -> None:
        self.x = x
        self.y1 = y1
        self.y1_err = y1_err
        self.y2 = y2
        self.y2_err = y2_err
        self.xlabel = xlabel
        self.ylabel = ylabel
    def __enter__(self) -> None:
        plt.subplots(2, 1, figsize = (6.4, 4.8 * 1.2), gridspec_kw={"height_ratios" : [5, 1]} , sharex = True) # set to that the ratio plot is 1/5th the default plot height
        plt.subplot(211)
        
        return self
    def __exit__(self, type, value, traceback) -> None:
        plt.subplot(212)
        if self.x is None:
            raise Exception("x has not been assigned")
        if self.y1 is None:
            raise Exception("y1 has not been assigned")
        if self.y2 is None:
            raise Exception("y2 has not been assigned")
        if (self.y2_err is None) and (self.y1_err is not None):
            self.y2_err = np.zeros(len(self.y2))
        if (self.y1_err is None) and (self.y2_err is not None):
            self.y1_err = np.zeros(len(self.y1))

        ratio = Utils.nandiv(self.y1, self.y2)
        
        if (self.y2_err is None) and (self.y1_err is None):
            ratio_err = None
        else:
            ratio_err = abs(ratio * np.sqrt(Utils.nandiv(self.y1_err, self.y1)**2 + Utils.nandiv(self.y2_err, self.y2)**2))

        Plot(self.x, ratio, yerr = ratio_err, xlabel = self.xlabel, ylabel = self.ylabel, marker = "o", color = "black", linestyle = "", newFigure = False)
        ticks = [0, 0.5, 1, 1.5, 2] # hardcode the yaxis to have 5 ticks
        plt.yticks(ticks, np.char.mod("%.2f", ticks))
        plt.ylim(0, 2)

        return True
    def subplot(n):
        plt.subplot(int(f"21{n}"))


def simple_sig_bkg_hist(
    prop_name, units, property, sig_mask,
    path="/users/wx21978/projects/pion-phys/plots/photon_pairs/", unique_save_id="",
    y_scaling='log', bin_size=None, bins=100, range=None, **kwargs
):
    """
    Produces a set of plots of the supplied `property` with configurable
    scalings, bins and ranges.

    Data to be plotted comes from `property`, and a `sig_mask` to indicate
    background vs. signal. Signal and background are plotted on the same axis
    for comparison.

    `prop_name` and `units` are required to correctly label and title the plot.

    `y_scaling`, `bin_size`, `bins`, and `range` can all be supplied as lists of
    values (that could be passed to a pyplot function). If a list is passed,
    the options supplied will be iterated through, and a plot will be produced
    for each value given. `bin_size`, `bins`, and `range` must all either have
    same length, or not be a list.

    Parameters
    ----------
    prop_name : str
        Name of the property plotted for title and saved file.
    units : str
        Unit the property is measured.
    property : ak.Array or np.ndarray
        Array containing the values to be plotted.
    sig_mask : ak.Array or np.ndarray
        Mask which indicates which of the values in `property`
        correspond to signal data.
    path : str, optional
        Directory in which to save the final plot(s). Default is
        '/users/wx21978/projects/pion-phys/plots/photon_pairs/'.
    unique_save_id : str, optional
        Extra string to add into the save name to avoid
        overwriting. Default is ''.
    y_scaling : str or list, optional
        Type of scalings to use on the y-axis. Should be
        'linear' or 'log'. Default is 'log'
    bin_size : str, list, or None, optional
        The size of the bins. If None, this will be automatically
        calculated, but can be manually specified, i.e. if using
        custom/variable bin widths. Default is None.
    bins : int, np.ndarray, or list, optional
        Number of bins if ``int``, or bin edges if ``ndarray``.
        Default is 100.
    range : float, None, tuple, or list, optional
        Range over which to produce the plot. Default is None.

    Other Parameters
    ----------------
    **kwargs
        Any additional keyword arguments to be passed to the ``plt.hist``
        calls. The same arguments will be passed to all plots.
    """
    if path[-1] != '/':
        path += '/'
    if not isinstance(y_scaling, list):
        y_scaling = [y_scaling]
    if not isinstance(range, list):
        range = [range]
    if not ((isinstance(bins, list) or isinstance(bins, np.ndarray)) and len(bins) == len(range)):
        # Strictly, this could be a problem on edge-case that we want to use one fewer bins than the
        # number of ranges  we investigate, but this is sufficiently likely that we can ignore that
        bins = [bins for _ in range]
    for y_scale in y_scaling:
        for i, r in enumerate(range):
            if r is None:
                path_end = "full"
                hist_range = np.max(property)
            elif isinstance(r, tuple):
                path_end = f"{r[0]}-{r[1]}"
                hist_range = r[1] - r[0]
                kwargs.update({"range": r})
            else:
                path_end = f"<{r}"
                hist_range = r
                kwargs.update({"range": (0, r)})

            if not y_scale == "linear":
                path_end += "_" + y_scale

            if bin_size is None:
                if isinstance(bins[i], int):
                    bin_size = f"{hist_range/bins[i]:.2g}" + units
                else:
                    bin_size = f"{hist_range/len(bins[i]):.2g}" + units
            plt.figure(figsize=(12, 9))
            plt.hist(ak.ravel(property[sig_mask]),
                     bins=100, label="signal", **kwargs)
            plt.hist(ak.ravel(property[np.logical_not(
                sig_mask)]), bins=100, label="background", **kwargs)
            plt.legend()
            plt.yscale(y_scale)
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Count/" + bin_size)
            plt.savefig(path + prop_name.replace(" ", "_") +
                        unique_save_id + "_hist_" + path_end + ".png")
            plt.close()
    return


def plot_pair_hists(
    prop_name, units, property, sig_count,
    path="/users/wx21978/projects/pion-phys/plots/photon_pairs/", unique_save_id="",
    inc_stacked=False, inc_norm=True, inc_log=True,
    bin_size=None, bins=100,
    range=None, weights=None, **kwargs
):
    """
    Produces a set of plots of the supplied `property` of a pair of PFOs with
    configurable types of plot, scalings, bins, ranges, weights.

    Data to be plotted comes from `property`, and a `sig_count` indicates
    how many signal type PFOs went into the given pair. 0, 1, and 2 signals
    are plotted on the same axis for comparison.

    `prop_name` and `units` are required to correctly label and title the plots.

    Whether to produce stacked, normalised, and log scale plots is controlled
    by `inc_stacked`, `inc_norm`, and `inc_log` respectively.

    `bin_size`, `bins`, `range`, and `weights` can all be supplied as lists of
    values (that could be passed to a pyplot function). If a list is passed,
    the options supplied will be iterated through, and a plot will be produced
    for each value given. `bin_size`, `bins`, and `range` must all either have
    same length, or not be a list.

    Parameters
    ----------
    prop_name : str
        Name of the property plotted for title and saved file.
    units : str
        Unit the property is measured.
    property : ak.Array or np.ndarray
        Array containing the pair values to be plotted.
    sig_count : ak.Array or np.ndarray
        And array with the same shape as `property` to indicate how
        many signal PFOs exist in the pair.
    path : str, optional
        Directory in which to save the final plots. Default is
        '/users/wx21978/projects/pion-phys/plots/photon_pairs/'.
    unique_save_id : str, optional
        Extra string to add into the save name to avoid
        overwriting. Default is ''.
    inc_stacked : bool, optional
        Whether to include a stacked histogram plot. Default is False.
    inc_norm : bool, optional
        Whether to include a normalised histogram. Default is True.
    inc_log : bool, optional
        Whether to include logarithmic scale histograms. Will
        also produce a logarithmic normalised plot if `inc_norm`
        is True. Default is True.
    bin_size : str, list, or None, optional
        The size of the bins. If None, this will be automatically
        calculated, but can be manually specified, i.e. if using
        custom/variable bin widths. Default is None.
    bins : int, np.ndarray, or list, optional
        Number of bins if ``int``, or bin edges if ``ndarray``.
        Default is 100.
    range : float, None, tuple, or list, optional
        Range over which to produce the plot. Default is None.
    weights : ak.Array, np.ndarray, list, or None, optional
        Weights to used for each value of the `property`. Default
        is None.

    Other Parameters
    ----------------
    **kwargs
        Any additional keyword arguments to be passed to the ``plt.hist``
        calls. The same arguments will be passed to all plots.
    """

    if path[-1] != '/':
        path += '/'

    if not isinstance(range, list):
        range = [range]
    if not ((isinstance(bins, list) or isinstance(bins, np.ndarray)) and len(bins) == len(range)):
        # Strictly, this could be a problem on edge-case that we want to use one fewer bins than the
        # number of ranges  we investigate, but this is sufficiently likely that we can ignore that
        bins = [bins] * len(range)

    # There is definitely a better way to do this...
    if not isinstance(weights, list):
        if inc_stacked:  # Need to keep track of the raw weights if producing a stacked plot
            raw_weights = [weights] * len(range)
        if weights is None:
            weights = [{2: None, 1: None, 0: None}] * len(range)
        else:
            weights = [{2: ak.ravel(weights[sig_count == 2]), 1:ak.ravel(
                weights[sig_count == 1]), 0:ak.ravel(weights[sig_count == 0])}] * len(range)
    else:
        if inc_stacked:  # Need to keep track of the raw weights if producing a stacked plot
            raw_weights = weights
        for i in np.arange(len(weights)):
            if weights[i] is None:
                weights[i] = {2: None, 1: None, 0: None}
            else:
                weights[i] = {2: ak.ravel(weights[i][sig_count == 2]), 1: ak.ravel(
                    weights[i][sig_count == 1]), 0: ak.ravel(weights[i][sig_count == 0])}

    sig_0 = ak.ravel(property[sig_count == 0])
    sig_1 = ak.ravel(property[sig_count == 1])
    sig_2 = ak.ravel(property[sig_count == 2])

    for i, r in enumerate(range):
        if r is None:
            path_end = unique_save_id + "_hist_full"
            hist_range = np.max(property)
        elif isinstance(r, tuple):
            path_end = unique_save_id + f"_hist_{r[0]}-{r[1]}"
            hist_range = r[1] - r[0]
            kwargs.update({"range": r})
        else:
            path_end = unique_save_id + f"_hist<{r}"
            hist_range = r
            kwargs.update({"range": (0, r)})

        if bin_size is None:
            if isinstance(bins[i], int):
                bin_size = f"{hist_range/bins[i]:.2g}" + units
            else:
                bin_size = f"{hist_range/len(bins[i]):.2g}" + units

        if inc_stacked:
            # Whoever wrote this disgusting way to deal with stacked weights ought to be shot...
            if raw_weights[i] is not None:
                all_weights = ak.ravel(raw_weights[i])
                no_bkg_weights = ak.ravel(raw_weights[i][sig_count != 0])
            else:
                all_weights = None
                no_bkg_weights = None

            plt.figure(figsize=(12, 9))
            plt.hist(ak.ravel(property),                 label="0 signal",
                     bins=bins[i], weights=all_weights,    color="C2", **kwargs)
            plt.hist(ak.ravel(property[sig_count != 0]), label="1 signal",
                     bins=bins[i], weights=no_bkg_weights, color="C1", **kwargs)
            plt.hist(sig_2,                              label="2 signal",
                     bins=bins[i], weights=weights[i][2],  color="C0", **kwargs)
            plt.legend()
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Count/" + bin_size)
            plt.savefig(path + "paired_" + prop_name.replace(" ",
                        "_") + path_end + "_stacked.png")
            plt.close()

        plt.figure(figsize=(12, 9))
        plt.hist(sig_2, histtype='step', label="2 signal",
                 bins=bins[i], weights=weights[i][2], **kwargs)
        plt.hist(sig_1, histtype='step', label="1 signal",
                 bins=bins[i], weights=weights[i][1], **kwargs)
        plt.hist(sig_0, histtype='step', label="0 signal",
                 bins=bins[i], weights=weights[i][0], **kwargs)
        plt.legend()
        plt.xlabel(prop_name.title() + "/" + units)
        plt.ylabel("Count/" + bin_size)
        plt.savefig(path + "paired_" +
                    prop_name.replace(" ", "_") + path_end + ".png")
        plt.close()

        if inc_log:
            plt.figure(figsize=(12, 9))
            plt.hist(sig_2, histtype='step', label="2 signal",
                     bins=bins[i], weights=weights[i][2], **kwargs)
            plt.hist(sig_1, histtype='step', label="1 signal",
                     bins=bins[i], weights=weights[i][1], **kwargs)
            plt.hist(sig_0, histtype='step', label="0 signal",
                     bins=bins[i], weights=weights[i][0], **kwargs)
            plt.legend()
            plt.yscale('log')
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Count/" + bin_size)
            plt.savefig(path + "paired_" + prop_name.replace(" ",
                        "_") + path_end + "_log.png")
            plt.close()

        if inc_norm:
            plt.figure(figsize=(12, 9))
            plt.hist(sig_2, histtype='step', density=True, label="2 signal",
                     bins=bins[i], weights=weights[i][2], **kwargs)
            plt.hist(sig_1, histtype='step', density=True, label="1 signal",
                     bins=bins[i], weights=weights[i][1], **kwargs)
            plt.hist(sig_0, histtype='step', density=True, label="0 signal",
                     bins=bins[i], weights=weights[i][0], **kwargs)
            plt.legend()
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Density/" + bin_size)
            plt.savefig(path + "paired_" + prop_name.replace(" ",
                        "_") + path_end + "_norm.png")
            plt.close()

        if inc_log and inc_norm:
            plt.figure(figsize=(12, 9))
            plt.hist(sig_2, histtype='step', density=True, label="2 signal",
                     bins=bins[i], weights=weights[i][2], **kwargs)
            plt.hist(sig_1, histtype='step', density=True, label="1 signal",
                     bins=bins[i], weights=weights[i][1], **kwargs)
            plt.hist(sig_0, histtype='step', density=True, label="0 signal",
                     bins=bins[i], weights=weights[i][0], **kwargs)
            plt.legend()
            plt.yscale('log')
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Density/" + bin_size)
            plt.savefig(path + "paired_" + prop_name.replace(" ",
                        "_") + path_end + "_norm_log.png")
            plt.close()
    return


def plot_rank_hist(
    prop_name, ranking,
    path="/users/wx21978/projects/pion-phys/plots/photon_pairs/", unique_save_id="",
    y_scaling='log', bins=None, **kwargs
):
    """
    Produces a plot displaying a set of ranks (positions) as a histogram.

    Ranked data must already be calculated and gets passed as `ranking`.

    `prop_name` is required to correctly title and save the plot.

    `y_scaling` can be supplied as list of scalings ('log' and 'linear'). If a
    list is passed, the options supplied will be iterated through, and a plot
    will be produced for each value given.

    Parameters
    ----------
    prop_name : str
        Name of the property plotted for title and saved file.
    ranking : ak.Array or np.ndarray
        Array containing the ranks to be plotted.
    path : str, optional
        Directory in which to save the final plot(s). Default is
        '/users/wx21978/projects/pion-phys/plots/photon_pairs/'.
    unique_save_id : str, optional
        Extra string to add into the save name to avoid
        overwriting. Default is ''.
    y_scaling : str or list, optional
        Type of scalings to use on the y-axis. Should be
        'linear' or 'log'. Default is 'log'
    bins : int, np.ndarray, or None, optional
        Number of bins if ``int``, or bin edges if ``ndarray``.
        Gives one bin per ranking if None. Default is None.

    Other Parameters
    ----------------
    **kwargs
        Any additional keyword arguments to be passed to the ``plt.hist``
        calls. The same arguments will be passed to all plots.
    """

    if path[-1] != '/':
        path += '/'

    if not isinstance(y_scaling, list):
        y_scaling = [y_scaling]
    if bins is None:
        bins = int(np.max(ranking) - 1)

    for y_scale in y_scaling:
        plt.figure(figsize=(12, 9))
        plt.hist(ranking, label="signal", bins=bins,  **kwargs)
        plt.legend()
        plt.yscale(y_scale)
        plt.xlabel(prop_name.title() + " ranking")
        plt.ylabel("Count")
        plt.savefig(path + "ranking_" + prop_name.replace(" ",
                    "_") + unique_save_id + ".png")
        plt.close()
    return


def make_truth_comparison_plots(
    events, photon_indicies,
    valid_events=None,
    prop_label=None, inc_log=True,
    path="/users/wx21978/projects/pion-phys/plots/photon_pairs/truth_method_comparisions/",
    **kwargs
):
    """
    Produces a set of plots displaying the errors of a set of PFOs with respect to the truth
    PFO they are representing. Errors in energy for the leading (highest energy) and sub-
    leading (lowest energy) photon are displayed in separate plots, and the cosine of the
    angular differences between the PFOs and truth particles is displayed for both photons
    on the same plot.

    `photons_indicies` can be a dictionary to allow comparision of multiple methods of generating
    truth photons to be compared on the same plot.

    Parameters
    ----------
    events : Data
        Events from which data is gathered.
    photon_indicies : dict or np.ndarray
        Indicies of the selected best particles. May be passed as a dictionary containing
        numpy arrays labelled by the method used to generate the indicies for comparision.
    valid_events : list, np.ndarray, ak.Array(), or None
        1D set of boolean values to optionally excluded known invalid events. Default is
        None.
    prop_label : str or None, optional
        Optional name of property to appear in legend if `photons_indicies` is not a
        dictionary.
    inc_log : bool, optional
        Whether to include logarithmically scaled plots. Default is True.
    path : str, optional
        Directory in which to save the final plots. Default is
        '/users/wx21978/projects/pion-phys/plots/photon_pairs/truth_method_comparisions/'.

    Other Parameters
    ----------------
    **kwargs
        Any additional keyword arguments to be passed to the ``plt.hist`` calls. The same
        arguments will be passed to all plots.
    """
    if path[-1] != '/':
        path += '/'

    # If we don't specify valid events, assume all events are OK, so convert valid events into a
    #   slice which selects all events
    if valid_events is None:
        valid_events = slice(None)

    # Ideally, we can use the trueParticle (not trueparticleBT} data, since trueParticle is likely
    #   already loaded, and trueParticleBT likely hasn't been
    true_energies = events.trueParticlesBT.energy[valid_events]
    # true_energies_mom = vector.magnitude(events.trueParticlesBT.momentum)[valid_events]
    true_dirs = events.trueParticlesBT.direction[valid_events]

    reco_energies = events.recoParticles.shower_energy[valid_events]
    reco_dirs = events.recoParticles.shower_direction[valid_events]

    # Warning - not sure this has been tested without photon_indicies as a dictionary...
    if not isinstance(photon_indicies, dict):
        photon_indicies = {prop_label: photon_indicies}

    fig_e_i, energy_i_axis = plt.subplots(figsize=(16, 12), layout="tight")
    fig_e_ii, energy_ii_axis = plt.subplots(figsize=(16, 12), layout="tight")
    fig_dirs, directions_axis = plt.subplots(figsize=(16, 12), layout="tight")

    if inc_log:
        fig_e_i_log, energy_i_axis_log = plt.subplots(
            figsize=(16, 12), layout="tight")
        fig_e_ii_log, energy_ii_axis_log = plt.subplots(
            figsize=(16, 12), layout="tight")
        fig_dirs_log, directions_axis_log = plt.subplots(
            figsize=(16, 12), layout="tight")

    for i, prop in enumerate(list(photon_indicies.keys())):
        if isinstance(prop, str):
            y1_label = "y1 " + prop
            y2_label = "y2 " + prop
        else:
            y1_label = None
            y2_label = None

        photon_i_indicies = np_to_ak_indicies(
            photon_indicies[prop][:, 0][valid_events])
        photon_ii_indicies = np_to_ak_indicies(
            photon_indicies[prop][:, 1][valid_events])

        # This might contain some useful stuff for moving to trueParticle informatio, rather than trueParticleBT
        # err_true_photon_i = np.zeros(np.sum(valid_events))
        # photon_i_ids = events.trueParticlesBT.number[valid_events][photon_i_indicies]
        # reco_energy_full = reco_energies[photon_i_indicies]
        # index = np.arange(photon_indicies[prop][:,0].shape[0])[valid_events]
        # for j in range(np.sum(valid_events)):
        #     true_ids = events.trueParticles.number[index[j]].to_numpy()
        #     true_energy = events.trueParticles.energy[index[j]].to_numpy()

        #     true_energy_i = true_energy[true_ids == photon_i_ids[j]]
        #     # true_energy_ii = true_energy[true_ids == pfo_truth_ids[photon_ii_indicies]]
        #     err_true_photon_i[j] = (reco_energy_full[j] / true_energy_i )[0] -1

        err_energy_photon_i = (
            reco_energies[photon_i_indicies] / true_energies[photon_i_indicies]) - 1
        err_energy_photon_ii = (
            reco_energies[photon_ii_indicies] / true_energies[photon_ii_indicies]) - 1

        err_direction_photon_i = vector.dot(
            reco_dirs[photon_i_indicies], true_dirs[photon_i_indicies])
        err_direction_photon_ii = vector.dot(
            reco_dirs[photon_ii_indicies], true_dirs[photon_ii_indicies])

        # Linear
        energy_i_axis.hist(err_energy_photon_i,  label=y1_label,
                           histtype="step", bins=100**kwargs)
        energy_ii_axis.hist(err_energy_photon_ii, label=y2_label,
                            histtype="step", bins=100**kwargs)

        directions_axis.hist(err_direction_photon_i,  label=y1_label,
                             histtype="step", bins=80, color=f"C{i}"**kwargs)
        directions_axis.hist(err_direction_photon_ii, label=y2_label,
                             histtype="step", bins=80, color=f"C{i}", ls="--"**kwargs)

        if inc_log:
            # Log
            energy_i_axis_log.hist(
                err_energy_photon_i,  label=y1_label, histtype="step", bins=100**kwargs)
            energy_ii_axis_log.hist(
                err_energy_photon_ii, label=y2_label, histtype="step", bins=100**kwargs)

            directions_axis_log.hist(
                err_direction_photon_i,  label=y1_label, histtype="step", bins=50, color=f"C{i}"**kwargs)
            directions_axis_log.hist(err_direction_photon_ii, label=y2_label,
                                     histtype="step", bins=50, color=f"C{i}", ls="--"**kwargs)

    # Linear
    energy_i_axis.set_xlabel("Fractional energy error")
    energy_i_axis.set_ylabel("Count")
    energy_i_axis.legend()

    energy_ii_axis.set_xlabel("Fractional energy error")
    energy_ii_axis.set_ylabel("Count")
    energy_ii_axis.legend()

    directions_axis.set_xlabel("Best photon vs. truth dot product")
    directions_axis.set_ylabel("Count")
    directions_axis.legend(loc="upper left")

    if inc_log:
        # Log
        energy_i_axis_log.set_xlabel("Fractional energy error")
        energy_i_axis_log.set_ylabel("Count")
        energy_i_axis_log.set_yscale("log")
        energy_i_axis_log.legend()

        energy_ii_axis_log.set_xlabel("Fractional energy error")
        energy_ii_axis_log.set_ylabel("Count")
        energy_i_axis_log.set_yscale("log")
        energy_ii_axis_log.legend()

        directions_axis_log.set_xlabel("Best photon vs. truth dot product")
        directions_axis_log.set_ylabel("Count")
        energy_i_axis_log.set_yscale("log")
        directions_axis_log.legend(loc="upper left")

    fig_e_i.savefig(path + "leading_photon_energy.png")
    fig_e_ii.savefig(path + "subleading_photon_energy.png")
    fig_dirs.savefig(path + "directions.png")
    if inc_log:
        fig_e_i_log.savefig(path + "leading_photon_energy_log.png")
        fig_e_ii_log.savefig(path + "subleading_photon_energy_log.png")
        fig_dirs_log.savefig(path + "directions_log.png")
    plt.close()
    return


def plot_region_data(
        regions_dict,
        plt_cfg=PlotConfig(),
        title=None,
        compare_max=0,
        log_norm=False,
        colourblind=False):
    """
    Creates a plot of the population of the tagged regions.

    Regions should be created using `EventSelection.create_regions()`.

    Parameters
    ----------
    regions_dict : dict
        Dictionary of containing a mask of which events fall in which
        classification regions. This should be created using
        `EventSelection.creat_regions()`.
    plt_cfg : Plots.PlotConfig, optional
        `PlotConfig` instance to control plotting parameters. Default
        is a new PlotConfig instance (no showing or saving).
    title : str or None, optional
        Title to display above the plot. If None, no title is shown.
        Default is None.
    compare_max : int or float, optional
        Maximum value that will be shown on the colour bar. Used to
        keep the colour scaling consistent if comparing multiple plots.
        This will not reduce the maximum colour value if it is set
        below the highest populated region. Default is 0.
    log_norm : boolean, optional
        Whether to use log scaling in the colour mapping. Default is
        False.
    colourblind : boolean, optional
        Changes the text colour to blue to improve constrast. Default
        is False.
    """
    x = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([0, 1, 0, 1, 0, 1])
    pi_prod_multi_pi0 = ak.sum(regions_dict["pion_prod_>1_pi0"])
    weights = np.array([
        ak.sum(regions_dict["absorption"]),
        ak.sum(regions_dict["pion_prod_0_pi0"]),
        ak.sum(regions_dict["charge_exchange"]),
        ak.sum(regions_dict["pion_prod_1_pi0"]),
        pi_prod_multi_pi0,
        pi_prod_multi_pi0])
    
    setup_kwargs = {}
    if title is not None:
        setup_kwargs.update({"title":title})
    text_kwargs = {
        "fontsize":16,
        "fontweight":"demibold",
        "horizontalalignment":"center"}
    if colourblind:
        colour=(0.55, 0.8, 1)
    else:
        colour = (1, 0.65, 0.8)

    plt_cfg.setup_figure(figsize=(14,8), **setup_kwargs)
    if log_norm:
        cnorm = matplotlib.colors.LogNorm(
            vmin=1, vmax=max(np.max(weights), compare_max))
    else:
        cnorm = matplotlib.colors.Normalize(
            vmin=0, vmax=max(np.max(weights), compare_max))
    cmap = plt.get_cmap("pink")
    plt.hist2d(x, y, weights=weights, range=[[-0.5, 2.5],[-0.5, 1.5]],
               bins=[3,2], norm=cnorm, cmap=cmap)
    plt.text(0,-0.05,f"Absorption\n{weights[0]}", **text_kwargs,
             color=_adjust_text_colour(weights[0], colour, cnorm))
    plt.text(0,0.9,f"Pion production,\n0 $\pi^{0}$\n{weights[1]}", **text_kwargs,
             color=_adjust_text_colour(weights[1], colour, cnorm))
    plt.text(1,-0.05,f"Charge exchange\n{weights[2]}", **text_kwargs,
             color=_adjust_text_colour(weights[2], colour, cnorm))
    plt.text(1,0.9,f"Pion production,\n1 $\pi^{0}$\n{weights[3]}", **text_kwargs,
             color=_adjust_text_colour(weights[3], colour, cnorm))
    plt.text(2,0.4,f"Pion production,\n>1 $\pi^{0}$\n{weights[4]}", **text_kwargs,
             color=_adjust_text_colour(weights[4], colour, cnorm))
    plt_cfg.format_axis(
        legend=False, xlabel="Number of $\pi^{0}$", ylabel="Number of $\pi^{+}$")
    plt.xticks(ticks=[0, 1, 2], labels=["0", "1", ">1"], minor=False)
    plt.yticks(ticks=[0, 1], labels=["0", ">=1"], minor=False)
    plt.minorticks_off()
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Number of events", rotation=90)
    plt_cfg.end_plot()
    return


def compare_truth_reco_regions(
        reco_regions,
        truth_regions,
        plt_cfg=PlotConfig(),
        title=None,
        log_norm=False,
        colourblind=False):
    """
    Comparing the classification coincidences between some
    `reco_regions` and `truth_regions`.

    Regions should be created using `EventSelection.create_regions()`.

    Parameters
    ----------
    reco_regions : dict
        Dictionary of containing a mask of which events fall in which
        classification regions from reconstructed data. This should be
        created using `EventSelection.creat_regions()`.
    reco_regions : dict
        Dictionary of containing a mask of which events fall in which
        classification regions from truth data. This should be created
        using `EventSelection.creat_regions()`.
    plt_cfg : Plots.PlotConfig, optional
        `PlotConfig` instance to control plotting parameters. Default
        is a new PlotConfig instance (no showing or saving).
    title : str or None, optional
        Title to display above the plot. If None, no title is shown.
        Default is None.
    log_norm : boolean, optional
        Whether to use log scaling in the colour mapping. Default is
        False.
    colourblind : boolean, optional
        Changes the text colour to blue to improve constrast. Default
        is False.
    """
    index_dict = {
        "absorption":0,
        "charge_exchange":1,
        "pion_prod_0_pi0":2,
        "pion_prod_1_pi0":3,
        "pion_prod_>1_pi0":4}
    tick_labels = [
        "absorbtion", "cex.", "pi+ prod (0)", "pi+ prod (1)", "pi+ prod (>1)"]
    ytick_labels = tick_labels.copy()
    values = np.repeat(np.expand_dims(np.arange(5, dtype=float),1), 5, axis=1)
    x = values.flatten("C")
    y = values.flatten("F")
    num_events = len(list(truth_regions.values())[0])
    for key_truth in truth_regions.keys():
        truth_i = index_dict[key_truth]
        truth_count = np.sum(truth_regions[key_truth])
        ytick_labels[truth_i] += (
            f"\n{truth_count} events" +
            f"\n({100*truth_count/num_events:.1f}% of total)")
        for key_reco in reco_regions.keys():
            reco_i = index_dict[key_reco]
            values[reco_i, truth_i] = 100*np.sum(
                np.logical_and(reco_regions[key_reco],
                               truth_regions[key_truth]))/truth_count
    values = values.flatten("C")

    setup_kwargs = {}
    if title is not None:
        setup_kwargs.update({"title":title})
    text_kwargs = {
        # "fontsize":16,
        "fontweight":"semibold",
        "horizontalalignment":"center"}
    if colourblind:
        colour=(0.55, 0.8, 1)
    else:
        colour = (1, 0.65, 0.8)

    plt_cfg.setup_figure(figsize=(16,12), **setup_kwargs)
    if log_norm:
        cnorm = matplotlib.colors.LogNorm(vmin=1, vmax=100)
    else:
        cnorm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    cmap = plt.get_cmap("pink")
    plt.hist2d(x, y, weights=values, range=[[-0.5, 4.5],[-0.5, 4.5]], bins=[5,5], norm=cnorm, cmap=cmap)
    for i, val in enumerate(values):
        plt.text(i//5, i%5, f"{val:.1f}%", **text_kwargs, color=_adjust_text_colour(val, colour, cnorm))
    plt_cfg.format_axis(legend=False, xlabel="Reco classifcation", ylabel="Truth classifcation")
    plt.xticks(ticks=list(index_dict.values()), labels=tick_labels, minor=False)
    plt.yticks(ticks=list(index_dict.values()), labels=ytick_labels, minor=False)
    plt.minorticks_off()
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("% of events from the truth region", rotation=90)
    plt_cfg.end_plot()
    return
