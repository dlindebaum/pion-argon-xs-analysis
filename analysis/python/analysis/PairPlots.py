"""
Created on: 13/12/2022

Author: Dennis Lindebaum

Description: Module containing plotting data for initial pair analyses
"""

import warnings
import math
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.stats import iqr
import copy as copy_lib


class PlotConfig():

    def __init__(self):
        self.PLT_STYLE              = "fivethirtyeight"
        self.FIG_SIZE               = (14,10)
        self.FIG_FACECOLOR          = 'white'
        self.AXIS_FACECOLOR         = 'white'
        self.BINS                   = 'best'                # Number of bins
        self.BEST_BINS_MULTIPLIER   = 1                     # Allows tweaking the number of bins produced by the "best" algorithm
        self.GRID_COLOR             = 'gray'                # Color of plot grid
        self.GRID_ALPHA             = 0.25                  # Transparency of the plot grid
        self.LEGEND_COLOR           = 'white'
        self.MINOR_GRID_X           = False
        self.MINOR_GRID_Y           = False
        self.SPINE_COLOR            = 'black'
        self.TICK_SIZE              = 18
        self.TICK_SIZE_MINOR        = 14
        self.TICK_LENGTH            = 14
        self.TICK_WIDTH             = 4
        self.MINOR_TICK_SPACING_X   = "auto"
        self.MINOR_TICK_SPACING_Y   = "auto"
        self.LEGEND_SIZE            = 18
        self.LEGEND_SIZE_DOUBLE     = 11
        self.TITLE_SIZE             = 28
        self.TITLE_SIZE_DOUBLE      = 22
        self.AX_ALPHA               = 0.5
        self.TEXT_SIZE              = 18
        self.DPI                    = 300

        self.SAVE_FOLDER            = None                  # Folder to save the plot
        self.SHOW_PLOT              = False                 # Show the plot?
        self.N_HITS_WHICH           = "count"               # Count to get number of individual hits per APA. "sum" to get total of hits per APA (Summed Hit Size)
        self.LABEL_SIZE             = 24
        self.LOGSCALE               = False
        self.LINEWIDTH              = 2

        self.HIST_COLOR1            = "tab:blue"
        self.HIST_COLOR2            = "tab:orange"
        self.HIST_COLOR3            = "tab:green"
        self.HIST_COLOR4            = "tab:purple"
        self.HIST_DENSITY           = False                 # Normalise the count of the histograms
        self.HIST_TYPE              = "step"

        self.HIST_COLOURS = {
            0:self.HIST_COLOR1,
            1:self.HIST_COLOR2,
            2:self.HIST_COLOR3,
            3:self.HIST_COLOR4
        }
    
    def __str__(self):
        return "\n".join(["{} = {}".format(str(var),vars(self)[var]) for var in vars(self)])

    # def __eq__(self, other):
        
    #     if not isinstance(other, AnalysisPlotConfig):
    #         # don't attempt to compare against unrelated types
    #         print("Warning! Comparing different instances/types is not implemented")
    #         return NotImplemented
        
    #     return (
    #         super().__eq__(self)    == super().__eq__(other)    and
    #         self.PLT_STYLE          == other.PLT_STYLE          and  
    #         self.HIST_COLOR1        == other.HIST_COLOR1        and 
    #         self.FIG_SIZE           == other.FIG_SIZE           and  
    #         self.FIG_FACECOLOR      == other.FIG_FACECOLOR      and 
    #         self.BINS               == other.BINS               and
    #         self.GRID_COLOR         == other.GRID_COLOR         and  
    #         self.GRID_ALPHA         == other.GRID_ALPHA         and
    #         self.SHOW_PLT           == other.SHOW_PLT           and  
    #         self.N_HITS_WHICH       == other.N_HITS_WHICH       and 
    #         self.LABEL_SIZE         == other.LABEL_SIZE         and   
    #         self.HIST_DENSITY       == other.HIST_DENSITY       and  
    #         self.LOGSCALE           == other.LOGSCALE           
    #     )

    # def __ne__(self, other):
        
    #     if not isinstance(other, AnalysisPlotConfig):
    #         # don't attempt to compare against unrelated types
    #         print("Warning! Comparing different instances/types is not implemented")
    #         return NotImplemented
        
    #     return (
    #         super().__ne__(self)    == super().__ne__(other)    or
    #         self.PLT_STYLE          != other.PLT_STYLE          or 
    #         self.HIST_COLOR1        != other.HIST_COLOR1        or
    #         self.FIG_SIZE           != other.FIG_SIZE           or 
    #         self.FIG_FACECOLOR      != other.FIG_FACECOLOR      or
    #         self.BINS               != other.BINS               or
    #         self.GRID_COLOR         != other.GRID_COLOR         or 
    #         self.GRID_ALPHA         != other.GRID_ALPHA         or
    #         self.SHOW_PLT           != other.SHOW_PLT           or 
    #         self.N_HITS_WHICH       != other.N_HITS_WHICH       or
    #         self.LABEL_SIZE         != other.LABEL_SIZE         or  
    #         self.HIST_DENSITY       != other.HIST_DENSITY       or 
    #         self.LOGSCALE           != other.LOGSCALE
    #     )

    # def __hash__(self):
    #     return hash(
    #         (
    #             super().__init__(), 
    #             self.PLT_STYLE, 
    #             self.HIST_COLOR1,  
    #             self.FIG_SIZE, 
    #             self.FIG_FACECOLOR,
    #             self.BINS,   
    #             self.GRID_COLOR,
    #             self.GRID_ALPHA,  
    #             self.SHOW_PLT,  
    #             self.N_HITS_WHICH, 
    #             self.LABEL_SIZE,
    #             self.HIST_DENSITY, 
    #             self.LOGSCALE
    #         )
    #     )

    def copy(self):
        return copy_lib.deepcopy(self)

    def setup_figure(self, sub_rows=1, sub_cols=1, title=None, **kwargs):
        # TODO add a scaling function to adjust all test sizes etc, base on either single scalling, or input dimensions

        plt.style.use(self.PLT_STYLE)

        fig_kwargs = {
            "figsize": self.FIG_SIZE,
            "facecolor": self.FIG_FACECOLOR
        }
        fig_kwargs.update(kwargs)

        fig, axs = plt.subplots(sub_rows, sub_cols, **fig_kwargs)

        if title is not None:
            plt.title(title, fontdict={"fontsize": self.TITLE_SIZE})

        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for ax in axs.flatten():
            ax.patch.set_alpha(self.AX_ALPHA)
            
            ax.tick_params(axis='both', which='major', labelsize=self.TICK_SIZE)
            ax.tick_params(axis='both', direction="out", length=self.TICK_LENGTH, width=self.TICK_WIDTH, grid_color=self.GRID_COLOR, grid_alpha=self.GRID_ALPHA)

            if self.MINOR_TICK_SPACING_X == "auto":
                ax.xaxis.set_minor_locator(AutoMinorLocator())
            else:
                ax.xaxis.set_minor_locator(MultipleLocator(self.MINOR_TICK_SPACING_X))
            if self.MINOR_TICK_SPACING_Y == "auto":
                ax.yaxis.set_minor_locator(AutoMinorLocator())
            else:
                ax.yaxis.set_minor_locator(MultipleLocator(self.MINOR_TICK_SPACING_X))
            ax.tick_params(axis="both", which='minor', labelsize=self.TICK_SIZE-2, length=self.TICK_LENGTH-2)

            ax.xaxis.grid(self.MINOR_GRID_X, which='minor')
            ax.yaxis.grid(self.MINOR_GRID_Y, which='minor')

            if self.LOGSCALE:
                ax.set_yscale("log")

        return fig, axs
      
    def gen_kwargs(self, type="plot", index=0, **kwargs):
        final_kwargs = {
            "linewidth"    : self.LINEWIDTH
        }
        
        if type == "hist":
            final_kwargs.update({
                "bins"          : self.BINS,
                "density": self.HIST_DENSITY,
                "histtype": self.HIST_TYPE
            })
            if index <= max(self.HIST_COLOURS.keys()):
                final_kwargs.update({"color":self.HIST_COLOURS[index]})
        elif type == "line":
            final_kwargs.pop("linewidth")
            final_kwargs.update({
                "linewidth" : self.LINEWIDTH,
                "colors"    : "red"
            })

        final_kwargs.update(kwargs)

        return final_kwargs

    def format_axis(self, *ax, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=True, xlog=None, ylog=None):
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
                plt.legend(prop={'size': self.LEGEND_SIZE}, facecolor=self.LEGEND_COLOR)

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
                if legend:
                    a.legend(prop={'size': self.LEGEND_SIZE}, facecolor=self.LEGEND_COLOR)

        return

    def end_plot(self, save_path=None):   
        # Need to get this to work with multiple figures open
        
        plt.tight_layout()   

        if save_path is not None:
            if "/" in save_path:
                plt.savefig(save_path, bbox_inches='tight', facecolor=self.FIG_FACECOLOR, dpi=self.DPI)
            elif self.SAVE_FOLDER is not None:
                if self.SAVE_FOLDER[-1] != "/":
                    self.SAVE_FOLDER += "/"
                plt.savefig(self.SAVE_FOLDER + save_path, bbox_inches='tight', facecolor=self.FIG_FACECOLOR, dpi=self.DPI)
            else:
                warnings.warn("Plot has been given a file name to save, but not path to folder in the PlotConfig")

        if self.SHOW_PLOT:
            plt.show()
        
        if self.SHOW_PLOT or (save_path is not None):
            plt.close()
        
        return

    def get_bins(self, data, array=False):
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
        bin_width = 2*iqrange/len(data)**(1/3)
        bins = int(round((np.max(data) - np.min(data))/bin_width))
        return bins, bin_width

    @staticmethod
    def expand_bins(bins, data):
        if min(data) < bins[0]:
            low_bin_width = bins[1] - bins[0]
            bins = np.concatenate((np.arange(bins[0] - low_bin_width, min(data) - low_bin_width, low_bin_width)[::-1], bins))
        if max(data) >= bins[-1]:
            high_bin_width = bins[-1] - bins[-2]
            # TODO nicer way to do this?
            bins = np.concatenate((bins, np.arange(bins[-1]+ high_bin_width, max(data) + high_bin_width*(1+1e-6), high_bin_width)))
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
            low_bins = np.linspace(curr_min - new_bins_count*bin_widths_low, curr_min - bin_widths_low, new_bins_count)
            
            empty_bin_data = np.zeros(new_bins_count, dtype=data.dtype)

            bins = np.concatenate((low_bins, bins))
            data = np.concatenate((empty_bin_data, data))
        
        if curr_max < new_max:
            new_bins_count = math.ceil((new_max - curr_max)/bin_widths_high)
            high_bins = np.linspace(curr_max + bin_widths_high, curr_max + new_bins_count*bin_widths_high, new_bins_count)

            empty_bin_data = np.zeros(new_bins_count, dtype=data.dtype)

            bins = np.concatenate((bins, high_bins))
            data = np.concatenate((data,empty_bin_data))
        
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
            counts, hist_bins = np.histogram(data, bins=bins, range=range, weights=weights)
            
            self.bins.update({hist_name:hist_bins})
            self.binned_data.update({hist_name:counts})

        else:
            if range is None:
                self._extend_bins(data, hist_name)
            counts, hist_bins = np.histogram(data, bins=self.bins[hist_name], range=range, weights=weights)

            if (hist_bins != self.bins[hist_name]).all():
                raise AssertionError("Bins from histogram do not match sorted bins.")
            new_binned_data = self.binned_data[hist_name] + counts
            self.binned_data.update({hist_name:new_binned_data})

        return self.binned_data[hist_name], self.bins[hist_name]

    def plot_hist(self, *ax, hist_name=None, plot_config=None, **plot_kwargs):
        if plot_config is None:
            plot_config = self.plt_cfg
        plot_config_kwargs = {
            "type"      : "hist",
            "bins"      : self.bins[hist_name],
            "weights"   : self.binned_data[hist_name]
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
        Produces a figure using the histogram(s) data from hist_name.

        Parameters
        ----------
        hist_name : list or str, optional
        xlabel
        ylabel
        title
        """
        if not isinstance(hist_name, list):
            hist_name = [hist_name]
        use_labels = len(hist_name) != 1

        fig, axes = self.plt_cfg.setup_figure(title = title)

        for h_name in hist_name:
            self.plot_hist(axes[0], hist_name=h_name, normed=self.plt_cfg.HIST_DENSITY, labelled=use_labels)
        
        self.plt_cfg.format_axis(axes[0], xlabel=xlabel, ylabel=ylabel)
        self.plt_cfg.end_plot(save_path=save_name)
        return fig, axes


class PairHistsBatchPlotter(HistogramBatchPlotter):
    def __init__(
        self, 
        prop_name, units,
        plot_config=None, unique_save_id="",
        inc_stacked=False, inc_norm=True, inc_log=True,
        # bin_size = None,
        bins = 100, range=None,
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
            self._plot_types.update({"norm_log": self._plot_types["norm"].copy()})
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
            raise ValueError(f"Bins length {len(bins)} and ranges length {len(range)} incompatible")
        
        # Get the larger number of supplied values
        num_repeats = max(len(range), len(bins))
        # Tile the smaller one until it has the length of the larger one
        range = range * (num_repeats // len(range))
        bins = bins * (num_repeats // len(bins))
        
        # Put the values in the list of parameters to run over
        self._range_bins_list = [ {"range": range[i], "bins": bins[i]} for i in np.arange(num_repeats)]

        # CURRENTLY NOT USED: see comment at start of __init__
        # if not isinstance(bin_size, list):
        #     bin_size = [bin_size]
        # if len(bin_size) != 1 and len(bin_size) != len(bins):
        #     raise ValueError(f"Bins length {len(bins)} and bin sizes length {len(bin_size)} incompatible")


        # Additional plotting parameters
        self._plot_kwargs = kwargs


        # Naming
        self.image_save_base = "paired_" + prop_name.replace(" ", "_") + unique_save_id + "_hist"
        # self.units = units
        self.plot_x_label = prop_name.title() + "/" + units # self.units

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
        if self.includes_stacked: # Need to keep track of the raw weights if producing a stacked plot
            raw_weights = weights
        if weights is None:
            weights = {2:None, 1:None, 0:None}
        else:
            weights = {2:ak.ravel(weights[sig_count == 2]), 1:ak.ravel(weights[sig_count == 1]), 0:ak.ravel(weights[sig_count == 0])}
        
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
                
                super().add_batch(two_signal                               , hist_name=self._stacked_names[0] + f"_ind_{index}", weights=weights[2]     , **params)
                super().add_batch(ak.ravel(data[sig_count != 0]).to_numpy(), hist_name=self._stacked_names[1] + f"_ind_{index}", weights=no_bkg_weights , **params)
                super().add_batch(ak.ravel(data).to_numpy()                , hist_name=self._stacked_names[2] + f"_ind_{index}", weights=all_weights    , **params)
            
            # Only need to do this once, since the changes between logs etc. are cosmetic
            super().add_batch(ak.ravel(data[sig_count == 0]).to_numpy(), hist_name=self._standard_names[0] + f"_ind_{index}", weights=weights[0], **params)
            super().add_batch(ak.ravel(data[sig_count == 1]).to_numpy(), hist_name=self._standard_names[1] + f"_ind_{index}", weights=weights[1], **params)
            super().add_batch(two_signal                               , hist_name=self._standard_names[2] + f"_ind_{index}", weights=weights[2], **params)
        
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
                self._stacked_config.format_axis(stack_axes[0], xlabel=self.plot_x_label, ylabel="Count")
                self._stacked_config.end_plot(save_path = self.image_save_base + save_path_end + "_stacked.png")

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
                self._plot_types[plot_type].format_axis(axes[0], xlabel=self.plot_x_label, ylabel=ylabel)
                self._plot_types[plot_type].end_plot(save_path = self.image_save_base + save_path_end + "_" + plot_type + ".png")
        
        return