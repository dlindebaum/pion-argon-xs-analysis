"""
Created on: 02/06/2023 10:37

Author: Shyam Bhuller

Description: Library for code used in the cross section analysis. Refer to the README to see which apps correspond to the cross section analysis.
"""
import os

from dataclasses import dataclass, field

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from particle import Particle
from scipy.stats import chi2
from scipy.interpolate import interp1d

from python.analysis import BeamParticleSelection, PFOSelection, EventSelection, SelectionTools, Fitting, Plots, vector, Tags, Processing, BetheBloch, ThinSlice, EnergySlice, NtupleProcessing, Slices
from python.analysis.Master import LoadObject, SaveObject, ReadHDF5, Data, timer, IO, FileDescriptor
from python.analysis.DetectorGeometry import ProtoDUNESPGeometry
from python.analysis.Utils import *

GEANT_XS = os.environ["PYTHONPATH"] + "/data/g4_xs_pi_KE_100.root"
# GEANT_XS = os.environ["PYTHONPATH"] + "/data/g4_xs.root"

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
        Plots.plt.style.use('ggplot')
        Plots.plt.rcParams.update({'patch.linewidth': 1})
        Plots.plt.rcParams.update({'font.size': font_scale * 10})
        Plots.plt.rcParams.update({"axes.titlecolor" : "#555555"})
        Plots.plt.rcParams.update({"axes.titlesize" : font_scale * 12})
        Plots.plt.rcParams['figure.dpi'] = dpi
        Plots.plt.rcParams['legend.fontsize'] = "small"
        Plots.plt.rcParams["font.family"] = font_style

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


def SetPlotStyle(extend_colors : bool = False, custom_colors : list = None, dpi : int = 300, dark : bool = False, font_scale : float = 1, font_style : str = "sans"):
    Plots.plt.style.use("default") # first load the default to reset any previous changes made by other styles
    Plots.plt.style.use('ggplot')
    Plots.plt.rcParams.update({'patch.linewidth': 1})
    Plots.plt.rcParams.update({'font.size': font_scale * 10})
    Plots.plt.rcParams.update({"axes.titlecolor" : "#555555"})
    Plots.plt.rcParams.update({"axes.titlesize" : font_scale * 12})
    Plots.plt.rcParams['figure.dpi'] = dpi
    Plots.plt.rcParams['legend.fontsize'] = "small"
    Plots.plt.rcParams["font.family"] = font_style

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


def KE(p, m):
    return (p**2 + m**2)**0.5 - m


def Efficiency(selected_count : np.array, total_count : np.array) -> tuple[np.array, np.array]:
    """ Calcualtes selection efficiency and binomial error.

    Args:
        selected_count (np.array): number of selected events
        total_count (np.array): number of total events

    Returns:
        tuple[np.array, np.array]: efficiency, error
    """
    p = selected_count / total_count
    p = np.nan_to_num(p)
    error = (p * (1 - p) / total_count)**0.5
    return p, error


def IsScraper(mc : Data, beam_scraper_args : dict) -> ak.Array:
    beam_inst_KE = KE(mc.recoParticles.beam_inst_P, Particle.from_pdgid(211).mass) # get kinetic energy from beam instrumentation
    true_ffKE = mc.trueParticles.beam_KE_front_face

    delta_KE_upstream = beam_inst_KE - true_ffKE

    scraper_ids = {}
    for k, v in beam_scraper_args.items():
        scraper_ids[k] = (beam_inst_KE > min(v["bins"])) & (beam_inst_KE < max(v["bins"]))
        threshold = v["mu_e_res"] + 3 * abs(v["sigma_e_res"])
        
        scraper_ids[k] = scraper_ids[k] & (delta_KE_upstream > threshold)
    scraper_ids = SelectionTools.CombineMasks(scraper_ids, operator = "or")
    return scraper_ids


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
    if hasattr(energy_slice.width, "__iter__"):
        width = energy_slice.width[:-1][::-1]
        xs_sim = GeantCrossSections(energy_range = [energy_slice.min_pos - energy_slice.width[0], energy_slice.max_pos + energy_slice.width[-1]])
    else:
        width = energy_slice.width
        xs_sim = GeantCrossSections(energy_range = [energy_slice.min_pos - energy_slice.width, energy_slice.max_pos + energy_slice.width])

    if colors is None:
        colors = {k : f"C{i}" for i, k in enumerate(xs)}

    sim_curve_interp = xs_sim.GetInterpolatedCurve(process)
    x = energy_slice.pos[:-1] - width/2

    if newFigure is True: Plots.plt.figure()
    chi_sqrs = {}
    for k, v in xs.items():
        w_chi_sqr = weighted_chi_sqr(v[0], sim_curve_interp(x), v[1])
        chi_sqrs[k] = w_chi_sqr
        if (chi2 is True) and (cv_only is False):
            chi2_l = ", $\\chi^{2}/ndf$ = " + f"{w_chi_sqr:.3g}"
        else:
            chi2_l = ""
        Plots.Plot(x, v[0], xerr = width / 2  if cv_only is False else None, yerr = v[1] if cv_only is False else None, label = k + chi2_l, color = colors[k], linestyle = "", marker = "x", newFigure = False, markersize = marker_size, capsize = marker_size/2)
    
    if process == "single_pion_production":
        Plots.Plot(xs_sim.KE, sim_curve_interp(xs_sim.KE), label = simulation_label, title = "Single pion production" if title is None else title.capitalize(), newFigure = False, xlabel = "$KE (MeV)$", ylabel = "$\\sigma$ (mb)", color = xs_sim_color)
    else:
        xs_sim.Plot(process, label = simulation_label, color = xs_sim_color, title = title.capitalize() if type(title) is str else title)

    Plots.plt.ylim(0)
    if max(Plots.plt.gca().get_ylim()) > np.nanmax(sim_curve_interp(xs_sim.KE).astype(float)) * 2:
        Plots.plt.ylim(0, max(sim_curve_interp(xs_sim.KE)) * 2)
    if hasattr(width, "__iter__"):
        Plots.plt.xlim(energy_slice.min_pos - (0.2 * width[0]), energy_slice.max_pos + (0.2 * width[-1]))
    else:
        Plots.plt.xlim(energy_slice.min_pos - (0.2 * width), energy_slice.max_pos + (0.2 * width))
    return chi_sqrs


class EnergyCorrection:
    @staticmethod
    def LinearCorrection(x, p0):
        return x / p0

    class ResponseFit(Fitting.FitFunction):
        n_params = 3

        @staticmethod
        def func(x, p0, p1, p2):
            return p0 * np.log(x - p1) + p2

        @staticmethod
        def p0(x, y):
            return None

    @staticmethod
    def ResponseCorrection(x, p0, p1, p2):
        return x / (EnergyCorrection.ResponseFit.func(x, p0, p1, p2) + 1)

    shower_energy_correction = {
        "linear" : LinearCorrection,
        "response": ResponseCorrection,
        None : None
    }


def TrackPitch(tracks : ak.Array) -> ak.Array:
    return vector.dist(tracks[:, :-1], tracks[:, 1:])


def TrackLength(tracks : ak.Array = None, pitches : ak.Array = None) -> ak.Array:
    if (tracks is None) & (pitches is not None):
        p = pitches
    elif (tracks is not None) & (pitches is None):
        p = TrackPitch(tracks)
    else:
        raise Exception("Track length requires either tracks or pitches to be supplied.")
    return ak.fill_none(ak.sum(p, -1), 0)


def TruncateTrack(tracks : ak.Array, x_trunc = None, y_trunc = None, z_trunc = None):
    trunc = {"x" : x_trunc, "y" : y_trunc, "z" : z_trunc}

    masks = {}
    for k, v in trunc.items():
        if v is None: continue
        masks[k] = (SelectionTools.cuts_to_func(trunc[k], "<")(tracks[k]))
    mask = SelectionTools.CombineMasks(masks, "and")

    index = ak.local_index(mask)
    max_ind = ak.argmin(~mask == False, -1) 
    max_ind = ak.where(max_ind == 0, ak.num(mask), max_ind)
    truncated_tracks = tracks[index <= max_ind]
    return truncated_tracks


def UpstreamEnergyLoss(KE_inst : ak.Array, params : np.ndarray, function : Fitting.FitFunction = Fitting.poly2d) -> ak.Array:
    """ compute the upstream loss based on a repsonse function and it's fit parameters.

    Args:
        KE_inst (ak.Array): kinetic energy measured by the beam instrumentation
        function (Fitting.FitFunction): repsonse function, defaults to Fitting.poly2d. 
        params (np.ndarray): function paramters

    Returns:
        ak.Array: upstream energy loss
    """
    return function.func(KE_inst, **params)


def RecoEndEnergy(tracks : ak.Array, KE_init: ak.Array, dEdX : ak.Array | None, method : str) -> ak.Array:
    """ Calculates the energy deposited by the beam particle in the TPC, either using calorimetric information or the bethe bloch formula (track information).

    Args:
        tracks (ak.Array): Track trajectory points.
        KE_init (ak.Array): Initial kinetic energy.
        dEdX (ak.Array | None): dEdX for each hit deposition, note this is only used for calo method.
        method (str): method to calcualte the deposited energy, either "calo" or "track".

    Returns:
        ak.Array: deposited energy
    """
    pitches = TrackPitch(tracks)

    if method == "calo":
        dE = ak.sum(dEdX[:, :-1] * pitches, -1)
        KE_end = KE_init - dE
    elif method == "track":
        track_length = TrackLength(pitches=pitches)
        KE_end = BetheBloch.InteractingKE(KE_init, track_length, 50)
    else:
        raise Exception(f"{method} not a valid method, pick 'calo' or 'track'")
    return KE_end


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
            Plots.Plot(self.KE, getattr(self, k), label = remove_(k), newFigure = False, xlabel = "$KE$ (MeV)", ylabel = "$\\sigma$ (mb)", title = title)
            # Plots.plt.fill_between(self.KE, getattr(self, k) - self.Stat_Error(k), getattr(self, k) + self.Stat_Error(k), color = Plots.plt.gca()._get_lines.get_next_color())


    def Plot(self, xs : str, color : str = None, label : str = None, title : str = None, simplified_pion_production : bool = False):
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
            elif xs == "pion_production" and simplified_pion_production is True:
                y = self.quasielastic + self.double_charge_exchange + self.pion_production
            else:
                y = getattr(self, xs)
            Plots.Plot(self.KE, y, label = label, title = title, newFigure = False, xlabel = "$KE$ (MeV)", ylabel = "$\\sigma$  (mb)", color = color)
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


    def NInteract(self, energy_slice : Slices, process : np.ndarray, mask : np.ndarray = None, weights : np.ndarray = None) -> np.ndarray:
        """ Exclusive interaction histogram using energy slice method.

        Args:
            energy_slice (Slices): energy slices
            process (np.ndarray): exclusive process mask
            mask (np.ndarray, optional): additional mask to apply. Defaults to None.
            weights (np.ndarray, optional): event weights. Defaults to None.

        Returns:
            np.ndarray: exclusive interaction histogram
        """
        if mask is None: mask = np.ones(len(self.df), dtype = bool)
        w = weights if weights is None else weights[mask]
        n_interact = EnergySlice.CountingExperiment(self.df.KE_int_smeared[mask].values, self.df.KE_init_smeared[mask].values, self.outside_tpc_smeared[mask].values, process[mask].values, energy_slice, interact_only = True, weights = w)
        return n_interact


@dataclass
class AnalysisInput:
    # masks
    regions : dict[np.ndarray]
    inclusive_process : dict[np.ndarray]
    exclusive_process : dict[np.ndarray]
    process_id : np.ndarray = field(init=False)
    region_id : np.ndarray = field(init=False)
    outside_fv_reco : np.ndarray
    outside_fv_true : np.ndarray
    outside_tpc_reco : np.ndarray
    outside_tpc_true : np.ndarray
    # observables
    track_length_reco : np.ndarray
    track_length_truncated_reco : np.ndarray
    end_x_reco : np.ndarray
    end_y_reco : np.ndarray
    end_z_reco : np.ndarray
    start_x_reco : np.ndarray
    start_y_reco : np.ndarray
    start_z_reco : np.ndarray
    P_inst_reco : np.ndarray
    KE_inst_reco : np.ndarray
    KE_int_reco : np.ndarray
    KE_init_reco : np.ndarray
    KE_ff_reco : np.ndarray
    KE_end_reco : np.ndarray
    mean_track_score : np.ndarray
    track_length_true : np.ndarray
    track_length_truncated_true : np.ndarray
    end_x_true : np.ndarray
    end_y_true : np.ndarray
    end_z_true : np.ndarray
    start_x_true : np.ndarray
    start_y_true : np.ndarray
    start_z_true : np.ndarray
    KE_int_true : np.ndarray
    KE_init_true : np.ndarray
    KE_ff_true : np.ndarray
    KE_end_true : np.ndarray
    # extras
    weights : np.ndarray = None
    event_num : np.ndarray = None
    run : np.ndarray = None
    sub_run : np.ndarray = None

    @property
    def has_regions(self):
        return type(self.regions) == dict

    @property
    def has_exclusive_process(self):
        return type(self.exclusive_process) == dict 

    @property
    def region_labels(self):
        if not self.has_regions:
            raise Exception("Analysis input does not have well defined regions.")
        return list(self.regions.keys())

    @property
    def process_labels(self):
        if not self.has_exclusive_process:
            raise Exception("Analysis input does not have well defined exclusive processes.")
        return list(self.exclusive_process.keys())


    def __post_init__(self):
        # need to set defaults for fields otherwise they are undefined (and python doesnt complain...)
        self.region_id = None
        self.process_id = None

        if self.regions is not None:
            self.region_id = self.IDFromSamples(self.regions)
        if self.exclusive_process is not None:
            self.process_id = self.IDFromSamples(self.exclusive_process)


    def __len__(self):
        for o in ["outside_tpc_reco", "outside_tpc_true", "track_length_reco", "KE_int_reco", "KE_init_reco", "mean_track_score", "track_length_true", "KE_int_true", "KE_init_true", "weights"]:
            if getattr(self, o) is not None:
                return len(getattr(self, o))
        return


    def ToFile(self, file : str):
        """ Save to dill file.

        Args:
            file (str): file path.
        """
        SaveObject(file, self)
        return


    def ToROOTFile(self, file : str):
        """ Save to a ROOT File as a flat TTree.

        Args:
            file (str): file path.
        """
        file_writer = IO(file)
        file_writer.WriteData(vars(self), None, True)
        return


    def ToSplitROOTFiles(self, directory : str, name : str):
        """ Save to regions and samples to multiple ROOT Files, each as a flat TTree.
            Compatible with MaCh3 ROOT input files.

        Args:
            file (str): file path.
        """
        dir_name = f"{directory}/root_analysis_input_{name}"
        os.makedirs(dir_name, exist_ok = True)

        #* create different files for a combination of regions and processes, could add support for more splits.
        if self.has_regions: # Data/MC.
            for r in self.region_labels:
                new_sample = self.SelectSample(self.regions[r])
                new_sample.ToROOTFile(f"{dir_name}/{name}_R{r}")
        else: # cheated MC
            for t in self.process_labels:
                new_sample = self.SelectSample(self.exclusive_process[t])
                new_sample.ToROOTFile(f"{dir_name}/{name}_T{t}")
        return

    @staticmethod
    def FromFile(file : str) -> "AnalysisInput": #* seems a bit extra but why not
        """ Load analysis input from dill file.

        Args:
            file (str): file path.

        Returns:
            AnalysisInput: analysis input.
        """
        obj = LoadObject(file)
        if type(obj) == AnalysisInput:
            return obj
        else:
            raise Exception("not an analysis input file")


    def NInteract(self, energy_slice : Slices, process: np.ndarray, mask : np.ndarray = None, reco : bool = True, weights : np.ndarray = None) -> np.ndarray:
        """ Calculate exclusive interaction histogram using the energy slice method.

        Args:
            energy_slice (Slices): energy slices
            process (np.ndarray): exclusive process mask
            mask (np.ndarray, optional): additional mask. Defaults to None.
            reco (bool, optional): use reco KE?. Defaults to True.
            weights (np.ndarray, optional): event weights. Defaults to None.

        Returns:
            np.ndarray: exclusive interaction histogram.
        """
        if mask is None: mask = np.ones(len(self.KE_int_reco), dtype = bool)
        if reco is True:
            KE_int = self.KE_int_reco
            KE_init = self.KE_init_reco
            outside_fv = self.outside_fv_reco
        else:
            KE_int = self.KE_int_true
            KE_init = self.KE_init_true
            outside_fv = self.outside_fv_true
        n_interact = EnergySlice.CountingExperiment(KE_int[mask], KE_init[mask], outside_fv[mask], process[mask], energy_slice, interact_only = True, weights = weights[mask] if weights is not None else weights)
        return n_interact


    def IDFromSamples(self, sample_masks : dict[np.ndarray], uncategorised : int = -1) -> ak.Array:
        """ Create ID from sample masks. Counts from 0 and includes override for uncategorised events.

        Args:
            sample_masks (dict[np.ndarray]): sample masks, such as reco regions.
            uncategorised (int, optional): id for uncategorised objects. Defaults to -1.

        Returns:
            ak.Array: _description_
        """
        id = ak.zeros_like(list(sample_masks.values())[0]) + uncategorised
        for i, v in enumerate(sample_masks.values()):
            id = ak.where(v, i, id)
        return id


    @staticmethod
    def CreateAnalysisInputToy(toy : Toy) -> "AnalysisInput":
        """ Create analysis input from a toy sample.

        Args:
            toy (Toy): toy sample

        Returns:
            AnalysisInput: analysis input object.
        """
        inclusive_events = np.array((toy.df.inclusive_process != "decay").values)

        regions = {k : np.array(v.values) for k, v in toy.reco_regions.items()}
        process = {k : np.array(v.values) for k, v in toy.truth_regions.items()}

        return AnalysisInput(
            regions = regions,
            inclusive_process = inclusive_events,
            exclusive_process = process,
            outside_fv_reco = np.array(toy.outside_tpc_smeared.values),
            outside_fv_true = np.array(toy.outside_tpc.values),
            outside_tpc_reco = np.array(toy.outside_tpc_smeared.values),
            outside_tpc_true = np.array(toy.outside_tpc.values),
            track_length_reco = np.array(toy.df.z_int_smeared.values),
            track_length_truncated_reco = None,
            end_x_reco = None,
            end_y_reco = None,
            end_z_reco = np.array(toy.df.z_int_smeared.values),
            start_x_reco = None,
            start_y_reco = None,
            start_z_reco = None,
            P_inst_reco = None,
            KE_inst_reco = None,
            KE_int_reco = np.array(toy.df.KE_int_smeared.values),
            KE_init_reco = None,
            KE_ff_reco = np.array(toy.df.KE_init_smeared.values),
            KE_end_reco = np.array(toy.df.KE_init_smeared.values),
            mean_track_score = np.array(toy.df.mean_track_score.values),
            track_length_true = np.array(toy.df.z_int.values),
            track_length_truncated_true = None,
            end_x_true = None,
            end_y_true = None,
            end_z_true = np.array(toy.df.z_int.values),
            start_x_true = None,
            start_y_true = None,
            start_z_true = None,
            KE_int_true = np.array(toy.df.KE_int.values),
            KE_init_true = None,
            KE_ff_true = np.array(toy.df.KE_init.values),
            KE_end_true = np.array(toy.df.KE_init.values),
            weights = None,
            event_num = None,
            run = None,
            sub_run = None,
            )

    @staticmethod
    def CreateAnalysisInputNtuple(events : Data, upstream_energy_loss_params : dict, reco_regions : dict[np.ndarray] = None, true_regions : dict[np.ndarray] = None, mc_reweight_params : dict = None, mc_reweight_stength : float = 3, fiducial_volume : list[float] = [0, 700], upstream_loss_func : callable = Fitting.poly2d, energy_method : str = "track") -> "AnalysisInput":
        """ Create analysis input from an ntuple sample.

        Args:
            events (Data): ntuple sample
            upstream_energy_loss_params (dict): upstream energy loss correction
            reco_regions (dict[np.ndarray]): reco region masks
            true_regions (dict[np.ndarray], optional): true process masks. Defaults to None.
            mc_reweight_params (dict, optional): mc reweight parameters. Defaults to None.

        Returns:
            AnalysisInput: analysis input.
        """
        if mc_reweight_params is not None:
            weights = RatioWeights(events.recoParticles.beam_inst_P, "gaussian", mc_reweight_params, mc_reweight_stength)
        else:
            weights = None

        KE_inst_reco = KE(events.recoParticles.beam_inst_P, Particle.from_pdgid(211).mass)
        upstream_loss_reco = UpstreamEnergyLoss(KE_inst_reco, upstream_energy_loss_params, upstream_loss_func)
        KE_ff_reco = KE_inst_reco - upstream_loss_reco

        if min(fiducial_volume) > 0:
            KE_init_reco = BetheBloch.InteractingKE(KE_ff_reco, min(fiducial_volume) * np.ones_like(KE_ff_reco), 50) # initial kinetic energy in the fiducial volume
        else:
            KE_init_reco = KE_ff_reco

        KE_int_reco = RecoEndEnergy(events.recoParticles.beam_calo_pos, KE_ff_reco, events.recoParticles.beam_dEdX, energy_method)

        truncated_track_reco = TruncateTrack(events.recoParticles.beam_calo_pos, z_trunc = max(fiducial_volume))
        track_length_truncated_reco = TrackLength(tracks=truncated_track_reco)
        KE_end_reco = RecoEndEnergy(truncated_track_reco, KE_ff_reco, events.recoParticles.beam_dEdX, energy_method)

        track_length_reco = events.recoParticles.beam_track_length
        outside_tpc_reco = ProtoDUNESPGeometry().outside_tpc(events.recoParticles.beam_endPos_SCE.x, events.recoParticles.beam_endPos_SCE.y, events.recoParticles.beam_endPos_SCE.z)

        outside_fv_reco = (events.recoParticles.beam_endPos_SCE.z < min(fiducial_volume)) | (events.recoParticles.beam_endPos_SCE.z > max(fiducial_volume))
        start_pos_reco = events.recoParticles.beam_startPos_SCE
        end_pos_reco = events.recoParticles.beam_endPos_SCE

        if true_regions is not None:
            KE_ff_true = events.trueParticles.beam_KE_front_face

            if min(fiducial_volume) > 0:
                KE_init_true = BetheBloch.InteractingKE(KE_ff_true, min(fiducial_volume) * np.ones_like(KE_ff_true), 50) # initial kinetic energy in the fiducial volume
            else:
                KE_init_true = KE_ff_true


            KE_int_true = events.trueParticles.beam_traj_KE[:, -2]
            track_length_true = events.trueParticles.beam_track_length
            start_pos_true = events.trueParticles.beam_traj_pos[:, 0]
            end_pos_true = events.trueParticles.beam_traj_pos[:, -1]

            outside_tpc_true = ProtoDUNESPGeometry().outside_tpc(events.trueParticles.endPos.x[:, 0], events.trueParticles.endPos.y[:, 0], events.trueParticles.endPos.z[:, 0])


            outside_fv_true = (events.trueParticles.beam_traj_pos.z[:, -1] < min(fiducial_volume)) | (events.trueParticles.beam_traj_pos.z[:, -1] > max(fiducial_volume))
            inelastic = events.trueParticles.true_beam_endProcess == "pi+Inelastic"

            truncated_tracks_true = TruncateTrack(events.trueParticles.beam_traj_pos[events.trueParticles.in_tpc_z], z_trunc = max(fiducial_volume))
            track_length_truncated_true = TrackLength(tracks=truncated_tracks_true)

            traj_KE = events.trueParticles.beam_traj_KE[events.trueParticles.in_tpc_z]
            KE_end_true = traj_KE[ak.local_index(traj_KE) == (ak.num(truncated_tracks_true)-2)]
            KE_end_true = ak.ravel(ak.fill_none(ak.pad_none(KE_end_true, 1, -1), -999, None)) # current null value for invalid true tracks is -999


        else:
            KE_int_true = None
            KE_init_true = None
            KE_ff_true = None
            track_length_true = None
            outside_fv_true = None
            outside_tpc_true = None
            inelastic = None
            start_pos_true = vector.vector([None], [None], [None])
            end_pos_true = vector.vector([None], [None], [None])
            KE_end_true = None
            track_length_truncated_true = None


        mean_track_score = ak.fill_none(ak.mean(events.recoParticles.track_score, axis = -1), -0.05) # fill null values in case empty events are supplied

        return AnalysisInput(
            regions = reco_regions,
            inclusive_process = inelastic,
            exclusive_process = true_regions,
            outside_fv_reco = outside_fv_reco,
            outside_fv_true = outside_fv_true,
            outside_tpc_reco = outside_tpc_reco,
            outside_tpc_true = outside_tpc_true,
            track_length_reco = track_length_reco,
            track_length_truncated_reco = track_length_truncated_reco,
            end_x_reco = end_pos_reco.x,
            end_y_reco = end_pos_reco.y,
            end_z_reco = end_pos_reco.z,
            start_x_reco = start_pos_reco.x,
            start_y_reco = start_pos_reco.y,
            start_z_reco = start_pos_reco.z,
            P_inst_reco = events.recoParticles.beam_inst_P,
            KE_inst_reco = KE_inst_reco,
            KE_int_reco = KE_int_reco,
            KE_init_reco = KE_init_reco,
            KE_ff_reco = KE_ff_reco,
            KE_end_reco = KE_end_reco,
            mean_track_score = mean_track_score,
            track_length_true = track_length_true,
            track_length_truncated_true = track_length_truncated_true,
            end_x_true = end_pos_true.x,
            end_y_true = end_pos_true.y,
            end_z_true = end_pos_true.z,
            start_x_true = start_pos_true.x,
            start_y_true = start_pos_true.y,
            start_z_true = start_pos_true.z,
            KE_int_true = KE_int_true,
            KE_init_true = KE_init_true,
            KE_ff_true = KE_ff_true,
            KE_end_true = KE_end_true,
            weights = weights,
            event_num = events.eventNum,
            run = events.run,
            sub_run = events.subRun,
            )

    @staticmethod
    def req_fields():
        return [k for k, v in AnalysisInput.__dataclass_fields__.items() if v.init is True]

    @staticmethod
    def Concatenate(ais : list["AnalysisInput"]):
        fields = NtupleProcessing.MergeOutputs([{f : getattr(a, f) for f in AnalysisInput.req_fields()} for a in ais])

        # check for null entries after merging outputs (null entries are list of Nones)
        for k in fields:
            if (type(fields[k]) == list) and (all(ak.is_none(fields[k]))):
                fields[k] = None

        return AnalysisInput(**fields)


    def SelectSample(self, mask : ak.Array) -> "AnalysisInput":
        """ Select a subset of the sample, creating a new AnalysisInput.

        Args:
            mask (ak.Array): Sample to select, can be a boolean mask or list of indices.

        Returns:
            AnalysisInput: Selected sample.
        """
        selection = {}
        for field in AnalysisInput.req_fields():
            value = getattr(self, field)
            if value is None:
                selection[field] = value # we allow Non types when defining data samples.
            if hasattr(value, "__iter__"):
                if type(value) is dict:
                    tmp_dict = {}
                    for k, v in value.items():
                        tmp_dict[k] = v[mask]
                    selection[field] = tmp_dict
                elif type(value) is list:
                    print(f"found {field} is a list type, cannot slice using an array, skipping.") #! might want to add specific logic to check if regions and processes are not null.
                    selection[field] = None
                else:
                    selection[field] = value[mask]
        return AnalysisInput(**selection)


    def CreateTrainTestSamples(self, seed : int, train_fraction : float = None) -> dict:
        """ Split analysis input into two samples

        Args:
            seed (int): seed for random permutation
            train_fraction (float, optional): fraction of events to assign to train, if None, sample is split 50/50. Defaults to None.

        Returns:
            dict: train and test samples.
        """
        rng = np.random.default_rng(seed)
        sample = rng.permutation(len(self.KE_init_reco))

        if train_fraction is None:
            fraction = len(sample) // 2
        else:
            fraction = round(train_fraction * len(sample))

        return {
            "train" : self.SelectSample(sample[:fraction]), "test" : self.SelectSample(sample[fraction:])}


    def CreateHistograms(self, energy_slice : Slices, exclusive_process : str, reco : bool, mask : np.ndarray = None) -> dict[np.ndarray]:
        """ Calculate Histogrames required for the cross section measurement using energy slicing. Note exclusive interaction histogram is without background subtraction.

        Args:
            energy_slice (Slices): energy slices
            exclusive_process (str): exclusive process
            reco (bool): use reco information?
            mask (np.ndarray, optional): additional mask. Defaults to None.

        Returns:
            dict[np.ndarray]: histograms
        """
        KE_int = self.KE_int_true if reco is False else self.KE_int_reco
        KE_init = self.KE_init_true if reco is False else self.KE_init_reco

        if mask is None: mask = np.zeros_like(self.outside_fv_reco, dtype = bool)

        if self.outside_fv_true is None:
            outside_tpc = self.outside_fv_reco | mask
        else:
            outside_tpc = self.outside_fv_true | mask

        if self.exclusive_process is not None:
            channel_mask = self.exclusive_process[exclusive_process]
        else:
            channel_mask = self.regions[exclusive_process]

        #! keep just in case
        # if efficiency is True:
        #     KE_int = KE_int[toy.df.beam_selection_mask]
        #     KE_init = KE_init[toy.df.beam_selection_mask]
        #     outside_tpc = outside_tpc[toy.df.beam_selection_mask]
        #     channel_mask = channel_mask[toy.df.beam_selection_mask]

        n_initial, n_interact_inelastic, n_interact_exclusive, n_incident = EnergySlice.CountingExperiment(KE_int, KE_init, outside_tpc, channel_mask, energy_slice, weights = self.weights)

        output = {"init" : n_initial, "int" : n_interact_inelastic, "int_ex" : n_interact_exclusive, "inc" : n_incident}
        return output



