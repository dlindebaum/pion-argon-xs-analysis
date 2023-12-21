"""
Created on: 02/06/2023 10:37

Author: Shyam Bhuller

Description: Library for code used in the cross section analysis. Refer to the README to see which apps correspond to the cross section analysis.
"""
import argparse
import copy
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
from pyunfold import iterative_unfold, Logger
from scipy.interpolate import interp1d, UnivariateSpline

from python.analysis import BeamParticleSelection, PFOSelection, EventSelection, Fitting, Plots, vector, Tags
from python.analysis.Master import LoadConfiguration, LoadObject, SaveObject, SaveConfiguration, ReadHDF5, Data, Ntuple_Type, timer
from python.analysis.shower_merging import SetPlotStyle

def bin_centers(bins : np.array) -> np.array:
    return (bins[1:] + bins[:-1]) / 2


def nandiv(num, den):
    return np.divide(num, np.where(den == 0, np.nan, den))


def KE(p, m):
    return (p**2 + m**2)**0.5 - m


def weighted_chi_sqr(observed, expected, uncertainties):
    u = np.array(uncertainties)
    u[u == 0] = np.nan
    return np.nansum((observed - expected)**2 / u**2) / len(observed)


def RatioWeights(mc : Data, func : str, params : list, truncate : int = 10):
    weights = 1/getattr(Fitting, func)(mc.recoParticles.beam_inst_P, *params)
    weights = np.where(weights > truncate, truncate, weights)
    return weights


def PlotXSComparison(xs : dict[np.array], energy_slice, process : str = None, colors : dict[str] = None, xs_sim_color : str = "k", newFigure : bool = True):
    xs_sim = GeantCrossSections(energy_range = [energy_slice.min_pos, energy_slice.max_pos + energy_slice.width])

    sim_curve_interp = xs_sim.GetInterpolatedCurve(process)
    x = energy_slice.pos - energy_slice.width/2

    if newFigure is True: Plots.plt.figure()
    chi_sqrs = {}
    for k, v in xs.items():
        w_chi_sqr = weighted_chi_sqr(v[0], sim_curve_interp(x), v[1])
        chi_sqrs[k] = w_chi_sqr
        Plots.Plot(x, v[0], xerr = energy_slice.width / 2, yerr = v[1], label = k + ", $\chi^{2}/ndf$ = " + f"{w_chi_sqr:.3g}", color = colors[k], linestyle = "", marker = "x", newFigure = False)
    
    if process == "single_pion_production":
        Plots.Plot(xs_sim.KE, sim_curve_interp(xs_sim.KE), label = "simulation", title = "single pion production", newFigure = False, xlabel = "$KE_{int} (MeV)$", ylabel = "$\sigma (mb)$", color = xs_sim_color)
    else:
        xs_sim.Plot(process, label = "simulation", color = xs_sim_color)

    Plots.plt.ylim(0)
    if max(Plots.plt.gca().get_ylim()) > np.nanmax(sim_curve_interp(xs_sim.KE).astype(float)) * 2:
        Plots.plt.ylim(0, max(sim_curve_interp(xs_sim.KE)) * 2)
    Plots.plt.xlim(energy_slice.min_pos, energy_slice.max_pos)
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


class BetheBloch:
    rho = 1.39 # [g/cm3] density of LAr
    K = 0.307075 # [MeV cm2 / mol]
    Z = 18 # LAr atomic number
    A = 39.948 # [g/mol] LAr atomic mass
    I = 188E-6 # [MeV] mean excitation energy
    me = Particle.from_pdgid(11).mass # [MeV] electron mass

    # density correction parameters
    C = 5.2146
    y0 = 0.2
    y1 = 3
    a = 0.19559
    k = 3

    @staticmethod
    def densityCorrection(beta : float | ak.Array, gamma : float | ak.Array) -> float | ak.Array:
        """ Correction to account for th fact a particles electric field flattens and spreads as the energy increases.

        Args:
            beta (float | ak.Array): velocity
            gamma (float | ak.Array): relativistic factor

        Returns:
            (float | ak.Array): density correction value
        """
        y = np.log10(beta * gamma)

        delta_0 = 2 * np.log(10) * y - BetheBloch.C
        delta_1 = delta_0 + BetheBloch.a * (BetheBloch.y1 - y)**BetheBloch.k

        if hasattr(y, "__iter__"):
            delta = ak.where(y >= BetheBloch.y1, delta_0, 0) 
            delta = ak.where((BetheBloch.y0 <= y) & (y < BetheBloch.y1), delta_1, delta)
        else:
            if y >= BetheBloch.y1:
                delta = delta_0
            elif y < BetheBloch.y0:
                delta = 0
            else:
                delta = delta_1

        return delta

    @staticmethod
    def meandEdX(KE : float | ak.Array, particle : Particle) -> float | ak.Array:
        """ Calculate the mean dEdX for a particle with given kinetic energy.

        Args:
            KE (float | ak.Array): particle kinetic energy
            particle (Particle): particle type

        Returns:
            float | ak.Array: mean dEdX
        """
        gamma = (KE / particle.mass) + 1
        beta = (1 - (1/gamma)**2)**0.5

        w_max = 2 * BetheBloch.me * (beta * gamma)**2 / (1 + (2 * BetheBloch.me * (gamma/particle.mass)) + (BetheBloch.me/particle.mass)**2)
        N = np.divide((BetheBloch.rho * BetheBloch.K * BetheBloch.Z * (particle.charge)**2), (BetheBloch.A * (beta**2)))
        A = 0.5 * np.log(2 * BetheBloch.me * (gamma**2) * (beta**2) * w_max / ((BetheBloch.I) **2))
        B = beta**2
        C = 0.5 * BetheBloch.densityCorrection(beta, gamma)

        dEdX = N * (A - B - C)

        dEdX = np.nan_to_num(dEdX)
        if hasattr(KE, "__iter__"):
            dEdX = ak.where(dEdX < 0, 0, dEdX) # handle when np.log is -infinity i.e. when KE = 0
        else:
            if dEdX < 0: dEdX = 0
        return dEdX

    @staticmethod
    def interp_KE_to_mean_dEdX(inital_KE : float, stepsize : float, particle : Particle = Particle.from_pdgid(211)) -> interp1d:
        """ Calculate the mean dEdX profile for a given initial kinetic energy and position step size.
            Then produce a function to map kinetic energy to dEdX given the outputs, and allow for interpolation.

        Args:
            inital_KE (float): Initial kinetic energy
            stepsize (float): position step size (cm)

        Returns:
            interp1d: interpolated map of KE and dEdX
        """
        e = inital_KE
        KE = []
        dEdX = []
        while e >= 0:
            KE.append(e)
            dEdX.append(BetheBloch.meandEdX(e, particle))
            e = e - stepsize * dEdX[-1]
            if dEdX[-1] <= 0: break # sometines bethebloch produces an unphysical value when KE is too small, so stop
        KE.append(0)
        dEdX.append(np.inf)
        return interp1d(KE, dEdX, fill_value = 0, bounds_error = False) # if outside the interpolation range, return 0

    @staticmethod
    def interp_range_to_KE(KE_init : float, precision = 0.05) -> interp1d:
        """ Create an interpolation object for the range of a particle and its kinetic energy

        Args:
            KE_init (float): kinetic energy
            precision (float, optional): position step. Defaults to 0.05.

        Returns:
            interp1d: interpolated map of range and KE
        """
        KE = [KE_init]
        track_length = [0]
        count = 0
        while KE[-1] > 0:
            KE.append(KE[-1] - precision * BetheBloch.meandEdX(KE[-1], Particle.from_pdgid(-13)))
            count += 1
            track_length.append(count * precision)
        track_length = np.array(track_length)

        return interp1d(max(track_length) - track_length, KE, fill_value = 0, bounds_error = False)


    @staticmethod
    def InteractingKE(KE_init : ak.Array, track_length : ak.Array, n : int) -> ak.Array:
        """ Compute the interacting energy from the particles initial kinetic energy and track length.

        Args:
            KE_init (ak.Array): initial kinetic energies
            track_length (ak.Array): track lengths
            n (int): number of iterations, higher results in more accurate values but takes exponentially longer.

        Returns:
            ak.Array: interacting KEs
        """
        interpolated_energy_loss = BetheBloch.interp_KE_to_mean_dEdX(2*max(KE_init), 1) # precompute the energy loss and create a function to interpolate between them
        steps = track_length/n

        KE_int = KE_init
        for i in range(n):
            KE_int = KE_int - interpolated_energy_loss(KE_int)*steps
        KE_int = ak.where(KE_int < 0, 0, KE_int)
        return KE_int

    @staticmethod
    def RangeFromKE(KE_init : np.array, particle : Particle, precision : float = 1) -> ak.Array:
        """ Compute the range of particles from the  initial kinetic energy.

        Args:
            KE_init (np.array): initial kinetic energies
            particle (Particle): particle type
            precision (float, optional): position step. Defaults to 1.

        Returns:
            ak.Array: ranges
        """
        interpolated_energy_loss = BetheBloch.interp_KE_to_mean_dEdX(2*max(KE_init), precision/2, particle) # precompute the energy loss and create a function to interpolate between them
        KE = np.array(KE_init)
        n = np.zeros(len(KE_init))
        while any(KE > 0):
            KE = KE - precision * interpolated_energy_loss(KE)
            n = n + (KE > 0)
        return n * precision


class ApplicationArguments:
    @staticmethod
    def Ntuples(parser : argparse.ArgumentParser, data : bool = False):
        parser.add_argument("-m", "--mc-file", dest = "mc_file", nargs = "+", help = "MC NTuple file to study.", required = False)
        if data: parser.add_argument("-d", "--data-file", dest = "data_file", nargs = "+", help = "Data Ntuple to study", required = False)
        parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Ntuple_Type, help = f"type of ntuple I am looking at {[m.value for m in Ntuple_Type]}.", required = False)
        return

    @staticmethod
    def SingleNtuple(parser : argparse.ArgumentParser, define_sample : bool = True):
        parser.add_argument(dest = "file", help = "NTuple file to study.")
        parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Ntuple_Type, help = f"type of ntuple I am looking at {[m.value for m in Ntuple_Type]}.", required = False)
        if define_sample : parser.add_argument("-S", "--sample-type", dest = "sample_type", type = str, choices = ["mc", "data"], help = f"type of sample I am looking at.", required = False)
        return

    @staticmethod
    def BeamQualityCuts(parser : argparse.ArgumentParser, data : bool = False):
        parser.add_argument("--mc_beam_quality_fit", dest = "mc_beam_quality_fit", type = str, help = "MC fit values for the beam quality cut.", required = False)
        if data: parser.add_argument("--data_beam_quality_fit", dest = "data_beam_quality_fit", type = str, default = None, help = "Data fit values for the beam quality cut.", required = False)
        return
    
    @staticmethod
    def Processing(parser : argparse.ArgumentParser):
        parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "Number of batches to split n tuple files into when parallel processing processing data.")
        parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "Number of events to process when parallel processing data.")
        parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "Number of threads to use when processsing")

    @staticmethod
    def Output(parser : argparse.ArgumentParser):
        parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "Directory to save plots")
        return

    @staticmethod
    def BeamSelection(parser : argparse.ArgumentParser):
        parser.add_argument("--scraper", action = "store_true", help = "Toggle to enable the beam scraper cut for the beam particle selection.")
        return

    @staticmethod
    def ShowerCorrection(parser : argparse.ArgumentParser):
        parser.add_argument("-C, --shower_correction", nargs = 2, dest = "correction", help = f"Shower energy correction method {tuple(EnergyCorrection.shower_energy_correction.keys())} followed by a correction parameters json file.", required = False)
        return

    @staticmethod
    def Plots(parser : argparse.ArgumentParser):
        parser.add_argument("--nbins", dest = "nbins", type = int, default = 50, help = "Number of bins to make for histogram plots.")
        parser.add_argument("-a", "--annotation", dest = "annotation", type = str, default = None, help = "Annotation to add to plots")
        return

    @staticmethod
    def Config(parser : argparse.ArgumentParser, required : bool = False):
        parser.add_argument("-c", "--config", dest = "config", type = str, default = None, required = required, help = "Analysis configuration file, if supplied will override command line arguments.")

    @staticmethod
    def ResolveArgs(args : argparse.Namespace) -> argparse.Namespace:
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

            if hasattr(args, "correction") and args.correction:
                args.correction_params = args.correction[1]
                args.correction = EnergyCorrection.shower_energy_correction[args.correction[0]]

        if hasattr(args, "out"):
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
            if head == "NTUPLE_FILE":
                args.mc_file = value["mc"]
                args.data_file = value["data"]
                args.ntuple_type = value["type"]
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
            elif head == "FINAL_STATE_PI0_SELECTION":
                args.pi0_selection = ApplicationArguments.__CreateSelection(value, EventSelection)
            elif head == "BEAM_QUALITY_FITS":
                args.mc_beam_quality_fit = LoadConfiguration(value["mc"])
                args.data_beam_quality_fit = LoadConfiguration(value["data"])
            elif head == "BEAM_SCAPER_FITS":
                args.mc_beam_scraper_fit = LoadConfiguration(value["mc"])
            elif head == "ENERGY_CORRECTION":
                args.correction = value["correction"]
                args.correction_params = value["correction_params"]
            elif head == "UPSTREAM_ENERGY_LOSS":
                args.upstream_loss_bins = value["bins"]
                args.upstream_loss_correction_params = LoadConfiguration(value["correction_params"])
            elif head == "BEAM_REWEIGHT":
                args.beam_reweight_params = LoadConfiguration(value["params"])
            elif head == "SELECTION_MASKS":
                args.selection_masks = {}
                for k, v in value.items():
                    args.selection_masks[k] = {i : LoadObject(j) for i, j in v.items()}
            elif head == "TOY_PARAMETERS":
                args.toy_parameters = {}
                for k, v in value.items():
                    if k == "beam_profile": 
                        args.toy_parameters[k] = getattr(Fitting, v)
                    else:
                        args.toy_parameters[k] = v
            else:
                setattr(args, head, value) # allow for generic configurations in the json file
        ApplicationArguments.DataMCSelectionArgs(args)
        if hasattr(args, "pi0_selection"):
            ApplicationArguments.AddEnergyCorrection(args)
        if hasattr(args, "beam_selection"):
            args.beam_selection["data_arguments"]["PiBeamSelection"]["use_beam_inst"] = True # make sure to set the correct settings for data.
        return args


    @staticmethod
    def AddEnergyCorrection(args):
        if hasattr(args, "correction"):
            method = EnergyCorrection.shower_energy_correction[args.correction]
            params = LoadConfiguration(args.correction_params)
        else:
            method = None
            params = None
            args.correction = None
            args.correction_params = None
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
                args.beam_selection["mc_arguments"][i]["fits"] = args.mc_beam_quality_fit
                args.beam_selection["data_arguments"][i]["fits"] = args.data_beam_quality_fit
            elif s == BeamParticleSelection.BeamScraperCut:
                args.beam_selection["mc_arguments"][i]["fits"] = args.mc_beam_scraper_fit
                args.beam_selection["data_arguments"][i]["fits"] = args.mc_beam_scraper_fit
            else:
                continue
        return args


def UpstreamEnergyLoss(KE_inst : ak.Array, params : np.array, function : Fitting.FitFunction = Fitting.poly2d) -> ak.Array:
    """ compute the upstream loss based on a repsonse function and it's fit parameters.

    Args:
        KE_inst (ak.Array): kinetic energy measured by the beam instrumentation
        function (Fitting.FitFunction): repsonse function, defaults to Fitting.poly2d. 
        params (np.array): function paramters

    Returns:
        ak.Array: upstream energy loss
    """
    return function.func(KE_inst, **params)

@timer
def RecoDepositedEnergy(events : Data, ff_KE : ak.Array, method : str) -> ak.Array:
    """ Calcuales the energy deposited by the beam particle in the TPC, either using calorimetric information or the bethe bloch formula (spatial information).

    Args:
        events (Data): events to look at
        ff_KE (ak.Array): front facing kinetic energy
        method (str): method to calcualte the deposited energy, either "calo" or "bb"

    Returns:
        ak.Array: depotisted energy
    """
    reco_pitch = vector.dist(events.recoParticles.beam_calo_pos[:, :-1], events.recoParticles.beam_calo_pos[:, 1:]) # distance between reconstructed calorimetry points
    
    if method == "calo":
        dE = ak.sum(events.recoParticles.beam_dEdX[:, :-1] * reco_pitch, -1)
    elif method == "bb":
        KE_int_bb = BetheBloch.InteractingKE(ff_KE, ak.sum(reco_pitch, -1), 50)
        dE = ff_KE - KE_int_bb
    else:
        raise Exception(f"{method} not a valid method, pick 'calo' or 'bb'")
    return dE


class Slices:
    """ Describes slices of a variable, equivilant to a list of bin edges but has more functionality. 

    Slice : a Single slice, has properies number (integer) and "position" in the parameter space of the value you want to slice up. 
    """
    Slice = namedtuple("Slice", "num pos")
    def __init__(self, width, _min, _max, reversed : bool = False):
        self.width = width
        self.min = _min
        self.max = _max
        self.reversed = reversed
        
        self.max_num = max(self.num)
        self.min_num = min(self.num)
        self.max_pos = max(self.pos)
        self.min_pos = min(self.pos)


    def __conversion__(self, x):
        """ convert a value to its slice number.

        Args:
            x: value, array of float

        Returns:
            slice: slice number/s
        """
        if self.reversed:
            numerator = self.max - x
        else:
            numerator = x
        c = np.floor(numerator // self.width)
        if hasattr(c, "__iter__"):
            return ak.values_astype(c, int)
        else:
            return int(c)


    def __create_slice__(self, i) -> Slice:
        """ using the slice number, create the Slice object.

        Args:
            i (int): slice number/s

        Returns:
            Slice: slice
        """
        if self.reversed:
            p = self.max - i * self.width
        else:
            p = i * self.width
        return self.Slice(i, p)


    def __call__(self, x):
        """ get the slice number for a set of values

        Args:
            x: values

        Returns:
            array or int: slice numbers
        """
        return self.__create_slice__(self.__conversion__(x))


    def __getitem__(self, i : int) -> Slice:
        """ Creates slices from slice numbers.

        Args:
            i (int): slice number

        Raises:
            StopIteration

        Returns:
            Slice: ith slice
        """
        if i * self.width > (self.max - self.min):
            raise StopIteration
        else:
            if self.reversed:
                return self.__create_slice__(i + self.__conversion__(self.max))
            else:
                return self.__create_slice__(i + self.__conversion__(self.min))

    @property
    def num(self) -> np.array:
        """ Return all slice numbers.

        Returns:
            np.array: slice numbers
        """
        return np.array([ s.num for s in self], dtype = int)

    @property
    def pos(self) -> np.array:
        """ Return all slice positions.

        Returns:
            np.array: slice positions
        """
        return np.array([ s.pos for s in self])


    def pos_to_num(self, pos):
        """ Convert slice positions to numbers

        Args:
            pos: positions

        Returns:
            array or int: slice numbers
        """
        slice_num = self.__conversion__(pos)
        if hasattr(pos, "__iter__"):
            slice_num = ak.where(slice_num > max(self.num), max(self.num), slice_num)
            slice_num = ak.where(slice_num < 0, min(self.num), slice_num)
        else:
            if pos > max(self.pos): 
                slice_num = max(self.num) # above range go into overflow bin
            if pos < 0:
                slice_num = min(self.num) # below range go into the underflow bin
        return slice_num


class GeantCrossSections:
    """ Object for accessing Geant 4 cross sections from the root file generated with Geant4reweight tools.
    """
    labels = {"abs_KE;1" : "absorption", "inel_KE;1" : "quasielastic", "cex_KE;1" : "charge_exchange", "dcex_KE;1" : "double_charge_exchange", "prod_KE;1" : "pion_production", "total_inel_KE;1" : "total_inelastic"}

    def __init__(self, file : str = os.environ["PYTHONPATH"] + "/data/g4_xs.root", energy_range : list = None, n_cascades : int = None) -> None:
        self.file = uproot.open(file) # open root file

        self.KE = self.file["abs_KE;1"].all_members["fX"] # load kinetic energy from one channel (shared for all cross section channels)

        if energy_range:
            self.KE = self.KE[(self.KE <= max(energy_range)) & (self.KE >= min(energy_range))]

        for k in self.file.keys():
            if "KE" in k:
                g = self.file[k]
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


    def Stat_Error(self, xs : str) -> np.array:
        """ Statisitical error of the simulation, done using binomial uncertainties. Only works if n_cascades is known.

        Args:
            xs (str): cross section process

        Returns:
            np.array: statistical error
        """
        if (self.n_cascades is None) or (not hasattr(self, xs + "_frac")):
            return 0 * getattr(self, xs)
        else:
            return getattr(self, xs) * np.sqrt(getattr(self, xs + "_frac") / self.n_cascades)


    def __PlotAll(self, title : str = None):
        """ Plot all cross section channels.
        """
        for k in self.labels.values():
            Plots.Plot(self.KE, getattr(self, k), label = k.replace("_", " "), newFigure = False, xlabel = "KE (MeV)", ylabel = "$\sigma (mb)$", title = title)
            Plots.plt.fill_between(self.KE, getattr(self, k) - self.Stat_Error(k), getattr(self, k) + self.Stat_Error(k), color = Plots.plt.gca()._get_lines.get_next_color())


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
                label = xs.replace("_", " ")
            else:
                if title is None:
                    title = xs.replace("_", " ")
            Plots.Plot(self.KE, getattr(self, xs), label = label, title = title, newFigure = False, xlabel = "$KE_{int} (MeV)$", ylabel = "$\sigma (mb)$", color = color)
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


class ThinSlice:
    """ Methods for implementing the thin slice measurement method.
    """
    @staticmethod
    def CountingExperiment(endPos : ak.Array, channel : ak.Array, slices : Slices) -> tuple[ak.Array, ak.Array]:
        """ Creates the interacting and incident histograms.

        Args:
            endPos (ak.Array): end position of particle or "interaction vertex"
            channel (ak.Array): mask which selects particles which interact in the channel you are interested in
            slices (Slices): spatial slices

        Returns:
            tuple[ak.Array, ak.Array]: n_interact and n_incident histograms
        """
        end_slice_pos = slices.pos_to_num(endPos)
        slice_nums = slices.num

        n_interact = np.histogram(end_slice_pos[channel], slice_nums)[0]

        total_interact = np.histogram(end_slice_pos, slice_nums)[0]
        n_incident = np.cumsum(total_interact[::-1])[::-1]
        return n_interact, n_incident

    @staticmethod
    def MeanSliceEnergy(energy : ak.Array, endPos : ak.Array, slices : Slices) -> tuple[ak.Array, ak.Array]:
        """ Compute the average energy in a spatial slice.

        Args:
            energy (ak.Array): particle energies over its lifetime in the tpc
            endPos (ak.Array): end position of particle or "interaction vertex"
            slices (Slices): spatial slices

        Returns:
            tuple[ak.Array, ak.Array]: means slice energy, error in the mean slice energy
        """
        beam_traj_slice = slices.pos_to_num(endPos)
        slice_nums = slices.num

        counts = np.histogram(ak.ravel(beam_traj_slice), slice_nums)[0] # histogram of positions will give the counts

        sum_energy = np.histogram(ak.ravel(beam_traj_slice), slice_nums, weights = ak.ravel(energy))[0] # total energy in each bin if you weight by energy
        sum_energy_sqr = np.histogram(ak.ravel(beam_traj_slice), slice_nums, weights = ak.ravel(energy)**2)[0] # same as above

        mean_energy = sum_energy / counts

        std_energy = np.divide(sum_energy_sqr, counts) - mean_energy**2
        error_mean_energy = np.sqrt(np.divide(std_energy, counts))

        return mean_energy, error_mean_energy

    @staticmethod 
    def TotalCrossSection(n_incident : np.array, n_interact : np.array, slice_width : float) -> tuple[np.array, np.array]:
        """ Returns cross section in mb.

        Args:
            n_incident (np.array): incident histogram
            n_interact (np.array): interacting histogram
            slice_width (float): spatial width of thin slice

        Returns:
            tuple[np.array, np.array]: cross section, statistical uncertainty
        """
        xs = np.log(n_incident / (n_incident - n_interact)) # calculate a dimensionless cross section

        v_incident = n_incident # poisson uncertainty
        v_interact = n_interact*(1- (n_interact/n_incident)) # binomial uncertainty

        xs_e = (1/n_incident) * (1/(n_incident - n_interact)) * (n_interact**2 * v_incident + n_incident**2 * v_interact)**0.5

        NA = 6.02214076e23
        factor = 10**27 * BetheBloch.A  / (BetheBloch.rho * NA * slice_width)

        return factor * xs, abs(factor * xs_e)


    def CrossSection(n_int_exclusive : np.array, n_int_inclusive : np.array, n_inc_inclusive : np.array, slice_width : float) -> tuple[np.array, np.array]:
        """ Cross section of exclusive process.

        Args:
            n_int_exclusive (np.array): exclusive interactions
            n_int_inclusive (np.array): interactions
            n_inc_inclusive (np.array): incident counts
            slice_width (float): slice width

        Returns:
            tuple[np.array, np.array]: cross section and error
        """
        NA = 6.02214076e23
        factor = 10**27 * BetheBloch.A  / (BetheBloch.rho * NA * slice_width)

        n_interact_ratio = nandiv(n_int_exclusive, n_int_inclusive)
        n_survived_inclusive = n_inc_inclusive - n_int_inclusive

        var_inc_inclusive = n_inc_inclusive # poisson variance
        var_int_inclusive = n_int_inclusive * (1 - nandiv(n_int_inclusive, n_inc_inclusive)) # binomial uncertainty
        var_int_exclusive = n_int_exclusive * (1 - nandiv(n_int_exclusive, n_inc_inclusive)) # binomial uncertainty

        xs = factor * n_interact_ratio * np.log(nandiv(n_inc_inclusive, n_inc_inclusive - n_int_inclusive))

        diff_n_int_exclusive = nandiv(xs, n_int_exclusive)
        diff_n_inc_inclusive = factor * n_interact_ratio * (nandiv(1, n_inc_inclusive) - nandiv(1, n_survived_inclusive))
        diff_n_int_inclusive = factor * n_interact_ratio * nandiv(1, n_survived_inclusive) - nandiv(xs, n_int_inclusive)

        xs_err = ((diff_n_int_exclusive**2 * var_int_exclusive) + (diff_n_inc_inclusive**2 * var_inc_inclusive) + (diff_n_int_inclusive**2 * var_int_inclusive))**0.5
        return xs, xs_err


class EnergySlice:
    """ Methods for implementing the energy slice measurement method.
    """
    @staticmethod
    def TrunacteSlices(slice_array : ak.Array, energy_slices : Slices) -> ak.Array:
        """ Method for truncating slice numbers due to the fact energy slices should be in reverse order vs kinetic energy.

        Args:
            slice_array (ak.Array): slices to truncate
            energy_slices (Slices): energy slices

        Returns:
            ak.Array: truncated slices
        """
        # set minimum to -1 (underflow i.e. energy > plim)
        slice_array = ak.where(slice_array < 0, -1, slice_array)
        # set maxslice (overflow i.e. energy < dE)
        slice_array = ak.where(slice_array > energy_slices.max_num, energy_slices.max_num, slice_array)
        return slice_array


    @staticmethod
    def NIncident(n_initial : np.array, n_end : np.array) -> np.array:
        """ Calculate number of incident particles

        Args:
            n_initial (np.array): initial particle counts
            n_end (np.array): interaction counts

        Returns:
            np.array: incident counts
        """
        n_survived_all = np.cumsum(n_initial - n_end)
        n_incident = n_survived_all + n_end
        return n_incident

    @staticmethod
    def SliceNumbers(int_energy : ak.Array, init_energy : ak.Array, outside_tpc : ak.Array, energy_slices : Slices) -> tuple[np.array, np.array]:
        """ Convert energies from physical units to slice numbers.

        Args:
            int_energy (ak.Array): interaction energy
            init_energy (ak.Array): initial energy
            outside_tpc (ak.Array): mask of particles which interact outside the fiducial volume
            energy_slices (Slices): energy slices

        Returns:
            tuple[np.array, np.array]: initial slice numbers and interacitng slice numbers
        """
        init_slice = energy_slices(init_energy).num + 1 # equivilant to ceil
        int_slice = energy_slices(int_energy).num

        init_slice = EnergySlice.TrunacteSlices(init_slice, energy_slices)
        int_slice = EnergySlice.TrunacteSlices(int_slice, energy_slices)

        # removes instances where the particle incident energy and interacting energy are in the same slice (Yinrui calls this an incomplete slice)
        # i.e. this happens if the particle interacting in its first slice, must be an artifact of the energy slicing but not sure why.
        bad_slices = (int_slice < init_slice) | outside_tpc
        init_slice = ak.where(bad_slices, -1, init_slice)
        int_slice = ak.where(bad_slices, -1, int_slice)
        return init_slice, int_slice

    @staticmethod
    def CountingExperiment(int_energy : ak.Array, init_energy : ak.Array, outside_tpc : ak.Array, process : ak.Array, energy_slices : Slices, interact_only : bool = False, weights : np.array = None) -> tuple[np.array]:
        """ Creates the interacting and incident histograms.

        Args:
            int_energy (ak.Array): interacting enrgy
            init_energy (ak.Array): initial energy
            outside_tpc (ak.Array): mask of particles which interact outside the fiducial volume
            process (ak.Array): mask of events for exclusive interactions
            energy_slices (Slices): energy slices
            interact_only (bool, optional): only return exclusive interaction histogram. Defaults to False.
            weights (np.array, optional): event weights. Defaults to None.

        Returns:
            np.array | tuple[np.array]: exclusive interaction histogram and/or initial histogram, incident histogram and interaction histogram 
        """
        init_slice, int_slice = EnergySlice.SliceNumbers(int_energy, init_energy, outside_tpc, energy_slices)

        slice_bins = np.arange(-1 - 0.5, energy_slices.max_num + 1.5)

        exclusive_weights = weights[process] if weights is not None else None

        n_interact_exclusive = np.histogram(np.array(int_slice[process]), slice_bins, weights = exclusive_weights)[0]
        if interact_only == False:
            n_initial = np.histogram(np.array(init_slice), slice_bins, weights = weights)[0]
            n_interact_inelastic = np.histogram(np.array(int_slice), slice_bins, weights = weights)[0]

            n_incident = EnergySlice.NIncident(n_initial, n_interact_inelastic)

            return n_initial, n_interact_inelastic, n_interact_exclusive, n_incident
        else:
            return n_interact_exclusive

    @staticmethod
    def CountingExperimentOld(int_energy : ak.Array, ff_energy : ak.Array, outside_tpc : ak.Array, channel : ak.Array, energy_slices : Slices) -> tuple[np.array, np.array]:
        """ (Legacy) Creates the interacting and incident histograms.

        Args:
            int_energy (ak.Array): interacting enrgy
            ff_energy (ak.Array): front facing energy
            outside_tpc (ak.Array): mask which selects particles decaying outside the tpc
            channel (ak.Array): mask which selects particles which interact in the channel you are interested in
            energy_slices (Slices): energy slices

        Returns:
            tuple[np.array, np.array]: n_interact and n_incident histograms
        """
        true_init_slice = energy_slices(ff_energy).num + 1 # equivilant to ceil
        true_int_slice = energy_slices(int_energy).num

        true_init_slice = EnergySlice.TrunacteSlices(true_init_slice, energy_slices)
        true_int_slice = EnergySlice.TrunacteSlices(true_int_slice, energy_slices)

        # just in case we encounter an instance where E_int > E_ini (unphysical)
        bad_slices = true_int_slice < true_init_slice
        true_init_slice = ak.where(bad_slices < 0, -1, true_init_slice)
        true_int_slice = ak.where(bad_slices, -1, true_int_slice)

        n_incident = np.zeros(energy_slices.max_num + 1)
        n_interact = np.zeros(energy_slices.max_num + 1)

        true_int_slice_in_tpc = true_int_slice[~outside_tpc]
        true_init_slice_in_tpc = true_init_slice[~outside_tpc]

        #! slowest but most explict version
        # n_incident = np.zeros(max_slice + 1)
        # for i in range(len(n_incident)):
        #     for p in range(len(true_int_slice_in_tpc)):
        #         if (true_init_slice_in_tpc[p] <= i) and (true_int_slice_in_tpc[p] >= i):
        #             n_incident[i] += 1
        #! faster, order log(n) because it skips checking for empty entries
        # true_init_slice_in_tpc = ak.where(true_init_slice_in_tpc == -1, 0, true_init_slice_in_tpc) #! done because -n index in python means you add to the last nth bin
        # for p in range(len(true_int_slice_in_tpc)):
        #     n_incident[true_init_slice_in_tpc[p] : true_int_slice_in_tpc[p] + 1] += 1
        # print(n_incident)

        #! fastest, vectorised version of the first but c++ loops are faster. 
        n_incident = np.array([ak.sum(ak.where((true_init_slice_in_tpc <= i) & (true_int_slice_in_tpc > i), 1, 0)) for i in range(energy_slices.max_num + 1)])
    
        n_interact = np.histogram(np.array(true_int_slice_in_tpc[channel[~outside_tpc]]), range(-1, energy_slices.max_num + 1))[0]
        n_interact = np.roll(n_interact, -1) # shift the underflow bin to the location of the overflow bin in n_incident i.e. merge them.
        return n_interact, n_incident + n_interact

    @staticmethod
    def Slice_dEdX(energy_slices : Slices, particle : Particle) -> np.array:
        """ Computes the mean dEdX between energy slices.

        Args:
            energy_slices (Slices): energy slices
            particle (Particle): particle

        Returns:
            np.array: mean dEdX
        """
        return BetheBloch.meandEdX(energy_slices.pos - energy_slices.width/2, particle)

    @staticmethod
    def TotalCrossSection(n_interact : np.array, n_incident : np.array, dEdX : np.array, dE : float) -> tuple[np.array, np.array]:
        """ Compute cross section using ThinSlice.CrossSection, by passing an effective spatial slice width.

        Args:
            n_interact (np.array): interacting histogram
            n_incident (np.array): incident histogram
            dEdX (np.array): mean slice dEdX
            dE (float): energy slice width

        Returns:
            tuple[np.array, np.array]: Cross section and statistical uncertainty.
        """
        return ThinSlice.TotalCrossSection(n_incident, n_interact, dE/dEdX)

    @staticmethod
    def CrossSection(n_int_ex : np.array, n_int : np.array, n_inc : np.array, dEdX : np.array, dE : float, n_int_ex_err : np.array = None, n_int_err : np.array = None, n_inc_err : np.array = None) -> tuple[np.array, np.array]:
        """ Compute exclusive cross sections. If interactions errors are not provided, staticial uncertainties are used (poisson for incident, binomial for interactions).

        Args:
            n_int_ex (np.array): exclusive interactions
            n_int (np.array): interactions
            n_inc (np.array): incident counts
            dEdX (np.array): slice dEdX
            dE (float): energy slice width
            n_int_ex_err (np.array, optional): exclusive interaction errors. Defaults to None.
            n_int_err (np.array, optional): interaction errors. Defaults to None.
            n_inc_err (np.array, optional): incident count errors. Defaults to None.

        Returns:
            tuple[np.array, np.array]: _description_
        """
        NA = 6.02214076e23
        factor = np.array(dEdX) * 10**27 * BetheBloch.A  / (BetheBloch.rho * NA * dE)

        n_interact_ratio = nandiv(n_int_ex, n_int)
        n_survived = n_inc - n_int

        if n_inc_err is not None:
            var_inc_inclusive = n_inc_err**2
        else:
            var_inc_inclusive = n_inc # poisson variance
    
        if n_int_err is not None:
            var_int = n_int_err**2
        else:
            var_int = n_int * (1 - nandiv(n_int, n_inc)) # binomial uncertainty
    
        if n_int_ex_err is not None:
            var_int_ex = n_int_ex_err**2
        else:
            var_int_ex = n_int_ex * (1 - nandiv(n_int_ex, n_inc)) # binomial uncertainty

        xs = factor * n_interact_ratio * np.log(nandiv(n_inc, n_inc - n_int))

        diff_n_int_ex = nandiv(xs, n_int_ex)
        diff_n_inc = factor * n_interact_ratio * (nandiv(1, n_inc) - nandiv(1, n_survived))
        diff_n_int = factor * n_interact_ratio * nandiv(1, n_survived) - nandiv(xs, n_int)

        xs_err = ((diff_n_int_ex**2 * var_int_ex) + (diff_n_inc**2 * var_inc_inclusive) + (diff_n_int**2 * var_int))**0.5
        return np.array(xs, dtype = float), np.array(xs_err, dtype = float)


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

    @staticmethod
    def ComputeCounts(true_regions : dict, reco_regions : dict, selection_efficincy : np.array = None) -> np.array:
        """ Computes the counts of each combination of reco and true regions.

        Args:
            true_regions (dict): true region masks
            reco_regions (dict): reco region masks
            return_counts (bool, optional): return matrix of counts. Defaults to False.

        Returns:
            np.array: counts.
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

    @staticmethod
    def GetCorrelationMatrix(toy : pd.DataFrame) -> np.array:
        """ Compute the confusion matrix for the reco/truth regions.

        Args:
            toy (pd.DataFrame): Toy

        Returns:
            np.array: confusion matrix
        """
        reco_regions = Toy.GetRegion(toy, "reco_regions_")
        true_regions = Toy.GetRegion(toy, "truth_regions_")
        return Toy.ComputeCounts(true_regions, reco_regions)


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


    def NInteract(self, energy_slice : Slices, process : np.array, mask : np.array = None, weights : np.array = None) -> np.array:
        """ Exclusive interaction histogram using energy slice method.

        Args:
            energy_slice (Slices): energy slices
            process (np.array): exclusive process mask
            mask (np.array, optional): additional mask to apply. Defaults to None.
            weights (np.array, optional): event weights. Defaults to None.

        Returns:
            np.array: exclusive interaction histogram
        """
        if mask is None: mask = np.ones(len(self.df), dtype = bool)
        w = weights if weights is None else weights[mask]
        n_interact = EnergySlice.CountingExperiment(self.df.KE_int_smeared[mask].values, self.df.KE_init_smeared[mask].values, self.outside_tpc_smeared[mask].values, process[mask].values, energy_slice, interact_only = True, weights = w)
        return n_interact


@dataclass
class AnalysisInput:
    # masks
    regions : dict[np.array]
    inclusive_process : dict[np.array]
    exclusive_process : dict[np.array]
    outside_tpc_reco : np.array
    outside_tpc_true : np.array
    # observables
    track_length_reco : np.array
    KE_int_reco : np.array
    KE_init_reco : np.array
    mean_track_score : np.array
    track_length_true : np.array
    KE_int_true : np.array
    KE_init_true : np.array
    # extras
    weights : np.array

    def ToFile(self, file : str):
        """ Save to dill file.

        Args:
            file (str): file path.
        """
        SaveObject(file, self)
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


    def NInteract(self, energy_slice : Slices, process: np.array, mask : np.array = None, reco : bool = True, weights : np.array = None) -> np.array:
        """ Calculate exclusive interaction histogram using the energy slice method.

        Args:
            energy_slice (Slices): energy slices
            process (np.array): exclusive process mask
            mask (np.array, optional): additional mask. Defaults to None.
            reco (bool, optional): use reco KE?. Defaults to True.
            weights (np.array, optional): event weights. Defaults to None.

        Returns:
            np.array: exclusive interaction histogram.
        """
        if mask is None: mask = np.ones(len(self.KE_int_reco), dtype = bool)
        if reco is True:
            KE_int = self.KE_int_reco
            KE_init = self.KE_init_reco
            outside_tpc = self.outside_tpc_reco
        else:
            KE_int = self.KE_int_true
            KE_init = self.KE_init_true
            outside_tpc = self.outside_tpc_true
        n_interact = EnergySlice.CountingExperiment(KE_int[mask], KE_init[mask], outside_tpc[mask], process[mask], energy_slice, interact_only = True, weights = weights[mask] if weights is not None else weights)
        return n_interact

    @staticmethod
    def CreateAnalysisInputToy(toy : Toy) -> "AnalysisInput":
        """ Create analysis input from a toy sample.

        Args:
            toy (Toy): toy sample

        Returns:
            AnalysisInput: analysis input object.
        """
        inclusive_events = (toy.df.inclusive_process != "decay").values

        regions = {k : v.values for k, v in toy.reco_regions.items()}
        process = {k : v.values for k, v in toy.truth_regions.items()}

        return AnalysisInput(
            regions,
            inclusive_events,
            process,
            toy.outside_tpc_smeared.values,
            toy.outside_tpc.values,
            toy.df.z_int_smeared.values,
            toy.df.KE_int_smeared.values,
            toy.df.KE_init_smeared.values,
            toy.df.mean_track_score.values,
            toy.df.z_int.values,
            toy.df.KE_int.values,
            toy.df.KE_init.values,
            None
            )

    @staticmethod
    def CreateAnalysisInputNtuple(events : Data, upstream_energy_loss_params : dict, reco_regions : dict[np.array], true_regions : dict[np.array] = None, mc_reweight_params : dict = None) -> "AnalysisInput":
        """ Create analysis input from an ntuple sample.

        Args:
            events (Data): ntuple sample
            upstream_energy_loss_params (dict): upstream energy loss correction
            reco_regions (dict[np.array]): reco region masks
            true_regions (dict[np.array], optional): true process masks. Defaults to None.
            mc_reweight_params (dict, optional): mc reweight parameters. Defaults to None.

        Returns:
            AnalysisInput: analysis input.
        """
        if mc_reweight_params is not None:
            weights = RatioWeights(events, "gaussian", [mc_reweight_params[k]["value"] for k in mc_reweight_params], 3)
        else:
            weights = None

        reco_KE_inst = KE(events.recoParticles.beam_inst_P, Particle.from_pdgid(211).mass)
        reco_upstream_loss = UpstreamEnergyLoss(reco_KE_inst, upstream_energy_loss_params)
        reco_KE_ff = reco_KE_inst - reco_upstream_loss
        reco_KE_int = reco_KE_ff - RecoDepositedEnergy(events, reco_KE_ff, "bb")
        reco_track_length = events.recoParticles.beam_track_length
        outside_tpc_reco = (events.recoParticles.beam_endPos_SCE.z < 0) | (events.recoParticles.beam_endPos_SCE.z > 700)


        if true_regions is not None:
            true_KE_ff = events.trueParticles.beam_KE_front_face
            true_KE_int = events.trueParticles.beam_traj_KE[:, -2]
            true_track_length = events.trueParticles.beam_track_length
            outside_tpc_true = (events.trueParticles.beam_traj_pos.z[:, -1] < 0) | (events.trueParticles.beam_traj_pos.z[:, -1] > 700)
            # inelastic = np.ones(len(events.eventNum), dtype = bool)
            inelastic = events.trueParticles.true_beam_endProcess == "pi+Inelastic"

        else:
            true_KE_int = None
            true_KE_ff = None
            true_track_length = None
            outside_tpc_true = None
            inelastic = None

        mean_track_score = ak.fill_none(ak.mean(events.recoParticles.track_score, axis = -1), -0.05) # fill null values in case empty events are supplied

        return AnalysisInput(
            reco_regions,
            inelastic,
            true_regions,
            outside_tpc_reco,
            outside_tpc_true,
            reco_track_length,
            reco_KE_int,
            reco_KE_ff,
            mean_track_score,
            true_track_length,
            true_KE_int,
            true_KE_ff,
            weights
            )


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

        train_indices = sample[:fraction]
        test_indices = sample[fraction:]

        train = {}
        test = {}
        for attr in vars(self):
            value = getattr(self, attr)
            if hasattr(value, "__iter__"):
                if type(value) is dict:
                    tmp_dict_train = {}
                    tmp_dict_test = {}
                    for k, v in value.items():
                        tmp_dict_train[k] = v[train_indices]
                        tmp_dict_test[k] = v[test_indices]
                    train[attr] = tmp_dict_train
                    test[attr] = tmp_dict_test
                else:
                    train[attr] = value[train_indices]
                    test[attr] = value[test_indices]

        return {"train" : AnalysisInput(**train), "test" : AnalysisInput(**test)}


    def CreateHistograms(self, energy_slice : Slices, exclusive_process : str, reco : bool, mask : np.array = None) -> dict[np.array]:
        """ Calculate Histogrames required for the cross section measurement using energy slicing. Note exclusive interaction histogram is without background subtraction.

        Args:
            energy_slice (Slices): energy slices
            exclusive_process (str): exclusive process
            reco (bool): use reco information?
            mask (np.array, optional): additional mask. Defaults to None.

        Returns:
            dict[np.array]: histograms
        """
        KE_int = self.KE_int_true if reco is False else self.KE_int_reco
        KE_init = self.KE_init_true if reco is False else self.KE_init_reco

        if mask is None: mask = np.zeros_like(self.outside_tpc_reco, dtype = bool)

        if self.outside_tpc_true is None:
            outside_tpc = self.outside_tpc_reco | mask
        else:
            outside_tpc = self.outside_tpc_reco | self.outside_tpc_true | mask

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


class RegionFit:

    @staticmethod    
    def Model(n_channels : int, KE_int_templates : np.array, mean_track_score_templates : np.array = None, mc_stat_unc : bool = False) -> pyhf.Model:
        def channel(channel_name : str, samples : np.array, mc_stat_unc : bool):
            ch = {
                "name": channel_name,
                "samples":[
                    {
                        "name" : f"sample_{i}",
                        "data" : s.tolist(),
                        "modifiers" : [
                            {'name': f"mu_{i}", 'type': 'normfactor', 'data': None},
                            ]
                    }
                for i, s in enumerate(samples)
                ]
            }
            if mc_stat_unc == True:
                for i in range(len(samples)):
                    ch["samples"][i]["modifiers"].append({'name': f"{channel_name}_sample_{i}_pois_err", 'type': 'shapesys', 'data': np.sqrt(samples[i]).astype(int).tolist()})
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
    def GenerateObservations(fit_input : AnalysisInput, energy_slices : Slices, mean_track_score_bins : np.array, model : pyhf.Model, verbose : bool = True) -> np.array:
        data = RegionFit.CreateObservedInputData(fit_input, energy_slices, mean_track_score_bins)
        if verbose is True: print(f"{model.config.suggested_init()=}")
        observations = np.concatenate(data + [model.config.auxdata])
        if verbose is True: print(f"{model.logpdf(pars=model.config.suggested_init(), data=observations)=}")
        return observations

    @staticmethod
    def Fit(observations, model : pyhf.Model, init_params : list[float] = None, par_bounds : list[tuple] = None, verbose : bool = True) -> FitResults:
        pyhf.set_backend(backend = "numpy", custom_optimizer = "minuit")
        if verbose is True: print(f"{init_params=}")
        result = cabinetry.fit.fit(model, observations, init_pars = init_params, custom_fit = True, tolerance = 0.001, par_bounds = par_bounds)

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
    def GetPredictedCorrelationMatrix(model : pyhf.Model, mu : np.array) -> np.array:
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
    def CreateKEIntTemplates(analysis_input : AnalysisInput, energy_slices : Slices) -> list[np.array]:
        model_input_data = []
        for c in analysis_input.regions:
            tmp = []
            for s in analysis_input.exclusive_process:
                tmp.append(analysis_input.NInteract(energy_slices, analysis_input.exclusive_process[s], analysis_input.regions[c], True, analysis_input.weights) + 1)
            model_input_data.append(tmp)
        return model_input_data

    @staticmethod
    def CreateMeanTrackScoreTemplates(analysis_input : AnalysisInput, bins : np.array, weights : np.array = None) -> np.array:
        templates = []
        for t in analysis_input.exclusive_process:
            mask = analysis_input.exclusive_process[t]
            templates.append(np.histogram(analysis_input.mean_track_score[mask], bins, weights = weights[mask] if weights is not None else weights)[0])
        return np.array(templates)

    @staticmethod
    def CreateModel(template : AnalysisInput, energy_slice : Slices, mean_track_score_bins : np.array, return_templates : bool = False, weights : np.array = None, mc_stat_unc : bool = True) -> pyhf.Model:
        templates_energy = RegionFit.CreateKEIntTemplates(template, energy_slice)
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
    def CreateObservedInputData(fit_input : AnalysisInput, slices : Slices, mean_track_score_bins : np.array = None) -> np.array:
        observed_binned = []
        if fit_input.inclusive_process is None:
            mask = np.ones_like(fit_input.KE_int_reco, dtype = bool)
        else:
            mask = fit_input.inclusive_process
        for v in fit_input.regions.values():
            observed_binned.append(fit_input.NInteract(slices, v & mask, reco = True))
        if mean_track_score_bins is not None:
            observed_binned.append(np.histogram(fit_input.mean_track_score[fit_input.inclusive_process], mean_track_score_bins)[0])
        return observed_binned

    @staticmethod
    def SliceModelPrediction(prediction : cabinetry.model_utils.ModelPrediction, slice : slice, label : str) -> cabinetry.model_utils.ModelPrediction:
        return cabinetry.model_utils.ModelPrediction(prediction.model, np.array(prediction.model_yields[slice]), np.array(prediction.total_stdev_model_bins[slice]), np.array(prediction.total_stdev_model_channels[slice]), label)

    @staticmethod
    def PlotPrefitPostFit(prefit, prefit_err, postfit, postfit_err, energy_bins):
        with Plots.RatioPlot(energy_bins[::-1], postfit, prefit, postfit_err, prefit_err, "$KE_{int}$ (MeV)", "fit/ actual") as ratio_plot:
            Plots.Plot(ratio_plot.x, ratio_plot.y2, yerr = ratio_plot.y2_err, color = "C0", label = "actual", style = "step", newFigure = False)
            Plots.Plot(ratio_plot.x, ratio_plot.y1, yerr = ratio_plot.y1_err, color = "C6", label = "fit", style = "step", ylabel = "Counts", newFigure = False)


class Unfold:
    @staticmethod
    def CorrelationMarix(observed : np.array, true : np.array, bins : np.array, remove_overflow : bool = True) -> np.array:
        """ Caclulate Correlation matrix of observed and true parameters.

        Args:
            observed (np.array): observed data (reco).
            true (np.array): true data (truth).
            bins (np.array): bins.
            remove_overflow (bool, optional): remove the first bins which are interpreted as overflow. Defaults to True.

        Returns:
            np.array: Correlation matrix.
        """
        corr = np.histogram2d(np.array(observed), np.array(true), bins = bins)[0]
        if remove_overflow is True:
            corr = corr[1:, 1:]
        return corr

    @staticmethod
    def ResponseMatrix(observed : np.array, true : np.array, bins : np.array, efficiencies : np.array = None, remove_overflow : bool = False) -> tuple[np.array, np.array]:
        """ Caclulate Correlation matrix of observed and true parameters.

        Args:
            observed (np.array): observed data (reco).
            true (np.array): true data (truth).
            bins (np.array): bins.
            efficiencies (np.array, optional): selection efficiency. Defaults to None.
            remove_overflow (bool, optional): remove the first bins which are interpreted as overflow. Defaults to True.

        Returns:
            tuple[np.array, np.array]: response matrix and the statistical error in the response matrix
        """
        if efficiencies is None:
            efficiencies = np.ones(len(bins) - 1 - int(remove_overflow))

        response_hist = Unfold.CorrelationMarix(observed, true, bins, remove_overflow)
        response_hist_err = np.sqrt(response_hist)

        column_sums = response_hist.sum(axis=0)
        if remove_overflow is True:
            normalization_factor = efficiencies[1:] / column_sums
        else:
            normalization_factor = efficiencies / column_sums
        response = response_hist * normalization_factor
        response_err = response_hist_err * normalization_factor
        
        response = np.nan_to_num(response)
        response_err = np.nan_to_num(response_err)

        return response, response_err

    @staticmethod #? move to Plots?
    def PlotMatrix(matrix : np.array, title : str = None, c_label : str = None):
        """ Plot numpy matrix.

        Args:
            matrix (np.array): matrix
            title (str, optional): plot title. Defaults to None.
            c_label (str, optional): colourbar label. Defaults to None.
        """
        #* cause = true, effect = reco
        Plots.plt.figure()
        Plots.plt.imshow(matrix, origin = "lower", cmap = "plasma")
        Plots.plt.xlabel("true")
        Plots.plt.ylabel("reco")
        Plots.plt.grid(False)
        Plots.plt.colorbar(label = c_label)
        Plots.plt.tight_layout()
        Plots.plt.title(title)
        return

    @staticmethod
    def CalculateResponseMatrices(template : AnalysisInput, process : str, energy_slice : Slices, book : Plots.PlotBook = None, efficiencies : dict[np.array] = None) -> dict[np.array]:
        """ Calculate response matrix of energy histograms from analysis input.

        Args:
            template (AnalysisInput): template analysis input
            process (str): exclusive process
            energy_slice (Slices): energy slices
            book (Plots.PlotBook, optional): plot book. Defaults to None.
            efficiencies (dict[np.array], optional): selection efficiencies. Defaults to None.

        Returns:
            dict[np.array]: response matrices with errors for each histogram
        """
        slice_bins = np.arange(-1 - 0.5, energy_slice.max_num + 1.5)

        outside_tpc_mask = template.outside_tpc_reco | template.outside_tpc_true

        true_slices = EnergySlice.SliceNumbers(template.KE_int_true, template.KE_init_true, outside_tpc_mask, energy_slice)
        reco_slices = EnergySlice.SliceNumbers(template.KE_int_reco, template.KE_init_reco, outside_tpc_mask, energy_slice)

        channel = template.exclusive_process[process][~outside_tpc_mask]

        slice_pairs = {"init" : [reco_slices[0], true_slices[0]], "int" : [reco_slices[1], true_slices[1]], "int_ex" : [reco_slices[1][channel], true_slices[1][channel]]}

        corr = {}
        resp = {}

        labels = {"init" : "$N_{init}$", "int" : "$N_{int}$", "int_ex" : "$N_{int, ex}$"}


        for k, v in slice_pairs.items():
            corr[k] = Unfold.CorrelationMarix(*v, bins = slice_bins, remove_overflow = False)
            resp[k] = Unfold.ResponseMatrix(*v, bins = slice_bins, efficiencies = None if efficiencies is None else efficiencies[k], remove_overflow = False)
            if book is not None:
                Unfold.PlotMatrix(corr[k], title = f"Response Marix: {labels[k]}", c_label = "Counts")
                book.Save()
            if book is not None:
                Unfold.PlotMatrix(resp[k][0], title = f"Normalised Response Matrix: {labels[k]}", c_label = "$P(E_{i}|C_{j})$")
                book.Save()
        return resp

    @staticmethod
    def Unfold(observed : dict[np.array], observed_err : dict[np.array], response_matrices : dict[np.array], priors : dict[np.array] = None, ts_stop = 0.01, max_iter = 100, ts = "ks", regularizers : dict[UnivariateSpline] = None, verbose : bool = False, efficiencies : dict[np.array] = None) -> dict[dict]:
        """ Run iterative bayesian unfolding for each histogram.

        Args:
            observed (dict[np.array]): observed data
            observed_err (dict[np.array]): observed data error
            response_matrices (dict[np.array]): repsonse matrices
            priors (dict[np.array], optional): pior distributions. Defaults to None.
            ts_stop (float, optional): tolerance of test statistic. Defaults to 0.01.
            max_iter (int, optional): maximum number of iterations. Defaults to 100.
            ts (str, optional): test statistic type. Defaults to "ks".
            regularizers (dict[UnivariateSpline], optional): splines to regularise the priors. Defaults to None.
            verbose (bool, optional): verbose printout. Defaults to False.
            efficiencies (dict[np.array], optional): selection efficiencies. Defaults to None.

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
        for k, n, n_e, v in zip(response_matrices.keys(), observed.values(), observed_err.values(), response_matrices.values()):

            cb = make_cb(k)

            if efficiencies is not None:
                efficiency = efficiencies[k]
            else:
                efficiency = np.ones_like(n) #! for the toy, assume perfect selection efficiency, so 1 +- 0
            efficiency_err = np.zeros_like(n)

            if priors is None:
                p = n/sum(n)
            else:
                p = priors[k] / sum(priors[k])

            results[k] = iterative_unfold(n, n_e, v[0], v[1], efficiency, efficiency_err, callbacks = cb, prior = p, ts_stopping = ts_stop, max_iter = max_iter, ts = ts)
        return results

    @staticmethod
    def PlotUnfoldingResults(obs : np.array, true : np.array, results : dict, energy_bins : np.array, label : str, book : Plots.PlotBook = Plots.PlotBook.null):
        """ Plot unfolded histogram in comparison to observed and true.

        Args:
            obs (np.array): observation
            true (np.array): truth
            results (dict): unfolding results
            energy_bins (np.array): energy bins
            label (str): x label (units of MeV are automatically applied)
            book (Plots.PlotBook, optional): plot book. Defaults to Plots.PlotBook.null.
        """
        Plots.Plot(energy_bins[::-1], obs, style = "step", label = "reco", xlabel = label, color = "C6")
        Plots.Plot(energy_bins[::-1], true, style = "step", label = "true", xlabel = label, color = "C0", newFigure = False)
        Plots.Plot(energy_bins[::-1], results["unfolded"], yerr = results["stat_err"], style = "step", label = f"unfolded, {results['num_iterations']} iterations", xlabel = label + " (MeV)", color = "C4", newFigure = False)
        book.Save() 
        Unfold.PlotMatrix(results["unfolding_matrix"], title = "Unfolded matrix: " + label, c_label = "$P(C_{j}|E_{i})$")
        book.Save() 
        return