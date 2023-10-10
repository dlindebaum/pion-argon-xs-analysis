"""
Created on: 02/06/2023 10:37

Author: Shyam Bhuller

Description: Library for code used in the cross section analysis. Refer to the README to see which apps correspond to the cross section analysis.
"""
import argparse
import copy
import json

from collections import namedtuple

import awkward as ak
import numpy as np
import pandas as pd
import dill
import uproot
from particle import Particle

from python.analysis import Master, BeamParticleSelection, PFOSelection, EventSelection, Fitting, Plots, vector
from python.analysis.shower_merging import SetPlotStyle


def KE(p, m):
    return (p**2 + m**2)**0.5 - m


def LoadSelectionFile(file : str):
    """ Opens and serialises object saved as a dill file. May be remaned to a more general method if dill files are used more commonly.

    Args:
        file (str): dill file

    Returns:
        any: loaded object
    """
    with open(file, "rb") as f:
        obj = dill.load(f)
    return obj

#! currently not used, wait until the new cut handling is done.
def GenerateSelectionAndMasks(events : Master.Data, fits : dict) -> dict:
    beam_selection_mask = BeamParticleSelection.CreateDefaultSelection(events, False, fits, False, False)
    events.Filter([beam_selection_mask], [beam_selection_mask])

    good_PFO_selection_mask = PFOSelection.GoodShowerSelection(events, False)
    events.Filter([good_PFO_selection_mask])

    pi_plus_selection_mask = PFOSelection.DaughterPiPlusSelection(events)
    photon_selection_mask = PFOSelection.InitialPi0PhotonSelection(events)

    pi0_selection_mask = EventSelection.Pi0Selection(events, photon_selection_mask)

    truth_regions = EventSelection.create_regions(events.trueParticles.nPi0, events.trueParticles.nPiPlus)

    reco_pi0_counts = EventSelection.count_pi0_candidates(events, exactly_two_photons = True)
    reco_pi_plus_counts_mom_cut = EventSelection.count_charged_pi_candidates(events,energy_cut = None)
    reco_regions = EventSelection.create_regions(reco_pi0_counts, reco_pi_plus_counts_mom_cut)
    masks  = {
        "beam_selection"      : beam_selection_mask,
        "valid_pfo_selection" : good_PFO_selection_mask,
        "pi_plus_selection"   : pi_plus_selection_mask,
        "photon_selection"    : photon_selection_mask,
        "pi0_selection"       : pi0_selection_mask,
        "truth_regions"       : truth_regions,
        "reco_regions"        : reco_regions
    }
    return masks


def SaveSelection(file : str, masks : dict):
    """ Saves Masks from selection to file. If not specified it will be left as None.

    Args:
        file (str): _description_
        beam_selection_mask (dict): dictionary of masks
    """
    with open(file, "wb") as f:
        dill.dump(masks, f)


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
        def bounds(x, y):
            return (-np.inf, np.inf)

        @staticmethod
        def p0(x, y):
            return None

        @staticmethod
        def mu():
            pass
        
        @staticmethod
        def var():
            pass

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
    def densityCorrection(beta, gamma):
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
    def meandEdX(KE, particle : Particle):
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


    def InteractingKE(KE_init : ak.Array, track_length : ak.Array, particle : Particle, n : int):
        KE_int = KE_init
        for i in range(n):
            KE_int = KE_int - BetheBloch.meandEdX(KE_int, particle)*track_length/n
        KE_int = ak.where(KE_int < 0, 0, KE_int)
        return KE_int


class ApplicationArguments:
    @staticmethod
    def Ntuples(parser : argparse.ArgumentParser, data : bool = False):
        parser.add_argument("-m", "--mc-file", dest = "mc_file", nargs = "+", help = "MC NTuple file to study.", required = False)
        if data: parser.add_argument("-d", "--data-file", dest = "data_file", nargs = "+", help = "Data Ntuple to study", required = False)
        parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {[m.value for m in Master.Ntuple_Type]}.", required = False)
        return

    @staticmethod
    def SingleNtuple(parser : argparse.ArgumentParser, define_sample : bool = True):
        parser.add_argument(dest = "file", help = "NTuple file to study.")
        parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {[m.value for m in Master.Ntuple_Type]}.", required = False)
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
    def Config(parser : argparse.ArgumentParser):
        parser.add_argument("-c", "--config", dest = "config", type = str, default = None, help = "Analysis configuration file, if supplied will override command line arguments.")

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
            elif head == "SELECTION_MASKS":
                args.selection_masks = {}
                for k, v in value.items():
                    args.selection_masks[k] = {i : LoadSelectionFile(j) for i, j in v.items()}
            else:
                setattr(args, head, value) # allow for generic configurations in the json file
        ApplicationArguments.DataMCSelectionArgs(args)
        ApplicationArguments.AddEnergyCorrection(args)
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


def LoadConfiguration(file : str) -> dict:
    """ Loads a json file.

    Args:
        file (str): file path

    Returns:
        dict: unpacked json 
    """
    with open(file, "rb") as f:
        config = json.load(f)
    return config


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


def RecoDepositedEnergy(events : Master.Data, ff_KE : ak.Array, method : str) -> ak.Array:
    """ Calcuales the energy deposited by the beam particle in the TPC, either using calorimetric information or the bethe bloch formula (spatial information).

    Args:
        events (Master.Data): events to look at
        ff_KE (ak.Array): front facing kinetic energy
        method (str): method to calcualte the deposited energy, either "calo" or "bb"

    Returns:
        ak.Array: depotisted energy
    """
    reco_pitch = vector.dist(events.recoParticles.beam_calo_pos[:, :-1], events.recoParticles.beam_calo_pos[:, 1:]) # distance between reconstructed calorimetry points
    
    if method == "calo":
        dE = ak.sum(events.recoParticles.beam_dEdX[:, :-1] * reco_pitch, -1)
    elif method == "bb":
        KE_int_bb = BetheBloch.InteractingKE(ff_KE, ak.sum(reco_pitch, -1), Particle.from_pdgid(211), 50)
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

    def __init__(self, file : str = "data/g4_xs.root", energy_range : list = None) -> None:
        self.file = uproot.open(file) # open root file

        self.KE = self.file["abs_KE;1"].all_members["fX"] # load kinetic energy from one channel (shared for all cross section channels)
        if energy_range:
            self.KE = self.KE[(self.KE <= max(energy_range)) & (self.KE >= min(energy_range))]

        for k in self.file.keys():
            if "KE" in k:
                g = self.file[k]
                setattr(self, self.labels[k], g.all_members["fY"][0:len(self.KE)]) # assign class variables for each cross section channel
        pass

    def __PlotAll(self):
        """ Plot all cross section channels.
        """
        for k in self.labels.values():
            Plots.Plot(self.KE, getattr(self, k), label = k.replace("_", " "), newFigure = False, xlabel = "KE (MeV)", ylabel = "$\sigma (mb)$")

    def Plot(self, xs : str, color : str = None, label : str = None, title : str = None):
        """ Plot cross sections. To be used in conjunction with other plots for comparisons.

        Args:
            xs (str): cross section channel to plot, if given all, will plot all cross section channels
            color (str, optional): colour of single plot. Defaults to None.
            label (str, optional): label of plot, if None, the channel name is used. Defaults to None.
            title (str, optional): title of plot, set to the channel name if label is provided. Defaults to None.
        """
        if xs == "all":
            self.__PlotAll()
        else:
            if label is None:
                label = xs.replace("_", " ")
            else:
                if title is None:
                    title = xs.replace("_", " ")
            Plots.Plot(self.KE, getattr(self, xs), label = label, title = title, newFigure = False, xlabel = "KE (MeV)", ylabel = "$\sigma (mb)$", color = color)


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
        end_slice_pos = slices.pos_to_num(endPos) # using trajectory points gives wierd results, compare the two to see what is different.
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
        # error_mean_energy = np.divide((sum_energy_sqr)**0.5, counts)

        return mean_energy, error_mean_energy

    @staticmethod 
    def CrossSection(n_incident : np.array, n_interact : np.array, slice_width : float) -> tuple[np.array, np.array]:
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


class EnergySlice:
    """ Methods for implementing the energy slice measurement method.
    """
    @staticmethod
    def TrunacteSlices(slice_array : ak.Array, energy_slices : Slices) -> ak.Array:
        """ Custom method for truncating slice numbers due to the fact energy slices should be in reverse order vs kinetic energy.

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
    def CountingExperiment(int_energy : ak.Array, ff_energy : ak.Array, outside_tpc : ak.Array, channel : ak.Array, energy_slices : Slices) -> tuple[np.array, np.array]:
        """ Creates the interacting and incident histograms.

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
    def CrossSection(n_interact : np.array, n_incident : np.array, dEdX : np.array, dE : float) -> tuple[np.array, np.array]:
        """ Compute cross section using ThinSlice.CrossSection, by passing an effective spatial slice width.

        Args:
            n_interact (np.array): interacting histogram
            n_incident (np.array): incident histogram
            dEdX (np.array): mean slice dEdX
            dE (float): energy slice width

        Returns:
            tuple[np.array, np.array]: Cross section and statistical uncertainty.
        """
        return ThinSlice.CrossSection(n_incident, n_interact, dE/dEdX)

    @staticmethod
    def ModifiedCrossSection(n_int_exclusive : np.array, n_inc_exclusive, n_int_inclusive : np.array, n_inc_inclusive : np.array, dEdX : float, dE : float) -> tuple[np.array, np.array]:
        def nandiv(num, den):
            return np.divide(num, np.where(den == 0, np.nan, den))

        NA = 6.02214076e23
        factor = np.array(dEdX) * 10**27 * BetheBloch.A  / (BetheBloch.rho * NA * dE)

        n_interact_ratio = nandiv(n_int_exclusive, n_int_inclusive)
        n_survived_inclusive = n_inc_inclusive - n_int_inclusive

        var_inc_inclusive = n_inc_inclusive # poisson variance
        var_int_inclusive = n_int_inclusive * (1 - nandiv(n_int_inclusive, n_inc_inclusive)) # binomial uncertainty
        var_int_exclusive = n_int_exclusive * (1 - nandiv(n_int_exclusive, n_inc_exclusive)) # binomial uncertainty

        xs = factor * n_interact_ratio * np.log(nandiv(n_inc_inclusive, n_inc_inclusive - n_int_inclusive))

        diff_n_int_exclusive = nandiv(xs, n_int_exclusive)
        diff_n_inc_inclusive = factor * n_interact_ratio * (nandiv(1, n_inc_inclusive) - nandiv(1, n_survived_inclusive))
        diff_n_int_inclusive = factor * n_interact_ratio * nandiv(1, n_survived_inclusive) - nandiv(xs, n_int_inclusive)

        xs_err = ((diff_n_int_exclusive**2 * var_int_exclusive) + (diff_n_inc_inclusive**2 * var_inc_inclusive) + (diff_n_int_inclusive**2 * var_int_inclusive))**0.5
        return xs, xs_err