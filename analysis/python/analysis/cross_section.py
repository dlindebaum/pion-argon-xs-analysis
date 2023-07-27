"""
Created on: 02/06/2023 10:37

Author: Shyam Bhuller

Description: Library for code used in the cross section analysis. Refer to the README to see which apps correspond to the cross section analysis.
"""
import argparse
import json

import awkward as ak
import numpy as np
import dill
from particle import Particle

from python.analysis import Master, BeamParticleSelection, PFOSelection, EventSelection, Fitting
from python.analysis.shower_merging import SetPlotStyle

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
        "response": ResponseCorrection
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

    pip_charge = 1

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
    def ResolveArgs(args : argparse.Namespace):

        if hasattr(args, "config"):
            args_copy = argparse.Namespace()
            for a, v in args._get_kwargs():
                setattr(args_copy, a, v)
            args = ApplicationArguments.ResolveConfig(LoadConfiguration(args.config))
            print(args.mc_file)
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
        print(args)

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
    def __CreateSelection(value : dict, module):
        selection = {"selections" : [], "arguments" : []}
        for func, opt in value.items():
            if opt["enable"] is True:
                selection["selections"].append(getattr(module, func))
                copy = opt.copy()
                copy.pop("enable")
                selection["arguments"].append(copy)
        return selection

    @staticmethod
    def ResolveConfig(config : dict):
        args = argparse.Namespace()
        for head, value in config.items():
            if head == "NTUPLE_FILE":
                args.mc_file = value["mc"]
                args.data_file = value["data"]
                args.ntuple_type = value["type"]
            elif head == "BEAM_QUALITY_FITS":
                args.mc_beam_quality_fit = value["mc"]
                args.data_beam_quality_fit = value["data"]
            elif head == "BEAM_SCAPER_FITS":
                args.mc_beam_scraper_fit = value["mc"]
            elif head == "ENERGY_CORRECTION":
                args.correction = value["correction"]
                args.correction_params = value["correction_params"]
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
            else:
                setattr(args, head, value) # allow for generic configurations in the json file
        return args


def LoadConfiguration(file : str):
    with open(file, "rb") as f:
        config = json.load(f)
    return config