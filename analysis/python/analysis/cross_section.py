"""
Created on: 02/06/2023 10:37

Author: Shyam Bhuller

Description: Library for code used used in the cross section analysis. Refer to the README to see which apps correspond to the cross section analysis
"""
import argparse

import awkward as ak
import numpy as np
import dill
from particle import Particle
from scipy.optimize import curve_fit

from python.analysis import Master, BeamParticleSelection, PFOSelection, EventSelection

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


def Gaussian(x : np.array, a : float, x0 : float, sigma : float) -> np.array:
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def Fit_Gaussian(data : ak.Array, bins : int) -> tuple:
    """ Fits a gaussian function to a histogram of data, using the least squares method.

    Args:
        data (ak.Array): data to fit
        bins (int): number of bins
        range (list, optional): range of values to fit to. Defaults to None.

    Returns:
        tuple : fit parameters and covariance matrix
    """
    y, bins_edges = np.histogram(np.array(data[~np.isnan(data)]), bins = bins, range = sorted([np.nanpercentile(data, 10), np.nanpercentile(data, 90)])) # fit only to  data within the 10th and 90th percentile of data to exclude large tails in the distriubtion.
    bin_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
    return curve_fit(Gaussian, bin_centers, y, p0 = (0, np.nanmedian(data), np.nanstd(data)))



def LinearCorrection(x, p0):
    return x / p0


def ResponseFit(x, p0, p1, p2):
    return p0 * np.log(x - p1) + p2


def ResponseCorrection(x, p0, p1, p2):
    return x / (ResponseFit(x, p0, p1, p2) + 1)

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

        delta = ak.where(y >= BetheBloch.y1, 2 * np.log(10)*y - BetheBloch.C, 0) 
        delta = ak.where((BetheBloch.y0 <= y) & (y < BetheBloch.y1), 2 * np.log(10)*y - BetheBloch.C + BetheBloch.a * (BetheBloch.y1 - y)**BetheBloch.k, delta)

        # if y >= BetheBloch.y1:
        #     delta = 2 * np.log(10)*y - BetheBloch.C
        # elif BetheBloch.y0 <= y < BetheBloch.y1:
        #     delta = 2 * np.log(10)*y - BetheBloch.C + BetheBloch.a * (BetheBloch.y1 - y)**BetheBloch.k
        # else:
        #     delta = 0

        return delta

    @staticmethod
    def meandEdX(KE, particle : Particle):
        gamma = (KE / particle.mass) + 1
        beta = (1 - (1/(gamma**2)))**0.5

        w_max = 2 * BetheBloch.me * (beta * gamma)**2 / (1 + (2 * BetheBloch.me * (gamma/particle.mass)) + (BetheBloch.me/particle.mass)**2)

        dEdX = (BetheBloch.rho * BetheBloch.K * BetheBloch.Z * (particle.charge)**2) / ( BetheBloch.A * beta**2 * (0.5 * np.log(2 * BetheBloch.me * (gamma**2) * (beta**2) * w_max / (BetheBloch.I**2))) - beta**2 - (BetheBloch.densityCorrection(beta, gamma) / 2) )
        return dEdX
    

class ApplicationArguments:
    @staticmethod
    def Ntuples(parser : argparse.ArgumentParser, data : bool = False):
        parser.add_argument(dest = "mc_file", nargs = "+", help = "MC NTuple file to study.")
        if data: parser.add_argument("-d", "--data-file", dest = "data_file", nargs = "+", help = "Data Ntuple to study")
        parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {[m.value for m in Master.Ntuple_Type]}.", required = True)
        return

    @staticmethod
    def SingleNtuple(parser : argparse.ArgumentParser, define_sample : bool = True):
        parser.add_argument(dest = "file", help = "NTuple file to study.")
        parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {[m.value for m in Master.Ntuple_Type]}.", required = True)
        if define_sample : parser.add_argument("-S", "--sample-type", dest = "sample_type", type = str, choices = ["mc", "data"], help = f"type of sample I am looking at.", required = True)
        return

    @staticmethod
    def BeamQualityCuts(parser : argparse.ArgumentParser, data : bool = False):
        parser.add_argument("--mc_beam_quality_fit", dest = "mc_beam_quality_fit", type = str, help = "mc fit values for the beam quality cut.", required = True)
        if data: parser.add_argument("--data_beam_quality_fit", dest = "data_beam_quality_fit", type = str, default = None, help = "data fit values for the beam quality cut.")
        return
    
    @staticmethod
    def Processing(parser : argparse.ArgumentParser):
        parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "number of batches to split n tuple files into when parallel processing processing data.")
        parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "number of events to process when parallel processing data.")
        parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "number of threads to use when processsing")

    @staticmethod
    def Output(parser : argparse.ArgumentParser):
        parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save plots")
        return

    @staticmethod
    def BeamSelection(parser : argparse.ArgumentParser):
        parser.add_argument("--scraper", action = "store_true", help = "Toggle to enable the beam scraper cut for the beam particle selection.")
        return

    @staticmethod
    def ShowerCorrection(parser : argparse.ArgumentParser):
        parser.add_argument("-c, --shower_correction", nargs = 2, dest = "correction", help = f"shower energy correction method {tuple(shower_energy_correction.keys())} followed by a correction parameters json file.", required = False)
        return

    @staticmethod
    def Plots(parser : argparse.ArgumentParser):
        parser.add_argument("--nbins", dest = "nbins", type = int, default = 50, help = "number of bins to make for histogram plots.")
        parser.add_argument("-a", "--annotation", dest = "annotation", type = str, default = None, help = "annotation to add to plots")
        return
    
    @staticmethod
    def ResolveArgs(args : argparse.Namespace):
        if hasattr(args, "out"):
            if args.out is None:
                filename = None
                if hasattr(args, "mc_file"):
                    filename = args.mc_file
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
                    args.out = args.file.split("/")[-1].split(".")[0] + "/"
            if args.out[-1] != "/": args.out += "/"

        if hasattr(args, "data_file") and hasattr(args, "data_beam_quality_fit"):
            if args.data_file is not None and args.data_beam_quality_fit is None:
                raise Exception("beam quality fit values for data are required")

        if hasattr(args, "correction") and args.correction:
            args.correction_params = args.correction[1]
            args.correction = shower_energy_correction[args.correction[0]]
        return