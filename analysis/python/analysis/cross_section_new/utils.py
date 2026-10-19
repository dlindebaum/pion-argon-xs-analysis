import awkward as ak
import numpy as np

from particle import Particle

from Master import Data

import SelectionTools
import Fitting
import BetheBloch
import vector

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

### Kinematics ###

def KE(p, m):
    return (p**2 + m**2)**0.5 - m

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
        KE_end = BetheBloch.KE_end(KE_init, track_length, 50)
    else:
        raise Exception(f"{method} not a valid method, pick 'calo' or 'track'")
    return KE_end
