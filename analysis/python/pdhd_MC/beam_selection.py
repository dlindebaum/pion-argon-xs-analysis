"""
Created on: 22/05/2026

Author: Samikshya Kar

Description: Contains cuts for Beam Particle Selection.
"""
import sys
sys.path.insert(1, '/users/gj23442/analysis/pion-argon-xs-analysis/analysis/')

import numpy as np
import awkward as ak

from python.analysis import Master, vector
from python.analysis.SelectionTools import *
#from python.analysis import SelectionEvaluation as seval
from python.analysis import Fitting, cross_section
from python.analysis.PFOSelection import Median

## Purity and Efficiency helper functions

def signal_mask(evts):
    mask = (evts.trueParticlesBT.beam_pdg == 211)  #Backtracked PDG for now because we're not clear how to match it to truth 
                                                   #& (ak.firsts(evts.trueParticlesBT.number)==ak.firsts(evts.trueParticles.number))
    return mask

def true_signal_mask(evts):
    beam_pdg = ak.firsts(evts.trueParticles.pdg[evts.trueParticles.number == 1])
    mask = beam_pdg == 211
    return mask

def BeamTriggerSelection(events: Data, pdgs : list[int] = [211, 13, -13], use_beam_inst : bool = False, return_property : bool = False):
    """ Beam particle selection using beam instrumentation information for Data and truth information if MC.

    Args:
        events (Master.Data): events to study
        pdgs (list[int], optional): list of particle pdgs. Defaults to [211, 13, -13].
        use_beam_inst (bool, optional): use beam instrumentation (enable if using data). Defaults to False.
        return_property (bool, optional): return beam pdg. Defaults to False.
    """
    def compare_beam_pdg(pdg : int):
        return ak.fill_none(ak.pad_none(events.recoParticles.beam_inst_PDG_candidates, 1, -1), -1, -1) == pdg

    if use_beam_inst:
        mask = events.recoParticles.beam_inst_valid
        mask = mask & (events.recoParticles.beam_inst_trigger != 8)
        mask = mask & (events.recoParticles.beam_inst_nTracks == 1) & (events.recoParticles.beam_inst_nMomenta == 1)

        beam_pdg = None
        for i in pdgs:
            tmp = compare_beam_pdg(i)
            if beam_pdg is None:
                beam_pdg = tmp
            else:
                beam_pdg = beam_pdg | tmp
        mask = mask & ak.any(beam_pdg, axis = -1)
        mask = mask & events.recoParticles.reco_reconstructable_beam_event
    else:
        #beam_pdg = ak.flatten(events.trueParticles.pdg[events.trueParticles.number == 1])
        #Flattening caused the empty entries to be removed which caused reduced evt num lengthed masks to shorten. 
        beam_pdg = ak.firsts(events.trueParticles.pdg[events.trueParticles.number == 1])
        beam_pdg = ak.fill_none(beam_pdg, -999)
        mask = ak.any([beam_pdg == i for i in pdgs], axis = 0)
    if return_property is True:
        return mask, beam_pdg
    else:
        return mask

def BeamValidSelection(events: Data, cut : int = 0, op : str = ">", return_property : bool = False) -> ak.Array:
    """ Beam particle selection using beam instrumentation information for Data and truth information if MC.

    Args:
        events (Master.Data): events to study
    """
    
    n_beam_particles = events.recoParticles.beam_particles
     # Analyser fills the empty entry with a -999
    n_beam_particles = n_beam_particles[n_beam_particles != -999]

    return CreateMask(cut, op, n_beam_particles, return_property)

def BeamTrackSelection(events: Data, cut : int = 13, op = "==", return_property : bool = False) -> ak.Array:
    """ Cut on Pandora slice tag, selects track like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """

    return CreateMask(cut, op, events.recoParticles.beam_pandora_tag, return_property)

def tpc_entrance_mask(evts):
    traj_z = evts.trueParticles.beam_traj_pos.z
    traj_z = traj_z[traj_z >= 0]
    mask = evts.trueParticles.beam_traj_pos.z == ak.firsts(traj_z)
    return mask

def BeamQualityPosition(events: Data, bins: int = 10, x_percentiles: tuple = (10, 90), y_percentiles: tuple = (20, 90)):
    """Fit Gaussians to the beam start position distributions in x and y.

    Args:
        events (Data): events to study.
        bins (int): number of histogram bins. Defaults to 10.
        x_percentiles (tuple): (lower, upper) percentiles for x fit range. Defaults to (10, 90).
        y_percentiles (tuple): (lower, upper) percentiles for y fit range. Defaults to (20, 90).

    Returns:
        tuple: (mu, mu_err, sigma, sigma_err) dicts keyed by "x" and "y".
    """
    def range_from_percentiles(data, lower=10, upper=90):
        return sorted([np.nanpercentile(data, lower), np.nanpercentile(data, upper)])

    reco_startpos = events.recoParticles.beam_startPos

    fit_ranges = {
        "x": range_from_percentiles(reco_startpos.x, *x_percentiles),
        "y": range_from_percentiles(reco_startpos.y, *y_percentiles),
    }

    mu, mu_err, sigma, sigma_err = {}, {}, {}, {}
    for i in ["x", "y"]:
        data = reco_startpos[i]
        # fit only within percentile range to exclude large tails
        y, bin_edges = np.histogram(np.array(data[~np.isnan(data)]), bins=bins, range=fit_ranges[i])
        yerr = np.sqrt(y)
        popt, perr = Fitting.Fit(cross_section.bin_centers(bin_edges), y, yerr, Fitting.gaussian)
        mu[i] = popt[1]
        sigma[i] = abs(popt[2])
        mu_err[i] = abs(perr[1])
        sigma_err[i] = abs(perr[2])

    return mu, mu_err, sigma, sigma_err

def BeamQualitySelection(events: Data, cut : int = 3, op : str = "<", return_property : bool = False) -> ak.Array:

    reco_startpos = events.recoParticles.beam_startPos

    mu, _, sigma, _ = BeamQualityPosition(events)

    beam_dx = (reco_startpos.x - mu["x"])/ sigma["x"] 
    beam_dy = (reco_startpos.y - mu["y"])/ sigma["y"]
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5

    return beam_dxy, CreateMask(cut, op, beam_dxy, return_property)

def MedianDEdXCut(events: Data, cut : float = 2.4, op = "<", return_property : bool = False) -> ak.Array:
    """ cut on median dEdX to exlude proton background.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    dEdX = events.recoParticles.beam_dEdX_noSCE

    median = Median(dEdX)
    
    return CreateMask(cut, op, median, return_property)


