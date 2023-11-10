"""
Created on: 06/03/2023 18:41

Author: Shyam Bhuller

Description: 
"""
import awkward as ak
import numpy as np

from python.analysis import Master
from python.analysis.SelectionTools import *


@CountsWrapper
def HasTruthInfo(events : Master.Data) -> ak.Array:
    #* remove events with no truth info
    return ak.num(events.trueParticles.number) > 0

@CountsWrapper
def Pi0InFinalState(events : Master.Data, n_pi0 : int = 1) -> ak.Array:
    #* only look at events with 1 primary pi0
    pi0 = events.trueParticles.PrimaryPi0Mask
    mask = ak.num(pi0[pi0]) == n_pi0
    return mask


def FinalStatePi0Cut(events : Master.Data) -> ak.Array:
    #* remove true particles which aren't primaries
    #! this cuts on truth information, not events
    return events.trueParticles.PrimaryPi0Mask | events.trueParticles.truePhotonMask 

@CountsWrapper
def BeamMCFilter(events : Master.Data, n_pi0 : int = 1) -> ak.Array:
    """ Filters BeamMC data to get events with only 1 pi0 which originates from the beam particle interaction.

    Args:
        events (Data): events to filter

    Returns:
        ak.Array : boolean mask
    """
    empty = HasTruthInfo(events)
    pi0 = Pi0InFinalState(events, n_pi0)
    truth_cut = FinalStatePi0Cut(events)
    truth_event_cut = ak.any(truth_cut, -1) # convert truth level cut to event level cut

    return (empty & pi0) & truth_event_cut

@CountsWrapper
def PiBeamSelection(events : Master.Data) -> ak.Array:
    true_beam = events.trueParticlesBT.pdg[events.recoParticles.beam_number == events.recoParticles.number]
    return ak.all(true_beam == 211, -1)

@CountsWrapper
def DiPhotonCut(events : Master.Data):
    return Master.Pi0TwoBodyDecayMask(events)

@CountsWrapper
def RecoBeamParticleCut(events : Master.Data) -> ak.Array:
    return (events.recoParticles.beam_number != -999) & (events.recoParticles.beam_endPos.x != -999)

@CountsWrapper
def HasPFO(events : Master.Data) -> ak.Array:
    return Master.NPFPMask(events, -1)

@CountsWrapper
def HasBacktracked(events : Master.Data):
    unique = events.trueParticlesBT.GetUniqueParticleNumbers(events.trueParticlesBT.number)
    return ak.num(unique) > 1

@CountsWrapper
def BothPhotonsBacktracked(events : Master.Data) -> ak.Array:
    pi0 = events.trueParticles.number[events.trueParticles.PrimaryPi0Mask]
    pi0 = ak.fill_none(ak.pad_none(pi0, ak.max(ak.num(pi0)), -1), -999, -1)

    f = [events.trueParticlesBT.mother == pi0[:, i] for i in range(ak.max(ak.num(pi0)))] # get PFOs which are daughters of primiary pi0s
    
    f_combined = None
    for i in range(ak.max(ak.num(pi0))):
        if hasattr(f_combined, "__iter__"):
            f_combined = f_combined | f[i]
        else:
            f_combined = f[i]

    not_null = np.logical_not(np.logical_or(events.recoParticles.shower_start_pos.x == -999, events.recoParticles.momentum.x == -999))
    f = f_combined & not_null

    f = f & (events.trueParticlesBT.number != -999)

    daughters = events.trueParticlesBT.number[f]
    unique_daughters = events.trueParticlesBT.GetUniqueParticleNumbers(daughters)
    unique_daughters = ak.num(unique_daughters)

    mask = unique_daughters == 2
    return mask


def CreateLegacyBeamParticleSelection(events : Master.Data, verbose : bool = True):
    selections = [
        BeamMCFilter,
        PiBeamSelection,
        DiPhotonCut,
        RecoBeamParticleCut,
        HasPFO,
        HasBacktracked,
        BothPhotonsBacktracked,
    ]
    return CombineSelections(events, selections, 0, verbose = verbose, return_table = True)