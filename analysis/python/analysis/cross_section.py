"""
Created on: 02/06/2023 10:37

Author: Shyam Bhuller

Description: Library for code used used in the cross section analysis. Refer to the README to see which apps correspond to the cross section analysis
"""

import awkward as ak
import dill

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