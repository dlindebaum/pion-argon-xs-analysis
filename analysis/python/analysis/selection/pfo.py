"""
Created on: 07/03/2023 14:43

Author: Shyam Bhuller

Description: Lists cuts which selects PFOs in an event or events.
TODO Documentation.
"""
import awkward as ak
import numpy as np

from .common import *

from python.analysis.ntuple import Data, NtupleType

def ValidRecoEnergyCut(events : Data) -> ak.Array:
    """ Select reconstructed PFOs with a non-sentinel shower energy.

        Args:
            events (Data): Event data containing reconstructed shower energies.

        Returns:
            ak.Array: Per-PFO mask that is true where ``shower_energy`` is not ``-999``.
    """
    return events.recoParticles.shower_energy != -999


def ValidRecoPositionCut(events : Data) -> ak.Array:
    """ Select reconstructed PFOs with all shower-start coordinates populated.

        Args:
            events (Data): Event data containing reconstructed shower-start positions.

        Returns:
            ak.Array: Per-PFO mask excluding sentinel x, y, or z coordinates.
    """
    return np.logical_and(
        np.logical_and(
            events.recoParticles.shower_start_pos.x != -999,
            events.recoParticles.shower_start_pos.y != -999
        ),
        events.recoParticles.shower_start_pos.z != 999
    )


def ValidRecoMomentumCut(events : Data) -> ak.Array:
    """ Select reconstructed PFOs with all momentum components populated.

        Args:
            events (Data): Event data containing reconstructed shower momenta.

        Returns:
            ak.Array: Per-PFO mask excluding momentum components equal to ``-999``.
    """
    return np.logical_and(
        np.logical_and(
            events.recoParticles.shower_momentum.x != -999,
            events.recoParticles.shower_momentum.y != -999
        ),
        events.recoParticles.shower_momentum.z != -999
    )


def ValidCNNScoreCut(events : Data, return_property : bool = False) -> ak.Array:
    """ Select reconstructed PFOs whose CNN score is not the missing-value sentinel.

        Args:
            events (Data): Event data containing reconstructed CNN scores.
            return_property (bool): If true, return the CNN-score values alongside the mask.

        Returns:
            ak.Array: Per-PFO validity mask, optionally paired with the CNN-score values.
    """
    return create_mask(-999, "!=", events.recoParticles.cnn_score, return_property)


def GoodShowerSelection(events : Data, return_table = False):
    """ Combine the reconstructed-position, momentum, and energy validity cuts.

        Args:
            events (Data): Event data containing reconstructed PFOs.
            return_table (Any): Whether to return the cut-performance table with the combined mask.

        Returns:
            Any: Combined per-PFO mask, optionally paired with its cut-performance table.
    """
    selections = [
        ValidRecoPositionCut,
        ValidRecoMomentumCut,
        ValidRecoEnergyCut,
    ]
    return combine_selections(events, selections, 1, return_table = return_table)


def EMScoreCut(events : Data, cut = 0.5, return_property : bool = False) -> ak.Array: #TODO Remove
    """ Compare reconstructed track scores with the configured threshold.

        Args:
            events (Data): Event data containing reconstructed track scores.
            cut (Any): Threshold compared with ``recoParticles.track_score``.
            return_property (bool): If true, return the track-score values alongside the mask.

        Returns:
            ak.Array: Per-PFO comparison mask, optionally paired with track-score values.
    """
    return create_mask(cut, "<", events.recoParticles.track_score, return_property)


def NHitsCut(events : Data, cut = 80, op = ">", return_property : bool = False) -> ak.Array:
    """ Compare each reconstructed PFO's hit count with a threshold.

        Args:
            events (Data): Event data containing reconstructed hit counts.
            cut (Any): Hit-count threshold or thresholds.
            op (Any): Comparison operator, or operators for a range cut.
            return_property (bool): If true, return hit counts alongside the mask.

        Returns:
            ak.Array: Per-PFO comparison mask, optionally paired with hit counts.
    """
    return create_mask(cut, op, events.recoParticles.n_hits, return_property)


def BeamParticleDistanceCut(events : Data, cut = [3, 90], op = [">", "<"], return_property : bool = False) -> ak.Array:
    # distance to beam end position in cm
    """ Select PFOs by their distance from the beam particle's end position.

        Args:
            events (Data): Event data used to calculate PFO-to-beam distances.
            cut (Any): Distance threshold or lower/upper bounds, in centimeters.
            op (Any): Comparison operator, or operators corresponding to the bounds.
            return_property (bool): If true, return distances alongside the mask.

        Returns:
            ak.Array: Per-PFO comparison mask, optionally paired with distances.
    """
    dist = find_beam_separations(events)
    return create_mask(cut, op, dist, return_property)


def BeamParticleIPCut(events : Data, cut = 20, op = "<", return_property : bool = False) -> ak.Array:
    """ Select PFOs by their impact parameter relative to the beam particle.

        Args:
            events (Data): Event data used to calculate beam-relative impact parameters.
            cut (Any): Impact-parameter threshold or thresholds.
            op (Any): Comparison operator, or operators for a range cut.
            return_property (bool): If true, return impact parameters alongside the mask.

        Returns:
            ak.Array: Per-PFO comparison mask, optionally paired with impact parameters.
    """
    ip = find_beam_impact_parameters(events)
    return create_mask(cut, op, ip, return_property)


def TrackScoreCut(events : Data, cut = 0.5, op = ">", return_property : bool = False):
    """ Compare reconstructed track scores with one or more thresholds.

        Args:
            events (Data): Event data containing reconstructed track scores.
            cut (Any): Track-score threshold or thresholds.
            op (Any): Comparison operator, or operators for a range cut.
            return_property (bool): If true, return track scores alongside the mask.

        Returns:
            Any: Per-PFO comparison mask, optionally paired with track scores.
    """
    return create_mask(cut, op, events.recoParticles.track_score, return_property)


def BeamDaughterCut(events : Data):
    """ Select reconstructed PFOs whose mother is the beam particle.

        Args:
            events (Data): Event data with reconstructed PFO mother and beam indices.

        Returns:
            Any: Per-PFO boolean mask identifying beam daughters.
    """
    beam_number = ak.where(events.recoParticles.beam_number == -999, -1, events.recoParticles.beam_number) # null beam number (no beam particle) and null mother (pf has no mother) are the same value, so change one of them to something else
    return events.recoParticles.mother == beam_number


def VetoBeamParticle(events : Data):
    """ Exclude the beam PFO from the reconstructed-particle selection.

        Args:
            events (Data): Event data with reconstructed PFO and beam-particle indices.

        Returns:
            Any: Per-PFO mask that is true for PFOs other than the beam particle.
    """
    return events.recoParticles.beam_number != events.recoParticles.number


def PiPlusSelection(events : Data, cut = [0.5, 2.8], op = [">", "<"], return_property : bool = False):
    """ Select reconstructed PFOs by the median track energy-deposition rate.

        Args:
            events (Data): Event data containing reconstructed track dE/dx values.
            cut (Any): dE/dX threshold or lower/upper bounds for pion candidates.
            op (Any): Comparison operator, or operators corresponding to the bounds.
            return_property (bool): If true, return median dE/dx values alongside the mask.

        Returns:
            Any: Per-PFO comparison mask, optionally paired with median dE/dx values.
    """
    median_dEdX = median(events.recoParticles.track_dEdX)
    return create_mask(cut, op, median_dEdX, return_property)


def ProtonSelection(events : Data, cut = 2.8, op = ">", return_property : bool = False):
    """ Select reconstructed PFOs by the median track energy-deposition rate.

        Args:
            events (Data): Event data containing reconstructed track dE/dx values.
            cut (Any): dE/dX threshold or thresholds for proton candidates.
            op (Any): Comparison operator, or operators for a range cut.
            return_property (bool): If true, return median dE/dx values alongside the mask.

        Returns:
            Any: Per-PFO comparison mask, optionally paired with median dE/dx values.
    """
    median_dEdX = median(events.recoParticles.track_dEdX)
    return create_mask(cut, op, median_dEdX, return_property)


def Chi2ProtonSelection(events : Data, cut = 80, op = "<", return_property : bool = False):
    """ Select reconstructed PFOs by proton-hypothesis chi-squared per degree of freedom.

        Args:
            events (Data): Event data containing proton-hypothesis chi-squared and degrees of freedom.
            cut (Any): Chi-squared-per-degree-of-freedom threshold or thresholds.
            op (Any): Comparison operator, or operators for a range cut.
            return_property (bool): If true, return the calculated ratio alongside the mask.

        Returns:
            Any: Per-PFO comparison mask, optionally paired with chi-squared ratios.
    """
    chi2_ndf = events.recoParticles.track_chi2_proton / events.recoParticles.track_chi2_proton_ndof
    return create_mask(cut, op, chi2_ndf, return_property)

def TrackLengthSelection(events : Data, cut = 27.1, op = ">", return_property : bool = False):
    """ Compare reconstructed PFO track lengths with one or more thresholds.

        Args:
            events (Data): Event data containing reconstructed track lengths.
            cut (Any): Track-length threshold or thresholds.
            op (Any): Comparison operator, or operators for a range cut.
            return_property (bool): If true, return track lengths alongside the mask.

        Returns:
            Any: Per-PFO comparison mask, optionally paired with track lengths.
    """
    return create_mask(cut, op, events.recoParticles.track_len, return_property)

def initial_pi0_photon_selection(
        events : Data,
        em_cut : float = 0.5,
        n_hits_cut : int = 80,
        distance_bounds_cm : tuple = [3, 90],
        max_impact_cm : float = 20,
        veto_beam_particle : bool = True,
        verbose : bool = False,
        return_table : bool = False):
    """ Combine photon-quality, beam-distance, and impact-parameter cuts for pi0 candidates.

        Args:
            events (Data): Event data containing reconstructed photon-candidate PFOs.
            em_cut (float): Maximum ``track_score`` threshold passed to ``EMScoreCut``.
            n_hits_cut (int): Minimum hit count required for a candidate.
            distance_bounds_cm (tuple): Lower and upper beam-distance bounds, in centimeters.
            max_impact_cm (float): Maximum beam-relative impact parameter, in centimeters.
            veto_beam_particle (bool): Whether to exclude the beam PFO from candidates.
            verbose (bool): Whether to print the cut-by-cut selection summary.
            return_table (bool): Whether to return the cut-performance table with the mask.

        Returns:
            Any: Combined PFO mask, optionally paired with the cut-performance table.
    """
    selections = [
        EMScoreCut,
        NHitsCut,
        BeamParticleDistanceCut,
        BeamParticleIPCut
    ]
    arguments = [
        {"cut": em_cut},
        {"cut": n_hits_cut},
        {"cut": distance_bounds_cm},
        {"cut": max_impact_cm}
    ]
    if veto_beam_particle:
        selections.append(VetoBeamParticle)
        arguments.append({})
    return combine_selections(events, selections, 1, arguments, verbose, return_table)


def daughter_pi_plus_selection(
        events : Data,
        track_cut : float = 0.5,
        n_hits_cut : int = 20,
        min_dEdX : float = 0.5,
        max_dEdX : float = 2.8,
        verbose : bool = False,
        return_table : bool = False):
    """ Select reconstructed daughter PFOs consistent with charged pions.

        Args:
            events (Data): Event data containing reconstructed PFOs.
            track_cut (float): Minimum track score required for a candidate.
            n_hits_cut (int): Minimum number of hits required for a candidate.
            min_dEdX (float): Lower bound on median track dE/dx.
            max_dEdX (float): Upper bound on median track dE/dx.
            verbose (bool): Whether to print the cut-by-cut selection summary.
            return_table (bool): Whether to return the cut-performance table with the mask.

        Returns:
            Any: Combined candidate mask, optionally paired with the cut-performance table.
    """
    selections = [
        TrackScoreCut,
        NHitsCut,
        PiPlusSelection
    ]
    arguments = [
        {"cut" : track_cut},
        {"cut" : n_hits_cut},
        {"cut" : [min_dEdX, max_dEdX]}
    ]
    if events.ntuple_type == NtupleType.SHOWER_MERGING:
        print("Apply beam daughter cuts")
        selections.insert(0, BeamDaughterCut)
        arguments.insert(0, {}) # the cut changes slightly depending on the ntuple type we use.
    return combine_selections(events, selections, 1, arguments, verbose, return_table)


def daughter_proton_selection(
        events : Data,
        track_cut : float = 0.5,
        n_hits_cut : int = 20,
        dEdX : float = 2.8,
        verbose : bool = False,
        return_table : bool = False):
    """ Select reconstructed daughter PFOs consistent with protons.

        Args:
            events (Data): Event data containing reconstructed PFOs.
            track_cut (float): Minimum track score required for a candidate.
            n_hits_cut (int): Minimum number of hits required for a candidate.
            dEdX (float): Minimum median track dE/dx for the proton selection.
            verbose (bool): Whether to print the cut-by-cut selection summary.
            return_table (bool): Whether to return the cut-performance table with the mask.

        Returns:
            Any: Combined candidate mask, optionally paired with the cut-performance table.
    """
    selections = [
        TrackScoreCut,
        NHitsCut,
        ProtonSelection
    ]
    arguments = [
        {"cut" : track_cut},
        {"cut" : n_hits_cut},
        {"cut" : dEdX}
    ]
    if events.ntuple_type == NtupleType.SHOWER_MERGING:
        selections.insert(0, BeamDaughterCut)
        arguments.insert(0, {}) # the cut changes slightly depending on the ntuple type we use.
    return combine_selections(events, selections, 1, arguments, verbose, return_table)
