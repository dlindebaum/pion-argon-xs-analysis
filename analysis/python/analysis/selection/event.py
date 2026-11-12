import time
import warnings
import awkward as ak
import numpy as np
import pandas as pd

from .common import *
from . import beam, pfo

from python.analysis import ntuple, tags
from python.analysis.ntuple import Data

#######################################################################
#######################################################################
##########                   EVENT TAGGING                   ##########
#######################################################################
#######################################################################


def generate_true_final_state_tags(events : Data = None) -> tags.Tags:
    """ Generate truth tags for final state of the beam interaction.

        Args:
            events (Data): Events to look at. Defaults to None.

        Returns:
            tags.Tags: Tags.
    """
    t = tags.Tags()
    t["charge_exchange" ] = tags.Tag("$1\\pi^{0} + 0\\pi^{+}$"              , "charge_exchange" , "#8EBA42", generate_truth_tags(events, 1, 0      , only_diphoton = False) if events is not None else None, 0)
    t["absorption"      ] = tags.Tag("$0\\pi^{0} + 0\\pi^{+}$"              , "absorption"      , "#777777", generate_truth_tags(events, 0, 0      , only_diphoton = False) if events is not None else None, 1)
    t["pion_prod_1_pi0" ] = tags.Tag("$1\\pi^{0} + \\geq 1\\pi^{+}$"         , "pion_prod_1_pi0" , "#E24A33", generate_truth_tags(events, 1, (1,)   , only_diphoton = False) if events is not None else None, 2)
    t["pion_prod_0_pi0" ] = tags.Tag("$0\\pi^{0} + \\geq 1\\pi^{+}$"         , "pion_prod_0_pi0" , "#988ED5", generate_truth_tags(events, 0, (1,)   , only_diphoton = False) if events is not None else None, 3)
    t["pion_prod_>1_pi0"] = tags.Tag("$> 1\\pi^{0} + \\geq 0\\pi^{+}$"       , "pion_prod_>1_pi0", "#348ABD", generate_truth_tags(events, (2,), (0,), only_diphoton = False) if events is not None else None, 4)
    return t


#######################################################################
#######################################################################
##########                  EVENT SELECTION                  ##########
#######################################################################
#######################################################################


def NPhotonCandidateSelection(events : Data, photon_mask : ak.Array, cut : int, op = "==", return_property : bool = False):
    """ Select events by the number of PFOs passing the photon-candidate mask.

        Args:
            events (Data): Event data associated with the candidate mask.
            photon_mask (ak.Array): Per-PFO boolean mask identifying photon candidates.
            cut (int): Required candidate count or count threshold.
            op (Any): Comparison operator applied to the per-event candidate count.
            return_property (bool): If true, return candidate counts alongside the mask.

        Returns:
            Any: Per-event selection mask, optionally paired with candidate counts.
    """
    n_photons = ak.sum(photon_mask, -1)
    return create_mask(cut, op, n_photons, return_property)


def Pi0OpeningAngleSelection(events : Data, photon_mask : ak.Array = None, photon_coords : ak.Array = None, cut = [10, 80], op = [">", "<"], return_property : bool = False):
    """ Select photon pairs by their reconstructed opening angle.

        Args:
            events (Data): Event data used to construct reconstructed shower pairs.
            photon_mask (ak.Array): Optional per-PFO mask used to form photon pairs.
            photon_coords (ak.Array): Optional explicit shower-pair coordinates used instead of a mask.
            cut (Any): Opening-angle threshold or bounds, in degrees.
            op (Any): Comparison operator, or operators corresponding to the bounds.
            return_property (bool): If true, return opening angles alongside the mask.

        Returns:
            Any: Pair-selection mask, optionally paired with opening angles.
    """
    if photon_mask is not None:
        shower_pairs = ntuple.ShowerPairs(events, shower_pair_mask = photon_mask)
    elif photon_coords is not None:
        shower_pairs = ntuple.ShowerPairs(events, pair_coords = photon_coords)
    else:
        raise Exception("photon mask and photon_coords cannot both be None")

    angle = ak.fill_none(ak.pad_none(shower_pairs.reco_angle, 1, -1), -999, -1)
    if hasattr(cut, "__getitem__"):
        cut = [c * np.pi / 180 for c in cut]
    else:
        cut = cut * np.pi / 180
    return create_mask(cut, op, angle, return_property)


def Pi0MassSelection(events : Data, photon_mask : ak.Array = None, photon_coords : ak.Array = None, cut = [50, 250], op = [">", "<"], correction = None, correction_params : list = None, return_property : bool = False):
    """ Select photon pairs by their reconstructed diphoton invariant mass.

        Args:
            events (Data): Event data used to construct reconstructed shower pairs.
            photon_mask (ak.Array): Optional per-PFO mask used to form photon pairs.
            photon_coords (ak.Array): Optional explicit shower-pair coordinates used instead of a mask.
            cut (Any): Invariant-mass threshold or bounds.
            op (Any): Comparison operator, or operators corresponding to the bounds.
            correction (Any): Optional function applied to each photon's reconstructed energy.
            correction_params (list): Parameters passed to ``correction``; its ``value`` mapping is used.
            return_property (bool): If true, return invariant masses alongside the mask.

        Returns:
            Any: Pair-selection mask, optionally paired with invariant masses.
    """
    if photon_mask is not None:
        shower_pairs = ntuple.ShowerPairs(events, shower_pair_mask = photon_mask)
    elif photon_coords is not None:
        shower_pairs = ntuple.ShowerPairs(events, pair_coords = photon_coords)
    else:
        raise Exception("photon mask and photon_coords cannot both be None")

    if correction is None:
        le = shower_pairs.reco_lead_energy
        se = shower_pairs.reco_sub_energy
    else:
        le = correction(shower_pairs.reco_lead_energy, **correction_params["value"])
        se = correction(shower_pairs.reco_sub_energy, **correction_params["value"])

    mass = shower_pairs.mass(le, se, shower_pairs.reco_angle)
    mass = ak.fill_none(ak.pad_none(mass, 1, -1), -999, -1)
    return create_mask(cut, op, mass, return_property)


def pi0_selection(
    events: Data,
    photon_candidates_mask : ak.Array = None,
    photon_candidates_coords : ak.Array = None,
    exact_photon_candidates : bool = True,
    n : int = 2,
    angle_cuts = [10, 80],
    mass_cuts = [50, 250],
    correction = None,
    correction_params : list = None,
    verbose : bool = False,
    return_table : bool = False):
    """ Combine photon-pair angle and mass cuts, with an optional candidate-count requirement.

        Args:
            events (Data): Event data containing reconstructed photon-candidate PFOs.
            photon_candidates_mask (ak.Array): Optional per-PFO mask used to form shower pairs.
            photon_candidates_coords (ak.Array): Optional explicit shower-pair coordinates.
            exact_photon_candidates (bool): Whether to require exactly ``n`` masked candidates.
            n (int): Required number of candidates when ``exact_photon_candidates`` is true.
            angle_cuts (Any): Opening-angle threshold or bounds, in degrees.
            mass_cuts (Any): Diphoton invariant-mass threshold or bounds.
            correction (Any): Optional function applied to each photon's reconstructed energy.
            correction_params (list): Parameters passed to ``correction``.
            verbose (bool): Whether to print the cut-by-cut selection summary.
            return_table (bool): Whether to return the cut-performance table with the mask.

        Returns:
            Any: Combined per-event selection mask, optionally paired with the cut-performance table.
    """
    selections = [
        Pi0OpeningAngleSelection,
        Pi0MassSelection
    ]
    arguments = [
        {"photon_mask" : photon_candidates_mask, "photon_coords" : photon_candidates_coords , "cut" : angle_cuts},
        {"photon_mask" : photon_candidates_mask, "photon_coords" : photon_candidates_coords , "cut" : mass_cuts, "correction" : correction, "correction_params" : correction_params}
    ]

    if (exact_photon_candidates is True) and (photon_candidates_mask is not None):
        selections.insert(0, NPhotonCandidateSelection)
        arguments.insert(0, {"photon_mask" : photon_candidates_mask, "cut" : n})

    print("Pi0Selection")
    return combine_selections(events, selections, 0, arguments, verbose, return_table)

#######################################################################
#######################################################################
##########             EVENT TYPE TRUTH TAGGING              ##########
#######################################################################
#######################################################################


def count_diphoton_decays(events, beam_daughters=True):
    """ Returns the number of truth pi0 particles which decay to yy in each
        event in `events`.

        Pi0 -> yy.

        Args:
            events (Any): Data Events in which to count pi0 occurances.
            beam_daughters (Any): Boolean, optional Whether to only accept a pi0 if it is a daughter of the beam particle. Default is True.

        Returns:
            counts: ak.Array containing the number of truth pi0 diphoton decays for each event.
    """
    def get_two_count_pi0s(pi0s):
        """ Return particle identifiers that occur exactly twice in a list.

            Args:
                pi0s (Any): Pi0 particle identifiers, repeated once for each daughter photon.

            Returns:
                Any: Identifiers whose occurrence count is exactly two.
        """
        pions, counts = np.unique(pi0s, return_counts=True)
        return pions[counts == 2]

    if beam_daughters:
        beam_daughter_filter = events.true_particles.mother == 1
    else:
        beam_daughter_filter = True
    beam_cadidate_pi0s = events.true_particles.number[np.logical_and(
        beam_daughter_filter,
        events.true_particles.pdg == 111)]
    try:
        pi0_daughters = events.true_particles.mother_pdg == 111
    except:
        pi0_daughters = pfo.get_mother_pdgs(events) == 111
    pi0_photon_mothers = events.true_particles.mother[np.logical_and(
        events.true_particles.pdg == 22,
        pi0_daughters)]
    counts = ak.Array(map(
        lambda pi0s: len(
            np.intersect1d(
                pi0s['beam'],
                get_two_count_pi0s(pi0s['photon']))),
        ak.zip(
            {"beam": beam_cadidate_pi0s, "photon": pi0_photon_mothers},
            depth_limit=1)))
    return counts


def count_all_pi0s(events, beam_daughters=True):
    """ Count truth pi0 particles in each event, optionally restricting them to beam daughters.

        Args:
            events (Any): Event data containing truth particles.
            beam_daughters (Any): Whether to count only pi0s whose mother is the beam particle. Defaults to True.

        Returns:
            counts: ak.Array containing the selected truth-pi0 count for each event.
    """
    if beam_daughters:
        beam_daughter_filter = events.true_particles.mother == 1
    else:
        beam_daughter_filter = True
    beam_cadidate_pi0s = events.true_particles.number[np.logical_and(
        beam_daughter_filter,
        events.true_particles.pdg == 111)]
    return ak.sum(beam_cadidate_pi0s, axis=-1)


def count_non_beam_charged_pi(events, beam_daughters=True):
    """ Returns the number of truth pi+ particles event in `events`.

        Args:
            events (Any): Data Events in which to count pi+ occurances.
            beam_daughters (Any): Boolean, optional Whether to only accept a pi+ if it is a daughter of the beam particle. Default is True.

        Returns:
            counts: ak.Array Array containing the number of pi+ particles for each event.
    """
    if beam_daughters:
        daughter_truth = events.true_particles.mother == 1
    else:
        daughter_truth = ak.ones_like(events.true_particles.mother, dtype=bool)
    non_beam_pi_mask = np.logical_and(
        daughter_truth,
        np.logical_and(events.true_particles.pdg == 211,
                       events.true_particles.number != 1))
    return ak.sum(non_beam_pi_mask, axis=-1)


def count_pi0_candidates(
        events,
        mass_cut=[50, 250],
        opening_angle_deg=[10, 80],
        exactly_two_photons=False,
        photon_mask=None,
        correction = None,
        correction_params : list = None,
        pair_coords=None):
    """ Count reconstructed photon pairs passing the pi0 angle and mass selection in each event.

        NOTE exactly_two_photons False is not properly implemented yet.
        Need to smartly work out the maximal set of passing pairs.

        Args:
            events (Any): Event data used to build and select reconstructed photon pairs.
            mass_cut (Any): Invariant-mass threshold or bounds for accepted pairs.
            opening_angle_deg (Any): Opening-angle threshold or bounds, in degrees.
            exactly_two_photons (Any): Whether to require exactly two candidates in the supplied mask.
            photon_mask (Any): Optional per-PFO mask used to form photon pairs.
            correction (Any): Optional function applied to each photon's reconstructed energy.
            correction_params (list): Parameters passed to ``correction``.
            pair_coords (Any): Optional explicit coordinates defining shower pairs.

        Returns:
            counts: ak.Array containing the number of selected reconstructed photon pairs for each event.
    """
    # if shower_pairs is None:
    #     if pair_coords is not None:
    #         shower_pairs = ntuple.ShowerPairs(events,
    #                                           pair_coords=pair_coords)
    #     elif photon_mask is not None:
    #         shower_pairs = ntuple.ShowerPairs(events,
    #                                           shower_pair_mask=photon_mask)
    #     else:
    #         shower_pairs = ntuple.ShowerPairs(
    #             events,
    #             shower_pair_mask=pfo.InitialPi0PhotonSelection(
    #                 events))
    
    # pass_angle = np.logical_and(
    #     shower_pairs.reco_angle > opening_angle_deg[0]*np.pi/180,
    #     shower_pairs.reco_angle < opening_angle_deg[1]*np.pi/180)
    # pass_mass = np.logical_and(
    #     shower_pairs.reco_mass > mass_cut[0],
    #     shower_pairs.reco_mass < mass_cut[1])

    # full_pass = np.logical_and(pass_angle, pass_mass)

    # if exactly_two_photons:
    #     number = np.logical_and(
    #         ak.sum(full_pass, axis=1),
    #         ak.num(shower_pairs.pairs['0']) == 1)
    #     counts = ak.values_astype(number, int)
    # else:
    #     counts = ak.sum(full_pass, axis=1)

    mask = pi0_selection(events = events,
        photon_candidates_mask = photon_mask,
        photon_candidates_coords = pair_coords,
        exact_photon_candidates = exactly_two_photons,
        n = 2,
        angle_cuts = opening_angle_deg,
        mass_cuts = mass_cut,
        correction = correction,
        correction_params = correction_params,
        verbose = False,
        return_table = False)
    return ak.sum(mask, axis=1)


def count_charged_pi_candidates(
        events,
        track_cut=None,
        n_hits_cut=None,
        dEdX_cuts=None,
        min_dEdX=None,
        max_dEdX=None,
        energy_cut=None,
        reco_pi_mask=None):
    """ Returns the number of pi+ particles identified from reconstructed
        PFOs for each event in `events`.

        Args:
            events (Any): Data Events in which to count pi+ occurances.
            track_cut (Any): float, optional Track score required for pi+ candidates. If not set, the default value in the `pfo` module is used.
            n_hits_cut (Any): int, optional Minimum number of hits required in pi+ candidates. If not set, the default value in the `pfo` module is used.
            dEdX_cuts (Any): Tuple (lower, upper), optional Lower and upper bounds for energy deposition rate required by pi+ candidates. If not set, the default values in the `pfo` module is used.
            min_dEdX (Any): float, optional Minimum energy deposition rate required by pi+ candidates. If not set, the default value in the `pfo` module is used. Overwritten by `dEdX_cuts` if specified.
            max_dEdX (Any): float, optional Maximum energy deposition rate required by pi+ candidates. If not set, the default value in the `pfo` module is used. Overwritten by `dEdX_cuts` if specified.
            energy_cut (Any): Optional minimum reconstructed shower energy for counted candidates.
            reco_pi_mask (Any): Optional precomputed reconstructed-pion mask; when supplied, other selection cuts are bypassed.

        Returns:
            counts: ak.Array Array containing the number of pi+ particles for each event.
    """
    if reco_pi_mask is None:
        selection_kwargs = {}
        if track_cut is not None:
            selection_kwargs.update({"track_cut":track_cut})
        if n_hits_cut is not None:
            selection_kwargs.update({"n_hits_cut":n_hits_cut})
        if dEdX_cuts is not None:
            min_dEdX = dEdX_cuts[0]
            max_dEdX = dEdX_cuts[1]
        if min_dEdX is not None:
            selection_kwargs.update({"min_dEdX":min_dEdX})
        if max_dEdX is not None:
            selection_kwargs.update({"max_dEdX":max_dEdX})
        reco_pi_mask = pfo.daughter_pi_plus_selection(
            events, **selection_kwargs)
        if energy_cut is not None:
            reco_pi_mask = np.logical_and(
                reco_pi_mask,
                events.recoParticles.shower_energy > energy_cut)
    return ak.sum(reco_pi_mask, axis=-1)


def _generate_selection(cut):
    """ Build a count predicate for an exact value, lower bound, range, or no cut.

        Args:
            cut (Any): ``None``, an exact count, or a one-/two-value tuple defining a lower bound or inclusive range.

        Returns:
            Any: Callable that returns a boolean selection for an input count.
    """
    if isinstance(cut, tuple):
        if len(cut) == 1:
            return lambda count: count >= cut[0]
        elif len(cut) == 2:
            return lambda count: np.logical_and(
                count >= min(cut), count <= max(cut))
        else:
            raise ValueError(f"Cut tuple {cut} must contain 1 or 2 values.")
    elif cut is None:
        return lambda count: True
    else:
        return lambda count: count == cut


def generate_truth_tags(
        events : Data,
        n_pi0,
        n_pi_charged,
        beam_daughters=True,
        only_diphoton=True,
        pi0_count=None,
        pi_charged_count=None):
    """ Generates a True/False tag for each event in `events` indicating
        whether they pass the truth level requirements of `n_pi0` and
        `n_pi_charged`.

        `n_pi0` and `n_pi_charged` may be integers, tuples, or None. If
        integer, only the specified number of occurances is selected. If a
        tuple of length 1, any events with occurances greater than or equal
        to the value in the tupled are selected. If a tuple of two values,
        he number of occurances must be equal to or between the values in
        the tuple. If None, no cut will be applied.

        Args:
            events (Data): Data Events to be tagged.
            n_pi0 (Any): None, int, or tuple Required number of pi0s that decay into two photons in an event for the event to pass the tag.
            n_pi_charged (Any): None, int, or tuple Required number of non-beam pi+ particles in an event for the event to pass the tag.
            beam_daughters (Any): Boolean, optional Whether to only accept a PFO if it is a daughter of the beam particle. Default is True.
            only_diphoton (Any): Boolean, optional Whether to only count a pi0 is it decay into two photons. Default is True.
            pi0_count (Any): ak.Array, None, optional Pre-created array of counts of pi0s in `events`. Computation time can be reduce if running multiple tags by creating this first using `EventSelection.count_diphoton_decays`, and passing the result here. If None, the counts will be calculated within this function. Default is None.
            pi_charged_count (Any): ak.Array, None, optional Pre-created array of counts of pi+s in `events`. Computation time can be reduce if running multiple tags by creating this first using `EventSelection.count_non_beam_charged_pi`, and passing the result here. If None, the counts will be calculated within this function. Default is None.

        Returns:
            tag: ak.Array Array matching the number of events in `events` containing a boolean of whether each event is selected by the tag.
    """
    pi0_cut: function = _generate_selection(n_pi0)
    pi_charged_cut: function = _generate_selection(n_pi_charged)
    if pi0_count is None:
        if only_diphoton:
            pi0_count = count_diphoton_decays(
                events,
                beam_daughters)
        else:
            if not beam_daughters:
                count_all_pi0s(events, beam_daughters)
            else:
                try:
                    pi0_count = events.true_particles.n_pi0
                except(AttributeError):
                    count_all_pi0s(events, beam_daughters)
    if pi_charged_count is None:
        if not beam_daughters:
            pi_charged_count = count_non_beam_charged_pi(
                events,
                beam_daughters)
        else:
            try:
                pi_charged_count = events.true_particles.n_pi_plus
            except(AttributeError):
                pi_charged_count = count_non_beam_charged_pi(
                    events,
                    beam_daughters)
    return np.logical_and(pi0_cut(pi0_count),
                          pi_charged_cut(pi_charged_count))


def create_regions(pi0_counts, pi_charged_counts):
    """ Classify events into analysis regions from their neutral- and charged-pion counts.
        ! Make use of SampleDefinitions class.
        Args:
            pi0_counts (Any): Per-event number of pi0 candidates.
            pi_charged_counts (Any): Per-event number of charged-pion candidates.

        Returns:
            Any: Mapping from region names to per-event boolean masks.
    """
    regions_dict = {
        "absorption": np.logical_and(pi0_counts==0, pi_charged_counts==0),
        "charge_exchange": np.logical_and(pi0_counts==1, pi_charged_counts==0),
        "pion_prod_0_pi0": np.logical_and(pi0_counts==0, pi_charged_counts>=1),
        "pion_prod_1_pi0": np.logical_and(pi0_counts==1, pi_charged_counts>=1),
        "pion_prod_>1_pi0": pi0_counts>=2
    }
    return regions_dict