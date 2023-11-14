#!/usr/bin/env python

# Imports
from python.analysis import Master, PFOSelection, BeamParticleSelection, SelectionTools, Tags
import time
import warnings
import awkward as ak
import numpy as np
import pandas as pd


#######################################################################
#######################################################################
##########                   EVENT TAGGING                   ##########
#######################################################################
#######################################################################


def GenerateTrueFinalStateTags(events : Master.Data = None) -> Tags.Tags:
    """ Generate truth tags for final state of the beam interaction.

    Args:
        events (Data, optional): events to look at. Defaults to None.

    Returns:
        Tags: tags
    """
    tags = Tags.Tags()
    tags["charge_exchange" ] = Tags.Tag("$1\pi^{0} + 0\pi^{+}$"              , "charge_exchange" , "#8EBA42", generate_truth_tags(events, 1, 0      , only_diphoton = False) if events is not None else None, 0)
    tags["absorption"      ] = Tags.Tag("$0\pi^{0} + 0\pi^{+}$"              , "absorption"      , "#777777", generate_truth_tags(events, 0, 0      , only_diphoton = False) if events is not None else None, 1)
    tags["pion_prod_1_pi0" ] = Tags.Tag("$1\pi^{0} + \geq 1\pi^{+}$"         , "pion_prod_1_pi0" , "#E24A33", generate_truth_tags(events, 1, (1,)   , only_diphoton = False) if events is not None else None, 2)
    tags["pion_prod_0_pi0" ] = Tags.Tag("$0\pi^{0} + \geq 1\pi^{+}$"         , "pion_prod_0_pi0" , "#988ED5", generate_truth_tags(events, 0, (1,)   , only_diphoton = False) if events is not None else None, 3)
    tags["pion_prod_>1_pi0"] = Tags.Tag("$> 1\pi^{0} + \geq 0\pi^{+}$"       , "pion_prod_>1_pi0", "#348ABD", generate_truth_tags(events, (2,), (0,), only_diphoton = False) if events is not None else None, 4)
    return tags


#######################################################################
#######################################################################
##########                  EVENT SELECTION                  ##########
#######################################################################
#######################################################################


def NPhotonCandidateSelection(events : Master.Data, photon_mask : ak.Array, cut : int, return_property : bool = False):
    n_photons = ak.sum(photon_mask, -1)
    return SelectionTools.CreateMask(cut, "==", n_photons, return_property)


def Pi0OpeningAngleSelection(events : Master.Data, photon_mask : ak.Array = None, photon_coords : ak.Array = None, cut = [10, 80], return_property : bool = False):
    if photon_mask is not None:
        shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photon_mask)
    elif photon_coords is not None:
        shower_pairs = Master.ShowerPairs(events, pair_coords = photon_coords)
    else:
        raise Exception("photon mask and photon_coords cannot both be None")

    angle = ak.fill_none(ak.pad_none(shower_pairs.reco_angle, 1, -1), -999, -1)
    cut = [c * np.pi / 180 for c in cut]
    return SelectionTools.CreateMask(cut, [">", "<"], angle, return_property)


def Pi0MassSelection(events : Master.Data, photon_mask : ak.Array = None, photon_coords : ak.Array = None, cut = [50, 250], correction = None, correction_params : list = None, return_property : bool = False):
    if photon_mask is not None:
        shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photon_mask)
    elif photon_coords is not None:
        shower_pairs = Master.ShowerPairs(events, pair_coords = photon_coords)
    else:
        raise Exception("photon mask and photon_coords cannot both be None")

    if correction is None:
        le = shower_pairs.reco_lead_energy
        se = shower_pairs.reco_sub_energy
    else:
        le = correction(shower_pairs.reco_lead_energy, **correction_params)
        se = correction(shower_pairs.reco_sub_energy, **correction_params)

    mass = shower_pairs.Mass(le, se, shower_pairs.reco_angle)
    mass = ak.fill_none(ak.pad_none(mass, 1, -1), -999, -1)
    print(cut)
    return SelectionTools.CreateMask(cut, [">", "<"], mass, return_property)


def Pi0Selection(
    events: Master.Data,
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
    return SelectionTools.CombineSelections(events, selections, 0, arguments, verbose, return_table)


def apply_function(
        name,
        history_list: list,
        func,
        events: Master.Data,
        *args,
        timed=True,
        ts=None,
        **kwargs):
    if timed and (ts is None):
        ts = time.time()
    result = func(events, *args, **kwargs)
    if timed:
        print(f"{name} done in {time.time() - ts}s")
    evt_remaining = ak.count(events.eventNum)
    history_list.append([name,
                         ak.count(events.trueParticlesBT.pdg),
                         ak.count(events.trueParticlesBT.pdg)/evt_remaining,
                         evt_remaining,
                         100*(evt_remaining/history_list[1][3])])
    return result


def apply_filter(events: Master.Data, filter, truth_filter=False):
    filter_true = []
    if truth_filter:
        filter_true = [filter]
    return events.Filter([filter], filter_true)


def filter_beam_slice(events: Master.Data):
    # TODO vectorise
    slice_mask = [[]] * ak.num(events.recoParticles.beam_number, axis=0)
    for i in range(ak.num(slice_mask, axis=0)):
        slices = events.recoParticles.slice_id[i]
        beam_slice = slices[
            events.recoParticles.number[i]
            == events.recoParticles.beam_number[i]]
        slice_mask[i] = list(slices == beam_slice)
    slice_mask = ak.Array(slice_mask)
    return events.Filter([slice_mask], [])


def load_and_cut_data(
        path,
        ntuple_type = "PDSPAnalyser",
        batch_size=-1, batch_start=-1,
        beam_selection=True,
        valid_momenta=True,
        n_hits_cut=80,
        cnn_cut=0.5,
        distance_bounds_cm=[3., 90.],
        max_impact_cm=20.,
        beam_slice_cut=False,
        truth_pi0_count=None,
        truth_pi_charged_count=None,
        print_summary=True,
        catch_warnings=True) -> Master.Data:
    """
    Loads the ntuple file from `path` and performs the initial cuts,
    returning the result as a `Data` instance.

    A single batch of data can be loaded an cut on by changing
    `batch_size` and `batch_start`.

    The cuts used can be partially controlled by the `two_photon`,
    `cnn_cut`, `valid_momenta`, `beam_slice_cut`, and `no_pi0`
    arguments.

    Parameters
    ----------
    path : str
        Path to the .root file to be loaded.
    ntuple_type : str {"PDSPAnalyser", "shower_merging"}
        What type of ntuple loaded. Default is PDSPAnalyser.
    batch_size : int, optional
        Sets how many events to read. Default is -1 (all).
    batch_start : int, optional
        Sets which event to start reading from. Default is -1
        (first event).
    cnn_cut : bool, optional
        Whether to perform a cut on PFOs requiring the CNN score to
        be > 0.36. Default is True.
    valid_momenta : bool, optional
        Whether to perform a cut removing PFOs with a momentum
        (-999., -999., -999.). Default is True.
    beam_slice_cut : bool, optional
        Whether to perform a cut requiring the PFOs to exist in the
        same slice as the beam particle. Default is True.

    Returns
    -------
    events : Data
        A Data instance containing the cut events.
    """
    # TODO add wther truth or MC to names (+ generally better names etc.)
    events = Master.Data(path,
                         nTuple_type=ntuple_type,
                         nEvents=batch_size,
                         start=batch_start)
    # Apply cuts:
    n = [["Event selection",
          "Number of PFOs",
          "Average PFOs per event",
          "Number of events",
          "Percentage of events remaining"]]
    n.append(["no selection",
              ak.count(events.trueParticlesBT.pdg),
              ak.count(events.trueParticlesBT.pdg)/ak.count(events.eventNum),
              ak.count(events.eventNum), 100])
    with warnings.catch_warnings():
        if catch_warnings:
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
        # Beam selection
        if beam_selection:
            ts = time.time()
            beam_selection_mask = BeamParticleSelection.CreateDefaultSelection(
                events, verbose=True, return_table=False)
            apply_function(
                "Beam selection (reco)", n,
                apply_filter, events, beam_selection_mask, truth_filter=True,
                ts=ts)
        # Only accept PFOs with a well reconstructed momentum/direction
        if valid_momenta:
            ts = time.time()
            valid_filter = PFOSelection.GoodShowerSelection(events)
            apply_function(
                "Valid momenta (reco)", n,
                apply_filter, events, valid_filter,
                ts=ts)
            del valid_filter
        # Only particles from the beam slice. Cheat this?
        if beam_slice_cut:
            apply_function(
                "Beam slice (reco)", n,
                filter_beam_slice, events)
        # Candidate photon PFO selection
        if ((not np.isclose(n_hits_cut, 0)) or (not np.isclose(cnn_cut, 0)) or
                (distance_bounds_cm is not None) or (max_impact_cm is not None)):
            ts = time.time()
            beam_selection_mask = PFOSelection.InitialPi0PhotonSelection(
                events,
                em_cut=cnn_cut,
                n_hits_cut=n_hits_cut,
                distance_bounds_cm=distance_bounds_cm,
                max_impact_cm=max_impact_cm,
                verbose=True,
                return_table=False)
            apply_function(
                "Beam selection (reco)", n,
                apply_filter, events, beam_selection_mask, truth_filter=True,
                ts=ts)
        # Require >= 2 PFOs. ETA ~90s:
        ts = time.time()
        at_least_2_pfos = Master.NPFPMask(events, -1)
        apply_function(
            "PFOs >= 2 (reco)", n,
            apply_filter, events, at_least_2_pfos,
            truth_filter=True, ts=ts)
        if (truth_pi0_count is not None) or (truth_pi_charged_count is not None):
            ts = time.time()
            truth_filter = generate_truth_tags(
                truth_pi0_count, truth_pi_charged_count)
            apply_function(
                f"{truth_pi0_count}pi0, {truth_pi_charged_count}pi+ (truth)", n,
                apply_filter, events, truth_filter,
                truth_filter=True, ts=ts)
    # Previously had plots of the distribution of PFOs per event after
    # each cut. Example code:
    # plt.figure(figsize=(8,6))
    # plt.hist(ak.count(events.trueParticlesBT.pdg, axis=-1), bins=100)
    # plt.xlabel("Number of PFOs in event")
    # plt.ylabel("Count")
    # plt.title("(beam, pi+, CNNScore) NPFPs >=2")
    # plt.savefig("/users/wx21978/projects/pion-phys/plots/photon_pairs/PFO_dist_nPFPs>2.png")
    # plt.close()
    if print_summary:
        print(pd.DataFrame(n[1:], columns=n[0]).to_string())
    return events


#######################################################################
#######################################################################
##########             EVENT TYPE TRUTH TAGGING              ##########
#######################################################################
#######################################################################


def count_diphoton_decays(events, beam_daughters=True):
    """
    Returns the number of truth pi0 particles which decay to yy in each
    event in `events`.

    pi0 -> yy

    Parameters
    ----------
    events : Data
        Events in which to count pi0 occurances.
    beam_daughters : boolean, optional
        Whether to only accept a pi0 if it is a daughter of the beam
        particle. Default is True.

    Returns
    -------
    counts : ak.Array
        Array containing the number of occurances of pi0 -> yy for each
        event.
    """
    def get_two_count_pi0s(pi0s):
        pions, counts = np.unique(pi0s, return_counts=True)
        return pions[counts == 2]

    if beam_daughters:
        beam_daughter_filter = events.trueParticles.mother == 1
    else:
        beam_daughter_filter = True
    beam_cadidate_pi0s = events.trueParticles.number[np.logical_and(
        beam_daughter_filter,
        events.trueParticles.pdg == 111)]
    try:
        pi0_daughters = events.trueParticles.mother_pdg == 111
    except:
        pi0_daughters = PFOSelection.get_mother_pdgs(events) == 111
    pi0_photon_mothers = events.trueParticles.mother[np.logical_and(
        events.trueParticles.pdg == 22,
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
    """
    Returns the number of truth pi0 particles which decay to yy in each
    event in `events`.

    pi0 -> yy

    Parameters
    ----------
    events : Data
        Events in which to count pi0 occurances.
    beam_daughters : boolean, optional
        Whether to only accept a pi0 if it is a daughter of the beam
        particle. Default is True.

    Returns
    -------
    counts : ak.Array
        Array containing the number of occurances of pi0 -> yy for each
        event.
    """
    if beam_daughters:
        beam_daughter_filter = events.trueParticles.mother == 1
    else:
        beam_daughter_filter = True
    beam_cadidate_pi0s = events.trueParticles.number[np.logical_and(
        beam_daughter_filter,
        events.trueParticles.pdg == 111)]
    return ak.sum(beam_cadidate_pi0s, axis=-1)


def count_non_beam_charged_pi(events, beam_daughters=True):
    """
    Returns the number of truth pi+ particles event in `events`.

    Parameters
    ----------
    events : Data
        Events in which to count pi+ occurances.
    beam_daughters : boolean, optional
        Whether to only accept a pi+ if it is a daughter of the beam
        particle. Default is True.

    Returns
    -------
    counts : ak.Array
        Array containing the number of pi+ particles for each event.
    """
    if beam_daughters:
        daughter_truth = events.trueParticles.mother == 1
    else:
        daughter_truth = ak.ones_like(events.trueParticles.mother, dtype=bool)
    non_beam_pi_mask = np.logical_and(
        daughter_truth,
        np.logical_and(events.trueParticles.pdg == 211,
                       events.trueParticles.number != 1))
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
    """
    Returns the number of truth pi0 particles which decay to yy in each
    event in `events`.

    pi0 -> yy

    NOTE exactly_two_photons False is not properly implemented yet.
    Need to smartly work out the maximal set of passing pairs.

    Parameters
    ----------
    events : Data
        Events in which to count pi0 occurances.
    beam_daughters : boolean, optional
        Whether to only accept a pi0 if it is a daughter of the beam
        particle. Default is True.

    Returns
    -------
    counts : ak.Array
        Array containing the number of occurances of pi0 -> yy for each
        event.
    """
    # if shower_pairs is None:
    #     if pair_coords is not None:
    #         shower_pairs = Master.ShowerPairs(events,
    #                                           pair_coords=pair_coords)
    #     elif photon_mask is not None:
    #         shower_pairs = Master.ShowerPairs(events,
    #                                           shower_pair_mask=photon_mask)
    #     else:
    #         shower_pairs = Master.ShowerPairs(
    #             events,
    #             shower_pair_mask=PFOSelection.InitialPi0PhotonSelection(
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

    mask = Pi0Selection(events = events,
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
    """
    Returns the number of pi+ particles identified from reconstructed
    PFOs for each event in `events`.

    Parameters
    ----------
    events : Data
        Events in which to count pi+ occurances.
    track_cut : float, optional
        Track score required for pi+ candidates. If not set, the
        default value in the `PFOSelection` module is used.
    n_hits_cut : int, optional
        Minimum number of hits required in pi+ candidates. If not set,
        the default value in the `PFOSelection` module is used.
    dEdX_cuts : tuple (lower, upper), optional
        Lower and upper bounds for energy deposition rate required by
        pi+ candidates. If not set, the default values in the
        `PFOSelection` module is used.
    min_dEdX : float, optional
        Minimum energy deposition rate required by pi+ candidates. If
        not set, the default value in the `PFOSelection` module is 
        used. Overwritten by `dEdX_cuts` if specified.
    max_dEdX : float, optional
        Maximum energy deposition rate required by pi+ candidates. If
        not set, the default value in the `PFOSelection` module is 
        used. Overwritten by `dEdX_cuts` if specified.

    Returns
    -------
    counts : ak.Array
        Array containing the number of pi+ particles for each event.
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
        reco_pi_mask = PFOSelection.DaughterPiPlusSelection(
            events, **selection_kwargs)
        if energy_cut is not None:
            reco_pi_mask = np.logical_and(
                reco_pi_mask,
                events.recoParticles.energy > energy_cut)
    return ak.sum(reco_pi_mask, axis=-1)


def _generate_selection(cut):
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
        events : Master.Data,
        n_pi0,
        n_pi_charged,
        beam_daughters=True,
        only_diphoton=True,
        pi0_count=None,
        pi_charged_count=None):
    """
    Generates a True/False tag for each event in `events` indicating
    whether they pass the truth level requirements of `n_pi0` and
    `n_pi_charged`.

    `n_pi0` and `n_pi_charged` may be integers, tuples, or None. If
    integer, only the specified number of occurances is selected. If a
    tuple of length 1, any events with occurances greater than or equal
    to the value in the tupled are selected. If a tuple of two values, 
    he number of occurances must be equal to or between the values in
    the tuple. If None, no cut will be applied.

    Parameters
    ----------
    events : Data
        Events to be tagged.
    n_pi0 : None, int, or tuple
        Required number of pi0s that decay into two photons in an event
        for the event to pass the tag.
    n_pi_charged : None, int, or tuple
        Required number of non-beam pi+ particles in an event for the
        event to pass the tag.
    beam_daughters : boolean, optional
        Whether to only accept a PFO if it is a daughter of the beam
        particle. Default is True.
    only_diphoton : boolean, optional
        Whether to only count a pi0 is it decay into two photons.
        Default is True.
    pi0_count : ak.Array, None, optional
        Pre-created array of counts of pi0s in `events`. Computation
        time can be reduce if running multiple tags by creating this
        first using `EventSelection.count_diphoton_decays`, and passing
        the result here. If None, the counts will be calculated within
        this function. Default is None.
    pi_charged_count : ak.Array, None, optional
        Pre-created array of counts of pi+s in `events`. Computation
        time can be reduce if running multiple tags by creating this
        first using `EventSelection.count_non_beam_charged_pi`, and
        passing the result here. If None, the counts will be calculated
        within this function. Default is None.

    Returns
    -------
    tag : ak.Array
        Array matching the number of events in `events` containing a
        boolean of whether each event is selected by the tag.
    """
    pi0_cut: function = _generate_selection(n_pi0)
    pi_charged_cut: function = _generate_selection(n_pi_charged)
    if pi0_count is None:
        if only_diphoton:
            pi0_count = count_diphoton_decays(
                events,
                beam_daughters=beam_daughters)
        else:
            if not beam_daughters:
                count_all_pi0s(events, beam_daughters=beam_daughters)
            else:
                try:
                    pi0_count = events.trueParticles.nPi0
                except(AttributeError):
                    count_all_pi0s(events, beam_daughters=beam_daughters)
    if pi_charged_count is None:
        if not beam_daughters:
            pi_charged_count = count_non_beam_charged_pi(
                events,
                beam_daughters=beam_daughters)
        else:
            try:
                pi_charged_count = events.trueParticles.nPiPlus
            except(AttributeError):
                pi_charged_count = count_non_beam_charged_pi(
                    events,
                    beam_daughters=beam_daughters)
    return np.logical_and(pi0_cut(pi0_count),
                          pi_charged_cut(pi_charged_count))


def generate_reco_tags(
    events : Master.Data,
    n_pi0,
    n_pi_charged,
    exactly_two_photons=False,
    pi0_count=None,
    pi_charged_count=None):
    """
    Generates a True/False tag for each event in `events` indicating
    whether they pass the truth level requirements of `n_pi0` and
    `n_pi_charged`.

    `n_pi0` and `n_pi_charged` may be integers, tuples, or None. If
    integer, only the specified number of occurances is selected. If a
    tuple of length 1, any events with occurances greater than or equal
    to the value in the tupled are selected. If a tuple of two values, 
    he number of occurances must be equal to or between the values in
    the tuple. If None, no cut will be applied.

    Parameters
    ----------
    events : Data
        Events to be tagged.
    n_pi0 : None, int, or tuple
        Required number of pi0s that decay into two photons in an event
        for the event to pass the tag.
    n_pi_charged : None, int, or tuple
        Required number of non-beam pi+ particles in an event for the
        event to pass the tag.
    exactly_two_photons : boolean, optional
        Whether or not to only count pi0 candiadtes if there are
        exactly 2 shower candidates in a event. This limits the count
        to 1 per event. Default is False.
    pi0_count : ak.Array, None, optional
        Pre-created array of counts of pi0s in `events`. Computation
        time can be reduce if running multiple tags by creating this
        first using `EventSelection.count_diphoton_decays`, and passing
        the result here. If None, the counts will be calculated within
        this function. Default is None.
    pi_charged_count : ak.Array, None, optional
        Pre-created array of counts of pi+s in `events`. Computation
        time can be reduce if running multiple tags by creating this
        first using `EventSelection.count_non_beam_charged_pi`, and
        passing the result here. If None, the counts will be calculated
        within this function. Default is None.

    Returns
    -------
    tag : ak.Array
        Array matching the number of events in `events` containing a
        boolean of whether each event is selected by the tag.
    """
    pi0_cut: function = _generate_selection(n_pi0)
    pi_charged_cut: function = _generate_selection(n_pi_charged)
    if pi0_count is None:
        pi0_count = count_pi0_candidates(
            events, exactly_two_photons=exactly_two_photons)
    if pi_charged_count is None:
        pi_charged_count = count_charged_pi_candidates(events)
    return np.logical_and(pi0_cut(pi0_count),
                          pi_charged_cut(pi_charged_count))


def create_regions(pi0_counts, pi_charged_counts):
    regions_dict = {
        "absorption": np.logical_and(pi0_counts==0, pi_charged_counts==0),
        "charge_exchange": np.logical_and(pi0_counts==1, pi_charged_counts==0),
        "pion_prod_0_pi0": np.logical_and(pi0_counts==0, pi_charged_counts>=1),
        "pion_prod_1_pi0": np.logical_and(pi0_counts==1, pi_charged_counts>=1),
        "pion_prod_>1_pi0": pi0_counts>=2
    }
    return regions_dict


def create_regions_new(pi0_counts, pi_charged_counts):
    regions_dict = {
        "absorption": np.logical_and(pi0_counts==0, pi_charged_counts==0),
        "charge_exchange": np.logical_and(pi0_counts==1, pi_charged_counts==0),
        "single_pion_production": np.logical_and(pi0_counts==0, pi_charged_counts==1),
        "pion_production": ((pi0_counts >= 0) & (pi_charged_counts > 1)) | ((pi0_counts > 1) & (pi_charged_counts >= 0)) | ((pi0_counts == 1) & (pi_charged_counts == 1)),
    }
    return regions_dict


#######################################################################
#######################################################################
##########                    DEPRECATED                     ##########
#######################################################################
#######################################################################


def candidate_photon_pfo_selection(
        events: Master.Data,
        cut_record=[[], [1, 1, 1, 1, 1]],
        n_hits_cut=80,
        cnn_cut=0.5,
        distance_bounds_cm=(3, 90),
        max_impact_cm=20.):
    """
    DEPRECATED
    Use PFOSelection.InitialPi0PhotonSelection to generate a filter and
    then apply the generated filter.

    Finds a set of PFOs passing cuts on number of hits, cnn score,
    distance from beam vertex and impact parameter with beam vertex.

    Parameters
    ----------
    events : Master.Data
        Events containing the PFOs to cut on.
    cut record : list, optional
        Record of the number of events cut. Default list avoids errors
        if not used.
    n_hits_cut : int, optional
        Minimum number of hits required by a PFO to pass the cut.
        Deafult is 80.
    cnn_cut : float, optional
        Minimum CNN score required by a PFO to pass the cut. Default is
        0.5.
    distance_bounds_cm : tuple, optional
        (lower bound, upper bound) for allowed distances between the
        PFO start point the beam vertex. Default is (3., 90.).
    max_impact_cm : float, optional
        Maximum allowed impact parameter with teh beam particle for a
        PFO to pass the cut. Default is 20.
    """
    warnings.warn("deprecated", DeprecationWarning)
    if max_impact_cm is not None:
        ts = time.time()
        impacts = PFOSelection.find_beam_impact_parameters(events)
        impact_mask = impacts < max_impact_cm
        apply_function(
            "Beam impact (reco)", cut_record,
            apply_filter, events, impact_mask,
            ts=ts)
    if distance_bounds_cm is not None:
        ts = time.time()
        distances = PFOSelection.find_beam_separations(events)
        distance_mask = np.logical_and(distances > distance_bounds_cm[0],
                                       distances < distance_bounds_cm[1])
        apply_function(
            "Distance from beam (reco)", cut_record,
            apply_filter, events, distance_mask,
            ts=ts)
    if not np.isclose(n_hits_cut, 0):
        ts = time.time()
        apply_function(
            f"n_hits > {n_hits_cut} (reco)", cut_record,
            apply_filter, events, events.recoParticles.n_hits > n_hits_cut,
            ts=ts)
    if not np.isclose(cnn_cut, 0):
        ts = time.time()
        apply_function(
            f"EMScore > {cnn_cut} (reco)", cut_record,
            apply_filter, events, events.recoParticles.em_score > cnn_cut,
            ts=ts)
    return events


def get_0_or_1_pi0(events: Master.Data):
    warnings.warn("deprecated", DeprecationWarning)
    # Copying the BeamMCFitler here to allow multiple valid numbers
    empty = ak.num(events.trueParticles.number) > 0
    events.Filter([empty], [empty])

    # * only look at events with 0 or 1 primary pi0s
    pi0 = events.trueParticles.PrimaryPi0Mask
    single_primary_pi0 = np.logical_or(
        ak.num(pi0[pi0]) == 0, ak.num(pi0[pi0]) == 1)
    events.Filter([single_primary_pi0], [single_primary_pi0])

    # * remove true particles which aren't primaries
    primary_pi0 = events.trueParticles.PrimaryPi0Mask
    # this is fine so long as we only care about pi0->gamma gamma
    primary_daughter = events.trueParticles.truePhotonMask
    primaries = np.logical_or(primary_pi0, primary_daughter)
    return events.Filter([], [primaries])


def old_event_selection_truth(events, pion_count, two_photon, cut_record):
    warnings.warn("deprecated", DeprecationWarning)
    if pion_count == 'one':
        apply_function(
            "single pi0", cut_record,
            Master.BeamMCFilter, events, returnCopy=False)
    elif pion_count == 'zero':
        apply_function(
            "no pi0s", cut_record,
            Master.BeamMCFilter, events, n_pi0=0, returnCopy=False)
    elif pion_count == 'both':
        apply_function(
            "0 or 1 pi0s", cut_record,
            get_0_or_1_pi0, events)
    if two_photon:
        apply_function(
            "diphoton decays", cut_record,
            apply_filter,
            events, Master.Pi0TwoBodyDecayMask(events), truth_filter=True)
    # Require a beam particle to exist. ETA ~15s:
    apply_function(
        "beam", cut_record,
        lambda e: e.ApplyBeamFilter(), events)
    # Require pi+ beam. ETA ~10s
    # Check with reco pi+? (Currently cheating the pi+ selection)
    ts = time.time()
    beam_pdg_codes = events.trueParticlesBT.pdg[
        events.recoParticles.beam_number == events.recoParticles.number]
    pi_beam_filter = ak.all(beam_pdg_codes == 211, -1)
    apply_function(
        "pi+ beam", cut_record,
        apply_filter, events, pi_beam_filter,
        truth_filter=True, ts=ts)
    del beam_pdg_codes
    del pi_beam_filter
    return events
