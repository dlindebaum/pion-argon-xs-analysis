#!/usr/bin/env python

# Imports
from python.analysis import Master, pfoProperties
import time
import awkward as ak
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '/users/wx21978/projects/pion-phys/pi0-analysis/analysis/')

# Functions

# This is a memory management tool, currently not really necessary


def del_prop(obj, property_name):
    """
    Deletes a properties from the supplied `RecoPaticleData` type
    object.

    Requires the `obj` to have a property
    ``_RecoPaticleData__{property_name}``.

    Parameters
    ----------
    obj : RecoPaticleData
        Object from which to remove the property.
    property_name : str
        Property to be deleted (should match the name of the property).
    """
    del(obj.__dict__["_RecoPaticleData__" + property_name])
    return


def apply_function(
        name,
        history_list: list,
        func,
        events: Master.Data,
        *args,
        timed=True,
        **kwargs):
    if timed:
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


def get_0_or_1_pi0(events: Master.Data):
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


def filter_beam_slice(events: Master.Data):
    slice_mask = [[]] * ak.num(events.recoParticles.beam_number, axis=0)
    for i in range(ak.num(slice_mask, axis=0)):
        slices = events.recoParticles.sliceID[i]
        beam_slice = slices[
            events.recoParticles.number[i]
            == events.recoParticles.beam_number[i]]
        slice_mask[i] = list(slices == beam_slice)
    slice_mask = ak.Array(slice_mask)
    return events.Filter([slice_mask], [])


def filter_impact_and_distance(
        events: Master.Data,
        distance_bounds_cm=None,
        max_impact_cm=None):
    if (distance_bounds_cm is None) and (max_impact_cm is None):
        return None
    mask = [[]] * ak.num(events.recoParticles.beam_number, axis=0)
    for i in range(ak.num(mask, axis=0)):
        starts = events.recoParticles.startPos[i]
        beam_vertex = events.recoParticles.beamVertex[i]
        distances = pfoProperties.get_separation(starts, beam_vertex)
        if distance_bounds_cm is not None:
            distance_mask = np.logical_and(distances > distance_bounds_cm[0],
                                           distances < distance_bounds_cm[1])
        if max_impact_cm is not None:
            directions = events.recoParticles.direction[i]
            impact_mask = pfoProperties.get_impact_parameter(
                directions,
                starts,
                beam_vertex) < max_impact_cm
        mask[i] = np.logical_and(distance_mask, impact_mask)
    mask = ak.Array(mask)
    return events.Filter([mask], [])


def load_and_cut_data(
        path, batch_size=-1,
        batch_start=-1,
        pion_count='one',
        two_photon=True,
        cnn_cut=0.36,
        valid_momenta=True,
        beam_slice_cut=True,
        n_hits_cut=0,
        distance_bounds_cm=None,
        max_impact_cm=None,
        print_summary=True):
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
    batch_size : int, optional
        Sets how many events to read. Default is -1 (all).
    batch_start : int, optional
        Sets which event to start reading from. Default is -1
        (first event).
    pion_count : str {zero, one, both, all}, optional
        Sets the number of pi0s created by the beam that must be
        present in the final events. `'zero'` selects only events
        with no pi0s, `'one'` selects only 1 pi0, `'both'` selects
        events with 0 or 1 pi0s, and `'all'` removes the cut.
        Default is `'one'`.
    two_photon : bool, optional
        If true, only accepts events with 1 pi0 which decays to two
        photons. Default is True.
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
                         includeBackTrackedMC=True,
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
    if pion_count == 'one':
        apply_function(
            "single pi0", n,
            Master.BeamMCFilter, events, returnCopy=False)
    elif pion_count == 'zero':
        apply_function(
            "no pi0s", n,
            Master.BeamMCFilter, events, n_pi0=0, returnCopy=False)
    elif pion_count == 'both':
        apply_function(
            "0 or 1 pi0s", n,
            get_0_or_1_pi0, events)
    if two_photon:
        apply_function(
            "diphoton decays", n,
            apply_filter,
            events, Master.Pi0TwoBodyDecayMask(events), truth_filter=True)
    # Require a beam particle to exist. ETA ~15s:
    apply_function(
        "beam", n,
        lambda e: e.ApplyBeamFilter(), events)
    # Require pi+ beam. ETA ~10s
    # Check with reco pi+? (Currently cheating the pi+ selection)
    beam_pdg_codes = events.trueParticlesBT.pdg[
        events.recoParticles.beam_number == events.recoParticles.number]
    pi_beam_filter = ak.all(beam_pdg_codes == 211, -1)
    apply_function(
        "pi+ beam", n,
        apply_filter, events, pi_beam_filter, truth_filter=True)
    del beam_pdg_codes
    del pi_beam_filter
    # Only look at PFOs with > n_hits hits. ETA ~30s:
    if not np.isclose(n_hits_cut, 0):
        apply_function(
            "nHits", n,
            apply_filter, events, events.recoParticles.nHits > n_hits_cut)
    # Cheat this?
    if beam_slice_cut:
        apply_function(
            "Beam slice", n,
            filter_beam_slice, events)
    if not np.isclose(cnn_cut, 0):
        apply_function(
            f"CNNScore > {cnn_cut}", n,
            apply_filter, events, events.recoParticles.cnnScore > cnn_cut)
    if valid_momenta:
        reco_momenta = events.recoParticles.momentum
        mom_filter = np.logical_and(
            np.logical_and(reco_momenta.x != -999., reco_momenta.y != -999.),
            reco_momenta.z != -999.)
        apply_function(
            "Valid momenta", n,
            apply_filter, events, mom_filter)
        del reco_momenta
        del mom_filter
    if (distance_bounds_cm is not None) or (max_impact_cm is not None):
        apply_function(
            "distance/impact", n,
            filter_impact_and_distance, events,
            distance_bounds_cm, max_impact_cm
        )
    # Require >= 2 PFOs. ETA ~90s:
    apply_function(
        "PFOs >= 2", n,
        apply_filter, events, Master.NPFPMask(events, -1), truth_filter=True)
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
