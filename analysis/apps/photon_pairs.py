#!/usr/bin/env python

# Imports
import sys
sys.path.insert(1, '/users/wx21978/projects/pion-phys/pi0-analysis/analysis/')
import os
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
from python.analysis import Master, vector, PairPlots
import time

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
        events : Master.Data,
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
                         100*(evt_remaining/history_list[1][3]) ])
    return result


def apply_filter(events : Master.Data, filter, truth_filter = False):
    filter_true = []
    if truth_filter:
        filter_true = [filter]
    return events.Filter([filter], filter_true)


def get_0_or_1_pi0(events : Master.Data):
    # Copying the BeamMCFitler here to allow multiple valid numbers
    empty = ak.num(events.trueParticles.number) > 0
    events.Filter([empty], [empty])

    #* only look at events with 0 or 1 primary pi0s
    pi0 = events.trueParticles.PrimaryPi0Mask
    single_primary_pi0 = np.logical_or(
        ak.num(pi0[pi0]) == 0, ak.num(pi0[pi0]) == 1)
    events.Filter([single_primary_pi0], [single_primary_pi0])

    #* remove true particles which aren't primaries
    primary_pi0 = events.trueParticles.PrimaryPi0Mask
    # this is fine so long as we only care about pi0->gamma gamma
    primary_daughter = events.trueParticles.truePhotonMask
    primaries = np.logical_or(primary_pi0, primary_daughter)
    return events.Filter([], [primaries])


def filter_beam_slice(events : Master.Data):
    slice_mask = [[]] * ak.num(events.recoParticles.beam_number, axis=0)
    for i in range(ak.num(slice_mask, axis=0)):
        slices = events.recoParticles.sliceID[i]
        beam_slice = slices[
            events.recoParticles.number[i] \
            == events.recoParticles.beam_number[i]]
        slice_mask[i] = list(slices == beam_slice)
    slice_mask = ak.Array(slice_mask)
    return events.Filter([slice_mask], [])


def filter_impact_and_distance(
        events : Master.Data,
        distance_bounds_cm=None,
        max_impact_cm=None):
    if (distance_bounds_cm is None) and (max_impact_cm is None):
        return None
    mask = [[]] * ak.num(events.recoParticles.beam_number, axis=0)
    for i in range(ak.num(mask, axis=0)):
        starts = events.recoParticles.startPos[i]
        beam_vertex = events.recoParticles.beamVertex[i]
        distances = get_separation(starts, beam_vertex)
        if distance_bounds_cm is not None:
            distance_mask = np.logical_and(distances > distance_bounds_cm[0],
                                           distances < distance_bounds_cm[1])
        if max_impact_cm is not None:
            directions = events.recoParticles.direction[i]
            impact_mask = get_impact_parameter(directions,
                                               starts,
                                               beam_vertex) < max_impact_cm
        mask[i] = np.logical_and(distance_mask, impact_mask)
    mask = ak.Array(mask)
    return events.Filter([mask], [])


def load_and_cut_data(
        path, batch_size = -1,
        batch_start = -1,
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
        reco_momenta=events.recoParticles.momentum
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
        print(pd.DataFrame(n[1:], columns = n[0]).to_string())
    return events


#######################################################################
#######################################################################
##########                PLOTTING FUCNTIONS                 ##########
#######################################################################
#######################################################################


def simple_sig_bkg_hist(
        prop_name, units, property, sig_mask,
        path="/users/wx21978/projects/pion-phys/plots/photon_pairs/", unique_save_id = "",
        y_scaling = 'log', bin_size = None, bins = 100, range = None, **kwargs
    ):
    """
    Produces a set of plots of the supplied `property` with configurable
    scalings, bins and ranges.

    Data to be plotted comes from `property`, and a `sig_mask` to indicate
    background vs. signal. Signal and background are plotted on the same axis
    for comparison.

    `prop_name` and `units` are required to correctly label and title the plot.

    `y_scaling`, `bin_size`, `bins`, and `range` can all be supplied as lists of
    values (that could be passed to a pyplot function). If a list is passed,
    the options supplied will be iterated through, and a plot will be produced
    for each value given. `bin_size`, `bins`, and `range` must all either have
    same length, or not be a list.

    Parameters
    ----------
    prop_name : str
        Name of the property plotted for title and saved file.
    units : str
        Unit the property is measured.
    property : ak.Array or np.ndarray
        Array containing the values to be plotted.
    sig_mask : ak.Array or np.ndarray
        Mask which indicates which of the values in `property`
        correspond to signal data.
    path : str, optional
        Directory in which to save the final plot(s). Default is
        '/users/wx21978/projects/pion-phys/plots/photon_pairs/'.
    unique_save_id : str, optional
        Extra string to add into the save name to avoid
        overwriting. Default is ''.
    y_scaling : str or list, optional
        Type of scalings to use on the y-axis. Should be
        'linear' or 'log'. Default is 'log'
    bin_size : str, list, or None, optional
        The size of the bins. If None, this will be automatically
        calculated, but can be manually specified, i.e. if using
        custom/variable bin widths. Default is None.
    bins : int, np.ndarray, or list, optional
        Number of bins if ``int``, or bin edges if ``ndarray``.
        Default is 100.
    range : float, None, tuple, or list, optional
        Range over which to produce the plot. Default is None.
    
    Other Parameters
    ----------------
    **kwargs
        Any additional keyword arguments to be passed to the ``plt.hist``
        calls. The same arguments will be passed to all plots.
    """
    if path[-1] != '/':
        path += '/'
    if not isinstance(y_scaling, list):
        y_scaling = [y_scaling]
    if not isinstance(range, list):
        range = [range]
    if not ( (isinstance(bins, list) or isinstance(bins, np.ndarray)) and len(bins) == len(range) ):
        # Strictly, this could be a problem on edge-case that we want to use one fewer bins than the
        # number of ranges  we investigate, but this is sufficiently likely that we can ignore that
        bins = [bins for _ in range]
    for y_scale in y_scaling:
        for i, r in enumerate(range):
            if r is None:
                path_end = "full"
                hist_range = np.max(property)
            elif isinstance(r, tuple):
                path_end = f"{r[0]}-{r[1]}"
                hist_range = r[1] - r[0]
                kwargs.update({"range":r})
            else:
                path_end = f"<{r}"
                hist_range = r
                kwargs.update({"range":(0,r)})
            
            if not y_scale == "linear":
                path_end += "_" + y_scale

            if bin_size is None:
                if isinstance(bins[i], int):
                    bin_size = f"{hist_range/bins[i]:.2g}" + units
                else:
                    bin_size = f"{hist_range/len(bins[i]):.2g}" + units
            plt.figure(figsize=(12,9))
            plt.hist(ak.ravel(property[sig_mask]),                 bins=100, label="signal", **kwargs)
            plt.hist(ak.ravel(property[np.logical_not(sig_mask)]), bins=100, label="background", **kwargs)
            plt.legend()
            plt.yscale(y_scale)
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Count/" + bin_size)
            plt.savefig(path + prop_name.replace(" ", "_") + unique_save_id + "_hist_" + path_end + ".png")
            plt.close()
    return
    

def plot_pair_hists(
        prop_name, units, property, sig_count,
        path="/users/wx21978/projects/pion-phys/plots/photon_pairs/", unique_save_id = "",
        inc_stacked=False, inc_norm=True, inc_log=True,
        bin_size = None, bins = 100,
        range=None, weights=None, **kwargs
    ):
    """
    Produces a set of plots of the supplied `property` of a pair of PFOs with
    configurable types of plot, scalings, bins, ranges, weights.

    Data to be plotted comes from `property`, and a `sig_count` indicates
    how many signal type PFOs went into the given pair. 0, 1, and 2 signals
    are plotted on the same axis for comparison.

    `prop_name` and `units` are required to correctly label and title the plots.

    Whether to produce stacked, normalised, and log scale plots is controlled
    by `inc_stacked`, `inc_norm`, and `inc_log` respectively.

    `bin_size`, `bins`, `range`, and `weights` can all be supplied as lists of
    values (that could be passed to a pyplot function). If a list is passed,
    the options supplied will be iterated through, and a plot will be produced
    for each value given. `bin_size`, `bins`, and `range` must all either have
    same length, or not be a list.

    Parameters
    ----------
    prop_name : str
        Name of the property plotted for title and saved file.
    units : str
        Unit the property is measured.
    property : ak.Array or np.ndarray
        Array containing the pair values to be plotted.
    sig_count : ak.Array or np.ndarray
        And array with the same shape as `property` to indicate how
        many signal PFOs exist in the pair.
    path : str, optional
        Directory in which to save the final plots. Default is
        '/users/wx21978/projects/pion-phys/plots/photon_pairs/'.
    unique_save_id : str, optional
        Extra string to add into the save name to avoid
        overwriting. Default is ''.
    inc_stacked : bool, optional
        Whether to include a stacked histogram plot. Default is False.
    inc_norm : bool, optional
        Whether to include a normalised histogram. Default is True.
    inc_log : bool, optional
        Whether to include logarithmic scale histograms. Will
        also produce a logarithmic normalised plot if `inc_norm`
        is True. Default is True.
    bin_size : str, list, or None, optional
        The size of the bins. If None, this will be automatically
        calculated, but can be manually specified, i.e. if using
        custom/variable bin widths. Default is None.
    bins : int, np.ndarray, or list, optional
        Number of bins if ``int``, or bin edges if ``ndarray``.
        Default is 100.
    range : float, None, tuple, or list, optional
        Range over which to produce the plot. Default is None.
    weights : ak.Array, np.ndarray, list, or None, optional
        Weights to used for each value of the `property`. Default
        is None.
    
    Other Parameters
    ----------------
    **kwargs
        Any additional keyword arguments to be passed to the ``plt.hist``
        calls. The same arguments will be passed to all plots.
    """
    
    if path[-1] != '/':
        path += '/'

    if not isinstance(range, list):
        range = [range]
    if not ( (isinstance(bins, list) or isinstance(bins, np.ndarray)) and len(bins) == len(range) ):
        # Strictly, this could be a problem on edge-case that we want to use one fewer bins than the
        # number of ranges  we investigate, but this is sufficiently likely that we can ignore that
        bins = [bins] * len(range)

    # There is definitely a better way to do this...
    if not isinstance(weights, list):
        if inc_stacked: # Need to keep track of the raw weights if producing a stacked plot
            raw_weights = [weights] * len(range)
        if weights is None:
            weights = [{2:None, 1:None, 0:None}] * len(range)
        else:
            weights = [{2:ak.ravel(weights[sig_count == 2]), 1:ak.ravel(weights[sig_count == 1]), 0:ak.ravel(weights[sig_count == 0])}] * len(range)
    else:
        if inc_stacked: # Need to keep track of the raw weights if producing a stacked plot
            raw_weights = weights
        for i in np.arange(len(weights)):
            if weights[i] is None:
                weights[i] = {2:None, 1:None, 0:None}
            else:
                weights[i] = {2:ak.ravel(weights[i][sig_count == 2]), 1:ak.ravel(weights[i][sig_count == 1]), 0:ak.ravel(weights[i][sig_count == 0])}

    sig_0 = ak.ravel(property[sig_count == 0])
    sig_1 = ak.ravel(property[sig_count == 1])
    sig_2 = ak.ravel(property[sig_count == 2])

    for i, r in enumerate(range):
        if r is None:
            path_end = unique_save_id + "_hist_full"
            hist_range = np.max(property)
        elif isinstance(r, tuple):
            path_end = unique_save_id + f"_hist_{r[0]}-{r[1]}"
            hist_range = r[1] - r[0]
            kwargs.update({"range":r})
        else:
            path_end = unique_save_id + f"_hist<{r}"
            hist_range = r
            kwargs.update({"range":(0,r)})
        
        if bin_size is None:
            if isinstance(bins[i], int):
                bin_size = f"{hist_range/bins[i]:.2g}" + units
            else:
                bin_size = f"{hist_range/len(bins[i]):.2g}" + units

        if inc_stacked:
            # Whoever wrote this disgusting way to deal with stacked weights ought to be shot...
            if raw_weights[i] is not None:
                all_weights = ak.ravel(raw_weights[i])
                no_bkg_weights = ak.ravel(raw_weights[i][sig_count != 0])
            else: 
                all_weights = None
                no_bkg_weights = None
            
            plt.figure(figsize=(12,9))
            plt.hist(ak.ravel(property),                 label="0 signal", bins = bins[i], weights=all_weights,    color="C2", **kwargs)
            plt.hist(ak.ravel(property[sig_count != 0]), label="1 signal", bins = bins[i], weights=no_bkg_weights, color="C1", **kwargs)
            plt.hist(sig_2,                              label="2 signal", bins = bins[i], weights=weights[i][2],  color="C0", **kwargs)
            plt.legend()
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Count/" + bin_size)
            plt.savefig(path + "paired_" + prop_name.replace(" ", "_") + path_end + "_stacked.png")
            plt.close()

        plt.figure(figsize=(12,9))
        plt.hist(sig_2, histtype='step', label="2 signal", bins = bins[i], weights=weights[i][2], **kwargs)
        plt.hist(sig_1, histtype='step', label="1 signal", bins = bins[i], weights=weights[i][1], **kwargs)
        plt.hist(sig_0, histtype='step', label="0 signal", bins = bins[i], weights=weights[i][0], **kwargs)
        plt.legend()
        plt.xlabel(prop_name.title() + "/" + units)
        plt.ylabel("Count/" + bin_size)
        plt.savefig(path + "paired_" + prop_name.replace(" ", "_") + path_end + ".png")
        plt.close()

        if inc_log:
            plt.figure(figsize=(12,9))
            plt.hist(sig_2, histtype='step', label="2 signal", bins = bins[i], weights=weights[i][2], **kwargs)
            plt.hist(sig_1, histtype='step', label="1 signal", bins = bins[i], weights=weights[i][1], **kwargs)
            plt.hist(sig_0, histtype='step', label="0 signal", bins = bins[i], weights=weights[i][0], **kwargs)
            plt.legend()
            plt.yscale('log')
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Count/" + bin_size)
            plt.savefig(path + "paired_" + prop_name.replace(" ", "_") + path_end + "_log.png")
            plt.close()

        if inc_norm:
            plt.figure(figsize=(12,9))
            plt.hist(sig_2, histtype='step', density=True, label="2 signal", bins = bins[i], weights=weights[i][2], **kwargs)
            plt.hist(sig_1, histtype='step', density=True, label="1 signal", bins = bins[i], weights=weights[i][1], **kwargs)
            plt.hist(sig_0, histtype='step', density=True, label="0 signal", bins = bins[i], weights=weights[i][0], **kwargs)
            plt.legend()
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Density/" + bin_size)
            plt.savefig(path + "paired_" + prop_name.replace(" ", "_") + path_end + "_norm.png")
            plt.close()

        if inc_log and inc_norm:
            plt.figure(figsize=(12,9))
            plt.hist(sig_2, histtype='step', density=True, label="2 signal", bins = bins[i], weights=weights[i][2], **kwargs)
            plt.hist(sig_1, histtype='step', density=True, label="1 signal", bins = bins[i], weights=weights[i][1], **kwargs)
            plt.hist(sig_0, histtype='step', density=True, label="0 signal", bins = bins[i], weights=weights[i][0], **kwargs)
            plt.legend()
            plt.yscale('log')
            plt.xlabel(prop_name.title() + "/" + units)
            plt.ylabel("Density/" + bin_size)
            plt.savefig(path + "paired_" + prop_name.replace(" ", "_") + path_end + "_norm_log.png")
            plt.close()
    return


def plot_rank_hist(
        prop_name, ranking,
        path="/users/wx21978/projects/pion-phys/plots/photon_pairs/", unique_save_id = "",
        y_scaling = 'log', bins = None, **kwargs
    ):
    """
    Produces a plot displaying a set of ranks (positions) as a histogram.

    Ranked data must already be calculated and gets passed as `ranking`.

    `prop_name` is required to correctly title and save the plot.

    `y_scaling` can be supplied as list of scalings ('log' and 'linear'). If a
    list is passed, the options supplied will be iterated through, and a plot
    will be produced for each value given.

    Parameters
    ----------
    prop_name : str
        Name of the property plotted for title and saved file.
    ranking : ak.Array or np.ndarray
        Array containing the ranks to be plotted.
    path : str, optional
        Directory in which to save the final plot(s). Default is
        '/users/wx21978/projects/pion-phys/plots/photon_pairs/'.
    unique_save_id : str, optional
        Extra string to add into the save name to avoid
        overwriting. Default is ''.
    y_scaling : str or list, optional
        Type of scalings to use on the y-axis. Should be
        'linear' or 'log'. Default is 'log'
    bins : int, np.ndarray, or None, optional
        Number of bins if ``int``, or bin edges if ``ndarray``.
        Gives one bin per ranking if None. Default is None.
    
    Other Parameters
    ----------------
    **kwargs
        Any additional keyword arguments to be passed to the ``plt.hist``
        calls. The same arguments will be passed to all plots.
    """

    if path[-1] != '/':
        path += '/'

    if not isinstance(y_scaling, list):
        y_scaling = [y_scaling]
    if bins is None:
        bins = int(np.max(ranking) -1)

    for y_scale in y_scaling:
        plt.figure(figsize=(12,9))
        plt.hist(ranking, label="signal", bins=bins,  **kwargs)
        plt.legend()
        plt.yscale(y_scale)
        plt.xlabel(prop_name.title() + " ranking")
        plt.ylabel("Count")
        plt.savefig(path + "ranking_" + prop_name.replace(" ", "_") + unique_save_id + ".png")
        plt.close()
    return


def make_truth_comparison_plots(
        events, photon_indicies,
        valid_events=None,
        prop_label=None, inc_log=True,
        path="/users/wx21978/projects/pion-phys/plots/photon_pairs/truth_method_comparisions/",
        **kwargs
    ):
    """
    Produces a set of plots displaying the errors of a set of PFOs with respect to the truth
    PFO they are representing. Errors in energy for the leading (highest energy) and sub-
    leading (lowest energy) photon are displayed in separate plots, and the cosine of the
    angular differences between the PFOs and truth particles is displayed for both photons
    on the same plot.

    `photons_indicies` can be a dictionary to allow comparision of multiple methods of generating
    truth photons to be compared on the same plot.

    Parameters
    ----------
    events : Data
        Events from which data is gathered.
    photon_indicies : dict or np.ndarray
        Indicies of the selected best particles. May be passed as a dictionary containing
        numpy arrays labelled by the method used to generate the indicies for comparision.
    valid_events : list, np.ndarray, ak.Array(), or None
        1D set of boolean values to optionally excluded known invalid events. Default is
        None.
    prop_label : str or None, optional
        Optional name of property to appear in legend if `photons_indicies` is not a
        dictionary.
    inc_log : bool, optional
        Whether to include logarithmically scaled plots. Default is True.
    path : str, optional
        Directory in which to save the final plots. Default is
        '/users/wx21978/projects/pion-phys/plots/photon_pairs/truth_method_comparisions/'.
    
    Other Parameters
    ----------------
    **kwargs
        Any additional keyword arguments to be passed to the ``plt.hist`` calls. The same
        arguments will be passed to all plots.
    """
    if path[-1] != '/':
        path += '/'
    
    # If we don't specify valid events, assume all events are OK, so convert valid events into a
    #   slice which selects all events
    if valid_events is None:
        valid_events = slice(None)

    # Ideally, we can use the trueParticle (not trueparticleBT} data, since trueParticle is likely
    #   already loaded, and trueParticleBT likely hasn't been
    true_energies = events.trueParticlesBT.energy[valid_events]
    # true_energies_mom = vector.magnitude(events.trueParticlesBT.momentum)[valid_events]
    true_dirs = events.trueParticlesBT.direction[valid_events]

    reco_energies = events.recoParticles.energy[valid_events]
    reco_dirs = events.recoParticles.direction[valid_events]

    # Warning - not sure this has been tested without photon_indicies as a dictionary...
    if not isinstance(photon_indicies, dict):
        photon_indicies = {prop_label:photon_indicies}
        
    
    fig_e_i, energy_i_axis    = plt.subplots(figsize=(16,12), layout="tight")
    fig_e_ii, energy_ii_axis  = plt.subplots(figsize=(16,12), layout="tight")
    fig_dirs, directions_axis = plt.subplots(figsize=(16,12), layout="tight")
    
    if inc_log:
        fig_e_i_log, energy_i_axis_log    = plt.subplots(figsize=(16,12), layout="tight")
        fig_e_ii_log, energy_ii_axis_log  = plt.subplots(figsize=(16,12), layout="tight")
        fig_dirs_log, directions_axis_log = plt.subplots(figsize=(16,12), layout="tight")

    for i, prop in enumerate(list(photon_indicies.keys())):
        if isinstance(prop, str):
            y1_label = "y1 " + prop
            y2_label = "y2 " + prop
        else:
            y1_label = None
            y2_label = None

        photon_i_indicies  = np_to_ak_indicies(photon_indicies[prop][:,0][valid_events])
        photon_ii_indicies = np_to_ak_indicies(photon_indicies[prop][:,1][valid_events])
        
        # This might contain some useful stuff for moving to trueParticle informatio, rather than trueParticleBT
        # err_true_photon_i = np.zeros(np.sum(valid_events))
        # photon_i_ids = events.trueParticlesBT.number[valid_events][photon_i_indicies]
        # reco_energy_full = reco_energies[photon_i_indicies]
        # index = np.arange(photon_indicies[prop][:,0].shape[0])[valid_events]
        # for j in range(np.sum(valid_events)):
        #     true_ids = events.trueParticles.number[index[j]].to_numpy()
        #     true_energy = events.trueParticles.energy[index[j]].to_numpy()
            
        #     true_energy_i = true_energy[true_ids == photon_i_ids[j]]
        #     # true_energy_ii = true_energy[true_ids == pfo_truth_ids[photon_ii_indicies]]
        #     err_true_photon_i[j] = (reco_energy_full[j] / true_energy_i )[0] -1


        err_energy_photon_i  = (reco_energies[photon_i_indicies ] / true_energies[photon_i_indicies ]) -1
        err_energy_photon_ii = (reco_energies[photon_ii_indicies] / true_energies[photon_ii_indicies]) -1
    
        err_direction_photon_i  = vector.dot(reco_dirs[photon_i_indicies ], true_dirs[photon_i_indicies ])
        err_direction_photon_ii = vector.dot(reco_dirs[photon_ii_indicies], true_dirs[photon_ii_indicies])

        # Linear
        energy_i_axis.hist( err_energy_photon_i,  label=y1_label, histtype="step", bins=100**kwargs)
        energy_ii_axis.hist(err_energy_photon_ii, label=y2_label, histtype="step", bins=100**kwargs)

        directions_axis.hist(err_direction_photon_i,  label=y1_label, histtype="step", bins=80, color=f"C{i}"**kwargs)
        directions_axis.hist(err_direction_photon_ii, label=y2_label, histtype="step", bins=80, color=f"C{i}", ls="--"**kwargs)

        if inc_log:
            # Log
            energy_i_axis_log.hist( err_energy_photon_i,  label=y1_label, histtype="step", bins=100**kwargs)
            energy_ii_axis_log.hist(err_energy_photon_ii, label=y2_label, histtype="step", bins=100**kwargs)

            directions_axis_log.hist(err_direction_photon_i,  label=y1_label, histtype="step", bins=50, color=f"C{i}"**kwargs)
            directions_axis_log.hist(err_direction_photon_ii, label=y2_label, histtype="step", bins=50, color=f"C{i}", ls="--"**kwargs)

    # Linear
    energy_i_axis.set_xlabel("Fractional energy error")
    energy_i_axis.set_ylabel("Count")
    energy_i_axis.legend()

    energy_ii_axis.set_xlabel("Fractional energy error")
    energy_ii_axis.set_ylabel("Count")
    energy_ii_axis.legend()

    directions_axis.set_xlabel("Best photon vs. truth dot product")
    directions_axis.set_ylabel("Count")
    directions_axis.legend(loc="upper left")

    if inc_log:
        # Log
        energy_i_axis_log.set_xlabel("Fractional energy error")
        energy_i_axis_log.set_ylabel("Count")
        energy_i_axis_log.set_yscale("log")
        energy_i_axis_log.legend()

        energy_ii_axis_log.set_xlabel("Fractional energy error")
        energy_ii_axis_log.set_ylabel("Count")
        energy_i_axis_log.set_yscale("log")
        energy_ii_axis_log.legend()

        directions_axis_log.set_xlabel("Best photon vs. truth dot product")
        directions_axis_log.set_ylabel("Count")
        energy_i_axis_log.set_yscale("log")
        directions_axis_log.legend(loc="upper left")

    fig_e_i.savefig(path + "leading_photon_energy.png")
    fig_e_ii.savefig(path + "subleading_photon_energy.png")
    fig_dirs.savefig(path + "directions.png")
    if inc_log:
        fig_e_i_log.savefig(path + "leading_photon_energy_log.png")
        fig_e_ii_log.savefig(path + "subleading_photon_energy_log.png")
        fig_dirs_log.savefig(path + "directions_log.png")
    plt.close()
    return


######################################################################################################
######################################################################################################
##########                                EVENT MANIPULATION                                ##########
######################################################################################################
######################################################################################################


def get_mother_pdgs(events):
    """
    DEPRECATED
    Use trueParticleBT.motherPdg

    Loops through each event (warning: slow) and creates a reco-type array of PDG codes
    of mother particle. E.g. a photon from a pi0 will be assigned a PDG code 111.

    Mother particle is defined as the mother of the truth particle which the PFO was
    backtracked to.

    Beam particle is hard-coded to have a PDG code of 211.
    
    Anything which cannot have a mother PDG code assigned is given 0.

    Future warning: Mother PDG codes are planned to be added into the ntuple data
    making this function redundant.

    Parameters
    ----------
    events : Data
        Set of events to produce the mother PDG codes for.

    Returns
    -------
    mother_pdgs : ak.Array
        PDG code of the mother of the truth particle that backtrackd to the PFO.
    """

    # This loops through every event and assigned the pdg code of the mother.
    # Assigns 211 (pi+) to daughters of the beam, and 0 to everything which is not found
    mother_ids = events.trueParticlesBT.mother
    truth_ids = events.trueParticles.number
    truth_pdgs = events.trueParticles.pdg
    mother_pdgs = mother_ids.to_list()
    ts = time.time()
    for i in range(ak.num(mother_ids, axis=0)):
        true_pdg_lookup = {truth_ids[i][d]:truth_pdgs[i][d] for d in range(ak.count(truth_ids[i]))}
        true_pdg_lookup.update({0:0, 1:211}) # I have no idea why the hell I did this
                                             # Presumably the beam particle gets a strange pdg code??
        for j in range(ak.count(mother_ids[i])):
            try:
                mother_pdgs[i][j] = true_pdg_lookup[mother_ids[i][j]]
            except:
                mother_pdgs[i][j] = 0

    print(f"Mother PDG codes found in {time.time()  - ts}s")
    mother_pdgs = ak.Array(mother_pdgs)
    return mother_pdgs


def get_MC_truth_beam_mask(event_mothers, event_ids, beam_id=1):
    """
    Loops through all PFOs in an event and generates a mask which selects all PFOs
    which have been backtracked to have been produced by the beam particle, or any
    particles in the daughter tree of the beam particle (i.e. a PFO from a
    daughter of a daughter of the beam particle is selected).

    Parameters
    ----------
    event_mothers : ak.Array
        1D array of all mother ids of the backtracked PFOs in a single event.
    event_ids : ak.Array
        1D array of all truth particle ids of the backtracked PFOs in a single event.
    beam_id : int
        ID code of the truth particle corresponding to the beam particle. Default is 1.

    Returns
    -------
    beam_mask : list
        List of boolean values which select all beam generated PFOs when used as a mask.
    """
    # IDs for particles related to the beam
    beam_ids = np.array([beam_id])
    # IDs of particles which correspond added compnents to beam_ids
    new_ids = event_ids[event_mothers == beam_ids[0]]

    while len(new_ids) != 0:
        # Add new ids into the beam related ids
        beam_ids = np.append(beam_ids, new_ids)
        # Loop through events and create a mask of daughters of the newly added ids
        #   Could be made more efficient by somehow removing PFOs that already exist
        #   in the beam_ids list, another mask?
        mask = [e in new_ids for e in event_mothers]
        # Flag the newly added PFOs as "to be added" and loop, unless no new PFOs are found
        new_ids = event_ids[mask]
    return [e in beam_ids for e in event_ids]


def np_to_ak_indicies(indicies):
    """
    Takes a numpy array of indicies for slicing and converts them to a format
    compatible with awkward arrays which selected PFOs in an event base on the
    index, rather than events themselves based on the index.

    Parameters
    ----------
    indicies: np.ndarray
        Array of indicies for conversion.

    Returns
    -------
    ak_indicies : ak.Array
        Array of indicies which selected PFOs when slicing an awkward array.
    """
    # 1. Expands the dimensions to ensure you hit one index per event
    # 2. Convert to list - this is necessary to ensure the final awkward array has variable size.
    #     Without variable size arrays, it tries to gather the event of the index, not the PFO at the index in the event.
    # 3. Convert to awkward array
    return ak.Array(np.expand_dims(indicies,1).tolist())


######################################################################################################
######################################################################################################
##########                                  EVENT PAIRING                                   ##########
######################################################################################################
######################################################################################################


def truth_pfos_in_two_photon_decay(events, sort=True):
    """
    Returns the truth IDs of the two photons in each event which come from
    pi0 -> yy decay. Requires a mask on `events` to select only events
    which contain exactly pion which decays into two photons.

    The IDs are return as a (num_events, 2) numpy nparray.

    The optional `sort` argument will cause the photons to be sorted by
    energy, such that index 0 of each event contains the leading (higher
    energy) photon.

    Parameters
    ----------
    events : Data
        Set of events in which to look for photons.
    sort : bool, optional
        Defines whether the photon IDs are sorted by energy or not.
        Default is True.
    
    Returns
    -------
    photon_ids : np.ndarray
        Array containing the truth IDs of the two photons created by
        pion decay in each event.
    """
    num_events = ak.num(events.trueParticlesBT.mother, axis=0)

    # This is only valid for the 1 pi0 in event, 2 photon decay cut
    photon_ids = np.zeros((num_events, 2))

    for i in range(num_events):
        truth_mothers = events.trueParticles.mother[i]
        truth_ids = events.trueParticles.number[i]
        truth_pdgs = events.trueParticles.pdg[i]
        truth_energy = events.trueParticles.energy[i].to_numpy()
        
        beam_mask = get_MC_truth_beam_mask(truth_mothers, truth_ids, beam_id=1)
        
        beam_photons_ids = truth_ids[(truth_pdgs == 22) & beam_mask]

        sorted_energies = np.flip(np.argsort(truth_energy[(truth_pdgs == 22) & beam_mask])) if sort else [0, 1]
        photon_ids[i,:] = beam_photons_ids[sorted_energies]
    return photon_ids


def get_best_pairs(
        events,
        # truth_photon_ids,
        method='mom', return_type="mask", valid_mom_cut=False, report=False, verbosity=0
    ):
    """
    Finds the 'best' pair of PFOs in `events` which match the truth
    photons assuming 1 pi0->yy in each event. The method to select the
    'best is chosen by `method`.

    The avaiable methods are as follows:
    - mom : Momentum method - best PFO is that with the greatest momentum
    projection along the direction of the true photon.
    - energy : Energy method - best PFO is that with the largest energy.
    - dir : Direction method - best PFO is that which is closest aligned
    to the direction of the true photon.
    - purity : Purity method - best PFO is that with the highest purity,
    i.e. is made up of the greatest fraction of hits which come from
    the true photon.
    - completeness : Completeness method - best PFO is that with the
    highest completeness, i.e. the PFO which contains the greatest
    number of hits generated by the true photon.
    
    Multiple methods may be used simulatenously by passing a list of
    desired methods. If more than one method is selected, the output PFOs
    will be contained in a dictionary labelled by the method used.

    The `return_type` argument allows selection of the format of the
    output pairs. Available formats are as follows:
    - mask : Returns a awkward boolean mask which picks out the two best
    PFOs for each event when applied to reco data in `events`. Note that
    this results in a loss of information regarding the sorting of
    values from `truth_photon_ids`.
    - id : Returns the reco IDs of the PFOs corresponding to the best
    PFOs in each event as a numpy array.
    - index : Returns the indicies corresponding to the best matching
    PFOs in each event as a numpy array.

    Additionally the function will note any events for which a best pair
    cannot be created due to one or both of the true photons not having
    any associated PFOs. A `valid_event_mask` is returned which will
    select only the events in which each true photon has at least one
    backtracked PFO when applied to `events`. Additionally the `report`
    argument will cause the function to print out a notice detailing the
    number of events dropped.

    `valid_mom_cut` will cause a cut to be made on objects which have an
    invalid momentum value (-999., -999., -999.). This is necessary if
    this cut hasn't yet been applied to `events`, and the method used is
    one of "mom", "energy", or "dir".

    Parameters
    ----------
    events : Data
        Set of events for which the pairs should be found.
    method : {'mom', 'energy', 'dir', 'purity', 'completeness', 'all'} or \
list, optional
        Method(s) to use to find the best PFOs. If passed as a list,
        multiple methods will be used. Default is 'mom'
    return_type : {'mask', 'id', 'index', str}, optional
        Format to output the best PFOs. If a non-matching string is passed,
        'index' will be used by default. Default is 'mask'.
    valid_mom_cut : bool, optional
        Whether to filter out PFOs which have an invalid momentum. Default
        is False.
    report : bool, optional
        Whether to print out two lines detailing how many events were
        dropped due to lack of backtracked PFOs. Default is False.
    verbosity : int, optional
        Controls the amount of information printed at each step for
        debugging purposes from none at 0, to full at 6. Default is 0.
    
    Returns
    -------
    truth_indicies_photon_beam : np.ndarray, ak.Array, or dict
        Best PFOs in each event formatted as specified by `return_type`.
        if `method` a list, a dictionary of results indexed by the values
        in `method` is returned.
    valid_event_mask : np.ndarray
        Boolean mask which selects events where both truth photons have
        backtracked PFOs when applied to `events`.
    """
    # Work out the method of determining the best pair to use. The dictionary allows for
    # aliasing of methods, but currently this is removed to avoid potential confusion
    known_methods = {
        # "momentum"      : "mom",
        "mom"           : "mom",
        # "e"             : "energy",
        "energy"        : "energy",
        # "d"             : "dir",
        "dir"           : "dir",
        # "direction"     : "dir",
        "purity"        : "purity",
        "completeness"  : "completeness",
        # "comp"          : "completeness",
        "all"           : "all"
    }
    if not isinstance(method, list):
        method = [method]
    bad_methods = []
    for i, m in enumerate(method):
        try:
            method[i] = known_methods[m]
        except(KeyError):
            bad_methods += [m]
            print(f'Method(s) "{m}" not found, please use one of:\nmomentum, energy, direction, purity, completeness')
    if len(bad_methods) != 0:
        join_str = '", "'
        raise ValueError(f'Method(s): "{join_str.join(bad_methods)}"\nnot found, please use:\n"mom", "energy", "dir", "purity", "completeness", or "all"')

    # Parse which methods to use
    if "all" in method:
        methods_to_use = ["mom", "energy", "dir", "purity", "completeness"]
    else:
        methods_to_use = method


    # Work out the number of events we have
    num_events = ak.num(events.trueParticlesBT.number, axis=0)

    # Currently only worrying about single pi0->yy in each event. In future, we could extent this by allowing
    # `truth_photon_ids` as an argument, and then iterating over each photon in each event, but that's a later problem!
    truth_photon_ids = truth_pfos_in_two_photon_decay(events, sort=return_type!="mask")

    # Definitions of what each method involves
    #   - testing_methods: The test to be performed between the true and reco data to get the pair ordering
    #   - reco_props: The reco data of the candidate PFOs
    #   - truth_props: The truth data to be tested against

    testing_methods = {
        "mom": lambda reco, true: np.argsort([ vector.dot(r, true)[0] for r in reco ]), # This is want enforces looping 
        "dir": lambda reco, true: np.argsort([ vector.dot(r, true)[0] for r in reco ]), # per event at the moment
        "energy": lambda reco, true: np.argsort(reco),
        "purity": lambda reco, true: np.argsort(reco),
        "completeness": lambda reco, true: np.argsort(reco)
    }
    # TODO Swap to lambda functions for we don't fetch these unless we actually need them
    reco_props = {
        "mom": events.recoParticles.momentum,
        "dir": events.recoParticles.direction,
        "energy": events.recoParticles.energy,
        "purity": events.trueParticlesBT.purity,
        "completeness": events.trueParticlesBT.completeness
    }
    # TODO Change energy ... to use a smaller zeros/like (events.trueParticles.number ?)
    true_props = {
        "mom": events.trueParticles.momentum,
        "dir": events.trueParticles.direction,
        "energy": ak.zeros_like(events.trueParticles.number),
        "purity": ak.zeros_like(events.trueParticles.number),
        "completeness": ak.zeros_like(events.trueParticles.number)
    }


    # Other properties which must be set up ahead of the loop
    valid_event_mask = np.full(num_events, True, dtype=bool)

    truth_indicies_photon_beam = {}
    if return_type == "mask":
        for m in methods_to_use:
            truth_indicies_photon_beam.update({m : [[]] * num_events})
    else:
        # This is only valid for the 1 pi0 in event, 2 photon decay cut
        for m in methods_to_use:
            truth_indicies_photon_beam.update({m : np.zeros((num_events, 2), dtype=int)})

    zero_count = 0
    one_count = 0

    # Loop over each event to determine the best pair
    for i in range(num_events):
        bt_ids = events.trueParticlesBT.number[i]
        indicies = np.arange(len(bt_ids))

        true_ids = events.trueParticles.number[i]

        # Sorted ids should be supplied as an argument generated by truth_pfos_in_two_photon_decay(evts)
        photon_i, photon_ii = truth_photon_ids[i]


        if verbosity >= 1:
            true_momenta = events.trueParticles.momentum[i]
            if verbosity >= 2:
                print("\nTrue particle energies (GeV)")
                print(vector.magnitude(true_momenta[true_ids == photon_i ]))
                print(vector.magnitude(true_momenta[true_ids == photon_ii]))
                print("True particle directions")
                print(vector.normalize(true_momenta[true_ids == photon_i ]))
                print(vector.normalize(true_momenta[true_ids == photon_ii]))
            else:
                print(true_momenta[true_ids == photon_i ])
                print(true_momenta[true_ids == photon_ii])

        if valid_mom_cut:
            # Maybe want to look at what happens when we use purity/completeness with no good data cut?
            reco_momenta = events.recoParticles.momentum[i]
            good_data = np.logical_and(np.logical_and(reco_momenta.x != -999., reco_momenta.y != -999.), reco_momenta.z != -999.)
        else:
            good_data = slice(None)

        # Count how many events are cut with no photons/only one photon having at least one daughter
        photon_i_exists = photon_i in bt_ids[good_data]
        photon_ii_exists = photon_ii in bt_ids[good_data]
        if (not photon_i_exists) or (not photon_ii_exists):
            if not (photon_i_exists or photon_ii_exists):
                zero_count += 1
            else:
                one_count += 1
            valid_event_mask[i] = False
            continue # Skips to next iteration if not both phhotons have daughters

        for m in methods_to_use:
            # Get the data to use
            reco_prop = reco_props[m][i]
            true_prop = true_props[m][i]
            # Get the truth property of each photon
            true_prop_i  = true_prop[true_ids == photon_i ]
            true_prop_ii = true_prop[true_ids == photon_ii]
            # Get a mask indicating the daughters of each photon in the reco data
            photon_i_mask  = [ bt_ids[good_data] == photon_i  ][0]
            photon_ii_mask = [ bt_ids[good_data] == photon_ii ][0]
            # Need the [0] because bt_ids[good_data] == photon_i is an array, so returns an array (one element)

            # Order the PFOs by the selected method
            reco_prop_ordering_i  = testing_methods[m](reco_prop[ indicies[good_data][photon_i_mask ] ], true_prop_i )
            reco_prop_ordering_ii = testing_methods[m](reco_prop[ indicies[good_data][photon_ii_mask] ], true_prop_ii)
            # Returns the index of the PFO selected as the best selection for each photon
            photon_i_index  = indicies[good_data][photon_i_mask ][ reco_prop_ordering_i[-1]]
            photon_ii_index = indicies[good_data][photon_ii_mask][reco_prop_ordering_ii[-1]]


            if verbosity >= 6:
                print(f"List of reco values {m} with good data:")
                print(*reco_props[m][good_data])
            if verbosity >= 4:
                print(f"Values of {m}:")
                print("Leading photon")
                print(reco_props[indicies[good_data][photon_i_mask ]])
                print("Sub-leading photon")
                print(reco_props[indicies[good_data][photon_ii_mask]])
            if verbosity >= 5:
                print(f"Index ordering of {m} values")
                print("Leading photon")
                print(reco_prop_ordering_i)
                print("Sub-leading photon")
                print(reco_prop_ordering_ii)
            if verbosity >= 3:
                print(f"Selected {m}:")
                print("Leading photon")
                print(reco_prop[photon_i_index])
                print("Sub-leading photon")
                print(reco_prop[photon_ii_index])

            # If we are returning IDs, we need to find the corresponding reco ID
            if return_type == "id":
                reco_ids = events.recoParticles.number[i]
                truth_indicies_photon_beam[m][i,:] = [reco_ids[photon_i_index], reco_ids[photon_ii_index]]
            # If returning a mask we need to construct the mask
            elif return_type == "mask":
                event_mask = [False] * (indicies[-1] + 1)
                event_mask[photon_i_index] = True
                event_mask[photon_ii_index] = True
                truth_indicies_photon_beam[m][i] = event_mask
            # Otherwise, we just return the reco indicies of the selected PFOs
            else:
                truth_indicies_photon_beam[m][i,:] = [photon_i_index, photon_ii_index]
    
    if report:
        print(f"{zero_count} events discarded due to no true photons having matched PFOs.")
        print(f"{one_count} events discarded due to only one true photon having matched PFOs.")

    if return_type == "mask":
        truth_indicies_photon_beam = ak.Array(truth_indicies_photon_beam)
    if len(methods_to_use) == 1:
        return truth_indicies_photon_beam[method[0]], valid_event_mask
    else:
        return truth_indicies_photon_beam, valid_event_mask


def pair_apply_sig_mask(truth_mask, pair_coords):
    """
    Returns a count of the number of PFOs which exist in `truth_mask` for
    each pair in a set of pairs defined by `pair_coords`.

    Parameters
    ----------
    truth_mask : ak.Array
        Array of boolean values masking a set of signal PFOs.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.
    
    Returns
    -------
    sig_counts : ak.Array()
        Number of signal PFOs in each pair.
    """
    # Convert the mask to integers
    true_counts = np.multiply(truth_mask, 1)

    # Add the results
    return true_counts[pair_coords["0"]] + true_counts[pair_coords["1"]]


def gen_pair_sig_counts(events, pair_coords):
    """
    Returns a count of the number of PFOs in a pair which
    contribute to a good signal:
    - 2 means the PFOs come from different photons produced
    by the same pi0
    - 1 means 1 PFO of a photon from a pi0 exists, or both
    PFOs were from the same photon
    - No photons from pi0s

    Parameters
    ----------
    truth_mask : ak.Array
        Array of boolean values masking a set of signal PFOs.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.
    
    Returns
    -------
    sig_counts : ak.Array()
        Number of signal PFOs in each pair.
    """
    photon_from_pions = np.logical_and(events.trueParticlesBT.pdg == 22, events.trueParticlesBT.motherPdg == 111)

    different_daughter = events.trueParticlesBT.number[pair_coords["0"]] != events.trueParticlesBT.number[pair_coords["1"]]
    same_mother = events.trueParticlesBT.mother[pair_coords["0"]] == events.trueParticlesBT.mother[pair_coords["1"]]

    same_mother_and_different_daughter = np.logical_and(same_mother, different_daughter)
    del same_mother
    del different_daughter

    both_photons = np.logical_and(photon_from_pions[pair_coords["0"]], photon_from_pions[pair_coords["1"]])

    photons_form_pi0 = np.logical_and(both_photons, same_mother_and_different_daughter)
    del same_mother_and_different_daughter
    del both_photons

    at_least_one_photon = np.logical_or(photon_from_pions[pair_coords["0"]], photon_from_pions[pair_coords["1"]])
    # Add the results
    return np.multiply(at_least_one_photon, 1) + np.multiply(photons_form_pi0, 1)


def get_sig_count(events, pair_coordinates, single_best=False, **kwargs):
    """
    Generates an awkward array matching the size of `pair_coordinates` indicating
    how many good signal PFOs are in the indexed pair.

    If `single_best` is set to `True`, a maximum of one pair may have two signal
    PFOs per event. This is determined by the `get_best_pairs()` function, and
    can take additional keyword arguments. See `get_best_pairs` for details.

    Parameters
    ----------
    events : Data
        Events used to generate the pairs.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.
    single_best : bool, optional
        Whether to select only one best pair per event, or if multiple
        pairs can have a signal count of 2. Default is False.
    kwargs
        Additional keyword arguments to be passed to the `get_best_pairs`
        function if `single_best` is set to `True`.
    """
    if single_best:
        truth_pair_indicies, valid_events = get_best_pairs(events, **kwargs)

        events.Filter([valid_events], [valid_events])
        truth_pair_indicies = truth_pair_indicies[valid_events]
        del valid_events
        return pair_apply_sig_mask(truth_pair_indicies, pair_coordinates)
    else:
        return gen_pair_sig_counts(events, pair_coordinates)


def pair_photon_counts(events, pair_coords, mother_pdgs):
    """
    Returns a count of the number of PFOs which have been backtracked
    to photons which are daughters of pi0s for each pair in a set of
    pairs defined by `pair_coords`.

    Future warning: `mother_pdgs` will be removed in later version, as
    mother PDG codes are planned to be added into ntuple data.

    Parameters
    ----------
    events : Data
        Events used to generate the pairs.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.
    mother_pdgs : ak.Array
        PDG codes of the mother of the backtracked truth particles.

    Returns
    -------
    sig_counts : ak.Array()
        Number of photons produced by pi0s in each pair.
    """
    true_photons = events.trueParticlesBT.pdg == 22
    # Get the locations wehere the pdg is 22 and mother pdg is 111
    first_sigs = np.logical_and(true_photons[pair_coords["0"]], mother_pdgs[pair_coords["0"]] == 111)
    # Multiplying by 1 sets the dtype to be int (1 where True, 0 where False)
    first_sigs = np.multiply(first_sigs, 1)
    #Same for second particle
    second_sigs = np.logical_and(true_photons[pair_coords["1"]], mother_pdgs[pair_coords["1"]] == 111)
    second_sigs = np.multiply(second_sigs, 1)
    # Add the results
    return first_sigs + second_sigs


######################################################################################################
######################################################################################################
##########                                   CALCULATIONS                                   ##########
######################################################################################################
######################################################################################################


def get_impact_parameter(direction, start_pos, beam_vertex):
    """
    Finds the impact parameter between a PFO and beam vertex.

    Parameters
    ----------
    direction : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Direction of PFO.
    start_pos : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of the PFO.
    beam_vertex : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Beam interaction point.

    Returns
    -------
    m : ak.Array
        Impact parameter between the PFO and beam vertex.
    """
    rel_pos = vector.sub(beam_vertex, start_pos)
    cross = vector.cross(rel_pos, direction)
    return vector.magnitude(cross)


def get_separation(pos1, pos2):
    """
    Finds the separation between positions `pos1` and `pos2`.

    Parameters
    ----------
    spos1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Postion 1.
    spos1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Postion 2.

    Returns
    -------
    separation : ak.Array
        Separation between positions 1 and 2.
    """
    return vector.magnitude(vector.sub(pos1, pos2))


def closest_approach(dir1, dir2, start1, start2):
    """
    Finds the closest approach between two showers.

    Parameters
    ----------
    dir1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Direction of first PFO.
    dir2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Direction of second PFO.
    start1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of first PFO.
    start2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of second PFO.

    Returns
    -------
    d : ak.Array
        Magnitude of the sortest direction between the two PFOs.
    """
    # x_1 - x_2 + lambda_1 v_1 - lambda_2 v_2 = d/sin(theta) v_1 x v_2
    cross = vector.normalize( vector.cross(dir1, dir2) )
    rel_start = vector.sub(start1, start2)
    # Separation between the lines
    d = vector.dot(rel_start, cross)
    return d


def get_shared_vertex(mom1, mom2, start1, start2):
    """
    Estimates a shared vertex for two vectors and starting positions
    by taking the midpoint of the line of closest approach.

    This is a projected point, not the difference between positions.

    Momenta are used instead of directions to allow for potential
    updates which weight the position of the vertex along the line
    of closest approach based on the relative momenta.

    Parameters
    ----------
    mom1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Momentum of first PFO.
    mom2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Momentum of second PFO.
    start1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of first PFO.
    start2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of second PFO.
    
    Returns
    -------
    vertex : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Shared vertex between the PFOs.
    """
    # We estimate the shared vertex by taking the midpoint of the line of closest approch
    # This is a projected point, NOT the difference between the reconstructed starts
    joining_dir = vector.normalize( vector.cross(mom1, mom2) )
    separation = vector.dot(vector.sub(start1, start2), joining_dir)
    dir1_selector = vector.cross(joining_dir, mom2)
    # We don't use the normalised momentum, because we later multiply by the momentum, so it cancels
    start1_offset = vector.dot( vector.sub(start2, start1), dir1_selector ) / vector.dot(mom1, dir1_selector)
    pos1 = vector.add(start1, vector.prod(start1_offset, mom1))
    return vector.add(pos1, vector.prod(separation/2, joining_dir))


def get_midpoints(x1, x2):
    """
    Returns the midpoint of the starting positions:
        ``mp = (x1 + x2)/2``

    Parameters
    ----------
    x1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Position of first point.
    x2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Position of second point.

    Returns
    -------
    mp : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Midpoint of `x1` and `x2`.
    """
    return vector.prod(0.5, vector.add(x1, x2))


######################################################################################################
######################################################################################################
##########                                 PAIR PROPERTIES                                  ##########
######################################################################################################
######################################################################################################

def paired_mass(events, pair_coords):
    """
    Finds the mass of the pairs given by `pair_coords` from `events` assuming
    relativistic limit.

    Parameters
    ----------
    events : Data
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    mass : ak.Array
        Invarient masses of the pairs.
    """
    # Get the momenta via the pair indicies
    first_mom = events.recoParticles.momentum[pair_coords["0"]]
    second_mom = events.recoParticles.momentum[pair_coords["1"]]
    # Calculate
    e = vector.magnitude(first_mom) + vector.magnitude(second_mom)
    p = vector.magnitude(vector.add(first_mom, second_mom))
    return np.sqrt(e**2 - p**2)


def paired_momentum(events, pair_coords):
    """
    Finds the summed momenta of the pairs given by `pair_coords` from `events`.

    Parameters
    ----------
    events : Data
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    mom : ak.Array
        Summed momenta of the pairs.
    """
    # Get the momenta via the pair indicies
    first_mom = events.recoParticles.momentum[pair_coords["0"]]
    second_mom = events.recoParticles.momentum[pair_coords["1"]]
    # Calculate
    return vector.add(first_mom, second_mom)


def paired_energy(events, pair_coords):
    """
    Finds the summed energies of the pairs given by `pair_coords` from `events`.

    Parameters
    ----------
    events : Data
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    energy : ak.Array
        Summed energies of the pairs.
    """
    # Get the energies via the pair indicies
    first_mom = events.recoParticles.momentum[pair_coords["0"]]
    second_mom = events.recoParticles.momentum[pair_coords["1"]]
    # Calculate
    return vector.magnitude(first_mom) + vector.magnitude(second_mom)


def paired_closest_approach(events, pair_coords):
    """
    Finds the closest approaches of the pairs given by `pair_coords` from `events`.

    Parameters
    ----------
    events : Data
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    closest_approach : ak.Array
        Distance of closest approach of the pairs.
    """
    first_dir = events.recoParticles.direction[pair_coords["0"]]
    first_pos = events.recoParticles.startPos[pair_coords["0"]]
    second_dir = events.recoParticles.direction[pair_coords["1"]]
    second_pos = events.recoParticles.startPos[pair_coords["1"]]
    return closest_approach(first_dir,second_dir,first_pos,second_pos)


def paired_beam_impact(events, pair_coords):
    """
    Finds the impact parameter between the beam and the combined momentum
    traced back from the midpoint of the closest approach of each pair in
    `pair_coords` for each event in `events`.

    Parameters
    ----------
    events : Data
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    impact_parameter : ak.Array
        Impact parameter between the beam vertex and each pairs.
    """
    first_mom = events.recoParticles.momentum[pair_coords["0"]]
    first_pos = events.recoParticles.startPos[pair_coords["0"]]
    second_mom = events.recoParticles.momentum[pair_coords["1"]]
    second_pos = events.recoParticles.startPos[pair_coords["1"]]
    # Direction of the summed momenta of the PFOs
    paired_direction = vector.normalize(vector.add(first_mom, second_mom))
    # Midpoint of the line of closest approach between the PFOs
    shared_vertex = get_shared_vertex(first_mom, second_mom, first_pos, second_pos)
    # Impact parameter between the PFOs and corresponding beam vertex
    return get_impact_parameter(paired_direction, shared_vertex, events.recoParticles.beamVertex)


def paired_separation(events, pair_coords):
    """
    Finds the separations between start positions of the pairs given
    by `pair_coords` from `events`.

    Parameters
    ----------
    events : Data
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    separation : ak.Array
        Separation between start points of the pairs.
    """
    # Get the positions via the pair indicies
    first_pos = events.recoParticles.startPos[pair_coords["0"]]
    second_pos = events.recoParticles.startPos[pair_coords["1"]]
    return get_separation(first_pos, second_pos)


def paired_beam_slice(events, pair_coords):
    """
    Finds the detector slice that both PFOs in the the pairs given
    by `pair_coords` from `events`. If they do not have a shared
    slice, returns -1 for the pair.

    Parameters
    ----------
    events : Data
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    slice : ak.Array
        Slice shared by the PFOs in the pairs.
    """
    # Get the positions via the pair indicies
    first_slice = events.recoParticles.sliceID[pair_coords["0"]]
    second_slice = events.recoParticles.sliceID[pair_coords["1"]]
    return (first_slice == second_slice) * (first_slice + 1) - 1


# N.B. this is phi in Shyam's scheme
def paired_opening_angle(events, pair_coords):
    """
    Finds the opening angles of the pairs given by `pair_coords` from `events`.

    Parameters
    ----------
    events : Data
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    angle : ak.Array
        Opening angle of the pairs.
    """
    # Get the momenta via the pair indicies
    first_dir = events.recoParticles.direction[pair_coords["0"]]
    second_dir = events.recoParticles.direction[pair_coords["1"]]
    # Calculate
    return np.arccos(vector.dot(first_dir, second_dir))


######################################################################################################
######################################################################################################
##########                                     RUNNING                                      ##########
######################################################################################################
######################################################################################################


if __name__ == "__main__":

    print("Running pair analysis")

    plot_config = PairPlots.PlotConfig()
    plot_config.SAVE_FOLDER = "/users/wx21978/projects/pion-phys/plots/photon_pairs_1GeV_multi_sig/"

    # Setting up the batch plotters:
    mass_plotter        = PairPlots.PairHistsBatchPlotter("mass", "MeV",            plot_config=plot_config, range=[None, 1000, 400], bins=[500, 100, 40])
    momentum_plotter    = PairPlots.PairHistsBatchPlotter("momentum", "MeV",        plot_config=plot_config, range=[None, 2000, 500], bins=[500,200,60])
    energy_plotter      = PairPlots.PairHistsBatchPlotter("energy", "MeV",          plot_config=plot_config, range=[None, 2000, 500], bins=[500,200,60])
    approach_plotter    = PairPlots.PairHistsBatchPlotter("closest approach", "cm", plot_config=plot_config, range=[None, (-100, 50)], bins=[100,50])
    separations_plotter = PairPlots.PairHistsBatchPlotter("pfo separation", "cm",   plot_config=plot_config, range=[None, 200], bins=[100,30])
    impact_plotter      = PairPlots.PairHistsBatchPlotter("beam impact", "cm",      plot_config=plot_config, range=[None, 200], bins=[100,30])
    angles_plotter      = PairPlots.PairHistsBatchPlotter("angle", "rad",           plot_config=plot_config, range=None, bins=100)

    # Make the bin widths that keep equal area on a sphere
    # Area cover over angle dTheta is r sin(Theta) dTheta (with r=1)
    # So we need constant sin(Theta) dTheta
    # In the range Theta = [0, pi), we have
    # \int^\pi_0 sin(\theta) d\theta = 2
    # So for 100 bins, we need: sin(Theta) dTheta = 2/100 = 0.02
    # \int^{\theta_new}_{\theta_old} sin(\theta) d\theta = 2/100
    # So 0.2 = cons(theta_old) - cos(theta_new)
    n_bins = 100
    sphere_bins = np.zeros(n_bins+1)
    for i in range(n_bins):
        sphere_bins[i+1] = np.arccos(np.max([np.cos(sphere_bins[i]) - 2/n_bins, -1]))

    angles_sphere_plotter =      PairPlots.PairHistsBatchPlotter("angle", "arcrad", plot_config=plot_config, range=None, bins=sphere_bins, unique_save_id = "_sphere", inc_norm=False)
    angles_sphere_norm_plotter = PairPlots.PairHistsBatchPlotter("angle", "arcrad", plot_config=plot_config, range=None, bins=sphere_bins, unique_save_id = "_sphere_norm", inc_norm=False)


    # Get the batches
    batch_folder = "/scratch/wx21978/pi0/root_files/1GeV_beam_v3/"

    batch_names = os.listdir(batch_folder)

    for batch in batch_names:
        print("Beginning batch: " + batch)

        evts = load_and_cut_data(
            batch_folder + batch,
            batch_size = -1, batch_start = -1,
            pion_count="both",
            cnn_cut=0.5,
            n_hits_cut=80,
            beam_slice_cut=False,
            distance_bounds_cm=(3,90),
            max_impact_cm=20)

        # truth_pair_indicies, valid_events = get_best_pairs(evts, method="mom", return_type="mask", report=True)

        # evts.Filter([valid_events], [valid_events])
        # truth_pair_indicies = truth_pair_indicies[valid_events]
        # del valid_events


        pair_coords = ak.argcombinations(evts.recoParticles.number, 2)


        # sig_count = pair_apply_sig_mask(truth_pair_indicies, pair_coords)
        # del truth_pair_indicies
        sig_count = get_sig_count(evts, pair_coords)


        print("Plotting masses...")
        mass_plotter.add_batch(paired_mass(evts, pair_coords), sig_count)

        print("Plotting momenta...")
        momentum_plotter.add_batch(vector.magnitude(paired_momentum(evts, pair_coords)), sig_count)
        
        print("Plotting energies...")
        energy_plotter.add_batch(paired_energy(evts, pair_coords), sig_count)

        print("Plotting closest approaches...")
        approach_plotter.add_batch(paired_closest_approach(evts, pair_coords), sig_count)

        print("Plotting separations...")
        separations_plotter.add_batch(paired_separation(evts, pair_coords), sig_count)

        print("Plotting beam impact parameters...")
        impact_plotter.add_batch(paired_beam_impact(evts, pair_coords), sig_count)

        print("Plotting opening angles...")
        angles = paired_opening_angle(evts, pair_coords)
        angles_plotter.add_batch(angles, sig_count)

        print("Plotting opening angles in bins of equal arcradians...")
        angles_sphere_plotter.add_batch(angles, sig_count)
        angles_sphere_norm_plotter.add_batch(angles, sig_count, weights=ak.full_like(angles, 1/(0.04*np.pi)))
        del angles

    print("Making plots...")
    mass_plotter.make_figures()
    momentum_plotter.make_figures()
    energy_plotter.make_figures()
    approach_plotter.make_figures()
    separations_plotter.make_figures()
    impact_plotter.make_figures()
    angles_plotter.make_figures()
    angles_sphere_plotter.make_figures()
    angles_sphere_norm_plotter.make_figures()

    print("All plots made.")