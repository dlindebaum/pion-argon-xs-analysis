#!/usr/bin/env python

# Imports
import sys
sys.path.insert(1, '/users/wx21978/projects/pion-phys/pi0-analysis/analysis/')
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from python.analysis import Master, vector
import time
from memory_profiler import profile

print("Running pair analysis")

# Edit the base classes to fit what we want to be doing with pairs
class RecoPairData(Master.RecoParticleData):
    @property
    def pandoraTag(self):
        self.LoadData("pandoraTag", "pandoraTag")
        return getattr(self, f"_{type(self).__name__}__pandoraTag")

class PairData(Master.Data):
    def __init__(self, _filename : str = None, includeBackTrackedMC : bool = False, _nEvents : int = -1, _start : int = 0) -> None:
        super().__init__(_filename, includeBackTrackedMC, _nEvents, _start)
        if self.filename != None:
            self.recoParticles = RecoPairData(self) # Set the reco particles as Pairs to inlucde pandora tag and CNN score
    

# Functions

# This is a memory management tool, currently not really necessary
def del_prop(obj, property_name):
    """
    Deletes a properties from the supplied `RecoPairData` object.

    Requires the `obj` to have a property ``_RecoPairData__{property_name}``.

    Parameters
    ----------
    obj : RecoPairData
        Object from which to remove the property.
    property_name : str
        Property to be deleted (should match the name of the property).
    """
    del(obj.__dict__["_RecoPairData__" + property_name])
    return


# # Load the events and performs cuts
# def load_and_cut_data(path, batch_size = -1, batch_start = -1, cnn_cut = True):
#     # Load the events (ETA ~1min for a single 6GeV beam v2 sample). ETA ~10s
#     events = PairData(path, includeBackTrackedMC=True, _nEvents=batch_size, _start=batch_start)

#     # Apply cuts:
#     n = [["Event selection", "Number of PFOs", "Average PFOs per event", "Number of events", "Percentage of events removed"]]
#     n.append(["no selection", ak.count(events.trueParticlesBT.pdg), ak.count(events.eventNum), 0])

#     plt.figure(figsize=(8,6))
#     plt.hist(ak.count(events.trueParticlesBT.pdg, axis=-1), bins=100)
#     plt.xlabel("Number of PFOs in event")
#     plt.ylabel("Count")
#     plt.title("No cut")
#     plt.savefig("/users/wx21978/projects/pion-phys/plots/photon_pairs/PFO_dist_no_cut.png")
#     plt.close()

#     # # Require a beam particle to exist. ETA ~60s:
#     # ts = time.time()
#     # events.ApplyBeamTypeFilter(211)
#     # print(f"pi+ beam done in {time.time()  - ts}s")
#     # evt_remaining = ak.count(events.eventNum)
#     # n.append(["pi+ beam", ak.count(events.trueParticlesBT.pdg), evt_remaining, 100*(n[-1][2] - evt_remaining)/n[-1][2]])

#     # Require a beam particle to exist. ETA ~15s:
#     ts = time.time()
#     events.ApplyBeamFilter() # apply beam filter if possible
#     print(f"beam done in {time.time()  - ts}s")
#     evt_remaining = ak.count(events.eventNum)
#     n.append([ "beam", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(n[-1][3] - evt_remaining)/n[-1][2] ])

#     plt.figure(figsize=(8,6))
#     plt.hist(ak.count(events.trueParticlesBT.pdg, axis=-1), bins=100)
#     plt.xlabel("Number of PFOs in event")
#     plt.ylabel("Count")
#     plt.title("Beam")
#     plt.savefig("/users/wx21978/projects/pion-phys/plots/photon_pairs/PFO_dist_beam_particle.png")
#     plt.close()

#     # Require pi+ beam. ETA ~10s
#     ts = time.time()
#     true_beam = events.trueParticlesBT.pdg[events.recoParticles.beam_number == events.recoParticles.number]
#     f = ak.all(true_beam == 211, -1)
#     events.Filter([f], [f])
#     del(true_beam)
#     print(f"pi+ beam done in {time.time()  - ts}s")
#     evt_remaining = ak.count(events.eventNum)
#     n.append(["pi+ beam", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(n[-1][3] - evt_remaining)/n[-1][2]])

#     plt.figure(figsize=(8,6))
#     plt.hist(ak.count(events.trueParticlesBT.pdg, axis=-1), bins=100)
#     plt.xlabel("Number of PFOs in event")
#     plt.ylabel("Count")
#     plt.title("(beam) pi+ beam")
#     plt.savefig("/users/wx21978/projects/pion-phys/plots/photon_pairs/PFO_dist_pi+_beam.png")
#     plt.close()

#     # # Only look at PFOs with > 50 hits. ETA ~30s:
#     # ts = time.time()
#     # events.Filter([events.recoParticles.nHits > 50], [])
#     # del_prop(events.recoParticles, "nHits")
#     # print(f"Hits > 50 done in {time.time()  - ts}s")
#     # evt_remaining = ak.count(events.eventNum)
#     # n.append(["nHits >= 51", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(n[-1][3] - evt_remaining)/n[-1][2]])

#     if cnn_cut:
#         # Take CNNScore > 0.36. ETA ~15s:
#         cnn_cut = 0.36
#         ts = time.time()
#         events.Filter([events.recoParticles.cnnScore > cnn_cut], [])
#         del_prop(events.recoParticles, "cnnScore")
#         print(f"CNNScore > 0.36 done in {time.time()  - ts}s")
#         evt_remaining = ak.count(events.eventNum)
#         n.append([f"CNNScore > {cnn_cut}", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(n[-1][3] - evt_remaining)/n[-1][2]])

#         plt.figure(figsize=(8,6))
#         plt.hist(ak.count(events.trueParticlesBT.pdg, axis=-1), bins=100)
#         plt.xlabel("Number of PFOs in event")
#         plt.ylabel("Count")
#         plt.title("(beam, pi+) CNNScore>0.36")
#         plt.savefig("/users/wx21978/projects/pion-phys/plots/photon_pairs/PFO_dist_cnn_0.36.png")
#         plt.close()

#     # Require >= 2 PFOs. ETA ~90s:
#     ts = time.time()
#     f = Master.NPFPMask(events, -1)
#     del_prop(events.recoParticles, "direction")
#     del_prop(events.recoParticles, "startPos")
#     events.Filter([f], [f])
#     print(f"PFOs >= 2 done in {time.time()  - ts}s")
#     evt_remaining = ak.count(events.eventNum)
#     n.append(["nPFP >= 2", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(n[-1][3] - evt_remaining)/n[-1][2]])

#     plt.figure(figsize=(8,6))
#     plt.hist(ak.count(events.trueParticlesBT.pdg, axis=-1), bins=100)
#     plt.xlabel("Number of PFOs in event")
#     plt.ylabel("Count")
#     plt.title("(beam, pi+, CNNScore) NPFPs >=2")
#     plt.savefig("/users/wx21978/projects/pion-phys/plots/photon_pairs/PFO_dist_nPFPs>2.png")
#     plt.close()

#     print("\n".join([str(l) for l in n]))
    
#     plt.close()

#     return events

def load_and_cut_data_two_photon(path, batch_size = -1, batch_start = -1, cnn_cut=True, valid_momenta=True, beam_slice_cut=True):
    """
    Loads the ntuple file from `path` and performs the initial cuts,
    returning the result as a `PairData` instance.

    A single batch of data can be loaded an cut on by changing `batch_size`
    and `batch_start`.

    The cuts used can be partially controlled by the `cnn_cut`,
    `valid_momenta` and `beam_slice_cut`.

    Parameters
    ----------
    path : str
        Path to the .root file to be loaded.
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
    events : PairData
        A PairData instance containing the cut events.
    """

    events = PairData(path, includeBackTrackedMC=True, _nEvents=batch_size, _start=batch_start)

    # Apply cuts:
    n = [["Event selection", "Number of PFOs", "Average PFOs per event", "Number of events", "Percentage of events remaining"]]
    n.append(["no selection", ak.count(events.trueParticlesBT.pdg),ak.count(events.trueParticlesBT.pdg)/ak.count(events.eventNum), ak.count(events.eventNum), 100])

    # Single pi0
    ts = time.time()
    Master.BeamMCFilter(events, returnCopy=False)
    print(f"single pi0 done in {time.time()  - ts}s")
    evt_remaining = ak.count(events.eventNum)
    n.append([ "beam -> pi0 + X", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(evt_remaining/n[1][3]) ])

    # Require diphoton decay from the pi0
    ts = time.time()
    f = Master.Pi0TwoBodyDecayMask(events)
    events.Filter([f], [f])
    print(f"pi+ beam done in {time.time()  - ts}s")
    evt_remaining = ak.count(events.eventNum)
    n.append(["diphoton decay", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(evt_remaining/n[1][3]) ])

    # Require a beam particle to exist. ETA ~15s:
    ts = time.time()
    events.ApplyBeamFilter() # apply beam filter if possible
    print(f"beam done in {time.time()  - ts}s")
    evt_remaining = ak.count(events.eventNum)
    n.append([ "beam", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(evt_remaining/n[1][3]) ])

    # Require pi+ beam. ETA ~10s
    ts = time.time()
    true_beam = events.trueParticlesBT.pdg[events.recoParticles.beam_number == events.recoParticles.number]
    f = ak.all(true_beam == 211, -1)
    events.Filter([f], [f])
    del(true_beam)
    print(f"pi+ beam done in {time.time()  - ts}s")
    evt_remaining = ak.count(events.eventNum)
    n.append(["pi+ beam", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(evt_remaining/n[1][3]) ])

    # # Only look at PFOs with > 50 hits. ETA ~30s:
    # ts = time.time()
    # events.Filter([events.recoParticles.nHits > 50], [])
    # del_prop(events.recoParticles, "nHits")
    # print(f"Hits > 50 done in {time.time()  - ts}s")
    # evt_remaining = ak.count(events.eventNum)
    # n.append(["nHits >= 51", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(evt_remaining/n[1][3]) ])

    if beam_slice_cut:
        slice_mask = [[]] * ak.num(events.recoParticles.beam_number, axis=0)
        ts = time.time()
        for i in range(ak.num(slice_mask, axis=0)):
            slices = events.recoParticles.sliceID[i]
            beam_slice = slices[events.recoParticles.number[i] == events.recoParticles.beam_number[i]]
            slice_mask[i] = list(slices == beam_slice)
        slice_mask = ak.Array(slice_mask)
        
        events.Filter([slice_mask], [])
        print(f"Beam slice done in {time.time()  - ts}s")
        evt_remaining = ak.count(events.eventNum)
        n.append(["Beam slice", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(evt_remaining/n[1][3]) ])

    if cnn_cut:
        # Take CNNScore > 0.36. ETA ~15s:
        cnn_cut = 0.36
        ts = time.time()
        events.Filter([events.recoParticles.cnnScore > cnn_cut], [])
        print(f"CNNScore > 0.36 done in {time.time()  - ts}s")
        evt_remaining = ak.count(events.eventNum)
        n.append([f"CNNScore > {cnn_cut}", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(evt_remaining/n[1][3]) ])

    if valid_momenta:
        ts = time.time()
        reco_momenta=events.recoParticles.momentum
        events.Filter([np.logical_and(np.logical_and(reco_momenta.x != -999., reco_momenta.y != -999.), reco_momenta.z != -999.)], [])
        print(f"Valid momenta done in {time.time()  - ts}s")
        evt_remaining = ak.count(events.eventNum)
        n.append(["Valid momenta", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(evt_remaining/n[1][3]) ])

    # Require >= 2 PFOs. ETA ~90s:
    ts = time.time()
    f = Master.NPFPMask(events, -1)
    events.Filter([f], [f])
    print(f"PFOs >= 2 done in {time.time()  - ts}s")
    evt_remaining = ak.count(events.eventNum)
    n.append(["nPFP >= 2", ak.count(events.trueParticlesBT.pdg), ak.count(events.trueParticlesBT.pdg)/evt_remaining, evt_remaining, 100*(evt_remaining/n[1][3]) ])
    

    print(pd.DataFrame(n[1:], columns = n[0]))
    # print("\n".join([str(l) for l in n]))
    
    plt.close()

    return events


######################################################################################################
######################################################################################################
##########                                PLOTTING FUCNTIONS                                ##########
######################################################################################################
######################################################################################################


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
        Any additional keyword arguments to be passed to the `plt.hist`
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
        inc_stacked=True, inc_norm=True, inc_log=True,
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
    inc_stacked : bool
        Whether to include a stacked histogram plot. Default is False.
    inc_norm : bool
        Whether to include a normalised histogram. Default is True.
    inc_log : bool
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
        Any additional keyword arguments to be passed to the `plt.hist`
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
        if weights is None:
            weights = [{2:None, 1:None, 0:None}] * len(range)
        else:
            weights = [{2:ak.ravel(weights[sig_count == 2]), 1:ak.ravel(weights[sig_count == 1]), 0:ak.ravel(weights[sig_count == 0])}] * len(range)
    else:
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
            plt.figure(figsize=(12,9))
            plt.hist(ak.ravel(property),                 label="0 signal", bins = bins[i], weights=weights[i][2], color="C2", **kwargs)
            plt.hist(ak.ravel(property[sig_count != 0]), label="1 signal", bins = bins[i], weights=weights[i][1], color="C1", **kwargs)
            plt.hist(ak.ravel(property[sig_count == 2]), label="2 signal", bins = bins[i], weights=weights[i][0], color="C0", **kwargs)
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
            plt.ylabel("Count/" + bin_size)
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
            plt.ylabel("Count/" + bin_size)
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
        Any additional keyword arguments to be passed to the `plt.hist`
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


######################################################################################################
######################################################################################################
##########                              NOT PLOTTING FUCNTIONS                              ##########
######################################################################################################
######################################################################################################


def get_mother_pdgs(events):
    # ETA ~30mins:
    # This loops through every event and assigned the pdg code of the mother.
    # Assigns 211 (pi+) to daughters of the beam, and 0 to everything which is not found
    mother_ids = events.trueParticlesBT.mother
    truth_ids = events.trueParticles.number
    truth_pdgs = events.trueParticles.pdg
    mother_pdgs = mother_ids.to_list()
    ts = time.time()
    for i in range(ak.count(events.eventNum)):
        true_pdg_lookup = {truth_ids[i][d]:truth_pdgs[i][d] for d in range(ak.count(truth_ids[i]))}
        true_pdg_lookup.update({0:0, 1:211})
        for j in range(ak.count(mother_ids[i])):
            try:
                mother_pdgs[i][j] = true_pdg_lookup[mother_ids[i][j]]
            except:
                mother_pdgs[i][j] = 0

    print(f"Mother PDG codes found in {time.time()  - ts}s")
    mother_pdgs = ak.Array(mother_pdgs)
    return mother_pdgs

# Number of signals in a pair
def paired_sig_count(events, pair_coords, mother_pdgs):
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


# Hopefully this is correct, not tested yet though!
# I fear it may require iterating through events to calculate though
def get_beam_impact_point(direction, start_pos, beam_vertex):
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

    return vector.magnitude(d)

def paired_mass(events, pair_coords):
    """
    Finds the mass of the pairs given by `pair_coords` from `events`.

    Parameters
    ----------
    events : PairData
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
    events : PairData
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
    events : PairData
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    energy : ak.Array
        Summed energies of the pairs.
    """
    # Get the momenta via the pair indicies
    first_mom = events.recoParticles.momentum[pair_coords["0"]]
    second_mom = events.recoParticles.momentum[pair_coords["1"]]

    # Calculate
    return vector.magnitude(first_mom) + vector.magnitude(second_mom)

def paired_closest_approach(events, pair_coords):
    """
    Finds the closest approaches of the pairs given by `pair_coords` from `events`.

    Parameters
    ----------
    events : PairData
        Events from which the pairs are drawn.
    pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
        Inidicies to construct the pairs.

    Returns
    -------
    closest_approach : ak.Array
        Distance of closest approach of the pairs.
    """
    # Get the momenta via the pair indicies
    first_dir = events.recoParticles.direction[pair_coords["0"]]
    first_pos = events.recoParticles.startPos[pair_coords["0"]]
    second_dir = events.recoParticles.direction[pair_coords["1"]]
    second_pos = events.recoParticles.startPos[pair_coords["1"]]

    return closest_approach(first_dir,second_dir,first_pos,second_pos)

# N.B. this is phi in Shyam's scheme
def paired_opening_angle(events, pair_coords):
    """
    Finds the opening angles of the pairs given by `pair_coords` from `events`.

    Parameters
    ----------
    events : PairData
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

if __name__ == "__main__":

    evts = load_and_cut_data_two_photon("/scratch/wx21978/pi0/root_files/6GeV_beam_v1/Prod4a_6GeV_BeamSim_00.root", batch_size = 8000, batch_start = 0)

    # print("Locals:")
    # local_vars = list(locals().items())
    # for var, obj in local_vars:
    #     print(f"{var}: {sys.getsizeof(obj)}")
    # print("Globals:")
    # global_vars = list(globals().items())
    # for var, obj in global_vars:
    #     print(f"{var}: {sys.getsizeof(obj)}")

    # Basic method: We come up with a set of pairs.
    # Argcombinations give the results as indicies, so we don't have to worry about
    # making sure combinations is consistent  between calls.
    # Now we just need to use these pair indicies to locate our pairs and do stuff with them
    @profile
    def get_pair_coords():
        return ak.argcombinations(evts.recoParticles.number, 2)
    # pair_coords = ak.argcombinations(evts.recoParticles.number, 2)
    pair_coords = get_pair_coords

    mother_pdgs = get_mother_pdgs(evts)

    print("\nPlotting nHits...")
    simple_sig_bkg_hist(
        "nHits", "Count", evts.recoParticles.nHits, np.logical_and(evts.trueParticlesBT.pdg == 22, mother_pdgs == 111),
        range=[None, 100], bins = [200, 100], histtype='step', density=True
    )
    del_prop(evts.recoParticles, "nHits")

    sig_count = paired_sig_count(evts, pair_coords, mother_pdgs)
    del mother_pdgs

    print("Plotting masses...")
    # masses = paired_mass(evts, pair_coords)
    plot_pair_hists("mass", "MeV(?)", paired_mass(evts, pair_coords), sig_count, range=[None, 1000, 100], bins=[1000, 1000, 200])

    print("Plotting momenta...")
    # momentum = paired_momentum(evts, pair_coords)
    # mom_mag = vector.magnitude(momentum)
    plot_pair_hists("momentum", "MeV(?)", vector.magnitude(paired_momentum(evts, pair_coords)), sig_count, range=[None, 4000, 200], bins=[1000,1000,200])
    
    # TODO Energies go -ve??
    print("Plotting energies...")
    # energies = paired_energy(evts, pair_coords)
    plot_pair_hists("energy", "MeV(?)", paired_energy(evts, pair_coords), sig_count, range=[None, 4000, 200], bins=[1000,1000,200])

    print("Plotting cloest approaches...")
    # approaches = paired_closest_approach(evts, pair_coords)
    plot_pair_hists("closest approach", "cm", paired_closest_approach(evts, pair_coords), sig_count, range=[None, (-1000, 500)], bins=[100,150])

    print("Plotting opening angles...")
    angles = paired_opening_angle(evts, pair_coords)
    plot_pair_hists("angle", "rad", angles, sig_count, range=None, bins=100, bin_size="(pi/50)rad")

    # Weight the bin widths to keep then with equal area on a sphere
    # Area cover over angle dTheta is r sin(Theta) dTheta (with r=1)
    # So we need constant sin(Theta) dTheta
    # In the range Theta = [0, pi), we have
    # \int^\pi_0 sin(\theta) d\theta = 2
    # So for 100 bins, we need: sin(Theta) dTheta = 2/100 = 0.02
    # \int^{\theta_new}_{\theta_old} sin(\theta) d\theta = 2/100
    # So 0.2 = cons(theta_old) - cos(theta_new)
    n_bins = 100
    bins = np.zeros(n_bins+1)
    for i in range(n_bins):
        bins[i+1] = np.arccos(np.max([np.cos(bins[i]) - 2/n_bins, -1]))
    
    print("Plotting opening angles in bins of equal arcradians...")
    # TODO Need to fix the normailisation, currently it's not working!
    plot_pair_hists("angle", "rad", angles, sig_count, range=None, bins=bins, bin_size="(pi/25)arcrad", unique_save_id = "_sphere")
    del angles

    print("All plots made.")