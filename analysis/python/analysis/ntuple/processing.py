from argparse import Namespace

import awkward as ak
import numpy as np

from python.analysis import processing, vector
from python.analysis.common import timer
from . import ntuple

def calculate_batches(args : Namespace) -> dict[str, any]:
    """ Calculate the number of events to process in eacn batch.

    Args:
        args (Namespace): Application Arguments.

    Returns:
        dict[str, any]: Multiprocessing arguments.
    """
    if "data" in args.ntuple_files:
        n_data = [ntuple.file_len(file_desc.file) for file_desc in args.ntuple_files["data"]]
    else:
        n_data = []

    if len(n_data) == 0:
        print("no data file was specified, 'normalisation', 'beam_reweight', 'toy_parameters' and 'analyse' will not run")

    n_mc = [ntuple.file_len(file_desc.file) for file_desc in args.ntuple_files["mc"]] # must have MC

    processing_args = {"events" : None, "batches" : None, "threads" : args.cpus}

    # pass multiprocessing args
    # if max([*n_data, *n_mc]) >= 7E5:
    #     processing_args["events"] = None
    #     processing_args["batches"] = int(2 * max([*n_data, *n_mc]) // 7E5)
    #     processing_args["threads"] = args.cpus

    return processing_args


def run_process(ntuple_files : list[ntuple.FileDescriptor], is_data : bool, args : Namespace, func : callable, merge : bool = True) -> list[dict] | dict:
    """ Run a process on Ntuple files.

    Args:
        ntuple_files (list[FileDescriptor]): Ntuple files.
        is_data (bool): Whether the Ntuple files are Data or MC.
        args (Namespace): Application arguments.
        func (callable): Process to run.
        merge (bool, optional): Whether to merge the outputs of the processing. Defaults to True.

    Returns:
        list[dict] | dict: Output of the processing.
    """
    func_args = vars(args)
    func_args["data"] = is_data
    output = processing.mutliprocess(func, ntuple_files, args.batches, args.events, func_args, args.threads)
    if merge:
        output = processing.merge_outputs(output)
    return output

@timer
def select_sample(events: ntuple.Data, nDaughters: int, merge: bool = False, backtracked: bool = False, cheatMerging: bool = False) -> ntuple.Data:
    """ Applies MC matching and shower merging to events with
        specificed number of objects

    Args:
        events (ntuple.Data): events to look at
        nDaughters (int): number of objects per event
        merge (bool, optional): should we do shower merging?. Defaults to False.

    Returns:
        ntuple.Data: selected sample
    """
    valid = pi0_mc_mask(events, nDaughters)  # get mask of events
    filtered = events.filter([valid], [valid], True)  # filter events with mask

    if backtracked == False:
        matched, unmatched, selection = filtered.mc_matching(apply_filters=False)
        filtered.filter([selection], [selection])  # apply the selection
    else:
        singleMatchedEvents = filtered.true_particles_bt.single_match
        filtered.filter([singleMatchedEvents], [singleMatchedEvents])
        best_match, selection = filtered.match_by_angle_bt()
        filtered.filter([selection], [selection])

    if merge is True and backtracked is False:
        filtered = filtered.mergeShower(
            filtered, matched[selection], unmatched[selection], 1, False)

    if merge is True and backtracked is True:
        if cheatMerging is True:
            filtered, null = filtered.merge_pfo_cheat()
            filtered.filter([null], [null])
        else:
            filtered = filtered.merge_shower_bt(best_match[selection])

    # if we don't merge showers, just get the showers matched to MC
    if merge is False and backtracked is False:
        filtered.filter([matched[selection]])
    if merge is False and backtracked is True:
        filtered.filter([best_match[selection]])
    return filtered


def n_pfp_mask(events: ntuple.Data, n_objects: int = None) -> ak.Array:
    """ Create a boolean mask of events with a specific number of objects.

    Args:
        events (ntuple.Data): events to filter
        n_objects (int, optional): keep events with specific number of daughters. Defaults to None.

    Returns:
        ak.Array: mask of events to filter
    """
    null_dir = events.recoParticles.shower_direction.x != -999
    null_pos = events.recoParticles.shower_start_pos.x != -999
    nObj = np.logical_and(null_dir, null_pos)
    # get number of showers which have a valid direction
    nObj = ak.num(nObj[nObj])

    if n_objects == None:
        r_mask = nObj > 1
    elif n_objects < 0:
        r_mask = nObj > abs(n_objects)
    else:
        r_mask = nObj == n_objects
    return r_mask


def pi0_two_body_decay_mask(events: ntuple.Data) -> ak.Array:
    """ Create boolean mask of events where pi0's have decayed in their primary decay mode
        i.e. pi0 -> gamma gamma

    Args:
        events (ntuple.Data): events to filter

    Returns:
        ak.Array: mask of events to filter
    """
    mask = ak.num(
        events.true_particles.true_photon_mask[events.true_particles.true_photon_mask], -1) == 2  # exclude pi0 -> e+ + e- + photons
    print(f"number of dalitz decays: {ak.count(mask[np.logical_not(mask)])}")
    return mask


@timer
def pi0_mc_mask(events: ntuple.Data, n_objects: int = None) -> ak.Array:
    """ A filter for Pi0 MC dataset, selects events with a specific number of objects
        which have a valid direction vector and removes events with the 3 body pi0 decay.

    Args:
        events (ntuple.Data): events being studied
        n_objects (int): keep events with specific number of daughters. Defaults to None

    Returns:
        ak.Array: mask of events to filter
    """
    r_mask = n_pfp_mask(events, n_objects)
    t_mask = pi0_two_body_decay_mask(events)
    valid = np.logical_and(r_mask, t_mask)
    return valid


def fractional_error(reco: ak.Array, true: ak.Array, null: ak.Array) -> tuple:
    """ Calcuate fractional error, filter null data and format data for plotting.

    Args:
        reco (ak.Array): reconstructed quantity
        true (ak.Array): true quantity
        null (ak.Array): mask for events without shower pairs, reco direction or energy

    Returns:
        tuple of np.array: flattened numpy array of errors, reco and truth
    """
    true = true[null]
    true = ak.where(ak.num(true, 1) > 0, true, np.nan)
    reco = ak.flatten(reco, 1)[null]
    print(f"number of reco pairs: {len(reco)}")
    print(f"number of true pairs: {len(true)}")
    error = (reco / true) - 1
    return ak.to_numpy(ak.ravel(error)), ak.to_numpy(ak.ravel(reco)), ak.to_numpy(ak.ravel(true))


@timer
def calculate_quantities(events: ntuple.Data, backtracked_truth: bool = False) -> tuple:
    """ Calcaulte reco/ true quantities of shower pairs, and format them for plotting

    Args:
        events (ntuple.Data): events to look at
        names (str): quantity names
        backtracked_truth (bool): calculate truth quantities from backtracked truth

    Returns:
        tuple of np.arrays: quantities to plot
    """
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    if backtracked_truth is True:
        mct = events.true_particles_bt.calculate_pair_quantities()
    else:
        mct = events.true_particles.calculate_pair_quantities()
    rmc = events.recoParticles.calculate_pair_quantities(use_bt=True)

    # keep track of events with no shower pairs
    null = ak.flatten(rmc[-1], -1)
    null = ak.num(null, 1) > 0

    error = []
    reco = []
    true = []
    for i in range(len(names)):
        print(names[i])
        e, r, t = fractional_error(rmc[i], mct[i], null)
        error.append(e)
        reco.append(r)
        true.append(t)

    error = np.nan_to_num(error, nan=-999)
    reco = np.nan_to_num(reco, nan=-999)
    true = np.nan_to_num(true, nan=-999)
    return true, reco, error


def shower_merge_performance(events: ntuple.Data, best_match: ak.Array) -> float:
    """ Checks the fraction of correctly merged showers. 
        Does so by checking if the showers to merge have the 
        same true particle number as the target shower.

    Args:
        events (ntuple.Data): events to study
        best_match (ak.Array): target PFP's to merge to.
    """
    if bool(ak.all(ak.num(events.recoParticles.shower_energy[events.recoParticles.shower_energy != -999]) == 2)) is True:
        counts = 0
    else:
        counts = 1
    index = ak.local_index(events.recoParticles.shower_energy)
    to_merge = [ak.where(index == best_match[:, i], False, True)
                for i in range(2)]
    to_merge = np.logical_and(*to_merge)

    # * calculate angle between each best matched shower and the showers to merge
    best_match_dir = events.recoParticles.shower_direction[best_match]
    to_merge_dir = events.recoParticles.shower_direction[to_merge]
    not_null = to_merge_dir.x != -999
    to_merge_dir = to_merge_dir[not_null]

    angles = [ak.unflatten(vector.angle(
        to_merge_dir, best_match_dir[:, i]), counts, -1) for i in range(2)]
    angles = ak.concatenate(angles, -1)
    # check which angle is the smallest i.e. decide which main shower to merge to
    min_mask_angle = ak.min(angles, -1) == angles

    index_to_merge = [(index[to_merge])[min_mask_angle[:, :, i]]
                      for i in range(2)]

    target_particle_number = events.true_particles_bt.partice_number[best_match]

    correct_merge = [events.true_particles_bt.partice_number[index_to_merge[i]]
                     == target_particle_number[:, i] for i in range(2)]

    correctly_merged = ak.count(
        correct_merge[0][correct_merge[0]]) + ak.count(correct_merge[1][correct_merge[1]])
    total = ak.count(correct_merge[0]) + ak.count(correct_merge[1])
    if total > 0:
        correct_merge = (correctly_merged/total) * 100
    else:
        correct_merge = 100
    print(f"nPFP's correctly merged: {correct_merge:.2f}%")
    return correct_merge
