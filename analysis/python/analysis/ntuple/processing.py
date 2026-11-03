from argparse import Namespace

import awkward as ak
import numpy as np

from python.analysis import processing, vector
from python.analysis.common import timer
from . import ntuple

def calculate_batches(args : Namespace) -> dict[str, any]:
    """Build the multiprocessing options from the application arguments.

        Args:
            args (Namespace): Parsed application arguments containing the input.
                Ntuple files and CPU count. The current implementation leaves
                event and batch limits unset and uses ``args.cpus`` as threads.

        Returns:
            dict[str, any]: Processing options with ``events`` and ``batches``
                set to ``None`` and ``threads`` set to the requested CPU count.
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
    """Run a processing function over ntuple files and optionally merge its outputs.

        Args:
            ntuple_files (list[ntuple.FileDescriptor]): Ntuple files.
            is_data (bool): Whether the Ntuple files are Data or MC.
            args (Namespace): Application arguments.
            func (callable): Process to run.
            merge (bool): Whether to merge the outputs of the processing. Defaults to True.

        Returns:
            list[dict] | dict: Per-file processing outputs when ``merge`` is.
                False, or the dictionary returned by
                ``processing.merge_outputs`` when ``merge`` is true.
    """
    func_args = vars(args)
    func_args["data"] = is_data
    output = processing.mutliprocess(func, ntuple_files, args.batches, args.events, func_args, args.threads)
    if merge:
        output = processing.merge_outputs(output)
    return output

@timer
def select_sample(events: ntuple.Data, nDaughters: int, merge: bool = False, backtracked: bool = False, cheatMerging: bool = False) -> ntuple.Data:
    """Select pi0 MC events and optionally match and merge their showers.

        Args:
            events (ntuple.Data): Event sample to select and process.
            nDaughters (int): Required number of reconstructed PFPs, passed to.
                The pi0 selection mask.
            merge (bool): Whether to merge matched showers. Defaults to False.
            backtracked (bool): Whether to match and merge using backtracked.
                Truth information instead of the ordinary MC matching.
            cheatMerging (bool): When ``backtracked`` and ``merge`` are true,
                whether to merge PFPs using their shared backtracked particle
                identity rather than angular matching.

        Returns:
            ntuple.Data: A filtered event sample. With ``merge=False``, only.
                Matched PFPs remain; with ``merge=True``, matched or merged
                shower data are returned.
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
    """Select events by the number of reconstructed showers with valid geometry.

        Args:
            events (ntuple.Data): Sample whose reconstructed shower directions.
                And start positions are checked for the ``-999`` null sentinel.
            n_objects (int): Exact valid-shower count to require. A negative.
                Value requires more than its absolute value; ``None`` requires
                more than one. Defaults to None.

        Returns:
            ak.Array: Boolean event mask indicating whether each event meets.
                The requested valid-shower count.
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
    """Select events whose primary pi0 daughters contain exactly two photons.

        Args:
            events (ntuple.Data): Sample providing the true-photon mask for.
                Each event.

        Returns:
            ak.Array: Boolean event mask that is true when the selected primary.
                Pi0 decay has two photons, excluding three-body Dalitz decays.
    """
    mask = ak.num(
        events.true_particles.true_photon_mask[events.true_particles.true_photon_mask], -1) == 2  # exclude pi0 -> e+ + e- + photons
    print(f"number of dalitz decays: {ak.count(mask[np.logical_not(mask)])}")
    return mask


@timer
def pi0_mc_mask(events: ntuple.Data, n_objects: int = None) -> ak.Array:
    """Select pi0 MC events by valid reconstructed showers and two-photon decay.

        Args:
            events (ntuple.Data): Sample to test for valid shower geometry and.
                The primary pi0 two-photon decay mode.
            n_objects (int): Shower-count requirement passed to.
                :func:`n_pfp_mask`; negative values require a minimum count.
                Defaults to None.

        Returns:
            ak.Array: Boolean mask combining the reconstructed-shower count.
                And true two-photon decay selections.
    """
    r_mask = n_pfp_mask(events, n_objects)
    t_mask = pi0_two_body_decay_mask(events)
    valid = np.logical_and(r_mask, t_mask)
    return valid


def fractional_error(reco: ak.Array, true: ak.Array, null: ak.Array) -> tuple:
    """Select events by mask and flatten reconstructed-to-true fractional errors.

        Args:
            reco (ak.Array): Reconstructed quantity array, flattened over its.
                Pair axis before event selection.
            true (ak.Array): Corresponding truth quantity array; empty selected.
                Events are replaced with ``nan`` before division.
            null (ak.Array): Boolean event-selection mask applied to both.
                Reconstructed and truth arrays before flattening.

        Returns:
            tuple: Three flattened NumPy arrays in the order fractional error.
                ``(reco / true) - 1``, reconstructed value, and truth value.
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
    """Calculate truth, reconstructed, and fractional-error shower-pair quantities.

        Args:
            events (ntuple.Data): Selected events containing reconstructed and.
                Truth shower-pair information.
            backtracked_truth (bool): Use backtracked truth pair quantities.
                Instead of the ordinary true-particle quantities.

        Returns:
            tuple: Three lists of NumPy arrays in the order truth values,
                reconstructed values, and fractional errors. Each list follows
                ``inv_mass``, ``angle``, leading energy, subleading energy, and
                pi0 momentum order; invalid values are replaced with ``-999``.
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
    """Measure the percentage of merged PFPs assigned to the correct truth particle.

        Args:
            events (ntuple.Data): Events with reconstructed shower and.
                Backtracked particle information after merging.
            best_match (ak.Array): Per-event indices of the two target PFPs;
                other valid PFPs are associated with the target whose shower
                direction has the smaller angular separation.

        Returns:
            float: Percentage of associated PFPs whose backtracked particle.
                Number matches that of the corresponding target PFP. Returns
                ``100`` when there are no PFPs to assess.
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
