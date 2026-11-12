#! Not used anywhere, should delete.

#!/usr/bin/env python

import numpy as np
import awkward as ak

from python.analysis import vector, selection

#######################################################################
#######################################################################
##########                  PAIR SELECTION                   ##########
#######################################################################
#######################################################################


def get_MC_truth_beam_mask(event_mothers, event_ids, beam_id=1):
    """ Create a mask for particles descended from the beam particle.

        Args:
            event_mothers (ak.Array): Mother-particle IDs for backtracked particles in one event.
            event_ids (ak.Array): Truth-particle IDs for the corresponding backtracked particles.
            beam_id (int): Truth-particle ID of the beam particle.

        Returns:
            list[bool]: Mask selecting particles whose ancestor chain includes the beam.
    """
    # IDs for particles related to the beam
    beam_ids = np.array([beam_id])
    # IDs of particles which correspond added compnents to beam_ids
    new_ids = event_ids[event_mothers == beam_ids[0]]

    while len(new_ids) != 0:
        # Add new ids into the beam related ids
        beam_ids = np.append(beam_ids, new_ids)
        # Loop through events and create a mask of daughters of the
        # newly added ids. Could be made more efficient by somehow
        # removing PFOs that already exist in the beam_ids list,
        # another mask?
        mask = [e in new_ids for e in event_mothers]
        # Flag the newly added PFOs as "to be added" and loop, unless
        # no new PFOs are found
        new_ids = event_ids[mask]
    return [e in beam_ids for e in event_ids]


def get_best_pairs(
    events,
    # truth_photon_ids,
    method='mom',
    return_type="mask",
    valid_mom_cut=False,
    report=False,
    verbosity=0
):
    """ Select the best reconstructed pair matched to each event's two decay photons.

        Args:
            events (Data): Event collection containing reconstructed and truth particles.
            method (str | list[str]): Matching method(s): ``"mom"``, ``"energy"``, ``"dir"``, ``"purity"``, ``"completeness"``, or ``"all"``.
            return_type (str): Pair representation: ``"mask"``, ``"id"``, or ``"index"``; other values use indices.
            valid_mom_cut (bool): Whether to ignore PFOs with invalid momentum components.
            report (bool): Whether to print counts of events missing one or both photon matches.
            verbosity (int): Debug-output level from 0 (quiet) through 6 (most detailed).

        Returns:
            tuple: Selected pairs (a mask, index array, or method-keyed dictionary) and a boolean event-validity mask.
    """
    # Work out the method of determining the best pair to use. The
    # dictionary allows for aliasing of methods, but currently this is
    # removed to avoid potential confusion
    known_methods = {
        # "momentum"      : "mom",
        "mom": "mom",
        # "e"             : "energy",
        "energy": "energy",
        # "d"             : "dir",
        "dir": "dir",
        # "direction"     : "dir",
        "purity": "purity",
        "completeness": "completeness",
        # "comp"          : "completeness",
        "all": "all"
    }
    if not isinstance(method, list):
        method = [method]
    bad_methods = []
    for i, m in enumerate(method):
        try:
            method[i] = known_methods[m]
        except(KeyError):
            bad_methods += [m]
            print(f'Method(s) "{m}" not found, please use one of:\nmomentum, '
                  + 'energy, direction, purity, completeness')
    if len(bad_methods) != 0:
        join_str = '", "'
        raise ValueError(f'Method(s): "{join_str.join(bad_methods)}"\nnot '
                         + 'found, please use:\n"mom", "energy", "dir", '
                         + '"purity", "completeness", or "all"')

    # Parse which methods to use
    if "all" in method:
        methods_to_use = ["mom", "energy", "dir", "purity", "completeness"]
    else:
        methods_to_use = method

    # Work out the number of events we have
    num_events = ak.num(events.true_particles_bt.number, axis=0)

    # Currently only worrying about single pi0->yy in each event. In
    # future, we could extent this by allowing `truth_photon_ids` as an
    # argument, and then iterating over each photon in each event, but
    # that's a later problem!
    truth_photon_ids = truth_pfos_in_two_photon_decay(
        events, sort=return_type != "mask")

    # Definitions of what each method involves
    #   - testing_methods: The test to be performed between the true
    #     and reco data to get the pair ordering
    #   - reco_props: The reco data of the candidate PFOs
    #   - truth_props: The truth data to be tested against

    testing_methods = {
        "mom": lambda reco, true: np.argsort(
            [vector.dot(r, true)[0] for r in reco]),
        "dir": lambda reco, true: np.argsort(
            [vector.dot(r, true)[0] for r in reco]),
        "energy": lambda reco, true: np.argsort(reco),
        "purity": lambda reco, true: np.argsort(reco),
        "completeness": lambda reco, true: np.argsort(reco)
    }
    # TODO Swap to lambda functions for we don't fetch these unless we
    # actually need them
    reco_props = {
        "mom": events.recoParticles.shower_momentum,
        "dir": events.recoParticles.shower_direction,
        "energy": events.recoParticles.shower_energy,
        "purity": events.true_particles_bt.purity,
        "completeness": events.true_particles_bt.completeness
    }
    # TODO Change energy ... to use a smaller zeros/like
    # (events.true_particles.number ?)
    true_props = {
        "mom": events.true_particles.momentum,
        "dir": events.true_particles.shower_direction,
        "energy": ak.zeros_like(events.true_particles.number),
        "purity": ak.zeros_like(events.true_particles.number),
        "completeness": ak.zeros_like(events.true_particles.number)
    }

    # Other properties which must be set up ahead of the loop
    valid_event_mask = np.full(num_events, True, dtype=bool)

    truth_indicies_photon_beam = {}
    if return_type == "mask":
        for m in methods_to_use:
            truth_indicies_photon_beam.update({m: [[]] * num_events})
    else:
        # This is only valid for the 1 pi0 in event, 2 photon decay cut
        for m in methods_to_use:
            truth_indicies_photon_beam.update(
                {m: np.zeros((num_events, 2), dtype=int)})

    zero_count = 0
    one_count = 0

    # Loop over each event to determine the best pair
    for i in range(num_events):
        bt_ids = events.true_particles_bt.number[i]
        indicies = np.arange(len(bt_ids))

        true_ids = events.true_particles.number[i]

        # Sorted ids should be supplied as an argument generated by
        # truth_pfos_in_two_photon_decay(evts)
        photon_i, photon_ii = truth_photon_ids[i]

        if verbosity >= 1:
            true_momenta = events.true_particles.momentum[i]
            if verbosity >= 2:
                print("\nTrue particle energies (GeV)")
                print(vector.magnitude(true_momenta[true_ids == photon_i]))
                print(vector.magnitude(true_momenta[true_ids == photon_ii]))
                print("True particle directions")
                print(vector.normalize(true_momenta[true_ids == photon_i]))
                print(vector.normalize(true_momenta[true_ids == photon_ii]))
            else:
                print(true_momenta[true_ids == photon_i])
                print(true_momenta[true_ids == photon_ii])

        if valid_mom_cut:
            # Maybe want to look at what happens when we use
            # purity/completeness with no good data cut?
            reco_momenta = events.recoParticles.shower_momentum[i]
            good_data = np.logical_and(
                np.logical_and(
                    reco_momenta.x != -999., reco_momenta.y != -999.),
                reco_momenta.z != -999.)
        else:
            good_data = slice(None)

        # Count how many events are cut with no photons/only one photon
        # having at least one daughter
        photon_i_exists = photon_i in bt_ids[good_data]
        photon_ii_exists = photon_ii in bt_ids[good_data]
        if (not photon_i_exists) or (not photon_ii_exists):
            if not (photon_i_exists or photon_ii_exists):
                zero_count += 1
            else:
                one_count += 1
            valid_event_mask[i] = False
            # Skips to next iteration if not both phhotons have
            # daughters
            continue

        for m in methods_to_use:
            # Get the data to use
            reco_prop = reco_props[m][i]
            true_prop = true_props[m][i]
            # Get the truth property of each photon
            true_prop_i = true_prop[true_ids == photon_i]
            true_prop_ii = true_prop[true_ids == photon_ii]
            # Get a mask indicating the daughters of each photon in the
            # reco data
            photon_i_mask = [bt_ids[good_data] == photon_i][0]
            photon_ii_mask = [bt_ids[good_data] == photon_ii][0]
            # Need the [0] because bt_ids[good_data] == photon_i is an
            # array, so returns an array (one element)

            # Order the PFOs by the selected method
            reco_prop_ordering_i = testing_methods[m](
                reco_prop[indicies[good_data][photon_i_mask]], true_prop_i)
            reco_prop_ordering_ii = testing_methods[m](
                reco_prop[indicies[good_data][photon_ii_mask]], true_prop_ii)
            # Returns the index of the PFO selected as the best
            # selection for each photon
            photon_i_index = indicies[good_data][photon_i_mask][
                reco_prop_ordering_i[-1]]
            photon_ii_index = indicies[good_data][photon_ii_mask][
                reco_prop_ordering_ii[-1]]

            if verbosity >= 6:
                print(f"List of reco values {m} with good data:")
                print(*reco_props[m][good_data])
            if verbosity >= 4:
                print(f"Values of {m}:")
                print("Leading photon")
                print(reco_props[indicies[good_data][photon_i_mask]])
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

            # If we are returning IDs, we need to find the
            # corresponding reco ID
            if return_type == "id":
                reco_ids = events.recoParticles.number[i]
                truth_indicies_photon_beam[m][i, :] = [
                    reco_ids[photon_i_index], reco_ids[photon_ii_index]]
            # If returning a mask we need to construct the mask
            elif return_type == "mask":
                event_mask = [False] * (indicies[-1] + 1)
                event_mask[photon_i_index] = True
                event_mask[photon_ii_index] = True
                truth_indicies_photon_beam[m][i] = event_mask
            # Otherwise, we just return the reco indicies of the
            # selected PFOs
            else:
                truth_indicies_photon_beam[m][i, :] = [
                    photon_i_index, photon_ii_index]

    if report:
        print(f"{zero_count} events discarded due to no true photons "
              + "having matched PFOs.")
        print(f"{one_count} events discarded due to only one true photon "
              + "having matched PFOs.")

    if return_type == "mask":
        truth_indicies_photon_beam = ak.Array(truth_indicies_photon_beam)
    if len(methods_to_use) == 1:
        return truth_indicies_photon_beam[method[0]], valid_event_mask
    else:
        return truth_indicies_photon_beam, valid_event_mask


def pair_apply_sig_mask(truth_mask, pair_coords):
    """ Count signal-marked particles in each indexed pair.

        Args:
            truth_mask (ak.Array): Boolean mask identifying signal particles.
            pair_coords (ak.Array): Pair indices stored under the ``"0"`` and ``"1"`` fields.

        Returns:
            ak.Array: Integer signal-particle count for each pair.
    """
    # Convert the mask to integers
    true_counts = np.multiply(truth_mask, 1)

    # Add the results
    return true_counts[pair_coords["0"]] + true_counts[pair_coords["1"]]


def gen_pair_sig_counts(events, pair_coords):
    """ Count photons from pi0 decays in each pair; two denotes distinct daughters of one pi0.

        Args:
            events (Data): Event collection containing backtracked truth particles.
            pair_coords (ak.Array): Pair indices stored under the ``"0"`` and ``"1"`` fields.

        Returns:
            ak.Array: Signal count for each pair, ranging from zero to two.
    """
    photon_from_pions = np.logical_and(
        events.true_particles_bt.pdg == 22,
        events.true_particles_bt.mother_pdg == 111)

    different_daughter = (
        events.true_particles_bt.number[pair_coords["0"]]
        != events.true_particles_bt.number[pair_coords["1"]])
    same_mother = (
        events.true_particles_bt.mother[pair_coords["0"]]
        == events.true_particles_bt.mother[pair_coords["1"]])

    same_mother_and_different_daughter = np.logical_and(same_mother,
                                                        different_daughter)
    del same_mother
    del different_daughter

    both_photons = np.logical_and(photon_from_pions[pair_coords["0"]],
                                  photon_from_pions[pair_coords["1"]])

    photons_form_pi0 = np.logical_and(both_photons,
                                      same_mother_and_different_daughter)
    del same_mother_and_different_daughter
    del both_photons

    at_least_one_photon = np.logical_or(photon_from_pions[pair_coords["0"]],
                                        photon_from_pions[pair_coords["1"]])
    # Add the results
    return (np.multiply(at_least_one_photon, 1)
            + np.multiply(photons_form_pi0, 1))


def get_sig_count(events, pair_coordinates, single_best=False, **kwargs):
    """ Count signal contributions for each pair, optionally using the pair selected by ``get_best_pairs``.

        Args:
            events (Data): Event collection used to classify signal particles.
            pair_coordinates (ak.Array): Pair indices stored under the ``"0"`` and ``"1"`` fields.
            single_best (bool): Whether to select one best signal pair per event.
            **kwargs (dict): Additional arguments forwarded to ``get_best_pairs`` when ``single_best`` is true.

        Returns:
            ak.Array: Signal count for each pair.
    """
    if single_best:
        truth_pair_indicies, valid_events = get_best_pairs(events, **kwargs)

        events.filter([valid_events], [valid_events])
        truth_pair_indicies = truth_pair_indicies[valid_events]
        del valid_events
        return pair_apply_sig_mask(truth_pair_indicies, pair_coordinates)
    else:
        return gen_pair_sig_counts(events, pair_coordinates)


def pair_photon_counts(events, pair_coords, mother_pdgs):
    """ Count pi0-decay photons among the backtracked particles in each pair.

        Args:
            events (Data): Event collection containing backtracked particle PDG codes.
            pair_coords (ak.Array): Pair indices stored under the ``"0"`` and ``"1"`` fields.
            mother_pdgs (ak.Array): Mother PDG codes for the backtracked particles.

        Returns:
            ak.Array: Number of paired particles identified as photons from pi0 decays.
    """
    true_photons = events.true_particles_bt.pdg == 22
    # Get the locations wehere the pdg is 22 and mother pdg is 111
    first_sigs = np.logical_and(true_photons[pair_coords["0"]],
                                mother_pdgs[pair_coords["0"]] == 111)
    # Multiplying by 1 sets the dtype to be int (1 where True, 0 where
    # False)
    first_sigs = np.multiply(first_sigs, 1)
    # Same for second particle
    second_sigs = np.logical_and(true_photons[pair_coords["1"]],
                                 mother_pdgs[pair_coords["1"]] == 111)
    second_sigs = np.multiply(second_sigs, 1)
    # Add the results
    return first_sigs + second_sigs


#######################################################################
#######################################################################
##########                  PAIR PROPERTIES                  ##########
#######################################################################
#######################################################################


def paired_beam_impact(shower_pairs, type="reco"):
    """ Calculate each pair's impact parameter relative to the beam vertex.

        Args:
            shower_pairs (ShowerPairs): Pair collection providing event data and pair directions.
            type (str): Direction and particle data variant: ``"reco"``, ``"true"``, or ``"cheated"``.

        Returns:
            ak.Array: Impact parameter for each shower pair.
    """
    if type == "reco":
        data = shower_pairs.events.recoParticles
        beam_vertex = data.beam_end_pos
        dir = shower_pairs.reco_direction
    else:
        data = shower_pairs.events.true_particles_bt
        beam_vertex = data.true_beam_vertex
        if type == "cheated":
            dir = shower_pairs.cheated_direction
        else:
            dir = shower_pairs.true_direction
    shared_vertex = shower_pairs.SharedVertex(
        data.shower_momentum[shower_pairs.leading],
        data.shower_momentum[shower_pairs.subleading],
        data.shower_start_pos[shower_pairs.leading],
        data.shower_start_pos[shower_pairs.subleading])
    # Impact parameter between the PFOs and corresponding beam vertex
    return selection.pfo.get_impact_parameter(
        dir, shared_vertex, beam_vertex)


def truth_pfos_in_two_photon_decay(events, sort=True):
    """ Return the truth IDs of beam-descended photons in each event, optionally ordered by energy.

        Args:
            events (Data): Event collection containing truth particle records.
            sort (bool): Whether to order the two photon IDs by descending energy.

        Returns:
            np.ndarray: Array of shape ``(num_events, 2)`` containing the photon IDs.
    """
    num_events = ak.num(events.true_particles_bt.mother, axis=0)

    # This is only valid for the 1 pi0 in event, 2 photon decay cut
    photon_ids = np.zeros((num_events, 2))

    for i in range(num_events):
        truth_mothers = events.true_particles.mother[i]
        truth_ids = events.true_particles.number[i]
        truth_pdgs = events.true_particles.pdg[i]
        truth_energy = events.true_particles.energy[i].to_numpy()

        beam_mask = get_MC_truth_beam_mask(
            truth_mothers, truth_ids, beam_id=1)

        beam_photons_ids = truth_ids[(truth_pdgs == 22) & beam_mask]

        sorted_energies = np.flip(np.argsort(
            truth_energy[(truth_pdgs == 22) & beam_mask])) if sort else [0, 1]
        photon_ids[i, :] = beam_photons_ids[sorted_energies]
    return photon_ids
