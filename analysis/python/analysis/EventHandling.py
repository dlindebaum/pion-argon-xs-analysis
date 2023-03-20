#!/usr/bin/env python

# Imports
from python.analysis import vector
import time
import awkward as ak
import numpy as np
import sys
sys.path.insert(1, '/users/wx21978/projects/pion-phys/pi0-analysis/analysis/')

#######################################################################
#######################################################################
##########                EVENT MANIPULATION                 ##########
#######################################################################
#######################################################################


# def get_mother_pdgs(events):
#     """
#     DEPRECATED: use events.trueParticles(BT).motherPdg

#     Loops through each event (warning: slow) and creates a reco-type
#     array of PDG codes of mother particle. E.g. a photon from a pi0
#     will be assigned a PDG code 111.

#     Mother particle is defined as the mother of the truth particle
#     which the PFO was backtracked to.

#     Beam particle is hard-coded to have a PDG code of 211.

#     Anything which cannot have a mother PDG code assigned is given 0.

#     Future warning: Mother PDG codes are planned to be added into the
#     ntuple data making this function redundant.

#     Parameters
#     ----------
#     events : Data
#         Set of events to produce the mother PDG codes for.

#     Returns
#     -------
#     mother_pdgs : ak.Array
#         PDG code of the mother of the truth particle that backtrackd to
#         the PFO.
#     """
#     # This loops through every event and assigned the pdg code of the
#     # mother. Assigns 211 (pi+) to daughters of the beam, and 0 to
#     # everything which is not found
#     mother_ids = events.trueParticles.mother
#     truth_ids = events.trueParticles.number
#     truth_pdgs = events.trueParticles.pdg
#     mother_pdgs = mother_ids.to_list()
#     ts = time.time()
#     for i in range(ak.num(mother_ids, axis=0)):
#         true_pdg_lookup = {
#             truth_ids[i][d]:
#             truth_pdgs[i][d] for d in range(ak.count(truth_ids[i]))}
#         true_pdg_lookup.update({0: 0, 1: 211})
#         # I have no idea why the hell I did this
#         # # Presumably the beam particle gets a strange pdg code??
#         for j in range(ak.count(mother_ids[i])):
#             try:
#                 mother_pdgs[i][j] = true_pdg_lookup[mother_ids[i][j]]
#             except:
#                 mother_pdgs[i][j] = 0

#     print(f"Mother PDG codes found in {time.time()  - ts}s")
#     mother_pdgs = ak.Array(mother_pdgs)
#     return mother_pdgs


# def get_MC_truth_beam_mask(event_mothers, event_ids, beam_id=1):
#     """
#     Loops through all PFOs in an event and generates a mask which
#     selects all PFOs which have been backtracked to have been produced
#     by the beam particle, or any particles in the daughter tree of the
#     beam particle (i.e. a PFO from a daughter of a daughter of the beam
#     particle is selected).

#     Parameters
#     ----------
#     event_mothers : ak.Array
#         1D array of all mother ids of the backtracked PFOs in a single
#         event.
#     event_ids : ak.Array
#         1D array of all truth particle ids of the backtracked PFOs in a
#         single event.
#     beam_id : int
#         ID code of the truth particle corresponding to the beam
#         particle. Default is 1.

#     Returns
#     -------
#     beam_mask : list
#         List of boolean values which select all beam generated PFOs
#         when used as a mask.
#     """
#     # IDs for particles related to the beam
#     beam_ids = np.array([beam_id])
#     # IDs of particles which correspond added compnents to beam_ids
#     new_ids = event_ids[event_mothers == beam_ids[0]]

#     while len(new_ids) != 0:
#         # Add new ids into the beam related ids
#         beam_ids = np.append(beam_ids, new_ids)
#         # Loop through events and create a mask of daughters of the
#         # newly added ids. Could be made more efficient by somehow
#         # removing PFOs that already exist in the beam_ids list,
#         # another mask?
#         mask = [e in new_ids for e in event_mothers]
#         # Flag the newly added PFOs as "to be added" and loop, unless
#         # no new PFOs are found
#         new_ids = event_ids[mask]
#     return [e in beam_ids for e in event_ids]


# def count_diphoton_decays(events, beam_daughters=True):
#     """
#     Returns the number of truth pi0 particles which decay to yy in each
#     event in `events`.

#     pi0 -> yy

#     Parameters
#     ----------
#     events : Data
#         Events in which to count pi0 occurances.
#     beam_daughters : boolean, optional
#         Whether to only accept a pi0 if it is a daughter of the beam
#         particle. Default is True.

#     Returns
#     -------
#     counts : ak.Array
#         Array containing the number of occurances of pi0 -> yy for each
#         event.
#     """
#     def get_two_count_pi0s(pi0s):
#         pions, counts = np.unique(pi0s, return_counts=True)
#         return pions[counts == 2]

#     if beam_daughters:
#         beam_daughter_filter = events.trueParticles.mother == 1
#     else:
#         beam_daughter_filter = True
#     beam_cadidate_pi0s = events.trueParticles.number[np.logical_and(
#         beam_daughter_filter,
#         events.trueParticles.pdg == 111)]
#     try:
#         pi0_daughters = events.trueParticles.mother_pdg == 111
#     except:
#         pi0_daughters = get_mother_pdgs(events) == 111
#     pi0_photon_mothers = events.trueParticles.mother[np.logical_and(
#         events.trueParticles.pdg == 22,
#         pi0_daughters)]
#     counts = ak.Array(map(
#         lambda pi0s: len(
#             np.intersect1d(
#                 pi0s['beam'],
#                 get_two_count_pi0s(pi0s['photon']))),
#         ak.zip(
#             {"beam": beam_cadidate_pi0s, "photon": pi0_photon_mothers},
#             depth_limit=1)))
#     return counts


# def count_non_beam_charged_pi(events, beam_daughters=True):
#     """
#     Returns the number of truth pi+ particles event in `events`.

#     Parameters
#     ----------
#     events : Data
#         Events in which to count pi+ occurances.
#     beam_daughters : boolean, optional
#         Whether to only accept a pi+ if it is a daughter of the beam
#         particle. Default is True.

#     Returns
#     -------
#     counts : ak.Array
#         Array containing the number of pi+ particles for each event.
#     """
#     if beam_daughters:
#         daughter_truth = events.trueParticles.mother == 1
#     else:
#         daughter_truth = ak.ones_like(events.trueParticles.mother, dtype=bool)
#     non_beam_pi_mask = np.logical_and(
#         daughter_truth,
#         np.logical_and(events.trueParticles.pdg == 211,
#                        events.trueParticles.number != 1))
#     return ak.sum(non_beam_pi_mask, axis=-1)


# def _generate_selection(cut):
#     if isinstance(cut, tuple):
#         if len(cut) == 1:
#             return lambda count: count >= cut[0]
#         elif len(cut) == 2:
#             return lambda count: np.logical_and(
#                 count >= min(cut), count <= max(cut))
#         else:
#             raise ValueError(f"Cut tuple {cut} must contain 1 or 2 values.")
#     elif cut is None:
#         return lambda count: True
#     else:
#         return lambda count: count == cut


# def generate_truth_tags(events, n_pi0, n_pi_charged, beam_daighters=True):
#     """
#     Generates a True/False tag for each event in `events` indicating
#     whether they pass the truth level requirements of `n_pi0` and
#     `n_pi_charged`.

#     `n_pi0` and `n_pi_charged` may be integers, tuples, or None. If
#     integer, only the specified number of occurances is selected. If a
#     tuple of length 1, any events with occurances greater than or equal
#     to the value in the tupled are selected. If a tuple of two values,
#     he number of occurances must be equal to or between the values in
#     the tuple. If None, no cut will be applied.

#     Parameters
#     ----------
#     events : Data
#         Events to be tagged.
#     n_pi0 : None, int, or tuple
#         Required number of pi0s that decay into two photons in an event
#         for the event to pass the tag.
#     n_pi_charged : None, int, or tuple
#         Required number of non-beam pi+ particles in an event for the
#         event to pass the tag.
#     beam_daughters : boolean, optional
#         Whether to only accept a PFO if it is a daughter of the beam
#         particle. Default is True.

#     Returns
#     -------
#     tag : ak.Array
#         Array matching the number of events in `events` containing a
#         boolean of whether each event is selected by the tag.
#     """
#     pi0_cut: function = _generate_selection(n_pi0)
#     pi_charged_cut: function = _generate_selection(n_pi_charged)
#     pi0_count = count_diphoton_decays(events)
#     pi_charged_count = count_non_beam_charged_pi(events)
#     return np.logical_and(pi0_cut(pi0_count),
#                           pi_charged_cut(pi_charged_count))


# def np_to_ak_indicies(indicies):
#     """
#     Takes a numpy array of indicies for slicing and converts them to a
#     format compatible with awkward arrays which selected PFOs in an
#     event base on the index, rather than events themselves based on the
#     index.

#     Parameters
#     ----------
#     indicies: np.ndarray
#         Array of indicies for conversion.

#     Returns
#     -------
#     ak_indicies : ak.Array
#         Array of indicies which selected PFOs when slicing an awkward
#         array.
#     """
#     # 1. Expands the dimensions to ensure you hit one index per event
#     # 2. Convert to list - this is necessary to ensure the final
#     #    awkward array has variable size. Without variable size arrays,
#     #    it tries to gather the event of the index, not the PFO at the
#     #    index in the event.
#     # 3. Convert to awkward array
#     return ak.Array(np.expand_dims(indicies, 1).tolist())


#######################################################################
#######################################################################
##########                   EVENT PAIRING                   ##########
#######################################################################
#######################################################################


# def truth_pfos_in_two_photon_decay(events, sort=True):
#     """
#     Returns the truth IDs of the two photons in each event which come
#     from pi0 -> yy decay. Requires a mask on `events` to select only
#     events which contain exactly pion which decays into two photons.

#     The IDs are return as a (num_events, 2) numpy nparray.

#     The optional `sort` argument will cause the photons to be sorted by
#     energy, such that index 0 of each event contains the leading
#     (higher energy) photon.

#     Parameters
#     ----------
#     events : Data
#         Set of events in which to look for photons.
#     sort : bool, optional
#         Defines whether the photon IDs are sorted by energy or not.
#         Default is True.

#     Returns
#     -------
#     photon_ids : np.ndarray
#         Array containing the truth IDs of the two photons created by
#         pion decay in each event.
#     """
#     num_events = ak.num(events.trueParticlesBT.mother, axis=0)

#     # This is only valid for the 1 pi0 in event, 2 photon decay cut
#     photon_ids = np.zeros((num_events, 2))

#     for i in range(num_events):
#         truth_mothers = events.trueParticles.mother[i]
#         truth_ids = events.trueParticles.number[i]
#         truth_pdgs = events.trueParticles.pdg[i]
#         truth_energy = events.trueParticles.energy[i].to_numpy()

#         beam_mask = get_MC_truth_beam_mask(
#             truth_mothers, truth_ids, beam_id=1)

#         beam_photons_ids = truth_ids[(truth_pdgs == 22) & beam_mask]

#         sorted_energies = np.flip(np.argsort(
#             truth_energy[(truth_pdgs == 22) & beam_mask])) if sort else [0, 1]
#         photon_ids[i, :] = beam_photons_ids[sorted_energies]
#     return photon_ids


# def get_best_pairs(
#     events,
#     # truth_photon_ids,
#     method='mom',
#     return_type="mask",
#     valid_mom_cut=False,
#     report=False,
#     verbosity=0
# ):
#     """
#     Finds the 'best' pair of PFOs in `events` which match the truth
#     photons assuming 1 pi0->yy in each event. The method to select the
#     'best is chosen by `method`.

#     The avaiable methods are as follows:
#     - mom : Momentum method - best PFO is that with the greatest
#     momentum projection along the direction of the true photon.
#     - energy : Energy method - best PFO is that with the largest
#     energy.
#     - dir : Direction method - best PFO is that which is closest
#     aligned to the direction of the true photon.
#     - purity : Purity method - best PFO is that with the highest
#     purity, i.e. is made up of the greatest fraction of hits which
#     comes from the true photon.
#     - completeness : Completeness method - best PFO is that with the
#     highest completeness, i.e. the PFO which contains the greatest
#     number of hits generated by the true photon.

#     Multiple methods may be used simulatenously by passing a list of
#     desired methods. If more than one method is selected, the output
#     PFOs will be contained in a dictionary labelled by the method used.

#     The `return_type` argument allows selection of the format of the
#     output pairs. Available formats are as follows:
#     - mask : Returns a awkward boolean mask which picks out the two
#     best PFOs for each event when applied to reco data in `events`.
#     Note that this results in a loss of information regarding the
#     sorting of values from `truth_photon_ids`.
#     - id : Returns the reco IDs of the PFOs corresponding to the best
#     PFOs in each event as a numpy array.
#     - index : Returns the indicies corresponding to the best matching
#     PFOs in each event as a numpy array.

#     Additionally the function will note any events for which a best
#     pair cannot be created due to one or both of the true photons not
#     having any associated PFOs. A `valid_event_mask` is returned which
#     will select only the events in which each true photon has at least
#     one backtracked PFO when applied to `events`. Additionally the
#     `report` argument will cause the function to print out a notice
#     detailing the number of events dropped.

#     `valid_mom_cut` will cause a cut to be made on objects which have
#     an invalid momentum value (-999., -999., -999.). This is necessary
#     if this cut hasn't yet been applied to `events`, and the method
#     used is one of "mom", "energy", or "dir".

#     Parameters
#     ----------
#     events : Data
#         Set of events for which the pairs should be found.
#     method : {'mom', 'energy', 'dir', 'purity', 'completeness', 'all'} \
# or list, optional
#         Method(s) to use to find the best PFOs. If passed as a list,
#         multiple methods will be used. Default is 'mom'
#     return_type : {'mask', 'id', 'index', str}, optional
#         Format to output the best PFOs. If a non-matching string is
#         passed, 'index' will be used by default. Default is 'mask'.
#     valid_mom_cut : bool, optional
#         Whether to filter out PFOs which have an invalid momentum.
#         Default is False.
#     report : bool, optional
#         Whether to print out two lines detailing how many events were
#         dropped due to lack of backtracked PFOs. Default is False.
#     verbosity : int, optional
#         Controls the amount of information printed at each step for
#         debugging purposes from none at 0, to full at 6. Default is 0.

#     Returns
#     -------
#     truth_indicies_photon_beam : np.ndarray, ak.Array, or dict
#         Best PFOs in each event formatted as specified by
#         `return_type`. If `method` a list, a dictionary of results
#         indexed by the values in `method` is returned.
#     valid_event_mask : np.ndarray
#         Boolean mask which selects events where both truth photons have
#         backtracked PFOs when applied to `events`.
#     """
#     # Work out the method of determining the best pair to use. The
#     # dictionary allows for aliasing of methods, but currently this is
#     # removed to avoid potential confusion
#     known_methods = {
#         # "momentum"      : "mom",
#         "mom": "mom",
#         # "e"             : "energy",
#         "energy": "energy",
#         # "d"             : "dir",
#         "dir": "dir",
#         # "direction"     : "dir",
#         "purity": "purity",
#         "completeness": "completeness",
#         # "comp"          : "completeness",
#         "all": "all"
#     }
#     if not isinstance(method, list):
#         method = [method]
#     bad_methods = []
#     for i, m in enumerate(method):
#         try:
#             method[i] = known_methods[m]
#         except(KeyError):
#             bad_methods += [m]
#             print(f'Method(s) "{m}" not found, please use one of:\nmomentum, '
#                   + 'energy, direction, purity, completeness')
#     if len(bad_methods) != 0:
#         join_str = '", "'
#         raise ValueError(f'Method(s): "{join_str.join(bad_methods)}"\nnot '
#                          + 'found, please use:\n"mom", "energy", "dir", '
#                          + '"purity", "completeness", or "all"')

#     # Parse which methods to use
#     if "all" in method:
#         methods_to_use = ["mom", "energy", "dir", "purity", "completeness"]
#     else:
#         methods_to_use = method

#     # Work out the number of events we have
#     num_events = ak.num(events.trueParticlesBT.number, axis=0)

#     # Currently only worrying about single pi0->yy in each event. In
#     # future, we could extent this by allowing `truth_photon_ids` as an
#     # argument, and then iterating over each photon in each event, but
#     # that's a later problem!
#     truth_photon_ids = truth_pfos_in_two_photon_decay(
#         events, sort=return_type != "mask")

#     # Definitions of what each method involves
#     #   - testing_methods: The test to be performed between the true
#     #     and reco data to get the pair ordering
#     #   - reco_props: The reco data of the candidate PFOs
#     #   - truth_props: The truth data to be tested against

#     testing_methods = {
#         "mom": lambda reco, true: np.argsort(
#             [vector.dot(r, true)[0] for r in reco]),
#         "dir": lambda reco, true: np.argsort(
#             [vector.dot(r, true)[0] for r in reco]),
#         "energy": lambda reco, true: np.argsort(reco),
#         "purity": lambda reco, true: np.argsort(reco),
#         "completeness": lambda reco, true: np.argsort(reco)
#     }
#     # TODO Swap to lambda functions for we don't fetch these unless we
#     # actually need them
#     reco_props = {
#         "mom": events.recoParticles.momentum,
#         "dir": events.recoParticles.direction,
#         "energy": events.recoParticles.energy,
#         "purity": events.trueParticlesBT.purity,
#         "completeness": events.trueParticlesBT.completeness
#     }
#     # TODO Change energy ... to use a smaller zeros/like
#     # (events.trueParticles.number ?)
#     true_props = {
#         "mom": events.trueParticles.momentum,
#         "dir": events.trueParticles.direction,
#         "energy": ak.zeros_like(events.trueParticles.number),
#         "purity": ak.zeros_like(events.trueParticles.number),
#         "completeness": ak.zeros_like(events.trueParticles.number)
#     }

#     # Other properties which must be set up ahead of the loop
#     valid_event_mask = np.full(num_events, True, dtype=bool)

#     truth_indicies_photon_beam = {}
#     if return_type == "mask":
#         for m in methods_to_use:
#             truth_indicies_photon_beam.update({m: [[]] * num_events})
#     else:
#         # This is only valid for the 1 pi0 in event, 2 photon decay cut
#         for m in methods_to_use:
#             truth_indicies_photon_beam.update(
#                 {m: np.zeros((num_events, 2), dtype=int)})

#     zero_count = 0
#     one_count = 0

#     # Loop over each event to determine the best pair
#     for i in range(num_events):
#         bt_ids = events.trueParticlesBT.number[i]
#         indicies = np.arange(len(bt_ids))

#         true_ids = events.trueParticles.number[i]

#         # Sorted ids should be supplied as an argument generated by
#         # truth_pfos_in_two_photon_decay(evts)
#         photon_i, photon_ii = truth_photon_ids[i]

#         if verbosity >= 1:
#             true_momenta = events.trueParticles.momentum[i]
#             if verbosity >= 2:
#                 print("\nTrue particle energies (GeV)")
#                 print(vector.magnitude(true_momenta[true_ids == photon_i]))
#                 print(vector.magnitude(true_momenta[true_ids == photon_ii]))
#                 print("True particle directions")
#                 print(vector.normalize(true_momenta[true_ids == photon_i]))
#                 print(vector.normalize(true_momenta[true_ids == photon_ii]))
#             else:
#                 print(true_momenta[true_ids == photon_i])
#                 print(true_momenta[true_ids == photon_ii])

#         if valid_mom_cut:
#             # Maybe want to look at what happens when we use
#             # purity/completeness with no good data cut?
#             reco_momenta = events.recoParticles.momentum[i]
#             good_data = np.logical_and(
#                 np.logical_and(
#                     reco_momenta.x != -999., reco_momenta.y != -999.),
#                 reco_momenta.z != -999.)
#         else:
#             good_data = slice(None)

#         # Count how many events are cut with no photons/only one photon
#         # having at least one daughter
#         photon_i_exists = photon_i in bt_ids[good_data]
#         photon_ii_exists = photon_ii in bt_ids[good_data]
#         if (not photon_i_exists) or (not photon_ii_exists):
#             if not (photon_i_exists or photon_ii_exists):
#                 zero_count += 1
#             else:
#                 one_count += 1
#             valid_event_mask[i] = False
#             # Skips to next iteration if not both phhotons have
#             # daughters
#             continue

#         for m in methods_to_use:
#             # Get the data to use
#             reco_prop = reco_props[m][i]
#             true_prop = true_props[m][i]
#             # Get the truth property of each photon
#             true_prop_i = true_prop[true_ids == photon_i]
#             true_prop_ii = true_prop[true_ids == photon_ii]
#             # Get a mask indicating the daughters of each photon in the
#             # reco data
#             photon_i_mask = [bt_ids[good_data] == photon_i][0]
#             photon_ii_mask = [bt_ids[good_data] == photon_ii][0]
#             # Need the [0] because bt_ids[good_data] == photon_i is an
#             # array, so returns an array (one element)

#             # Order the PFOs by the selected method
#             reco_prop_ordering_i = testing_methods[m](
#                 reco_prop[indicies[good_data][photon_i_mask]], true_prop_i)
#             reco_prop_ordering_ii = testing_methods[m](
#                 reco_prop[indicies[good_data][photon_ii_mask]], true_prop_ii)
#             # Returns the index of the PFO selected as the best
#             # selection for each photon
#             photon_i_index = indicies[good_data][photon_i_mask][
#                 reco_prop_ordering_i[-1]]
#             photon_ii_index = indicies[good_data][photon_ii_mask][
#                 reco_prop_ordering_ii[-1]]

#             if verbosity >= 6:
#                 print(f"List of reco values {m} with good data:")
#                 print(*reco_props[m][good_data])
#             if verbosity >= 4:
#                 print(f"Values of {m}:")
#                 print("Leading photon")
#                 print(reco_props[indicies[good_data][photon_i_mask]])
#                 print("Sub-leading photon")
#                 print(reco_props[indicies[good_data][photon_ii_mask]])
#             if verbosity >= 5:
#                 print(f"Index ordering of {m} values")
#                 print("Leading photon")
#                 print(reco_prop_ordering_i)
#                 print("Sub-leading photon")
#                 print(reco_prop_ordering_ii)
#             if verbosity >= 3:
#                 print(f"Selected {m}:")
#                 print("Leading photon")
#                 print(reco_prop[photon_i_index])
#                 print("Sub-leading photon")
#                 print(reco_prop[photon_ii_index])

#             # If we are returning IDs, we need to find the
#             # corresponding reco ID
#             if return_type == "id":
#                 reco_ids = events.recoParticles.number[i]
#                 truth_indicies_photon_beam[m][i, :] = [
#                     reco_ids[photon_i_index], reco_ids[photon_ii_index]]
#             # If returning a mask we need to construct the mask
#             elif return_type == "mask":
#                 event_mask = [False] * (indicies[-1] + 1)
#                 event_mask[photon_i_index] = True
#                 event_mask[photon_ii_index] = True
#                 truth_indicies_photon_beam[m][i] = event_mask
#             # Otherwise, we just return the reco indicies of the
#             # selected PFOs
#             else:
#                 truth_indicies_photon_beam[m][i, :] = [
#                     photon_i_index, photon_ii_index]

#     if report:
#         print(f"{zero_count} events discarded due to no true photons "
#               + "having matched PFOs.")
#         print(f"{one_count} events discarded due to only one true photon "
#               + "having matched PFOs.")

#     if return_type == "mask":
#         truth_indicies_photon_beam = ak.Array(truth_indicies_photon_beam)
#     if len(methods_to_use) == 1:
#         return truth_indicies_photon_beam[method[0]], valid_event_mask
#     else:
#         return truth_indicies_photon_beam, valid_event_mask


# def pair_apply_sig_mask(truth_mask, pair_coords):
#     """
#     Returns a count of the number of PFOs which exist in `truth_mask`
#     for each pair in a set of pairs defined by `pair_coords`.

#     Parameters
#     ----------
#     truth_mask : ak.Array
#         Array of boolean values masking a set of signal PFOs.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     sig_counts : ak.Array()
#         Number of signal PFOs in each pair.
#     """
#     # Convert the mask to integers
#     true_counts = np.multiply(truth_mask, 1)

#     # Add the results
#     return true_counts[pair_coords["0"]] + true_counts[pair_coords["1"]]


# def gen_pair_sig_counts(events, pair_coords):
#     """
#     Returns a count of the number of PFOs in a pair which
#     contribute to a good signal:
#     - 2 means the PFOs come from different photons produced
#     by the same pi0
#     - 1 means 1 PFO of a photon from a pi0 exists, or both
#     PFOs were from the same photon
#     - No photons from pi0s

#     Parameters
#     ----------
#     truth_mask : ak.Array
#         Array of boolean values masking a set of signal PFOs.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     sig_counts : ak.Array()
#         Number of signal PFOs in each pair.
#     """
#     photon_from_pions = np.logical_and(
#         events.trueParticlesBT.pdg == 22,
#         events.trueParticlesBT.motherPdg == 111)

#     different_daughter = (
#         events.trueParticlesBT.number[pair_coords["0"]]
#         != events.trueParticlesBT.number[pair_coords["1"]])
#     same_mother = (
#         events.trueParticlesBT.mother[pair_coords["0"]]
#         == events.trueParticlesBT.mother[pair_coords["1"]])

#     same_mother_and_different_daughter = np.logical_and(same_mother,
#                                                         different_daughter)
#     del same_mother
#     del different_daughter

#     both_photons = np.logical_and(photon_from_pions[pair_coords["0"]],
#                                   photon_from_pions[pair_coords["1"]])

#     photons_form_pi0 = np.logical_and(both_photons,
#                                       same_mother_and_different_daughter)
#     del same_mother_and_different_daughter
#     del both_photons

#     at_least_one_photon = np.logical_or(photon_from_pions[pair_coords["0"]],
#                                         photon_from_pions[pair_coords["1"]])
#     # Add the results
#     return (np.multiply(at_least_one_photon, 1)
#             + np.multiply(photons_form_pi0, 1))


# def get_sig_count(events, pair_coordinates, single_best=False, **kwargs):
#     """
#     Generates an awkward array matching the size of `pair_coordinates`
#     indicating how many good signal PFOs are in the indexed pair.

#     If `single_best` is set to `True`, a maximum of one pair may have
#     two signal PFOs per event. This is determined by the
#     `get_best_pairs()` function, and can take additional keyword
#     arguments. See `get_best_pairs` for details.

#     Parameters
#     ----------
#     events : Data
#         Events used to generate the pairs.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.
#     single_best : bool, optional
#         Whether to select only one best pair per event, or if multiple
#         pairs can have a signal count of 2. Default is False.
#     kwargs
#         Additional keyword arguments to be passed to the
#         `get_best_pairs` function if `single_best` is set to `True`.
#     """
#     if single_best:
#         truth_pair_indicies, valid_events = get_best_pairs(events, **kwargs)

#         events.Filter([valid_events], [valid_events])
#         truth_pair_indicies = truth_pair_indicies[valid_events]
#         del valid_events
#         return pair_apply_sig_mask(truth_pair_indicies, pair_coordinates)
#     else:
#         return gen_pair_sig_counts(events, pair_coordinates)


# def pair_photon_counts(events, pair_coords, mother_pdgs):
#     """
#     Returns a count of the number of PFOs which have been backtracked
#     to photons which are daughters of pi0s for each pair in a set of
#     pairs defined by `pair_coords`.

#     Future warning: `mother_pdgs` will be removed in later version, as
#     mother PDG codes are planned to be added into ntuple data.

#     Parameters
#     ----------
#     events : Data
#         Events used to generate the pairs.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.
#     mother_pdgs : ak.Array
#         PDG codes of the mother of the backtracked truth particles.

#     Returns
#     -------
#     sig_counts : ak.Array()
#         Number of photons produced by pi0s in each pair.
#     """
#     true_photons = events.trueParticlesBT.pdg == 22
#     # Get the locations wehere the pdg is 22 and mother pdg is 111
#     first_sigs = np.logical_and(true_photons[pair_coords["0"]],
#                                 mother_pdgs[pair_coords["0"]] == 111)
#     # Multiplying by 1 sets the dtype to be int (1 where True, 0 where
#     # False)
#     first_sigs = np.multiply(first_sigs, 1)
#     # Same for second particle
#     second_sigs = np.logical_and(true_photons[pair_coords["1"]],
#                                  mother_pdgs[pair_coords["1"]] == 111)
#     second_sigs = np.multiply(second_sigs, 1)
#     # Add the results
#     return first_sigs + second_sigs
