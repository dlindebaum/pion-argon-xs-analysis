# Created 17/01/24
# Dennis Lindebaum
# Functions for preparing graph data from analysis.Master.Data data

import os
import warnings
import json
import dill # dill over pickle to allo wsaving local functions
import numpy as np
import awkward as ak
import tensorflow as tf
import tensorflow_gnn as tfgnn
import matplotlib.pyplot as plt
from python.analysis import Master, EventSelection, PFOSelection, vector


# =====================================================================
#    Default definitions of available context, node, and edge data.

def _abs_def(regions):
    return regions["absorption"]
def _cex_def(regions):
    return regions["charge_exchange"]
def _sing_pion_def(regions):
    return regions["single_pion_production"]
def _mutli_pion_def(regions):
    return regions["pion_production"]
def _all_pion_def(regions):
    return np.logical_or(regions["single_pion_production"],
                          regions["pion_production"])
def _all_bkg_def(regions):
    return np.logical_or(regions["charge_exchange"])
default_classification_definitions = {
    "absorption": _abs_def,
    "charge_exchange": _cex_def,
    "single_pion_production": _sing_pion_def,
    "multi_pion_production": _mutli_pion_def,
    "all_pion_production": _all_pion_def,
    "all_background": _all_bkg_def}

def _track_def(evts):
    return evts.recoParticles.track_score
def _nhits_def(evts):
    return evts.recoParticles.n_hits
def _dEdx_med_def(evts):
    return PFOSelection.Median(evts.recoParticles.track_dEdX)
def _chi_pion_def(evts):
    return (evts.recoParticles.track_chi2_pion
            / evts.recoParticles.track_chi2_pion_ndof)
def _chi_muon_def(evts):
    return (evts.recoParticles.track_chi2_muon
            / evts.recoParticles.track_chi2_muon_ndof)
def _chi_proton_def(evts):
    return (evts.recoParticles.track_chi2_proton
            / evts.recoParticles.track_chi2_proton_ndof)
def _len_track_def(evts):
    return evts.recoParticles.track_len
def _len_alt_track_def(evts):
    return evts.recoParticles.track_len_alt
def _track_vert_michel_def(evts):
    return evts.recoParticles.track_vertex_michel
def _energy_shower_def(evts):
    return evts.recoParticles.shower_energy
default_kinematic_definitions = {
    "track": _track_def,
    "n_hits": _nhits_def,
    "dEdx_med": _dEdx_med_def,
    "chi2_pion_reduced": _chi_pion_def,
    "chi2_muon_reduced": _chi_muon_def,
    "chi2_proton_reduced": _chi_proton_def,
    "track_len": _len_track_def,
    "track_len_alt": _len_alt_track_def,
    "track_vertex_michel": _track_vert_michel_def,
    "shower_energy": _energy_shower_def}

def _empty_edge(evts, pairs):
    return ak.ones_like(pairs["0"])
def _track_sep_def(evts, pairs):
    return vector.dist(evts.recoParticles.track_start_pos[pairs["1"]],
                       evts.recoParticles.track_end_pos[pairs["0"]])
def _shower_sep_def(evts, pairs):
    return vector.dist(evts.recoParticles.shower_start_pos[pairs["1"]],
                       evts.recoParticles.track_end_pos[pairs["0"]])
def _track_approach_def(evts, pairs):
    return Master.ShowerPairs.ClosestApproach(
        evts.recoParticles.track_start_dir[pairs["1"]],
        evts.recoParticles.track_start_dir[pairs["0"]],
        evts.recoParticles.track_start_pos[pairs["1"]],
        evts.recoParticles.track_start_pos[pairs["0"]])
def _shower_approach_def(evts, pairs):
    return Master.ShowerPairs.ClosestApproach(
        evts.recoParticles.shower_direction[pairs["1"]],
        evts.recoParticles.track_start_dir[pairs["0"]],
        evts.recoParticles.shower_start_pos[pairs["1"]],
        evts.recoParticles.track_start_pos[pairs["0"]])
def _track_impact_def(evts, pairs):
    return PFOSelection.get_impact_parameter(
        evts.recoParticles.track_start_dir[pairs["1"]],
        evts.recoParticles.track_start_pos[pairs["1"]],
        evts.recoParticles.track_end_pos[pairs["0"]])
def _shower_impact_def(evts, pairs):
    return PFOSelection.get_impact_parameter(
        evts.recoParticles.shower_direction[pairs["1"]],
        evts.recoParticles.shower_start_pos[pairs["1"]],
        evts.recoParticles.track_end_pos[pairs["0"]])
def _shower_shower_sep_def(evts, pairs):
    return vector.dist(evts.recoParticles.shower_start_pos[pairs["1"]],
                       evts.recoParticles.shower_start_pos[pairs["0"]])
def _shower_shower_approach_def(evts, pairs):
    return Master.ShowerPairs.ClosestApproach(
        evts.recoParticles.shower_direction[pairs["1"]],
        evts.recoParticles.shower_direction[pairs["0"]],
        evts.recoParticles.shower_start_pos[pairs["1"]],
        evts.recoParticles.shower_start_pos[pairs["0"]])
default_geometric_definitions = {
    "no_edge": _empty_edge,
    "separation_track": _track_sep_def,
    "separation_shower": _shower_sep_def,
    "closest_approach_track": _track_approach_def,
    "closest_approach_shower": _shower_approach_def,
    "impact_parameter_track": _track_impact_def,
    "impact_parameter_shower": _shower_impact_def,
    "separation_shower_shower": _shower_shower_sep_def,
    "closest_approach_shower_shower": _shower_shower_approach_def}

def _beam_impact(evts):
    return PFOSelection.find_beam_impact_parameters(evts)
def _beam_separation(evts):
    return PFOSelection.find_beam_separations(evts)
default_beam_definitions = {
    "impact": _beam_impact,
    "separation": _beam_separation}

def _true_shared_mother(evts, pairs):
    return 1. * (evts.trueParticlesBT.mother[pairs["0"]]
                 == evts.trueParticlesBT.mother[pairs["1"]])
def _true_shared_mother_no_beam(evts, pairs):
    return 1. * np.logical_and(
        (evts.trueParticlesBT.mother[pairs["0"]]
         == evts.trueParticlesBT.mother[pairs["1"]]),
        np.logical_and(evts.trueParticlesBT.mother[pairs["0"]] != 1,
                       evts.trueParticlesBT.mother[pairs["0"]] != 0))
def _true_closest_approach(evts, pairs):
    return Master.ShowerPairs.ClosestApproach(
        evts.trueParticlesBT.direction[pairs["1"]],
        evts.trueParticlesBT.direction[pairs["0"]],
        evts.trueParticlesBT.shower_start_pos[pairs["1"]],
        evts.trueParticlesBT.shower_start_pos[pairs["0"]])
def _true_pfo_sep(evts, pairs):
    return vector.dist(evts.trueParticlesBT.shower_start_pos[pairs["1"]],
                       evts.trueParticlesBT.shower_start_pos[pairs["0"]])
def _true_pi0(evts, pairs):
    shared_mother = _true_shared_mother(evts, pairs)
    mother_pdgs = PFOSelection.get_mother_pdgs(evts, bt=True)
    mother_pi0 = np.logical_and( # Should only need, one, but be safe!
        mother_pdgs[pairs["0"]] == 211, mother_pdgs[pairs["1"]] == 211)
    return shared_mother * mother_pi0
def _true_inv_mass(evts, pairs):
    mom1 = evts.trueParticlesBT.momentum[pairs["0"]]
    mom2 = evts.trueParticlesBT.momentum[pairs["1"]]
    mag1 = vector.magnitude(mom1)
    mag2 = vector.magnitude(mom2)
    dot = vector.dot(mom1, mom2)
    return 2*(mag1*mag2 - dot)
truth_geometric_definitions = {
    "no_edge": _empty_edge,
    "shared_mother": _true_shared_mother,
    "shared_mother_no_beam": _true_shared_mother_no_beam,
    "closest_approach": _true_closest_approach,
    "separation": _true_pfo_sep,
    "pi0": _true_pi0,
    "invariant_mass": _true_inv_mass}

def _get_beam_vertex_from_true(evts):
    return evts.trueParticles.endPos[:,0]
def _beam_connection(evts):
    return 1. * (evts.trueParticlesBT.mother == 1)
def _beam_gen2_connection(evts):
    gen1_mothers = evts.trueParticles.number[evts.trueParticles.mother == 1]
    return 1. * PFOSelection.get_ak_intersection(
        evts.trueParticlesBT.mother, gen1_mothers)
def _true_beam_impact(evts):
    return PFOSelection.get_impact_parameter(
        evts.trueParticlesBT.direction,
        evts.trueParticlesBT.shower_start_pos,
        evts.trueParticlesBT.trueBeamVertex)
def _true_beam_separation(evts):
    return vector.dist(evts.trueParticlesBT.shower_start_pos,
                       evts.trueParticlesBT.trueBeamVertex)
def _true_true_beam_impact(evts):
    return PFOSelection.get_impact_parameter(
        evts.trueParticlesBT.direction,
        evts.trueParticlesBT.shower_start_pos,
        _get_beam_vertex_from_true(evts))
def _true_true_beam_separation(evts):
    return vector.dist(evts.trueParticlesBT.shower_start_pos,
                       _get_beam_vertex_from_true(evts))
truth_beam_definitions = {
    "beam_daughter" : _beam_connection,
    "impact": _true_beam_impact,
    "separation": _true_beam_separation,
    "impact_true": _true_true_beam_impact,
    "separation_true": _true_true_beam_separation}

def _mc_true_shared_mother(evts, pairs):
    return 1. * (evts.trueParticles.mother[pairs["0"]]
                 == evts.trueParticles.mother[pairs["1"]])
def _mc_true_closest_approach(evts, pairs):
    return Master.ShowerPairs.ClosestApproach(
        evts.trueParticles.direction[pairs["1"]],
        evts.trueParticles.direction[pairs["0"]],
        evts.trueParticles.shower_start_pos[pairs["1"]],
        evts.trueParticles.shower_start_pos[pairs["0"]])
def _mc_true_pfo_sep(evts, pairs):
    return vector.dist(evts.trueParticles.shower_start_pos[pairs["1"]],
                       evts.trueParticles.shower_start_pos[pairs["0"]])
mc_truth_geometric_definitions = {
    "shared_mother": _mc_true_shared_mother,
    "closest_approach": _mc_true_closest_approach,
    "separation": _mc_true_pfo_sep}

def _get_mc_beam_vertex_from_true(evts):
    return evts.trueParticles.endPos[:,0]
def _mc_beam_connection(evts):
    return 1. * (evts.trueParticles.mother == 1)
def _mc_beam_gen2_connection(evts):
    gen1_mothers = evts.trueParticles.number[evts.trueParticles.mother == 1]
    return 1. * PFOSelection.get_ak_intersection(
        evts.trueParticles.mother, gen1_mothers)
def _mc_true_beam_impact(evts):
    return PFOSelection.get_impact_parameter(
        evts.trueParticles.direction,
        evts.trueParticles.shower_start_pos,
        _get_mc_beam_vertex_from_true(evts))
def _mc_true_beam_separation(evts):
    return vector.dist(evts.trueParticles.shower_start_pos,
                       _get_mc_beam_vertex_from_true(evts))
mc_truth_beam_definitions = {
    "beam_daughter" : _mc_beam_connection,
    "impact_true": _mc_true_beam_impact,
    "separation_true": _mc_true_beam_separation}

def _reco_mom_x(evts):
    return evts.recoParticles.shower_momentum.x
def _reco_mom_y(evts):
    return evts.recoParticles.shower_momentum.y
def _reco_mom_z(evts):
    return evts.recoParticles.shower_momentum.z
def _true_mom_x(evts):
    return evts.trueParticlesBT.momentum.x
def _true_mom_y(evts):
    return evts.trueParticlesBT.momentum.y
def _true_mom_z(evts):
    return evts.trueParticlesBT.momentum.z
def _mc_true_mom_x(evts):
    return evts.trueParticles.momentum.x
def _mc_true_mom_y(evts):
    return evts.trueParticles.momentum.y
def _mc_true_mom_z(evts):
    return evts.trueParticles.momentum.z
default_momenta_definitions = {
    "reco_x": _reco_mom_x,
    "reco_y": _reco_mom_y,
    "reco_z": _reco_mom_z,
    "true_x": _true_mom_x,
    "true_y": _true_mom_y,
    "true_z": _true_mom_z,
    "mc_true_x": _mc_true_mom_x,
    "mc_true_y": _mc_true_mom_y,
    "mc_true_z": _mc_true_mom_z}

available_classification_definitions = [
    key for key in default_classification_definitions.keys()]
available_kinematic_definitions = [
    key for key in default_kinematic_definitions.keys()]
available_geometric_definitions = [
    key for key in default_geometric_definitions.keys()]
available_beam_definitions = [
    key for key in default_beam_definitions.keys()]
available_truth_geometric_definitions = [
    key for key in truth_geometric_definitions.keys()]
available_truth_beam_definitions = [
    key for key in truth_beam_definitions.keys()]


def _apply_not_beam(events, other_mask, mc_true=False):
    """Adds a not-beam PFO requirement to the mask, returns as float"""
    if mc_true:
        particles = events.trueParticles
        beam_num = 1
    else:
        particles = events.recoParticles
        beam_num = particles.beam_number
    not_beam = (particles.number
                != beam_num)
    return 1. * np.logical_and(not_beam, other_mask)
def _get_multi_mask(events, ids, mc_true=False):
    """Combine mask from events together where pdg codes match ids"""
    if mc_true:
        particles = events.trueParticles
    else:
        particles = events.trueParticlesBT
    mask = False
    for id in ids:
        mask = np.logical_or(mask, particles.pdg == id)
    return mask
def _create_truth_definition_func(ids, index=None, mc_true=False):
    """Makes the functions to go into truth definitions"""
    if mc_true:
        get_particles = lambda events: events.trueParticles
    else:
        get_particles = lambda events: events.trueParticlesBT
    if index is None:
        def other_func(events):
            mask = True
            for id in ids:
                if isinstance(id, tuple):
                    this_mask = _get_multi_mask(events, id, mc_true=mc_true)
                else:
                    this_mask = get_particles(events).pdg == id
                mask = np.logical_and(mask, np.logical_not(this_mask))
            return _apply_not_beam(events, mask, mc_true=mc_true)
        return other_func
    elif index == "beam":
        def bkg_func(events):
            return 1. - _apply_not_beam(events, True, mc_true=mc_true) 
        return bkg_func       
    else:
        def ind_func(events):
            id = ids[index]
            if isinstance(id, tuple):
                mask = _get_multi_mask(events, id, mc_true=mc_true)
            else:
                mask = get_particles(events).pdg == id
            return _apply_not_beam(events, mask, mc_true=mc_true)
        return ind_func

def gen_truth_definitions(ids, names, mc_true=False):
    """
    Takes a list of `ids` containing integers or tuples of two
    integers, combined is a list of `names` of equal length to `ids`
    containing strings.

    Returns a dictionary classifing the truth particles in `events` as
    defined by `ids` and `names`. If an entry is `ids` is a tuple, all
    elements in that tuple will be classified together, i.e. positive
    and negative pions: `(211, -211)`.

    In addition, the beam particle will be uniquely classified, and an
    "other" category will be created to take any unaccoutned PFOs.

    Parameters
    ----------
    events : Master.Data
        Event data from which to get values
    ids : list [ int or tuple ]
        List of PDG codes indexing which particles form each truth
        class. If passing a tuple of integers, all elements in the
        tuple will be treated as one class.
    names : list [ str ]
        Names of the truth classes to use. Must be the same length as
        `ids`.
    
    returns
    dict
        Dictionary containing truth class definitions.
    """
    res = {"beam_truth": _create_truth_definition_func(
        ids, "beam", mc_true=mc_true)}
    for i, name in enumerate(names):
        res.update({f"{name}_truth":_create_truth_definition_func(
            ids, i, mc_true=mc_true)})
    res.update({"other_truth":_create_truth_definition_func(
        ids, None, mc_true=mc_true)})
    return res
    # res = {"beam_truth": 1. - _apply_not_beam(events, True)}
    # other_mask = True
    # for i, name in enumerate(names):
    #     if isinstance(ids[i], tuple):
    #         this_mask = _get_multi_mask(events, ids[i])
    #     else:
    #         this_mask = events.trueParticlesBT.pdg == ids[i]
    #     res.update({f"{name}_truth":_apply_not_beam(events, this_mask)})
    #     other_mask = np.logical_and(other_mask,
    #                                 np.logical_not(this_mask))
    # res.update({"other_truth":_apply_not_beam(events, other_mask)})
    # return res   


# =====================================================================
#                   Functions to generate graph data

class NormalisableProperty():
    """
    Stores a set of properties with normalisation to mean 0, variance
    1.

    Label like values (values with only 2 unique inputs, i.e. 0/1) are
    not normalised.

    Attributes
    ----------
    unnormed : dict
        Dictionary of properties without normalisation.
    normed : dict
        Dictionary of normalised properties.
    norm_params : dict
        Dictionary containing the calulated mean and standard
        deviations used in normalisation.
    use_norm : bool
        Whether to set normed or unnormed as the default used value
    use_val : dict
        Default values to use (`normed` or `unnormed`), set by
        `use_norm`.
    
    Methods
    -------
    normalise(unnormed)
        Returns the input dictionary with normalisation applied.
    denormalise(normed)
        Returns the input dictionary with denormalisation applied.
    """
    def __init__(self, unnormed_props, use_norm=True):
        """
        Create a NormalisableProperty instance from the dictionary
        supplied.

        Parameters
        ----------
        unnormed_props : dict
            Set of properties from which to pull normalisation. This
            data will also have normed and unnormed versions stored as
            attributes.
        use_norm : bool, optional
            Whether to return the normed or unnormed data as the
            default `use_val`. Default is True.
        """
        self.use_norm = use_norm
        self.unnormed = unnormed_props
        self.norm_params = self._generate_normalisations()
        self.normed = self.normalise(self.unnormed)
        return

    def _generate_normalisations(self):
        result = {}
        for (name, data) in self.unnormed.items():
            flat_data = ak.ravel(data)
            if len(set(flat_data)) == 2:
                # Label type property (0/1), don't norm
                result.update({
                    name: {"mean": 0.,
                           "std": 1.,
                           "label": True}})
            else:
                result.update({
                    name: {"mean": np.mean(flat_data),
                           "std": np.std(flat_data),
                           "label": False}})
        return result
    
    def normalise(self, unnormed):
        """
        Apply stored normalisation parameters to input data.

        Parameters
        ----------
        unnormed : dict
            Dictionary of unnormalised properties.

        Returns
        -------
        dict
            Properties applied with normalisation parameters stored in
            the instance.
        """
        scaling = lambda name: ( (unnormed[name]
                                  - self.norm_params[name]["mean"])
                                / self.norm_params[name]["std"])
        return {name: scaling(name) if not self.norm_params[name]["label"]
                else unnormed[name]
                for name in unnormed.keys()}

    def denormalise(self, normed):
        """
        Denormalise input data based on normalisation parameters stored
        in the instance.

        Parameters
        ----------
        normed : dict
            Dictionary of normalised properties.

        Returns
        -------
        dict
            Properties denormalised by the normalisation parameters
            stored in the instance.
        """
        scaling = lambda name: (normed[name]
                                * self.norm_params[name]["std"]
                                + self.norm_params[name]["mean"])
        return {name: scaling(name) if not self.norm_params[name]["label"]
                else normed[name]
                for name in normed.keys()}

    def copy(self):
        """
        Create a copy of the instance (note normalisations are
        recalculated for the copy).

        Returns
        -------
        NormalisationProperty
            A copy of this instance.
        """
        return NormalisableProperty(self.unnormed, self.use_norm)

    def get_subset(self, mask, renorm=True):
        """
        Return a NormalisableProperty using the subset of properties
        given by the mask.

        Parameters
        ----------
        mask : ak.array
            Data mask to apply.
        renorm : bool, optional
            If True, normalisation will be recalculated, else use the
            normalisation present in this instance. Default is True.
        
        Returns
        -------
        NormalisableProperty
            Instance with poerties filtered by mask.
        """
        new_data = {name: val[mask] for (name, val) in self.unnormed.items()}
        if renorm:        
            return NormalisableProperty(new_data, self.use_norm)
        new_instance = self.copy()
        new_instance.unnormed = new_data
        new_instance.normed = new_instance.normalise(new_data)
        return new_instance

    @property
    def use_val(self):
        if self.use_norm:
            return self.normed
        else:
            return self.unnormed

def generate_directed_pair_indicies(events, mc_true=False):
    """
    Creates a zipped awkward array containing directed edge pairs.

    Parameters
    ----------
    events : analysis.Master.Data
        Events for which the pairs should be created.

    Returns
    -------
    ak.Array
        Zipped array with the indicies of source PFOs as `"0"` and
        indicies of target PFOs as `"1"`.
    """
    if mc_true:
        particles = events.trueParticles.number
    else:
        particles = events.recoParticles.number
    node_pairs = ak.argcombinations(particles, 2)
    node_pairs_reversed = ak.zip((node_pairs["1"], node_pairs["0"]))
    return ak.concatenate((node_pairs, node_pairs_reversed),-1)

def _particle_counts_to_one_hot_classification(
        pion_counts, pi0_counts,
        classifications, classification_definitions):
    """Converts particle counts per event to one-hot classification"""
    regions_dict = EventSelection.create_regions_new(pi0_counts, pion_counts)
    one_hot_classes = np.array([
        classification_definitions[which](regions_dict)
            for which in classifications]).T
    if len(classifications) == 1:
        # Don't use vector encoding if using binary classification
        one_hot_classes = np.squeeze(one_hot_classes, axis=-1)
    elif np.any(one_hot_classes.sum(axis=1) != 1):
        warnings.warn("Not all rows have exactly one non-zero entry")
    return one_hot_classes.astype(np.float32)

def make_evt_classifications(
        events,
        classifications,
        classification_definitions=default_classification_definitions):
    """
    Creates an array with ones in the column corresponding to valid
    classifications of the event, and zero in other columns.

    Raises a warning if some multiple classifications are specified and
    there is not exactly one non-zero entry in each row.

    Parameters
    ----------
    events : analysis.Master.Data
        Events which should be classified.
    classifications : list [ str ]
        List of names of classifications to include. Must be in the
        keys available in `classification_definitions`. To get a list
        of default keys, get the
        `DataPreparation.available_classification_definitions`
        variable.
    classification_definitions : dict {str : func}, optional
        Defintions of the available event classifications. Functions
        are passed one argument, which is a dictionary of regions as
        created by `analysis.EventSelection.create_regions_new()`.
        Default is
        `DataPreparation.default_classification_definitions`.
        
    Returns
    -------
    np.ndarray
        An array of integers with shape
        `(n_events, len(classifications))` if
        `len(classifications) > 1`, else `(n_events,)`, where
        `n_events` is the number of events in `events`.
    """
    pi0_counts = EventSelection.count_all_pi0s(events)
    pion_counts = EventSelection.count_non_beam_charged_pi(events)
    return _particle_counts_to_one_hot_classification(
        pion_counts, pi0_counts,
        classifications, classification_definitions)

def classifiy_true_pfos(evts, mc_pfos=False):
    """
    Returns a PFO-like awkward array with 0 for tracks, 1 for
    showers, and 2 for pi0s
    
    Parameters
    ----------
    evts : Master.Data
        Events from which to draw PFOs.
    
    Returns
    -------
    ak.Array
        Array containing PFO labels
    """
    pdgs = evts.trueParticles.pdg if mc_pfos else evts.trueParticlesBT.pdg
    pfos = ak.zeros_like(pdgs)
    shower_mask = np.logical_and(
        pdgs == 22,
        np.logical_and(pdgs == 11, pdgs == -11))
    pfos += 1. * shower_mask
    pfos += 2. * pdgs == 111
    return pfos

def split_pfo_option(func):
    """
    Decorate a function to split the results between tracks and
    showers.
    """
    def decorated(*args, split_pfos=None, data_type="reco", **kwargs):
        props = func(*args, **kwargs)
        if split_pfos is None:
            return props
        events = args[0]
        if data_type != "reco":
            labels = classifiy_true_pfos(events, mc_pfos=data_type == "mc")
            shower_mask = labels == 1.
            track_mask  = labels == 0.
        else:
            shower_mask = events.recoParticles.cnn_score > split_pfos
            track_mask = np.logical_not(shower_mask)
        return props.get_subset(track_mask), props.get_subset(shower_mask)
    return decorated

def make_evt_ids(events):
    """
    Generate an array containing information to uniqul idenitfy
    events.

    Parameters
    ----------
    events : Data
        Events from which to gather IDs.
    
    Returns
    np.ndarray
        (N, 3) array of IDs.    
    """
    to_numpy = lambda arr : np.array(arr)[:, np.newaxis].astype(np.int32)
    run_num = to_numpy(events.run)
    subrun_num = to_numpy(events.subRun)
    event_num = to_numpy(events.eventNum)
    return np.concatenate([run_num, subrun_num, event_num], axis=1)

def make_evt_kinematics(
        events,
        kinematics,
        kinematic_definitions=default_kinematic_definitions,
        norm=True):
    """
    Creates a `NormalisableProperty` instance containing the properties
    defined by `kinematic_definitions` which are included in the
    `kinematics` list.

    Parameters
    ----------
    events : analysis.Master.Data
        Events from which to take the properties.
    kinematics : list [ str ]
        List of names of kinematic properties to include. Must be in
        the keys available in `kinematic_definitions`. To get a list of
        default keys, get the
        `DataPreparation.available_kinematic_definitions` variable.
    kinematic_definitions : dict {str : func}, optional
        Defintions of the available kinematics. Functions are passed
        the `events` as an argument. Default is
        `DataPreparation.default_kinematic_definitions`.
    norm : bool, optional
        Whether or not to use the normalised version of the values.
        Default is True.
        
    Returns
    -------
    DataPreparation.NormalisableProperty
        An instance containing a dictionary of unnormalised property
        values as `NormalisableProperty.unnormed`, normalised values
        as `NormalisableProperty.normed`, and selected normalisation
        as `NormalisableProperty.use_vals`
    """
    evaluated_kinematics = {name: kinematic_definitions[name](events)
                            for name in kinematics}
    return NormalisableProperty(evaluated_kinematics, norm)

def make_evt_momenta(
        events,
        momenta,
        momenta_definitions=default_momenta_definitions,
        norm=True):
    """
    Creates a `NormalisableProperty` instance containing the momenta of
    the particles.

    Parameters
    ----------
    events : analysis.Master.Data
        Events from which to take the momenta.
    momenta : list [ str ]
        List of which momenta to include. Must be in the keys available
        in `momenta_definitions`.
    momenta_definitions : dict {str : func}, optional
        Defintions of the available momenta. Functions are passed
        the `events` as an argument. Default is
        `DataPreparation.default_momenta_definitions`.
    norm : bool, optional
        Whether or not to use the normalised version of the values.
        Default is True.
        
    Returns
    -------
    DataPreparation.NormalisableProperty
        An instance containing a dictionary of unnormalised property
        values as `NormalisableProperty.unnormed`, normalised values
        as `NormalisableProperty.normed`, and selected normalisation
        as `NormalisableProperty.use_vals`
    """
    evaluated_momenta = {name: momenta_definitions[name](events)
                            for name in momenta}
    return NormalisableProperty(evaluated_momenta, norm)

def make_evt_geometries(
        events,
        geomtries,
        geometric_definitions=default_geometric_definitions,
        directed_pairs=None,
        norm=True):
    """
    Creates a `NormalisableProperty` instance containing the geometric
    properties defined by `geometric_definitions` which are included in
    the `geomtries` list.
    
    Geometric properties are defined between the source PFO and target
    PFO contained in `directed_pairs`. If unspecified, geometric data
    will be taken between every pairing of PFOs in the event.

    Parameters
    ----------
    events : analysis.Master.Data
        Events from which to take the properties.
    geomtries : list [ str ]
        List of names of geomtric properties to include. Must be in
        the keys available in `geometric_definitions`. To get a list of
        default keys, get the
        `DataPreparation.available_geometric_definitions` variable.
    geometric_definitions : dict {str : func}, optional
        Defintions of the available geomtric properties. Functions are
        passed the `*(events, directed_pairs)` as arguments. Default is
        `DataPreparation.default_geometric_definitions`.
    directed_pairs : ak.Array, optional
        Zipped array containing the source PFOs as field `'0'` and
        target PFOs as field `'1'` between which to generate
        geometric data. If None, pairs will be created between every
        PFO in each event. Default is None.
    norm : bool, optional
        Whether or not to use the normalised version of the values.
        Default is True.
    
    Returns
    -------
    DataPreparation.NormalisableProperty
        An instance containing a dictionary of unnormalised property
        values as `NormalisableProperty.unnormed`, normalised values
        as `NormalisableProperty.normed`, and selected normalisation
        as `NormalisableProperty.use_vals`
    """
    if directed_pairs is None:
        directed_pairs = generate_directed_pair_indicies(events)
    evaluated_kinematics = {
        name: geometric_definitions[name](
            events, directed_pairs)
        for name in geomtries}
    return NormalisableProperty(evaluated_kinematics, norm)

def make_beam_props(events, beam_props, beam_definitions, norm=True):
    """
    Creates a `NormalisableProperty` instance containing the properties
    defined by `beam_definitions` which are included in the
    `beam_props` list.

    Parameters
    ----------
    events : analysis.Master.Data
        Events from which to take the properties.
    beam_props : list [ str ]
        List of names of beam connection properties on edges connecting
        to the beam node to include. Must be in the keys available in
        `beam_definitions`. To get a list of default keys, get the
        `DataPreparation.available_beam_definitions` variable.
    beam_definitions : dict {str : func}, optional
        Defintions of the available beam connections. Functions are
        passed the `events` as an argument. Default is
        `DataPreparation.default_beam_definitions`.
    norm : bool, optional
        Whether or not to use the normalised version of the values.
        Default is True.
        
    Returns
    -------
    DataPreparation.NormalisableProperty
        An instance containing a dictionary of unnormalised property
        values as `NormalisableProperty.unnormed`, normalised values
        as `NormalisableProperty.normed`, and selected normalisation
        as `NormalisableProperty.use_vals`
    """
    evaluated_beam = {name: beam_definitions[name](events)
                      for name in beam_props}
    return NormalisableProperty(evaluated_beam, norm)

def make_pfo_truth_info(events, mc=False):
    # This is inefficient, since there are shared bits of data
    #   caclulated between the different types of truth info
    pdgs = [(211, -211),    22   ]
    names= [  "pion"   , "photon"]
    if mc:
        pdgs.append(111)
        names.append("pi0")
        beam_particles = _mc_beam_connection(events)
        mothers = events.trueParticles.mother
    else:
        beam_particles = _beam_connection(events)
        mothers = events.trueParticlesBT.mother

    particle_defs = gen_truth_definitions(
        pdgs, names, mc_true=mc)
    
    pions = particle_defs["pion_truth"](events)
    photons = particle_defs["photon_truth"](events)

    mc_beam_daughters = events.trueParticles.mother == 1
    mc_beam_pi0s = np.logical_and(mc_beam_daughters,
                                      events.trueParticles.pdg == 111)
    all_granddaughters = 1. * PFOSelection.get_ak_intersection(
        mothers, events.trueParticles.number[mc_beam_daughters])
    pi0_granddaughters = 1. * PFOSelection.get_ak_intersection(
        mothers, events.trueParticles.number[mc_beam_pi0s])

    pfo_dict = {
        "beam_daughter": beam_particles,
        "beam_granddaughter": all_granddaughters,
        "pi0_granddaughter": pi0_granddaughters,
        "beam_related": beam_particles + all_granddaughters,
        "beam_relevant": beam_particles + pi0_granddaughters,
        "pion": pions,
        "photon": photons,
        "beam_pion": pions * beam_particles,
        "beam_photon": photons * pi0_granddaughters}
    if mc:
        pi0s = particle_defs["pi0_truth"](events)
        pfo_dict.update({
            "pi0": pi0s,
            "beam_pi0": pi0s * beam_particles})
    return pfo_dict

def make_neighbour_truth_info(events, pairs, mc=False):
    # This is inefficient, since there are shared bits of data
    #   caclulated between the different types of truth info
    if mc:
        shared_mother = _mc_true_shared_mother(events, pairs)
        pfo_mothers = events.trueParticles.mother
    else:
        shared_mother = _true_shared_mother(events, pairs)
        pfo_mothers = events.trueParticlesBT.mother
    pi0_mask = events.trueParticles.pdg == 111
    true_pi0s = events.trueParticles.number[pi0_mask]
    true_beam_pi0s = events.trueParticles.number[np.logical_and(
        pi0_mask, events.trueParticles.mother == 1)]
    pi0_photons = PFOSelection.get_ak_intersection(
        pfo_mothers, true_pi0s)
    beam_pi0_photons = PFOSelection.get_ak_intersection(
        pfo_mothers, true_beam_pi0s)

    mother_pi0 = shared_mother * np.logical_and(
        pi0_photons[pairs["0"]], pi0_photons[pairs["1"]])
    beam_mother_pi0 = shared_mother * np.logical_and(
        beam_pi0_photons[pairs["0"]], beam_pi0_photons[pairs["1"]])

    neighbour_dict = {
        "true_pi0": mother_pi0,
        "beam_pi0": beam_mother_pi0}
    return neighbour_dict

def make_beam_conn_truth_info(events, mc=False):
    # This is inefficient, since there are shared bits of data
    #   caclulated between the different types of truth info
    if mc:
        beam_particles = _mc_beam_connection(events)
        mothers = events.trueParticles.mother
    else:
        beam_particles = _beam_connection(events)
        mothers = events.trueParticlesBT.mother

    mc_beam_daughters = events.trueParticles.mother == 1
    mc_beam_pi0s = np.logical_and(mc_beam_daughters,
                                      events.trueParticles.pdg == 111)
    all_granddaughters = 1. * PFOSelection.get_ak_intersection(
        mothers, events.trueParticles.number[mc_beam_daughters])
    pi0_granddaughters = 1. * PFOSelection.get_ak_intersection(
        mothers, events.trueParticles.number[mc_beam_pi0s])
    beam_conn_dict = {
        "true_daughter": beam_particles,
        "true_granddaughter": all_granddaughters,
        "pi0_granddaughter": pi0_granddaughters,
        "beam_related": beam_particles + all_granddaughters,
        "beam_relevant": beam_particles + pi0_granddaughters}
    return beam_conn_dict

def make_context_truth_info(
        events,
        classifications, classification_definitions,
        mc=False):
    # This is inefficient, since there are shared bits of data
    #   caclulated between the different types of truth info
    pdgs = [(211, -211),    22   , 111]
    names= [  "pion"   , "photon", "pi0"]
    mc_particle_defs = gen_truth_definitions(
        pdgs, names, mc_true=True)

    beam_particles = _mc_beam_connection(events)
    mothers = events.trueParticles.mother
    mc_beam_daughters = events.trueParticles.mother == 1
    mc_beam_pi0s = np.logical_and(mc_beam_daughters,
                                      events.trueParticles.pdg == 111)
    pi0_granddaughters = 1. * PFOSelection.get_ak_intersection(
        mothers, events.trueParticles.number[mc_beam_pi0s])
    
    # Dumb way to do it - get the extra context dimension by
    #   passing as numpy over and ak.Array
    context_dict = {
        "mc_pions": ak.sum(mc_particle_defs["pion_truth"](events)
                     * beam_particles, axis=-1).to_numpy(),
        "mc_photons": ak.sum(mc_particle_defs["photon_truth"](events)
                       * pi0_granddaughters, axis=-1).to_numpy(),
        "mc_pi0s": ak.sum(mc_particle_defs["pi0_truth"](events)
                     * mc_beam_pi0s, axis=-1).to_numpy()}
    if not mc:
        bt_particle_defs = gen_truth_definitions(
            pdgs, names, mc_true=False)

        bt_beam_particles = _beam_connection(events)
        bt_mothers = events.trueParticlesBT.mother
        bt_pi0_granddaughters = 1. * PFOSelection.get_ak_intersection(
            bt_mothers, events.trueParticles.number[mc_beam_pi0s])

        # Unique list of pi0 IDs which are mothers of backtracked particles
        bt_pi0_mothers = bt_mothers[bt_pi0_granddaughters == 1]
        bt_pi0_unique = PFOSelection.add_event_offset(bt_pi0_mothers)
        unique_pi0s = np.unique(ak.ravel(bt_pi0_unique))
        # Convert the unique list into only the event IDs, so we get
        # n copies of the event ID, where n is the numebr of mother
        # pi0s in that event
        n_events = ak.num(bt_pi0_mothers, axis=0)
        event_bits = len(bin(ak.num(bt_pi0_mothers, axis=0))) - 2
        event_numbers = unique_pi0s%(2**event_bits)
        # inds is the event index, n_pi0s is the number of pi0s in that event
        inds, n_pi0s = np.unique(event_numbers, return_counts=True)
        bt_pi0_counts = np.zeros(n_events)
        bt_pi0_counts[inds] = n_pi0s

        bt_pi_counts = ak.sum(bt_particle_defs["pion_truth"](events)
                            * bt_beam_particles, axis=-1).to_numpy()
        bt_photon_counts = ak.sum(bt_particle_defs["photon_truth"](events)
                                  * bt_pi0_granddaughters, axis=-1).to_numpy()

        context_dict.update({
            "bt_pions": bt_pi_counts,
            "bt_photons": bt_photon_counts,
            "bt_pi0s": bt_pi0_counts,
            "reco_class":
                _particle_counts_to_one_hot_classification(
                    bt_pi_counts, bt_pi0_counts,
                    classifications, classification_definitions)})
    return context_dict

def get_event_index(
        event_index,
        prop,
        prop_order=None):
    """
    Returns the value of the property at the supplied event index as a
    tf.Tensor to be used to create a graph.

    `prop` must be a numpy array or NormalisableProperty created by
    one of the `DataPreparation.make_evt_classifications()`,
    `DataPreparation.make_evt_kinematics()`, or
    `DataPreparation.make_evt_geometries()` functions.

    Requires a prop_order if a NormalisableProperty is passed to ensure
    the values are loaded in a known order.

    This is a dumb function, it shouldn't be so generic.
    It relies on the type to tell what response is desired.
    np.ndarray => context like
    NormalisableProperty requires special context
    ak.Array doesn't have the leading dimension

    Parameters
    ----------
    event_index : int
        Index of the event to fetch.
    prop : np.ndarray or DataPerpataion.NormalisableProperty
        Property to convert to a tensor
    prop_order : list[str], optional
        Order of properties to read out. List of strings found in the
        NormalisableProperty instance. Default is None.

    Returns
    -------
    tf.Tensor
        Property at the supplied index as a tensor ready to generate an
        event graph.
    
    """
    if isinstance(prop, np.ndarray):
        # This is a context value
        # We need and outer dimension to be one, so return an array
        return tf.convert_to_tensor(
            prop[event_index:event_index+1])
    # If these are node/edge properties (as a NormalisableProperty
    elif isinstance(prop, NormalisableProperty):
        if prop_order is None:
            raise ValueError("prop_order must be specified for "
                             +"NormalisableProperty")
        np_from_ak = lambda data: ak.to_numpy(data[event_index])
        return tf.convert_to_tensor(
            np.array([np_from_ak(prop.unnormed[which])
                      for which in prop_order]).T,
                     dtype=np.float32)
    # Else an ak.Array, we don't keep outer dimension
    return tf.convert_to_tensor(
        prop[event_index])
    
def get_property_normalisations(prop, prop_order):
    """
    Returns the means and standard deviation to normalise the
    properties.
    
    Will indicate mean 0. and standard deviation 1. for properties
    which should not be normalised (such that normalisation returns the
    input). This applies to label type properties, i.e. where only two
    unique values are present (i.e. 1 and 0).

    Also includes a list of strings referencing the properties inside
    the NormalisableProperty to ensure the normalisations are correctly
    mapped.

    Parameters
    ----------
    prop : DataPerpataion.NormalisableProperty
        Class containing the data with normalisation parameters.
    
    Returns
    -------
    means : tf.Tensor
        Tensor containing the mean of each property in `prop`.
    stds : tf.Tensor
        Tensor containing the standard deviation of each property in
        `prop`.
    prop_order : str[list]
        The order the noirmalisation are given.
    """
    norm_params = prop.norm_params
    means = np.array([norm_params[which]["mean"]
                      for which in prop_order],
                     dtype=np.float32)
    stds =  np.array([norm_params[which]["std"]
                      for which in prop_order],
                     dtype=np.float32)
    return means, stds

def create_filepath_dictionary(folder_path):
    """
    Create a dictionary containing the default paths to stored data
    given the storage folder.

    Dictionary contains `"folder_path"`, `"schema_path"`,
    `"train_path"`, `"val_path"`, "`test_path"`, and `"dict_path"`.

    Parameters
    ----------
    folder_path : str
        Path to main folder storing the data.

    Returns
    -------
    dict
        Dictionary containing default paths to saved data.
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return {
        "folder_path": folder_path,
        "schema_path": os.path.join(folder_path, "schema.pbtxt"),
        "train_path": os.path.join(folder_path, "train_data.tfrecords"),
        "val_path": os.path.join(folder_path, "val_data.tfrecords"),
        "test_path": os.path.join(folder_path, "test_data.tfrecords"),
        "dict_path": os.path.join(folder_path, "params_dict.dill"),
        "norm_path": os.path.join(folder_path, "norm_params.json")}

def create_parameter_dictionary(
        folder_path,
        events,
        classifications,
        kinematics,
        geometries,
        directed_pairs=None,
        beam_connections=None,
        momenta=None,
        truth_info=False,
        split_pfos=None,
        data_type="reco",
        train_weight=0.8,
        val_weight=0.1,
        test_weight=0.1,
        norm_kinematics=True,
        norm_geometries=True,
        norm_beam=True,
        norm_momenta=True,
        classification_definitions=default_classification_definitions,
        kinematic_definitions=default_kinematic_definitions,
        geometric_definitions=default_geometric_definitions,
        beam_definitions=default_beam_definitions,
        momenta_definitions=default_momenta_definitions):
    """
    Creates a dictionary storing the events, properties to use, and
    property definitions for graph generation.

    Note stupid mc understanding. Use the mc_data to set the truth
    info, but other stuff is based upon dictionaries.

    Parameters
    ----------
    folder_path : str
        Path to folder in which to save results.
    events : analysis.Master.Data
        Events from which to generate graphs
    classifications : list [ str ]
        List of event classifications to use.
    kinematics : list [ str ]
        List of PFO kinematic proeprties to use.
    geometries : list [ str ]
        List of neighbour geometric properties to use.
    directed_pairs : ak.Array, optional
        Zipped awkward array containing source and target PFOs for
        directed edges. If None, pairs will be created between all
        PFOs. Default is None.
    beam_connections : bool, optional
        Edge properties to include in connections to an optional beam
        particle node. If None, no beam node will be created. Default
        is None.
    momenta : list [ str ], optional
        If passed, this list of momentum generators will be used to
        generate a momentum field in the graph verticies. Entries must
        correspond to entries in the `momenta_definitions` argument.
        Default is None.
    truth_info : bool, optional
        If True, include extra information in the context, pfo,
        connection, and beam connection fields with BT truth properties
        - to be useable for extra loss parameters. Default is False.
    split_pfos : float, optional
        If not None, will create two different node classes
        corresponding to pfos: `pfo_shower` and `pfo_track`, instead of
        a unified `pfo` node. `neighbour` connections will only be made
        between `pfo_shower` nodes. The value supplied should be in
        (0,1) which gives the CNN score cut to distingush types. If a
        PFO has a CNN score `<= split_pfos`, it will be classified as a
        `pfo_track`, else as a `pfo_shower`.
        
        If using Monte-Carlo data (`data_type` is 'mc' or 'bt'), the PFOs will be
        split based on PDG values as follows:
         - 22, 11, -11: `pfo_track` (photon, electron, positron)
         - 111: removed - no node created (pi0)
         - otherwise: `pfo_track`
        Default is None.
    data_type : str ['reco', 'bt', 'mc'], optional
        Flag indicating whether the graph is using Monte-Carlo truth
        data, backtracked data or recoconstructed data. Used to
        generate the correct truth information if `truth_info` is True,
        or to use pdg codes to separate shower/track like particles if
        `split_pfos` is not None. Default is 'reco'.
    train_weight : float, optional
        Proportion of events to use in the training dataset. Weighting
        is normalised to the other supplied weightings to ensure the
        sum is 1. Default is 0.8.
    val_weight : float, optional
        Proportion of events to use in the validation dataset.
        Weighting is normalised to the other supplied weightings to
        ensure the sum is 1. Default is 0.1.
    test_weight : float, optional
        Proportion of events to use in the test dataset. Weighting is
        normalised to the other supplied weightings to ensure the sum
        is 1. Default is 0.1.
    norm_kinematics : bool, optional
        Whether or not to normalise the kinematic properties. Default
        is True.
    norm_geometries : bool, optional
        Whether or not to normalise the geometric properties. Default
        is True.
    norm_beam : bool, optional
        Whether or not to normalise the beam connection properties.
        Default is True.
    classification_definitions : dict, optional
        Dictionary containing definitions of event classifications.
        Default is
        `DataPreparation.default_classification_definitions`.
    kinematic_definitions : dict, optional
        Dictionary containing definitions of kinematic PFO variables.
        Default is `DataPreparation.default_kinematic_definitions`.
    geometric_definitions : dict, optional
        Dictionary containing definitions of geometric properties
        between PFOs. Default is
        `DataPreparation.default_geometric_definitions`.
    beam_definitions : dict, optional
        Dictionary containing definitions of edge properties for edges
        connecting to the beam node between PFOs. Default is
        `DataPreparation.default_beam_definitions`.

    Returns
    -------
    dict:
        Dictionary containing the input parameters, and the calculated
        classification, kinematic, and geometric values, and completed
        save paths.
    """
    if directed_pairs is None:
        directed_pairs = generate_directed_pair_indicies(
            events, mc_true = data_type=="mc")
    params = {
        "events": events,
        "classifications": classifications,
        "kinematics": kinematics,
        "geometries": geometries,
        "directed_pairs": directed_pairs,
        "beam_node": beam_connections is not None,
        "beam_connections": beam_connections,
        "classification_definitions": classification_definitions,
        "kinematic_definitions": kinematic_definitions,
        "geometric_definitions": geometric_definitions,
        "momenta_definitions": momenta_definitions,
        "beam_definitions": beam_definitions,
        "norm_kinematics": norm_kinematics,
        "norm_geometries": norm_geometries,
        "norm_beam": norm_beam,
        "truth_info": truth_info,
        "split_pfos": split_pfos,
        "data_type": data_type,
        "momenta": momenta,
        "use_momenta": momenta is not None,
        "norm_momenta": norm_momenta}
    # if not isinstance(truth_info, list):
    #     if truth_info:
    #         truth = ["context", "pfos", "beam", "neighbours"]
    #     else:
    #         truth = []
    # truth_options = ["truth_context", "truth_pfos", "truth_beam", "truth_neighbours"]
    # for t in truth_options:
    #     params.update({t: t[6:] in truth_info})
    if truth_info:
        params.update({
            "truth_context": make_context_truth_info(
                params["events"], params["classifications"],
                params["classification_definitions"],
                mc=params["data_type"] == "mc"),
            "truth_pfos": make_pfo_truth_info(
                params["events"], mc=params["data_type"] == "mc"),
            "truth_beam": make_beam_conn_truth_info(
                params["events"], mc=params["data_type"] == "mc"),
            "truth_neighbours": make_neighbour_truth_info(
                params["events"], params["directed_pairs"],
                mc=params["data_type"] == "mc")})

    weighting_sum = train_weight + val_weight + test_weight
    params.update({
        "train_weight": train_weight/weighting_sum,
        "val_weight": val_weight/weighting_sum,
        "test_weight": test_weight/weighting_sum})
    params.update({
        "classification_values":
        make_evt_classifications(params["events"], params["classifications"],
                                 params["classification_definitions"])})
    params.update({"id_values": make_evt_ids(params["events"])})
    params.update({
        "kinematic_values":
        make_evt_kinematics(params["events"], params["kinematics"],
                            params["kinematic_definitions"],
                            params["norm_kinematics"])})
    kine_mean, kine_std = get_property_normalisations(
        params["kinematic_values"], params["kinematics"])
    norms_dict = {"pfo_mean": kine_mean, "pfo_std": kine_std}
    params.update({
        "geometric_values":
        make_evt_geometries(params["events"], params["geometries"],
                            params["geometric_definitions"],
                            params["directed_pairs"],
                            params["norm_geometries"])})
    geom_mean, geom_std = get_property_normalisations(
        params["geometric_values"], params["geometries"])
    norms_dict.update({"neighbours_mean": geom_mean,
                       "neighbours_std": geom_std})
    if params["beam_node"]:
        params.update({
            "beam_values":
            make_beam_props(params["events"], params["beam_connections"],
                            params["beam_definitions"],
                            params["norm_beam"])})
        conn_mean, conn_std = get_property_normalisations(
        params["beam_values"], params["beam_connections"])
        norms_dict.update({"beam_connections_mean": conn_mean,
                           "beam_connections_std": conn_std})
    else:
        params.update({"beam_values": None})
    if params["use_momenta"]:
        params.update({
            "momentum_values":
            make_evt_momenta(params["events"], params["momenta"],
                             params["momenta_definitions"],
                             params["norm_momenta"])
        })
    params.update(create_filepath_dictionary(folder_path))
    with open(params["dict_path"], "wb") as f:
        dill.dump(params, f)
    with open(params["norm_path"], "w") as f:
        json.dump(_norms_json_formatter(norms_dict), f, indent=4)
    return params

def _norms_json_formatter(dict, invert=False):
    if invert:
        return {key: np.array([np.float32(v) for v in val])
                for key, val in dict.items()}
    return {key: [str(v) for v in val] for key, val in dict.items()}
    

def generate_event_graph(event_index, param_dict):
    """
    Generates a `tfgnn.GraphTensor` for the event at `event_index`
    based on the parameters specified in `param_dict`.

    Parameters
    ----------
    event_index : int
        Index of the event for which to generate the GraphTensor.
    param_dict : dict
        Dictionary containing the parameters used to create the graph.
        Created by `DataPreparation.create_parameter_dictionary()`.
    
    Returns
    -------
    tfgnn.GraphTensor
        Graph tensor encoding the event indexed by `event_index`.
    """
    class_tensor = get_event_index(
        event_index, param_dict["classification_values"])
    id_tensor = get_event_index(
        event_index, param_dict["id_values"])
    kine_tensor = get_event_index(
        event_index, param_dict["kinematic_values"],
        prop_order=param_dict["kinematics"])
    geom_tensor = get_event_index(
        event_index, param_dict["geometric_values"],
        prop_order=param_dict["geometries"])
    
    context_values = {"classification": class_tensor,
                      "id": id_tensor}

    pairs = param_dict["directed_pairs"]
    n_nodes = kine_tensor.shape[0]
    n_edges = geom_tensor.shape[0]
    pfo_features = {tfgnn.HIDDEN_STATE: kine_tensor}
    edge_features = {tfgnn.HIDDEN_STATE: geom_tensor}
    
    def get_dict_event_index(dict):
        res = {}
        for k, v in dict.items():
            res[k] = get_event_index(event_index, v)
        return res

    if param_dict["truth_info"]:
        context_values.update(
            get_dict_event_index(param_dict["truth_context"]))
        pfo_features.update(
            get_dict_event_index(param_dict["truth_pfos"]))
        edge_features.update(
            get_dict_event_index(param_dict["truth_neighbours"]))
    
    node_values = {
        "pfo": tfgnn.NodeSet.from_fields(
            sizes = tf.constant([n_nodes]),
            features=pfo_features)}
    edge_values={
        "neighbours":
            tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([n_edges]),
                features=edge_features,
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("pfo", pairs['0'][event_index]),
                    target=("pfo", pairs['1'][event_index])))}

    if param_dict["use_momenta"]:
        pfo_features.update({
            "momentum" : get_event_index(
                event_index, param_dict["momentum_values"])})
    if param_dict["beam_node"]:
        beam_conn_tensor = get_event_index(
            event_index, param_dict["beam_values"],
            prop_order=param_dict["beam_connections"])
        n_beam_conns = beam_conn_tensor.shape[0]
        beam_conn_features = {tfgnn.HIDDEN_STATE: beam_conn_tensor}
        if param_dict["truth_info"]:
            beam_conn_features.update(
                get_dict_event_index(param_dict["truth_beam"]))
        target_inds = np.arange(n_nodes)
            # ak.count(param_dict["events"].recoParticles.number[event_index]),
            # dtype=int)
        source_inds = np.full_like(target_inds, 0)
        node_values.update({
            "beam": tfgnn.NodeSet.from_fields(
                sizes = tf.constant([1]),
                features={tfgnn.HIDDEN_STATE: tf.constant([1.])})})
        edge_values.update({
            "beam_connections": tfgnn.EdgeSet.from_fields(
                sizes = tf.constant([n_beam_conns]),
                features=beam_conn_features,
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("beam", source_inds),
                    target=("pfo", target_inds)))})
    
    return tfgnn.GraphTensor.from_pieces(
        context = tfgnn.Context.from_fields(
            sizes = tf.constant([1]),
            features=context_values),
        node_sets=node_values,
        edge_sets=edge_values)
    
def load_params_dict(dict_path):
    """
    Load the params dict (or any dill file) from the supplied path.

    Parameters
    ----------
    dict_path : str
        Path to dill file to load
    
    Returns
    -------
    dict
        Loaded parameter dictionary
    """
    with open(dict_path, "rb") as f:
        params = dill.load(f)
    return params

def load_param_events(path, new_ntuples_folder=None, depth=-2):
    if isinstance(path, dict):
        path = path["dict_path"]
    with open(path, "rb") as f:
        load_params = dill.load(f)
    if new_ntuples_folder is None:
        return load_params["events"]
    else:
        evts = load_params["events"]
        old_path = evts.io.filename
        if new_ntuples_folder[-1] == "/":
            new_ntuples_folder = new_ntuples_folder[:-1]
        new_path = "/".join(
            [new_ntuples_folder] + old_path.split("/")[depth:])
        evts.io.filename = new_path
        evts.filename = new_path
        return evts


# =====================================================================
#            Functions to create and load TensorFlow records

def generate_graph_schema(params_dict):
    """
    Create a graph tensor spec and a sceham which is saved based on the
    `params_dict` supplied.

    Schema matches the graphs generated by
    `DataPreparation.generate_event_graph()`. Schema will be saved at
    the `"schema_path"` location defined in `params_dict`.

    Parameters
    ----------
    param_dict : dict
        Dictionary containing the parameters used to create the schema.
        Created by `DataPreparation.create_parameter_dictionary()`.
    
    Returns
    -------
    tfgnn.GraphTensorSpec
        Graph specification matching graphs generated by the events
        stored in `params_dict`.
    """
    pfo_features = {
        # HIDDEN_STATE denotes params the network will use
        tfgnn.HIDDEN_STATE: tf.TensorSpec(
            (None, len(params_dict["kinematics"])),
            tf.float32)}
    if params_dict["use_momenta"]:
        pfo_features.update({
            "momentum": tf.TensorSpec(
                (None, len(params_dict["momenta"])),
                tf.float32)})
    
    edge_features = {
                tfgnn.HIDDEN_STATE: tf.TensorSpec(
                    (None, len(params_dict["geometries"])),
                    tf.float32)}
    context_features = {
            "classification": tf.TensorSpec(
                shape=(1, len(params_dict["classifications"])),
                dtype=tf.float32),
            "id": tf.TensorSpec(
                shape=(1, params_dict["id_values"].shape[1]),
                dtype=params_dict["id_values"].dtype)}
    if params_dict["truth_info"]:
        for key in params_dict["truth_context"].keys():
            if key == "reco_class":
                context_features.update({
                    key: tf.TensorSpec(
                    shape=(1, len(params_dict["classifications"])),
                    dtype=tf.float32)})
            else:
                context_features.update({key: tf.TensorSpec(
                    shape=(1, 1),
                    dtype=tf.float32)})
        for key in params_dict["truth_pfos"].keys():
            pfo_features.update({key: tf.TensorSpec(
                shape=(None, 1),
                dtype=tf.float32)})
        for key in params_dict["truth_neighbours"].keys():
            edge_features.update({key: tf.TensorSpec(
                shape=(None, 1),
                dtype=tf.float32)})
    
    nodes_spec = {
        "pfo": tfgnn.NodeSetSpec.from_field_specs(
            features_spec=pfo_features,
            sizes_spec=tf.TensorSpec((1,), tf.int64))}
    edges_spec = {
        "neighbours": tfgnn.EdgeSetSpec.from_field_specs(
            features_spec=edge_features,
            sizes_spec=tf.TensorSpec((1,), tf.int64),
            adjacency_spec=(
                tfgnn.AdjacencySpec.from_incident_node_sets(
                    "pfo", "pfo")))}
    
    if params_dict["beam_node"]:
        nodes_spec.update({
            "beam": tfgnn.NodeSetSpec.from_field_specs(
                features_spec={#},
                    tfgnn.HIDDEN_STATE: tf.TensorSpec(
                        (1), tf.float32)},
                sizes_spec=tf.TensorSpec((1,), tf.int64))})
        beam_connection_features = {
            tfgnn.HIDDEN_STATE: tf.TensorSpec(
                (None, len(params_dict["beam_connections"])),
                tf.float32)}
        if params_dict["truth_info"]:
            for key in params_dict["truth_beam"].keys():
                beam_connection_features.update({key: tf.TensorSpec(
                    shape=(None, 1),
                    dtype=tf.float32)})
        edges_spec.update({
            "beam_connections": tfgnn.EdgeSetSpec.from_field_specs(
            features_spec=beam_connection_features,
            sizes_spec=tf.TensorSpec((1,), tf.int64),
            adjacency_spec=(
                tfgnn.AdjacencySpec.from_incident_node_sets(
                    "beam", "pfo")))})
    evt_graph_tensor_spec = tfgnn.GraphTensorSpec.from_piece_specs(
        context_spec = tfgnn.ContextSpec.from_field_specs(
            features_spec=context_features),
        node_sets_spec=nodes_spec,
        edge_sets_spec=edges_spec)
    with open(params_dict["schema_path"], "w") as f:
        f.write(str(
            tfgnn.create_schema_pb_from_graph_spec(evt_graph_tensor_spec)))
    return evt_graph_tensor_spec

def _write_graph_data(path, init_index, end_index, params_dict):
    """
    Writes out the graphs between `init_index` and `end_index` to
    `path`.
    """
    evt_range = range(init_index, end_index)
    if len(evt_range) == 0:
        return
    with tf.io.TFRecordWriter(path) as writer:
        for i in evt_range:
            this_graph = generate_event_graph(i, params_dict)
            this_as_example = tfgnn.write_example(this_graph)
            writer.write(this_as_example.SerializeToString())
    return


def generate_records(params_dict, max_total=None):
    """
    Generate training, validation, and test record files based on the
    parameters in the `params_dict`.

    Files are written out to `params_dict["train_path"]`,
    `params_dict["val_path"]`, andF+?W!8Kg13Q
    `params_dict["test_path"]` for training, validation and test
    records respectively.

    Parameters
    ----------
    param_dict : dict
        Dictionary containing the parameters used to create the
        records. Created by
        `DataPreparation.create_parameter_dictionary()`.
    max_total : int, optional
        Total numebr of events to use. Intended for debugging on
        smaller datasets. If None, uses all available events. Default
        is None.
    """
    if max_total is None:
        n_total = ak.num(params_dict["events"].recoParticles.number, axis=0)
    else:
        n_total = max_total
    i_train = int(n_total * params_dict["train_weight"])
    i_val = int(n_total * (params_dict["train_weight"]
                           + params_dict["val_weight"]))
    _write_graph_data(params_dict["train_path"],
                      0,       i_train, params_dict)
    _write_graph_data(params_dict["val_path"],
                      i_train, i_val,   params_dict)
    _write_graph_data(params_dict["test_path"],
                      i_val,   n_total, params_dict)
    return

def generate_training_data(params_dict):
    """
    Generate training, validation, and test record files with the
    corresponding schema to allow reading these files based on the
    parameters in the `params_dict`.

    Locations in which the files are written are returned.

    Parameters
    ----------
    param_dict : dict
        Dictionary containing the parameters used to create the data.
        Created by `DataPreparation.create_parameter_dictionary()`.
    
    Returns
    -------
    (str, str, str, str)
        Path to the schema, training record, validation record, and
        test record respectively.
    """
    _ = generate_graph_schema(params_dict)
    generate_records(params_dict)
    return (params_dict["schema_path"], params_dict["train_path"],
            params_dict["val_path"],    params_dict["test_path"])

def _get_reco_class(context):
    n_pions = context["bt_pions"]
    n_pi0s = context["bt_pi0s"]
    conds = 1. * np.array([
        int((n_pions + n_pi0s) == 0),
        int((n_pions == 0) * (n_pi0s==1)),
        int((n_pions == 1) * (n_pi0s==0)),
        int((n_pions + n_pi0s) >= 2)])
    return conds

_context_truths = [
    "mc_pions", "mc_photons", "mc_pi0s",
    "bt_pions", "bt_photons", "bt_pi0s", "reco_class"]
_pfo_truths = [
    "beam_daughter", "beam_granddaughter", "pi0_granddaughter",
    "beam_related", "beam_relevant", "pion", "photon",
    "beam_pion", "beam_photon"]
_neighbour_truths = ["true_pi0", "beam_pi0"]
_beam_conn_truths = [
    "true_daughter", "true_granddaughter", "pi0_granddaughter",
    "beam_related", "beam_relevant"]
known_truths = (_context_truths + _pfo_truths
                + _neighbour_truths + _beam_conn_truths)

def _make_decode_func(schema_path, extra_losses=None):
    """Create a decoding function to parse records"""
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(
        tfgnn.read_schema(schema_path))
    class_shape = graph_spec.context_spec["classification"].shape

    if (len(class_shape) > 1) and (class_shape[0] == 1):
        # If using a vector, want to remove the first dimension
        class_label = lambda graph_context: tf.squeeze(
            graph_context.pop('classification'), axis=0)
    else:
        class_label = lambda graph_context: graph_context.pop(
            'classification')
    if extra_losses is None:
        def append_extra_losses(graph, label):
            return graph, label
    else:
        def append_extra_losses(graph, label):
            labels = [label]
            context_values = [
                "mc_pions", "mc_photons", "mc_pi0s",
                "bt_pions", "bt_photons", "bt_pi0s", "reco_class"]
            pfo_values = [
                "beam_daughter", "beam_granddaughter", "pi0_granddaughter",
                "beam_related", "beam_relevant", "pion", "photon",
                "beam_pion", "beam_photon"]
            neighbour_values = ["true_pi0", "beam_pi0"]
            beam_conn_values = [
                "true_daughter", "true_granddaughter", "pi0_granddaughter",
                "beam_related", "beam_relevant"]
            new_context = graph.context.get_features_dict()
            # new_nodes = graph.
            # new_edges = graph.
            for loss in extra_losses:
                if loss in context_values:
                    info = tf.squeeze(
                        new_context.pop(loss), axis=0)
                    labels.append(info)
                elif loss in pfo_values:
                    info = graph.node_sets["pfo"].features[loss]
                    # info = tfgnn.keras.layers.Readout(
                    #     node_set_name="pfo", feature_name=loss)(graph)
                    labels.append(info)
                elif loss in neighbour_values:
                    info = graph.edge_sets["neighbours"].features[loss]
                    # info = tfgnn.keras.layers.Readout(
                    #     edge_set_name="neighbours", feature_name=loss)(graph)
                    labels.append(info)
                elif loss in beam_conn_values:
                    info = graph.edge_sets["beam_connections"].features[loss]
                    labels.append(info)
                else:
                    raise ValueError(f"Unknown loss: {loss}")
            new_graph = graph.replace_features(context=new_context)
            return new_graph, tuple(labels)
    def decode_func(record_bytes):
        graph = tfgnn.parse_single_example(graph_spec,
                                            record_bytes,
                                            validate=True)
        # Extract the classification sperately as truth data
        graph_context = graph.context.get_features_dict()
        label = class_label(graph_context)
        new_graph = graph.replace_features(context=graph_context)
        return append_extra_losses(new_graph, label)
    return decode_func

def _get_data_only(data, label):
    """Function to grab only data when used in `Dataset.map()`"""
    return data

def load_record(
        schema_path, record_path,
        no_label=False, extra_losses=None,
        start_ind=None, n_graphs=None):
    """
    Load the specified records as TensorFlow Datasets using the schema
    in `schema_path`.

    `record_path` can be iterable, or a string. If iterable, results
    will be a list of the same size.
    If `no_label` is `True`, only the data is returned, without a the
    `"classification"` context label.
    If both `no_label` and `record_path` are iterable, the
    classification label will be removed for each index where
    `no_label` is `True`.

    Parameters
    ----------
    schema_path : str
        Path to the schema file defining the format of the records.
    record_path : str or list [ str ]
        Path, or iterable of paths, to the records to be read. If
        iterable, output will be a tuple of the same size.
    no_label : bool or list [ bool ], optional
        Whether or not to include classification label data in the
        Dataset. If iterable of size `record_path`, the check will be
        applied per index.
    extra_losses : list [ str ], optional
        Extra truth information which should be appended to the basic
        classification information to facilitate additional losses in
        the network. See Examples below for available information.
        Currently, only context information works.
    start_ind, int, optional
        If passed, which graph to start reading from. Default is None.
    n_graphs : int, optional
        If passed, number of graphs to read, else reads all graphs.
        Default is None
    
    Returns
    -------
    tf.data.Dataset or tuple [ tf.data.Dataset ]
        Datasets loaded from the records. Is a tuple with the same size
        as `record_path`, if `record_path` is iterable.

    Examples
    --------
    Avaiable losses come from labels in the network used by the extra
    `truth_info`.
    
    Context available information is:
    [`mc_pions`, `mc_photons`, `mc_pi0s`, `bt_pions`, `bt_photons`,
    `bt_pi0s`]
    Note bt_... properties are only available for data type graphs.
    
    PFO available information is:
    [`beam_daughter`, `beam_granddaughter`, `pi0_granddaughter`,
    `beam_related`, `beam_relevant`, `pion`, `photon`, `beam_pion`,
    `beam_photon`, `pi0`, `beam_pi0`]
    Note the final two properties are only available for MC type data.

    Neighbour connection available information is:
    [`true_pi0`, `beam_pi0`]

    Beam connections available information is:
    [`true_daughter`, `true_granddaughter`, `pi0_granddaughter`,
    `beam_related`, `beam_relevant`]
    """
    decode_func = _make_decode_func(schema_path, extra_losses=extra_losses)
    def _load_dataset(path):
        raw = tf.data.TFRecordDataset([path])
        if start_ind is not None:
            raw = raw.skip(start_ind)
        if n_graphs is not None:
            raw = raw.take(n_graphs)
        return raw.map(decode_func)
    if isinstance(record_path, str):
        datasets = _load_dataset(record_path)
        if no_label:
            datasets = datasets.map(_get_data_only)
    else: # we iterate through record paths to get multiple datasets
        datasets = tuple(_load_dataset(path) for path in record_path)
        if not isinstance(no_label, bool):
            # Only remove label if the no_label is true at the index
            datasets = tuple(ds.map(_get_data_only)
                             if no_label[i] else ds
                             for i, ds in enumerate(datasets))
        elif no_label:
            datasets = tuple(ds.map(_get_data_only) for ds in datasets)
    return datasets


# =====================================================================
#                           Displaying graphs
# W.I.P.

def get_odd_circle_coords(n_points, r=1):
    extra_node = n_points%2 == 0
    n_points += extra_node
    rads = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = r * np.sin(rads)
    y = r * np.cos(rads)
    if extra_node:
        x = np.delete(x, n_points//2)
        y = np.delete(y, n_points//2)
    return x, y

def get_odd_circle_angs(n_points, r=1):
    extra_node = n_points%2 == 0
    n_points += extra_node
    rads = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    if extra_node:
        rads = np.delete(rads, n_points//2)
    return rads

def pos_ind_from_octant(octant):
    """Creates a sequence like [0, -1, -1, -1, 0, 1, 1, 1]"""
    return (octant % 4) // (8*(octant // 4) - 4)

def get_text_alignment(angle):
    octant = int(((angle + np.pi/8) // (np.pi/4)) % 8)
    horizontals = ["center", "left",  "left",  "left",
                   "center", "right", "right", "right"]
    verticals = ["bottom", "bottom", "center", "top",
                 "top",    "top",    "center", "bottom"]
    return {"horizontalalignment": horizontals[octant],
            "verticalalignment": verticals[octant]}

def feature_text(values, labels):
    text = []
    for val, lab in zip(values, labels):
        text.append(f"{lab}: {val:.4e}")
    "\n".join(text)
    return text

def print_graph(
        graph, params,
        show_node_props=True, show_edge_props=True):
    if not "kinematics" in params:
        params = load_params_dict(params["dict_path"])
    fig = plt.figure(figsize=(8,8))
    renderer = fig.canvas.get_renderer()
    n_pfos = graph.node_sets["pfo"].sizes.numpy()[0]
    has_beam = "beam" in graph.node_sets.keys()
    has_momenta = "momentum" in graph.node_sets["pfo"].features.keys()
    angs = get_odd_circle_angs(n_pfos)
    pfo_labels = params["kinematics"]
    pfo_features = graph.node_sets["pfo"].features[tfgnn.HIDDEN_STATE].numpy()
    if has_momenta:
        pfo_momenta = graph.node_sets["pfo"].features["momentum"].numpy()
        mom_labels = params["momenta"]
    pfo_pos = np.zeros((2, n_pfos))
    for pfo_i in range(pfo_features.shape[0]):
        ang = angs[pfo_i]
        pfo_pos[:, pfo_i] = np.array([np.sin(ang), np.cos(ang)])
        if show_node_props:
            text = feature_text(pfo_features[pfo_i], pfo_labels)
            if has_momenta:
                text += "\n------\n"
                text += feature_text(pfo_momenta[pfo_i], mom_labels)
            alignments = get_text_alignment(ang)
            text_pos = pfo_pos[:, pfo_i] * 1.1
            plt.text(*text_pos, text, color="mediumblue", **alignments)
    plt.plot(pfo_pos[0:1, pfo_i], pfo_pos[1:-1, pfo_i], "o", ms=400, color="mediumblue")
    return plt.show()


# =====================================================================
#                 MUTAG dataset definitions for testing

# Taken from :
# https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/intro_mutag_example.ipynb
mutag_tensor_spec = tfgnn.GraphTensorSpec.from_piece_specs(
    context_spec=tfgnn.ContextSpec.from_field_specs(features_spec={
                  'label': tf.TensorSpec(shape=(1,), dtype=tf.int32)
    }),
    node_sets_spec={
        'atoms':
            tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        tf.TensorSpec((None, 7), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32))
    },
    edge_sets_spec={
        'bonds':
            tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        tf.TensorSpec((None, 4), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    'atoms', 'atoms'))
    })

def _mutag_decode_func(record_bytes):
        graph = tfgnn.parse_single_example(
            mutag_tensor_spec, record_bytes, validate=True)
        context_features = graph.context.get_features_dict()
        label = context_features.pop('label')
        new_graph = graph.replace_features(context=context_features)
        return new_graph, label

def get_mutag_data(mutag_folder):
    train_path = os.path.join(mutag_folder, "train.tfrecords")
    val_path = os.path.join(mutag_folder, "val.tfrecords")
    train_ds = tf.data.TFRecordDataset([train_path]).map(_mutag_decode_func)
    val_ds = tf.data.TFRecordDataset([val_path]).map(_mutag_decode_func)
    return train_ds, val_ds
