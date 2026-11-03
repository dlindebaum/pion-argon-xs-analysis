"""NTuple and data container definitions."""
import argparse
import itertools
import warnings

from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum

import awkward as ak
import numpy as np
import pandas as pd

from particle import Particle
from rich import print

from python.analysis import cuts, io, vector

from python.analysis.common import timer, match_dimension

FileDescriptor = namedtuple("FileDescriptor", ["file", "type", "momentum_scale"])

class Sample(str, Enum):
    """Classify an input sample as simulated or real data.
    """
    MC = "mc"
    DATA = "data"


def __is_mask_compatiable__(var : any, mask : ak.Array, verbose : bool = False):
    """Check whether a boolean mask has dimensions compatible with an array.

        Args:
            var (any): Array or record array that the mask would index. For
                record arrays, the first field is used to determine shape.
            mask (ak.Array): Boolean mask whose event and nested dimensions are
                compared with ``var``.
            verbose (bool): Print the compared shapes and compatibility
                decisions when true. Defaults to False.

        Returns:
            bool: Whether ``var`` can be indexed with ``mask`` without a
                dimension or per-event length mismatch. Returns ``None`` only
                if shape analysis reaches no explicit compatibility branch.
    """
    mask_shape = ak.num(mask, -1)

    var_shape = ak.num(var, -1)
    if type(var) == ak.highlevel.Array:
        if len(var.fields) > 0:
            var_shape = ak.num(var[var.fields[0]], -1) # assume records are vectors only (should be true)

    if verbose: print(f"{var_shape=}")
    if verbose: print(f"{mask_shape=}")
    if (type(var_shape) == np.ndarray) and (type(mask_shape) == np.ndarray): # mask is flat, variable is flat
        if var_shape == mask_shape:
            if verbose: print("compatible")
            return True
        else:
            if verbose: print("not compatible, event numbers not the same")
            return False
    elif (type(var_shape) == np.ndarray) and (type(mask_shape) != np.ndarray): # mask is not flat, variable is flat
        if verbose: print("not compatible, mask and variable have different dimensions")
        return False
    elif (type(var_shape) != np.ndarray) and (type(mask_shape) == np.ndarray): # mask is flat, variable is not flat
        if match_dimension(mask_shape, var_shape) == mask_shape:
            if verbose: print("compatible")
            return True
        else:
            if verbose: print("not compatible, event numbers not the same")
            return False
    else: # mask is not flat, variable is not flat
        if ak.all(match_dimension(mask_shape, var_shape) == mask_shape):
            if verbose: print("compatible")
            return True
        else:
            if verbose: print("not compatible, length in higher dimensions not equal")
            return False
    return

def __generic_filter__(data, filters: list, verbose : bool = True):
    """ Applies boolean masks (filters) to data.

        Args:
            data (any): data class which has instance variables with compatible shapes to the filters
            filters (list): list of filters
            verbose (bool): Print the data and each variable being tested for
                filtering. Defaults to True.
    """
    if verbose: print(data)
    for f in filters:
        for var in vars(data):
            if var == "filters": continue # skip filters list
            if hasattr(getattr(data, var), "__getitem__") and (type(getattr(data, var)) != str):
                try:
                    if verbose: print(f"{var, getattr(data, var), type(getattr(data, var))=}")
                    if __is_mask_compatiable__(getattr(data, var), f, verbose):
                        setattr(data, var, getattr(data, var)[f])
                except:
                    if verbose: warnings.warn(f"Couldn't apply filters to {var}.")
    return


def file_len(file : str) -> int:
    """ Get number of events of an PDSP Analyser Ntuple file.

        Args:
            file (str): path of Ntuple.

        Returns:
            int: Number of events
    """
    return len(io.RootIO(file).get(["EventID", "event"]))


class NtupleType(str, Enum):
    """Identify the supported ntuple layouts and branch conventions.
    """
    PDSP = "PDSPAnalyser"
    SHOWER_MERGING = "shower_merging"


class Data:
    """ Class to hold all event information read from the NTuple.

        Args:
            file_descriptor (FileDescriptor): File path, ntuple layout, and
                momentum scale used to construct the reader. If omitted, no
                file-backed event data are loaded.
            n_events (int): Maximum number of entries to load; ``-1`` requests
                all available entries. Defaults to -1.
            start (int): First entry to read from the file. Must be
                non-negative. Defaults to 0.
            verbose (bool): Whether to print filtering and shape diagnostics.
                Defaults to False.

        Attributes:
            io (IO): IO class which is used for interfacing with the NTuple file (using uproot).
            filename (str): NTuple file path passed to the reader. Defaults to
                None when no file descriptor is supplied.
            n_events (int): Maximum number of events to study. Defaults to -1
                (all available events).
            start (int): Zero-based entry offset in the ROOT file. Defaults to
                0.
            run (ak.Array): run number of each event.
            sub_run (ak.Array): sub run number of each event.
            event_num (ak.Array): event number of each event.
            true_particles (TrueParticles): class to hold truth information.
            recoParticles (RecoParticles): class to hold reco information.
            true_particles_bt (TrueParticlesBT): class to hold backtracked truth information.

            Property Methods:
            sorted_true_energy_mask (ak.Array): Indices of true photons ordered
                by ascending energy.

            Methods:
            find_events (dict): Return run, subrun, and event identifiers
                selected by a boolean event mask.
            mc_matching (tuple | Data | None): Match reconstructed showers
                to true photons and either return masks, filter a copy, or
                filter this instance in place.
            match_by_angle_bt (tuple): Match showers to backtracked particles
                by angular separation and return indices with an event mask.
            create_specific_event_filter (ak.Array): Build a boolean mask from
                requested subrun and event-number pairs.
            filter (Data | None): Apply reconstructed and truth filters in
                place or return a filtered copy.
            apply_beam_filter (None): Filter the event sample to entries with
                a beam PFO and a valid beam endpoint.
            merge_pfo_cheat (tuple): Merge PFPs that backtrack to the same
                truth particle and return the merged data with a validity mask.
            merge_shower (Data): Merge showers using reconstructed information.
            merge_shower_bt (Data): Merge showers using backtracked matching.
    """

    def __init__(self, file_descriptor: FileDescriptor = None, n_events: int = -1, start: int = 0, verbose : bool = False):

        """Create a file-backed event container and its particle-data views.

            Args:
                file_descriptor (FileDescriptor): File path, ntuple layout, and
                    momentum scale used to construct the reader. If omitted, no
                    file-backed event data are loaded.
                n_events (int): Maximum number of entries to load; ``-1`` requests
                    all available entries. Defaults to -1.
                start (int): First entry to read from the file. Must be
                    non-negative. Defaults to 0.
                verbose (bool): Whether to print filtering and shape diagnostics.
                    Defaults to False.
        """
        self.verbose = verbose
    
        if start < 0:
            raise ValueError("start cannot be less than zero")

        if file_descriptor is None:
            self.filemane = None
            self.ntuple_type = None
            self.momentum_scale = None
        else:
            self.filename = file_descriptor.file

            if (self.filename is not None) and (file_descriptor.type is None):
                warnings.warn(f"ntuple type is not specified, assuming it is {NtupleType.PDSP}")
                self.ntuple_type = NtupleType.SHOWER_MERGING # TODO remove shower merging related data IO
            else:
                self.ntuple_type = file_descriptor.type
            if (self.filename != None):
                if file_descriptor.momentum_scale is None:
                    if ("_data_" in self.filename):
                        self.momentum_scale = 1
                    else:
                        if "GeV" in self.filename:
                            self.momentum_scale = float(self.filename.split("GeV")[0][-1])
                        else:
                            self.momentum_scale = file_descriptor.momentum_scale
                else:
                    self.momentum_scale = file_descriptor.momentum_scale
                self.n_events = n_events
                self.start = start
                self.io = io.RootIO(self.filename, self.n_events, self.start)
                self.run = self.io.get(["Run", "run"])
                self.sub_run = self.io.get(["SubRun", "subrun"])
                self.event_num = self.io.get(["EventID", "event"])
                if self.event_num is not None:
                    self.event_index = ak.local_index(self.event_num) + start # unique index for each event
                self.true_particles = TrueParticleData(self)
                self.recoParticles = RecoParticleData(self)
                self.true_particles_bt = TrueParticleDataBT(self)
                self.cut_table = cuts.CutHandler(self)

    @property
    def sorted_true_energy_mask(self) -> ak.Array:
        """Return indices of true photons ordered by ascending energy.

            Returns:
                ak.Array: Per-event indices produced by sorting the true
                    photon energies from lowest to highest.
        """
        return ak.argsort(self.true_particles.energy[self.true_particles.true_photon_mask], ascending=True)

    def find_events(self, event_filter: ak.Array = None) -> dict:
        """ Returns the run, subrun and event number for events which pass a particlular filter.

            Args:
                event_filter (ak.Array): mask which selects events (1D array)

            Returns:
                dict: event, subrun and run.
        """
        if ak.ravel(event_filter).type.type != ak.types.PrimitiveType("bool"):
            raise TypeError(
                f'event_filter must have primitive type "bool", but has primitive type {event_filter.type.type}')
        if ak.count(event_filter) != ak.count(self.event_num):
            raise Exception(
                f"event_fitler doesn't have an array length equal to the number of events")
        return {"event": self.event_num[event_filter], "subrun": self.sub_run[event_filter], "run": self.run[event_filter]}

    @timer
    def mc_matching(self, cut = 0.25, apply_filters: bool = True, return_copy: bool = False):
        """ Function which matched reco showers to true photons for pi0 decays.
            Does so by looking at angular separation, it will create masks which
            can be applied to self to select matched showers, or return them. ALso
            calculates mask of unmatched showers, needed for shower merging.

            Args:
                cut (float, optional): cut on angle for event selection. Defaults to 0.25.
                apply_filters (bool): if true applies matching filters to self, false returns the filters for the user to use. Defaults to True.
                return_copy (bool): if true returns a filtered copy of the class, false applies to self. Defaults to False.

            Returns:
                tuple | Data | None: If ``apply_filters`` is false, returns
                ``(matched_mask, unmatched_mask, selection_mask)``. If filters
                are applied with ``return_copy=True``, returns a filtered
                ``Data`` copy; otherwise it filters this instance and returns
                ``None``.
        """
        # angle of all reco showers wrt to each true photon per event i.e. error
        null_shower_dir = self.recoParticles.shower_direction.x == -999
        photon_dir = vector.normalize(self.true_particles.momentum)[
            self.true_particles.true_photon_mask]
        angle_0 = vector.angle(self.recoParticles.shower_direction, photon_dir[:, 0])
        angle_0 = ak.where(null_shower_dir == True, 1E8, angle_0)
        angle_1 = vector.angle(self.recoParticles.shower_direction, photon_dir[:, 1])
        angle_1 = ak.where(null_shower_dir == True, 1E8, angle_1)
        # create array of indices to keep track of shower index
        ind = ak.sort(ak.argsort(angle_0, -1), -1)

        # get smallest angle wrt to each true photon
        m_0 = ak.unflatten(ak.min(angle_0, -1), 1)
        m_1 = ak.unflatten(ak.min(angle_1, -1), 1)

        # get the first photon with the smallest angle
        first_matched_photon = ak.argmin(
            ak.concatenate([m_0, m_1], -1), -1, keepdims=True)
        # get the other photon which needs to be matched
        remaining_photon = ak.where(first_matched_photon == 0, 1, 0)

        # get the index of the matched shower
        m_0 = ind[m_0 == angle_0]
        m_1 = ind[m_1 == angle_1]
        # get index of the matched shower wrt to each photon
        first_matched_shower = ak.where(first_matched_photon == 0, m_0, m_1)

        # get angles of remaining showers to look at
        remaining_showers_0 = ak.flatten(
            angle_0[first_matched_shower]) != angle_0
        remaining_showers_1 = ak.flatten(
            angle_1[first_matched_shower]) != angle_1
        new_angle_0 = angle_0[remaining_showers_0]
        new_angle_1 = angle_1[remaining_showers_1]

        # get index of next closest shower
        m_0 = ak.unflatten(ak.min(new_angle_0, -1), 1)
        m_1 = ak.unflatten(ak.min(new_angle_1, -1), 1)
        m_0 = ind[m_0 == angle_0]
        m_1 = ind[m_1 == angle_1]
        # get index of the matched shower wrt to each photon
        second_matched_shower = ak.where(remaining_photon == 0, m_0, m_1)

        # concantenate matched photons in each pass, then get the indices in which they should be sorted to preserve the order of photons in MC truth
        photon_order = ak.argsort(ak.concatenate(
            [first_matched_photon, remaining_photon], -1), -1)
        # concantenate matched showers and then sort by photon number
        matched_mask = ak.concatenate(
            [first_matched_shower, second_matched_shower], -1)[photon_order]

        # get unmatched showers by checking which showers are matched
        t_0 = matched_mask[:, 0] == ind
        t_1 = matched_mask[:, 1] == ind
        # use logical not to get the showers not in matched mask i.e. what is unmatched
        unmatched_mask = np.logical_not(np.logical_or(t_0, t_1))

        # get events where both reco MC angles are less than the threshold value
        angle_0 = vector.angle(
            self.recoParticles.shower_direction[matched_mask][:, 0], photon_dir[:, 0])
        angle_1 = vector.angle(
            self.recoParticles.shower_direction[matched_mask][:, 1], photon_dir[:, 1])
        selection = np.logical_and(angle_0 < cut, angle_1 < cut)

        # either apply the filters or return them
        if apply_filters is True:
            if return_copy is True:
                return self.filter([matched_mask, selection], [selection], return_copy=True)
            else:
                self.filter([matched_mask, selection], [
                            selection], return_copy=False)
        else:
            return matched_mask, unmatched_mask, selection

    @timer
    def match_by_angle_bt(self) -> tuple:
        """ Equivlant to shower matching/angluar closeness cut, but for backtracked MC.

            Returns:
                tuple: indices of reco particles with the smallest angular closeness ak.Array: boolean mask of events which pass the angle cut

            Raises:
                Exception: if all reco PFP's backtracked to the same true particle
        """
        null_direction = self.recoParticles.shower_direction.x == - \
            999  # boolean mask of PFP's with undefined direction
        null_momentum = self.recoParticles.shower_momentum.x == - \
            999  # boolean mask of PFP's with undefined momentum
        null_position = self.recoParticles.shower_start_pos.x == -999
        null = np.logical_or(null_direction, null_momentum)
        null = np.logical_or(null, null_position)
        # calculate angular closeness
        angle_error = vector.angle(
            self.recoParticles.shower_direction, self.true_particles_bt.direction)
        # if direction is undefined, angler error is massive (so not the best match)
        angle_error = ak.where(null, 999, angle_error)
        # create index array of angles to use later
        ind = ak.local_index(angle_error, -1)

        # get unique true particle numbers per event i.e. the photons which the reco PFP's backtrack to
        mcIndex = self.true_particles_bt.partice_number
        unqiueIndex = self.true_particles_bt.get_unique_particle_numbers(mcIndex)

        if(ak.any(ak.num(unqiueIndex) == 1)):
            raise Exception(
                "data contains events with reco particles matched to only one photon, did you forget to apply singleMatch filter?")

        # get PFP's which match to the same true particle
        mcp_0 = mcIndex == unqiueIndex[:, 0]
        mcp_1 = mcIndex == unqiueIndex[:, 1]

        # get the smallest angle error of each sorted PFP's
        angle_error_0 = ak.min(angle_error[mcp_0], -1)
        angle_error_1 = ak.min(angle_error[mcp_1], -1)

        # create boolean mask for the event selection
        selection_bt = np.logical_and(
            angle_error_0 < 0.25, angle_error_1 < 0.25)

        # get recoPFP indices which had the smallest angular closeness
        indices_0 = ind[angle_error == angle_error_0]
        indices_1 = ind[angle_error == angle_error_1]
        best_match = ak.concatenate([indices_0, indices_1], -1)
        best_match = best_match[:, 0:2]
        return best_match, selection_bt

    def create_specific_event_filter(self, event_nums: list) -> ak.Array:
        """ Create boolean mask which selects specific events and subruns to look at.
            example: if you want to look at subrun 1 events 5, 6, 8 then event_nums is
            [(1, 5), (1, 6), (1, 8)]
            or the first event of subruns 1, 40 and 268:
            [(1, 1), (40, 1), (268, 1)]

            Args:
                event_nums (list): list of tuples of the form (subrun, event_num)

            Returns:
                ak.Array: boolean mask of selected events
        """
        re = ak.concatenate([ak.unflatten(self.sub_run, 1),
                            ak.unflatten(self.event_num, 1)], 1)
        f = np.logical_and(re[:, 0] == event_nums[0][0],
                           re[:, 1] == event_nums[0][1])
        for i in range(1, len(event_nums)):
            f = np.logical_or(f, np.logical_and(
                re[:, 0] == event_nums[i][0], re[:, 1] == event_nums[i][1]))
        return f

    def filter(self, reco_filters: list = [], true_filters: list = [], return_copy: bool = False):
        """ Filter events or entries per event.

            Args:
                reco_filters (list): list of filters to apply to reconstructed data. Defaults to [].
                true_filters (list): list of filters to apply to true data. Defaults to [].
                return_copy (bool): if true return a copy of filtered events, else change self

            Returns:
                Data | None: A filtered ``Data`` copy when ``return_copy`` is
                true; otherwise filters this instance in place and returns
                ``None``.
        """
        if return_copy is False:
            self.true_particles.filter(true_filters, return_copy)
            self.recoParticles.filter(reco_filters, return_copy)
            if hasattr(self, "true_particles_bt"):
                # has same shape as reco data
                self.true_particles_bt.filter(reco_filters, return_copy)

            __generic_filter__(self, reco_filters, self.verbose)
            cuts.apply_filters(self, reco_filters)
            self.true_particles.events = self
            self.recoParticles.events = self
            if hasattr(self, "true_particles_bt"):
                self.true_particles_bt.events = self
        else:
            # copy filtered attributes into new instance
            filtered = Data(FileDescriptor(self.filename, self.ntuple_type, self.momentum_scale))
            filtered.filename = self.filename
            filtered.verbose = self.verbose
            filtered.momentum_scale = self.momentum_scale
            filtered.n_events = self.n_events
            filtered.start = self.start
            filtered.io = io.RootIO(filtered.filename,
                             filtered.n_events, filtered.start)
            filtered.cut_table = self.cut_table.copy()
            filtered.event_index = self.event_index
            filtered.event_num = self.event_num
            filtered.sub_run = self.sub_run
            filtered.run = self.run
            filtered.true_particles = self.true_particles.filter(true_filters)
            filtered.recoParticles = self.recoParticles.filter(reco_filters)
            filtered.recoParticles.events = filtered
            filtered.true_particles.events = filtered
            if hasattr(self, "true_particles_bt"):
                filtered.true_particles_bt = self.true_particles_bt.filter(
                    reco_filters)
                filtered.true_particles_bt.events = filtered
            # ? should true_filters also be applied?
            __generic_filter__(filtered, reco_filters, self.verbose)
            cuts.apply_filters(filtered, reco_filters)
            return filtered

    @timer
    def apply_beam_filter(self):
        """ Applies a beam filter to the sample, which selects objects
            in events which are the beam particle.
        """
        if self.recoParticles.beam_number is None:
            print("data doesn't contain beam number, can't apply filter.")
            return
        hasBeam = self.recoParticles.beam_number != - \
            999  # check if event has a beam particle
        hasBeam = np.logical_and(
            self.recoParticles.beam_end_pos.x != -999, hasBeam)
        self.filter([hasBeam], [hasBeam])  # filter data

    @timer
    def merge_pfo_cheat(self, merge_method: int = 0):
        """ Merges all PFPs which backtrack to the same True particle.

            Args:
                merge_method (int): method to calculate merged momenta/energy
        """
        def sum_hits(hits: ak.Array):
            """Sum hit values along the innermost axis.

                Args:
                    hits (ak.Array): Per-PFO hit-count array to combine for
                        each merged particle.

                Returns:
                    ak.Array: Hit-value sums for each event and merged
                        particle.
            """
            sum_hits = ak.where(hits < 0, 0, hits)  # exclude null data
            sum_hits = ak.sum(hits, -1)
            return sum_hits

        mc_index = self.true_particles_bt.number
        if mc_index is None:
            # required for data missing backtracked number (pure pi0 sample)
            mc_index = self.true_particles_bt.partice_number

        unique_index = self.true_particles_bt.get_unique_particle_numbers(
            mc_index)  # get unique list of true particles

        max_pfos = ak.max(ak.num(unique_index))
        unique_index = ak.pad_none(unique_index, max_pfos)

        merged_data = {
            "truePFPMask": None,
            "p": None,
            "e": None,
            "dir": None,
            "n_hits": None,
            "true_hits": None,
            "shared_hits": None,
            "reco_cluster_hits": None,
            "true_collection_hits": None,
            "shared_collection_hits": None,
            "reco_cluster_collection_hits": None,
            "purity": None,
            "completeness": None,
            "purity_collection": None,
            "completeness_collection": None,
        }

        for i in range(max_pfos):
            data = {}
            # sort pfps based on which true particle they back-track to
            pfos = mc_index == unique_index[:, i]
            # get index of backtracked particle
            data["truePFPMask"] = ak.local_index(pfos, -1)
            data["truePFPMask"] = ak.min(data["truePFPMask"][pfos], -1)

            p = self.recoParticles.shower_momentum[pfos]
            # dont merge PFP's with null data
            p = ak.where(p.x == -999, {"x": 0, "y": 0, "z": 0}, p)
            if merge_method == 0:
                # sum momenta and use magitude as energy and norm as direction
                data["p"] = ak.sum(p, -1)  # sum all momentum vectors
                data["e"] = vector.magnitude(data["p"])
                data["dir"] = vector.normalize(data["p"])
            if merge_method == 1:
                # sum energy, sum momenta, use norm as direction and summed energy * direction as summed momenta
                e = self.recoParticles.shower_energy[pfos]
                e = ak.where(e < 0, 0, e)
                data["e"] = ak.sum(e, -1)
                data["dir"] = vector.normalize(ak.sum(p, -1))
                data["p"] = vector.prod(data["e"], data["dir"])

            data["n_hits"] = sum_hits(self.recoParticles.n_hits_collection[pfos])

            # recompute the purity/completeness of the PFO
            # true hits for all PFOs are the same
            data["true_hits"] = self.true_particles_bt.n_hits[pfos][:, 0]
            if self.true_particles_bt.n_hits_collection is not None:
                data["true_collection_hits"] = self.true_particles_bt.n_hits_collection[pfos][:, 0]

            data["shared_hits"] = sum_hits(
                self.true_particles_bt.sharedHits[pfos])
            if self.true_particles_bt.shared_hits_collection is not None:
                data["shared_collection_hits"] = sum_hits(
                    self.true_particles_bt.shared_hits_collection[pfos])

            data["reco_cluster_hits"] = sum_hits(
                self.true_particles_bt.hit_in_reco_cluster[pfos])
            if self.true_particles_bt.hit_in_reco_cluster_collection is not None:
                data["reco_cluster_collection_hits"] = sum_hits(
                    self.true_particles_bt.hit_in_reco_cluster_collection[pfos])

            data["purity"] = data["shared_hits"] / data["reco_cluster_hits"]
            data["completeness"] = data["shared_hits"] / data["true_hits"]
            if "shared_collection_hits" in data.keys():
                data["purity_collection"] = data["shared_collection_hits"] / \
                    data["reco_cluster_collection_hits"]
                data["completeness_collection"] = data["shared_collection_hits"] / \
                    data["true_collection_hits"]

            for k in data:
                if i == 0:
                    merged_data[k] = ak.unflatten(data[k], 1, -1)
                else:
                    merged_data[k] = ak.concatenate(
                        [merged_data[k], ak.unflatten(data[k], 1, -1)], axis=-1)

        merged_data["truePFPMask"] = ak.fill_none(
            merged_data["truePFPMask"], -999)
        merged_data["truePFPMask"] = merged_data["truePFPMask"][merged_data["truePFPMask"] != -999]

        # ? should we implement corrections for other values like start position or n_hits?
        self.recoParticles._RecoParticleData__shower_momentum = merged_data["p"]
        self.recoParticles._RecoParticleData__shower_energy = merged_data["e"]
        self.recoParticles._RecoParticleData__shower_direction = merged_data["dir"]
        self.recoParticles._RecoParticleData__n_hits = merged_data["n_hits"]

        self.true_particles_bt._TrueParticleDataBT__n_hits = merged_data["true_hits"]
        self.true_particles_bt._TrueParticleDataBT__sharedHits = merged_data["shared_hits"]
        self.true_particles_bt._TrueParticleDataBT__hit_in_reco_cluster = merged_data[
            "reco_cluster_hits"]
        self.true_particles_bt._TrueParticleDataBT__purity = merged_data["purity"]
        self.true_particles_bt._TrueParticleDataBT__completeness = merged_data["completeness"]

        self.true_particles_bt._TrueParticleDataBT__n_hits_collection = merged_data[
            "true_collection_hits"]
        self.true_particles_bt._TrueParticleDataBT__shared_hits_collection = merged_data[
            "shared_collection_hits"]
        self.true_particles_bt._TrueParticleDataBT__hit_in_reco_cluster_collection = merged_data[
            "reco_cluster_collection_hits"]
        self.true_particles_bt._TrueParticleDataBT__purity_collection = merged_data[
            "purity_collection"]
        self.true_particles_bt._TrueParticleDataBT__completeness_collection = merged_data[
            "completeness_collection"]

        # filter to get the merged PFOs only
        self.filter([merged_data["truePFPMask"]])

        # exlucde events where all PFP's merged had no valid momentmum vector
        null = np.logical_not(ak.any(self.recoParticles.shower_energy == 0, -1))
        print(
            f"Events where one merged PFP had undefined momentum: {ak.count(null[np.logical_not(null)])}")

    def merge_shower(self, matched: ak.Array, unmatched: ak.Array):
        """ Merge shower not matched to MC to the spatially closest matched shower.

            Args:
                matched (ak.Array): matched shower indicies
                unmatched (ak.Array): boolean mask of unmatched showers

            Returns:
                Data: events with matched reco showers after merging
        """

        events_matched = self.filter([matched], return_copy=True)
        # filter reco for matched/unmatched only
        unmatched_reco = self.filter(
            [unmatched], return_copy=True).recoParticles
        # should only be needed for unmatched sample
        null_dir = unmatched_reco.shower_direction.x == -999

        angle_0 = ak.where(null_dir == True, 1E8, vector.angle(
            unmatched_reco.shower_direction, events_matched.recoParticles.shower_direction[:, 0]))
        angle_1 = ak.where(null_dir == True, 1E8, vector.angle(
            unmatched_reco.shower_direction, events_matched.recoParticles.shower_direction[:, 1]))
        angle_0 = ak.unflatten(angle_0, 1, -1)
        angle_1 = ak.unflatten(angle_1, 1, -1)
        angle = ak.concatenate([angle_0, angle_1], -1)
        mergeMask = ak.min(angle, -1) == angle

        # * create Array which contains the amount of energy to merge to the showers
        # * will be zero for the shower we don't want to merge to
        null = {"x": 0, "y": 0, "z": 0}
        to_merge = ak.where(mergeMask, unmatched_reco.shower_momentum, null)
        to_merge = ak.where(to_merge.x != -999, to_merge, null)
        merge_0 = ak.unflatten(ak.sum(to_merge[:, :, 0], -1), 1, -1)
        merge_1 = ak.unflatten(ak.sum(to_merge[:, :, 1], -1), 1, -1)
        momentum_to_merge = ak.concatenate([merge_0, merge_1], -1)
        momentum_to_merge = ak.where(
            events_matched.recoParticles.shower_momentum.x != -999, momentum_to_merge, null)
        new_momentum = vector.add(
            events_matched.recoParticles.shower_momentum, momentum_to_merge)
        events_matched.recoParticles._RecoParticleData__shower_momentum = new_momentum

        new_direction = vector.normalize(events_matched.recoParticles.shower_momentum)
        new_direction = ak.where(events_matched.recoParticles.shower_momentum.x !=
                                -999, new_direction, vector.null_vector)
        events_matched.recoParticles._RecoParticleData__shower_direction = new_direction

        new_energy = vector.magnitude(events_matched.recoParticles.shower_momentum)
        events_matched.recoParticles._RecoParticleData__shower_energy = ak.where(
            events_matched.recoParticles.shower_momentum.x != -999, new_energy, -999)

        return events_matched

    def merge_shower_bt(self, best_match: ak.Array):
        """ Merge showers using bactacked matching.

            Args:
                best_match (ak.Array): mask of starting showers

            Returns:
                Data: events with merged showers
        """
        if bool(ak.all(ak.num(self.recoParticles.shower_energy[self.recoParticles.shower_energy != -999]) == 2)) is True:
            counts = 0
        else:
            counts = 1
        # * get boolean mask of PFP's to merge
        index = ak.local_index(self.recoParticles.shower_energy)
        to_merge = [ak.where(index == best_match[:, i], False, True)
                    for i in range(2)]
        to_merge = np.logical_and(*to_merge)

        # * calculate angle between each best matched shower and the showers to merge
        best_match_dir = self.recoParticles.shower_direction[best_match]
        to_merge_dir = self.recoParticles.shower_direction[to_merge]
        not_null = to_merge_dir.x != -999
        to_merge_dir = to_merge_dir[not_null]

        angles = [ak.unflatten(vector.angle(
            to_merge_dir, best_match_dir[:, i]), counts, -1) for i in range(2)]
        angles = ak.concatenate(angles, -1)
        # check which angle is the smallest i.e. decide which main shower to merge to
        minMask_angle = ak.min(angles, -1) == angles

        # * merge momenta
        momentumToMerge = self.recoParticles.shower_momentum[to_merge]
        momentumToMerge = momentumToMerge[not_null]
        valid_mom = momentumToMerge.x != -999  # we don't want to merge invalid PFP's
        momentumToMerge = momentumToMerge[valid_mom]
        minMask_angle = minMask_angle[valid_mom]
        minMask_angle = [minMask_angle[:, :, i] for i in range(2)]

        momentumToMerge = [
            ak.sum(momentumToMerge[minMask_angle[i]], -1) for i in range(2)]
        leading_momentum = self.recoParticles.shower_momentum[best_match]

        # add the merged momenta to their respective leading showers
        mergedMomentum = [ak.unflatten(vector.add(
            leading_momentum[:, i], momentumToMerge[i]), 1, -1) for i in range(2)]
        mergedMomentum = ak.concatenate(mergedMomentum, -1)

        mergedDirection = [ak.unflatten(vector.normalize(
            mergedMomentum[:, i]), 1, -1) for i in range(2)]
        merged_direction = ak.concatenate(mergedDirection, -1)

        merged_energy = [ak.unflatten(vector.magnitude(
            mergedMomentum[:, i]), 1, -1) for i in range(2)]
        merged_energy = ak.concatenate(merged_energy, -1)

        merged_events = self.filter(return_copy=True)
        merged_events.recoParticles._RecoParticleData__shower_momentum = mergedMomentum
        merged_events.recoParticles._RecoParticleData__shower_direction = merged_direction
        merged_events.recoParticles._RecoParticleData__shower_energy = merged_energy

        merged_events.true_particles_bt.filter([best_match], return_copy=False)
        return merged_events


    def save_selected_event_indices(self, output_path : str = "selected_events.hdf5"):
        """ Save selected event indices.

            Args:
                output_path (str): HDF5 output path for the selected event indices.
                    Defaults to ``selected_events.hdf5``.
        """
        pd.DataFrame(self.event_index).to_hdf(output_path, "df")
        print(f"event indices saved to : {output_path}")

    def load_selected_events(self, indices_file : str):
        """ Load selected events.

            Args:
                indices_file (str): HDF5 file containing event indices to retain.
        """
        selection = io.read_hdf5(indices_file).values.flatten()

        mask = []
        for i in self.event_index:
            mask.append(any(i == selection))
        mask = ak.Array(mask)

        self.filter([mask], [mask])

    @property
    def table(self, **kwargs):
        """Return the cut table generated for this event sample.

            Args:
                **kwargs (Any): Keyword arguments forwarded to
                    ``cut_table.get_table``.

            Returns:
                Any: Table returned by ``cut_table.get_table(**kwargs)``.
        """
        return self.cut_table.get_table(**kwargs)



class ParticleData(ABC):
    """ Template for data classes which store particle information in events.

        Args:
            events (Data): Parent event container that supplies the ntuple
                reader, verbosity setting, and event-level filters.

        Attributes:
            events (Data): parent class.
            filters (list): list of filters applied/to apply to data

            Abstrct Methods:
            __init__:
            calculate_pair_quantities: Calculate Shower pair quantities. #! shower pair quantities should be its own data class
            Methods:
            read_data: Reads data from NTuple to hidden attribute specific to the subclass.
            Filter (object): Filter data.
    """
    @abstractmethod
    def __init__(self, events: Data) -> None:
        """Attach the parent event container and initialize queued filters.

            Args:
                events (Data): Parent event container whose ntuple data are
                    exposed by this particle-data object.
        """
        self.events = events  # keep reference of parent class
        self.filters = []  # list of filters to apply to newly loaded data
        pass

    @abstractmethod
    def calculate_pair_quantities(self):
        """Return the particle-pair quantities implemented by a concrete subclass.
        """
        pass


    def filter_variable(self, var : str):
        """Apply queued event masks to a loaded particle-data attribute.

            Args:
                var (str): Name of the hidden particle-data attribute to which
                    queued filters should be applied.
        """
        var_name = f"_{type(self).__name__}__{var}"
        for f in self.filters:
            try:
                if __is_mask_compatiable__(getattr(self, var_name), f, self.events.verbose):
                    setattr(self, var_name, getattr(self, var_name)[f])
            except:
                Warning(f"couldn't apply a filter to {var}")


    def read_data(self, name: str, ntuple_name, is_vector : bool = False):
        """ Reads data from ntuple and assigns it to a hidden variable
            which can then be called by the property method.

            Args:
                name (str): variable name
                ntuple_name (Any): Branch name to load, or three branch names
                    for the x, y, and z components when ``is_vector`` is true.
                is_vector (bool): Zip the three loaded branches into an
                    Awkward record array when true. Defaults to False.
        """
        var_name = f"_{type(self).__name__}__{name}"  # variable name which chose to be hidden
        if hasattr(self.events, "io") and not hasattr(self, var_name):
            if is_vector:
                setattr(self, var_name, ak.zip({
                    "x": self.events.io.get(ntuple_name[0]),
                    "y": self.events.io.get(ntuple_name[1]),
                    "z": self.events.io.get(ntuple_name[2])
                }))
            else:
                setattr(self, var_name, self.events.io.get(ntuple_name))
            # apply any filters to data when loading
            self.filter_variable(name)


    def filter(self, filters: list, return_copy: bool = True):
        """Apply event masks to particle-data attributes in place or on a copy.

            Args:
                filters (list): list of filters to apply to particle data.
                return_copy (bool): Return a new instance with the filters
                    applied when true; otherwise mutate this object. Defaults
                    to True.

            Returns:
                ParticleData | None: A filtered instance when
                ``return_copy`` is true; otherwise applies the filters in place
                and returns ``None``.
        """
        if return_copy is False:
            __generic_filter__(self, filters, self.events.verbose)
            # append list of filters to apply to newly loaded data
            self.filters.extend(filters)
        else:
            # get the class which is of type ParticleData
            subclass = globals()[type(self).__name__]
            filtered = subclass(Data())
            # populate new instance
            for var in vars(self):
                setattr(filtered, var, getattr(self, var))
            __generic_filter__(filtered, filters, self.events.verbose)
            filtered.filters = list(self.filters + filters)
            return filtered


class TrueParticleData(ParticleData):
    """ Particle data class for truth information.

        Args:
            events (Data): Parent event container whose true-particle branches
                are exposed by this object.

        Attributes:
            (1 hidden attribute for each property method)
            Property Methods:
            pdg (ak.Array): pdg code
            number (ak.Array): particle number
            mother (ak.Array): number of mother particle
            mother_pdg (ak.Array): mother pdg code
            mass (ak.Array):
            energy (ak.Array):
            momentum (ak.Record):
            direction (ak.Record):
            shower_start_pos (ak.Record):
            end_pos (ak.Record):
            pi0_mc (bool): check if we are looking at pure pi0 MC or beam MC
            primary_pi0_mask (ak.Array): Pi0 produced from beam or generated by particle gun
            true_photon_mask (ak.Array): get mask of photons which are daughter of the primary pi0

            Methods:
            calculate_pair_quantities (tuple): Calculate true shower pair quantities.
    """

    def __init__(self, events: Data):
        """Initialize the truth-particle view of the parent event sample.

            Args:
                events (Data): Parent event container whose true-particle
                    branches are exposed by this object.
        """
        super().__init__(events)

    def pdsp_data(self, name : str, beam : str, daughter : str, grand_daughter : str):
        """Combine a PDSP beam branch with its daughter and granddaughter branches.

            Args:
                name (str): Hidden attribute name in which to cache the
                    concatenated values.
                beam (str): Branch containing the beam particle's values.
                daughter (str): Branch containing daughter-particle values.
                grand_daughter (str): Branch containing granddaughter values.
        """
        var_name = f"_{type(self).__name__}__{name}"
        if hasattr(self.events, "io") and not hasattr(self, var_name):
            value = ak.concatenate([ak.unflatten(self.events.io.get(beam), 1, -1), self.events.io.get(daughter), self.events.io.get(grand_daughter)], -1)
            setattr(self, var_name, value)
            self.filter_variable(name)

    def pdsp_data_vector(self, name : str, beam : str, daughter : str, grand_daughter : str):
        """Build a three-component vector from PDSP beam and daughter branches.

            Args:
                name (str): Hidden attribute name in which to cache the vector.
                beam (str): Three branch names for the beam particle's x, y,
                    and z components.
                daughter (str): Three branch names for the daughter
                    particles' x, y, and z components.
                grand_daughter (str): Three branch names for the
                    granddaughter particles' x, y, and z components.
        """
        var_name = f"_{type(self).__name__}__{name}"
        if hasattr(self.events, "io") and not hasattr(self, var_name):
            bv = [self.events.io.get(n) for n in beam]
            dv = [self.events.io.get(n) for n in daughter]
            gv = [self.events.io.get(n) for n in grand_daughter]
            value = [ak.concatenate([ak.unflatten(bv[i], 1, -1), dv[i], gv[i]], -1) for i in range(len(beam))]
            value = vector.vector(x = value[0], y = value[1], z = value[2])
            setattr(self, var_name, value)
            self.filter_variable(name)

    @property
    def beam_traj_KE(self) -> ak.Array:
        """Return the true beam kinetic energy at each recorded trajectory point.

            Returns:
                ak.Array: Beam-trajectory kinetic-energy values from the
                    ``true_beam_traj_KE`` branch.
        """
        self.read_data("beam_traj_KE", "true_beam_traj_KE")
        return getattr(self, f"_{type(self).__name__}__beam_traj_KE")

    @property
    def beam_traj_pos(self) -> ak.Record:
        """Return the true beam trajectory positions as x, y, and z components.

            Returns:
                ak.Record: Vector record assembled from the true beam trajectory
                    coordinate branches.
        """
        self.read_data("beam_traj_pos", ["true_beam_traj_X", "true_beam_traj_Y", "true_beam_traj_Z"], is_vector = True)
        return getattr(self, f"_{type(self).__name__}__beam_traj_pos")

    @property
    def pdg(self) -> ak.Array:
        """Return the PDG code of each true particle.

            Returns:
                ak.Array: Particle PDG codes assembled from PDSP beam,
                    daughter, and granddaughter branches or read from the
                    shower-merging ``g4_Pdg`` branch.
        """
        if self.events.ntuple_type == NtupleType.PDSP:
            self.pdsp_data("pdg", "true_beam_PDG", "true_beam_daughter_PDG", "true_beam_Pi0_decay_PDG")
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            self.read_data("pdg", "g4_Pdg")
        return getattr(self, f"_{type(self).__name__}__pdg")

    @property
    def number(self) -> ak.Array:
        """Return the identifier of each true particle.

            Returns:
                ak.Array: Particle identifiers assembled from PDSP branches or
                    read from the shower-merging ``g4_num`` branch.
        """
        if self.events.ntuple_type == NtupleType.PDSP:
            self.pdsp_data("number", "true_beam_ID", "true_beam_daughter_ID", "true_beam_Pi0_decay_ID")
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            self.read_data("number", "g4_num")
        return getattr(self, f"_{type(self).__name__}__number")

    @property
    def mother(self) -> ak.Array:
        """Return the mother-particle identifier for each true particle.

            Returns:
                ak.Array: Mother identifiers read from the shower-merging
                    branch or constructed for PDSP beam, daughter, and
                    granddaughter particles.
        """
        var_name = f"_{type(self).__name__}__mother"
        if self.events.ntuple_type == NtupleType.PDSP and not hasattr(self, var_name):
            beam = ak.Array([0] * ak.count(self.events.io.get("true_beam_ID"))) #! need to check null values for these
            daughter = ak.Array([[1]*i for i in ak.num(self.events.io.get("true_beam_daughter_ID"))]) #! need to check null values for these
            grand_daughter = self.events.io.get("true_beam_Pi0_decay_parID")
            mother = ak.concatenate([ak.unflatten(beam, 1, -1), daughter, grand_daughter], -1)
            setattr(self, var_name, mother)
            self.filter_variable("mother")
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            self.read_data("mother", "g4_mother")
        return getattr(self, var_name)

    @property
    def mother_pdg(self) -> ak.Array:
        """Return the PDG code of each true particle's mother.

            Returns:
                ak.Array: Mother PDG codes from ``g4_mother_Pdg`` for
                    shower-merging ntuples. PDSP ntuples raise
                    ``AttributeError`` because this field is unavailable.
        """
        if self.events.ntuple_type == NtupleType.PDSP:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                + "'mother_pdg' for 'PDSP' ntuple type.")
        self.read_data("mother_pdg", "g4_mother_Pdg") # PDSPAnalyzer does not contain this information
        return getattr(self, f"_{type(self).__name__}__mother_pdg")

    @property
    def mass(self) -> ak.Array:
        """Return each true particle's mass.

            Returns:
                ak.Array: Mass values read from the shower-merging branch or
                    obtained by mapping PDG codes to particle masses for PDSP
                    ntuples.
        """
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            self.read_data("mass", "g4_mass") # PDSPAnalyzer does not contain this information, but we can create this ourselves
        if self.events.ntuple_type == NtupleType.PDSP:
            unique_pdgs = np.unique(ak.ravel(self.events.true_particles.pdg)) # get all unique pdg codes

            # make a "mask" for each pdg code, mapping pdg code to mass
            masses = []
            for p in unique_pdgs:
                m = Particle.from_pdgid(p).mass 
                if m is None: m = 0
                masses.append(ak.where(self.events.true_particles.pdg == p, m, -1))

            # merge the "masks" into the mass array
            mass = None
            for m in masses:
                if type(mass) == ak.highlevel.Array:
                    mass = ak.where(mass == -1, m, mass)
                else:
                    mass = m
            setattr(self, f"_{type(self).__name__}__mass", mass)
            self.filter_variable("mass")
        return getattr(self, f"_{type(self).__name__}__mass")

    @property
    def energy(self) -> ak.Array:
        """Return each true particle's energy.

            Returns:
                ak.Array: Energy values read from the shower-merging branch or
                    calculated for PDSP particles from momentum magnitude and
                    mass.
        """
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            self.read_data("energy", "g4_startE") # PDSPAnalyzer does not contain this information, but we can create this ourselves
        if self.events.ntuple_type == NtupleType.PDSP:
            e = (vector.magnitude(self.momentum) ** 2 - self.mass ** 2) ** 0.5
            setattr(self, f"_{type(self).__name__}__energy", e)
            self.filter_variable("energy")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def momentum(self) -> ak.Record:
        """Return each true particle's three-momentum.

            Returns:
                ak.Record: Momentum vector built from PDSP start-momentum
                    branches (scaled to MeV) or shower-merging component
                    branches.
        """
        var_name = f"_{type(self).__name__}__momentum"
        apply_scaling = not hasattr(self, var_name)
        if self.events.ntuple_type == NtupleType.PDSP:
            self.pdsp_data_vector("momentum", ["true_beam_startPx", "true_beam_startPy", "true_beam_startPz"], ["true_beam_daughter_startPx", "true_beam_daughter_startPy", "true_beam_daughter_startPz"], ["true_beam_Pi0_decay_startPx", "true_beam_Pi0_decay_startPy", "true_beam_Pi0_decay_startPz"])
            if apply_scaling: # PDSP analyser stores true energy information in GeV
                setattr(self, var_name, vector.prod(1000, getattr(self, var_name)))
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            self.read_data("momentum", ["g4_pX", "g4_pY", "g4_pZ"], is_vector = True)
        return getattr(self, var_name)

    @property
    def direction(self) -> ak.Record:
        """Return unit direction vectors derived from true-particle momentum.

            Returns:
                ak.Record: Normalized momentum vectors with x, y, and z
                    components.
        """
        if not hasattr(self, f"_{type(self).__name__}__direction"):
            self.__direction = vector.normalize(self.momentum)
        return self.__direction

    @property
    def start_pos(self) -> ak.Record:
        """Return the starting position of each true particle.

            Returns:
                ak.Record: Position vectors assembled from PDSP start-coordinate
                    branches or shower-merging start-coordinate branches.
        """
        if self.events.ntuple_type == NtupleType.PDSP:
            self.pdsp_data_vector("shower_start_pos", ["true_beam_startX", "true_beam_startY", "true_beam_startZ"], ["true_beam_daughter_startX", "true_beam_daughter_startY", "true_beam_daughter_startZ"], ["true_beam_Pi0_decay_startX", "true_beam_Pi0_decay_startY", "true_beam_Pi0_decay_startZ"])
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            self.read_data("shower_start_pos", ["g4_startX", "g4_startY", "g4_startZ"], is_vector = True)
        return getattr(self, f"_{type(self).__name__}__shower_start_pos")

    @property
    def end_pos(self) -> ak.Record:
        """Return the endpoint of each true particle.

            Returns:
                ak.Record: Position vectors assembled from PDSP end-coordinate
                    branches or shower-merging end-coordinate branches.
        """
        if self.events.ntuple_type == NtupleType.PDSP:
            self.pdsp_data_vector("end_pos", ["true_beam_endX", "true_beam_endY", "true_beam_endZ"], ["true_beam_daughter_endX", "true_beam_daughter_endY", "true_beam_daughter_endZ"], ["true_beam_Pi0_decay_startX", "true_beam_Pi0_decay_startY", "true_beam_Pi0_decay_startZ"])
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            self.read_data("end_pos", ["g4_endX", "g4_endY", "g4_endZ"], is_vector = True)
        return getattr(self, f"_{type(self).__name__}__end_pos")

    @property
    def n_pi0(self) -> type:
        """Return the PDSP count of daughter pi-zero particles.

            Returns:
                type: Values read from ``true_daughter_nPi0``. This field is
                    unavailable in shower-merging ntuples.
        """
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                + f"'n_pi0' for {NtupleType.SHOWER_MERGING} ntuple type.")
        self.read_data("n_pi0", "true_daughter_nPi0")
        return getattr(self, f"_{type(self).__name__}__n_pi0")

    @property
    def n_pi_plus(self) -> type:
        """Return the PDSP count of daughter positive pions.

            Returns:
                type: Values read from ``true_daughter_nPiPlus``. This field is
                    unavailable in shower-merging ntuples.
        """
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                + f"'n_pi_plus' for {NtupleType.SHOWER_MERGING} ntuple type.")
        self.read_data("n_pi_plus", "true_daughter_nPiPlus")
        return getattr(self, f"_{type(self).__name__}__n_pi_plus")

    @property
    def n_pi_minus(self) -> type:
        """Return the PDSP count of daughter negative pions.

            Returns:
                type: Values read from ``true_daughter_nPiMinus``. This field
                    is unavailable in shower-merging ntuples.
        """
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                + f"'n_pi_minus' for {NtupleType.SHOWER_MERGING} ntuple type.")
        self.read_data("n_pi_minus", "true_daughter_nPiMinus")
        return getattr(self, f"_{type(self).__name__}__n_pi_minus")

    @property
    def n_proton(self) -> type:
        """Return the PDSP count of daughter protons.

            Returns:
                type: Values read from ``true_daughter_nProton``. This field is
                    unavailable in shower-merging ntuples.
        """
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                + f"'n_proton' for {NtupleType.SHOWER_MERGING} ntuple type.")
        self.read_data("n_proton", "true_daughter_nProton")
        return getattr(self, f"_{type(self).__name__}__n_proton")

    @property
    def n_neutron(self) -> type:
        """Return the PDSP count of daughter neutrons.

            Returns:
                type: Values read from ``true_daughter_nNeutron``. This field
                    is unavailable in shower-merging ntuples.
        """
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                + f"'n_neutron' for {NtupleType.SHOWER_MERGING} ntuple type.")
        self.read_data("n_neutron", "true_daughter_nNeutron")
        return getattr(self, f"_{type(self).__name__}__n_neutron")

    @property
    def n_nucleus(self) -> type:
        """Return the PDSP count of daughter nuclei.

            Returns:
                type: Values read from ``true_daughter_nNucleus``. This field
                    is unavailable in shower-merging ntuples.
        """
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                + f"'n_nucleus' for {NtupleType.SHOWER_MERGING} ntuple type.")
        self.read_data("n_nucleus", "true_daughter_nNucleus")
        return getattr(self, f"_{type(self).__name__}__n_nucleus")

    @property
    def true_beam_end_process(self) -> ak.Array:
        """Return the true beam particle's recorded end-process code.

            Returns:
                ak.Array: Values from the ``true_beam_endProcess`` branch.
        """
        self.read_data("true_beam_end_process", "true_beam_endProcess")
        return getattr(self, f"_{type(self).__name__}__true_beam_end_process")

    @property
    def beam_ind_in_tpc(self) -> ak.Array:
        """Return the first true beam trajectory index with positive z.

            Returns:
                ak.Array: Per-event index of the first trajectory point whose
                    z coordinate is greater than zero.
        """
        return ak.argmax(self.beam_traj_pos.z > 0, -1, keepdims = True)

    @property
    def beam_KE_front_face(self) -> ak.Array:
        """Return true beam kinetic energy at the first positive-z trajectory point.

            Returns:
                ak.Array: Per-event kinetic energy selected by
                    ``beam_ind_in_tpc`` from ``beam_traj_KE``.
        """
        return ak.flatten(self.beam_traj_KE[self.beam_ind_in_tpc])

    @property
    def in_tpc_z(self) -> ak.Array:
        """Identify true beam trajectory points inside the configured z range.

            Returns:
                ak.Array: Boolean mask true where ``0 < z < 700``.
        """
        return (self.beam_traj_pos.z > 0) & (self.beam_traj_pos.z < 700)

    @property
    def beam_track_length(self) -> ak.Array:
        """Sum consecutive true beam trajectory distances within the z range.

            Returns:
                ak.Array: Per-event track length calculated from points selected
                    by ``in_tpc_z``.
        """
        pos_in_tpc = self.beam_traj_pos[self.in_tpc_z]
        return ak.sum(vector.dist(pos_in_tpc[:, 1:], pos_in_tpc[:, :-1]), -1)


    @property
    def pi0_mc(self) -> bool:
        """Indicate whether the truth record contains a primary pi-zero particle.

            Returns:
                bool: Whether any particle has identifier 1 and PDG code 111.
        """
        if not hasattr(self, f"_{type(self).__name__}__pi0_mc"):
            # check if we are looking at pure pi0 MC or beam MC
            self.__pi0_mc = ak.any(np.logical_and(
                self.number == 1, self.pdg == 111))
        return self.__pi0_mc

    @property
    def primary_pi0_mask(self) -> ak.Array:
        """Select primary pi-zero particles in the current truth sample.

            Returns:
                ak.Array: Boolean mask selecting particle 1 for pi0-gun MC, or
                    particles with mother 1 and PDG code 111 for beam MC.
        """
        if self.pi0_mc:
            return np.logical_and(self.number == 1, self.pdg == 111)
        else:
            # assume we have beam MC
            # get pi0s produced from beam interaction
            return np.logical_and(self.mother == 1, self.pdg == 111)

    @property
    def true_photon_mask(self) -> ak.Array:
        """Select true photons produced by the primary pi-zero decay.

            Returns:
                ak.Array: Boolean mask selecting primary photon daughters with
                    PDG code 22, using the primary pi-zero identifiers for
                    beam MC samples.
        """
        if self.pi0_mc:
            photons = self.mother == 1  # get only primary daughters
            photons = np.logical_and(photons, self.pdg == 22)
        else:
            # assume we have beam MC, with potentially more than 1 pi0
            primary_pi0 = self.primary_pi0_mask
            # get particle number of each pi0
            primary_pi0_num = self.number[primary_pi0]
            # get the largest number of primary pi0s per event
            n = ak.max(ak.num(primary_pi0_num))
            # pad empty elements with None so we can do index slicing
            primary_pi0_num = ak.pad_none(primary_pi0_num, n)

            # * loop through slices of pi0s per event, and get all particles which are their daughters
            null = self.number == -1  # never should be negative so all values are false
            primary_daughters = self.mother == primary_pi0_num[:, 0]
            # check none since boolean logic ignores None values
            primary_daughters = ak.where(ak.is_none(
                primary_daughters), null, primary_daughters)

            i = 1
            while i < n:
                next_primaries = self.mother == primary_pi0_num[:, i]
                next_primaries = ak.where(ak.is_none(
                    next_primaries), null, next_primaries)
                primary_daughters = np.logical_or(
                    primary_daughters, next_primaries)
                i += 1
            photons = np.logical_and(
                primary_daughters, self.pdg == 22)  # only want photons
        return photons

    def calculate_pair_quantities(self) -> tuple:
        """ Calculate true shower pair quantities.

            Returns:
                tuple: calculated quantities
        """
        mask_pi0 = self.primary_pi0_mask
        if ak.all(ak.num(mask_pi0[mask_pi0]) <= 1) == False:
            raise ValueError(
                "function currently only works for samples with 1 pi0 per event.")
        photons = self.true_photon_mask
        sort_energy = self.events.sorted_true_energy_mask

        # * compute start momentum of daughters
        p_daughter = self.momentum[photons]
        sum_p = ak.sum(p_daughter, axis=1)
        sum_p = vector.magnitude(sum_p)
        p_daughter_mag = vector.magnitude(p_daughter)
        p_daughter_mag = p_daughter_mag[sort_energy]

        # * compute true opening angle
        angle = np.arccos(vector.dot(
            p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

        # * compute invariant mass
        e_daughter = self.energy[photons]
        inv_mass = (2 * e_daughter[:, 1:] *
                    e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

        # * pi0 momentum
        p_pi0 = self.momentum[mask_pi0]
        p_pi0 = vector.magnitude(p_pi0)
        return inv_mass, angle, p_daughter_mag[:, 1:], p_daughter_mag[:, :-1], p_pi0


class RecoParticleData(ParticleData):
    """ Particle data class for reconstructed information.

        Args:
            events (Data): Parent event container whose reconstructed-particle
                branches are exposed by this object.

        Attributes:
            (1 hidden attribute for each property method)

            Property Methods:
            beam_number (ak.Array): beam PFO number
            slice_id (ak.Array): slice the PFO corresponds to
            beam_slice_id (ak.Array): slice the beam PFO corresponds to
            beam_cosmic_score (ak.Array): whether the reconstruction chain used was for a neutrino vertex (beam) or cosmics
            beam_end_pos (ak.Record): end point of the beam particle
            beam_start_pos (ak.Record): start point of the beam particle
            beam_calo_wire (ak.Array): wire numbers of beam calorimetry
            beam_michel_score (ak.Array): beam particle michel score
            beam_n_hits (ak.Array): beam particle number of hits
            beam_dEdx (ak.Array): beam particle dEdx
            pandora_tag (ak.Array): label given to particles by pandora; track, shower or -999
            number (ak.Array): PFO number
            mother (ak.Array): number of mother PFO
            n_hits (ak.Array): number of hits
            n_hits_collection (ak.Array): number of collection plane hits
            energy (ak.Array):
            momentum (ak.Record):
            shower_direction (ak.Record):
            shower_start_pos (ak.Record):
            shower_length (ak.Array): length of shower (if applicable)
            showerConeAngle (ak.Array): width of shower (if applicable)
            em_score (ak.Array): shower like score
            track_score (ak.Array): track like score
            cnn_score (ak.Array): em_score/(em_score + track_score)
            space_points (ak.Record): hit space points
            channel (ak.Array): hit channel
            peak_time (ak.Array): hit peak time
            integral (ak.Array): hit integral
            hit_energy (ak.Array): hit energy
            track_dEdX (ak.Array): dEdX (if PFO is interpreted as track like)


            Methods:
            calculate_pair_quantities (tuple): Calculate reconstructed shower pair quantities.
            get_pair_values (ak.Array): Get shower pair values, in pairs
            all_shower_pairs (list): Get all unique shower pair combinations.
            shower_pairs_by_hits (ak.list): get shower pairs by leading in energy #! not used
    """

    def __init__(self, events: Data) -> None:
        """Initialize the reconstructed-particle view of the parent event sample.

            Args:
                events (Data): Parent event container whose reconstructed
                    particle branches are exposed by this object.
        """
        super().__init__(events)

    @property
    def beam_number(self) -> ak.Array:
        """Return the PFO identifier assigned to the beam particle.

            Returns:
                ak.Array: Per-event beam PFO identifiers read from the
                    available beam-number branch.
        """
        self.read_data("beam_number", ["beamNum", "reco_beam_PFP_ID"])
        return getattr(self, f"_{type(self).__name__}__beam_number")

    @property
    def beam_slice_id(self) -> ak.Array:
        """Return the reconstruction slice identifier of the beam PFO.

            Returns:
                ak.Array: Per-event values from ``reco_beam_slice_id``.
        """
        self.read_data("beam_slice_id", "reco_beam_slice_id") # dont store this in PDSPAnalyser
        return getattr(self, f"_{type(self).__name__}__beam_slice_id")

    @property
    def beam_cosmic_score(self) -> ak.Array:
        """Return the beam candidate's cosmic-classification score.

            Returns:
                ak.Array: Per-event values from
                    ``reco_daughter_allShower_beamCosmicScore``.
        """
        self.read_data("beam_cosmic_score",
                      "reco_daughter_allShower_beamCosmicScore") # dont store this in PDSPAnalyser
        return getattr(self, f"_{type(self).__name__}__beam_cosmic_score")

    @property
    def beam_end_pos_sce(self) -> ak.Record:
        """Return the space-charge-corrected beam endpoint.

            Returns:
                ak.Record: Per-event x, y, and z endpoint coordinates loaded
                    from the SCE-corrected beam branches.
        """
        ntuples = [
            ["reco_beam_calo_endX"],
            ["reco_beam_calo_endY"],
            ["reco_beam_calo_endZ"]
        ]
        self.read_data("beam_end_pos_sce", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__beam_end_pos_sce")

    @property
    def beam_end_pos(self) -> ak.Record:
        """Return the uncorrected beam endpoint.

            Returns:
                ak.Record: Per-event x, y, and z endpoint coordinates loaded
                    from the beam endpoint branches.
        """
        ntuples = [
            ["reco_beam_endX"],
            ["reco_beam_endY"],
            ["reco_beam_endZ"]
        ]
        self.read_data("beam_end_pos", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__beam_end_pos")

    @property
    def beam_start_pos_sce(self) -> ak.Record:
        """Return the space-charge-corrected beam start position.

            Returns:
                ak.Record: Per-event x, y, and z start coordinates loaded from
                    the SCE-corrected beam branches.
        """
        ntuples = [
            ["reco_beam_calo_startX"], # the first name in the list is prioritised.
            ["reco_beam_calo_startY"],
            ["reco_beam_calo_startZ"]
        ]
        self.read_data("beam_start_pos_sce", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__beam_start_pos_sce")

    @property
    def beam_start_pos(self) -> ak.Record:
        """Return the uncorrected beam start position.

            Returns:
                ak.Record: Per-event x, y, and z start coordinates loaded from
                    the beam start-position branches.
        """
        ntuples = [
            ["reco_beam_startX"], # the first name in the list is prioritised.
            ["reco_beam_startY"],
            ["reco_beam_startZ"]
        ]
        self.read_data("beam_start_pos", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__beam_start_pos")

    @property
    def beam_calo_pos(self) -> ak.Record:
        """Return the beam calorimetry position coordinates.

            Returns:
                ak.Record: Per-event x, y, and z positions assembled from
                    ``reco_beam_calo_X``, ``reco_beam_calo_Y``, and
                    ``reco_beam_calo_Z``.
        """
        self.read_data("beam_calo_pos", ["reco_beam_calo_X", "reco_beam_calo_Y", "reco_beam_calo_Z"], is_vector = True)
        return getattr(self, f"_{type(self).__name__}__beam_calo_pos")

    @property
    def beam_calo_wire(self) -> ak.Array:
        """Return wire numbers for beam calorimetry points.

            Returns:
                ak.Array: Per-event wire values from ``reco_beam_calo_wire``.
        """
        self.read_data("beam_calo_wire", "reco_beam_calo_wire")
        return getattr(self, f"_{type(self).__name__}__beam_calo_wire")

    @property
    def beam_michel_score(self) -> ak.Array:
        """Return the beam-vertex Michel-electron score.

            Returns:
                ak.Array: Per-event values from
                    ``reco_beam_vertex_michel_score``.
        """
        self.read_data("beam_michel_score", "reco_beam_vertex_michel_score")
        return getattr(self, f"_{type(self).__name__}__beam_michel_score")

    @property
    def beam_track_score(self) -> ak.Array:
        """Return the collection-plane track score for the beam PFO.

            Returns:
                ak.Array: Per-event values from
                    ``reco_beam_PFP_trackScore_collection``.
        """
        self.read_data("beam_track_score", "reco_beam_PFP_trackScore_collection")
        return getattr(self, f"_{type(self).__name__}__beam_track_score")

    @property
    def beam_n_hits(self) -> ak.Array:
        """Return the number of hits associated with the beam vertex.

            Returns:
                ak.Array: Per-event values from ``reco_beam_vertex_nHits``.
        """
        self.read_data("beam_n_hits", "reco_beam_vertex_nHits")
        return getattr(self, f"_{type(self).__name__}__beam_n_hits")

    @property
    def beam_dEdX(self) -> ak.Array:
        """Return space-charge-corrected beam dE/dx values.

            Returns:
                ak.Array: Per-event values from ``reco_beam_dEdX_SCE``.
        """
        self.read_data("beam_dEdX", "reco_beam_dEdX_SCE")
        return getattr(self, f"_{type(self).__name__}__beam_dEdX")

    @property
    def beam_dEdX_calibrated(self) -> ak.Array:
        """Return calibrated, space-charge-corrected beam dE/dx values.

            Returns:
                ak.Array: Per-event values from
                    ``reco_beam_calibrated_dEdX_SCE``.
        """
        self.read_data("beam_dEdX_calibrated", "reco_beam_calibrated_dEdX_SCE")
        return getattr(self, f"_{type(self).__name__}__beam_dEdX_calibrated")

    @property
    def reco_reconstructable_beam_event(self) -> ak.Array:
        """Return the beam-event reconstructability flag.

            Returns:
                ak.Array: Per-event values from
                    ``reco_reconstructable_beam_event``.
        """
        self.read_data("reco_reconstructable_beam_event", "reco_reconstructable_beam_event")
        return getattr(self, f"_{type(self).__name__}__reco_reconstructable_beam_event")

    @property
    def beam_inst_valid(self) -> ak.Array:
        """Return the validity flag for beam instrumentation data.

            Returns:
                ak.Array: Per-event values from ``beam_inst_valid``.
        """
        self.read_data("beam_inst_valid", "beam_inst_valid")
        return getattr(self, f"_{type(self).__name__}__beam_inst_valid")

    @property
    def beam_inst_n_tracks(self) -> ak.Array:
        """Return the number of instrumentation tracks in each event.

            Returns:
                ak.Array: Per-event values from ``beam_inst_nTracks``.
        """
        self.read_data("beam_inst_n_tracks", "beam_inst_nTracks")
        return getattr(self, f"_{type(self).__name__}__beam_inst_n_tracks")

    @property
    def beam_inst_n_momenta(self) -> ak.Array:
        """Return the number of beam-instrumentation momentum measurements.

            Returns:
                ak.Array: Per-event values from ``beam_inst_nMomenta``.
        """
        self.read_data("beam_inst_n_momenta", "beam_inst_nMomenta")
        return getattr(self, f"_{type(self).__name__}__beam_inst_n_momenta")

    @property
    def beam_inst_trigger(self) -> ak.Array:
        """Return the beam-instrumentation trigger values.

            Returns:
                ak.Array: Per-event values from ``beam_inst_trigger``.
        """
        self.read_data("beam_inst_trigger", "beam_inst_trigger")
        return getattr(self, f"_{type(self).__name__}__beam_inst_trigger")

    @property
    def beam_inst_momentum(self) -> ak.Array:
        """Return the beam momentum measured by the instrumentation.

            Returns:
                ak.Array: Per-event values from ``beam_inst_P``.
        """
        self.read_data("beam_inst_momentum", "beam_inst_P")
        return self.events.momentum_scale * 1000 * getattr(self, f"_{type(self).__name__}__beam_inst_momentum")

    @property
    def beam_inst_dir(self) -> ak.Record:
        """Return the beam-instrumentation direction vectors.

            Returns:
                ak.Record: Per-event x, y, and z components assembled from
                    instrumentation direction branches.
        """
        ntuples = [
            ["beam_inst_dirX"], # the first name in the list is prioritised.
            ["beam_inst_dirY"],
            ["beam_inst_dirZ"]
        ]
        self.read_data("beam_inst_dir", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__beam_inst_dir")


    @property
    def beam_inst_pos(self) -> ak.Record:
        """Return the beam-instrumentation position vectors.

            Returns:
                ak.Record: Per-event x, y, and z coordinates assembled from
                    ``beam_inst_X``, ``beam_inst_Y``, and ``beam_inst_Z``.
        """
        self.read_data("beam_inst_pos", ["beam_inst_X", "beam_inst_Y", "beam_inst_Z"], is_vector = True)
        return getattr(self, f"_{type(self).__name__}__beam_inst_pos")

    @property
    def beam_inst_pdg_candidates(self) -> ak.Array:
        """Return the PDG hypotheses recorded by beam instrumentation.

            Returns:
                ak.Array: Per-event candidate values from
                    ``beam_inst_PDG_candidates``.
        """
        self.read_data("beam_inst_pdg_candidates", "beam_inst_PDG_candidates")
        return getattr(self, f"_{type(self).__name__}__beam_inst_pdg_candidates")

    @property
    def pandora_tag(self) -> ak.Array:
        """Return Pandora's particle-type tag for each reconstructed PFO.

            Returns:
                ak.Array: Tag values from ``pandoraTag`` identifying the
                    reconstruction classification.
        """
        self.read_data("pandora_tag", "pandoraTag") # This information is not stored for daughter PFOs in PDSPAnalyser
        return getattr(self, f"_{type(self).__name__}__pandora_tag")

    @property
    def slice_id(self) -> ak.Array:
        """Return the reconstruction slice identifier for each PFO.

            Returns:
                ak.Array: Values from ``reco_daughter_allShower_sliceID``.
        """
        self.read_data("slice_id", "reco_daughter_allShower_sliceID") # dont store this in PDSPAnalyser
        return getattr(self, f"_{type(self).__name__}__slice_id")

    @property
    def number(self) -> ak.Array:
        """Return the identifier of each reconstructed PFO.

            Returns:
                ak.Array: PFO identifiers from the available reconstructed
                    particle ID branch.
        """
        self.read_data("number", ["reco_PFP_ID", "reco_daughter_PFP_ID"])
        return getattr(self, f"_{type(self).__name__}__number")

    @property
    def mother(self) -> ak.Array:
        """Return the parent PFO identifier for each reconstructed PFO.

            Returns:
                ak.Array: Values from ``reco_PFP_Mother``.
        """
        self.read_data("mother", "reco_PFP_Mother") # not needed for PDSPAnalyser (only beam daughter PFOs are kept)
        return getattr(self, f"_{type(self).__name__}__mother")

    @property
    def n_hits(self) -> ak.Array:
        """Return the total number of hits assigned to each reconstructed PFO.

            Returns:
                ak.Array: Values from ``reco_daughter_PFP_nHits``.
        """
        self.read_data("n_hits", "reco_daughter_PFP_nHits")
        return getattr(self, f"_{type(self).__name__}__n_hits")

    @property
    def n_hits_collection(self) -> ak.Array:
        """Return the collection-plane hit count for each reconstructed PFO.

            Returns:
                ak.Array: Values from ``reco_daughter_PFP_nHits_collection``.
        """
        self.read_data("n_hits_collection", "reco_daughter_PFP_nHits_collection")
        return getattr(self, f"_{type(self).__name__}__n_hits_collection")

    @property
    def track_start_pos(self) -> ak.Record:
        """Return the start-position vector for each reconstructed track.

            Returns:
                ak.Record: Per-track x, y, and z coordinates assembled from
                    the track-start branches.
        """
        ntuples = [
            "reco_daughter_allTrack_startX",
            "reco_daughter_allTrack_startY",
            "reco_daughter_allTrack_startZ"
        ]
        self.read_data("track_start_pos", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__track_start_pos")

    @property
    def shower_start_pos(self) -> ak.Record:
        """Return the start-position vector for each reconstructed shower.

            Returns:
                ak.Record: Per-shower x, y, and z coordinates assembled from
                    the shower-start branches.
        """
        ntuples = [
            "reco_daughter_allShower_startX",
            "reco_daughter_allShower_startY",
            "reco_daughter_allShower_startZ"
        ]
        self.read_data("shower_start_pos", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__shower_start_pos")

    @property
    def track_start_dir(self) -> ak.Record:
        """Return the initial direction vector for each reconstructed track.

            Returns:
                ak.Record: Per-track x, y, and z components assembled from the
                    track-start direction branches.
        """
        ntuples = [
            "reco_daughter_allTrack_startDirX",
            "reco_daughter_allTrack_startDirY",
            "reco_daughter_allTrack_startDirZ"
        ]
        self.read_data("track_start_dir", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__track_start_dir")

    @property
    def shower_direction(self) -> ak.Record:
        """Return the direction vector for each reconstructed shower.

            Returns:
                ak.Record: Per-shower x, y, and z components assembled from
                    the shower direction branches.
        """
        ntuples = [
            "reco_daughter_allShower_dirX",
            "reco_daughter_allShower_dirY",
            "reco_daughter_allShower_dirZ"
        ]
        self.read_data("shower_direction", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__shower_direction")

    @property
    def track_end_pos(self) -> ak.Record:
        """Return the endpoint vector for each reconstructed track.

            Returns:
                ak.Record: Per-track x, y, and z coordinates assembled from
                    the track-end branches.
        """
        ntuples = [
            "reco_daughter_allTrack_endX",
            "reco_daughter_allTrack_endY",
            "reco_daughter_allTrack_endZ"
        ]
        self.read_data("track_end_pos", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__track_end_pos")

    @property
    def shower_energy(self) -> ak.Array:
        """Return the reconstructed energy of each shower.

            Returns:
                ak.Array: Values from ``reco_daughter_allShower_energy``.
        """
        self.read_data("shower_energy", "reco_daughter_allShower_energy")
        return getattr(self, f"_{type(self).__name__}__shower_energy")

    @property
    def shower_momentum(self) -> ak.Record:
        """Calculate shower momentum from reconstructed energy and direction.

            Returns:
                ak.Record: Momentum vectors; entries with a null direction or
                    negative energy are replaced with the null vector.
        """
        if not hasattr(self, f"_{type(self).__name__}__shower_momentum"):
            mom = vector.prod(self.shower_energy, self.shower_direction)
            mom = ak.where(self.shower_direction.x == -999,
                           vector.null_vector, mom)
            mom = ak.where(self.shower_energy < 0, vector.null_vector, mom)
            self.__shower_momentum = mom
        return self.__shower_momentum

    @property
    def shower_length(self) -> ak.Array:
        """Return the reconstructed length of each shower.

            Returns:
                ak.Array: Values from the available shower-length branch.
        """
        self.read_data("shower_length", ["reco_daughter_allShower_length", "reco_daughter_allShower_len"])
        return getattr(self, f"_{type(self).__name__}__shower_length")

    @property
    def shower_cone_angle(self) -> ak.Array:
        """Return the reconstructed cone angle of each shower.

            Returns:
                ak.Array: Values from ``reco_daughter_allShower_coneAngle``.
        """
        self.read_data("shower_cone_angle", "reco_daughter_allShower_coneAngle")
        return getattr(self, f"_{type(self).__name__}__shower_cone_angle") # PDSPAnalyser does not store this value

    @property
    def track_score(self) -> ak.Array:
        """Return the collection-plane track-likeness score for each PFO.

            Returns:
                ak.Array: Values from ``reco_daughter_PFP_trackScore_collection``.
        """
        self.read_data("track_score", "reco_daughter_PFP_trackScore_collection")
        return getattr(self, f"_{type(self).__name__}__track_score")

    @property
    def em_score(self) -> ak.Array:
        """Return the collection-plane electromagnetic-likeness score for each PFO.

            Returns:
                ak.Array: Values from ``reco_daughter_PFP_emScore_collection``.
        """
        self.read_data("em_score", "reco_daughter_PFP_emScore_collection")
        return getattr(self, f"_{type(self).__name__}__em_score")

    @property
    def cnn_score(self) -> ak.Array:
        """Return the CNN classification score stored for each PFO.

            Returns:
                ak.Array: Values from ``CNNScore_collection``.
        """
        self.read_data("cnn_score", "CNNScore_collection")
        return getattr(self, f"_{type(self).__name__}__cnn_score") # specific to the shower merging, but can be recalculated

    @property
    def space_points(self) -> ak.Record:
        """Return the reconstructed space-point coordinates for PFO hits.

            Returns:
                ak.Record: Hit-position coordinates assembled from the
                    reconstructed space-point branches.
        """
        ntuples = [
            "reco_daughter_allShower_spacePointX",
            "reco_daughter_allShower_spacePointY",
            "reco_daughter_allShower_spacePointZ"
        ]
        self.read_data("space_points", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__space_points") # not in PDSPAnalyser

    @property
    def channel(self) -> ak.Array:
        """Return the readout channel associated with each reconstructed hit.

            Returns:
                ak.Array: Channel values from the reconstructed shower hit
                    branch.
        """
        self.read_data("channel", "reco_daughter_allShower_hit_channel") # not in PDSPAnalyser
        return getattr(self, f"_{type(self).__name__}__channel")

    @property
    def peak_time(self) -> ak.Array:
        """Return the peak time of each reconstructed shower hit.

            Returns:
                ak.Array: Values from the reconstructed shower hit peak-time
                    branch.
        """
        self.read_data("peak_time", "reco_daughter_allShower_hit_peakTime") # not in PDSPAnalyser
        return getattr(self, f"_{type(self).__name__}__peak_time")

    @property
    def integral(self) -> ak.Array:
        """Return the charge integral for each reconstructed shower hit.

            Returns:
                ak.Array: Values from the reconstructed shower hit integral
                    branch.
        """
        self.read_data("integral", "reco_daughter_allSHower_hit_integral") # not in PDSPAnalyser
        return getattr(self, f"_{type(self).__name__}__integral")

    @property
    def hit_energy(self) -> ak.Array:
        """Return the reconstructed energy associated with each shower hit.

            Returns:
                ak.Array: Values from the reconstructed shower hit energy
                    branch.
        """
        self.read_data("hit_energy", "reco_daughter_allShower_hit_energy") # not in PDSPAnalyser
        return getattr(self, f"_{type(self).__name__}__hit_energy")

    @property
    def track_dEdX(self) -> ak.Array:
        """Return calibrated space-charge-corrected dE/dx for reconstructed tracks.

            Returns:
                ak.Array: Values from ``reco_daughter_allTrack_calibrated_dEdX_SCE``.
        """
        self.read_data("track_dEdX", "reco_daughter_allTrack_calibrated_dEdX_SCE")
        return getattr(self, f"_{type(self).__name__}__track_dEdX")

    @property
    def track_dQdX(self) -> ak.Array:
        """Return calibrated space-charge-corrected dQ/dx for reconstructed tracks.

            Returns:
                ak.Array: Values from ``reco_daughter_allTrack_calibrated_dQdX_SCE``.
        """
        self.read_data("track_dQdX", "reco_daughter_allTrack_calibrated_dQdX_SCE")
        return getattr(self, f"_{type(self).__name__}__track_dQdX")
    
    @property
    def residual_range(self) -> ak.Array:
        """Return the space-charge-corrected residual range along each track.

            Returns:
                ak.Array: Values from ``reco_daughter_allTrack_resRange_SCE``.
        """
        self.read_data("residual_range", "reco_daughter_allTrack_resRange_SCE")
        return getattr(self, f"_{type(self).__name__}__residual_range")
    
    @property
    def track_chi2_proton(self) -> ak.Array:
        """Return the proton-hypothesis track-fit chi-squared values.

            Returns:
                ak.Array: Values from ``reco_daughter_allTrack_Chi2_proton``.
        """
        self.read_data("track_chi2_proton", "reco_daughter_allTrack_Chi2_proton")
        return getattr(self, f"_{type(self).__name__}__track_chi2_proton")
    
    @property
    def track_chi2_proton_ndof(self) -> ak.Array:
        """Return the degrees of freedom for proton-hypothesis track fits.

            Returns:
                ak.Array: Values from the track-fit degrees-of-freedom branch.
        """
        self.read_data("track_chi2_proton_ndof", "reco_daughter_allTrack_Chi2_ndof")
        return getattr(self, f"_{type(self).__name__}__track_chi2_proton_ndof")

    @property
    def track_chi2_pion(self) -> ak.Array:
        """Return the pion-hypothesis track-fit chi-squared values.

            Returns:
                ak.Array: Values from ``reco_daughter_allTrack_Chi2_pion``.
        """
        self.read_data("track_chi2_pion", "reco_daughter_allTrack_Chi2_pion")
        return getattr(self, f"_{type(self).__name__}__track_chi2_pion")
    
    @property
    def track_chi2_pion_ndof(self) -> ak.Array:
        """Return the degrees of freedom for pion-hypothesis track fits.

            Returns:
                ak.Array: Values from the pion track-fit degrees-of-freedom
                    branch.
        """
        self.read_data("track_chi2_pion_ndof", "reco_daughter_allTrack_Chi2_ndof_pion")
        return getattr(self, f"_{type(self).__name__}__track_chi2_pion_ndof")
    
    @property
    def track_chi2_muon(self) -> ak.Array:
        """Return the muon-hypothesis track-fit chi-squared values.

            Returns:
                ak.Array: Values from ``reco_daughter_allTrack_Chi2_muon``.
        """
        self.read_data("track_chi2_muon", "reco_daughter_allTrack_Chi2_muon")
        return getattr(self, f"_{type(self).__name__}__track_chi2_muon")
    
    @property
    def track_chi2_muon_ndof(self) -> ak.Array:
        """Return the degrees of freedom for muon-hypothesis track fits.

            Returns:
                ak.Array: Values from the muon track-fit degrees-of-freedom
                    branch.
        """
        self.read_data("track_chi2_muon_ndof", "reco_daughter_allTrack_Chi2_ndof_muon")
        return getattr(self, f"_{type(self).__name__}__track_chi2_muon_ndof")

    @property
    def track_len(self) -> ak.Array:
        """Return the reconstructed length of each track.

            Returns:
                ak.Array: Values from ``reco_daughter_allTrack_len``.
        """
        self.read_data("track_len", "reco_daughter_allTrack_len")
        return getattr(self, f"_{type(self).__name__}__track_len")
    
    @property
    def track_len_alt(self) -> ak.Array:
        """Return the alternate reconstructed length of each track.

            Returns:
                ak.Array: Values from ``reco_daughter_allTrack_alt_len``.
        """
        self.read_data("track_len_alt", "reco_daughter_allTrack_alt_len")
        return getattr(self, f"_{type(self).__name__}__track_len_alt")
    
    @property
    def track_vertex_michel(self) -> ak.Array:
        """Return the Michel score at each reconstructed track vertex.

            Returns:
                ak.Array: Values from
                    ``reco_daughter_allTrack_vertex_michel_score``.
        """
        self.read_data("track_vertex_michel", "reco_daughter_allTrack_vertex_michel_score")
        return getattr(self, f"_{type(self).__name__}__track_vertex_michel")
    
    @property
    def track_vertex_nhits(self) -> ak.Array:
        """Return the number of hits at each reconstructed track vertex.

            Returns:
                ak.Array: Values from ``reco_daughter_allTrack_vertex_nHits``.
        """
        self.read_data("track_vertex_nhits", "reco_daughter_allTrack_vertex_nHits")
        return getattr(self, f"_{type(self).__name__}__track_vertex_nhits")

    @property
    def beam_pandora_tag(self) -> ak.Array:
        """Return Pandora's reconstruction tag for the beam PFO.

            Returns:
                ak.Array: Per-event beam tags read from PDSP data or selected
                    from the shower-merging PFO tags; missing tags are filled
                    with ``-999``.
        """
        if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
            tag = self.events.recoParticles.pandora_tag[self.events.recoParticles.beam_number == self.events.recoParticles.number]
            tag = ak.flatten(ak.fill_none(ak.pad_none(tag, 1), -999))
            setattr(self, f"_{type(self).__name__}__beam_pandora_tag", tag)
        if self.events.ntuple_type == NtupleType.PDSP:
            self.read_data("beam_pandora_tag", "reco_beam_type")
        return getattr(self, f"_{type(self).__name__}__beam_pandora_tag")


    @property
    def beam_track_length(self) -> ak.Array:
        """Calculate beam track length from its calorimetry trajectory points.

            Returns:
                ak.Array: Per-event sum of distances between consecutive beam
                    calorimetry positions.
        """
        setattr(self, f"_{type(self).__name__}__beam_track_length", self.track_length(self.beam_calo_pos))
        return getattr(self, f"_{type(self).__name__}__beam_track_length")

    def track_length(self, trajectory_points, threshold = int(2E5)):
        """Sum distances between consecutive points in each trajectory.

            Args:
                trajectory_points (Any): Per-event sequence of ordered
                    three-dimensional trajectory points.
                threshold (Any): Maximum number of events processed per chunk
                    before concatenating chunk results. Defaults to 200000.

            Returns:
                Any: Per-event trajectory lengths computed as the sum of
                    distances between adjacent points.
        """
        if ak.num(trajectory_points, 0) > threshold:
            n_chunks =  ak.num(trajectory_points, 0) // threshold
            trk_len = []
            for i in range(n_chunks + 1):
                chunk = trajectory_points[i*threshold:(i+1)*threshold]
                trk_len.append(ak.sum(vector.dist(chunk[:, 1:], chunk[:, :-1]), axis = -1))
            trk_len = ak.concatenate(trk_len)
        else:
            trk_len = ak.sum(vector.dist(trajectory_points[:, 1:], trajectory_points[:, :-1]), axis = -1)
        return trk_len


    def calculate_pair_quantities(self, use_bt: bool = False) -> tuple:
        """ Calculate reconstructed shower pair quantities.

            Args:
                use_bt (bool): Sort reconstructed showers using backtracked
                    truth energies when true; otherwise use the true-photon
                    energy ordering from the parent event data. Defaults to
                    False.

            Returns:
                tuple: Invariant mass, opening angle, leading shower energy,
                    subleading shower energy, and summed pi0 momentum magnitude,
                    in that order. Invalid pair values are marked ``-999``.
        """
        if use_bt is True:
            sort_energy = ak.argsort(self.events.true_particles_bt.energy)
        else:
            sort_energy = self.events.sorted_true_energy_mask
        sortedPairs = ak.unflatten(self.shower_energy[sort_energy], 1, 0)
        leading = sortedPairs[:, :, 1:]
        secondary = sortedPairs[:, :, :-1]

        # * opening angle
        direction_pair = ak.unflatten(self.shower_direction[sort_energy], 1, 0)
        direction_pair_mag = vector.magnitude(direction_pair)
        angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (
            direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

        # * Invariant Mass
        inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

        # * pi0 momentum
        pi0_momentum = vector.magnitude(ak.sum(self.shower_momentum, axis=-1))

        # mask shower pairs with invalid direction vectors
        null_dir = np.logical_or(
            direction_pair[:, :, 1:].x == -999, direction_pair[:, :, :-1].x == -999)
        # mask of shower pairs with invalid energy
        null = np.logical_or(leading < 0, secondary < 0)

        # * filter null data
        pi0_momentum = np.where(null_dir, -999, pi0_momentum)
        pi0_momentum = np.where(null, -999, pi0_momentum)

        leading = np.where(null, -999, leading)
        leading = np.where(null_dir, -999, leading)
        secondary = np.where(null, -999, secondary)
        secondary = np.where(null_dir, -999, secondary)

        angle = np.where(null, -999, angle)
        angle = np.where(null_dir, -999, angle)

        inv_mass = np.where(null, -999, inv_mass)
        inv_mass = np.where(null_dir, -999, inv_mass)

        return inv_mass, angle, leading, secondary, pi0_momentum

    @timer
    def get_pair_values(pairs, value) -> ak.Array:
        """ Get shower pair values, in pairs.

            Args:
                pairs (list): shower pairs per event
                value (ak.Array): values to retrieve

            Returns:
                ak.Array: paired showers values per event
        """
        paired = []
        for i in range(len(pairs)):
            pair = pairs[i]
            evt = []
            for j in range(len(pair)):
                if len(pair[j]) > 0:
                    evt.append([value[i][pair[j][0]], value[i][pair[j][1]]])
                else:
                    evt.append([])
            paired.append(evt)
        return ak.Array(paired)

    @timer
    def all_shower_pairs(nd) -> list:
        """ Get all unique shower pair combinations.

            Args:
                nd (ak.Array): number of daughters in an event

            Returns:
                list: Jagged array of pairs per event
        """
        pairs = []
        for i in range(len(nd)):
            comb = itertools.combinations(range(nd[i]), 2)
            pairs.append(list(comb))
        return pairs

    @timer
    def shower_pairs_by_hits(hits: ak.Array) -> list:
        """Pair reconstructed showers by descending collection-plane hit count.

            Args:
                hits (ak.Array): Per-event collection-plane hit counts for
                    reconstructed daughter PFPs.

            Returns:
                list: Per-event lists containing at most one pair: the indices
                    of the two highest-hit showers, or an empty list when an
                    event has fewer than two showers.
        """
        showers = ak.argsort(
            hits, ascending=False)  # shower number sorted by n_hits
        mask = ak.count(showers, 1) > 1
        # only keep two largest showers
        showers = ak.pad_none(showers, 2, clip=True)
        showers = ak.where(mask, showers, [[]]*len(mask))
        pairs = ak.unflatten(showers, 1)
        return ak.to_list(pairs)


class TrueParticleDataBT(ParticleData):
    """ Particle data class for backtracked truth information.

        Args:
            events (Data): Parent event container whose reconstructed-particle
                truth-by-hits branches are exposed by this object.

        Attributes:
            (1 hidden attribute for each property method)
            Property Methods:
            number (ak.Array): backtracked particle number
            mother (ak.Array): backtracked particle number of mother
            pdg (ak.Array): pdg code of backtracked particle
            mother_pdg (ak.Array): pdg code of mother of backtracked particle
            shower_start_pos (ak.Record):
            end_pos (ak.Record):
            momentum (ak.Record):
            energy (ak.Array):
            energy_by_hits (ak.Array): energy of the true particle, calculated from the true hits
            direction (ak.Record):
            hit_in_reco_cluster (ak.Array): total hits in a reco cluster
            n_hits (ak.Array): true particle hits
            sharedHits (ak.Array): hits shared by both reco and backtracked true particle
            true_beam_vertex (ak.Record): end poisition of backtracked true beam particle
            purity (ak.Array): fraction of shared hits in reconstructed object
            completeness (ak.Array): fraction of shared hits in the backtrackted true particle
            hit_in_reco_cluster_collection (ak.Array): total hits in a reco cluster
            n_hits_collection (ak.Array): true particle hits
            shared_hits_collection (ak.Array): hits shared by both reco and backtracked true particle
            true_beam_vertex_collection (ak.Record): end poisition of backtracked true beam particle
            purity_collection (ak.Array): fraction of shared hits in reconstructed object
            completeness_collection (ak.Array): fraction of shared hits in the backtrackted true particle
            space_points (ak.Record): hit space points
            channel (ak.Array): hit channel
            peak_time (ak.Array): hit peak time
            integral (ak.Array): hit integral
            hit_energy (ak.Array): hit energy

            partice_number (ak.Array): placeholder for ROOT files which don't have the particle number stored as NTuples
            single_match (ak.Array): mask of events with only 1 unique backtracked particle

            Methods:
            GetUniqueParticleNumbers (ak.Array): Get unique array of particle numbers
            calculate_pair_quantities (tuple): Calculate backtracked shower pair quantities.
    """

    def __init__(self, events: Data):
        """Initialize the backtracked-truth view of the parent event sample.

            Args:
                events (Data): Parent event container whose backtracked truth
                    branches are exposed by this object.
        """
        super().__init__(events)

    @property
    def number(self) -> ak.Array:
        """Return the true-particle identifier assigned to each reconstructed PFO.

            Returns:
                ak.Array: Values from ``reco_daughter_PFP_true_byHits_ID``.
        """
        self.read_data("number", "reco_daughter_PFP_true_byHits_ID")
        return getattr(self, f"_{type(self).__name__}__number")

    @property
    def mother(self) -> ak.Array:
        """Return the identifier of each backtracked particle's mother.

            Returns:
                ak.Array: Mother identifiers from the available
                    true-by-hits mother or parent-ID branch.
        """
        self.read_data("mother", ["reco_daughter_PFP_true_byHits_Mother", "reco_daughter_PFP_true_byHits_parID"])
        return getattr(self, f"_{type(self).__name__}__mother")

    @property
    def pdg(self) -> ak.Array:
        """Return the PDG code of each particle matched by backtracking.

            Returns:
                ak.Array: Values from the available true-by-hits PDG branch.
        """
        self.read_data("pdg", ["reco_daughter_PFP_true_byHits_pdg", "reco_daughter_PFP_true_byHits_PDG"])
        return getattr(self, f"_{type(self).__name__}__pdg")

    @property
    def mother_pdg(self) -> ak.Array:
        """Return the PDG code of each backtracked particle's mother.

            Returns:
                ak.Array: Values from the available true-by-hits mother-PDG or
                    parent-PDG branch.
        """
        self.read_data("mother_pdg", ["reco_daughter_PFP_true_byHits_Mother_pdg", "reco_daughter_PFP_true_byHits_parPDG"])
        return getattr(self, f"_{type(self).__name__}__mother_pdg")

    @property
    def shower_start_pos(self) -> ak.Record:
        """Return the starting position of the truth particle matched to each PFO.

            Returns:
                ak.Record: x, y, and z coordinates assembled from the
                    true-by-hits start-position branches.
        """
        ntuples = [
            "reco_daughter_PFP_true_byHits_startX",
            "reco_daughter_PFP_true_byHits_startY",
            "reco_daughter_PFP_true_byHits_startZ"
        ]
        self.read_data("shower_start_pos", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__shower_start_pos")

    @property
    def end_pos(self) -> ak.Record:
        """Return the endpoint of the truth particle matched to each PFO.

            Returns:
                ak.Record: x, y, and z coordinates assembled from the
                    true-by-hits end-position branches.
        """
        ntuples = [
            "reco_daughter_PFP_true_byHits_endX",
            "reco_daughter_PFP_true_byHits_endY",
            "reco_daughter_PFP_true_byHits_endZ"
        ]
        self.read_data("end_pos", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__end_pos")

    @property
    def momentum(self) -> ak.Record:
        """Return the momentum vector of the truth particle matched to each PFO.

            Returns:
                ak.Record: x, y, and z components loaded from the available
                    true-by-hits momentum branches.
        """
        ntuples = [
            ["reco_daughter_PFP_true_byHits_pX", "reco_daughter_PFP_true_byHits_startPx"],
            ["reco_daughter_PFP_true_byHits_pY", "reco_daughter_PFP_true_byHits_startPy"],
            ["reco_daughter_PFP_true_byHits_pZ", "reco_daughter_PFP_true_byHits_startPz"]
        ]
        self.read_data("momentum", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__momentum")

    @property
    def direction(self) -> ak.Record:
        """Return unit direction vectors computed from backtracked momentum.

            Returns:
                ak.Record: Normalized momentum vectors with x, y, and z
                    components.
        """
        if not hasattr(self, f"_{type(self).__name__}__direction"):
            self.__direction = vector.normalize(self.momentum)
        return self.__direction

    @property
    def energy(self) -> ak.Array:
        """Return the starting energy of each backtracked truth particle.

            Returns:
                ak.Array: Values from the true-by-hits start-energy branch,
                    multiplied by 1000 for PDSP ntuples.
        """
        self.read_data("energy", "reco_daughter_PFP_true_byHits_startE")
        factor = 1000 if self.events.ntuple_type == NtupleType.PDSP else 1
        return factor * getattr(self, f"_{type(self).__name__}__energy")

    @property
    def energy_by_hits_uncorrected(self) -> ak.Array:
        """Return the uncorrected energy-by-hits estimate for each particle.

            Returns:
                ak.Array: Values from
                    ``reco_daughter_PFP_true_byHits_EnergyByHits``; used by
                    shower-merging ntuples.
        """
        self.read_data("energy_by_hits_uncorrected", "reco_daughter_PFP_true_byHits_EnergyByHits")
        return getattr(self, f"_{type(self).__name__}__energy_by_hits_uncorrected") # for shower merging only

    @property
    def energy_by_hits_correction(self) -> ak.Array:
        """Return the correction applied to each energy-by-hits estimate.

            Returns:
                ak.Array: Values from the true-by-hits energy-correction
                    branch; used by shower-merging ntuples.
        """
        self.read_data("energy_by_hits_correction", "reco_daughter_PFP_true_byHits_EnergyByHits_correction")
        return getattr(self, f"_{type(self).__name__}__energy_by_hits_correction") # for shower merging only

    @property
    def energy_by_hits(self) -> ak.Array:
        """Return the energy-by-hits estimate after subtracting its correction.

            Returns:
                ak.Array: Uncorrected energy-by-hits values minus the
                    corresponding correction, or ``None`` if either input is
                    unavailable.
        """
        if type(self.energy_by_hits_uncorrected) == ak.highlevel.Array and type(self.energy_by_hits_correction) == ak.highlevel.Array:
            v = self.energy_by_hits_uncorrected - self.energy_by_hits_correction
        else:
            v = None
        setattr(self, f"_{type(self).__name__}__hit_in_reco_cluster__energy_by_hits", v)
        return getattr(self, f"_{type(self).__name__}__hit_in_reco_cluster__energy_by_hits")

    @property
    def momentum_by_hits(self) -> ak.Record:
        """Calculate a momentum vector using energy-by-hits and the truth direction.

            Returns:
                ak.Record: Momentum vectors formed from energy-by-hits and
                    normalized truth momentum; invalid directions or negative
                    starting energies produce null vectors.
        """
        if not hasattr(self, f"_{type(self).__name__}__momentum_by_hits"):
            if type(self.energy_by_hits) == ak.highlevel.Array and type(self.direction) == ak.highlevel.Array:
                mom = vector.prod(self.energy_by_hits, self.direction)
                mom = ak.where(self.direction.x == -999, vector.null_vector, mom)
                mom = ak.where(self.energy < 0, vector.null_vector, mom)
            else:
                mom = None
            setattr(self, f"_{type(self).__name__}__momentum_by_hits", mom)
        return getattr(self, f"_{type(self).__name__}__momentum_by_hits") # for shower merging only

    @property
    def hit_in_reco_cluster(self) -> ak.Array:
        """Return the number of true-particle hits found in each reconstructed cluster.

            Returns:
                ak.Array: Values from the true-by-hits hits-in-reconstructed-
                    cluster branch.
        """
        self.read_data("hit_in_reco_cluster", "reco_daughter_PFP_true_byHits_hitsInRecoCluster")
        return getattr(self, f"_{type(self).__name__}__hit_in_reco_cluster") # PDSPAnalyser does not store this

    @property
    def n_hits(self) -> ak.Array:
        """Return the total true-particle hit count matched to each PFO.

            Returns:
                ak.Array: Values from the true-by-hits particle hit-count
                    branch.
        """
        self.read_data("n_hits", "reco_daughter_PFP_true_byHits_nHits")
        return getattr(self, f"_{type(self).__name__}__n_hits") # PDSPAnalyser does not store this

    @property
    def sharedHits(self) -> ak.Array:
        """Return the hit count shared by each PFO and its matched truth particle.

            Returns:
                ak.Array: Values from the true-by-hits shared-hit branch.
        """
        self.read_data("sharedHits", "reco_daughter_PFP_true_byHits_sharedHits")
        return getattr(self, f"_{type(self).__name__}__sharedHits")

    @property
    def hit_in_reco_cluster_collection(self) -> ak.Array:
        """Return collection-plane true-particle hits in each reconstructed cluster.

            Returns:
                ak.Array: Values from the collection-plane hits-in-cluster
                    branch.
        """
        self.read_data("hit_in_reco_cluster_collection", "reco_daughter_PFP_true_byHits_hitsInRecoCluster_collection")
        return getattr(self, f"_{type(self).__name__}__hit_in_reco_cluster_collection") # for shower merging only

    @property
    def n_hits_collection(self) -> ak.Array:
        """Return the collection-plane hit count for each backtracked particle.

            Returns:
                ak.Array: Values from the true-by-hits collection hit-count
                    branch.
        """
        self.read_data("n_hits_collection", "reco_daughter_PFP_true_byHits_nHits_collection")
        return getattr(self, f"_{type(self).__name__}__n_hits_collection") # for shower merging only

    @property
    def shared_hits_collection(self) -> ak.Array:
        """Return collection-plane hits shared by each PFO and its truth particle.

            Returns:
                ak.Array: Values from the collection-plane shared-hit branch.
        """
        self.read_data("shared_hits_collection", "reco_daughter_PFP_true_byHits_sharedHits_collection")
        return getattr(self, f"_{type(self).__name__}__shared_hits_collection") # for shower merging only

    @property
    def true_beam_vertex(self) -> ak.Record:
        """Return the backtracked true beam endpoint coordinates.

            Returns:
                ak.Record: x, y, and z coordinates assembled from the
                    true-by-hits beam endpoint branches.
        """
        ntuples = [
            "reco_beam_PFP_true_byHits_endX",
            "reco_beam_PFP_true_byHits_endY",
            "reco_beam_PFP_true_byHits_endZ"
        ]
        self.read_data("true_beam_vertex", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__true_beam_vertex")

    @property
    def purity(self) -> ak.Array:
        """Return the fraction of reconstructed-cluster hits shared with truth.

            Returns:
                ak.Array: Shared hits divided by hits in the reconstructed
                    cluster for shower-merging data, or the stored PDSP purity.
        """
        if not hasattr(self, f"_{type(self).__name__}__purity"):
            if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
                if type(self.sharedHits) == ak.highlevel.Array and type(self.hit_in_reco_cluster) == ak.highlevel.Array:
                    v = self.sharedHits / self.hit_in_reco_cluster
                else:
                    v = None
                setattr(self, f"_{type(self).__name__}__purity", v)
            if self.events.ntuple_type == NtupleType.PDSP:
                self.read_data("purity", "reco_daughter_PFP_true_byHits_purity")
        return getattr(self, f"_{type(self).__name__}__purity")

    @property
    def completeness(self) -> ak.Array:
        """Return the fraction of backtracked truth hits present in the PFO.

            Returns:
                ak.Array: Shared hits divided by total true-particle hits for
                    shower-merging data, or the stored PDSP completeness.
        """
        if not hasattr(self, f"_{type(self).__name__}__completeness"):
            if self.events.ntuple_type == NtupleType.SHOWER_MERGING:
                if type(self.sharedHits)  == ak.highlevel.Array and type(self.n_hits) == ak.highlevel.Array:
                    v = self.sharedHits / self.n_hits
                else:
                    v = None
                setattr(self, f"_{type(self).__name__}__completeness", v)
            if self.events.ntuple_type == NtupleType.PDSP:
                self.read_data("completeness", "reco_daughter_PFP_true_byHits_completeness")
        return getattr(self, f"_{type(self).__name__}__completeness")

    @property
    def purity_collection(self) -> ak.Array:
        """Return collection-plane hit purity for each reconstructed PFO.

            Returns:
                ak.Array: Collection-plane shared hits divided by hits in the
                    reconstructed cluster, or ``None`` when the inputs are
                    unavailable.
        """
        if not hasattr(self, f"_{type(self).__name__}__purity_collection"):
            if type(self.shared_hits_collection)  == ak.highlevel.Array and type(self.hit_in_reco_cluster_collection)  == ak.highlevel.Array:
                v = self.shared_hits_collection / self.hit_in_reco_cluster_collection
            else:
                v = None
            setattr(self, f"_{type(self).__name__}__purity_collection", v)
        return getattr(self, f"_{type(self).__name__}__purity_collection") # for shower merging only

    @property
    def completeness_collection(self) -> ak.Array:
        """Return collection-plane hit completeness for each reconstructed PFO.

            Returns:
                ak.Array: Collection-plane shared hits divided by total
                    collection-plane truth hits, or ``None`` when inputs are
                    unavailable.
        """
        if not hasattr(self, f"_{type(self).__name__}__completeness_collection"):
            if type(self.shared_hits_collection) == ak.highlevel.Array and type(self.n_hits_collection) == ak.highlevel.Array:
                v = self.shared_hits_collection / self.n_hits_collection
            else:
                v = None
            setattr(self, f"_{type(self).__name__}__completeness_collection", v)
        return getattr(self, f"_{type(self).__name__}__completeness_collection") # for shower merging only

    @property
    def space_points(self) -> type:
        """Return hit space-point coordinates for the backtracked particle.

            Returns:
                type: x, y, and z coordinates assembled from true-by-hits
                    space-point branches.
        """
        ntuples = [
            "reco_daughter_PFP_true_byHits_spacePointX",
            "reco_daughter_PFP_true_byHits_spacePointY",
            "reco_daughter_PFP_true_byHits_spacePointZ"
        ]
        self.read_data("space_points", ntuples, is_vector = True)
        return getattr(self, f"_{type(self).__name__}__space_points") # for shower merging only

    @property
    def channel(self) -> ak.Array:
        """Return the readout channel for each backtracked hit.

            Returns:
                ak.Array: Values from the true-by-hits hit-channel branch.
        """
        self.read_data("channel", "reco_daughter_PFP_true_byHits_hit_channel")
        return getattr(self, f"_{type(self).__name__}__channel") # for shower merging only

    @property
    def peak_time(self) -> ak.Array:
        """Return the peak time for each backtracked hit.

            Returns:
                ak.Array: Values from the true-by-hits hit peak-time branch.
        """
        self.read_data("peak_time", "reco_daughter_PFP_true_byHits_hit_peakTime")
        return getattr(self, f"_{type(self).__name__}__peak_time") # for shower merging only

    @property
    def integral(self) -> ak.Array:
        """Return the charge integral for each backtracked hit.

            Returns:
                ak.Array: Values from the true-by-hits hit-integral branch.
        """
        self.read_data("integral", "reco_daughter_PFP_true_byHits_hit_integral")
        return getattr(self, f"_{type(self).__name__}__integral") # for shower merging only

    @property
    def hit_energy(self) -> ak.Array:
        """Return the energy recorded for each backtracked hit.

            Returns:
                ak.Array: Values from the true-by-hits hit-energy branch.
        """
        self.read_data("hit_energy", "reco_daughter_PFP_true_byHits_hit_energy")
        return getattr(self, f"_{type(self).__name__}__hit_energy") # for shower merging only

    @property
    def beam_pdg(self) -> ak.Array:
        """Return the backtracked PDG code of the reconstructed beam particle.

            Returns:
                ak.Array: Values from the available beam true-by-hits PDG
                    branch.
        """
        self.read_data("beam_pdg", ["reco_beam_PFP_true_byHits_pdg", "reco_beam_true_byHits_PDG"])
        return getattr(self, f"_{type(self).__name__}__beam_pdg")

    @property
    def beam_origin(self) -> ak.Array:
        """Return the origin code of the backtracked beam particle.

            Returns:
                ak.Array: Values from ``reco_beam_true_byHits_origin``.
        """
        self.read_data("beam_origin", "reco_beam_true_byHits_origin")
        return getattr(self, f"_{type(self).__name__}__beam_origin")

    @property
    def beam_end_process(self) -> ak.Array:
        """Return the end-process code of the backtracked beam particle.

            Returns:
                ak.Array: Values from ``reco_beam_true_byHits_endProcess``.
        """
        self.read_data("beam_end_process", "reco_beam_true_byHits_endProcess")
        return getattr(self, f"_{type(self).__name__}__beam_end_process")

    @property
    def is_beam_pi0(self) -> ak.Array:
        """Identify backtracked particles that are daughters of a beam pi-zero.

            Returns:
                ak.Array: Boolean mask based on the backtracked mother PDG code
                    and the identifiers of primary pi-zero particles in the
                    event truth record.
        """
        if not hasattr(self, f"_{type(self).__name__}__is_beam_pi0"):
            if ak.count(self.mother_pdg) != 0:
                pi0_daughter = self.mother_pdg == 111

                beam_pi0 = self.events.true_particles.number[(self.events.true_particles.pdg == 111) & (self.events.true_particles.mother == 1)]
                beam_pi0 = ak.fill_none(ak.pad_none(beam_pi0, 1, axis = -1), False)

                beam_pi0_bt = []
                for pi0, m in zip(beam_pi0, self.mother):
                    beam_pi0_bt.append(np.isin(m, pi0))
                beam_pi0_bt = ak.Array(beam_pi0_bt) & pi0_daughter
            else:
                beam_pi0_bt = self.mother_pdg

            setattr(self, f"_{type(self).__name__}__is_beam_pi0", beam_pi0_bt)
        return getattr(self, f"_{type(self).__name__}__is_beam_pi0")

    @property
    def partice_number(self) -> ak.Array:
        """ Gets the true particle number of each true particle backtracked to reco

            Returns:
                ak.Array: awkward array of true particle indices
        """
        photonEnergies = self.events.true_particles.energy[
            self.events.true_particles.true_photon_mask]  # assign true particle number by comparing energies
        photonNum = self.events.true_particles.number[self.events.true_particles.true_photon_mask]

        # pad these arrays since occasionaly we can get more than two particles from the photon mask
        photonEnergies = ak.pad_none(photonEnergies, 2)
        photonNum = ak.pad_none(photonNum, 2)

        # loop through all true photons and check the energy matches
        index = None
        for i in range(2):
            # mask which checked the backtracked MC is the same as the ith photon slice
            matched = self.energy == photonEnergies[:, i]
            if i == 1:
                # where -1, see below, else leave index unchanged
                index = ak.where(
                    index == -1, ak.where(matched, photonNum[:, i], -1), index)
            else:
                # where true add slice, elseset indices to -1
                index = ak.where(matched, photonNum[:, i], -1)
        return index

    @property
    def single_match(self) -> ak.Array:
        """ Get a boolean mask of events with more than one tagged true particle
            in the back tracker.

            Returns:
                ak.Array: boolean mask
        """
        unique_index = self.get_unique_particle_numbers(self.partice_number)
        match = ak.num(unique_index) < 2
        return np.logical_not(match)

    def get_unique_particle_numbers(self, index: ak.Array) -> ak.Array:
        """ Gets the unique true particle numbers backtracked to each reco particle.
            e.g. if indices are [1, 2, 1, 3, 3, 3], unique is [1, 2, 3]

            Args:
                index (ak.Array): particle numbers of the true particles which were backtracked.

            Returns:
                ak.Array: unique particle numbers
        """
        max_n_PFP = ak.max(
            ak.num(index))  # get the larget number of PFParticles in an event
        # pad index array so all nested arrays are equal (can convert to numpy array)
        index_padded = ak.pad_none(index, max_n_PFP)
        # fill None entries in nested arrays
        index_padded = ak.fill_none(index_padded, -1)
        # fill events with no counts with [-1]*nPFP
        index_padded = ak.fill_none(index_padded, [-1]*max_n_PFP, 0)

        index_padded_np = ak.to_numpy(index_padded)
        uniqueIndex = [np.unique(index_padded_np[i, :]) for i in range(
            len(index_padded_np))]  # get the unique entries per event
        uniqueIndex = ak.Array(uniqueIndex)

        # remove -1 entries as these just represent padded entries
        return uniqueIndex[uniqueIndex != -1]

    def calculate_pair_quantities(self) -> tuple:
        """ Calculate reconstructed shower pair quantities.

            Returns:
                tuple: calculated quantities + array which masks null shower pairs
        """
        sort_energy = ak.argsort(self.energy)
        sorted_pairs = ak.unflatten(self.energy[sort_energy], 1, 0)
        leading = sorted_pairs[:, :, 1:]
        secondary = sorted_pairs[:, :, :-1]

        # * opening angle
        direction_pair = ak.unflatten(self.direction[sort_energy], 1, 0)
        direction_pair_mag = vector.magnitude(direction_pair)
        angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (
            direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

        # * Invariant Mass
        inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

        # * pi0 momentum
        pi0_momentum = vector.magnitude(ak.sum(self.momentum, axis=-1))

        # mask shower pairs with invalid direction vectors
        null_dir = np.logical_or(
            direction_pair[:, :, 1:].x == -999, direction_pair[:, :, :-1].x == -999)
        # mask of shower pairs with invalid energy
        null = np.logical_or(leading < 0, secondary < 0)

        # * filter null data
        pi0_momentum = np.where(null_dir, -999, pi0_momentum)
        pi0_momentum = np.where(null, -999, pi0_momentum)

        leading = leading
        secondary = secondary

        leading = np.where(null, -999, leading)
        leading = np.where(null_dir, -999, leading)
        secondary = np.where(null, -999, secondary)
        secondary = np.where(null_dir, -999, secondary)

        angle = np.where(null, -999, angle)
        angle = np.where(null_dir, -999, angle)

        inv_mass = inv_mass
        inv_mass = np.where(null, -999, inv_mass)
        inv_mass = np.where(null_dir, -999, inv_mass)

        return inv_mass, angle, leading, secondary, pi0_momentum


class ShowerPairs:
    """Pair reconstructed PFPs and expose paired reconstructed and truth quantities.

        Args:
            events (Data): Event sample providing reconstructed and backtracked
                particle data.
            pair_coords (Any): Precomputed pairs of PFP indices. When omitted,
                all unique reconstructed-PFP pairs are generated.
            shower_pair_mask (ak.Array): Per-event boolean masks from which
                pairs of selected PFP indices are generated. Takes precedence
                over ``pair_coords`` when provided.

        Attributes:
            events (Data): Parent event sample.
            pairs (ak.Array): Per-event indices for each pair of PFPs.
            leading (ak.Array): Index of the higher-energy PFP in each pair,
                ordered using backtracked truth energy when available and
                reconstructed energy otherwise.
            subleading (ak.Array): Index of the lower-energy PFP in each pair,
                using the same energy source as ``leading``.
            Static Methods:
            OpeningAngle
            Mass
            Pi0Mom
            FractionalError
            Property Methods:
            reco_lead_energy (ak.Array) : reco leading pair energy
            reco_sub_energy (ak.Array) : reco subleading pair energy
            reco_angle (ak.Array) : reco opening angle
            reco_mass (ak.Array) : reco invariant mass
            reco_pi0_mom (ak.Array) : reco sum momentum magnitude
            true_lead_energy (ak.Array) : true leading pair energy
            true_sub_energy (ak.Array) : true subleading pair energy
            true_angle (ak.Array) : true opening angle
            true_mass (ak.Array) : true invariant mass
            true_pi0_mom (ak.Array) : true sum momentum magnitude
            error_lead_energy (ak.Array) : error leading pair energy
            error_sub_energy (ak.Array) : error subleading pair energy
            error_angle (ak.Array) : error opening angle
            error_mass (ak.Array) : errore invariant mass
            error_pi0_mom (ak.Array) : error sum momentum magnitude
            Methods:
            SortByBacktrackedEnergy : create leading and sub leading masks
            calculate_all (pd.DataFrame) : calculates all quantities
            save_to_csv : saves quantities to a csv
    """

    def __init__(self, events: Data, pair_coords = None,
                 shower_pair_mask: ak.Array = None):
        """Build PFP pairs and order their members by the available energy values.

            Args:
                events (Data): Event sample providing reconstructed and
                    backtracked particle data.
                pair_coords (Any): Precomputed pairs of PFP indices. When
                    omitted, all unique reconstructed-PFP pairs are generated.
                shower_pair_mask (ak.Array): Per-event boolean masks from
                    which pairs of selected PFP indices are generated. Takes
                    precedence over ``pair_coords`` when provided.
        """
        self.events = events
        if shower_pair_mask is not None:
            self.pairs = self.get_pairs_from_mask(shower_pair_mask)
        else:
            if pair_coords is None:
                pair_coords = ak.argcombinations(
                    self.events.recoParticles.number, 2)
            self.pairs = pair_coords
        if ak.count(self.events.true_particles_bt.energy) > 0:
            self.sort_by_energy()
        else:
            print("no truth information found, sorting showers by reco energy instead")
            self.sort_by_energy(backtracked = False)
        return

    @staticmethod
    def get_pairs_from_mask(mask):
        """Create all two-index combinations selected by a per-event mask.

            Args:
                mask (Any): Per-event boolean masks selecting the PFPs to pair.

            Returns:
                Any: Awkward array of all unique index pairs for each event.
        """
        inverse_mask = ak.Array(map(
            lambda e: np.squeeze(np.argwhere(e), axis=1),
            mask))
        return ak.combinations(inverse_mask, 2)

    def sort_by_energy(self, backtracked : bool = True):
        """ Orders the argcombinations by the shower with higher energy.

            Args:
                backtracked (bool): sort by backtreacked energy, otherwise reco energy. Defaults to True.
        """
        if backtracked:
            pd_e = self.events.true_particles_bt.energy
        else:
            pd_e = self.events.recoParticles.shower_energy

        mask_0_leading = (
            pd_e[self.pairs['0']]
            >= pd_e[self.pairs['1']])
        mask_1_leading = np.logical_not(mask_0_leading)
        self.leading = (self.pairs['0'] * mask_0_leading
                        + self.pairs['1'] * mask_1_leading)
        self.subleading = (self.pairs['0'] * mask_1_leading
                           + self.pairs['1'] * mask_0_leading)

    @staticmethod
    def opening_angle(dir0: ak.Array, dir1: ak.Array) -> ak.Array:
        """ Angle between shower pair directions.

            Args:
                dir0 (ak.Array): Direction vectors of the first shower in
                    each pair.
                dir1 (ak.Array): Direction vectors of the second shower in
                    each pair.

            Returns:
                ak.Array: angles (rad).
        """
        mag0 = vector.magnitude(dir0)
        mag1 = vector.magnitude(dir1)
        return np.arccos(vector.dot(dir0, dir1) / (mag0 * mag1))

    @staticmethod
    def mass(energy0: ak.Array, energy1: ak.Array, angle: ak.Array) -> ak.Array:
        """ Invariant mass of shower pairs.

            Args:
                energy0 (ak.Array): Energy of the first shower in each pair.
                energy1 (ak.Array): Energy of the second shower in each pair.
                angle (ak.Array): Opening angle in radians between each
                    shower pair.

            Returns:
                ak.Array: mass (MeV)
        """
        return (2 * energy0 * energy1 * (1 - np.cos(angle)))**0.5

    @staticmethod
    def pi0_mom(p_0: ak.Array, p_1: ak.Array) -> ak.Array:
        """ Summed momentum magnitude of shower pairs.

            Args:
                p_0 (ak.Array): Momentum vector of the first shower.
                p_1 (ak.Array): Momentum vector of the second shower.

            Returns:
                ak.Array: sum momentum (MeV)
        """
        return vector.add(p_0, p_1)

    @staticmethod
    def energy(e_0: ak.Array, e_1: ak.Array):
        """ Finds the summed energies of the shower pairs.

            Args:
                e_0 (ak.Array): Energy of the first shower in each pair.
                e_1 (ak.Array): Energy of the second shower in each pair.

            Returns:
                energy: ak.Array Summed energies of the pairs.
        """
        return e_0 + e_1

    @staticmethod
    def fractional_error(reco: ak.Array, true: ak.Array) -> ak.Array:
        """ percentage difference between reco and truth.

            Args:
                reco (ak.Array): reconstructed value
                true (ak.Array): true value

            Returns:
                ak.Array: fractional error (%)
        """
        def f(r, t):
            """Compute the component-wise fractional difference from truth.

                Args:
                    r (Any): Reconstructed component or scalar value.
                    t (Any): Corresponding truth component or scalar value.

                Returns:
                    Any: ``(r / t) - 1`` for the supplied component or value.
            """
            return (r / t) -1
        if hasattr(reco, "x") and hasattr(true, "x"):
            # vector like quantity
            return vector.vector(f(reco.x, true.x), f(reco.y, true.y), f(reco.z, true.z))
        else:
            # scalar quantity
            return f(reco, true)

    @staticmethod
    def mid_points(x1, x2):
        """ Returns the midpoint of the starting positions:
            ``mp = (x1 + x2)/2``

            Args:
                x1 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Position of first point.
                x2 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Position of second point.

            Returns:
                mp: ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Midpoint of `x1` and `x2`.
        """
        return vector.prod(0.5, vector.add(x1, x2))

    @staticmethod
    def shared_vertex(mom1, mom2, start1, start2):
        """ Estimates a shared vertex for two vectors and starting positions
            by taking the midpoint of the line of closest approach.

            This is a projected point, not the difference between positions.

            Momenta are used instead of directions to allow for potential
            updates which weight the position of the vertex along the line
            of closest approach based on the relative momenta.

            Args:
                mom1 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Momentum of first PFO.
                mom2 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Momentum of second PFO.
                start1 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Initial point of first PFO.
                start2 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Initial point of second PFO.

            Returns:
                vertex: ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Shared vertex between the PFOs.
        """
        # We estimate the shared vertex by taking the midpoint of the line of closest approch
        # This is a projected point, NOT the difference between the reconstructed starts
        joining_dir = vector.normalize(vector.cross(mom1, mom2))
        separation = vector.dot(vector.sub(start1, start2), joining_dir)
        dir1_selector = vector.cross(joining_dir, mom2)
        # We don't use the normalised momentum, because we later multiply by the momentum, so it cancels
        start1_offset = vector.dot(vector.sub(
            start2, start1), dir1_selector) / vector.dot(mom1, dir1_selector)
        pos1 = vector.add(start1, vector.prod(start1_offset, mom1))
        return vector.add(pos1, vector.prod(separation/2, joining_dir))

    @staticmethod
    def closest_approach(dir1, dir2, start1, start2):
        """ Finds the closest approach between two showers.

            Args:
                dir1 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Direction of first PFO.
                dir2 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Direction of second PFO.
                start1 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Initial point of first PFO.
                start2 (Any): ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array}) Initial point of second PFO.

            Returns:
                d: ak.Array Magnitude of the sortest direction between the two PFOs.
        """
        # x_1 - x_2 + lambda_1 v_1 - lambda_2 v_2 = d/sin(theta) v_1 x v_2
        cross = vector.normalize(vector.cross(dir1, dir2))
        rel_start = vector.sub(start1, start2)
        # Separation between the lines
        d = vector.dot(rel_start, cross)
        return d

    ##############################################################################################
    @property
    def reco_lead_energy(self):
        """Return the reconstructed energy of the leading shower in each pair.

            Returns:
                Any: Shower-energy values selected using the ``leading``
                    indices.
        """
        return self.events.recoParticles.shower_energy[self.leading]

    @property
    def reco_sub_energy(self):
        """Return the reconstructed energy of the subleading shower in each pair.

            Returns:
                Any: Shower-energy values selected using the ``subleading``
                    indices.
        """
        return self.events.recoParticles.shower_energy[self.subleading]

    @property
    def reco_lead_nHits(self):
        """Return the total reconstructed hit count for each leading PFO.

            Returns:
                Any: ``n_hits`` values selected by the ``leading`` indices.
        """
        return self.events.recoParticles.n_hits[self.leading]

    @property
    def reco_sub_nHits(self):
        """Return the total reconstructed hit count for each subleading PFO.

            Returns:
                Any: ``n_hits`` values selected by the ``subleading`` indices.
        """
        return self.events.recoParticles.n_hits[self.subleading]

    @property
    def reco_lead_nHits_collection(self):
        """Return the ``n_hits`` values for leading PFPs in each pair.

            Returns:
                Any: Total-hit counts selected by the ``leading`` indices.
                    Despite this property's name, it indexes ``n_hits`` rather
                    than ``n_hits_collection``.
        """
        return self.events.recoParticles.n_hits[self.leading]

    @property
    def reco_sub_nHits_collection(self):
        """Return the ``n_hits`` values for subleading PFPs in each pair.

            Returns:
                Any: Total-hit counts selected by the ``subleading`` indices.
                    Despite this property's name, it indexes ``n_hits``
                    rather than ``n_hits_collection``.
        """
        return self.events.recoParticles.n_hits[self.subleading]

    @property
    def reco_lead_purity(self):
        """Return the backtracked hit purity for each leading PFO.

            Returns:
                Any: Purity values selected using the ``leading`` indices.
        """
        return self.events.true_particles_bt.purity[self.leading]

    @property
    def reco_sub_purity(self):
        """Return the backtracked hit purity for each subleading PFO.

            Returns:
                Any: Purity values selected using the ``subleading`` indices.
        """
        return self.events.true_particles_bt.purity[self.subleading]

    @property
    def reco_lead_completeness(self):
        """Return the backtracked hit completeness for each leading PFO.

            Returns:
                Any: Completeness values selected using the ``leading``
                    indices.
        """
        return self.events.true_particles_bt.completeness[self.leading]

    @property
    def reco_sub_completeness(self):
        """Return the backtracked hit completeness for each subleading PFO.

            Returns:
                Any: Completeness values selected using the ``subleading``
                    indices.
        """
        return self.events.true_particles_bt.completeness[self.subleading]

    @property
    def reco_lead_purity_collection(self):
        """Return collection-plane backtracked hit purity for leading PFOs.

            Returns:
                Any: Collection-plane purity values selected by ``leading``.
        """
        return self.events.true_particles_bt.purity_collection[self.leading]

    @property
    def reco_sub_purity_collection(self):
        """Return collection-plane backtracked hit purity for subleading PFOs.

            Returns:
                Any: Collection-plane purity values selected by
                    ``subleading``.
        """
        return self.events.true_particles_bt.purity_collection[self.subleading]

    @property
    def reco_lead_completeness_collection(self):
        """Return collection-plane backtracked hit completeness for leading PFOs.

            Returns:
                Any: Collection-plane completeness values selected by
                    ``leading``.
        """
        return self.events.true_particles_bt.completeness_collection[self.leading]

    @property
    def reco_sub_completeness_collection(self):
        """Return collection-plane backtracked hit completeness for subleading PFOs.

            Returns:
                Any: Collection-plane completeness values selected by
                    ``subleading``.
        """
        return self.events.true_particles_bt.completeness_collection[self.subleading]

    @property
    def reco_angle(self):
        """Calculate the opening angle between reconstructed shower directions.

            Returns:
                Any: Per-pair opening angles in radians.
        """
        return ShowerPairs.opening_angle(
            self.events.recoParticles.shower_direction[self.leading],
            self.events.recoParticles.shower_direction[self.subleading])

    @property
    def reco_mass(self):
        """Calculate reconstructed invariant mass for each shower pair.

            Returns:
                Any: Per-pair invariant masses from the two reconstructed
                    shower energies and their opening angle.
        """
        return ShowerPairs.mass(
            self.events.recoParticles.shower_energy[self.leading],
            self.events.recoParticles.shower_energy[self.subleading],
            self.reco_angle)

    @property
    def reco_energy(self):
        """Sum the reconstructed energies of each shower pair.

            Returns:
                Any: Per-pair sums of the leading and subleading shower
                    energies.
        """
        return ShowerPairs.energy(
            self.events.recoParticles.shower_energy[self.leading],
            self.events.recoParticles.shower_energy[self.subleading])

    @property
    def reco_pi0_mom(self):
        """Add the leading reconstructed shower momentum to itself.

            Returns:
                Any: Per-pair vector sum computed from the leading shower
                    momentum twice.
        """
        return ShowerPairs.pi0_mom(
            self.events.recoParticles.shower_momentum[self.leading],
            self.events.recoParticles.shower_momentum[self.leading])

    @property
    def reco_pi0_mom_mag(self):
        """Return the magnitude of the doubled leading-shower momentum.

            Returns:
                Any: Per-pair magnitudes of ``reco_pi0_mom``.
        """
        return vector.magnitude(ShowerPairs.pi0_mom(
            self.events.recoParticles.shower_momentum[self.leading],
            self.events.recoParticles.shower_momentum[self.leading]))

    @property
    def reco_direction(self):
        """Normalize the reconstructed pair momentum vector.

            Returns:
                Any: Per-pair unit vectors derived from ``reco_pi0_mom``.
        """
        return vector.normalize(self.reco_pi0_mom)

    @property
    def reco_closest_approach(self):
        """Calculate the signed closest separation of the two reconstructed shower lines.

            Returns:
                Any: Per-pair signed perpendicular separation derived from
                    shower directions and start positions.
        """
        return ShowerPairs.closest_approach(
            self.events.recoParticles.shower_direction[self.leading],
            self.events.recoParticles.shower_direction[self.subleading],
            self.events.recoParticles.shower_start_pos[self.leading],
            self.events.recoParticles.shower_start_pos[self.subleading])

    @property
    def reco_separation(self):
        """Calculate the distance between reconstructed shower start positions.

            Returns:
                Any: Per-pair Euclidean distances between the leading and
                    subleading shower starts.
        """
        return vector.dist(
            self.events.recoParticles.shower_start_pos[self.leading],
            self.events.recoParticles.shower_start_pos[self.subleading])

    @property
    def reco_beam_slice(self):
        """Return the shared reconstruction slice ID for each shower pair.

            Returns:
                Any: The slice ID when both PFPs share a slice, or ``-1`` when
                    their slice IDs differ.
        """
        first_slice = self.events.recoParticles.slice_id[self.leading]
        second_slice = self.events.recoParticles.slice_id[self.subleading]
        return (first_slice == second_slice) * (first_slice + 1) - 1

    ##############################################################################################

    @property
    def true_lead_energy(self):
        """Return the backtracked truth energy of each leading PFP.

            Returns:
                Any: Truth-energy values selected using the ``leading``
                    indices.
        """
        return self.events.true_particles_bt.energy[self.leading]

    @property
    def true_sub_energy(self):
        """Return the backtracked truth energy of each subleading PFP.

            Returns:
                Any: Truth-energy values selected using the ``subleading``
                    indices.
        """
        return self.events.true_particles_bt.energy[self.subleading]

    @property
    def true_angle(self):
        """Calculate the opening angle between backtracked truth directions.

            Returns:
                Any: Per-pair truth opening angles in radians.
        """
        return ShowerPairs.opening_angle(
            self.events.true_particles_bt.direction[self.leading],
            self.events.true_particles_bt.direction[self.subleading])

    @property
    def true_mass(self):
        """Calculate the invariant mass of each backtracked truth pair.

            Returns:
                Any: Per-pair invariant masses from the two backtracked
                    particle energies and their truth opening angle.
        """
        return ShowerPairs.mass(
            self.events.true_particles_bt.energy[self.leading],
            self.events.true_particles_bt.energy[self.subleading],
            self.true_angle)

    @property
    def true_energy(self):
        """Sum the backtracked truth energies of each pair.

            Returns:
                Any: Per-pair sums of leading and subleading truth energies.
        """
        return ShowerPairs.energy(
            self.events.true_particles_bt.energy[self.leading],
            self.events.true_particles_bt.energy[self.subleading])

    @property
    def true_pi0_mom(self):
        """Sum the backtracked momentum vectors of the two PFPs in each pair.

            Returns:
                Any: Per-pair vector sums of leading and subleading truth
                    momenta.
        """
        return ShowerPairs.pi0_mom(
            self.events.true_particles_bt.momentum[self.leading],
            self.events.true_particles_bt.momentum[self.subleading])

    @property
    def true_pi0_mom_mag(self):
        """Return the magnitude of the summed backtracked truth momentum.

            Returns:
                Any: Per-pair magnitudes of ``true_pi0_mom``.
        """
        return vector.magnitude(ShowerPairs.pi0_mom(
            self.events.true_particles_bt.momentum[self.leading],
            self.events.true_particles_bt.momentum[self.subleading]))

    @property
    def true_direction(self):
        """Normalize the summed backtracked truth momentum.

            Returns:
                Any: Per-pair unit vectors derived from ``true_pi0_mom``.
        """
        return vector.normalize(self.true_pi0_mom)

    @property
    def true_closest_approach(self):
        """Calculate the signed closest separation of the two truth trajectories.

            Returns:
                Any: Per-pair signed perpendicular separation derived from
                    backtracked directions and start positions.
        """
        return ShowerPairs.closest_approach(
            self.events.true_particles_bt.direction[self.leading],
            self.events.true_particles_bt.direction[self.subleading],
            self.events.true_particles_bt.shower_start_pos[self.leading],
            self.events.true_particles_bt.shower_start_pos[self.subleading])

    @property
    def true_separation(self):
        """Calculate the distance between backtracked truth start positions.

            Returns:
                Any: Per-pair Euclidean distances between the leading and
                    subleading truth-particle starts.
        """
        return vector.dist(
            self.events.true_particles_bt.shower_start_pos[self.leading],
            self.events.true_particles_bt.shower_start_pos[self.subleading])

    ##############################################################################################
    @property
    def cheated_lead_energy(self):
        """Return the leading backtracked energy-by-hits estimate.

            Returns:
                Any: Energy-by-hits values selected using the ``leading``
                    indices.
        """
        return self.events.true_particles_bt.energy_by_hits[self.leading]

    @property
    def cheated_sub_energy(self):
        """Return the subleading backtracked energy-by-hits estimate.

            Returns:
                Any: Energy-by-hits values selected using the ``subleading``
                    indices.
        """
        return self.events.true_particles_bt.energy_by_hits[self.subleading]

    @property
    def cheated_angle(self):
        """Use the reconstructed shower opening angle for the energy-by-hits comparison.

            Returns:
                Any: The same per-pair angle values returned by
                    ``reco_angle``.
        """
        return self.reco_angle

    @property
    def cheated_mass(self):
        """Calculate pair invariant mass using backtracked energy-by-hits values.

            Returns:
                Any: Per-pair invariant masses from the two energy-by-hits
                    values and ``cheated_angle``.
        """
        return ShowerPairs.mass(
            self.events.true_particles_bt.energy_by_hits[self.leading],
            self.events.true_particles_bt.energy_by_hits[self.subleading],
            self.cheated_angle)

    @property
    def cheated_energy(self):
        """Sum the backtracked energy-by-hits estimates for each pair.

            Returns:
                Any: Per-pair sums of the leading and subleading energy-by-hits
                    values. This property definition is shadowed later in the
                    class by the ``cheated_energy`` fractional-error property.
        """
        return ShowerPairs.energy(
            self.events.true_particles_bt.energy_by_hits[self.leading],
            self.events.true_particles_bt.energy_by_hits[self.subleading])

    @property
    def cheated_pi0_mom(self):
        """Sum the backtracked energy-by-hits momentum vectors in each pair.

            Returns:
                Any: Per-pair vector sums of the leading and subleading
                    momentum-by-hits vectors.
        """
        return ShowerPairs.pi0_mom(
            self.events.true_particles_bt.momentum_by_hits[self.leading],
            self.events.true_particles_bt.momentum_by_hits[self.subleading])

    @property
    def cheated_pi0_mom_mag(self):
        """Return the magnitude of the summed momentum-by-hits vectors.

            Returns:
                Any: Per-pair magnitudes of ``cheated_pi0_mom``.
        """
        return vector.magnitude(ShowerPairs.pi0_mom(
            self.events.true_particles_bt.momentum_by_hits[self.leading],
            self.events.true_particles_bt.momentum_by_hits[self.subleading]))

    @property
    def cheated_direction(self):
        """Normalize the summed momentum-by-hits vector.

            Returns:
                Any: Per-pair unit vectors derived from ``cheated_pi0_mom``.
        """
        return vector.normalize(self.cheated_pi0_mom)

    @property
    def cheated_closest_approach(self):
        """Reuse the backtracked truth trajectories' closest-approach value.

            Returns:
                Any: The same per-pair signed separation returned by
                    ``true_closest_approach``.
        """
        return self.true_closest_approach

    @property
    def cheated_separation(self):
        """Reuse the distance between the backtracked truth start positions.

            Returns:
                Any: The same per-pair distance returned by ``true_separation``.
        """
        return self.true_separation

    ##############################################################################################
    @property
    def error_lead_energy(self):
        """Calculate the fractional energy error for each leading shower.

            Returns:
                Any: ``(reconstructed / truth) - 1`` for each leading shower.
        """
        return ShowerPairs.fractional_error(self.reco_lead_energy, self.true_lead_energy)

    @property
    def error_sub_energy(self):
        """Calculate the fractional energy error for each subleading shower.

            Returns:
                Any: ``(reconstructed / truth) - 1`` for each subleading
                    shower.
        """
        return ShowerPairs.fractional_error(self.reco_sub_energy, self.true_sub_energy)

    @property
    def error_angle(self):
        """Calculate the fractional error in reconstructed opening angle.

            Returns:
                Any: ``(reconstructed angle / truth angle) - 1`` per pair.
        """
        return ShowerPairs.fractional_error(self.reco_angle, self.true_angle)

    @property
    def error_mass(self):
        """Calculate the fractional error in reconstructed pair invariant mass.

            Returns:
                Any: ``(reconstructed mass / truth mass) - 1`` per pair.
        """
        return ShowerPairs.fractional_error(self.reco_mass, self.true_mass)

    @property
    def cheated_energy(self):
        """Calculate the fractional error in summed reconstructed shower energy.

            Returns:
                Any: ``(reconstructed pair energy / truth pair energy) - 1``.
        """
        return ShowerPairs.fractional_error(self.reco_energy, self.true_energy)

    @property
    def error_pi0_mom(self):
        """Calculate the component-wise fractional error in pair momentum.

            Returns:
                Any: Per-pair x, y, and z fractional differences between
                    reconstructed and backtracked truth momentum.
        """
        return ShowerPairs.fractional_error(self.reco_pi0_mom, self.true_pi0_mom)

    @property
    def error_pi0_mom_mag(self):
        """Calculate the fractional error in pair momentum magnitude.

            Returns:
                Any: ``(reconstructed magnitude / truth magnitude) - 1`` per
                    pair.
        """
        return ShowerPairs.fractional_error(self.reco_pi0_mom_mag, self.true_pi0_mom_mag)

    @property
    def error_closest_approach(self):
        """Calculate the fractional difference in signed closest-approach values.

            Returns:
                Any: ``(reconstructed / truth) - 1`` for each pair's signed
                    perpendicular separation.
        """
        return ShowerPairs.fractional_error(self.reco_closest_approach,
                                           self.true_closest_approach)

    @property
    def error_separation(self):
        """Calculate the fractional error in shower-start separation.

            Returns:
                Any: ``(reconstructed separation / truth separation) - 1``
                    per pair.
        """
        return ShowerPairs.fractional_error(self.reco_separation,
                                           self.true_separation)

    def calculate_all(self) -> pd.DataFrame:
        """ Calculates all possible quantities and stores the results in a dataframe.

            Returns:
                pd.DataFrame: dataframe.
        """
        #* search terms are based on the prefix (caae sensitive) of the property methods, when adding new quantities they should conform to this standard
        search_terms = ["reco", "true", "cheated", "error"]

        df = {}
        for search in search_terms:
            for k in vars(ShowerPairs):
                if search in k:
                    print(k)
                    try:
                        q = getattr(self, k)
                        if hasattr(q, "x"):
                            # add individual vector components separately to the dataframe
                            df.update({k + "_x" : ak.ravel(q.x), k + "_y" : ak.ravel(q.y), k + "_z" : ak.ravel(q.z)})
                        else:
                            df.update({k: ak.ravel(q)})
                    except:
                        print(f"couldn't calculate {k}")
        df = pd.DataFrame(df)

        metadata = {
            "event": self.events.event_num,
            "run": self.events.run,
            "subrun": self.events.sub_run
        }
        df = pd.concat([pd.DataFrame(metadata), df], 1)

        print(df)
        return df

    def save_to_csv(self, name: str = "shower-pair-quantities"):
        """ Save all quantities to a csv file.
            Will overwrite existing files!

            Args:
                name (str): file name (exluding extentions). Defaults to "shower-pair-quantities".
        """
        self.calculate_all().to_csv(name.split(".csv")[0] + ".csv")


__all__ = [
    "FileDescriptor",
    "Sample",
    "file_len",
    "NtupleType",
    "Data",
    "ParticleData",
    "TrueParticleData",
    "RecoParticleData",
    "TrueParticleDataBT",
    "ShowerPairs",
]
