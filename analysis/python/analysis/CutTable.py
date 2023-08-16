import warnings
import numpy as np
import awkward as ak
import pandas as pd
import copy
import functools
from particle import Particle
from python.analysis import Tags
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from python.analysis import Master


class MaskIter():
    def __init__(self, start_sigs, simultaneous, masks):
        self.sigs = start_sigs
        self.masks = masks
        self.iter_max = len(masks)
        self.index = 0

        if self.iter_max != len(self.sigs):
            raise ValueError(f"masks and sigs must have the same length: {self.iter_max} and {len(self.sigs)}")
        if self.index >= self.iter_max:
            raise IndexError(f"Start index {self.index} is out of bounds for masks length of {self.iter_max}")

        self.curr_evts = []
        self.curr_pfos = []

        self.mask_inds = list(range(self.iter_max))
        self.next_simul = simultaneous[1:] + [False]
        self.init_simul = []
        last_simul = 0
        for i, this_simul in enumerate(simultaneous):
            if not this_simul:
                last_simul = i
            self.init_simul.append(last_simul)
        return
    
    def __iter__(self):
        self.index = 0
        self.curr_mask = self.masks[0]
        return self

    def check_simul_masks(self, sig1, sig2):
        return (sig1[0] == sig2[0]) and ((sig1[1] == sig2[-1]) or (sig1[1] == -1) or (sig2[1] == -1))

    def comb_masks(self, mask_list):
        return functools.reduce(lambda x, y: np.logical_and(x, y), mask_list)

    def gen_mask_applier(self):
        if len(self.curr_evts) == 0:
            mask = self.comb_masks(self.curr_pfos)
            def mask_applier(data):
                return data[mask]
        elif len(self.curr_pfos) == 0:
            mask = self.comb_masks(self.curr_evts)
            def mask_applier(data):
                return data[mask]
        else:
            evt_mask = self.comb_masks(self.curr_evts)
            pfo_mask = self.comb_masks(self.curr_pfos)
            def mask_applier(data):
                return data[pfo_mask][evt_mask]
        return mask_applier
    
    def add_mask_at_index(self, index):
        if self.sigs[index][1] == -1:
            self.curr_evts.append(self.masks[index])
        else:
            self.curr_pfos.append(self.masks[index])
        return

    def get_mask_func(self, apply_func, next_simul):
        if next_simul:
            def apply_mask_func(data):
                return apply_func, data
        else:
            def apply_mask_func(data):
                return apply_func, apply_func(data)
        return apply_mask_func

    def __next__(self):
        if self.index >= self.iter_max:
            raise StopIteration
        if self.init_simul[self.index] == self.index:
            self.curr_evts = []
            self.curr_pfos = []
        self.add_mask_at_index(self.index)
        self.curr_func = self.gen_mask_applier()
        self.index += 1
        return self.get_mask_func(self.curr_func, self.next_simul[self.index-1])
        
    def __getitem__(self, sli : slice):
        """
        NOT YET UPDATED TO PROPERLY HANDLE PFO/EVENT MIXING
        """
        raise NotImplementedError
        if (sli.step is not None) and (sli.step < 0):
            rev_slice = slice(None, None, -1)
            sli = slice(sli.stop + 1, sli.start + 1, abs(sli.step))
        else:
            rev_slice = slice(None, None, None)
        this_index = self.mask_inds[sli]
        init_masks = self.init_simul[sli]
        next_simuls = self.next_simul[sli]
        res = []
        last_end = init_masks[0]
        curr_mask = self.masks[last_end]
        for this_i, init_i, next_simul in zip(this_index, init_masks, next_simuls):
            if init_i > last_end:
                last_end = init_i
                curr_mask = self.masks[last_end]
            for new_mask in self.masks[last_end+1:this_i+1]:
                curr_mask = np.logical_and(curr_mask, new_mask)
            res.append(self.get_mask_func(curr_mask, next_simul))
        return res[rev_slice]
    

class CutHandler():
    _default_particles = [
        211, -211, 13, -13,
        11, -11, 22, 2212, 321]
    _event_table_cols = np.array([
            "Remaining events", "Percentage of total events remaining",
            "Relative percentage events"])
    _pfo_table_cols = np.array([
            "Remaining PFOs", "Percentage of total PFOs remaining",
            "Relative percentage of PFOs", "Average PFOs per event"])
    _particle_table_cols = [
            "remaining", "percentage remaining",
            "relative percentage remaining", "average per event"]

    # @staticmethod
    # def _init_doc(func):
    #     documentation = """
    #     Create new a `CutHandler` object to store applied masks
    #     and return a table of tracked particles when asked.

    #     Parameters
    #     ----------
    #     evts : Master.Data, optional
    #         Initial events the masks are being applied from.
    #         If not set here, it must be set for getting a
    #         table using `self.set_init_evts(evts)`. Default
    #         is None.
    #     particles_to_tag : list, optional
    #         List of pdg codes to be tracked by the tables.
    #         Default is
    #     """ + f"{CutHandler._default_particles}." + """
        
    #     Returns
    #     -------
    #     CutHandler
    #         New `CutHandler` instance.
    #     """
    #     func.__doc__ = documentation
    #     return func
    # @_init_doc
    def __init__(
            self,
            evts=None,#: Master.Data = None,
            particles_to_tag: list = _default_particles):
        """
        Create new a `CutHandler` object to store applied masks
        and return a table of tracked particles when asked.

        Parameters
        ----------
        evts : Master.Data, optional
            Initial events the masks are being applied from.
            If not set here, it must be set for getting a
            table using `self.set_init_evts(evts)`. Default
            is None.
        particles_to_tag : list, optional
            List of pdg codes to be tracked by the tables.
            Default is [211, -211, 13, -13,11, -11, 22, 2212, 321]
        
        Returns
        -------
        CutHandler
            New `CutHandler` instance.
        """
        self._masks = []
        self._signatures = []
        self._simultaneous = []
        self._start_sig = ()
        self._end_sig = ()
        self.curr_mask_index = 0
        self.concat_index = 0
        self.concat_indicies = [0]
        self._pfos_init = False
        self._init_data = None
        self._particle_tags = None
        self._data_changed = True
        self.init_events_set = False
        if evts is not None:
            self.set_init_evts(evts)
        self._last_pfo_counts = None
        self._names=[]

        self._particles = particles_to_tag

        self._table_data = {}
        return

    def _gen_basic_counts(self):
        init_evt_count = ak.num(self._init_data, axis=0)
        init_pfo_count = ak.count(self._init_data)
        results = {
            "Name":["Initial data"],
            "Remaining events":[init_evt_count],
            "Percentage of total events remaining":[100.],
            "Relative percentage events":[100.],
            "Remaining PFOs":[init_pfo_count],
            "Percentage of total PFOs remaining":[100.],
            "Relative percentage of PFOs":[100.],
            "Average PFOs per event":[init_pfo_count/init_evt_count]}
        last_evt_count = init_evt_count
        last_pfo_count = init_pfo_count
        data = self._init_data
        for name, application_func in zip(self._names, self):
            this_applier, new_data = application_func(data)
            cut_data = this_applier(data)
            this_evt_count = ak.num(cut_data, axis=0)
            this_pfo_count = ak.count(cut_data)
            results["Name"].append(name)
            results["Remaining events"].append(this_evt_count)
            results["Percentage of total events remaining"].append(100.*this_evt_count/init_evt_count)
            results["Relative percentage events"].append(100.*this_evt_count/last_evt_count)
            results["Remaining PFOs"].append(this_pfo_count)
            results["Percentage of total PFOs remaining"].append(100.*this_pfo_count/init_pfo_count)
            results["Relative percentage of PFOs"].append(100.*this_pfo_count/last_pfo_count)
            results["Average PFOs per event"].append(this_pfo_count/this_evt_count)
            last_evt_count = this_evt_count
            last_pfo_count = this_pfo_count
            data = new_data
        return results

    def _gen_particle_counts(self, init_list):
        init_count = np.sum(init_list)
        results = {
            "remaining": [init_count],
            "percentage remaining": [100.],
            "relative percentage remaining": [100.],
            "average per event": [init_count/ak.num(init_list, axis=0)]}
        data = init_list
        for application_func in self:
            this_applier, new_data = application_func(data)
            cut_data = this_applier(data)
            particle_count = np.sum(cut_data)
            results["remaining"].append(particle_count)
            results["percentage remaining"].append(100. * particle_count/init_count)
            results["relative percentage remaining"].append(
                100. * particle_count/results["remaining"][-2])
            results["average per event"].append(particle_count/ak.num(cut_data, axis=0))
            data = new_data
        return results

    # def GeneratePi0Tags(self, evts : Master.Data):
    #     photons_mask = evts.trueParticlesBT.pdg == 22
    #     return Tags.GeneratePi0Tags(evts, photons_mask)

    def set_init_evts(self, evts):#: Master.Data):
        """
        Set the initial event counts to be cut using the masks.
        This is automatically performed if as `Master.Data`
        object is passed in initialisation.
        
        This must be performed prior to generating a table.

        Use `self.init_data_set` for a boolean indicating if
        the events have been set.

        N.B. if we want to to this for truth, we must change evts
        to be an acutal akward array (i.e.
        `evts.recoParticles.number` for reco and
        `evts.trueParticles.number` for truth). At the moment,
        we take a Master.Data object an pull the
        `evts.recoParticles.number` from it.

        Parameters
        ----------
        evts : Master.Data
            Initial events the masks are being applied from.
        """
        self._init_data = ak.ones_like(evts.recoParticles.number, dtype=bool)
        self._init_data_sig = self._get_mask_signature(self._init_data)
        self._particle_tags = Tags.GenerateTrueParticleTags(evts)
        self._data_changed = True
        self.init_events_set = True
        return

    def _comb_masks(self, mask_list):
        if len(mask_list) >= 2:
            return [functools.reduce(lambda x, y: np.logical_and(x, y), mask_list)]
        else:
            return mask_list

    def _group_masks(self, start_index=None, end_index=None, rcount=np.inf):
        groups_count = 0
        curr_pfos_like = []
        curr_evts_like = []
        ordered_results = []
        if start_index is not None:
            if start_index == 0:
                start_index = None
            else:
                start_index -= 1
        if end_index is not None:
            if end_index == 0:
                return []
            end_index -= 1
        for mask, sig, simul in zip(
                self._masks[end_index:start_index:-1],
                self._signatures[end_index:start_index:-1],
                self._simultaneous[end_index:start_index:-1]):
            if groups_count >= rcount:
                break
            if sig[1] == -1:
                curr_evts_like = [mask] + curr_evts_like
            else:
                curr_pfos_like = [mask] + curr_pfos_like
            if (not simul) is True:
                ordered_results = (
                    self._comb_masks(curr_pfos_like)
                    + self._comb_masks(curr_evts_like)
                    + ordered_results)
                curr_evts_like = []
                curr_pfos_like = []
                groups_count += 1
        ordered_results = (
            self._comb_masks(curr_pfos_like)
            + self._comb_masks(curr_evts_like)
            + ordered_results)
        return ordered_results

    def _get_mask_signature(self, mask, end=False):
        flat_array = isinstance(ak.count(mask, axis=0), int)
        if not end:
            pfo_level = -1 if flat_array else ak.count(mask)
            start_sig = (ak.num(mask, axis=0), pfo_level)
            return start_sig
        curr_simul_masks = self._group_masks(rcount=1)
        mask = curr_simul_masks[0]
        for m in curr_simul_masks[1:]:
            mask = mask[m]
        if flat_array:
            end_sig = (ak.sum(mask), -1)
        else:
            end_sig = (ak.num(mask, axis=0), ak.sum(mask))
        return end_sig
    
    def _validate_signature(self, signature, raise_exception=True, return_simul=False):
        if self._start_sig == ():
            return False
        simul_mask = (self._start_sig[0] == signature[0]) and ((self._start_sig[-1] == signature[-1]) or (self._start_sig[-1] == -1) or (signature[-1] == -1))
        good_mask = simul_mask or (self._end_sig[0] == signature[0] and ((self._end_sig[-1] == signature[-1]) or (self._end_sig[-1] == -1) or (signature[-1] == -1)))
        if (not good_mask) and raise_exception:
            raise ValueError(f"Mask signature ({signature[0]} events, {signature[1]} PFOs) does not match"
                             + f" the required signature of ({self._start_sig[0]} events, "
                             + f"{self._start_sig[1]} PFOs) for a simultaenous mask, or ({self._end_sig[0]}"
                             + f" events, {self._end_sig[1]} PFOs) for a consequtive mask.")
        if return_simul:
            return simul_mask
        return good_mask

    def add_mask(self, mask: ak.Array, name: str = "-"):
        """
        Add a mask to the instance.

        Parameters
        ----------
        mask : ak.Array
            Boolean mask of the cut applied.
        name : str
            Name indicating what the cut did.
        """
        this_sig = self._get_mask_signature(mask)
        self._simultaneous.append(self._validate_signature(this_sig, return_simul=True))
        self._start_sig = this_sig
        self._masks.append(mask)
        self._signatures.append(this_sig)
        # Must occur after _masks, _signatures, and _simultaneous are set
        self._end_sig = self._get_mask_signature(mask, end=True)
        self._names.append(name)
        self.curr_mask_index += 1
        return

    def _mask_appliers(self):
        return MaskIter(self._signatures, self._simultaneous, self._masks)

    def __iter__(self):
        return self._mask_appliers()

    def copy(self):
        return copy.deepcopy(self)
    
    def get_masks(
            self,
            initial_concat_index: int = 0,
            final_concat_index: int = None,
            initial_true_index: int = None,
            final_true_index: int = None):
        """
        Returns the stored masks

        Parameters
        ----------
        initial_concat_index : int, optional
            Concatenation from which to begin fetching masks.
            Will be overwritten by `true_index_start` if
            passed. Default is 0.
        final_concat_index : int, optional
            Concatenation from which to finish fetching
            masks (the first resut of this concatenation
            will _not_ be inculded). Will be overwritten by
            `true_index_end` if `true_index_start` is passed.
            Default is None.
        initial_true_index : int, optional
            Index from which to begin fetching masks.
            Overwrites `concat_index_start` if passed.
            Default is None.
        final_true_index : int, optional
            Index to finish fetching masks (the supplied
            index will not be included). Only used if
            `true_index_start` is passed. Default is None.
        
        Returns
        -------
        list
            List of requested masks
        """
        if initial_true_index is None:
            initial_true_index = self.concat_indicies[initial_concat_index]
            if final_concat_index is not None:
                final_true_index = self.concat_indicies[final_concat_index]+1
            else:
                final_true_index = None
        reco_masks = self._group_masks(
            start_index=initial_true_index, end_index=final_true_index)
        truth_masks = [m for m in reco_masks if isinstance(ak.count(m, axis=0), int)]
        return reco_masks, truth_masks

    def apply_masks(
            self, data: ak.Array,
            return_table: bool = False,
            application_concat_index: int = 0,
            application_true_index: int = None):
        """
        DO NOT USE - yet to be fully implmented.

        Apply the masks in the instance to the data.

        Parameters
        ----------
        data : ak.Array
            Data to which the masks are applied. Must
            match the signature of the first mask.
        return_table : bool, optional
            If `True`, the both the filtered mask and
            corresponding table will be returned. Default
            is False.
        application_concat_index : int, optional
            Initial concatenation from which to apply masks.
            If data has already been filtered, use this to
            change from where the mask application begins.
            Current concatenation index can be seen using
            `self.concate_index`. Default is 0.
        application_true_index : int, optional
            Selects an arbitray mask as the starting point from
            which to begin applying masks. This can be used
            selections have not been grouped via concatenations.
            Current mask index can be seen using
            `self.curr_mask_index`. This will override
            `initial_concat_index` if set. Default is None.
        
        Returns
        -------
        ak.Array
            Filtered data.
        pd.DataFrame, optional
            Cuts table, only returned if `return_table` is
            True.
        """
        raise NotImplementedError
        # TODO exception throwing
        if application_true_index is not None:
            initial_index = application_true_index
        else:
            initial_index = self.concat_indicies[application_concat_index]
        result = data
        for application_func in self._mask_appliers()[initial_index:]:
            _, result = application_func(result)
        if return_table:
            return result, self.get_table()
        else:
            return result
    
    # def get_filters_list(self, application_concat_index=0):
    #     new_data = data
    #     for application_func in self._mask_appliers()[self.concat_indicies[application_concat_index]:]:
    #         mask, _ = application_func(result)
        

    def _gen_table(self):
        if self._init_data is None:
            raise Exception("Initial data not added, supply events using set_init_events.")
        if len(self._masks) == 0:
            raise IndexError("No masks have been added")
        if not self._data_changed:
            return
        self._table_data.update(self._gen_basic_counts())
        for particle in self._particles:
            p_data = self._gen_particle_counts(self._particle_tags[f"${Particle.from_pdgid(particle).latex_name}$"].mask)
            self._table_data.update(dict((f"{particle} {key}", value) for (key, value) in p_data.items()))
        self._data_changed = False
        return

        # if not self._check_signature(self._get_mask_signature(self._masks[0]), self._init_data_sig):
        #     raise ValueError(f"Initial data signature {self._init_data_sig} does not match the first avaiable mask {self._get_mask_signature(self._masks[0])}")
    
    def _gen_particle_cols_array(self, pdg):
        return np.array(list(map(lambda c: f"{pdg} {c}", self._particle_table_cols)))
    def _gen_particle_cols_array_read(self, pdg, latex):
        if latex:
            return np.array(list(map(lambda c: f"${Particle.from_pdgid(pdg).latex_name}$ {c}", self._particle_table_cols)))
        else:
            return np.array(list(map(lambda c: f"{Particle.from_pdgid(pdg)} {c}", self._particle_table_cols)))

    def _new_init_events(self, col, index, init_name):
        if index == 0:
            result = self._table_data[col]
        else:
            result = self._table_data[col][index:]
            if "percentage" in col.lower():
                if "relative" not in col.lower():
                    init_percent = result[0]
                    result = list(map(lambda p: 100 * p/init_percent, result))
                else:
                    result[0] = 100.
        if col == "Name":
            result[0] = init_name
        return result

    def get_table(
            self, init_data_name: str = "Initial data",
            initial_concat_index: int = 0,
            final_concat_index: int = None,
            initial_true_index: int = None,
            final_true_index: int = None,
            latex: bool = False,
            particles_list: list = _default_particles,
            events: bool = True, pfos: bool = True,
            counts: bool = True, percent_remain: bool = True,
            relative_percent: bool = True, ave_per_event: bool = True):
        """
        Generate the cuts table for the masks shown to the instance.

        The initial events must have been set prior to calling this,
        either in the initialisation, or the `set_init_events`
        method.

        Parameters
        ----------
        init_data_name : str, optional
            Name to be displayed for the pre-cuts row. Default is
            `"Initial data"`
        initial_concat_index : int, optional
            Initial concatenation from which to generate the table
            This can be used to i.e. removing initial event
            selection cuts if they are not of interest. Current
            concatenation index can be seen using
            `self.concate_index`. Default is 0.
        final_concat_index : int, optional
            Final concatenation at which to stop generating the
            table. Current concatenation index can be seen using
            `self.concate_index`. Default is None.
        initial_true_index : int, optional
            Selects an arbitray mask as the starting point from
            which to generate the table. This can be used
            selections have not been grouped via concatenations.
            Current mask index can be seen using
            `self.curr_mask_index`. This will override
            `initial_concat_index` if set. Default is None.
        final_true_index : int, optional
            Selects an arbitray mask as the final point at which
            to stop generating the table. Indexed mask will _not_
            be included. Current mask index can be seen using
            `self.curr_mask_index`. This will override
            `final_concat_index1` if `initial_concat_index` is
            set. Default is None.
        latex : bool, optional
            If `True`, result is a string which may be copied
            directly into LaTeX. Default is False.
        particles_list: list, optional
            List of pdg codes of particles to be included in the
            table. Default is [211, -211, 13, -13,11, -11, 22,
            2212, 321]
        events: bool, optional
            If `True`, columns indicating the number of events
            present are included. Default is True.
        pfos: bool, optional
            If `True`, columns indicating the total number of
            PFOs present are included. Default is True.
        counts: bool, optional
            If `True`, columns indicating the number of
            occurances for each tracked item are included.
            Default is True.
        percent_remain: bool, optional
            If `True`, columns indicating the percentage of the
            intial count for each tracked item are included.
            Default is True.
        relative_percent: bool, optional
            If `True`, columns indicating the relative percentage
            remaining compared to the previous row for each
            tracked item are included. Default is True.
        ave_per_event: bool, optional
            If `True`, columns indicating the mean number of
            occurances per event for each tracked item are
            included. Not included for events data. Default is
            True.
        
        Returns
        -------
        pd.DataFrame or str
            Table of results as a DataFrame if `latex` is False,
            or a string if `latex` is True.
        """
        self._gen_table()

        if initial_concat_index > self.concat_index:
            raise IndexError(
                f"Concatenation index {initial_concat_index} out of range, "
                + f"only {self.concat_index} concatenation(s) present.")
        cols_to_use = ["Name"]
        col_names = ["Name"]
        cols_to_keep = np.array([counts, percent_remain, relative_percent, ave_per_event])
        if events:
            cols = self._event_table_cols[cols_to_keep[:-1]].tolist()
            cols_to_use += cols
            col_names += cols
        if pfos:
            cols = self._pfo_table_cols[cols_to_keep].tolist()
            cols_to_use += cols
            col_names += cols
        for p in particles_list:
            cols_to_use += self._gen_particle_cols_array(p)[cols_to_keep].tolist()
            col_names += self._gen_particle_cols_array_read(p, latex)[cols_to_keep].tolist()

        if initial_true_index is None:
            initial_true_index = self.concat_indicies[initial_concat_index]
            if final_concat_index is not None:
                final_true_index = self.concat_indicies[final_concat_index]+1
            else:
                final_true_index = None
        data = {}
        for col, name in zip(cols_to_use, col_names):
            data[name] = self._new_init_events(
                col, initial_true_index, init_data_name)[:final_true_index]
        if latex:
            return pd.DataFrame(data).to_latex()
        else:
            return pd.DataFrame(data)

    def concatenate(self, other, return_copy=False):
        """
        Concatenates a second `CutHandler` instance.

        Also usable via the addition operator (which returns
        a copy).

        The combined result will include data from both
        instances with this instance first and the `other`
        instance second.

        The `other` instance have a compatible initial mask
        signature with that of the final mask in this
        instance.

        The index of this concatenation is stored in
        `self.concat_indicies`, and the current index is
        stored as `self.concat_index`. The concatenation
        index may be used in `get_masks` and `get_table` to
        index a particular set of cuts.

        Parameters
        ----------
        other : CutHandler
            Other CutHandler object to be concatenate to
            this instance.
        return_copy : bool, optional
            If True, a new instance is created to contain
            the concatenation, whilst this instance does
            not change. Default is False.
        
        Returns
        -------
        CutHandler, optional
            Combine CutHandler instance, if `return_copy`
            was set to True.
        """
        if not isinstance(other, CutHandler):
            raise NotImplementedError("CutHandler object can only be added to another CutHandler object")
        this_set = (self._start_sig != ())
        # Make sure the new set is consequtive
        if (not self._validate_signature(other._signatures[0], raise_exception=False)) and this_set:
            raise ValueError(f"Combining CutHandler objects requires consequtively matching signatures: "
                             + f"{self._end_sig} and {other._signatures[0]} do not match.")
        if return_copy:
            result = self.copy()
        else:
            result = self
        result.concat_indicies += [i + len(result._masks) for i in other.concat_indicies]
        result._masks += other._masks
        result._names += other._names
        result._signatures += other._signatures
        result._simultaneous += other._simultaneous
        result._start_sig = other._start_sig
        result._end_sig = other._end_sig
        result.concat_index += 1
        result._data_changed = True
        if (result._init_data is None) and (other._init_data is not None):
            result._init_data = other._init_data
            result._init_data_sig = other._init_data_sig
            result.init_events_set = other.init_events_set
            if result._particles != other._particles:
                warnings.warn("Concatenated data has a different set of watched particles, re-running set_init_evts() is recommended.")
            result._particle_tags = other._particle_tags
        if return_copy:
            return result
        else:
            return

    def __add__(self, other):
        return self.concatenate(other, return_copy=True)

def apply_filters(obj, filters):#: Master.Data, filters):
    for f in filters:
        obj.cutTable.add_mask(f)