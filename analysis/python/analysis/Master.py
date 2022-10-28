"""
Created on: 04/02/2022 12:26

Author: Shyam Bhuller

Description: Module containing core components of analysis code. 
"""
import itertools
import time
import warnings
from abc import ABC, abstractmethod

import awkward as ak
import numpy as np
import uproot

# custom modules
from python.analysis import vector


def timer(func):
    """ Decorator which times a function.

    Args:
        func (function): function to time
    """
    def wrapper_function(*args, **kwargs) -> object:
        """ Times funcions, returns outputs
        Returns:
            any: func output
        """
        s = time.time()
        out = func(*args,  **kwargs)
        print(f'{func.__name__!r} executed in {(time.time()-s):.4f}s')
        return out
    return wrapper_function


def __GenericFilter__(data, filters : list):
    """ Applies boolean masks (filters) to data.

    Args:
        data (any): data class which has instance variables with compatible shapes to the filters
        filters (list): list of filters
    """
    for f in filters:
        for var in vars(data):
            if hasattr(getattr(data, var), "__getitem__"):
                try:
                    setattr(data, var, getattr(data, var)[f])
                except:
                    warnings.warn(f"Couldn't apply filters to {var}.")


class IO:
    """ Interface to ROOT file using uproot
    Attributes:
        filename (str): ROOT filename
        nEvents (int): number of events to read. Defaults to -1 (all events)
        start (int): starting point to read from. Defaults to None.
    Methods:
        Get (ak.Array): Load nTuple from root file as awkward array.
        ListNTuples : list all NTuple entries produced by the analyser.
    """
    def __init__(self, _filename : str, _nEvents : int = -1, _start : int = None):
        self.filename = _filename
        self.start = _start
        self.nEvents = _nEvents


    def Get(self, item : str) -> ak.Array:
        """ Load nTuple from root file as awkward array.

        Args:
            item (str): nTuple name in root file

        Returns:
            ak.Array: nTuple loaded
        """
        with uproot.open(self.filename) as file:
            try:
                if self.nEvents > 0:
                    for batch in file["pduneana/beamana"].iterate(entry_start=self.start, entry_stop=self.start+self.nEvents, step_size=self.nEvents, filter_name=item):
                        return batch[item]
                else:
                    return file["pduneana/beamana"][item].array()
            except uproot.KeyInFileError:
                print(f"{item} not found in file, moving on...")
                return None


    def ListNTuples(self, search : str = ""):
        """ list all NTuple entries produced by the analyser.

        Args:
            search (str, optional): filter NTuple by name (not case sensitive). Defaults to "".
        """
        with uproot.open(self.filename) as file:
            names = file["pduneana/beamana"].keys()
            print([x for x in names if search.casefold() in x.casefold()])


class Data:
    """ Class to hold all event information read from the NTuple.
    
    Attributes:
        io (IO): IO class which is used for interfacing with the NTuple file (using uproot).
        includeBacktrackedMC (bool): Should backtracked MC be loaded. Defaults to False. #! to be removed!
        filename (str): NTuple file name. Defaults to None. #? does this need to be an attribute of the class if already defined in IO?
        nEvents (int): number of events to study. Defaults to -1 (all events). #? does this need to be an attribute of the class if already defined in IO?
        start (int): start point in which to read events from the root file. Defaults to 0. #? does this need to be an attribute of the class if already defined in IO?
        run (ak.Array): run number of each event.
        subRun (ak.Array): sub run number of each event.
        eventNum (ak.Array): event number of each event.
        trueParticles (TrueParticles): class to hold truth information.
        recoParticles (RecoParticles): class to hold reco information.
        trueParticlesBT (TrueParticlesBT): class to hold backtracked truth information.

    Property Methods:    
        SortedTrueEnergyMask (ak.Array): index of shower pairs sorted by true energy for pure pi0 sample.

    Methods:
        #? should the objects which apply to specific samples be moved? 
        FindEvents (dict): Returns the run, subrun and event number for events which pass a particlular filter.
        MCMatching (object): truth-reco matching algorithm for pure pi0 sample.
        MatchByAngleBT (tuple): shower matching algorithm using bracktracked info for pure pi0 sample.
        CreateSpecificEventFilter (ak.Array): boolean mask which selects events based on the event number/run number.
        Filter (object): Filter events or entries per event.
        ApplyBeamFilter: select events which have a beam particle ID'd. #? should this return a boolean mask rather than filtering?
        MergePFPCheat (tuple): merge PFOs which backtrack to the same particle.
        MergeShower (Data): merge PFOs using reco information for pure pi0 sample.
        MergeShowerBT (Data): Merge showers using bactacked matching for pure pi0 sample.
    """
    def __init__(self, filename : str = None, includeBackTrackedMC : bool = True, nEvents : int = -1, start : int = 0):
        self.filename = filename
        if self.filename != None:
            self.nEvents = nEvents
            self.start = start
            self.io = IO(self.filename, self.nEvents, self.start)
            self.run = self.io.Get("Run")            
            self.subRun = self.io.Get("SubRun")
            self.eventNum = self.io.Get("EventID")
            self.trueParticles = TrueParticleData(self)
            self.recoParticles = RecoParticleData(self)
            if includeBackTrackedMC is True:
                self.trueParticlesBT = TrueParticleDataBT(self)

    @property
    def SortedTrueEnergyMask(self) -> ak.Array:
        """ Returns index of shower pairs sorted by true energy (highest first).
        Returns:
            ak.Array: sorted indices of true photons
        """
        return ak.argsort(self.trueParticles.energy[self.trueParticles.truePhotonMask], ascending=True)

                
    def FindEvents(self, event_filter : ak.Array = None) -> dict:
        """ Returns the run, subrun and event number for events which pass a particlular filter.

        Args:
            event_filter (ak.Array): mask which selects events (1D array)

        Returns:
            dict: event, subrun and run.
        """
        if ak.ravel(event_filter).type.type != ak.types.PrimitiveType("bool"):
            raise TypeError(f'event_filter must have primitive type "bool", but has primitive type {event_filter.type.type}')
        if ak.count(event_filter) != ak.count(self.eventNum):
            raise Exception(f"event_fitler doesn't have an array length equal to the number of events")
        return {"event" : self.eventNum[event_filter], "subrun" : self.subRun[event_filter], "run" : self.run[event_filter]}

    @timer
    def MCMatching(self, cut=0.25, applyFilters : bool = True, returnCopy : bool = False):
        """ Function which matched reco showers to true photons for pi0 decays.
            Does so by looking at angular separation, it will create masks which
            can be applied to self to select matched showers, or return them. ALso
            calculates mask of unmatched showers, needed for shower merging.

        Args:
            cut (float, optional): cut on angle for event selection. Defaults to 0.25.
            applyFilters (bool, optional): if true applies matching filters to self, false returns the filters for the user to use. Defaults to True.
            returnCopy (bool, optional): if true returns a filtered copy of the class, false applies to self. Defaults to False.

        Returns (optional):
            Data: filtered copy of self
            ak.Array: mask which selects macthed showers
            ak.Array: mask which selects unmacthed showers
            ak.Array: mask which selects events that pass the selection
        """
        # angle of all reco showers wrt to each true photon per event i.e. error
        null_shower_dir = self.recoParticles.direction.x == -999
        photon_dir = vector.normalize(self.trueParticles.momentum)[self.trueParticles.truePhotonMask]
        angle_0 = vector.angle(self.recoParticles.direction, photon_dir[:, 0])
        angle_0 = ak.where(null_shower_dir == True, 1E8, angle_0)
        angle_1 = vector.angle(self.recoParticles.direction, photon_dir[:, 1])
        angle_1 = ak.where(null_shower_dir == True, 1E8, angle_1)
        ind = ak.sort(ak.argsort(angle_0, -1), -1) # create array of indices to keep track of shower index

        # get smallest angle wrt to each true photon
        m_0 = ak.unflatten(ak.min(angle_0, -1), 1)
        m_1 = ak.unflatten(ak.min(angle_1, -1), 1)

        first_matched_photon = ak.argmin(ak.concatenate([m_0, m_1], -1), -1, keepdims=True) # get the first photon with the smallest angle
        remaining_photon = ak.where(first_matched_photon == 0, 1, 0) # get the other photon which needs to be matched
        
        # get the index of the matched shower
        m_0 = ind[m_0 == angle_0]
        m_1 = ind[m_1 == angle_1]
        first_matched_shower = ak.where(first_matched_photon == 0, m_0, m_1) # get index of the matched shower wrt to each photon

        # get angles of remaining showers to look at
        remaining_showers_0 = ak.flatten(angle_0[first_matched_shower]) != angle_0
        remaining_showers_1 = ak.flatten(angle_1[first_matched_shower]) != angle_1
        new_angle_0 = angle_0[remaining_showers_0]
        new_angle_1 = angle_1[remaining_showers_1]

        # get index of next closest shower
        m_0 = ak.unflatten(ak.min(new_angle_0, -1), 1)
        m_1 = ak.unflatten(ak.min(new_angle_1, -1), 1)
        m_0 = ind[m_0 == angle_0]
        m_1 = ind[m_1 == angle_1]
        second_matched_shower = ak.where(remaining_photon == 0, m_0, m_1) # get index of the matched shower wrt to each photon

        # concantenate matched photons in each pass, then get the indices in which they should be sorted to preserve the order of photons in MC truth
        photon_order = ak.argsort(ak.concatenate([first_matched_photon, remaining_photon], -1), -1)
        # concantenate matched showers and then sort by photon number
        matched_mask = ak.concatenate([first_matched_shower, second_matched_shower], -1)[photon_order]

        # get unmatched showers by checking which showers are matched
        t_0 = matched_mask[:, 0] == ind
        t_1 = matched_mask[:, 1] == ind
        # use logical not to get the showers not in matched mask i.e. what is unmatched
        unmatched_mask = np.logical_not(np.logical_or(t_0, t_1))

        # get events where both reco MC angles are less than the threshold value
        angle_0 = vector.angle(self.recoParticles.direction[matched_mask][:, 0], photon_dir[:, 0])
        angle_1 = vector.angle(self.recoParticles.direction[matched_mask][:, 1], photon_dir[:, 1])
        selection = np.logical_and(angle_0 < cut, angle_1 < cut)
        
        # either apply the filters or return them
        if applyFilters is True:
            if returnCopy is True:
                return self.Filter([matched_mask, selection], [selection], returnCopy=True)
            else:
                self.Filter([matched_mask, selection], [selection], returnCopy=False)
        else:
            return matched_mask, unmatched_mask, selection

    @timer
    def MatchByAngleBT(self) -> tuple:
        """ Equivlant to shower matching/angluar closeness cut, but for backtracked MC.

        Args:
            events (Master.Data): events to look at

        Raises:
            Exception: if all reco PFP's backtracked to the same true particle

        Returns:
            ak.Array: indices of reco particles with the smallest angular closeness
            ak.Array: boolean mask of events which pass the angle cut
        """
        null_direction = self.recoParticles.direction.x == -999 # boolean mask of PFP's with undefined direction
        null_momentum = self.recoParticles.momentum.x == -999 # boolean mask of PFP's with undefined momentum
        null_position = self.recoParticles.startPos.x == -999
        null = np.logical_or(null_direction, null_momentum)
        null = np.logical_or(null, null_position)
        angle_error = vector.angle(self.recoParticles.direction, self.trueParticlesBT.direction) # calculate angular closeness
        angle_error = ak.where(null, 999, angle_error) # if direction is undefined, angler error is massive (so not the best match)
        ind = ak.local_index(angle_error, -1) # create index array of angles to use later

        # get unique true particle numbers per event i.e. the photons which the reco PFP's backtrack to
        mcIndex = self.trueParticlesBT.particleNumber
        unqiueIndex = self.trueParticlesBT.GetUniqueParticleNumbers(mcIndex)

        if(ak.any(ak.num(unqiueIndex) == 1)):
            raise Exception("data contains events with reco particles matched to only one photon, did you forget to apply singleMatch filter?")

        # get PFP's which match to the same true particle
        mcp_0 = mcIndex == unqiueIndex[:, 0]
        mcp_1 = mcIndex == unqiueIndex[:, 1]

        # get the smallest angle error of each sorted PFP's
        angle_error_0 = ak.min(angle_error[mcp_0], -1)
        angle_error_1 = ak.min(angle_error[mcp_1], -1)

        selection_bt = np.logical_and(angle_error_0 < 0.25, angle_error_1 < 0.25) # create boolean mask for the event selection

        # get recoPFP indices which had the smallest angular closeness
        indices_0 = ind[angle_error == angle_error_0]
        indices_1 = ind[angle_error == angle_error_1]
        best_match = ak.concatenate([indices_0, indices_1], -1)
        best_match = best_match[:, 0:2]
        return best_match, selection_bt


    def CreateSpecificEventFilter(self, eventNums : list) -> ak.Array:
        """ Create boolean mask which selects specific events and subruns to look at.
            example: if you want to look at subrun 1 events 5, 6, 8 then eventNums is
            [(1, 5), (1, 6), (1, 8)]
            or the first event of subruns 1, 40 and 268:
            [(1, 1), (40, 1), (268, 1)]
        Args:
            eventNums (list): list of tuples of the form (subrun, eventNum)

        Returns:
            ak.Array: boolean mask of selected events
        """
        re = ak.concatenate( [ak.unflatten(self.subRun, 1), ak.unflatten(self.eventNum, 1)], 1)
        f = np.logical_and(re[:, 0] == eventNums[0][0], re[:, 1] == eventNums[0][1])
        for i in range(1, len(eventNums)):
            f = np.logical_or(f, np.logical_and(re[:, 0] == eventNums[i][0], re[:, 1] == eventNums[i][1]))
        return f


    def Filter(self, reco_filters : list = [], true_filters : list = [], returnCopy : bool = False):
        """ Filter events or entries per event.

        Args:
            reco_filters (list, optional): list of filters to apply to reconstructed data. Defaults to [].
            true_filters (list, optional): list of filters to apply to true data. Defaults to [].
            returnCopy   (bool, optional): if true return a copy of filtered events, else change self

        Returns (optional):
            Event: filtered events
        """
        if returnCopy is False:
            self.trueParticles.Filter(true_filters, returnCopy)
            self.recoParticles.Filter(reco_filters, returnCopy)
            if hasattr(self, "trueParticlesBT"):
                self.trueParticlesBT.Filter(reco_filters, returnCopy) # has same shape as reco data

            __GenericFilter__(self, reco_filters)
            self.trueParticles.events = self
            self.recoParticles.events = self
            if hasattr(self, "trueParticlesBT"):
                self.trueParticlesBT.events = self
        else:
            # copy filtered attributes into new instance
            filtered = Data()
            filtered.filename = self.filename
            filtered.nEvents = self.nEvents
            filtered.start = self.start
            filtered.io = IO(filtered.filename, filtered.nEvents, filtered.start)
            filtered.eventNum = self.eventNum
            filtered.subRun = self.subRun
            filtered.run = self.run
            filtered.trueParticles = self.trueParticles.Filter(true_filters)
            filtered.recoParticles = self.recoParticles.Filter(reco_filters)
            filtered.recoParticles.events = filtered
            filtered.trueParticles.events = filtered
            if hasattr(self, "trueParticlesBT"):
                filtered.trueParticlesBT = self.trueParticlesBT.Filter(reco_filters)
                filtered.trueParticlesBT.events = filtered
            __GenericFilter__(filtered, reco_filters) #? should true_filters also be applied?
            return filtered

    @timer
    def ApplyBeamFilter(self):
        """ Applies a beam filter to the sample, which selects objects
            in events which are the beam particle, or daughters of the beam.
        """
        if self.recoParticles.beam_number is None:
            print("data doesn't contain beam number, can't apply filter.")
            return
        hasBeam = self.recoParticles.beam_number != -999 # check if event has a beam particle
        hasBeam = np.logical_and(self.recoParticles.beamVertex.x != -999, hasBeam)
        self.Filter([hasBeam], [hasBeam]) # filter data
    
    @timer
    def MergePFPCheat(self) -> tuple:
        """ Merges all PFPs which backtrack to the same True particle.

        Returns:
            Tuple: Data class with merged PFOs and a list of PFOs which have underfined momentum after merging 
        """
        mcIndex = self.trueParticlesBT.number
        if mcIndex is None:
            mcIndex = self.trueParticlesBT.particleNumber # required for data missing backtracked number (pure pi0 sample)

        unqiueIndex = self.trueParticlesBT.GetUniqueParticleNumbers(mcIndex) # get unique list of true particles

        maxPFPs = ak.max(ak.num(unqiueIndex))
        unqiueIndex = ak.pad_none(unqiueIndex, maxPFPs)
        for i in range(maxPFPs):
            pfps = mcIndex == unqiueIndex[:, i] # sort pfps based on which true particle they back-track to
            
            # get index of backtracked particle
            ind = ak.local_index(pfps, -1)
            ind = ak.min(ind[pfps], -1)

            # sum momenta of PFPs to merge
            p = self.recoParticles.momentum[pfps]
            p = ak.where(p.x == -999, {"x": 0,"y": 0,"z": 0}, p) # dont merge PFP's with null data
            p_m = ak.sum(p, -1) # sum all momentum vectors
            e_m = vector.magnitude(p_m)
            dir_m = vector.normalize(p_m)

            if i == 0:
                truePFPMask = ak.unflatten(ind, 1, -1)
                p_new = ak.unflatten(p_m, 1, -1)
                e_new = ak.unflatten(e_m, 1, -1)
                dir_new = ak.unflatten(dir_m, 1, -1)
            else:
                truePFPMask = ak.concatenate([truePFPMask, ak.unflatten(ind, 1, -1)], axis=-1)
                p_new = ak.concatenate([p_new, ak.unflatten(p_m, 1, -1)], axis=-1)
                e_new = ak.concatenate([e_new, ak.unflatten(e_m, 1, -1)], axis=-1)
                dir_new = ak.concatenate([dir_new, ak.unflatten(dir_m, 1, -1)], axis=-1)        

        truePFPMask = ak.fill_none(truePFPMask, -999)
        truePFPMask = truePFPMask[truePFPMask != -999]

        events_merge = self.Filter(returnCopy=True) # easy way to make copies of the class
        events_merge.recoParticles._RecoParticleData__momentum = p_new
        events_merge.recoParticles._RecoParticleData__energy = e_new
        events_merge.recoParticles._RecoParticleData__direction = dir_new
        
        events_merge.trueParticlesBT.Filter([truePFPMask], False) # filter to get the true particles the merged PFP's relate to
        null = np.logical_not(ak.any(events_merge.recoParticles.energy == 0, -1)) # exlucde events where all PFP's merged had no valid momentmum vector
        print(f"Events where one merged PFP had undefined momentum: {ak.count(null[np.logical_not(null)])}")
        return events_merge, null


    def MergeShower(self, matched : ak.Array, unmatched : ak.Array):
        """ Merge shower not matched to MC to the spatially closest matched shower.

        Args:
            events (Master.Event): events to study
            matched (ak.Array): matched shower indicies
            unmatched (ak.Array): boolean mask of unmatched showers
            mergeMethod (int): method 1 merges by closest angular distance, method 2 merges by closest spatial distance
            energyScalarSum (bool): False does a sum of momenta, then magnitude, True does magnitude of momenta, then sum

        Returns:
            Data: events with matched reco showers after merging
        """

        events_matched = self.Filter([matched], returnCopy=True)
        unmatched_reco = self.Filter([unmatched], returnCopy=True).recoParticles # filter reco for matched/unmatched only
        null_dir = unmatched_reco.direction.x == -999 # should only be needed for unmatched sample

        angle_0 = ak.where(null_dir == True, 1E8, vector.angle(unmatched_reco.direction, events_matched.recoParticles.direction[:, 0]))
        angle_1 = ak.where(null_dir == True, 1E8, vector.angle(unmatched_reco.direction, events_matched.recoParticles.direction[:, 1]))
        angle_0 = ak.unflatten(angle_0, 1, -1)
        angle_1 = ak.unflatten(angle_1, 1, -1)
        angle = ak.concatenate([angle_0, angle_1], -1)
        mergeMask = ak.min(angle, -1) == angle

        #* create Array which contains the amount of energy to merge to the showers
        #* will be zero for the shower we don't want to merge to
        null = {"x": 0, "y": 0, "z": 0}
        toMerge = ak.where(mergeMask, unmatched_reco.momentum, null)
        toMerge = ak.where(toMerge.x != -999, toMerge, null)
        merge_0 = ak.unflatten(ak.sum(toMerge[:, :, 0], -1), 1, -1)
        merge_1 = ak.unflatten(ak.sum(toMerge[:, :, 1], -1), 1, -1)
        momentumToMerge = ak.concatenate([merge_0, merge_1], -1)
        momentumToMerge = ak.where(events_matched.recoParticles.momentum.x != -999, momentumToMerge, null)
        new_momentum = vector.add(events_matched.recoParticles.momentum, momentumToMerge)
        events_matched.recoParticles._RecoParticleData__momentum = new_momentum

        new_direction = vector.normalize(events_matched.recoParticles.momentum)
        new_direction = ak.where(events_matched.recoParticles.momentum.x != -999, new_direction, {"x": -999, "y": -999, "z": -999})
        events_matched.recoParticles._RecoParticleData__direction = new_direction

        new_energy = vector.magnitude(events_matched.recoParticles.momentum)
        events_matched.recoParticles._RecoParticleData__energy = ak.where(events_matched.recoParticles.momentum.x != -999, new_energy, -999)

        return events_matched


    def MergeShowerBT(self, best_match : ak.Array):
        """ Merge showers using bactacked matching.

        Args:
            best_match (ak.Array): mask of starting showers

        Returns:
            Data: events with merged showers
        """
        if bool(ak.all(ak.num(self.recoParticles.energy[self.recoParticles.energy != -999]) == 2)) is True:
            counts = 0
        else:
            counts = 1
        #* get boolean mask of PFP's to merge
        index = ak.local_index(self.recoParticles.energy)
        to_merge = [ ak.where(index == best_match[:, i], False, True) for i in range(2) ]
        to_merge = np.logical_and(*to_merge)

        #* calculate angle between each best matched shower and the showers to merge
        best_match_dir = self.recoParticles.direction[best_match]
        to_merge_dir = self.recoParticles.direction[to_merge]
        not_null = to_merge_dir.x != -999
        to_merge_dir = to_merge_dir[not_null]

        angles = [ak.unflatten(vector.angle(to_merge_dir, best_match_dir[:, i]), counts, -1) for i in range(2)]
        angles = ak.concatenate(angles, -1)
        # check which angle is the smallest i.e. decide which main shower to merge to
        minMask_angle = ak.min(angles, -1) == angles

        #* merge momenta
        momentumToMerge = self.recoParticles.momentum[to_merge]
        momentumToMerge = momentumToMerge[not_null]
        valid_mom = momentumToMerge.x != -999 # we don't want to merge invalid PFP's
        momentumToMerge = momentumToMerge[valid_mom]
        minMask_angle = minMask_angle[valid_mom]
        minMask_angle = [minMask_angle[:, :, i] for i in range(2)]

        momentumToMerge = [ak.sum(momentumToMerge[minMask_angle[i]], -1) for i in range(2)]
        leading_momentum = self.recoParticles.momentum[best_match]

        # add the merged momenta to their respective leading showers
        mergedMomentum = [ak.unflatten(vector.add(leading_momentum[:, i], momentumToMerge[i]), 1, -1) for i in range(2)]
        mergedMomentum = ak.concatenate(mergedMomentum, -1)

        mergedDirection = [ak.unflatten(vector.normalize(mergedMomentum[:, i]), 1, -1) for i in range(2)]
        merged_direction = ak.concatenate(mergedDirection, -1)

        merged_energy = [ak.unflatten(vector.magnitude(mergedMomentum[:, i]), 1, -1) for i in range(2)]
        merged_energy = ak.concatenate(merged_energy, -1)

        merged_events = self.Filter(returnCopy=True)
        merged_events.recoParticles._RecoParticleData__momentum = mergedMomentum
        merged_events.recoParticles._RecoParticleData__direction = merged_direction
        merged_events.recoParticles._RecoParticleData__energy = merged_energy

        merged_events.trueParticlesBT.Filter([best_match], returnCopy=False)
        return merged_events


class ParticleData(ABC):
    """ Template for data classes which store particle information in events.
    Attributes:
        events (Data): parent class.
        filters (list): list of filters applied/to apply to data

    Abstrct Methods:
        __init__:
        CalculatePairQuantities: Calculate Shower pair quantities. #! shower pair quantities should be its own data class
    Methods:
        LoadData: Reads data from NTuple to hidden attribute specific to the subclass.
        Filter (object): Filter data.
    """
    @abstractmethod
    def __init__(self, events : Data) -> None:
        self.events = events # keep reference of parent class
        self.filters = [] # list of filters to apply to newly loaded data
        pass

    @abstractmethod
    def CalculatePairQuantities(self):
        pass


    def LoadData(self, name : str, nTupleName):
        """ Reads data from ntuple and assigns it to a hidden variable
            which can then be called by the property method.

        Args:
            name (str): variable name
            nTupleName (str, list): branch/es to read from root file
        """
        var_name = f"_{type(self).__name__}__{name}" # variable name which chose to be hidden
        if hasattr(self.events, "io") and not hasattr(self, var_name):
            if type(nTupleName) is list:
                if len(nTupleName) != 3:
                    raise Exception("nTuple list for vector must be of length 3")
                setattr(self, var_name, ak.zip({
                    "x" : self.events.io.Get(nTupleName[0]),
                    "y" : self.events.io.Get(nTupleName[1]),
                    "z" : self.events.io.Get(nTupleName[2])
                }))
            else:
                setattr(self, var_name, self.events.io.Get(nTupleName))
            # apply any filters to data when loading
            for f in self.filters:
                try:
                    setattr(self, var_name, getattr(self, var_name)[f])
                except:
                    Warning(f"couldn't apply a filter to {var_name}")


    def Filter(self, filters : list, returnCopy : bool = True):
        """ Filter data.

        Args:
            filters (list): list of filters to apply to particle data.

        Returns (optional):
            subClass(ParticleData): filtered data.
        """
        if returnCopy is False:
            __GenericFilter__(self, filters)
            self.filters.extend(filters) # append list of filters to apply to newly loaded data
        else:
            subclass = globals()[type(self).__name__] # get the class which is of type ParticleData
            filtered = subclass(Data()) # create a new instance of the class
            # populate new instance
            for var in vars(self):
                setattr(filtered, var, getattr(self, var))
            __GenericFilter__(filtered, filters)
            filtered.filters = list(self.filters + filters)
            return filtered


class TrueParticleData(ParticleData):
    """ Particle data class for truth information. 

    Attributes:
        (1 hidden attribute for each property method)
    Property Methods:
        pdg (ak.Array): pdg code
        number (ak.Array): particle number
        mother (ak.Array): number of mother particle
        energy (ak.Array):
        momentum (ak.Record):
        direction (ak.Record):
        startPos (ak.Record):
        endPos (ak.Record):
        pi0_MC (bool): check if we are looking at pure pi0 MC or beam MC
        PrimaryPi0Mask (ak.Array): Pi0 produced from beam or generated by particle gun
        truePhotonMask (ak.Array): get mask of photons which are duaghter of the primary pi0

    Methods:
        CalculatePairQuantities (tuple): Calculate true shower pair quantities.
    """
    def __init__(self, events : Data):
        super().__init__(events)

    @property
    def pdg(self) -> ak.Array:
        self.LoadData("pdg", "g4_Pdg")
        return getattr(self, f"_{type(self).__name__}__pdg")

    @property
    def number(self) -> ak.Array:
        self.LoadData("number", "g4_num")
        return getattr(self, f"_{type(self).__name__}__number")

    @property
    def mother(self) -> ak.Array:
        self.LoadData("mother", "g4_mother")
        return getattr(self, f"_{type(self).__name__}__mother")

    @property
    def energy(self) -> ak.Array:
        self.LoadData("energy", "g4_startE")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def momentum(self) -> ak.Record:
        self.LoadData("momentum", ["g4_pX", "g4_pY", "g4_pZ"])
        return getattr(self, f"_{type(self).__name__}__momentum")
    
    @property
    def direction(self) -> ak.Record:
        if not hasattr(self, f"_{type(self).__name__}__direction"):
            self.__direction = vector.normalize(self.momentum)
        return self.__direction

    @property
    def startPos(self) -> ak.Record:
        self.LoadData("startPos", ["g4_startX", "g4_startY", "g4_startZ"])
        return getattr(self, f"_{type(self).__name__}__startPos")

    @property
    def endPos(self) -> ak.Record:
        self.LoadData("endPos", ["g4_endX", "g4_endY", "g4_endZ"])
        return getattr(self, f"_{type(self).__name__}__endPos")

    @property
    def pi0_MC(self) -> bool:
        if not hasattr(self, f"_{type(self).__name__}__pi0_MC"):
            self.__pi0_MC = ak.any(np.logical_and(self.number == 1, self.pdg == 111)) # check if we are looking at pure pi0 MC or beam MC
        return self.__pi0_MC

    @property
    def PrimaryPi0Mask(self) -> ak.Array:
        if self.pi0_MC:
            return np.logical_and(self.number == 1, self.pdg == 111)
        else:
            # assume we have beam MC
            # get pi0s produced from beam interaction
            return np.logical_and(self.mother == 1, self.pdg == 111)

    @property
    def truePhotonMask(self) -> ak.Array:
        if self.pi0_MC:
            photons = self.mother == 1 # get only primary daughters
            photons = np.logical_and(photons, self.pdg == 22)
        else:
            # assume we have beam MC, with potentially more than 1 pi0
            primary_pi0 = self.PrimaryPi0Mask
            primary_pi0_num = self.number[primary_pi0] # get particle number of each pi0
            n = ak.max(ak.num(primary_pi0_num)) # get the largest number of primary pi0s per event
            primary_pi0_num = ak.pad_none(primary_pi0_num, n) # pad empty elements with None so we can do index slicing
            
            #* loop through slices of pi0s per event, and get all particles which are their daughters
            null = self.number == -1 # never should be negative so all values are false
            primary_daughters = self.mother == primary_pi0_num[:, 0]
            primary_daughters = ak.where(ak.is_none(primary_daughters), null, primary_daughters) # check none since boolean logic ignores None values
            
            i = 1
            while i < n:
                next_primaries = self.mother == primary_pi0_num[:, i]
                next_primaries = ak.where(ak.is_none(next_primaries), null, next_primaries)
                primary_daughters = np.logical_or(primary_daughters, next_primaries)
                i += 1
            photons = np.logical_and(primary_daughters, self.pdg == 22) # only want photons
        return photons


    def CalculatePairQuantities(self) -> tuple:
        """ Calculate true shower pair quantities.

        Args:
            events (Master.Event): events to process

        Returns:
            tuple of ak.Array: calculated quantities
        """
        mask_pi0 = self.PrimaryPi0Mask
        if ak.all(ak.num(mask_pi0[mask_pi0]) <= 1) == False:
            raise ValueError("function currently only works for samples with 1 pi0 per event.")
        photons = self.truePhotonMask
        sortEnergy = self.events.SortedTrueEnergyMask
        
        #* compute start momentum of dauhters
        p_daughter = self.momentum[photons]
        sum_p = ak.sum(p_daughter, axis=1)
        sum_p = vector.magnitude(sum_p)
        p_daughter_mag = vector.magnitude(p_daughter)
        p_daughter_mag = p_daughter_mag[sortEnergy]

        #* compute true opening angle
        angle = np.arccos(vector.dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

        #* compute invariant mass
        e_daughter = self.energy[photons]
        inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

        #* pi0 momentum
        p_pi0 = self.momentum[mask_pi0]
        p_pi0 = vector.magnitude(p_pi0)
        return inv_mass, angle, p_daughter_mag[:, 1:], p_daughter_mag[:, :-1], p_pi0


class RecoParticleData(ParticleData):
    """ Particle data class for reconstructed information.

    Attributes:
        (1 hidden attribute for each property method)
    
    Property Methods:
        beam_number (ak.Array): beam PFO number
        sliceID (ak.Array): slice the PFO corresponds to
        beamCosmicScore (ak.Array): whether the reconstruction chain used was for a neutrino vertex (beam) or cosmics
        pandoraTag (ak.Array): label given to particles by pandora; track, shower or -999
        number (ak.Array): PFO number
        mother (ak.Array): number of mother PFO
        nHits (ak.Array): number of collection plane hits
        energy (ak.Array):
        momentum (ak.Record):
        direction (ak.Record):
        startPos (ak.Record):
        showerLength (ak.Array): length of shower (if applicable)
        showerConeAngle (ak.Array): width of shower (if applicable)
        cnnScore (ak.Array): shower-track like score.
        beamVertex (ak.Record): end point of the beam particle     

        
    Methods:
        CalculatePairQuantities (tuple): Calculate reconstructed shower pair quantities.
        GetPairValues (ak.Array): Get shower pair values, in pairs
        AllShowerPairs (list): Get all unique shower pair combinations.
        ShowerPairsByHits (ak.list): get shower pairs by leading in energy #! not used
    """

    def __init__(self, events : Data) -> None:
        super().__init__(events)

    @property
    def beam_number(self) -> ak.Array:
        self.LoadData("beam_number", "beamNum")
        return getattr(self, f"_{type(self).__name__}__beam_number")

    @property
    def sliceID(self) -> ak.Array:
        self.LoadData("sliceID", "reco_daughter_allShower_sliceID")
        return getattr(self, f"_{type(self).__name__}__sliceID")

    @property
    def beamCosmicScore(self) -> ak.Array:
        self.LoadData("beamCosmicScore", "reco_daughter_allShower_beamCosmicScore")
        return getattr(self, f"_{type(self).__name__}__beamCosmicScore")

    @property
    def pandoraTag(self) -> ak.Array:
        self.LoadData("pandoraTag", "pandoraTag")
        return getattr(self, f"_{type(self).__name__}__pandoraTag")

    @property
    def number(self) -> ak.Array:
        self.LoadData("number", "reco_PFP_ID")
        return getattr(self, f"_{type(self).__name__}__number")

    @property
    def mother(self) -> ak.Array:
        self.LoadData("mother", "reco_PFP_Mother")
        return getattr(self, f"_{type(self).__name__}__mother")
    
    @property
    def nHits(self) -> ak.Array:
        self.LoadData("nHits", "reco_daughter_PFP_nHits_collection")
        return getattr(self, f"_{type(self).__name__}__nHits")

    @property
    def startPos(self) -> ak.Record:
        nTuples = [
            "reco_daughter_allShower_startX",
            "reco_daughter_allShower_startY",
            "reco_daughter_allShower_startZ"
        ]
        self.LoadData("startPos", nTuples)
        return getattr(self, f"_{type(self).__name__}__startPos")

    @property
    def direction(self) -> ak.Record:
        nTuples = [
            "reco_daughter_allShower_dirX",
            "reco_daughter_allShower_dirY",
            "reco_daughter_allShower_dirZ"
        ]
        self.LoadData("direction", nTuples)
        return getattr(self, f"_{type(self).__name__}__direction")

    @property
    def energy(self) -> ak.Array:
        self.LoadData("energy", "reco_daughter_allShower_energy")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def momentum(self) -> ak.Record:
        if not hasattr(self, f"_{type(self).__name__}__momentum"):
            mom = vector.prod(self.energy, self.direction)
            mom = ak.where(self.direction.x == -999, {"x": -999,"y": -999,"z": -999}, mom)
            mom = ak.where(self.energy < 0, {"x": -999,"y": -999,"z": -999}, mom)
            self.__momentum = mom
        return self.__momentum

    @property
    def showerLength(self) -> ak.Array:
        self.LoadData("showerLength", "reco_daughter_allShower_length")
        return getattr(self, f"_{type(self).__name__}__showerLength")

    @property
    def showerConeAngle(self) -> ak.Array:
        self.LoadData("coneAngle", "reco_daughter_allShower_coneAngle")
        return getattr(self, f"_{type(self).__name__}__coneAngle")

    @property
    def cnnScore(self) -> ak.Array:
        self.LoadData("cnnScore", "CNNScore_collection")
        return getattr(self, f"_{type(self).__name__}__cnnScore")

    @property
    def beamVertex(self) -> ak.Record:
        nTuples = [
            "reco_beam_endX",
            "reco_beam_endY",
            "reco_beam_endZ",
        ]
        self.LoadData("beamVertex", nTuples)
        return getattr(self, f"_{type(self).__name__}__beamVertex")


    def CalculatePairQuantities(self, useBT : bool = False) -> tuple:
        """ Calculate reconstructed shower pair quantities.

        Returns:
            tuple of ak.Array: calculated quantities + array which masks null shower pairs
        """
        if useBT is True:
            sortEnergy = ak.argsort(self.events.trueParticlesBT.energy)
        else:
            sortEnergy = self.events.SortedTrueEnergyMask
        sortedPairs = ak.unflatten(self.energy[sortEnergy], 1, 0)
        leading = sortedPairs[:, :, 1:]
        secondary = sortedPairs[:, :, :-1]

        #* opening angle
        direction_pair = ak.unflatten(self.direction[sortEnergy], 1, 0)
        direction_pair_mag = vector.magnitude(direction_pair)
        angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

        #* Invariant Mass
        inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

        #* pi0 momentum
        pi0_momentum = vector.magnitude(ak.sum(self.momentum, axis=-1))/1000

        null_dir = np.logical_or(direction_pair[:, :, 1:].x == -999, direction_pair[:, :, :-1].x == -999) # mask shower pairs with invalid direction vectors
        null = np.logical_or(leading < 0, secondary < 0) # mask of shower pairs with invalid energy
        
        #* filter null data
        pi0_momentum = np.where(null_dir, -999, pi0_momentum)
        pi0_momentum = np.where(null, -999, pi0_momentum)

        leading = leading/1000
        secondary = secondary/1000

        leading = np.where(null, -999, leading)
        leading = np.where(null_dir, -999, leading)
        secondary = np.where(null, -999, secondary)
        secondary = np.where(null_dir, -999, secondary)

        angle = np.where(null, -999, angle)
        angle = np.where(null_dir, -999, angle)

        inv_mass = inv_mass/1000
        inv_mass = np.where(null, -999, inv_mass)
        inv_mass = np.where(null_dir, -999, inv_mass)

        return inv_mass, angle, leading, secondary, pi0_momentum

    @timer
    def GetPairValues(pairs, value) -> ak.Array:
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
                    evt.append( [value[i][pair[j][0]], value[i][pair[j][1]]] )
                else:
                    evt.append([])
            paired.append(evt)
        return ak.Array(paired)

    @timer
    def AllShowerPairs(nd) -> list:
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
    def ShowerPairsByHits(hits : ak.Array) -> list:
        """ Pair reconstructed showers in an event by the number of hits.
            Pairs the two largest showers per event.
            TODO figure out a way to do this without sorting events (or re-sort events?)
            #! not used

        Args:
            hits (ak.Array): number of collection plane hits of daughters per event

        Returns:
            list: shower pairs (maximum of one per event), note lists are easier to iterate through than np or ak arrays, hence the conversion
        """
        showers = ak.argsort(hits, ascending=False) # shower number sorted by nHits
        mask = ak.count(showers, 1) > 1
        showers = ak.pad_none(showers, 2, clip=True) # only keep two largest showers
        showers = ak.where( mask, showers, [[]]*len(mask) )
        pairs = ak.unflatten(showers, 1)
        return ak.to_list(pairs)


class TrueParticleDataBT(ParticleData):
    """ Particle data class for backtracked truth information.

    Attributes:
        (1 hidden attribute for each property method)
    Property Methods:
        number (ak.Array): backtracked particle number
        mother (ak.Array): backtracked particle number of mother
        pdg (ak.Array): pdg code of backtracked particle
        startPos (ak.Record):
        endPos (ak.Record):
        momentum (ak.Record):
        energy (ak.Array):
        direction (ak.Record):
        matchedHits (ak.Array): true hits matched to reco hits
        hitsInRecoCluster (ak.Array): total hits in a reco cluster
        mcParticleHits (ak.Array): true particle hits
        sharedHits (ak.Array): true hits shared by other particles (i think?)
        particleNumber: placeholder for ROOT files which don't have the particle number stored as NTuples
        SingleMatch (ak.Array): mask of events with only 1 unique backtracked particle

    Methods:
        GetUniqueParticleNumbers (ak.Array): Get unique array of particle numbers
        CalculatePairQuantities (tuple): Calculate backtracked shower pair quantities.
    """
    def __init__(self, events: Data):
        super().__init__(events)

    @property
    def number(self) -> ak.Array:
        self.LoadData("number", "reco_daughter_PFP_true_byHits_ID")
        return getattr(self, f"_{type(self).__name__}__number")

    @property
    def mother(self) -> ak.Array:
        self.LoadData("mother", "reco_daughter_PFP_true_byHits_Mother")
        return getattr(self, f"_{type(self).__name__}__mother")

    @property
    def pdg(self) -> ak.Array:
        self.LoadData("pdg", "reco_daughter_PFP_true_byHits_pdg")
        return getattr(self, f"_{type(self).__name__}__pdg")

    @property
    def startPos(self) -> ak.Record:
        nTuples = [
            "reco_daughter_PFP_true_byHits_startX",
            "reco_daughter_PFP_true_byHits_startY",
            "reco_daughter_PFP_true_byHits_startZ"
        ]
        self.LoadData("startPos", nTuples)
        return getattr(self, f"_{type(self).__name__}__startPos")

    @property
    def endPos(self) -> ak.Record:
        nTuples = [
            "reco_daughter_PFP_true_byHits_endX",
            "reco_daughter_PFP_true_byHits_endY",
            "reco_daughter_PFP_true_byHits_endZ"
        ]
        self.LoadData("endPos", nTuples)
        return getattr(self, f"_{type(self).__name__}__endPos")

    @property
    def momentum(self) -> ak.Record:
        nTuples = [
            "reco_daughter_PFP_true_byHits_pX",
            "reco_daughter_PFP_true_byHits_pY",
            "reco_daughter_PFP_true_byHits_pZ"
        ]
        self.LoadData("momentum", nTuples)
        return getattr(self, f"_{type(self).__name__}__momentum")

    @property
    def direction(self) -> ak.Record:
        if not hasattr(self, f"_{type(self).__name__}__direction"):
            self.__direction = vector.normalize(self.momentum)
        return self.__direction

    @property
    def energy(self) -> ak.Array:
        self.LoadData("energy", "reco_daughter_PFP_true_byHits_startE")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def matchedHits(self) -> ak.Array:
        self.LoadData("matchedHits", "reco_daughter_PFP_true_byHits_matchedHits")
        return getattr(self, f"_{type(self).__name__}__matchedHits")

    @property
    def hitsInRecoCluster(self) -> ak.Array:
        self.LoadData("hitsInRecoCluster", "reco_daughter_PFP_true_byHits_hitsInRecoCluster")
        return getattr(self, f"_{type(self).__name__}__hitsInRecoCluster")

    @property
    def mcParticleHits(self) -> ak.Array:
        self.LoadData("mcParticleHits", "reco_daughter_PFP_true_byHits_mcParticleHits")
        return getattr(self, f"_{type(self).__name__}__mcParticleHits")

    @property
    def sharedHits(self) -> ak.Array:
        self.LoadData("sharedHits", "reco_daughter_PFP_true_byHits_sharedHits")
        return getattr(self, f"_{type(self).__name__}__sharedHits")

    @property
    def trueBeamVertex(self) -> ak.Array:
        nTuples = [
            "reco_beam_PFP_true_byHits_endX"
            "reco_beam_PFP_true_byHits_endY"
            "reco_beam_PFP_true_byHits_endZ"
        ]
        self.LoadData("trueBeamVertex", nTuples)
        return getattr(self, f"_{type(self).__name__}__trueBeamVertex")

    @property
    def particleNumber(self) -> ak.Array:
        """ Gets the true particle number of each true particle backtracked to reco

        Args:
            events (Master.Data): events to look at

        Returns:
            ak.Array: awkward array of true particle indices
        """
        photonEnergies = self.events.trueParticles.energy[self.events.trueParticles.truePhotonMask] # assign true particle number by comparing energies
        photonNum = self.events.trueParticles.number[self.events.trueParticles.truePhotonMask]

        photonEnergies = ak.pad_none(photonEnergies, 2) # pad these arrays since occasionaly we can get more than two particles from the photon mask 
        photonNum = ak.pad_none(photonNum, 2)

        # loop through all true photons and check the energy matches
        index = None
        for i in range(2):
            matched = self.energy == photonEnergies[:, i] # mask which checked the backtracked MC is the same as the ith photon slice 
            if i == 1:
                index = ak.where(index == -1, ak.where(matched, photonNum[:, i], -1), index) # where -1, see below, else leave index unchanged
            else:
                index = ak.where(matched, photonNum[:, i], -1) # where true add slice, elseset indices to -1
        return index

    @property
    def SingleMatch(self) -> ak.Array:
        """ Get a boolean mask of events with more than one tagged true particle
            in the back tracker.

        Returns:
            ak.Array: boolean mask
        """
        unqiueIndex = self.GetUniqueParticleNumbers(self.particleNumber)
        singleMatch = ak.num(unqiueIndex) < 2
        return np.logical_not(singleMatch)


    def GetUniqueParticleNumbers(self, index : ak.Array) -> ak.Array:
        """ Gets the unique true particle numbers backtracked to each reco particle.
            e.g. if indices are [1, 2, 1, 3, 3, 3], unique is [1, 2, 3] 

        Args:
            index (ak.Array): particle numbers of the true particles which were backtracked.

        Returns:
            ak.Array: unique particle numbers
        """
        maxNPFP = ak.max(ak.num(index)) # get the larget number of PFParticles in an event
        index_padded = ak.pad_none(index, maxNPFP) # pad index array so all nested arrays are equal (can convert to numpy array)
        index_padded = ak.fill_none(index_padded, -1) # fill None entries in nested arrays
        index_padded = ak.fill_none(index_padded, [-1]*maxNPFP, 0) # fill events with no counts with [-1]*nPFP

        index_padded_np = ak.to_numpy(index_padded)
        uniqueIndex = [np.unique(index_padded_np[i, :]) for i in range(len(index_padded_np))] # get the unique entries per event
        uniqueIndex = ak.Array(uniqueIndex)

        # remove -1 entries as these just represent padded entries
        return uniqueIndex[uniqueIndex != -1]


    def CalculatePairQuantities(self) -> tuple:
        """ Calculate reconstructed shower pair quantities.

        Returns:
            tuple of ak.Array: calculated quantities + array which masks null shower pairs
        """
        sortEnergy = ak.argsort(self.energy)
        sortedPairs = ak.unflatten(self.energy[sortEnergy], 1, 0)
        leading = sortedPairs[:, :, 1:]
        secondary = sortedPairs[:, :, :-1]

        #* opening angle
        direction_pair = ak.unflatten(self.direction[sortEnergy], 1, 0)
        direction_pair_mag = vector.magnitude(direction_pair)
        angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

        #* Invariant Mass
        inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

        #* pi0 momentum
        pi0_momentum = vector.magnitude(ak.sum(self.momentum, axis=-1))

        null_dir = np.logical_or(direction_pair[:, :, 1:].x == -999, direction_pair[:, :, :-1].x == -999) # mask shower pairs with invalid direction vectors
        null = np.logical_or(leading < 0, secondary < 0) # mask of shower pairs with invalid energy
        
        #* filter null data
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


def NPFPMask(events : Data, nObjects : int = None) -> ak.Array:
    """ Create a boolean mask of events with a specific number of objects.

    Args:
        events (Data): events to filter
        nObjects (int, optional): keep events with specific number of daughters. Defaults to None.

    Returns:
        ak.Array: mask of events to filter
    """
    null_dir = events.recoParticles.direction.x != -999
    null_pos = events.recoParticles.startPos.x != -999
    nObj = np.logical_and(null_dir, null_pos)
    nObj = ak.num(nObj[nObj]) # get number of showers which have a valid direction

    if nObjects == None:
        r_mask = nObj > 1
    elif nObjects < 0:
        r_mask = nObj > abs(nObjects)
    else:
        r_mask = nObj == nObjects
    return r_mask


def Pi0TwoBodyDecayMask(events : Data) -> ak.Array:
    """ Create boolean mask of events where pi0's have decayed in their primary decay mode
        i.e. pi0 -> gamma gamma

    Args:
        events (Data): events to filter

    Returns:
        ak.Array: mask of events to filter
    """
    mask = ak.num(events.trueParticles.truePhotonMask[events.trueParticles.truePhotonMask], -1) == 2 # exclude pi0 -> e+ + e- + photons
    print(f"number of dalitz decays: {ak.count(mask[np.logical_not(mask)])}")
    return mask

@timer
def Pi0MCMask(events : Data, nObjects : int = None) -> ak.Array:
    """ A filter for Pi0 MC dataset, selects events with a specific number of objects
        which have a valid direction vector and removes events with the 3 body pi0 decay.

    Args:
        events (Event): events being studied
        nObjects (int): keep events with specific number of daughters. Defaults to None

    Returns:
        ak.Array: mask of events to filter
    """
    r_mask = NPFPMask(events, nObjects)
    t_mask = Pi0TwoBodyDecayMask(events)
    valid = np.logical_and(r_mask, t_mask)
    return valid

@timer
def BeamMCFilter(events : Data, n_pi0 : int = 1, returnCopy=True) -> Data:
    """ Filters BeamMC data to get events with only 1 pi0 which originates from the beam particle interaction.

    Args:
        events (Data): events to filter

    Returns:
        Data: selected events
    """
    #* remove events with no truth info
    empty = ak.num(events.trueParticles.number) > 0
    if returnCopy is True:
        events = events.Filter([empty], [empty], returnCopy=True)
    else:
        events.Filter([empty], [empty])

    #* only look at events with 1 primary pi0
    pi0 = events.trueParticles.PrimaryPi0Mask
    single_primary_pi0 = ak.num(pi0[pi0]) == n_pi0 # only look at events with 1 pi0
    events.Filter([single_primary_pi0], [single_primary_pi0])

    #* remove true particles which aren't primaries
    primary_pi0 = events.trueParticles.PrimaryPi0Mask
    primary_daughter = events.trueParticles.truePhotonMask # this is fine so long as we only care about pi0->gamma gamma
    primaries = np.logical_or(primary_pi0, primary_daughter)
    events.Filter([], [primaries])
    if returnCopy is True: return events


def FractionalError(reco : ak.Array, true : ak.Array, null : ak.Array) -> tuple:
    """Calcuate fractional error, filter null data and format data for plotting.

    Args:
        reco (ak.Array): reconstructed quantity
        true (ak.Array): true quantity
        null (ak.Array): mask for events without shower pairs, reco direction or energy

    Returns:
        tuple of np.array: flattened numpy array of errors, reco and truth
    """
    true = true[null]
    true = ak.where( ak.num(true, 1) > 0, true, np.nan)
    reco = ak.flatten(reco, 1)[null]
    print(f"number of reco pairs: {len(reco)}")
    print(f"number of true pairs: {len(true)}")
    error = (reco / true) - 1
    return ak.to_numpy(ak.ravel(error)), ak.to_numpy(ak.ravel(reco)), ak.to_numpy(ak.ravel(true))

@timer
def CalculateQuantities(events : Data, backtrackedTruth : bool = False) -> tuple:
    """ Calcaulte reco/ true quantities of shower pairs, and format them for plotting

    Args:
        events (Master.Event): events to look at
        names (str): quantity names
        backtrackedTruth (bool): calculate truth quantities from backtracked truth

    Returns:
        tuple of np.arrays: quantities to plot
    """
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    if backtrackedTruth is True:
        mct = events.trueParticlesBT.CalculatePairQuantities()
    else:
        mct = events.trueParticles.CalculatePairQuantities()
    rmc = events.recoParticles.CalculatePairQuantities(useBT=True)

    # keep track of events with no shower pairs
    null = ak.flatten(rmc[-1], -1)
    null = ak.num(null, 1) > 0

    error = []
    reco = []
    true = []
    for i in range(len(names)):
        print(names[i])
        e, r, t = FractionalError(rmc[i], mct[i], null)
        error.append(e)
        reco.append(r)
        true.append(t)

    error = np.nan_to_num(error, nan=-999)
    reco = np.nan_to_num(reco, nan=-999)
    true = np.nan_to_num(true, nan=-999)
    return true, reco, error


def ShowerMergePerformance(events : Data, best_match : ak.Array) -> float:
    """ Checks the fraction of correctly merged showers. 
        Does so by checking if the showers to merge have the 
        same true particle number as the target shower.

    Args:
        events (Data): events to study
        best_match (ak.Array): target PFP's to merge to.
    """
    if bool(ak.all(ak.num(events.recoParticles.energy[events.recoParticles.energy != -999]) == 2)) is True:
        counts = 0
    else:
        counts = 1
    index = ak.local_index(events.recoParticles.energy)
    to_merge = [ ak.where(index == best_match[:, i], False, True) for i in range(2) ]
    to_merge = np.logical_and(*to_merge)

    #* calculate angle between each best matched shower and the showers to merge
    best_match_dir = events.recoParticles.direction[best_match]
    to_merge_dir = events.recoParticles.direction[to_merge]
    not_null = to_merge_dir.x != -999
    to_merge_dir = to_merge_dir[not_null]

    angles = [ak.unflatten(vector.angle(to_merge_dir, best_match_dir[:, i]), counts, -1) for i in range(2)]
    angles = ak.concatenate(angles, -1)
    # check which angle is the smallest i.e. decide which main shower to merge to
    minMask_angle = ak.min(angles, -1) == angles

    index_to_merge = [(index[to_merge])[minMask_angle[:, :, i]] for i in range(2)]

    targetParticleNumber = events.trueParticlesBT.particleNumber[best_match]

    correct_merge = [events.trueParticlesBT.particleNumber[index_to_merge[i]] == targetParticleNumber[:, i] for i in range(2)]

    correctlyMerged = ak.count(correct_merge[0][correct_merge[0]]) + ak.count(correct_merge[1][correct_merge[1]])
    total = ak.count(correct_merge[0]) + ak.count(correct_merge[1])
    if total > 0:
        correct_merge = (correctlyMerged/total) * 100
    else:
        correct_merge = 100
    print(f"nPFP's correctly merged: {correct_merge:.2f}%")
    return correct_merge

@timer
def SelectSample(events : Data, nDaughters : int, merge : bool = False, backtracked : bool = False, cheatMerging : bool = False) -> Data:
    """ Applies MC matching and shower merging to events with
        specificed number of objects

    Args:
        events (Master.Data): events to look at
        nDaughters (int): number of objects per event
        merge (bool, optional): should we do shower merging?. Defaults to False.

    Returns:
        Master.Data: selected sample
    """
    valid = Pi0MCMask(events, nDaughters) # get mask of events
    filtered = events.Filter([valid], [valid], True) # filter events with mask

    if backtracked == False:
        matched, unmatched, selection = filtered.MCMatching(applyFilters=False)
        filtered.Filter([selection],[selection]) # apply the selection
    else:
        singleMatchedEvents = filtered.trueParticlesBT.SingleMatch
        filtered.Filter([singleMatchedEvents], [singleMatchedEvents])
        best_match, selection = filtered.MatchByAngleBT()
        filtered.Filter([selection], [selection])

    if merge is True and backtracked is False:
        filtered = filtered.mergeShower(filtered, matched[selection], unmatched[selection], 1, False)

    if merge is True and backtracked is True:
        if cheatMerging is True:
            filtered, null = filtered.MergePFPCheat()
            filtered.Filter([null], [null])
        else:
            filtered = filtered.MergeShowerBT(best_match[selection])

    # if we don't merge showers, just get the showers matched to MC
    if merge is False and backtracked is False:
        filtered.Filter([matched[selection]])
    if merge is False and backtracked is True:
        filtered.Filter([best_match[selection]])
    return filtered
