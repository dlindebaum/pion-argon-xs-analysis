"""
Created on: 04/02/2022 12:26

Author: Shyam Bhuller

Description: Module containing core components of analysis code. 
"""

from abc import ABC, abstractmethod
import warnings
import uproot
import awkward as ak
import time
import numpy as np
import itertools
# custom modules
import vector

def timer(func):
    """ Decorator which times a function.

    Args:
        func (function): function to time
    """
    def wrapper_function(*args, **kwargs):
        """times func, returns outputs
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
    def __init__(self, _filename : str, _nEvents : int=-1, _start : int=None) -> None:
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


class Data:
    def __init__(self, _filename : str = None, includeBackTrackedMC : bool = False, _nEvents : int = -1, _start : int = 0) -> None:
        self.filename = _filename
        if self.filename != None:
            self.nEvents = _nEvents
            self.start = _start
            self.io = IO(self.filename, self.nEvents, self.start)
            self.eventNum = self.io.Get("EventID")
            self.subRun = self.io.Get("SubRun")
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
    def MatchByAngleBT(self):
        """ Equivlant to shower matching/angluar closeness cut, but for backtracked MC.

        Args:
            events (Master.Data): events to look at

        Raises:
            Exception: if all reco PFP's backtracked to the same true particle

        Returns:
            ak.Array: angular closeness of "matched" particles
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


    def CreateSpecificEventFilter(self, eventNums : list):
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
        """ Filter events.

        Args:
            reco_filters (list, optional): list of filters to apply to reconstructed data. Defaults to [].
            true_filters (list, optional): list of filters to apply to true data. Defaults to [].
            returnCopy   (bool, optional): if true return a copy of filtered events, else change self

        Returns:
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
            filtered = Data()
            filtered.filename = self.filename
            filtered.nEvents = self.nEvents
            filtered.start = self.start
            filtered.io = IO(filtered.filename, filtered.nEvents, filtered.start)
            filtered.eventNum = self.eventNum
            filtered.subRun = self.subRun
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
        beamParticle = self.recoParticles.number == self.recoParticles.beam_number # get beam particle
        beamParticleDaughters = self.recoParticles.mother == self.recoParticles.beam_number # get daugter of beam particle
        # combine masks
        particle_mask = np.logical_or(beamParticle, beamParticleDaughters)
        #? which one to do?
        #self.Filter([hasBeam], [hasBeam]) # filter data
        self.Filter([hasBeam, particle_mask[hasBeam]], [hasBeam]) # filter data
    

    def MergePFPCheat(self):
        mcIndex = self.trueParticlesBT.particleNumber
        unqiueIndex = self.trueParticlesBT.GetUniqueParticleNumbers(mcIndex) # get unique list of true particles
        p_new = []
        e_new = []
        dir_new = []
        truePFPMask = []
        for i in range(2):
            pfps = mcIndex == unqiueIndex[:, i] # sort pfps based on which true particle they back-track to
            
            # get single copy of true PFP
            ind = ak.local_index(pfps, -1)
            ind = ak.min(ind[pfps], -1)
            truePFPMask.append(ak.unflatten(ind, 1, -1))

            p = self.recoParticles.momentum[pfps]
            p = ak.where(p.x == -999, {"x": 0,"y": 0,"z": 0}, p) # dont merge PFP's with null data
            p_m = ak.sum(p, -1) # sum all momentum vectors
            e_m = vector.magnitude(p_m)
            dir_m = vector.normalize(p_m)
            p_new.append(ak.unflatten(p_m, 1, -1))
            e_new.append(ak.unflatten(e_m, 1, -1))
            dir_new.append(ak.unflatten(dir_m, 1, -1))
        
        truePFPMask = ak.concatenate(truePFPMask, axis=-1)

        events_merge = self.Filter(returnCopy=True) # easy way to make copies of the class
        mergedMomentum = ak.concatenate(p_new, axis=-1)
        mergedEnergy = ak.concatenate(e_new, axis=-1)
        mergedDirection = ak.concatenate(dir_new, axis=-1)
        events_merge.recoParticles._RecoParticleData__momentum = mergedMomentum
        events_merge.recoParticles._RecoParticleData__energy = mergedEnergy
        events_merge.recoParticles._RecoParticleData__direction = mergedDirection
        
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
                setattr(self, var_name, getattr(self, var_name)[f])


    def Filter(self, filters : list, returnCopy : bool = True):
        """ Filter data.

        Args:
            filters (list): list of filters to apply to particle data.

        Returns:
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
    def __init__(self, events : Data) -> None:
        super().__init__(events)

    @property
    def pdg(self):
        self.LoadData("pdg", "g4_Pdg")
        return getattr(self, f"_{type(self).__name__}__pdg")

    @property
    def number(self):
        self.LoadData("number", "g4_num")
        return getattr(self, f"_{type(self).__name__}__number")

    @property
    def mother(self):
        self.LoadData("mother", "g4_mother")
        return getattr(self, f"_{type(self).__name__}__mother")

    @property
    def energy(self):
        self.LoadData("energy", "g4_startE")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def momentum(self):
        self.LoadData("momentum", ["g4_pX", "g4_pY", "g4_pZ"])
        return getattr(self, f"_{type(self).__name__}__momentum")
    
    @property
    def direction(self):
        if not hasattr(self, f"_{type(self).__name__}__direction"):
            self.__direction = vector.normalize(self.momentum)
        return self.__direction

    @property
    def startPos(self):
        self.LoadData("startPos", ["g4_startX", "g4_startY", "g4_startZ"])
        return getattr(self, f"_{type(self).__name__}__startPos")

    @property
    def endPos(self):
        self.LoadData("startPos", ["g4_endX", "g4_endY", "g4_endZ"])
        return getattr(self, f"_{type(self).__name__}__endPos")

    @property
    def pi0_MC(self):
        if not hasattr(self, f"_{type(self).__name__}__pi0_MC"):
            self.__pi0_MC = ak.any(np.logical_and(self.number == 1, self.pdg == 111)) # check if we are looking at pure pi0 MC or beam MC
        return self.__pi0_MC

    @property
    def PrimaryPi0Mask(self):
        if self.pi0_MC:
            return np.logical_and(self.number == 1, self.pdg == 111)
        else:
            # assume we have beam MC
            # get pi0s produced from beam interaction
            return np.logical_and(self.mother == 1, self.pdg == 111)

    @property
    def truePhotonMask(self):
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


    def CalculatePairQuantities(self):
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
    def __init__(self, events : Data) -> None:
        super().__init__(events)

    @property
    def beam_number(self):
        self.LoadData("beam_number", "beamNum")
        return getattr(self, f"_{type(self).__name__}__beam_number")

    @property
    def sliceID(self):
        self.LoadData("sliceID", "reco_daughter_allShower_sliceID")
        return getattr(self, f"_{type(self).__name__}__sliceID")

    @property
    def beamCosmicScore(self):
        self.LoadData("beamCosmicScore", "reco_daughter_allShower_beamCosmicScore")
        return getattr(self, f"_{type(self).__name__}__beamCosmicScore")

    @property
    def number(self):
        self.LoadData("number", "reco_PFP_ID")
        return getattr(self, f"_{type(self).__name__}__number")

    @property
    def mother(self):
        self.LoadData("mother", "reco_PFP_Mother")
        return getattr(self, f"_{type(self).__name__}__mother")
    
    @property
    def nHits(self):
        self.LoadData("nHits", "reco_daughter_PFP_nHits_collection")
        return getattr(self, f"_{type(self).__name__}__nHits")

    @property
    def startPos(self):
        nTuples = [
            "reco_daughter_allShower_startX",
            "reco_daughter_allShower_startY",
            "reco_daughter_allShower_startZ"
        ]
        self.LoadData("startPos", nTuples)
        return getattr(self, f"_{type(self).__name__}__startPos")

    @property
    def direction(self):
        nTuples = [
            "reco_daughter_allShower_dirX",
            "reco_daughter_allShower_dirY",
            "reco_daughter_allShower_dirZ"
        ]
        self.LoadData("direction", nTuples)
        return getattr(self, f"_{type(self).__name__}__direction")

    @property
    def energy(self):
        self.LoadData("energy", "reco_daughter_allShower_energy")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def momentum(self):
        if not hasattr(self, f"_{type(self).__name__}__momentum"):
            mom = vector.prod(self.energy, self.direction)
            mom = ak.where(self.direction.x == -999, {"x": -999,"y": -999,"z": -999}, mom)
            mom = ak.where(self.energy < 0, {"x": -999,"y": -999,"z": -999}, mom)
            self.__momentum = mom
        return self.__momentum

    @property
    def showerLength(self):
        self.LoadData("showerLength", "reco_daughter_allShower_length")
        return getattr(self, f"_{type(self).__name__}__showerLength")

    @property
    def showerConeAngle(self):
        self.LoadData("coneAngle", "reco_daughter_allShower_coneAngle")
        return getattr(self, f"_{type(self).__name__}__coneAngle")


    def CalculatePairQuantities(self, useBT : bool = False):
        #? have each value as an @property?
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
        """ Get shower pair values, in pairs

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
        """Get all shower pair combinations, excluding duplicates and self paired.

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
    def ShowerPairsByHits(hits) -> list:
        """pair reconstructed showers in an event by the number of hits.
        pairs the two largest showers per event.
        TODO figure out a way to do this without sorting events (or re-sort events?)

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
    def __init__(self, events: Data) -> None:
        super().__init__(events)

    @property
    def number(self):
        self.LoadData("number", "reco_daughter_PFP_true_byHits_ID")
        return getattr(self, f"_{type(self).__name__}__number")

    @property
    def mother(self):
        self.LoadData("mother", "reco_daughter_PFP_true_byHits_Mother")
        return getattr(self, f"_{type(self).__name__}__mother")

    @property
    def pdg(self):
        self.LoadData("pdg", "reco_daughter_PFP_true_byHits_pdg")
        return getattr(self, f"_{type(self).__name__}__pdg")

    @property
    def startPos(self):
        nTuples = [
            "reco_daughter_PFP_true_byHits_startX",
            "reco_daughter_PFP_true_byHits_startY",
            "reco_daughter_PFP_true_byHits_startZ"
        ]
        self.LoadData("startPos", nTuples)
        return getattr(self, f"_{type(self).__name__}__startPos")

    @property
    def endPos(self):
        nTuples = [
            "reco_daughter_PFP_true_byHits_endX",
            "reco_daughter_PFP_true_byHits_endY",
            "reco_daughter_PFP_true_byHits_endZ"
        ]
        self.LoadData("direction", nTuples)
        return getattr(self, f"_{type(self).__name__}__endPos")

    @property
    def momentum(self):
        nTuples = [
            "reco_daughter_PFP_true_byHits_pX",
            "reco_daughter_PFP_true_byHits_pY",
            "reco_daughter_PFP_true_byHits_pZ"
        ]
        self.LoadData("momentum", nTuples)
        return getattr(self, f"_{type(self).__name__}__momentum")

    @property
    def direction(self):
        if not hasattr(self, f"_{type(self).__name__}__direction"):
            self.__direction = vector.normalize(self.momentum)
        return self.__direction

    @property
    def energy(self):
        self.LoadData("energy", "reco_daughter_PFP_true_byHits_startE")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def matchedHits(self):
        self.LoadData("energy", "reco_daughter_PFP_true_byHits_matchedHits")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def hitsInRecoCluster(self):
        self.LoadData("energy", "reco_daughter_PFP_true_byHits_hitsInRecoCluster")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def mcParticleHits(self):
        self.LoadData("energy", "reco_daughter_PFP_true_byHits_mcParticleHits")
        return getattr(self, f"_{type(self).__name__}__energy")

    @property
    def sharedHits(self):
        self.LoadData("energy", "reco_daughter_PFP_true_byHits_sharedHits")
        return getattr(self, f"_{type(self).__name__}__energy")


    @property
    def particleNumber(self):
        """ Gets the true particle number of each true particle backtracked to reco

        Args:
            events (Master.Data): events to look at

        Returns:
            ak.Array(): awkward array of true particle indices
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


    def GetUniqueParticleNumbers(self, index : ak.Array):
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
        #? can I do this without loops?
        uniqueIndex = [np.unique(index_padded_np[i, :]) for i in range(len(index_padded_np))] # get the unique entries per event
        uniqueIndex = ak.Array(uniqueIndex)

        # remove -1 entries as these just represent padded entries
        return uniqueIndex[uniqueIndex != -1]

    @property
    def SingleMatch(self):
        """ Get a boolean mask of events with more than one tagged true particle
            in the back tracker.

        Returns:
            ak.Array: boolean mask
        """
        unqiueIndex = self.GetUniqueParticleNumbers(self.particleNumber)
        singleMatch = ak.num(unqiueIndex) < 2 #!
        return np.logical_not(singleMatch)


    def CalculatePairQuantities(self):
        #? have each value as an @property?
        #? method is almost identical as reconsturected particleData equivilant. do we keep this function in the Data class?
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


def NPFPMask(events : Data, daughters : int = None):
    """ Create a boolean mask of events with a specific number of events.

    Args:
        events (Data): events to filter
        daughters (int, optional): keep events with specific number of daughters. Defaults to None.

    Returns:
        ak.Array: mask of events to filter
    """
    null_dir = events.recoParticles.direction.x != -999
    null_pos = events.recoParticles.startPos.x != -999
    nDaughter = np.logical_and(null_dir, null_pos)
    nDaughter = ak.num(nDaughter[nDaughter]) # get number of showers which have a valid direction

    if daughters == None:
        r_mask = nDaughter > 1
    elif daughters < 0:
        r_mask = nDaughter > abs(daughters)
    else:
        r_mask = nDaughter == daughters
    return r_mask


def Pi0TwoBodyDecayMask(events : Data):
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
def Pi0MCMask(events : Data, daughters : int = None):
    """ A filter for Pi0 MC dataset, selects events with a specific number of daughters
        which have a valid direction vector and removes events with the 3 body pi0 decay.

    Args:
        events (Event): events being studied
        daughters (int): keep events with specific number of daughters. Defaults to None

    Returns:
        ak.Array: mask of events to filter
        ak.Array: mask of true photons to apply to true data
    """
    r_mask = NPFPMask(events, daughters)
    t_mask = Pi0TwoBodyDecayMask(events)
    valid = np.logical_and(r_mask, t_mask)
    return valid

@timer
def BeamMCFilter(events : Data, n_pi0 : int = 1, returnCopy=True):
    """ Filters BeamMC data to get events with only 1 pi0 which originates from the beam particle interaction.

    Args:
        events (Data): events to filter

    Returns:
        Data: selected events
    """
    #* remove events with no truth info aka beam filter
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


def FractionalError(reco : ak.Array, true : ak.Array, null : ak.Array):
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
def CalculateQuantities(events : Data, backtrackedTruth : bool = False):
    #? add to Data class?
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


def ShowerMergePerformance(events : Data, best_match : ak.Array):
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
def SelectSample(events : Data, nDaughters : int, merge : bool = False, backtracked : bool = False, cheatMerging : bool = False):
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
