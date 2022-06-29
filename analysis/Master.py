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
import sys
# custom modules
sys.path.insert(1, "../")
import analysis.vector as vector

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


def __GenericFilter__(particleData, filters):
    for f in filters:
        for var in vars(particleData):
            if hasattr(getattr(particleData, var), "__getitem__"):
                try:
                    setattr(particleData, var, getattr(particleData, var)[f])
                except:
                    warnings.warn(f"Couldn't apply filters to {var}.")


class IO:
    #? handle opening root file, and setting the ith event
    def __init__(self, _filename : str) -> None:
        self.file = uproot.open(_filename)["pduneana/beamana"]
        self.nEvents = len(self.file["EventID"].array())
    def Get(self, item : str) -> ak.Array:
        """Load nTuple from root file as awkward array.

        Args:
            item (str): nTuple name in root file

        Returns:
            ak.Array: nTuple loaded
        """        
        try:
            return self.file[item].array()
        except uproot.KeyInFileError:
            print(f"{item} not found in file, moving on...")
            return None
    def List(self) -> list:
        """Print the parameters in the uproot file.

        Returns:
            list: Top level parameters
        """        
        return self.file.keys()


class Data:
    def __init__(self, _filename : str = None, includeBackTrackedMC : bool = False) -> None:
        self.filename = _filename
        if self.filename != None:
            self.io = IO(self.filename)
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
        """ function which matched reco showers to true photons for pi0 decays.
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
        """ equivlant to shower matching/angluar closeness cut, but for backtracked MC.

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
        null = np.logical_or(null_direction, null_momentum)
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

            __GenericFilter__(self, reco_filters) #? should true_filters also be applied?
            self.trueParticles.events = self
            self.recoParticles.events = self
            if hasattr(self, "trueParticlesBT"):
                self.trueParticlesBT.events = self
        else:
            filtered = Data()
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
        self.Filter([hasBeam, particle_mask[hasBeam]], [hasBeam]) # filter data
    

    def mergePFPCheat(self):
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
            e_m = vector.magntiude(p_m)
            dir_m = vector.normalize(p_m)
            p_new.append(ak.unflatten(p_m, 1, -1))
            e_new.append(ak.unflatten(e_m, 1, -1))
            dir_new.append(ak.unflatten(dir_m, 1, -1))
        
        truePFPMask = ak.concatenate(truePFPMask, axis=-1)

        events_merge = self.Filter(returnCopy=True) # easy way to make copies of the class
        events_merge.recoParticles.momentum = ak.concatenate(p_new, axis=-1)
        events_merge.recoParticles.energy = ak.concatenate(e_new, axis=-1)
        events_merge.recoParticles.direction = ak.concatenate(dir_new, axis=-1)
        
        events_merge.trueParticlesBT.Filter([truePFPMask], False) # filter to get the true particles the merged PFP's relate to
        null = np.logical_not(ak.any(events_merge.recoParticles.energy == 0, -1)) # exlucde events where all PFP's merged had no valid momentmum vector
        print(f"Events where one merged PFP had undefined momentum: {ak.count(null[np.logical_not(null)])}")
        return events_merge, null

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
        mergedMomentum = [ak.unflatten(vector.Add(leading_momentum[:, i], momentumToMerge[i]), 1, -1) for i in range(2)]
        mergedMomentum = ak.concatenate(mergedMomentum, -1)

        mergedDirection = [ak.unflatten(vector.normalize(mergedMomentum[:, i]), 1, -1) for i in range(2)]
        merged_direction = ak.concatenate(mergedDirection, -1)

        merged_energy = [ak.unflatten(vector.magntiude(mergedMomentum[:, i]), 1, -1) for i in range(2)]
        merged_energy = ak.concatenate(merged_energy, -1)

        merged_events = self.Filter(returnCopy=True)
        merged_events.recoParticles.momentum = mergedMomentum
        merged_events.recoParticles.direction = merged_direction
        merged_events.recoParticles.energy = merged_energy

        merged_events.trueParticlesBT.Filter([best_match], returnCopy=False)
        return merged_events


class ParticleData(ABC):
    @abstractmethod
    def __init__(self, events : Data) -> None:
        self.events = events # keep reference of parent class
        pass

    @abstractmethod
    def CalculatePairQuantities(self):
        pass


    def Filter(self, filters : list, returnCopy : bool = True):
        """Filter reconstructed data.

        Args:
            filters (list): list of filters to apply to particle data.

        Returns:
            subClass(ParticleData): filtered data.
        """
        if returnCopy is False:
            __GenericFilter__(self, filters)
        else:
            subclass = globals()[type(self).__name__] # get the class which is of type ParticleData
            filtered = subclass(Data()) # create a new instance of the class
            for var in vars(self):
                setattr(filtered, var, getattr(self, var))
            __GenericFilter__(filtered, filters)
            return filtered
    
    def GetValues(self, value):
        if hasattr(self.events, "io"):
            return self.events.io.Get(value)


class TrueParticleData(ParticleData):
    def __init__(self, events : Data) -> None:
        super().__init__(events)
        if hasattr(self.events, "io"):
            self.number = self.events.io.Get("g4_num")
            self.mother = self.events.io.Get("g4_mother")
            self.pdg = self.events.io.Get("g4_Pdg")
            self.energy = self.events.io.Get("g4_startE")
            self.momentum = ak.zip({"x" : self.events.io.Get("g4_pX"),
                                    "y" : self.events.io.Get("g4_pY"),
                                    "z" : self.events.io.Get("g4_pZ")})
            self.direction = vector.normalize(self.momentum)
            self.startPos = ak.zip({"x" : self.events.io.Get("g4_startX"),
                                    "y" : self.events.io.Get("g4_startY"),
                                    "z" : self.events.io.Get("g4_startZ")})
            self.endPos = ak.zip({"x" : self.events.io.Get("g4_endX"),
                                  "y" : self.events.io.Get("g4_endY"),
                                  "z" : self.events.io.Get("g4_endZ")})
            self.pi0_MC = ak.any(np.logical_and(self.number == 1, self.pdg == 111)) # check if we are looking at pure pi0 MC or beam MC

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
        
        #* compute start momentum of daughters
        p_daughter = self.momentum[photons]
        sum_p = ak.sum(p_daughter, axis=1)
        sum_p = vector.magntiude(sum_p)
        p_daughter_mag = vector.magntiude(p_daughter)
        p_daughter_mag = p_daughter_mag[sortEnergy]

        #* compute true opening angle
        angle = np.arccos(vector.dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

        #* compute invariant mass
        e_daughter = self.energy[photons]
        inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

        #* pi0 momentum
        p_pi0 = self.momentum[mask_pi0]
        p_pi0 = vector.magntiude(p_pi0)
        return inv_mass, angle, p_daughter_mag[:, 1:], p_daughter_mag[:, :-1], p_pi0


class RecoParticleData(ParticleData):
    def __init__(self, events : Data) -> None:
        super().__init__(events)
        if hasattr(self.events, "io"):
            self.beam_number = self.events.io.Get("beamNum")
            self.number = self.events.io.Get("reco_PFP_ID")
            self.mother = self.events.io.Get("reco_PFP_Mother")
            self.nHits = self.events.io.Get("reco_daughter_PFP_nHits_collection")
            self.direction = ak.zip({"x" : self.events.io.Get("reco_daughter_allShower_dirX"),
                                     "y" : self.events.io.Get("reco_daughter_allShower_dirY"),
                                     "z" : self.events.io.Get("reco_daughter_allShower_dirZ")})
            self.startPos = ak.zip({"x" : self.events.io.Get("reco_daughter_allShower_startX"),
                                    "y" : self.events.io.Get("reco_daughter_allShower_startY"),
                                    "z" : self.events.io.Get("reco_daughter_allShower_startZ")})
            self.energy = self.events.io.Get("reco_daughter_allShower_energy")
            self.momentum = self.GetMomentum()
            self.showerLength = self.events.io.Get("reco_daughter_allShower_length")
            self.showerConeAngle = self.events.io.Get("reco_daughter_allShower_coneAngle")


    def GetMomentum(self):
        # TODO turn into attribute with @property
        mom = vector.prod(self.energy, self.direction)
        mom = ak.where(self.direction.x == -999, {"x": -999,"y": -999,"z": -999}, mom)
        mom = ak.where(self.energy < 0, {"x": -999,"y": -999,"z": -999}, mom)
        return mom


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
        direction_pair_mag = vector.magntiude(direction_pair)
        angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

        #* Invariant Mass
        inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

        #* pi0 momentum
        pi0_momentum = vector.magntiude(ak.sum(self.momentum, axis=-1))/1000

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
        """get shower pair values, in pairs

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
        if hasattr(self.events, "io"):
            self.pdg = events.io.Get("reco_daughter_PFP_true_byHits_pdg")
            self.startPos = ak.zip({"x" : events.io.Get("reco_daughter_PFP_true_byHits_startX"),
                            "y" : events.io.Get("reco_daughter_PFP_true_byHits_startY"),
                            "z" : events.io.Get("reco_daughter_PFP_true_byHits_startZ")})
            self.endPos = ak.zip({"x" : events.io.Get("reco_daughter_PFP_true_byHits_endX"),
                            "y" : events.io.Get("reco_daughter_PFP_true_byHits_endY"),
                            "z" : events.io.Get("reco_daughter_PFP_true_byHits_endZ")})
            self.momentum = ak.zip({"x" : events.io.Get("reco_daughter_PFP_true_byHits_pX"),
                            "y" : events.io.Get("reco_daughter_PFP_true_byHits_pY"),
                            "z" : events.io.Get("reco_daughter_PFP_true_byHits_pZ")})
            self.direction = vector.normalize(self.momentum)
            self.energy = events.io.Get("reco_daughter_PFP_true_byHits_startE")
            #! multiplying energy by 1000 seems to ruin everything?
            #self.energy = ak.where(self.energy < 0, -999, self.energy*1000)
    
    @property
    def particleNumber(self):
        """ Gets the true particle number of each true particle batckracked to reco

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
        singleMatch = ak.num(unqiueIndex) != 2
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
        direction_pair_mag = vector.magntiude(direction_pair)
        angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

        #* Invariant Mass
        inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

        #* pi0 momentum
        pi0_momentum = vector.magntiude(ak.sum(self.momentum, axis=-1))/1000

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


def NPFPMask(events : Data, daughters : int = None):
    """ Create a boolean mask of events with a specific number of events.

    Args:
        events (Data): events to filter
        daughters (int, optional): keep events with specific number of daughters. Defaults to None.

    Returns:
        ak.Array: mask of events to filter
    """
    nDaughter = events.recoParticles.direction.x != -999
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


def BeamMCFilter(events : Data, n_pi0 : int = 1):
    """ Filters BeamMC data to get events with only 1 pi0 which originates from the beam particle interaction.

    Args:
        events (Data): events to filter

    Returns:
        Data: selected events
    """
    #* remove events with no truth info aka beam filter
    empty = ak.num(events.trueParticles.number) > 0 
    filtered = events.Filter([empty], [empty], returnCopy=True)

    #* only look at events with 1 primary pi0
    pi0 = filtered.trueParticles.PrimaryPi0Mask
    single_primary_pi0 = ak.num(pi0[pi0]) == n_pi0 # only look at events with 1 pi0
    filtered.Filter([single_primary_pi0], [single_primary_pi0], False)

    #* remove true particles which aren't primaries
    primary_pi0 = filtered.trueParticles.PrimaryPi0Mask
    primary_daughter = filtered.trueParticles.truePhotonMask # this is fine so long as we only care about pi0->gamma gamma
    primaries = np.logical_or(primary_pi0, primary_daughter)
    filtered.Filter([], [primaries], False)
    return filtered


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
def CalculateQuantities(events : Data, names : str):
    """Calcaulte reco/ true quantities of shower pairs, and format them for plotting

    Args:
        events (Master.Event): events to look at
        names (str): quantity names

    Returns:
        tuple of np.arrays: quantities to plot
    """
    mct = events.trueParticles.CalculatePairQuantities()
    rmc = events.recoParticles.CalculatePairQuantities()

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
