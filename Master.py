"""
Created on: 04/02/2022 12:26

Author: Shyam Bhuller

Description: Module containing core components of analysis code. 
"""

import warnings
import uproot
import awkward as ak
import time
import numpy as np
import itertools
# custom modules
import vector


def timer(func):
    """Decorator which times a function.

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


def GenericFilter(particleData, filters):
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


class Data:
    def __init__(self, _filename : str = None) -> None:
        self.filename = _filename
        if self.filename != None:
            self.io = IO(self.filename)
            self.eventNum = self.io.Get("EventID")
            self.subRun = self.io.Get("SubRun")
            self.trueParticles = TrueParticleData(self)
            self.recoParticles = RecoParticleData(self)

    @property
    def SortedTrueEnergyMask(self) -> ak.Array:
        """ Returns index of shower pairs sorted by true energy (highest first).

        Args:
            primary (bool): sort index of primary pi0 decay photons (particle gun MC)

        Returns:
            ak.Array: sorted indices of true photons
        """
        mask = self.trueParticles.pdg == 22
        mask = np.logical_and(mask, self.trueParticles.mother == 1)
        return ak.argsort(self.trueParticles.energy[mask], ascending=True)

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


    def Filter(self, reco_filters : list = [], true_filters : list = [], returnCopy : bool = True):
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
            GenericFilter(self, reco_filters) #? should true_filters also be applied?
        else:
            filtered = Data()
            filtered.eventNum = self.eventNum
            filtered.subRun = self.subRun
            filtered.trueParticles = self.trueParticles.Filter(true_filters)
            filtered.recoParticles = self.recoParticles.Filter(reco_filters)
            GenericFilter(filtered, reco_filters) #? should true_filters also be applied?
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
        self.Filter([hasBeam, particle_mask[hasBeam]], [hasBeam], returnCopy=False) # filter data


class TrueParticleData:
    def __init__(self, events : Data) -> None:
        self.events = events # parent of TrueParticles
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

    @property
    def truePhotonMask(self):
        photons = self.mother == 1 # get only primary daughters
        photons = np.logical_and(photons, self.pdg == 22)
        return photons


    def Filter(self, filters : list, returnCopy : bool = True):
        """Filter true particle data.

        Args:
            filters (list): list of filters to apply to true data.

        Returns:
            TrueParticles: filtered data.
        """
        if returnCopy is False:
            GenericFilter(self, filters)
        else:
            filtered = TrueParticleData(Data())
            for var in vars(self):
                setattr(filtered, var, getattr(self, var))
            GenericFilter(filtered, filters)
            return filtered


class RecoParticleData:
    def __init__(self, events : Data) -> None:
        self.events = events # parent of RecoParticles
        print(events.filename)
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


    def Filter(self, filters : list, returnCopy : bool = True):
        """Filter reconstructed data.

        Args:
            filters (list): list of filters to apply to reconstructed data.

        Returns:
            RecoParticles: filtered data.
        """
        if returnCopy is False:
            GenericFilter(self, filters)
        else:
            filtered = RecoParticleData(Data())
            for var in vars(self):
                setattr(filtered, var, getattr(self, var))
            GenericFilter(filtered, filters)
            return filtered

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


def Pi0MCMask(events : Data, daughters : int = None):
    #?do filtering within function i.e. return object of type Data or mutate events argument?
    """ A filter for Pi0 MC dataset, selects events with a specific number of daughters
        which have a valid direction vector and removes events with the 3 body pi0 decay.

    Args:
        events (Event): events being studied
        daughters (int): keep events with specific number of daughters. Defaults to None

    Returns:
        ak.Array: mask of events to filter
        ak.Array: mask of true photons to apply to true data
    """
    nDaughter = events.recoParticles.direction.x != -999
    nDaughter = ak.num(nDaughter[nDaughter]) # get number of showers which have a valid direction

    if daughters == None:
        r_mask = nDaughter > 1
    elif daughters < 0:
        r_mask = nDaughter > abs(daughters)
    else:
        r_mask = nDaughter == daughters
    
    t_mask = ak.num(events.trueParticles.truePhotonMask[events.trueParticles.truePhotonMask], -1) == 2 # exclude pi0 -> e+ + e- + photons
    print(f"number of dalitz decays: {ak.count(t_mask[np.logical_not(t_mask)])}")

    valid = np.logical_and(r_mask, t_mask) # events which have 2 reco daughters and correct pi0 decay
    return valid

@timer
def MCTruth(events : Data):
    #? move functionality into trueParticleData class and have each value as an @property?
    """ Calculate true shower pair quantities.

    Args:
        events (Master.Event): events to process

    Returns:
        tuple of ak.Array: calculated quantities
    """
    #* get the primary pi0
    mask_pi0 = np.logical_and(events.trueParticles.number == 1, events.trueParticles.pdg == 111)
    photons = events.trueParticles.truePhotonMask
    sortEnergy = events.SortedTrueEnergyMask

    #* compute start momentum of dauhters
    p_daughter = events.trueParticles.momentum[photons]
    sum_p = ak.sum(p_daughter, axis=1)
    sum_p = vector.magntiude(sum_p)
    p_daughter_mag = vector.magntiude(p_daughter)
    p_daughter_mag = p_daughter_mag[sortEnergy]

    #* compute true opening angle
    angle = np.arccos(vector.dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

    #* compute invariant mass
    e_daughter = events.trueParticles.energy[photons]
    inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

    #* pi0 momentum
    p_pi0 = events.trueParticles.momentum[mask_pi0]
    p_pi0 = vector.magntiude(p_pi0)
    return inv_mass, angle, p_daughter_mag[:, 1:], p_daughter_mag[:, :-1], p_pi0

@timer
def RecoQuantities(events : Data):
    #? move functionality into recoParticleData class and have each value as an @property?
    """ Calculate reconstructed shower pair quantities.

    Args:
        events(Master.Event): events to process

    Returns:
        tuple of ak.Array: calculated quantities + array which masks null shower pairs
    """
    sortEnergy = events.SortedTrueEnergyMask
    sortedPairs = ak.unflatten(events.recoParticles.energy[sortEnergy], 1, 0)
    leading = sortedPairs[:, :, 1:]
    secondary = sortedPairs[:, :, :-1]

    #* opening angle
    direction_pair = ak.unflatten(events.recoParticles.direction[sortEnergy], 1, 0)
    direction_pair_mag = vector.magntiude(direction_pair)
    angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

    #* Invariant Mass
    inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

    #* pi0 momentum
    pi0_momentum = vector.magntiude(ak.sum(events.recoParticles.momentum, axis=-1))/1000

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

    return inv_mass, angle, leading, secondary, pi0_momentum, ak.unflatten(null, 1, 0)


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
    true = ak.where( ak.num(true, 1) > 0, true, [np.nan]*len(true) )
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
    mct = MCTruth(events)
    rmc = RecoQuantities(events)

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