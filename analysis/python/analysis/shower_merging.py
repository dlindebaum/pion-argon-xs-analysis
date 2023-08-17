"""
Created on: 13/03/2023 21:47

Author: Shyam Bhuller

Description: 
"""
import warnings

import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
from tabulate import tabulate

from python.analysis import Master, Plots, vector, EventSelection
from python.analysis import LegacyBeamParticleSelection, BeamParticleSelection, PFOSelection


def SetPlotStyle(extend_colors : bool = False, dpi : int = 300):
    plt.style.use("default") # first load the default to reset any previous changes made by other styles
    plt.style.use('ggplot')
    plt.rcParams.update({'patch.linewidth': 1})
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({"axes.titlecolor" : "#555555"})
    plt.rcParams.update({"axes.titlesize" : 12})
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['legend.fontsize'] = "small"
    if extend_colors:
        plt.rcParams.update({"axes.prop_cycle" : plt.cycler("color", get_cmap("tab20").colors)})


class ShowerMergeQuantities:
    xlabels = [
        "$\\alpha$ (rad)",
        "$\delta x$ (cm)",
        "$\delta x_{l}$ (cm)",
        "$\delta x_{t}$ (cm)",
        "$\delta\phi$ (rad)",
        "d (cm)",
        "t (cm)",
        "p (cm)",
    ]
    selectionVariables = [
        "alpha",    
        "delta_x",
        "delta_xl",
        "delta_xt",
        "delta_phi",
        "d",
        "t",
        "p",
    ]

    bestCut = "purity"

    def __init__(self, events : Master.Data = None, to_merge = None, analysedCuts : str = None):
        self.analysedCuts = analysedCuts
        if events:
            #* collect positions and directions of PFOs
            self.to_merge_dir = events.recoParticles.direction[to_merge]
            self.to_merge_pos = events.recoParticles.startPos[to_merge]
            # check null positions/directions, throw warning if this is the case
            null = np.logical_or(self.to_merge_dir.x != -999, self.to_merge_pos.x != -999)
            if not (ak.all(null == True) and not ak.any(null == False)):
                warnings.warn("events passed to ShowerMergeQuantities contains PFOs with undefined positions or directions!", RuntimeWarning)


    @Master.timer
    def Evaluate(self, events : Master.Data, start_showers : ak.Array):
        """ Calculate quantities which may help select PFOs to merge

        Args:
            events (Master.Data): events to study
            start_showers (ak.Array): initial showers to merge to
        """
        # collect relavent parameters
        start_shower_pos = events.recoParticles.startPos[np.logical_or(*start_showers)]
        start_shower_dir = events.recoParticles.direction[np.logical_or(*start_showers)]

        # calculate
        self.t = ak.Array([vector.magnitude(vector.cross(vector.sub(start_shower_pos[:, i], self.to_merge_pos), self.to_merge_dir)) for i in range(2)])

        v3 = ak.Array([vector.normalize(vector.cross(start_shower_dir[:, i], self.to_merge_dir)) for i in range(2)])
        self.d = np.abs(ak.Array([vector.dot(vector.sub(start_shower_pos[:, i], self.to_merge_pos), v3[i]) for i in range(2)]))

        pi0_vertex = events.recoParticles.beam_endPos # assume the pi0 lifetime is short so these are approximately the same (should be even at 6GeV beam energy)
        self.p = ak.Array([vector.magnitude(vector.cross(vector.sub(pi0_vertex, self.to_merge_pos), self.to_merge_dir)) for i in range(2)])

        self.delta_phi = ak.Array([vector.angle(start_shower_dir[:, i], self.to_merge_dir) for i in range(2)])
        displacement = ak.Array([vector.sub(self.to_merge_pos, start_shower_pos[:, i]) for i in range(2)])
        self.alpha = ak.Array([vector.angle(displacement[i], start_shower_dir[:, i]) for i in range(2)])
        self.delta_x = ak.Array([vector.dist(start_shower_pos[:, i], self.to_merge_pos) for i in range(2)])
        self.delta_xl = ak.Array([self.delta_x[i] * np.abs(np.cos(self.alpha[i])) for i in range(2)])
        self.delta_xt = ak.Array([self.delta_x[i] * np.abs(np.sin(self.alpha[i])) for i in range(2)])
        if self.analysedCuts is not None:
            self.mask = self.SelectPFOsToMerge(BestCut(pd.read_csv(self.analysedCuts), self.selectionVariables, self.bestCut), False)


    def SelectPFOsToMerge(self, cuts : list, applyCuts : bool) -> ak.Array:
        """ Get mask of PFOs which pass a set of cuts.

        Args:
            cuts (list): list of cuts to apply in order of selectionVariables
            applyCuts (bool): apply cuts to quantities?

        Returns:
            ak.Array: boolean mask
        """
        for v in range(len(self.selectionVariables)):
            print(f"cutting on: {self.selectionVariables[v]}")
            if v == 0:
                mask = getattr(self, self.selectionVariables[v]) < cuts[v]
            else:
                mask = np.logical_and(mask, getattr(self, self.selectionVariables[v]) < cuts[v])
        if(hasattr(self, "signal")):
            new_list = self.selectionVariables + ["signal", "background"]
        else:
            new_list = self.selectionVariables
        if applyCuts:
            for v in range(len(new_list)):
                d = getattr(self, new_list[v])
                setattr(self, new_list[v], [d[i][mask[i]] for i in range(2)])
        return mask


    def GenerateDataFrame(self, signal : ak.Array, background : ak.Array):
        """ Generates a pandas dataframe with the geometric quantities.

        Args:
            signal (ak.Array): signal PFO mask
            background (ak.Array): background PFO mask
        """
        #* create dataframe and add all calculated quantities
        for i in range(len(self.selectionVariables)):
            print(self.selectionVariables[i])
            if i == 0:
                df = ak.to_pandas(getattr(self, self.selectionVariables[i]), anonymous=self.selectionVariables[i])
            else:
                df = pd.concat([df, ak.to_pandas(getattr(self, self.selectionVariables[i]), anonymous=self.selectionVariables[i])], axis = 1)
        
        #* add signal and background boolean masks
        df = pd.concat([df, ak.to_pandas(signal, anonymous="signal")], axis = 1)
        df = pd.concat([df, ak.to_pandas([background, background], anonymous="background")], axis = 1)
        return df

    def SaveQuantitiesToCSV(self, signal : ak.Array, background : ak.Array, filename : str = "merge-quantities.csv"):
        """ Saves merge quantities as a pandas dataframe to file.

        Args:
            signal (ak.Array): signal PFO mask
            background (ak.Array): background PFO mask
            filename (str, optional): _description_. Defaults to "merge-quantities.csv".
        """
        self.GenerateDataFrame(signal, background).to_csv(filename)


    def LoadQuantitiesFromCSV(self, filename : str):
        """ Load merge quantities data and populate instance variables.

        Args:
            filename (str): compatible data file
        """
        #! data is flattened but indices and subindices are kept
        #! so it should be possible to do per event studies
        #* read data and populate instance variables
        data = pd.read_csv(filename)
        for n in self.selectionVariables:
            d = ak.Array(data[n].values.tolist())
            setattr(self, n, ak.unflatten(d, ak.count(d)//2))

        signal = ak.Array(data["signal"].values.tolist())
        background = ak.Array(data["background"].values.tolist())

        self.signal = ak.unflatten(signal, ak.count(signal)//2)
        self.background = ak.unflatten(background, ak.count(background)//2)
        if self.analysedCuts is not None:
            self.SelectPFOsToMerge(BestCut(pd.read_csv(self.analysedCuts), self.selectionVariables, self.bestCut), applyCuts=True)


    def PlotQuantities(self, signal : ak.Array, background : ak.Array, min : bool = True, annotate : str = None, save : bool = False, outDir : str = "geometric_quantities/"):
        """ Plot geometric quantities to cosndier for shower merging

        Args:
            signal (ak.Array): signal PFOs
            background (ak.Array): background PFOs
        """
        #* plot and save
        labels = ["background", "signal"]
        for i in range(len(self.selectionVariables)):
            data = getattr(self, self.selectionVariables[i])
            print(data)
            
            #* collect signal PFOs
            if min is True:
                s = [data[j][np.logical_or(*signal)] for j in range(2)] # get all PFOs related to the pi0 decay
                s = ak.min(ak.concatenate(ak.unflatten(s, 1, -1), -1), -1) # take the smallest value as the signal
            else:
                s = ak.ravel([data[j][signal[j]] for j in range(2)]) # signal is found using truth info i.e. bt shower ID

            #* collect background PFOs
            b = ak.ravel([data[j][background[0]] for j in range(2)])

            Plots.PlotHistComparison([b, s], bins = 50, xlabel = self.xlabels[i], labels = labels, density = True, annotation = annotate)
            if save: Plots.Save(self.selectionVariables[i], outDir)


def BestCut(cuts : pd.DataFrame, q_names : list, type="balanced") -> list:
    """ Select a cut from the cut based scan using signal/background metrics.
        Currently has three types of cut it will pick: high purity, balanced and high efficiency

    Args:
        cuts (pd.DataFrame): dataframe of cuts + metrics
        q_names (list): quantity names
        type (str, optional): type of cut to pick. Defaults to "balanced".

    Returns:
        list : selected cut + metrics
    """
    types = ["balanced", "purity", "efficiency", "s_x_purity"]
    if type not in types:
        raise Exception(f"cut type must be either {types}")

    print("finding best cut")
    c = cuts[cuts["$\\epsilon$"] > 0] # pick cuts which dont exlude all PFOs

    if type == "balanced":
        # pick a cut which has reasonable signal efficiency, then pick the highest purity cut
        # c = c[c["$\\epsilon_{s}$"] > 0.5]
        c = c[(c["s"] > 1000) & (c["s/b"] > 2)]
        max_index = c["purity"].idxmax()

    if type == "purity":
        # pick a cut with small background efficiency and > 10% signal efficiency, then pick the highest purity cut
        c = c[c["$\\epsilon_{b}$"] < 0.1]
        c = c[c["$\\epsilon_{s}$"] > 0.1]
        max_index = c["purity"].idxmax()

    if type == "s_x_purity":
        # pick the cut with highest signal times purity
        sp = c["s"] * c["purity"]
        max_index = sp.idxmax()

    if type == "efficiency":
        # pick a cut with > 10% purity and then pick the highest efficiency cut
        c = c[c["purity"] > 0.1]
        max_index = c["$\\epsilon$"].idxmax()

    best_cuts = c[c.index == max_index]
    
    print("Best cut: ")
    print(best_cuts)
    return best_cuts[q_names].values.tolist()[0]

@Master.timer
def SplitSample(events : Master.Data, method="spatial") -> tuple:
    """ Select starting showers to merge for the pi0 decay.
        The starting showers are guarenteed to originate 
        from the pi0 decay (using truth information).
        Then return the start_showers and a mask of all 
        other PFOs (things to try merging).

    Args:
        events (Master.Data): events to look at
        method (str, optional): _description_. Defaults to "spatial".

    Raises:
        Exception: undefined method
        Exception: if all reco PFP's backtracked to the same true particle

    Returns:
        tuple: starting showers and PFOs to merge
    """
    #TODO fix a bug where occasionally a starting shower is not a daughter of the pi0.
    if method not in ["angular", "spatial"]:
        raise Exception('method for selecting start showers must be either "angular" or "spatial"')

    pi0 = ak.flatten(events.trueParticles.number[events.trueParticles.PrimaryPi0Mask]) # get pi0 id
    pi0_daughters = events.trueParticlesBT.mother == pi0 # get taggged daughters by matching pi0 pdg id to mother id

    null_position = events.recoParticles.startPos.x == -999 # boolean mask of PFP's with undefined position
    null_momentum = events.recoParticles.momentum.x == -999 # boolean mask of PFP's with undefined momentum
    null = np.logical_or(null_position, null_momentum)
    
    if method == "angular":
        separation = vector.angle(events.recoParticles.direction, events.trueParticlesBT.direction) # calculate angular closeness
    else:
        separation = vector.dist(events.recoParticles.startPos, events.trueParticlesBT.startPos) # calculate spatial closeness
    separation = ak.where(null, 9999999, separation) # if direction is undefined, separation is massive (so is never picked as a starting shower)
    ind = ak.local_index(separation, -1) # create index array of separations to use later

    mcID = events.trueParticlesBT.number
    uniqueID = events.trueParticlesBT.GetUniqueParticleNumbers(mcID[pi0_daughters]) # get unique true particle IDs

    if(ak.any(ak.num(uniqueID) == 1)):
        raise Exception("data contains events with reco particles matched to only one photon, did you forget to apply singleMatch filter?")

    # get PFP's which match to the same true particle
    mcp = [mcID == uniqueID[:, i] for i in range(2)]
    
    # mother = [events.trueParticlesBT.GetUniqueParticleNumbers(events.trueParticlesBT.mother[mcp[i]]) for i in range(2)]

    min_sorted_spearation = [ak.min(separation[mcp[i]], -1) for i in range(2)] # get minimum separation sorted by ID

    # select start showers by minimum separation
    indices = [ind[separation == min_sorted_spearation[i]] for i in range(2)]
    start_showers = [ind == indices[i][:, 0] for i in range(2)]
    
    to_merge = np.logical_not(np.logical_or(*start_showers)) # get boolean mask of PFP's to merge
    return start_showers, to_merge


def SignalBackground(events : Master.Data, start_showers : list, to_merge : ak.Array) -> tuple:
    """ Get boolean masks of events classified as signal and background.
        Signal are defined as PFOs which backtrack to the same true particle as the start showers
        Background are all other PFOs

    Args:
        events (Master.Data): events to look at
        start_showers (list): starting photon showers
        to_merge (ak.Array): PFOs to merge 

    Returns:
        tuple: signal mask, background mask and signal mask not accounting for which start shower the signal PFO relates to.
    """
    #* get boolean mask of PFP's which are actual fragments of the starting showers
    start_shower_ID = events.trueParticlesBT.number[np.logical_or(*start_showers)]
    to_merge_ID = events.trueParticlesBT.number[to_merge]
    signal = [start_shower_ID[:, i] == to_merge_ID for i in range(2)] # signal are the PFOs which is a fragment of the ith starting shower

    #* define signal and background
    signal_all = np.logical_or(*signal)
    background = np.logical_not(signal_all) # background is all other PFOs unrelated to the pi0 decay
    return signal, background, signal_all


@Master.timer
def ShowerMerging(events : Master.Data, start_showers : ak.Array, to_merge : ak.Array, quantities : ShowerMergeQuantities, n_merge : int = -1, merge_method : int = 0, make_copy : bool = False, cheat : bool = False) -> Master.Data:
    """ Shower merging algorithm based on reco data.

    Args:
        events (Master.Data): events to study
        start_showers (ak.Array): mask of starting showers
        quantities (ShowerMergeQuantities): quantities to determine which PFOs to merge to which starting shower
        n_merge (int, optional): maximum number of PFOs to merge per event. Defaults to -1 (no maximum)
        merge_method (int, optional): how to compute the new values of momentum/energy
        merge_copy (bool, optional): return a copy of the events class or modify the existing object
        cheat (bool, optional): cheat the merging by passing the signal PFO mask rather than one based on cuts.

    Returns:
        Master.Data: events with PFOs merged to start showers (so PFOs merged are removed from the events and start shower values are updated accordingly)
    """
    def SortByStartingShower(data : ak.Array) -> ak.Array:
        """ Sorts shower quantities in the order the starting showers appear in each event

        Args:
            data (ak.Array): shower quantity (two arrays, one for each starting shower)

        Returns:
            ak.Array: sorted shower quantity
        """
        return ak.concatenate([ak.unflatten(data[i], 1, -1) for i in range(2)], -1) # concantenate to group quantities per event together


    def ClosestQuantity(q : ak.Array, mask : ak.Array) -> ak.Array:
        """ Get the shower which has the smallest geometric quantity wrt to the starting shower
            i.e. if q is the angle of the PFO wrt to starting showers it will return the index
            of the starting shower which it is closest to: 0, 1, or 9999999 if the PFO doesn't
            pass the cuts defined by the mask.

        Args:
            q (ak.Array): quantity
            mask (ak.Array): mask defined by cut based selection

        Returns:
            ak.Array: array of closest start shower indices per PFO
        """
        masked_q = ak.where(mask, q, 9999999) # if value is 9999999 then it never can be chosen as the minimum value (hopefully)
        q_to_merge = ak.argmin(masked_q, -1, keepdims = True)
        # returns:
        # 0 if cloest shower is start shower 0
        # 1 if cloest shower is start shower 1
        # -1 if PFO shouldn't be merged
        return ak.where(ak.min(masked_q, -1, keepdims = True) == 9999999, -1, q_to_merge)


    def ReplaceShowerPairValue(mask : ak.Array, quantity : ak.Array, values : ak.Array) -> ak.Array:
        """ Replaces the shower pair values in a quantitiy for another set.
            ? Should this be made a more generic method in Master.ParticleData?

        Args:
            mask (ak.Array): shower pair mask
            quantity (ak.Array): quantity
            values (ak.Array): shower pair values

        Returns:
            ak.Array: new Array
        """
        new = ak.where(mask[0], values[:, 0], quantity)
        new = ak.where(mask[1], values[:, 1], new)
        return new


    def AssignQuantities(events : Master.Data):
        events.recoParticles._RecoParticleData__momentum =  ReplaceShowerPairValue(start_showers, events.recoParticles.momentum , corrected_data["momentum"])
        events.recoParticles._RecoParticleData__energy =    ReplaceShowerPairValue(start_showers, events.recoParticles.energy   , corrected_data["energy"])
        events.recoParticles._RecoParticleData__direction = ReplaceShowerPairValue(start_showers, events.recoParticles.direction, corrected_data["direction"])
        events.recoParticles._RecoParticleData__nHits = ReplaceShowerPairValue(start_showers, events.recoParticles.nHits, corrected_data["nHits"])
        events.recoParticles._RecoParticleData__nHits_collection = ReplaceShowerPairValue(start_showers, events.recoParticles.nHits_collection, corrected_data["nHits"])

        events.trueParticlesBT._TrueParticleDataBT__sharedHits = ReplaceShowerPairValue(start_showers, events.trueParticlesBT.sharedHits, corrected_data["shared_hits"])
        events.trueParticlesBT._TrueParticleDataBT__hitsInRecoCluster = ReplaceShowerPairValue(start_showers, events.trueParticlesBT.hitsInRecoCluster, corrected_data["reco_cluster_hits"])
        events.trueParticlesBT._TrueParticleDataBT__purity = ReplaceShowerPairValue(start_showers, events.trueParticlesBT.purity, corrected_data["purity"])
        events.trueParticlesBT._TrueParticleDataBT__completeness = ReplaceShowerPairValue(start_showers, events.trueParticlesBT.completeness, corrected_data["completeness"])

        events.trueParticlesBT._TrueParticleDataBT__sharedHits_collection = ReplaceShowerPairValue(start_showers, events.trueParticlesBT.sharedHits_collection, corrected_data["shared_hits_collection"])
        events.trueParticlesBT._TrueParticleDataBT__hitsInRecoCluster_collection = ReplaceShowerPairValue(start_showers, events.trueParticlesBT.hitsInRecoCluster_collection, corrected_data["reco_cluster_hits_collection"])
        events.trueParticlesBT._TrueParticleDataBT__purity_collection = ReplaceShowerPairValue(start_showers, events.trueParticlesBT.purity_collection, corrected_data["purity_collection"])
        events.trueParticlesBT._TrueParticleDataBT__completeness_collection = ReplaceShowerPairValue(start_showers, events.trueParticlesBT.completeness_collection, corrected_data["completeness_collection"])


    #* retrieve quantities and find which start shower is closest to each PFO for each variable
    quantities.Evaluate(events, start_showers)

    if cheat:
        all_showers = np.logical_or(*start_showers)
        s_num = events.trueParticlesBT.number[all_showers]

        signal = [events.trueParticlesBT.number == s_num[:, i] for i in range(2)] # showers we should have merged
        signal = [signal[i] & ~all_showers for i in range(2)] # starting showers are excluded from the signal
        # mask, _, _ = SignalBackground(events, start_showers, to_merge)
        mask = SortByStartingShower(signal)
    else:
        mask = SortByStartingShower(quantities.mask) # PFOs we want to merge after cut based study is done
    

    start_showers_ID = ak.concatenate([events.recoParticles.number[start_showers[i]] for i in range(2)], -1) # id of start showers to recreate mask after merging

    mask = [(np.logical_or(*start_showers) != mask[:, :, i]) & mask[:, :, i] for i in range(2)]
    mask_all = np.logical_or(*mask)
    mask = SortByStartingShower(mask)
    print(f"merging {ak.count(mask_all[mask_all])} PFOs")

    alpha = ClosestQuantity(SortByStartingShower(quantities.alpha), mask) # can use this to determine which starting shower the PFO is closest to in angle
    x = ClosestQuantity(SortByStartingShower(quantities.delta_x), mask) # can use this to determine which starting shower the PFO is closest to in space
    phi = ClosestQuantity(SortByStartingShower(quantities.delta_phi), mask) # can use this to determine which starting shower the PFO direction is most aligned to

    #* figure out which is the common start shower between all variables
    if cheat:
        # for the cheated merging the mask is already sorted by the PFOs that should be merged to said shower
        score_0 = ak.where(mask[:, :, 0], 1, -1) # if we need to merge to shower 0 it has score 1
        score_1 = ak.where(mask[:, :, 1], 2, -1) # if we need to merge to shower 1 it has score 2
        
        # sum the scores
        # note that there are 4 possible outcomes:
        # -2, where the PFO should not be merged (i.e. background)
        # 0, merge PFO to shower 0
        # 1, merge PFO to shower 1
        # 3, merge PFO to both showers, this only happens when the initial shower selection fails and picks two PFOs which backtrack to the saem photon i.e. this can happen when using the reco selection.
        # Note when doing the reco shower merging, it is guarenteed only 1 PFO is merged to 1 shower, even if whe have a case like score == 3.
        scores = score_0 + score_1
        scores = ak.where(scores == -2, -3, scores) # conform to the reco score definition of not merged

    else:
        # if min phi, alpha and x are all the same then merge to that shower
        # if two are the same, merge to the most common shower
        # if none agree (shouldn't be possible)
        #! should replace this with calculating the mode of the scores
        print(ak.concatenate([phi, x, alpha], -1))
        scores = ak.sum(ak.concatenate([phi, x, alpha], -1), -1)
        scores = ak.where(scores == 1, 0, scores) # [1, 0, 0]
        scores = ak.where(scores == 2, 1, scores) # [1, 1, 0]
        scores = ak.where(scores == 3, 1, scores) # [1, 1, 1]

    #* at this point we can check the performance of the shower merging
    event_performance_table = []
    pfo_performance_table = []
    event_performance_table = ShowerMergingEventPerformance(events, start_showers, to_merge, scores)
    pfo_performance_table = ShowerMergingPFOPerformance(events, start_showers, scores)

    def MaxPFOToMerge(quantity, null = 0):
        padded_quantity = ak.pad_none(quantity, ak.max(ak.count(quantity, -1)), -1) # pad jagged array for easier slicing 
        padded_quantity = ak.fill_none(padded_quantity, null, 1) # None -> zero momentum
        return padded_quantity[:, :n_merge] # get max PFO number to merge

    #* get momenta of PFOs to merge
    data_to_merge = {
        "momentum" : events.recoParticles.momentum,
        "nHits" : events.recoParticles.nHits,
        "nHits_collection" : events.recoParticles.nHits_collection,
        "shared_hits" : events.trueParticlesBT.sharedHits,
        "shared_hits_collection" : events.trueParticlesBT.sharedHits_collection,
        "reco_cluster_hits" : events.trueParticlesBT.hitsInRecoCluster,
        "reco_cluster_hits_collection" : events.trueParticlesBT.hitsInRecoCluster_collection,
    }
    if n_merge > 0:
        for d in data_to_merge:
            if d == "momentum":
                null = {"x": 0, "y": 0, "z": 0}
            else:
                null = 0
            data_to_merge[d] = MaxPFOToMerge(data_to_merge[d], null)

        scores = ak.pad_none(scores, ak.max(ak.count(scores, -1)), -1)
        scores = ak.fill_none(scores, -1) # padded scores are -1 i.e. not considered for merging
        scores = scores[:, :n_merge]

    if merge_method == 1:
        data_to_merge["energy"] = events.recoParticles.energy
        if n_merge > 0:
            data_to_merge["energy"] = MaxPFOToMerge(data_to_merge["energy"])

    #* merge all PFOs based on which starting shower they should be merged with i.e. this value is the total amount we correct the shower momenta by
    sorted_data_to_merge = {k : [] for k in data_to_merge}

    for i in range(2): #? is this always 2? What happens when we want to study events with > 1 pi0?
        val = scores == i
        if cheat: val = val | (scores == 3) # see above when determining the score for cheated merging.

        for k in sorted_data_to_merge:
            if k == "momentum":
                null = vector.prod(0, data_to_merge["momentum"])
            else:
                null = 0
            sorted_data = ak.where(val, data_to_merge[k], null)
            sorted_data_to_merge[k].append(ak.sum(sorted_data, -1))

    #* add correction to each starting shower and calculate shower properties
    corrected_data = {}
    
    original_hit_data = {
        "nHits" : events.recoParticles.nHits,
        "nHits_collection" : events.recoParticles.nHits_collection,
        "shared_hits" : events.trueParticlesBT.sharedHits,
        "shared_hits_collection" : events.trueParticlesBT.sharedHits_collection,
        "reco_cluster_hits" : events.trueParticlesBT.hitsInRecoCluster,
        "reco_cluster_hits_collection" : events.trueParticlesBT.hitsInRecoCluster_collection,
    }
    for k in original_hit_data:
        original_hit_data[k] = [original_hit_data[k][start_showers[i]] for i in range(2)]

    #* merge via momentum sum
    if merge_method == 0:
        momentum = [vector.add(events.recoParticles.momentum[start_showers[i]], sorted_data_to_merge["momentum"][i]) for i in range(2)]
        corrected_data["energy"] = ak.concatenate([vector.magnitude(momentum[i]) for i in range(2)], -1)
        corrected_data["direction"] = ak.concatenate([vector.normalize(momentum[i]) for i in range(2)], -1)
        corrected_data["momentum"] = ak.concatenate(momentum, -1)

    #* merge using energy i.e. summing hits, direction calculation is unchanged
    if merge_method == 1:
        momentum = [vector.add(events.recoParticles.momentum[start_showers[i]], sorted_data_to_merge["momentum"][i]) for i in range(2)]
        energy = [events.recoParticles.energy[start_showers[i]] + sorted_data_to_merge["energy"][i] for i in range(2)]
        direction = [vector.normalize(momentum[i]) for i in range(2)]
        momentum = [vector.prod(energy[i], direction[i]) for i in range(2)]

        corrected_data["energy"] = ak.concatenate(energy, -1)
        corrected_data["direction"] = ak.concatenate(direction, -1)
        corrected_data["momentum"] = ak.concatenate(momentum, -1)

    for k, v in sorted_data_to_merge.items():
        if k in ["momentum", "energy"]: continue
        corrected_data[k] = ak.concatenate([original_hit_data[k][i] + v[i] for i in range(2)], -1)

    true_hits = ak.concatenate([events.trueParticlesBT.nHits[start_showers[i]] for i in range(2)], -1)
    true_hits_collection = ak.concatenate([events.trueParticlesBT.nHits_collection[start_showers[i]] for i in range(2)], -1)

    corrected_data["purity"] = corrected_data["shared_hits"] / corrected_data["reco_cluster_hits"]
    corrected_data["completeness"] = corrected_data["shared_hits"] / true_hits
    corrected_data["purity_collection"] = corrected_data["shared_hits_collection"] / corrected_data["reco_cluster_hits_collection"]
    corrected_data["completeness_collection"] = corrected_data["shared_hits_collection"] / true_hits_collection

    # now we need to remove the merged PFOs from the data
    if make_copy:
        merged = events.Filter(returnCopy = True)
        AssignQuantities(merged)
        merged.Filter([~mask_all])
        new_start_showers = [merged.recoParticles.number == start_showers_ID[:, i] for i in range(2)]
        return merged, new_start_showers, event_performance_table, pfo_performance_table
    else:
        AssignQuantities(events)
        events.Filter([~mask_all])
        return [events.recoParticles.number == start_showers_ID[:, i] for i in range(2)], event_performance_table, pfo_performance_table


def Percentage(a, b):
    return  100 * (a - b)/ a

@Master.timer
def EventSelection(events : Master.Data, matchBy : str = "spatial", invertFinal : bool = False):
    """ Applies the event selection for this study and plots a table of how each cut performs.

    Args:
        events (Master.Data): events to apply selection to
        matchBy (str, optional): how to determine start showers. Defaults to "spatial".
        invertFinal (bool, optional): invert last selections boolean mask. Defaults to False.

    """    
    n = [["event selection", "type", "number of events", "percentage of events removed", "percentage of events remaining"]]
    n.append(["no selection", "-",  ak.count(events.eventNum), "-", "-"])

    #################### SELECTION USING TRUTH INFORMATION #################### 
    #* select events with 1 pi0 from the pi+
    mask = LegacyBeamParticleSelection.BeamMCFilter(events)
    events.Filter([mask], [mask])
    truth_mask = LegacyBeamParticleSelection.FinalStatePi0Cut(events)
    events.Filter([], [truth_mask])
    n.append(["beam -> pi0 + X", "truth", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100])

    #* pi+ beam selection!
    mask = LegacyBeamParticleSelection.PiBeamSelection(events)
    events.Filter([mask], [mask])
    n.append(["pi+ beam", "backtracked", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100 - Percentage(n[2][2], ak.count(events.eventNum))])

    #* select only two body decay
    mask = LegacyBeamParticleSelection.DiPhotonCut(events)
    events.Filter([mask], [mask])
    n.append(["diphoton decay", "truth", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100 - Percentage(n[2][2], ak.count(events.eventNum))])

    #################### SELECTION USING RECO INFORMATION #################### 
    #* select events with beam particle
    # events.ApplyBeamFilter()
    mask = LegacyBeamParticleSelection.RecoBeamParticleCut(events)
    events.Filter([mask], [mask])
    n.append(["beam particle", "reco", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100 - Percentage(n[2][2], ak.count(events.eventNum))])

    #* select events with >1 PFP
    mask = LegacyBeamParticleSelection.HasPFO(events)
    events.Filter([mask], [mask])
    n.append(["nPFP > 1", "reco", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100 - Percentage(n[2][2], ak.count(events.eventNum))])

    #################### SELECTION USING BACKTRACKED INFORMATION #################### 
    #* select events with more than one backtracked true particle
    mask = LegacyBeamParticleSelection.HasBacktracked(events)
    events.Filter([mask], [mask])
    n.append(["at least 1 true particle", "backtracked", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100 - Percentage(n[2][2], ak.count(events.eventNum))])

    #* select events with both true pi0 photons
    mask = LegacyBeamParticleSelection.BothPhotonsBacktracked(events)
    label = "both true photons are backtracked"
    if invertFinal is True:
        mask = np.logical_not(mask)
        label = "both true photons are not backtracked"
    events.Filter([mask], [mask])
    n.append([label, "backtracked", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100 - Percentage(n[2][2], ak.count(events.eventNum))])

    print(tabulate(n, tablefmt="latex"))

@Master.timer
def ValidPFOSelection(events : Master.Data):
    """ Applys PFO cuts and plots a table to show the performance of each selection.

    Args:
        events (Master.Data): events to look at

    Methods:
        ApplySelection : applies filters PFOs and updates table to show performance
    """
    # create table
    n = [["PFO selection", "type", "number of PFOs", "percentage of PFOs removed", "percentage of PFOs remaining"]]
    n.append(["no selection", "-",  ak.count(events.recoParticles.number), "-", 100])

    def ApplySelection(mask : ak.Array, mask_name : str, data_type : str):
        events.Filter([mask])
        count = ak.count(events.recoParticles.number)
        n.append([mask_name, data_type, count, Percentage(n[-1][2], count), 100 - Percentage(n[1][2], count)])    

    # selections 
    mask = events.recoParticles.startPos.x != -999
    ApplySelection(mask, "valid start position", "reco")

    mask = events.recoParticles.momentum.x != -999
    ApplySelection(mask, "valid momentum", "reco")

    mask = events.recoParticles.cnnScore != -999
    ApplySelection(mask, "valid CNN score", "reco")

    print(tabulate(n, tablefmt="latex"))


def StartShowerByDistance(events : Master.Data) -> ak.Array:
    """ Select a PFO per photon shower to us as a start for merging.
        Based on PFOs which have the smallest spatial separation.

    Args:
        events (Master.Data): events to look at

    Raises:
        Exception: if all reco PFP's backtracked to the same true particle

    Returns:
        ak.Array: indices of reco particles with the smallest spatial closeness
    """
    null_position = events.recoParticles.startPos.x == -999 # boolean mask of PFP's with undefined position
    null_momentum = events.recoParticles.momentum.x == -999 # boolean mask of PFP's with undefined momentum
    null = np.logical_or(null_position, null_momentum)
    distance_error = vector.dist(events.recoParticles.startPos, events.trueParticlesBT.startPos) # calculate angular closeness
    distance_error = ak.where(null, 999999, distance_error) # if direction is undefined, angler error is massive (so not the best match)
    ind = ak.local_index(distance_error, -1) # create index array of angles to use later

    # get unique true particle numbers per event i.e. the photons which the reco PFP's backtrack to
    mcIndex = events.trueParticlesBT.number
    unqiueIndex = events.trueParticlesBT.GetUniqueParticleNumbers(mcIndex)

    if(ak.any(ak.num(unqiueIndex) == 1)):
        raise Exception("data contains events with reco particles matched to only one photon, did you forget to apply singleMatch filter?")

    # get PFP's which match to the same true particle
    mcp = [mcIndex == unqiueIndex[:, i] for i in range(2)]

    # get the smallest distance error of each sorted PFP's
    distance_error_0 = ak.min(distance_error[mcp[0]], -1)
    distance_error_1 = ak.min(distance_error[mcp[1]], -1)

    # get recoPFP indices which had the smallest spatial closeness
    indices_0 = ind[distance_error == distance_error_0]
    indices_1 = ind[distance_error == distance_error_1]
    start_showers = ak.concatenate([indices_0, indices_1], -1)
    start_showers = start_showers[:, 0:2]
    return start_showers


def Selection(events : Master.Data, event_type : str, pfo_type : str, veto_daughter_pip : bool = True, select_photon_candidates : bool = True) -> list[pd.DataFrame]:
    """ Applies an event selecion and PFO selection.

    Args:
        events (Master.Data): events to look at
        event_type (str): event selection type
        pfo_type (str): pfo selection type
        veto_daughter_pip (bool) : Exlude events which have a daughter pi+ in the final state.
        select_photon_candidates (bool): whether to select events with 2 photon candidates or not

    Returns:
        list[pd.DataFrame]: table of performance metrics for each selection
    """
    match event_type:
        case "cheated":
            mask, event_table = LegacyBeamParticleSelection.CreateLegacyBeamParticleSelection(events, False)
        case "reco":
            mask, event_table = BeamParticleSelection.CreateDefaultSelection(events, None, False, True)
        case _:
            raise Exception(f"event selection type {event_type} not understood.")
    events.Filter([mask], [mask])

    mask, pfo_table = PFOSelection.GoodShowerSelection(events, True)
    events.Filter([mask])

    if pfo_type == "reco":

        mask, photon_candidate_table = PFOSelection.InitialPi0PhotonSelection(events, verbose = False, return_table = True)
        if select_photon_candidates:
            event_mask = ak.num(mask[mask]) == 2 # select events with 2 photon candidates only #TODO handle > 2 candidates
            events.Filter([event_mask], [event_mask])

        #* veto_daughter_pip can take a long time and a lot of resources, so do this cut after the pi0 photon selection
        mask, daughter_pi_candidate_table = PFOSelection.DaughterPiPlusSelection(events, verbose = False, return_table = True)
        if veto_daughter_pip:
            event_mask = ak.num(mask[mask]) == 0 # exclude all events which have a pi+ in the final state
            events.Filter([event_mask], [event_mask])

    if pfo_type == "reco":
        return event_table, pfo_table, photon_candidate_table, daughter_pi_candidate_table
    if pfo_type == "cheated":
        return event_table, pfo_table


def SplitSampleReco(events : Master.Data)-> tuple:
    """ Creates two boolean mask, one which is just the starting showers, the second are all other PFOs in the event.

    Args:
        events (Master.Data): events to look at

    Returns:
        tuple: starting showers mask and all other PFOs mask
    """
    mask = PFOSelection.InitialPi0PhotonSelection(events, False, False)

    #* need to compute start shower masks and to_merge masks to follow the same convention as SplitSample
    index = ak.local_index(mask)
    start_showers = [index == index[mask][:, i] for i in range(2)]
    to_merge = np.logical_not(np.logical_or(*start_showers))

    return start_showers, to_merge


def CountMask(mask : ak.Array, axis : int = None):
    """ Count the number of true entries in a boolean mask.

    Args:
        mask (ak.Array): boolean mask
        axis (int, optional): axis of arrays to count. Defaults to None.

    Returns:
        int or ak.Array: count depending on the axis
    """
    return ak.count(mask[mask], axis = axis)


def ShowerMergingPFOPerformance(events : Master.Data, start_showers : ak.Array, scores : ak.Array):
    """ Calculates performance metrics for how well the shower merging performs on a per PFO basis.

    Args:
        events (Master.Data): evnets to study
        start_showers (ak.Array): starting shower masks
        to_merge (ak.Array): to merge mask
        scores (ak.Array): score attributed to each PFO to merge, see it's definition as ShowerMerging
        quantities (ShowerMergeQuantities): ShowerMergeQuantities for each PFO
    """
    # false negative - showers we should have merged but didn't
    # false positive - showers we merged but shouldn't have
    # true positive - showers we should have merged and did
    # true negative - showers we should have merged but didn't
    # mismatch - of the showers merged, which were assigned to the wrong start shower

    #! not using SignalBackground method here to be explicit when defining signal and background masks
    all_showers = np.logical_or(*start_showers)
    s_num = events.trueParticlesBT.number[all_showers]

    signal = np.logical_or(*[events.trueParticlesBT.number == s_num[:, i] for i in range(2)]) # showers we should have merged
    signal = signal & ~all_showers # starting showers are excluded from the signal
    background = ~signal # showers we shouldn't have merged

    merged = scores != -3 # PFOs actually merged
    not_merged = ~merged

    tp = merged & signal # true positive, signal pfos merged
    tn = ~(merged | signal) # true negative, background not merged

    xor = merged != signal

    fp = xor & (signal == False) # false positive, background PFOs merged
    fn = xor & (signal == True) # false negative, signal PFOs not merged

    # signal_num = events.trueParticlesBT.number[to_merge][signal_all]
    target_num = events.trueParticlesBT.number[tp]
    actual_num = ak.where((scores == 0) | (scores == 3), s_num[:, 0], scores) # see shower merging score determination for cheated merging for why score == 3 is allowed.
    actual_num = ak.where((actual_num == 1) | (actual_num == 3), s_num[:, 1], actual_num)
    actual_num = actual_num[tp]

    actual_num = actual_num[ak.num(actual_num) > 0]
    target_num = target_num[ak.num(target_num) > 0]

    match = actual_num == target_num

    tags = EventSelection.GenerateTrueFinalStateTags(events)

    data = {
        "PFOs" : signal,
        "signal PFOs" : signal,
        "background PFOs" : background,
        "merged PFOs" : merged,
        "signal PFOs merged" : tp,
        "background PFOs merged" : fp,
        "unmerged PFOs" : not_merged,
        "signal PFOs not merged" : fn,
        "background PFOs not merged" : tn,
        "signal PFOs matched" : match
    }
    for d in data:
        print(d)
        if d == "signal PFOs matched":
            data[d] = [data[d]] + [data[d][tags[t].mask[ak.any(tp, -1)]] for t in tags]
        else:
            data[d] = [data[d]] + [data[d][tags[t].mask] for t in tags]

    counts = {}
    for d in data:
        if d == "PFOs":
            counts[d] = [ak.count(t) for t in data[d]]
        if d == "signal PFOs matched":
            counts[d] = [ak.sum(t) for t in data[d]]
        else:
            counts[d] = [CountMask(t) for t in data[d]]

    counts = pd.DataFrame(counts, index = ["all"] + list(tags.keys())).T
    print(counts)
    return counts


def ShowerMergingEventPerformance(events : Master.Data, start_showers : ak.Array, to_merge : ak.Array, scores : ak.Array):
    """ Calculates performance metrics for how well the shower merging performs on a per PFO basis.

    Args:
        events (Master.Data): evnets to study
        start_showers (ak.Array): starting shower masks
        to_merge (ak.Array): to merge mask
        scores (ak.Array): score attributed to each PFO to merge, see it's definition as ShowerMerging
    """
    all_showers = np.logical_or(*start_showers)
    s_num = events.trueParticlesBT.number[all_showers]
    tm_num = events.trueParticlesBT.number[to_merge]

    signal = np.logical_or(*[events.trueParticlesBT.number == s_num[:, i] for i in range(2)]) # showers we should have merged
    signal = signal & ~all_showers # starting showers are excluded from the signal

    background = ~signal # showers we shouldn't have merged

    merged = scores != -3 # PFOs actually merged

    nMerged = CountMask(merged, -1)
    nSignal = CountMask(signal, -1)

    tp = merged & signal # true positive
    nTp = CountMask(tp, -1)

    xor = merged != signal

    fp = xor & (signal == False) # false positive
    nFp = CountMask(fp, -1)
    t = (nFp > 0) & (nTp > 0)

    signal_only = (nFp == 0) & (nTp > 0)
    background_only = (nFp > 0) & (nTp == 0)

    tags = EventSelection.GenerateTrueFinalStateTags(events)

    data = {
        "events after selection" : events.eventNum,
        "events with PFOs to merge" : nSignal,
        "events where PFOs are merged" : nMerged,
        "events where we merge signal" : nTp,
        "events where we merge background" : nFp,
        "events where we merge signal and background" : t,
        "events where we merge only signal" : signal_only,
        "events where we merge only background" : background_only
        }

    for d in data:
        data[d] = [data[d]] + [data[d][tags[t].mask] for t in tags]

    counts = {k : [] for k in data}

    def EventCountNonZero(c, i):
        x = data[c][i]
        counts[c].append(ak.count(x[x > 0]))

    def EventCountMask(c, i):
        counts[c].append(CountMask(data[c][i]))

    for i in range(len(tags) + 1):
        counts["events after selection"].append(ak.count(data["events after selection"][i]))
        EventCountNonZero("events with PFOs to merge", i)
        EventCountNonZero("events where PFOs are merged", i)
        EventCountNonZero("events where we merge signal", i)
        EventCountNonZero("events where we merge background", i)
        EventCountMask("events where we merge signal and background", i)
        EventCountMask("events where we merge only signal", i)
        EventCountMask("events where we merge only background", i)

    counts = pd.DataFrame(counts, index = ["all"] + list(tags.keys())).T
    print(counts)
    return counts