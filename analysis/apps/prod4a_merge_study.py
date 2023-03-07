#!/usr/bin/env python3
"""
Created on: 09/08/2022 14:41

Author: Shyam Bhuller

Description: Process both ROOT and csv data for the shower merging analysis with production 4a MC.
"""
import argparse
import itertools
import os
import warnings

import awkward as ak
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from rich import print
from tabulate import tabulate

from apps import CutOptimization
from python.analysis import Master, Plots, vector, LegacyBeamParticleSelection


def Percentage(a, b):
    return  100 * (a - b)/ a


def BestCut(cuts : pd.DataFrame, q_names : list, type="balanced"):
    """ Select a cut from the cut based scan using signal/background metrics.
        Currently has three types of cut it will pick: high purity, balanced and high efficiency

    Args:
        cuts (pd.DataFrame): dataframe of cuts + metrics
        q_names (list): quantity names
        type (str, optional): type of cut to pick. Defaults to "balanced".

    Returns:
        list : selected cut + metrics
    """
    types = ["balanced", "purity", "efficiency"]
    if type not in types:
        raise Exception(f"cut type must be either {types}")

    print("finding best cut")
    c = cuts[cuts["$\\epsilon$"] > 0] # pick cuts which dont exlude all PFOs

    if type == "balanced":
        # pick a cut which has reasonable signal efficiency, then pick the highest purity cut
        c = c[c["$\\epsilon_{s}$"] > 0.5]
        max_index = c["purity"].idxmax()

    if type == "purity":
        # pick a cut with small background efficiency and > 10% signal efficiency, then pick the highest purity cut
        c = c[c["$\\epsilon_{b}$"] < 0.1]
        c = c[c["$\\epsilon_{s}$"] > 0.1]
        max_index = c["purity"].idxmax()

    if type == "efficiency":
        # pick a cut with > 10% purity and then pick the highest efficiency cut
        c = c[c["purity"] > 0.1]
        max_index = c["$\\epsilon$"].idxmax()
    best_cuts = c[c.index == max_index]
    
    print("Best cut: ")
    print(best_cuts.to_markdown())
    return best_cuts[q_names].values.tolist()[0]


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


    def SaveQuantitiesToCSV(self, signal : ak.Array, background : ak.Array, filename : str = "merge-quantities.csv"):
        """ Saves merge quantities as a pandas dataframe to file.

        Args:
            signal (ak.Array): signal PFO mask
            background (ak.Array): background PFO mask
            filename (str, optional): _description_. Defaults to "merge-quantities.csv".
        """
        #* create dataframe and add all calculated quantities
        for i in range(len(self.selectionVariables)):
            print(self.selectionVariables[i])
            if i == 0:
                df = ak.to_pandas(getattr(self, self.selectionVariables[i]), anonymous=self.selectionVariables[i])
            else:
                df = pd.concat([df, ak.to_pandas(getattr(self, self.selectionVariables[i]), anonymous=self.selectionVariables[i])], 1)
        
        #* add signal and background boolean masks
        df = pd.concat([df, ak.to_pandas(signal, anonymous="signal")], 1)
        df = pd.concat([df, ak.to_pandas([background, background], anonymous="background")], 1)
        df.to_csv(f"{outDir}{filename}")


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


    def PlotQuantities(self, signal : ak.Array, background : ak.Array, min : bool = True):
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

            Plots.PlotHistComparison([b, s], bins=50, xlabel=self.xlabels[i], labels=labels, density=norm, y_scale=scale, annotation=args.dataset)
            if save: Plots.Save(self.selectionVariables[i], outDir)


def GetMin(quantity : ak.Array) -> ak.Array:
    """ Get smallest geometric quantitity wrt to a start shower

    Args:
        quantity (ak.Array): geometric quantity

    Returns:
        ak.Array: smallest quantity per event
    """
    min_q = [ak.unflatten(quantity[i], 1, -1) for i in range(2)]
    min_q = ak.concatenate(min_q, -1)
    return ak.min(min_q, -1)


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
    n.append(["pi+ beam", "backtracked", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100-Percentage(n[2][2], ak.count(events.eventNum))])

    #* select only two body decay
    mask = LegacyBeamParticleSelection.DiPhotonCut(events)
    events.Filter([mask], [mask])
    n.append(["diphoton decay", "truth", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100-Percentage(n[2][2], ak.count(events.eventNum))])

    #################### SELECTION USING RECO INFORMATION #################### 
    #* select events with beam particle
    # events.ApplyBeamFilter()
    mask = LegacyBeamParticleSelection.RecoBeamParticleCut(events)
    events.Filter([mask], [mask])
    n.append(["beam particle", "reco", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100-Percentage(n[2][2], ak.count(events.eventNum))])

    #* select events with >1 PFP
    mask = LegacyBeamParticleSelection.HasPFO(events)
    events.Filter([mask], [mask])
    n.append(["nPFP > 1", "reco", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100-Percentage(n[2][2], ak.count(events.eventNum))])

    #################### SELECTION USING BACKTRACKED INFORMATION #################### 
    #* select events with more than one backtracked true particle
    mask = LegacyBeamParticleSelection.HasBacktracked(events)
    events.Filter([mask], [mask])
    n.append(["at least 1 true particle", "backtracked", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100-Percentage(n[2][2], ak.count(events.eventNum))])

    #* select events with both true pi0 photons
    mask = LegacyBeamParticleSelection.BothPhotonsBacktracked(events)
    label = "both true photons are backtracked"
    if invertFinal is True:
        mask = np.logical_not(mask)
        label = "both true photons are not backtracked"
    events.Filter([mask], [mask])
    n.append([label, "backtracked", ak.count(events.eventNum), Percentage(n[-1][2], ak.count(events.eventNum)), 100-Percentage(n[2][2], ak.count(events.eventNum))])

    print(tabulate(n, tablefmt="latex"))

@Master.timer
def PFOSelection(events : Master.Data):
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
        n.append([mask_name, data_type, count, Percentage(n[-1][2], count), 100-Percentage(n[1][2], count)])    

    # selections 
    mask = events.recoParticles.startPos.x != -999
    ApplySelection(mask, "valid start position", "reco")

    mask = events.recoParticles.momentum.x != -999
    ApplySelection(mask, "valid momentum", "reco")

    mask = events.recoParticles.cnnScore != -999
    ApplySelection(mask, "valid CNN score", "reco")

    print(tabulate(n, tablefmt="latex"))

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
    print(ak.any(ak.count(uniqueID, -1) != 2))

    # get PFP's which match to the same true particle
    mcp = [mcID == uniqueID[:, i] for i in range(2)]
    [print(ak.count(mcp[i], -1)) for i in range(2)]
    print(ak.any(ak.count(separation, -1) != ak.count(mcp[0], -1)))
    print(ak.any(ak.count(separation, -1) != ak.count(mcp[1], -1)))
    
    mother = [events.trueParticlesBT.GetUniqueParticleNumbers(events.trueParticlesBT.mother[mcp[i]]) for i in range(2)]
    print(ak.all(mother[0] == pi0))
    print(ak.all(mother[1] == pi0))
    print(ak.all(events.trueParticles.pdg[events.trueParticles.PrimaryPi0Mask] == 111))

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


def ROOTWorkFlow():
    """ Analysis that can be done when supplied a NTuple file.
        Can calculate:
         - basic quantities
         - geometric quantities for signal/background discrimination
         - pair quantities and shower merging
    """
    events = Master.Data(file, nEvents = args.nEvents[0], start = args.nEvents[1])
    EventSelection(events)
    PFOSelection(events)
    start_showers, to_merge = SplitSample(events, args.matchBy)
    
    #* class to calculate quantities
    q = ShowerMergeQuantities(events, to_merge, args.analysedCuts)

    if args.merge == None:
        signal, background, signal_all = SignalBackground(events, start_showers, to_merge)

        #* plot number of signal and background per event
        nSignal = ak.count(signal_all[signal_all], -1)
        nBackground = ak.count(background[background], -1)

        print(f"Total number of Signal PFOs :{ak.sum(nSignal)}")
        print(f"Total number of background PFOs :{ak.sum(nBackground)}")

        if plotsToMake == "all":
            subDir = "basic_quantities/"
            os.makedirs(outDir+subDir, exist_ok=True)
            labels = ["background", "signal"]
            
            Plots.PlotHist(ak.ravel(nSignal), xlabel="Start shower multiplicity", density=norm, y_scale=scale, annotation=args.dataset)
            if save: Plots.Save("multiplicity", outDir+subDir)

            Plots.PlotHistComparison([nBackground, nSignal], xlabel="Number of PFOs", bins=20, labels=labels, density=norm, y_scale=scale, annotation=args.dataset)
            if save: Plots.Save("nPFO", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(vector.magnitude(events.trueParticles.momentum[events.trueParticles.PrimaryPi0Mask])), nSignal, 50, xlabel = "True $\pi^{0}}$ momentum (GeV)", ylabel="Number of signal PFO", annotation=args.dataset)
            if save: Plots.Save("pi0_p_vs_nPFO_signal", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(vector.magnitude(events.trueParticles.momentum[events.trueParticles.PrimaryPi0Mask])), nBackground, 50, xlabel = "True $\pi^{0}}$ momentum (GeV)", ylabel="Number of background PFO", annotation=args.dataset)
            if save: Plots.Save("pi0_p_vs_nPFO_background", outDir+subDir)

            nbins =  max(nSignal) - min(nSignal)
            Plots.PlotHist(nSignal, xlabel="Number of signal PFOs", bins=np.arange(nbins)-0.5, y_scale=scale, annotation=args.dataset)
            if save: Plots.Save("nPFO_signal", outDir+subDir)

            Plots.PlotHist(nBackground, xlabel="Number of background PFOs", bins=20, y_scale=scale, annotation=args.dataset)
            if save: Plots.Save("nPFO_background", outDir+subDir)

            Plots.PlotHistComparison([ak.ravel(events.recoParticles.energy[to_merge][background]), ak.ravel(events.recoParticles.energy[to_merge][np.logical_or(*signal)])], xlabel="Energy (MeV)", bins=20, labels=labels, density = norm, y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("energy", outDir+subDir)

            Plots.PlotHistComparison([ak.ravel(events.recoParticles.nHits[to_merge][background]), ak.ravel(events.recoParticles.nHits[to_merge][np.logical_or(*signal)])], xlabel="Number of hits", bins=20, labels=labels, density = norm, y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("hits", outDir+subDir)

            Plots.PlotHistComparison([ak.ravel(events.recoParticles.cnnScore[to_merge][background]), ak.ravel(events.recoParticles.cnnScore[to_merge][np.logical_or(*signal)])], xlabel="CNN score", bins=20, labels=labels, density = norm, y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("cnn", outDir+subDir)

            purity = events.trueParticlesBT.purity
            completeness = events.trueParticlesBT.completeness

            start_showers_all = np.logical_or(*start_showers)
            Plots.PlotHist(ak.ravel(purity[start_showers_all]), xlabel="start shower purity", y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("ss-purity", outDir+subDir)
            Plots.PlotHist(ak.ravel(completeness[start_showers_all]), xlabel="start shower completeness", y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("ss-completeness", outDir+subDir)

            Plots.PlotHistComparison([ak.ravel(purity[to_merge][background]), ak.ravel(purity[to_merge][np.logical_or(*signal)])], labels=labels, xlabel="purity", y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("purity", outDir+subDir)
            Plots.PlotHistComparison([ak.ravel(completeness[to_merge][background]), ak.ravel(completeness[to_merge][np.logical_or(*signal)])], labels=labels, xlabel="completeness", y_scale = scale, annotation=args.dataset)
            if save: Plots.Save("completeness", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(purity), ak.ravel(completeness), xlabel="purity", ylabel="completeness")
            if save: Plots.Save("purity_vs_completeness", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(purity[to_merge][np.logical_or(*signal)]), ak.ravel(completeness[to_merge][np.logical_or(*signal)]), bins = 25, xlabel="purity", ylabel="completeness", title = "signal")
            if save: Plots.Save("purity_vs_completeness_s", outDir+subDir)

            Plots.PlotHist2D(ak.ravel(purity[to_merge][background]), ak.ravel(completeness[to_merge][background]), bins = 25, xlabel="purity", ylabel="completeness", title = "background")
            if save: Plots.Save("purity_vs_completeness_b", outDir+subDir)

        #* calculate geometric quantities
        if save is True and plotsToMake is None:
            q.Evaluate(events, start_showers)
            if args.csv is None:
                q.SaveQuantitiesToCSV(signal, background)
            else:
                q.SaveQuantitiesToCSV(signal, background, args.csv)
    else:
        if args.merge == "reco":
            q.bestCut = args.cut_type
            q.to_merge_dir = events.recoParticles.direction
            q.to_merge_pos = events.recoParticles.startPos
            start_showers = ShowerMerging(events, start_showers, q, -1)
            start_showers_all = np.logical_or(*start_showers)

        elif args.merge == "unmerged":
            start_showers_all = np.logical_or(*start_showers)
            #events.Filter([np.logical_or(*start_showers)])

        elif args.merge == "cheat":
            # start_shower_ID = events.trueParticlesBT.number[np.logical_or(*start_showers)]
            # pi0_PFOs = [events.trueParticlesBT.number == start_shower_ID[:, i] for i in range(2)]
            # pi0_PFOs = np.logical_or(*pi0_PFOs)
            # events.Filter([pi0_PFOs])
            events.MergePFOCheat(1)
            start_showers_all = events.trueParticlesBT.mother == ak.flatten(events.trueParticles.number[events.trueParticles.PrimaryPi0Mask])

        else:
            raise Exception("Don't understand the merge type")

        pairs = Master.ShowerPairs(events, start_showers_all)
        pairs.SaveToCSV(args.outDir + args.csv)
        # p = Master.CalculateQuantities(s, True)
        # PairQuantitiesToCSV(p)


def ShowerMergingCriteria(q : ShowerMergeQuantities):
    """ Performs a cut based scan on various criteria that can be used for shower merging

    Args:
        q (ShowerMergeQuantities): quantities to perform cut based scan on
    """
    def Spinner(counter : int, spinner="lines") -> str:
        """ Janky spinner, cause why not?

        Args:
            counter (int): iteration
            spinner (str, optional): type of spinner. Defaults to "lines".

        Returns:
            str: string to print at this interval
        """
        spinners = {
            "lines" : "-\|/",
            "box"   : "⠦⠆⠖⠒⠲⠰⠴⠤",
        }
        return spinners[spinner][counter % len(spinners[spinner])]

    min_val = [] # min range of each variable
    max_val = [] # max range
    for i in range(len(q.selectionVariables)):
        min_val.append(0)
        max_val.append(ak.max(getattr(q, q.selectionVariables[i])))

    min_val = np.array(min_val)
    max_val = np.array(max_val)

    values = np.linspace(min_val+(0.1*max_val), max_val-(0.1*max_val), 3, True) # values that are used to create combinations of cuts to optimize
    metric_labels = ["s", "b", "s/b", "$s\\sqrt{b}$", "purity", "$\\epsilon_{s}$", "$\\epsilon_{b}$", "$\\epsilon$"] # performance metrics to choose cuts #? add purity*efficiency?

    #* create input data strutcure
    counter = 0

    cuts = []
    for v in q.selectionVariables:
        operator = CutOptimization.Operator.GREATER if v == "cnn" else CutOptimization.Operator.LESS
        cuts.append(CutOptimization.Cuts(v, operator, None))

    print("list of cut types:")
    print(cuts)

    if args.csv is None:
        output_path = f"{outDir}analysedCuts.csv"
    else:
        output_path = f"{outDir}{args.csv}"

    #* loop through all combination of values for each parameter and optmize the final cut
    for initial_cuts in itertools.product(*values.T):
        for i in range(len(cuts)):
            cuts[i].value = initial_cuts[i]
        cutOptimization = CutOptimization.OptimizeSingleCut(q, cuts, False)
        c, m = cutOptimization.Optimize(10, CutOptimization.MaxSRootBRatio) # scan over 10 bins and optimize cut by looking for max s/sqrt(b)

        o = [c[i] + m[i] for i in range(len(c))] # combine output
        o = pd.DataFrame(o, columns = q.selectionVariables + metric_labels)
        o.to_csv(output_path, mode = "a", header = not os.path.exists(output_path))

        counter += 1
        end = '\n' if counter == 0 else '\r'
        print(f" {Spinner(counter, 'box')} progess: {counter/(len(values)**len(initial_cuts))*100:.3f}% | {counter} | {len(values)**len(initial_cuts)}", end=end)


def CSVWorkFlow():
    """ Analysis that can be done with the csv files produced by this program
    """
    q = ShowerMergeQuantities(analysedCuts=args.analysedCuts) # can apply cuts to shower quantities
    q.LoadQuantitiesFromCSV(file)
    if args.cut is True:
        ShowerMergingCriteria(q)
        return
    if plotsToMake in ["all", "quantities"]:
        q.PlotQuantities(q.signal, q.background, False)


@Master.timer
def ShowerMerging(events : Master.Data, start_showers : ak.Array, quantities : ShowerMergeQuantities, n_merge : int = -1, merge_method : int = 0, make_copy : bool = False) -> Master.Data:
    """ Shower merging algorithm based on reco data.

    Args:
        events (Master.Data): events to study
        start_showers (ak.Array): mask of starting showers
        quantities (ShowerMergeQuantities): quantities to determine which PFOs to merge to which starting shower
        n_merge (int, optional): maximum number of PFOs to merge per event. Defaults to -1 (no maximum)
        merge_method (int, optional): how to compute the new values of momentum/energy
        merge_copy (bool, optional): return a copy of the events class or modify the existing object

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
        q_to_merge = ak.argmin(masked_q, -1, keepdims=True)
        # returns:
        # 0 if cloest shower is start shower 0
        # 1 if cloest shower is start shower 1
        # -1 if PFO shouldn't be merged
        return ak.where(ak.min(masked_q, -1, keepdims=True) == 9999999, -1, q_to_merge)

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
        events.recoParticles._RecoParticleData__momentum = ReplaceShowerPairValue(start_showers, events.recoParticles.momentum, momentum)
        events.recoParticles._RecoParticleData__energy = ReplaceShowerPairValue(start_showers, events.recoParticles.energy, energy)
        events.recoParticles._RecoParticleData__direction = ReplaceShowerPairValue(start_showers, events.recoParticles.direction, direction)

    #* retrieve quantities and find which start shower is closest to each PFO for each variable
    quantities.Evaluate(events, start_showers)
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
    # if min phi, alpha and x are all the same then merge to that shower
    # if two are the same, merge to the most common shower
    # if none agree (shouldn't be possible)
    #! should replace this with calculating the mode of the scores
    print(ak.concatenate([phi, x, alpha], -1))
    scores = ak.sum(ak.concatenate([phi, x, alpha], -1), -1)
    scores = ak.where(scores == 1, 0, scores) # [1, 0, 0]
    scores = ak.where(scores == 2, 1, scores) # [1, 1, 0]
    scores = ak.where(scores == 3, 1, scores) # [1, 1, 1]

    #* get momenta of PFOs to merge
    momentum = events.recoParticles.momentum
    if merge_method == 1:
        energy = events.recoParticles.energy
    if n_merge > 0:
        momentum = ak.pad_none(momentum, ak.max(ak.count(momentum, -1)), -1) # pad jagged array for easier slicing
        momentum = ak.fill_none(momentum, {"x": 0, "y": 0, "z": 0}, 1) # None -> zero momentum
        momentum = momentum[:, :n_merge] # get max PFO number to merge

        if merge_method == 1:
            energy = ak.pad_none(energy, ak.max(ak.count(energy, -1)), -1) 
            energy = ak.fill_none(energy, 0, 1)
            energy = energy[:, :n_merge]

        scores = ak.pad_none(scores, ak.max(ak.count(scores, -1)), -1)
        scores = ak.fill_none(scores, -1) # padded scores are -1 i.e. not considered for merging
        scores = scores[:, :n_merge]
 
    #* merge all PFOs based on which starting shower they should be merged with i.e. this value is the total amount we correct the shower momenta by
    sorted_momentum_to_merge = []
    if merge_method == 1:
        sorted_energy_to_merge = []
    for i in range(2): #? is this always 2? What happens when we want to study events with > 1 pi0?
        val = scores == i
        sorted_momenta = ak.where(val, momentum, vector.prod(0, momentum))
        sorted_momentum_to_merge.append(ak.sum(sorted_momenta, -1))
        if merge_method == 1:
            sorted_energy = ak.where(val, energy, 0)
            sorted_energy_to_merge.append(ak.sum(sorted_energy, -1))

    #* add correction to each starting shower and calculate shower properties
    #* merge via momentum sum
    if merge_method == 0:
        momentum = [vector.add(events.recoParticles.momentum[start_showers[i]], sorted_momentum_to_merge[i]) for i in range(2)]
        energy = ak.concatenate([vector.magnitude(momentum[i]) for i in range(2)], -1)
        direction = ak.concatenate([vector.normalize(momentum[i]) for i in range(2)], -1)
        momentum = ak.concatenate(momentum, -1)

    #* merge using energy i.e. summing hits, direction calculation is unchanged
    if merge_method == 1:
        momentum = [vector.add(events.recoParticles.momentum[start_showers[i]], sorted_momentum_to_merge[i]) for i in range(2)]
        energy = [events.recoParticles.energy[start_showers[i]] + sorted_energy_to_merge[i] for i in range(2)]
        direction = [vector.normalize(momentum[i]) for i in range(2)]
        momentum = [vector.prod(energy[i], direction[i]) for i in range(2)]

        energy = ak.concatenate(energy, -1)
        direction = ak.concatenate(direction, -1)
        momentum = ak.concatenate(momentum, -1)

    # now we need to remove the merged PFOs from the data
    if make_copy:
        merged = events.Filter(returnCopy = True)
        AssignQuantities(merged)
        merged.Filter([~mask_all])
        new_start_showers = [merged.recoParticles.number == start_showers_ID[:, i] for i in range(2)]
        return merged, new_start_showers
    else:
        AssignQuantities(events)
        events.Filter([~mask_all])
        return [events.recoParticles.number == start_showers_ID[:, i] for i in range(2)]


@Master.timer
def main():
    plt.style.use('ggplot')
    plt.rcParams.update({'patch.linewidth': 1})
    plt.rcParams.update({'font.size': 10})
    if save:
        os.makedirs(outDir, exist_ok = True)
    fileFormat = file.split('.')[-1]
    if fileFormat == "root":
        ROOTWorkFlow()
    if fileFormat == "csv":
        CSVWorkFlow()

if __name__ == "__main__":
    example_usage = """Example Uasge:
    Open a ROOT file and plot basic quantities:
        prod4a_merge_study.py <ROOT file> -s -p <plot type> -d <out directory> -n

    Open a ROOT file and save merge quantities to file:
        prod4a_merge_study.py <ROOT file> -s -d <out directory>

    Open a csv file with merge quantities and plot them:
        prod4a_merge_study.py <merge quantity csv> -p <plot type> -s -d <out directory> -n

    Open a csv file with merge quantities and scan for cut values:
        prod4a_merge_study.py <merge quantity csv> -p <plot type> -s -d <out directory> -c

    Open a ROOT file and csv with list of cuts and merge PFOs based on the cut type:
        prod4a_merge_study.py <ROOT file> --cuts <cuts csv> --cut-type <cut type> -m reco -s -o <output filename> -d <out directory> -a

    Open a ROOT file, merge PFOs based on truth information and save shower pair quantities to file:
        prod4a_merge_study.py <ROOT file> -m cheat -s -o <output filename> -d <out directory>
    """

    parser = argparse.ArgumentParser(description = "Shower merging study for beamMC, plots quantities used to decide which showers to merge.", formatter_class = argparse.RawDescriptionHelpFormatter, epilog = example_usage)
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-e", "--events", dest="nEvents", type=int, nargs=2, default=[-1, 0], help="number of events to analyse and number to skip (-1 is all)")
    parser.add_argument("-n", "--normalize", dest="norm", action="store_true", help="normalise plots to compare shape")
    parser.add_argument("-l" "--log", dest="log", action="store_true", help="plot y axis on log scale")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="prod4a_merge_study/", help="directory to save plots")
    parser.add_argument("-p", "--plots", dest="plotsToMake", type=str, choices=["all", "quantities", "multiplicity", "nPFO", "2D"], help="what plots we want to make")
    parser.add_argument("-c", "--cutScan", dest="cut", action="store_true", help="whether to do a cut based scan")
    parser.add_argument("--start-showers", dest="matchBy", type=str, choices=["angular", "spatial"], default="spatial", help="method to detemine start showers")
    parser.add_argument("--cuts", dest="analysedCuts", default=None, type=str, help="data produced by ShowerMergingCriteria i.e. use the -c option")
    parser.add_argument("-a", "--apply-cuts", dest="applyCuts", action="store_true", help="apply cuts to shower merge quantities")
    parser.add_argument("-m", "--merge", dest="merge", type=str, choices=["unmerged", "reco", "cheat", None], default=None, help="Do shower merging (cuts required)")
    parser.add_argument("--cut-type", dest="cut_type", type=str, choices=["purity", "balanced", "efficiency"], default="balanced", help="type of cut to pick from cut scan.")
    parser.add_argument("-o", "--out-csv", dest="csv", type=str, default=None, help="output csv filename (will default to whatever type of data is produced)")
    parser.add_argument("--annotation", dest="dataset", type=str, help="annotation for plots.")
    args = parser.parse_args() #! run in command line
    print(vars(args))
    
    file = args.file
    save = args.save
    outDir = args.outDir
    plotsToMake = args.plotsToMake
    norm = args.norm

    if plotsToMake is not None:
        print("making directory")
        os.makedirs(outDir, exist_ok=True)
        print(f"made {outDir}")

    if args.csv is not None: args.csv += ".csv"
    
    if args.log is True:
        scale = "log"
    else:
        scale = "linear"

    main()