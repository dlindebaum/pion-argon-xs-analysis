import os
import argparse
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate import tabulate
import itertools

import Master
import vector
import Plots


class ShowerMergeQuantities:
    xlabels = [
        "$\delta\phi$ (rad)",
        "$\delta x$ (cm)",
        "$\delta x_{l}$ (cm)",
        "$\delta x_{t}$ (cm)",
        "$\delta e$ (cm)",
        "$\\alpha$ (rad)",
        "$\delta\\alpha$ (rad)"
    ]
    names = [
        "delta_phi",
        "delta_x",
        "delta_xl",
        "delta_xt",
        "delta_e",
        "alpha",
        "delta_alpha"
    ]
    selectionVariables = [
        "delta_phi",
        "delta_x",
        "delta_xl",
        "delta_xt",
        "alpha",        
    ]
    def __init__(self, events : Master.Data = None, to_merge = None):
        if events:
            #* collect positions and directions of PFOs
            self.to_merge_dir = events.recoParticles.direction[to_merge]
            self.to_merge_pos = events.recoParticles.startPos[to_merge]

            self.null = np.logical_or(self.to_merge_dir.x != -999, self.to_merge_pos.x != -999)
            self.to_merge_dir = self.to_merge_dir[self.null]
            self.to_merge_pos = self.to_merge_pos[self.null]

    @Master.timer
    def Evaluate(self, events : Master.Data, start_showers):
        """ Calculate quantities which may help select PFOs to merge

        Args:
            events (Master.Data): events to study
            start_showers (_type_): initial showers to merge to
            start_shower_pos (_type_): _description_
            start_shower_dir (_type_): _description_
        """
        start_shower_pos = events.recoParticles.startPos[start_showers]
        start_shower_dir = events.recoParticles.direction[start_showers]
        start_shower_length = events.recoParticles.showerLength[start_showers]
        start_shower_cone_angle = events.recoParticles.showerConeAngle[start_showers]
        start_shower_end = vector.add(start_shower_pos, vector.prod(start_shower_length, start_shower_dir))

        self.delta_phi = [vector.angle(start_shower_dir[:, i], self.to_merge_dir) for i in range(2)]
        displacement = [vector.sub(self.to_merge_pos, start_shower_pos[:, i]) for i in range(2)]
        self.alpha = [vector.angle(displacement[i], start_shower_dir[:, i]) for i in range(2)]
        self.delta_alpha = [self.alpha[i] - start_shower_cone_angle[:, i] for i in range(2)]
        self.delta_x = [vector.dist(start_shower_pos[:, i], self.to_merge_pos) for i in range(2)]
        self.delta_xl = [self.delta_x[i] * np.abs(np.cos(self.alpha[i])) for i in range(2)]
        self.delta_xt = [self.delta_x[i] * np.abs(np.sin(self.alpha[i])) for i in range(2)]
        self.delta_e = [vector.dist(start_shower_end[:, i], self.to_merge_pos) for i in range(2)]

    def SaveQuantitiesToCSV(self, signal, background, filename : str = "merge-quantities.csv"):
        for i in range(len(self.names)):
            if i == 0:
                df = ak.to_pandas(getattr(self, self.names[i]), anonymous=self.names[i])
            else:
                df = pd.concat([df, ak.to_pandas(getattr(self, self.names[i]), anonymous=self.names[i])], 1)
        df = pd.concat([df, ak.to_pandas(signal, anonymous="signal")], 1)
        df = pd.concat([df, ak.to_pandas([background, background], anonymous="background")], 1)
        df.to_csv(f"{outDir}/{filename}")

    def LoadQuantitiesToCSV(self, filename):
        data = pd.read_csv(filename)
        for n in self.names:
            d = ak.Array(data[n].values.tolist())
            setattr(self, n, ak.unflatten(d, ak.count(d)//2))

        signal = ak.Array(data["signal"].values.tolist())
        background = ak.Array(data["background"].values.tolist())

        self.signal = ak.unflatten(signal, ak.count(signal)//2)
        self.background = ak.unflatten(background, ak.count(background)//2)

    def SignalBackgroundRatio(self, signal, background, printMetrics=False):
        n_signal = ak.count(signal)
        n_background = ak.count(background)
        if n_background == 0:
            return -1, -1
        sb = n_signal/n_background
        srootb = n_signal/np.sqrt(n_background)
        if printMetrics is True:
            print(f"signal: {n_signal}")
            print(f"background: {n_background}")
            print(f"s/b {sb}")
            print(f"s/sqrt(b) {srootb}")
        return sb, srootb

    def BruteForceScan(self):
        initial_signal = ak.flatten(self.signal)[ak.flatten(self.signal)]
        initial_background = ak.flatten(self.background)[ak.flatten(self.background)]
        self.SignalBackgroundRatio(initial_signal, initial_background)

        # first pick a baseline value for cuts
        # define an initial range of values to cut
        initial_cuts = []
        for var in self.selectionVariables:
            minVal = ak.min(getattr(self, var))
            maxVal = ak.max(getattr(self, var))
            initial_cuts.append(np.linspace(minVal, maxVal, 10))
        print(tabulate((self.selectionVariables,initial_cuts), tablefmt="fancy_grid"))

        ratios = []
        rootRatios = []
        permutations = itertools.product(*initial_cuts)
        for p in permutations:
            masks = []
            for j in range(len(self.selectionVariables)):
                masks.append(ak.flatten(getattr(self, self.selectionVariables[j])) < p[j])
            for i in range(len(masks)):
                if i == 0:
                    passed = masks[i]
                else:
                    passed = np.logical_and(passed, masks[i])

            signal = ak.flatten(self.signal)[passed]
            background = ak.flatten(self.background)[passed]
            sb, srootb = self.SignalBackgroundRatio(signal[signal], background[background])
            ratios.append(sb)
            rootRatios.append(srootb)
        plt.figure()
        plt.plot(ratios)
        plt.figure()
        plt.plot(rootRatios)

    @Master.timer
    def PlotQuantities(self, signal : ak.Array, background : ak.Array):
        """ Plot geometric quantities to cosndier for shower merging

        Args:
            phi (ak.Array): angle between PFO and shower
            x (ak.Array): distance between PFO and shower start 
            xl (ak.Array): distance along axis of shower
            xt (ak.Array): distance transverse to shower
            e (ak.Array): distance between PFO and shower end
            alpha (ak.array): angle from axis of start shower to x
            dalpha (ak.Array): difference betweeen alpha and cone angle
            signal (ak.Array): signal PFOs
            background (ak.Array): background PFOs
        """
        #* plot and save
        #labels = ["$b_{0}$", "$b_{1}$", "$s_{0}$", "$s_{1}$"]
        labels = ["background", "signal"]
        for i in range(len(self.names)):
            data = getattr(self, self.names[i])
            print(data)
            #* collect signal PFOs
            s = ak.ravel([data[j][signal[j]] for j in range(2)])

            #* collect background PFOs
            b = ak.ravel([data[j][background[0]] for j in range(2)])

            #Plots.PlotHistComparison([ak.ravel((b+s)[j]) for j in range(4)], bins=50, xlabel=self.xlabels[i], labels=labels, density=norm, y_scale=scale)
            Plots.PlotHistComparison([b, s], bins=50, xlabel=self.xlabels[i], labels=labels, density=norm, y_scale=scale)
            if save: Plots.Save(self.names[i], outDir)


    def Plot2DQuantities(self, signal, background):
        background = background[0]
        labels = ["background", "signal"]
        colours = ["blue", "red"]

        legend = []
        for i in range(len(labels)):
            legend.append(mpatches.Patch(color=colours[i], label=labels[i]))

        s_alpha = ak.ravel([self.alpha[i][signal[i]] for i in range(2)])
        b_alpha = ak.ravel([self.alpha[i][background] for i in range(2)])
        s_x = ak.ravel([self.delta_x[i][signal[i]] for i in range(2)])
        b_x = ak.ravel([self.delta_x[i][background] for i in range(2)])
        s_xl = ak.ravel([self.delta_xl[i][signal[i]] for i in range(2)])
        b_xl = ak.ravel([self.delta_xl[i][background] for i in range(2)])
        s_xt = ak.ravel([self.delta_xt[i][signal[i]] for i in range(2)])
        b_xt = ak.ravel([self.delta_xt[i][background] for i in range(2)])
        s_phi = ak.ravel([self.delta_phi[i][signal[i]] for i in range(2)])
        b_phi = ak.ravel([self.delta_phi[i][background] for i in range(2)])

        PlotContour(s_alpha, s_x, b_alpha, b_x, colours, labels, legend, self.xlabels[5], self.xlabels[1])
        if save: Plots.Save(f"{self.names[5]}-{self.names[1]}", outDir)

        PlotContour(s_xl, s_xt, b_xl, b_xt, colours, labels, legend, self.xlabels[2], self.xlabels[3])
        if save: Plots.Save(f"{self.names[2]}-{self.names[3]}", outDir)

        PlotContour(s_alpha, s_phi, b_alpha, b_phi, colours, labels, legend, self.xlabels[5], self.xlabels[0])
        if save: Plots.Save(f"{self.names[5]}-{self.names[0]}", outDir)



def CutBasedScan(quantities : ShowerMergeQuantities, initial_cuts, startPoint=0, stepSize=10, plot=False, plotFinal=False):
    if len(initial_cuts) != len(quantities.selectionVariables):
        raise Exception("lenth of initial_cuts must be equal to number of cut variables")

    initial_signal = ak.flatten(quantities.signal)[ak.flatten(quantities.signal)]
    initial_background = ak.flatten(quantities.background)[ak.flatten(quantities.background)]
    quantities.SignalBackgroundRatio(initial_signal, initial_background, True)

    final_cuts = list(initial_cuts)
    nm1 = -1
    while nm1 != startPoint:
        if nm1 == -1: nm1 = startPoint # define the variable to produce n-1 plot
    
        #* first apply all but one cut
        mask = [] # mask of entries passing the cuts
        for i in range(len(final_cuts)):
            if i == nm1: continue # dont cut on nth variable
            if len(mask) == 0:
                mask = ak.flatten(getattr(quantities, quantities.selectionVariables[i])) < final_cuts[i]
            else:
                mask = np.logical_and(mask, ak.flatten(getattr(quantities, quantities.selectionVariables[i])) < final_cuts[i])

        #* make n-1 plot of variable
        sig = ak.flatten(quantities.signal)[mask]
        bkg = ak.flatten(quantities.background)[mask]
        ns_i = ak.count(sig[sig])
        nb_i = ak.count(bkg[bkg])
        var = ak.flatten(getattr(quantities, quantities.selectionVariables[nm1]))[mask]

        #* cut variable at various points, calculate s/b, s/rootb
        cuts = np.linspace(ak.min(var), ak.max(var), stepSize)
    
        metrics = {
            "s/b" : np.empty(shape=len(cuts)),
            "s/rootb" : np.empty(shape=len(cuts)),
            "signal efficiency" : np.empty(shape=len(cuts)),
            "background rejection" : np.empty(shape=len(cuts))
        }
        for i in range(len(cuts)):
            mask = var < cuts[i]
            ns = ak.count(sig[mask][sig[mask]])
            nb = ak.count(bkg[mask][bkg[mask]])
            se = ns / ns_i
            br = 1 - (nb / nb_i)
            if(nb > 0):
                sb = ns / nb
                srootb = ns / np.sqrt(nb)
            else:
                sb = -1
                srootb = -1
            metrics["signal efficiency"][i] = se
            metrics["background rejection"][i] = br
            metrics["s/b"][i] = sb
            metrics["s/rootb"][i] = srootb
            print(f"signal: {ns}, background: {nb}, s/b: {sb:.3f}, s/sqrtb: {srootb:.3f}, signal efficiency: {se:.3f}, background rejection : {br:.3f}")
        if plot:
            Plots.PlotHistComparison([var[bkg], var[sig]], labels=["background", "signal"], bins=25, xlabel=quantities.selectionVariables[nm1], density=False)
            Plots.Plot(cuts[metrics["s/b"]>0], metrics["s/b"][metrics["s/b"]>0], quantities.selectionVariables[nm1], "$s/b$")
            Plots.Plot(cuts[metrics["s/rootb"]>0], metrics["s/rootb"][metrics["s/rootb"]>0], quantities.selectionVariables[nm1], "$s/\\sqrt{b}$")
            Plots.Plot(cuts, metrics["signal efficiency"], quantities.selectionVariables[nm1], label="signal efficiency")
            Plots.Plot(cuts, metrics["background rejection"], quantities.selectionVariables[nm1], label="background rejection", newFigure=False)
            Plots.Plot(metrics["signal efficiency"], metrics["background rejection"], "signal efficiency", "background rejection")
        #* criteria to pick a starting shower should be something like, highest s/b or se == br
        #? make multiple criteria?
        best_cut = cuts[np.argmax(metrics["s/rootb"])]
        print(f"best cut for {quantities.selectionVariables[nm1]} is {best_cut}")
        final_cuts[nm1] = best_cut # this is the new best cut for this variable
        
        nm1 = (nm1 + 1)%len(quantities.selectionVariables)
        print(f"next look at : {quantities.selectionVariables[nm1]}")
    print(f"final cuts are: {final_cuts}")
    #* cut one final time to look at the results
    mask = [] # mask of entries passing the cuts
    for i in range(len(final_cuts)):
        if len(mask) == 0:
            mask = ak.flatten(getattr(quantities, quantities.selectionVariables[i])) < final_cuts[i]
        else:
            mask = np.logical_and(mask, ak.flatten(getattr(quantities, quantities.selectionVariables[i])) < final_cuts[i])
    print(f"number of PFOs after cuts: {ak.count(mask[mask])}")

    if plotFinal is True:
        selected_quantities = []
        for i in range(len(quantities.selectionVariables)):
            selected_quantities.append(ak.flatten(getattr(quantities, quantities.selectionVariables[i]))[mask])
        final_signal = ak.flatten(quantities.signal)[mask]
        final_background = ak.flatten(quantities.background)[mask]
        labels = ["background", "signal"]
        for i in range(len(quantities.selectionVariables)):
            data = selected_quantities[i]
            print(data)
            Plots.PlotHistComparison([data[final_background], data[final_signal]], bins=50, xlabel=quantities.selectionVariables[i], labels=labels, density=norm, y_scale=scale)
            if save: Plots.Save(quantities.names[i], outDir)
    return final_cuts


def PlotContour(xs, ys, xb, yb, colours, labels, legend, xlabel, ylabel):
    counts, xbins, ybins = np.histogram2d(xs, ys)
    contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=colours[1], label=labels[1])
    plt.clabel(contours, inline=True, fontsize=8)
    counts, xbins, ybins = np.histogram2d(xb, yb)
    contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=colours[0], label=labels[0])
    plt.clabel(contours, inline=True, fontsize=8)
    plt.legend(handles=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def GetMin(quantity : ak.Array):
    """ Get smallest geometric quantitity wrt to a start shower

    Args:
        quantity (ak.Array): geometric quantity

    Returns:
        ak.Array: smallest quantity per event
    """
    min_q = [ak.unflatten(quantity[i], 1, -1) for i in range(2)]
    min_q = ak.concatenate(min_q, -1)
    return ak.min(min_q, -1)


def Percentage(a, b):
    return  100 * (a - b)/ a

@Master.timer
def EventSelection(events : Master.Data):
    n = [["event selection", "number of events", "percentage of events removed"]]
    n.append(["no selection",  ak.count(events.eventNum), "-"])

    events.ApplyBeamFilter() # apply beam filter if possible
    n.append(["beam particle", ak.count(events.eventNum), Percentage(n[-1][1], ak.count(events.eventNum))])

    Master.BeamMCFilter(events, returnCopy=False)
    n.append(["single pi0", ak.count(events.eventNum), Percentage(n[-1][1], ak.count(events.eventNum))])

    f = Master.NPFPMask(events, -1)
    events.Filter([f], [f]) # filter events with mask
    n.append(["nPFP > 1", ak.count(events.eventNum), Percentage(n[-1][1], ak.count(events.eventNum))])

    f = Master.Pi0TwoBodyDecayMask(events)
    events.Filter([f], [f]) # filter events with mask
    n.append(["diphoton decay", ak.count(events.eventNum), Percentage(n[-1][1], ak.count(events.eventNum))])

    unique = events.trueParticlesBT.GetUniqueParticleNumbers(events.trueParticlesBT.number)
    f = ak.num(unique) > 1
    events.Filter([f], [f])
    n.append(["nUnique > 1", ak.count(events.eventNum), Percentage(n[-1][1], ak.count(events.eventNum))])

    pi0 = ak.flatten(events.trueParticles.number[events.trueParticles.PrimaryPi0Mask])
    f = events.trueParticlesBT.mother == pi0
    not_null = np.logical_or(events.recoParticles.startPos.x != -999, events.recoParticles.direction.x != -999)
    f = np.logical_and(f, not_null)
    daughters = events.trueParticlesBT.number[f]
    unique_daughters = events.trueParticlesBT.GetUniqueParticleNumbers(daughters)
    unique_daughters = ak.count(unique_daughters, -1)
    f = unique_daughters == 2
    events.Filter([f], [f])
    n.append(["unique pi0 daughters == 2", ak.count(events.eventNum), Percentage(n[-1][1], ak.count(events.eventNum))])

    start_showers = GetStartShowers(events, args.matchBy)

    start_shower_pos = events.recoParticles.startPos[start_showers]
    start_shower_dir = events.recoParticles.direction[start_showers]
    f = ak.all(np.logical_or(start_shower_dir.x != -999, start_shower_pos.x != -999), -1) # ignore null directions/positions for starting showers

    events.Filter([f], [f])
    n.append(["valid start shower", ak.count(events.eventNum), Percentage(n[-1][1], ak.count(events.eventNum))])
    start_showers = start_showers[f]

    print(tabulate(n, tablefmt="fancy_grid"))

    return start_showers


@Master.timer
def GetStartShowers(events : Master.Data, method="spatial"):
    #TODO fix a bug where occasionally a starting shower is a daughter of the pi0.
    """ Select starting showers to merge for the pi0 decay.
        The starting showers are guarenteed to originate 
        from the pi0 decay (using truth information).

    Args:
        events (Master.Data): events to look at

    Raises:
        Exception: if all reco PFP's backtracked to the same true particle

    Returns:
        ak.Array: indices of reco particles with the smallest angular closeness
    """
    global mcID, uniqueID, mcp, mother, pi0, indices
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
    [print(ak.count(indices[i], -1)) for i in range(2)]
    indices = [ak.unflatten(indices[i][:, 0], 1, -1) for i in range(2)]
    

    start_showers = ak.concatenate(indices, -1)
    particle = events.trueParticlesBT.pdg[start_showers]
    Plots.PlotBar(ak.ravel(particle))
    #start_showers = start_showers[:, 0:2] # only get the smallest separations, not the second smallest etc.
    return start_showers 


def StartShowerByDistance(events : Master.Data):
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
    events = Master.Data(file, includeBackTrackedMC=True)
    start_showers = EventSelection(events)

    #* get boolean mask of PFP's to merge
    index = ak.local_index(events.recoParticles.energy)
    to_merge = [ ak.where(index == start_showers[:, i], False, True) for i in range(2) ]
    to_merge = np.logical_and(*to_merge)

    #* get boolean mask of PFP's which are actual fragments of the starting showers
    start_shower_ID = events.trueParticlesBT.number[start_showers]
    to_merge_ID = events.trueParticlesBT.number[to_merge]
    signal = [to_merge_ID == start_shower_ID[:, i] for i in range(2)] # signal are the PFOs which is a fragment of the ith starting shower
    nSignal = [ak.count(signal[i][signal[i]], -1) for i in range(2)]

    if plotsToMake in ["all", "multiplicity"]:
        #* plot shower multiplicity
        Plots.PlotHist(ak.ravel(nSignal), xlabel="start shower multiplicity")
        if save: Plots.Save("shower-multiplicity", outDir)

    events.recoParticles.nHits
    #* class to calculate quantities
    q = ShowerMergeQuantities(events, to_merge)

    #* define signal and background
    signal_all = np.logical_or(*signal)
    signal_all = signal_all[q.null]
    background = np.logical_not(signal_all) # background is all other PFOs unrelated to the pi0 decay
    signal = [signal[i][q.null] for i in range(2)]

    #* plot number of signal and background per event
    nSignal = ak.count(signal_all[signal_all], -1)
    nBackground = ak.count(background[background], -1)

    if plotsToMake in ["all", "nPFO"]:
        labels = ["background", "signal"]
        Plots.PlotHistComparison([nBackground, nSignal], xlabel="number of PFOs per event", bins=20, labels=labels, density=False)
        if save: Plots.Save("nPFO", outDir)

    #* calculate geometric quantities
    if plotsToMake in ["all", "quantities", "2D"] or save is True: q.Evaluate(events, start_showers)
    if plotsToMake in ["all", "quantities"]:
        q.PlotQuantities(signal, background)
    if plotsToMake in ["2D"]:
        q.Plot2DQuantities(signal, background)

    if save is True and plotsToMake is None:
        q.SaveQuantitiesToCSV(signal, background)


def CSVWorkFlow():
    q = ShowerMergeQuantities()
    q.LoadQuantitiesToCSV(file)
    if plotsToMake in ["all", "quantities"]:
        q.PlotQuantities(q.signal, q.background)
    if plotsToMake in ["all", "2D"]:
        q.Plot2DQuantities(q.signal, q.background)
    if args.cut is True:
        initial_cuts = [0.7, 100, 50, 50, 1.5]
        #initial_cuts = [9999]*5
        passes = 0
        total = 10
        plot = False
        while passes < total:
            if passes == 0:
                final_cuts = CutBasedScan(q, initial_cuts, startPoint=0, stepSize=15)
            else:
                if passes == total-1: plot=True
                final_cuts = CutBasedScan(q, final_cuts, startPoint=0, stepSize=15, plotFinal=plot)
            passes += 1
    return

@Master.timer
def main():
    plt.rcParams.update({'font.size': 12})
    if save:
        os.makedirs(outDir, exist_ok=True)
    fileFormat = file.split('.')[-1]
    if fileFormat == "root":
        ROOTWorkFlow()
    if fileFormat == "csv":
        CSVWorkFlow()


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})

    parser = argparse.ArgumentParser(description="Shower merging study for beamMC, plots quantities used to decide which showers to merge.")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-n", "--normalize", dest="norm", action="store_true", help="normalise plots to compare shape")
    parser.add_argument("-l" "--log", dest="log", action="store_true", help="plot y axis on log scale")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="prod4a_merge_study", help="directory to save plots")
    parser.add_argument("-p", "--plots", dest="plotsToMake", type=str, choices=["all", "quantities", "multiplicity", "nPFO", "2D"], help="what plots we want to make")
    parser.add_argument("-c", "--cutScan", dest="cut", action="store_true", help="whether to do a cut based scan")
    parser.add_argument("--start-showers", dest="matchBy", type=str, choices=["angular", "spatial"], default="spatial", help="method to detemine start showers")
    args = parser.parse_args("test/merge-quantities.csv -c".split()) #! to run in Jutpyter notebook
    #args = parser.parse_args() #! run in command line

    # if args.file.split('.')[-1] != "root":
    #     files = []
    #     with open(args.file) as filelist:
    #         file = filelist.read().splitlines() 
    # else:
    #     file = args.file
    file = args.file
    save = args.save
    outDir = args.outDir
    plotsToMake = args.plotsToMake
    norm = args.norm
    if args.log is True:
        scale="log"
    else:
        scale="linear"

    main()