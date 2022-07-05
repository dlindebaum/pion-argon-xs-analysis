import os
import argparse
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

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
        self.background = ak.unflatten(background, ak.count(background)//2)[0]


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
        labels = ["$b_{0}$", "$b_{1}$", "$s_{0}$", "$s_{1}$"]
        for i in range(len(self.names)):
            data = getattr(self, self.names[i])
            print(data)
            #* collect signal PFOs
            s = [data[j][signal[j]] for j in range(2)]

            #* collect background PFOs
            b = [data[j][background] for j in range(2)]

            Plots.PlotHistComparison([ak.ravel((b+s)[j]) for j in range(4)], bins=50, xlabel=self.xlabels[i], labels=labels, density=norm, y_scale=scale)
            if save: Plots.Save(self.names[i], outDir)


    def Plot2DQuantities(self, signal, background):
        labels = ["$b_{0}$", "$b_{1}$", "$s_{0}$", "$s_{1}$"]
        s_alpha = [self.alpha[i][signal[i]] for i in range(2)]
        b_alpha = [self.alpha[i][background] for i in range(2)]
        s_x= [self.delta_x[i][signal[i]] for i in range(2)]
        b_x= [self.delta_x[i][background] for i in range(2)]
        counts, xbins, ybins = np.histogram2d(ak.ravel(s_alpha[0]), ak.ravel(s_x[0]))
        contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=["red"])
        plt.clabel(contours, inline=True, fontsize=8)
        counts, xbins, ybins = np.histogram2d(ak.ravel(s_alpha[1]), ak.ravel(s_x[1]))
        contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=["purple"])
        plt.clabel(contours, inline=True, fontsize=8)
        counts, xbins, ybins = np.histogram2d(ak.ravel(b_alpha[0]), ak.ravel(b_x[0]))
        contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=["blue"])
        plt.clabel(contours, inline=True, fontsize=8)
        counts, xbins, ybins = np.histogram2d(ak.ravel(b_alpha[1]), ak.ravel(b_x[1]))
        contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=["green"])
        plt.clabel(contours, inline=True, fontsize=8)
        plt.xlabel(self.xlabels[5])
        plt.ylabel(self.xlabels[1])
        if save: Plots.Save(f"{self.names[5]}-{self.names[1]}", outDir)

        s_xl= [self.delta_xl[i][signal[i]] for i in range(2)]
        b_xl= [self.delta_xl[i][background] for i in range(2)]
        s_xt= [self.delta_xt[i][signal[i]] for i in range(2)]
        b_xt= [self.delta_xt[i][background] for i in range(2)]
        counts, xbins, ybins = np.histogram2d(ak.ravel(s_xl[0]), ak.ravel(s_xt[0]))
        contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=["red"])
        plt.clabel(contours, inline=True, fontsize=8)
        counts, xbins, ybins = np.histogram2d(ak.ravel(s_xl[1]), ak.ravel(s_xt[1]))
        contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=["purple"])
        plt.clabel(contours, inline=True, fontsize=8)
        counts, xbins, ybins = np.histogram2d(ak.ravel(b_xl[0]), ak.ravel(b_xt[0]))
        contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=["blue"])
        plt.clabel(contours, inline=True, fontsize=8)
        counts, xbins, ybins = np.histogram2d(ak.ravel(b_xl[1]), ak.ravel(b_xt[1]))
        contours = plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=0.5, colors=["green"])
        plt.clabel(contours, inline=True, fontsize=8)
        plt.xlabel(self.xlabels[2])
        plt.ylabel(self.xlabels[3])

        if save: Plots.Save(f"{self.names[2]}-{self.names[3]}", outDir)

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

    f = events.trueParticlesBT.SingleMatch
    events.Filter([f], [f])
    n.append(["nUnique > 1", ak.count(events.eventNum), Percentage(n[-1][1], ak.count(events.eventNum))])

    if args.matchBy == "spatial":
        start_showers = StartShowerByDistance(events)
    else:
        start_showers, _ = events.MatchByAngleBT() # pick starting showers by looking at angular closeness

    start_shower_pos = events.recoParticles.startPos[start_showers]
    start_shower_dir = events.recoParticles.direction[start_showers]
    f = ak.all(np.logical_or(start_shower_dir.x != -999, start_shower_pos.x != -999), -1) # ignore null directions/positions for starting showers

    events.Filter([f], [f])
    n.append(["valid start shower", ak.count(events.eventNum), Percentage(n[-1][1], ak.count(events.eventNum))])
    start_showers = start_showers[f]

    print(tabulate(n, tablefmt="fancy_grid"))

    return start_showers

@Master.timer
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
    mcIndex = events.trueParticlesBT.particleNumber #! can also use trueParticlesBT.number instead for BeamMC
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
    parser.add_argument("--start-showers", dest="matchBy", type=str, choices=["angular", "spatial"], default="angular", help="method to detemine start showers")
    #args = parser.parse_args("work/ROOTFiles/Prod4a_6GeV_BeamSim_00.root -p all".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

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