"""
Created on: 17/03/2022 10:24

Author: Shyam Bhuller

Description: Plot basic Truth information of data sample
"""
import argparse
import awkward as ak
import os
import matplotlib.pyplot as plt
# custom imports
import Master
import Plots

@Master.timer
def main():
    events = Master.Data(file) # load file
    if selection:
        if sampleType == "purePi0":
            events.ApplyBeamFilter() # beam filter
            valid = Master.Pi0MCMask(events, None) # two photon decay + > 1 shower
            events.Filter([valid], [valid], False)
            _, _, valid = events.MCMatching(applyFilters=False) # matching
            events.Filter([valid], [valid], False)
        if sampleType == "beam":
            events = Master.BeamMCFilter(events)

    t = events.trueParticles.CalculatePhotonPairProperties() # calculate truth info

    #* plotting
    if save is True:
        os.makedirs(outDir, exist_ok=True)
    for i in range(len(t_l)):
        d = ak.ravel(t[i])
        if len(t_r[i]) == 2:
            d = d[d > t_r[i][0]]
            d = d[d < t_r[i][1]]
        Plots.PlotHist(d, bins, xlabel=t_l[i], newFigure=True)
        if save is True:
            Plots.Save(names[i], outDir)
    if save is False:
        plt.show()


if __name__ == "__main__":
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    # plot labels
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading photon energy (GeV)", "True sub leading photon energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    t_r = [[]]*5
    t_r = [[], [0, 1.5], [], [], []]

    parser = argparse.ArgumentParser(description="Plot truth information of pi0 decays")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="truth_info/", help="directory to save plots")
    parser.add_argument("-e", "--event_selection", dest="selection", action="store_true", help="apply event selection")
    parser.add_argument("-t", "--type", dest="sampleType", type=str, choices=["purePi0", "beam"], default="purePi0", help="Type of MC")
    #args = parser.parse_args("work/ROOTFiles/Prod4a_6GeV_BeamSim_00.root -e -t beam".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    outDir = args.outDir
    selection = args.selection
    sampleType = args.sampleType
    main()