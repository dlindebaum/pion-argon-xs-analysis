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
    t = Master.MCTruth(events) # calculate truth info
    
    #* plotting
    if save is True:
        os.makedirs(outDir, exist_ok=True)
    for i in range(len(t_l)):
        Plots.PlotHist(ak.ravel(t[i]), bins, xlabel=t_l[i])
        if save is True:
            Plots.Save(names[i], outDir)
    if save is False:
        plt.show()


if __name__ == "__main__":
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    # plot labels
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading photon energy (GeV)", "True Sub leading photon energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]

    parser = argparse.ArgumentParser(description="Plot truth information of pi0 decays")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="truth_info/", help="directory to save plots")
    parser.add_argument("-p", "--plots", dest="plotsToMake", type=str, choices=["all", "truth", "reco", "error", "2D"], default="all", help="what plots we want to make")
    #args = parser.parse_args("-f ROOTFiles/pi0_multi_9_3_22.root".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    outDir = args.outDir
    plotsToMake = args.plotsToMake
    main()