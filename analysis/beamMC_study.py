"""
Created on: 12/05/2022 12:12

Author: Shyam Bhuller

Description: Code to run analysis of pi0 reconstruction for beamMC
"""
import argparse

import Master
import Plots

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt

@matplotlib.rc_context({'font.size': 12})
@Master.timer
def main():
    events = Master.Data(file, True)
    if analysis == "nPFP":
        global particleNum, unique, counts, max_unique
        events = Master.BeamMCFilter(events)
        particleNum = events.trueParticlesBT.particleNumber
        unique = events.trueParticlesBT.GetUniqueParticleNumbers(particleNum)
        max_unique = ak.max(ak.num(unique))
        unique = ak.pad_none(unique, max_unique)
        counts = [ak.num(particleNum[particleNum == unique[:, i]]) for i in range(max_unique)]
        counts = ak.ravel(counts)
        Plots.PlotHist(counts)
        plt.show()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot quantities to study shower reconstruction")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="pi0_0p5GeV_100K/shower_merge/", help="directory to save plots")
    parser.add_argument("-a", "--analysis", dest="analysis", type=str, choices=["nPFP"], default="nPFP", help="what analysis we want to run")
    #args = parser.parse_args("work/ROOTFiles/Prod4a_1GeV_BeamSim_02.root".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    outDir = args.outDir
    analysis = args.analysis

    main()