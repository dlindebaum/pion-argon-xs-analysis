#!/usr/bin/env python3
"""
Created on: 14/03/2023 15:15

Author: Shyam Bhuller

Description: Calculate geometric quantities of PFOs to be used for the shower merging analysis.
#TODO support input of multiple root files
#TODO switch file format to hdf5
"""
import argparse
import os

from rich import print

from python.analysis import Master, shower_merging


def main(args):
    events = Master.Data(args.file, nEvents = args.nEvents[0], start = args.nEvents[1])
    shower_merging.Selection(events, args.selection_type, args.selection_type)

    if args.selection_type == "cheated":
        start_showers, to_merge = shower_merging.SplitSample(events)

    if args.selection_type == "reco":
        start_showers, to_merge = shower_merging.SplitSampleReco(events)

    signal, background, _ = shower_merging.SignalBackground(events, start_showers, to_merge)

    q = shower_merging.ShowerMergeQuantities(events, to_merge)
    q.Evaluate(events, start_showers)

    os.makedirs(args.outDir, exist_ok = True)
    q.SaveQuantitiesToCSV(signal, background, args.outDir + args.out)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Calculate geometric quantities of PFOs to be used for the shower merging analysis.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", type = str, help = "NTuple file to study.")
    parser.add_argument("-e", "--events", dest = "nEvents", type = int, nargs = 2, default = [-1, 0], help = "number of events to analyse and number to skip (-1 is all)")

    parser.add_argument("-s", "--selection", dest = "selection_type", type = str, choices = ["cheated", "reco"], help = "type of selection to use.", required = True)

    parser.add_argument("-d", "--directory", dest = "outDir", type = str, default = "", help = "directory to save files")
    parser.add_argument("-o", "--output-file", dest = "out", type = str, help = "output file name.")

    args = parser.parse_args()

    if args.out is None:
        args.out = args.file.split("/")[-1].split(".")[0] + "_geometric_quantities"
    
    if "." not in args.out:
        args.out += ".csv"
    else:
        args.out.split(".")[0] += ".csv"

    if args.outDir == "":
        args.outDir = args.file.split("/")[-1].split(".")[0] + "/"

    if args.outDir != "" and args.outDir[-1] != "/":
        args.outDir += "/"

    print(vars(args))
    main(args)