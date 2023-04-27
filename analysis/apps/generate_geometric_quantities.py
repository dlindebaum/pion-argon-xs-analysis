#!/usr/bin/env python3
"""
Created on: 14/03/2023 15:15

Author: Shyam Bhuller

Description: Calculate geometric quantities of PFOs to be used for the shower merging analysis.

#? switch file format to hdf5?
"""
import argparse
import os
import sys

from rich import print

from python.analysis import Master, shower_merging, Processing

@Processing.log_process
def run(i, file, n_events, start, args):
    print(f"proccess: {i}")
    events = Master.Data(file, nEvents = n_events, start = start)
    shower_merging.Selection(events, args["selection_type"], args["selection_type"])

    if args["selection_type"] == "cheated":
        start_showers, to_merge = shower_merging.SplitSample(events)

    if args["selection_type"] == "reco":
        start_showers, to_merge = shower_merging.SplitSampleReco(events)

    signal, background, _ = shower_merging.SignalBackground(events, start_showers, to_merge)

    q = shower_merging.ShowerMergeQuantities(events, to_merge)
    q.Evaluate(events, start_showers)
    df = q.GenerateDataFrame(signal, background)
    return df


def main(args):
    os.makedirs(args.out, exist_ok = True)
    output = Processing.mutliprocess(run, args.file, args.batches, args.events, vars(args), args.threads)
    file_path = args.out + "geometric_quantities.csv"
    mode = "w"
    for o in output:
        o.to_csv(file_path, mode = mode, header = not os.path.exists(file_path))
        mode = "a" # swtich to append mode after writing the first entry
    print(f"geometric quantities saved to: {file_path}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Calculate geometric quantities of PFOs to be used for the shower merging analysis.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", nargs = "+", help = "NTuple file to study.")

    parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "number of batches to split n tuple files into when parallel processing processing data.")
    parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "number of events to process when parallel processing data.")

    parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "number of threads to use when processsing")

    parser.add_argument("-s", "--selection", dest = "selection_type", type = str, choices = ["cheated", "reco"], help = "type of selection to use.", required = True)

    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save files")

    args = parser.parse_args()

    if args.out is None:
        if len(args.file) == 1:
            args.out = args.file[0].split("/")[-1].split(".")[0] + "/"
        else:
            args.out = "geometric_quantities/" #? how to make a better name for multiple input files?
    if args.out[-1] != "/": args.out += "/"

    print(vars(args))
    main(args)