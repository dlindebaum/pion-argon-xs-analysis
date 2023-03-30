#!/usr/bin/env python3
"""
Created on: 14/03/2023 15:18

Author: Shyam Bhuller

Description: Applies event selection and PFO selection, based on the cuts dataframe will select showers to merge, and write out the shower pair properties to file for plotting.
"""
import argparse

import awkward as ak
import numpy as np
import pandas as pd
from rich import print

from python.analysis import Master, shower_merging


def RecoShowerPairsDataFrame(events : Master.Data, start_showers : ak.Array, to_merge : ak.Array) -> pd.DataFrame:
    copy = events.Filter(returnCopy = True) # make a local copy of the events object so that we can do both the cheated merging and regular merging
    quantities = shower_merging.ShowerMergeQuantities(copy, to_merge, args.cuts)
    quantities.bestCut = args.cut_type
    quantities.to_merge_dir = copy.recoParticles.direction
    quantities.to_merge_pos = copy.recoParticles.startPos
    pair_mask = shower_merging.ShowerMerging(copy, start_showers, to_merge, quantities, -1)
    pairs = Master.ShowerPairs(copy, shower_pair_mask = np.logical_or(*pair_mask))
    return pairs.CalculateAll()


def Filter(df : pd.DataFrame, value : str) -> pd.DataFrame:
    out = df.filter(regex = value + "*")
    out.columns = [out.columns[i].replace(value + "_", "") for i in range(len(out.columns))]
    return out


def main(args):
    events = Master.Data(args.file, nEvents = args.nEvents[0], start = args.nEvents[1])

    shower_merging.Selection(events, args.selection_type, args.selection_type)

    #* tag the shower pairs based on event topology
    tags = shower_merging.GenerateTruthTags(events)

    tags_number = [-1] * len(tags["$\geq 1\pi^{0} + X$"].mask)
    tags_map = {"not tagged" : [-1]}
    for i, k in enumerate(tags):
        tags_map[k] = [i]
        tags_number = ak.where(tags[k].mask == True, i, tags_number)

    tags_number = pd.DataFrame({"tag": ak.to_list(tags_number)})
    tags_map = pd.DataFrame(tags_map)

    if args.selection_type == "cheated":
        start_showers, to_merge = shower_merging.SplitSample(events)

    if args.selection_type == "reco":
        start_showers, to_merge = shower_merging.SplitSampleReco(events)

    unmerged_pairs = Master.ShowerPairs(events, shower_pair_mask = np.logical_or(*start_showers))
    u_df = unmerged_pairs.CalculateAll()

    metadata = pd.concat([u_df[["run", "subrun", "event"]], tags_number], axis = 1)

    r_df = RecoShowerPairsDataFrame(events, start_showers, to_merge)

    cheat_merge = [pd.DataFrame([])]*2
    if args.selection_type == "cheated":
        events.MergePFOCheat(0)
        pairs = events.trueParticlesBT.mother == ak.flatten(events.trueParticles.number[events.trueParticles.PrimaryPi0Mask])
        cheated_pairs = Master.ShowerPairs(events, shower_pair_mask = pairs)
        c_df = cheated_pairs.CalculateAll()
        cheat_merge[0] = Filter(c_df, "reco")
        cheat_merge[1] = Filter(c_df, "error")

    data = { # output data in hierarchical order
        "tag_map" : tags_map,
        "metadata" : metadata,
        "true" : Filter(u_df, "true"),
        "cheat": Filter(u_df, "cheated"),
        "unmerged/reco" : Filter(u_df, "reco"),
        "unmerged/error" : Filter(u_df, "error"),
        "merged/reco" : Filter(r_df, "reco"),
        "merged/error" : Filter(r_df, "error"),
        "merged_cheat/reco" : cheat_merge[0],
        "merged_cheat/error" : cheat_merge[1],
    }

    file = pd.HDFStore(args.outDir + args.out)
    for k, v in data.items():
        v.to_hdf(file, k + "/")
    file.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Calculate geometric quantities of PFOs to be used for the shower merging analysis.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", type = str, help = "NTuple file to study.")
    parser.add_argument("-e", "--events", dest = "nEvents", type = int, nargs = 2, default = [-1, 0], help = "number of events to analyse and number to skip (-1 is all)")

    parser.add_argument("-s", "--selection", dest = "selection_type", type = str, choices = ["cheated", "reco"], help = "type of selection to use.", required = True)

    parser.add_argument("-c", "--cuts", dest = "cuts", type = str, help = "list of cuts to choose from", required = True)
    parser.add_argument("-t", "--type", dest = "cut_type", type = str, choices = ["purity", "efficiency"], help = "type of cut to pick.", required = True)

    parser.add_argument("-d", "--directory", dest = "outDir", type = str, default = "", help = "directory to save files")
    parser.add_argument("-o", "--output-file", dest = "out", type = str, help = "output file name.")

    args = parser.parse_args()

    if args.out is None:
        args.out = args.file.split("/")[-1].split(".")[0]
    
    if "." not in args.out:
        args.out += "_shower_pairs.hdf5"

    else:
        args.out.split(".")[0] += "_shower_pairs.hdf5"
    
    if args.outDir != "" and args.outDir[-1] != "/":
        args.outDir += "/"

    print(vars(args))
    main(args)