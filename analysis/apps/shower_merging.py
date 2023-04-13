#!/usr/bin/env python3
"""
Created on: 14/03/2023 15:18

Author: Shyam Bhuller

Description: Applies event selection and PFO selection, based on the cuts dataframe will select showers to merge, and write out the shower pair properties to file for plotting.
#TODO fix performance tables metric for parallel processing.
"""
import argparse
import os
import sys

import awkward as ak
import numpy as np
import pandas as pd
from rich import print as rprint

from python.analysis import Master, shower_merging, Processing


def RecoShowerPairsDataFrame(events : Master.Data, start_showers : ak.Array, to_merge : ak.Array, cuts : str, cut_type : str) -> tuple:
    copy = events.Filter(returnCopy = True) # make a local copy of the events object so that we can do both the cheated merging and regular merging
    quantities = shower_merging.ShowerMergeQuantities(copy, to_merge, cuts)
    quantities.bestCut = cut_type
    quantities.to_merge_dir = copy.recoParticles.direction
    quantities.to_merge_pos = copy.recoParticles.startPos
    pair_mask, event_performance_table, pfo_performance_table = shower_merging.ShowerMerging(copy, start_showers, to_merge, quantities, -1)
    pairs = Master.ShowerPairs(copy, shower_pair_mask = np.logical_or(*pair_mask))
    return pairs.CalculateAll(), event_performance_table, pfo_performance_table


def Filter(df : pd.DataFrame, value : str) -> pd.DataFrame:
    out = df.filter(regex = value + "*")
    out.columns = [out.columns[i].replace(value + "_", "") for i in range(len(out.columns))]
    return out


def run(i, file, n_events, start, args):
    batch_num = i # function arguments are pass by reference, so keep track of the batch number internally
    with open(f"out_{batch_num}.log", "w") as sys.stdout, open(f"out_{batch_num}.log", "w") as sys.stderr:
        events = Master.Data(file, nEvents = n_events, start = start)

        shower_merging.Selection(events, args["selection_type"], args["selection_type"])

        #* tag the shower pairs based on event topology
        tags = shower_merging.GenerateTruthTags(events)

        tags_number = [-1] * len(tags.number[0].mask)
        tags_map = {"not tagged" : [-1]}
        for i, k in enumerate(tags):
            tags_map[k] = [tags[k].number]
            tags_number = ak.where(tags[k].mask == True, tags[k].number, tags_number)

        tags_number = pd.DataFrame({"tag": ak.to_list(tags_number)})
        tags_map = pd.DataFrame(tags_map)

        if args["selection_type"] == "cheated":
            start_showers, to_merge = shower_merging.SplitSample(events)

        if args["selection_type"] == "reco":
            start_showers, to_merge = shower_merging.SplitSampleReco(events)

        unmerged_pairs = Master.ShowerPairs(events, shower_pair_mask = np.logical_or(*start_showers))
        u_df = unmerged_pairs.CalculateAll()

        metadata = pd.concat([u_df[["run", "subrun", "event"]], tags_number], axis = 1)

        r_df, event_performance_table, pfo_performance_table = RecoShowerPairsDataFrame(events, start_showers, to_merge, args["cuts"], args["cut_type"])

        cheat_merge = [pd.DataFrame([])]*2
        if args["selection_type"] == "cheated":
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
        return data, event_performance_table, pfo_performance_table


def MergeTables(tables : list) -> dict:
    """ Combine tables produced from run().

    Args:
        tables (list): list of tables

    Returns:
        dict: combined tables
    """
    table = {}
    for t in tables:
        for k, v in t.items():
            if k in table:
                table[k][0] = table[k][0] + v
            else:
                table[k] = [v]
    return pd.DataFrame(table, index = ["counts"]).T


def main(args):
    output = Processing.mutliprocess(run, args.file, args.batches, args.events, vars(args), args.threads)
    rprint(len(output))

    data = {}
    event_perf_tables = []
    pfo_perf_tables = []

    for o in output:
        event_perf_tables.append(o[1])
        pfo_perf_tables.append(o[2])
        
        for k, v in o[0].items():

            if k in data:
                if k == "tag_map": continue # map is a copy for each batch so doesn't need to get appended
                data[k] = pd.concat([data[k], v], axis = 0)
            else:
                data[k] = v

    event_table = MergeTables(event_perf_tables)
    event_table["percentage"] = 100 * event_table["counts"] / event_table["counts"][0]
    pfo_table = MergeTables(pfo_perf_tables)
    pfo_table["percentage"] = 100 * pfo_table["counts"] / pfo_table["counts"][0]

    rprint(event_table)
    rprint(pfo_table)

    os.makedirs(args.out, exist_ok = True)
    event_table.to_latex(args.out + "event_performance_table.tex")
    pfo_table.to_latex(args.out + "pfo_performance_table.tex")
    file = pd.HDFStore(args.out + "shower_pairs.hdf5")
    for k, v in data.items():
        v.to_hdf(file, k + "/")
    file.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Calculate geometric quantities of PFOs to be used for the shower merging analysis.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", nargs = "+", help = "NTuple file to study.")

    parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "number of batches to split n tuple files into when parallel processing processing data.")
    parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "number of events to process when parallel processing data.")

    parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "number of threads to use when processsing")

    parser.add_argument("-s", "--selection", dest = "selection_type", type = str, choices = ["cheated", "reco"], help = "type of selection to use.", required = True)

    parser.add_argument("-c", "--cuts", dest = "cuts", type = str, help = "list of cuts to choose from", required = True)
    parser.add_argument("-T", "--type", dest = "cut_type", type = str, choices = ["purity", "efficiency"], help = "type of cut to pick.", required = True)

    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save files")

    args = parser.parse_args()

    if args.out is None:
        if len(args.file) == 1:
            args.out = args.file[0].split("/")[-1].split(".")[0] + "/"
        else:
            args.out = "shower_merging/" #? how to make a better name for multiple input files?
    if args.out[-1] != "/": args.out += "/"

    rprint(vars(args))
    main(args)