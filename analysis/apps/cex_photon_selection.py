#!/usr/bin/env python3
"""
Created on: 23/05/2023 12:58

Author: Shyam Bhuller

Description: Applies beam particle selection, photon shower candidate selection and writes out shower energies.
"""
import argparse
import json
import os

import awkward as ak
import pandas as pd
from rich import print

from python.analysis import Master, Processing, BeamParticleSelection, PFOSelection

def run(i, file, n_events, start, selected_events, args):
    output = {}

    events = Master.Data(file, n_events, start, args["ntuple_type"])

    with open(args["beam_quality_fit"], "r") as f:
        fit_values = json.load(f)


    mask = BeamParticleSelection.CreateDefaultSelection(events, False, fit_values, True, False)
    events.Filter([mask], [mask])

    mask = PFOSelection.InitialPi0PhotonSelection(events, verbose = True, return_table = False)
    events.Filter([mask])

    output["reco_energy"] = ak.flatten(events.recoParticles.energy)
    output["true_energy"] = ak.flatten(events.trueParticlesBT.energy)
    output["true_mother"] = ak.flatten(events.trueParticlesBT.motherPdg)
    return output

def main(args):
    outputs = Processing.mutliprocess(run, args.file, args.batches, args.events, vars(args), args.threads)

    output = {}
    for o in outputs:
        for k, v in o.items():
            if k not in output:
                output[k] = v
            else:
                output[k] = ak.concatenate([output[k], v])
    output = pd.DataFrame(output)
    print(output)
    os.makedirs(args.out, exist_ok = True)
    output.to_hdf(args.out + "photon_energies.hdf5", "df")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", nargs = "+", help = "NTuple file to study.")
    parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {[m.value for m in Master.Ntuple_Type]}.", required = True)

    parser.add_argument("--beam_quality_fit", dest = "beam_quality_fit", type = str, help = "fit values for the beam quality cut.", required = True)

    parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "number of batches to split n tuple files into when parallel processing processing data.")
    parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "number of events to process when parallel processing data.")

    parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "number of threads to use when processsing")

    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save plots")

    args = parser.parse_args()

    if args.out is None:
        if len(args.file) == 1:
            args.out = args.file[0].split("/")[-1].split(".")[0] + "/"
        else:
            args.out = "photon_energy/" #? how to make a better name for multiple input files?
    if args.out[-1] != "/": args.out += "/"

    print(vars(args))
    main(args)