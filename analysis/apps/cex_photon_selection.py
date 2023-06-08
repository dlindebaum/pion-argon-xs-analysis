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

from python.analysis import Master, Processing, BeamParticleSelection, PFOSelection, cross_section

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
    for i in ["x", "y", "z"]:
        output["reco_dir_{i}"] = ak.flatten(events.recoParticles.direction[i])
        output["true_dir_{i}"] = ak.flatten(events.trueParticlesBT.direction[i])
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
    parser = argparse.ArgumentParser(description = "Applies beam particle selection and saves properties of photon shower candidate PFOs to hdf5 file (MC only)", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.SingleNtuple(parser, define_sample = False)
    cross_section.ApplicationArguments.BeamQualityCuts(parser, data = False)
    cross_section.ApplicationArguments.BeamSelection(parser)
    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)

    args = parser.parse_args()

    cross_section.ApplicationArguments.ResolveArgs(parser)

    print(vars(args))
    # main(args)