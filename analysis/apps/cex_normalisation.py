#!/usr/bin/env python3
"""
Created on: 19/04/2024 17:05

Author: Shyam Bhuller

Description: Computes normalisation for beam pion analysis.
"""

import os
import argparse

import awkward as ak

from rich import print

from python.analysis import Master, Plots, cross_section, BeamParticleSelection, Tags


def run(i : int, file : str, n_events : int, start : int, selected_events, args : dict):
    events = Master.Data(file, n_events, start, args["nTuple_type"], args["pmom"])
    mask = BeamParticleSelection.PiBeamSelection(events, args["data"])
    if args["data"] is False:
        tags = Tags.GenerateTrueBeamParticleTags(events)
    else:
        tags = None
    return {"mask" : mask, "tags" : tags}


@Master.timer
def main(args):
    cross_section.SetPlotStyle(extend_colors = True)
    out = args.out + "beam_norm/"
    os.makedirs(out, exist_ok = True)
    

    output_mc = cross_section.RunProcess(args.ntuple_files["mc"], False, args, run)
    output_data = cross_section.RunProcess(args.ntuple_files["data"], True, args, run)

    n_data = ak.sum(output_data["mask"])
    n_mc = ak.sum(output_mc["mask"])
    norm = round(n_data / n_mc, 3)

    with Plots.PlotBook(out + "plots.pdf") as book:
        Plots.PlotTags(output_mc["tags"], "True particle ID")
        book.Save()

    Master.SaveConfiguration({"norm" : norm, "mc" : int(n_mc), "data" : int(n_data)}, out + "norm.json")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Computes normalisation for beam pion analysis.", formatter_class = argparse.RawDescriptionHelpFormatter)

    # cross_section.ApplicationArguments.Ntuples(parser, data = True)
    cross_section.ApplicationArguments.Config(parser, True)
    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)
    print(vars(args))
    main(args)