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

from python.analysis import Master, Plots, cross_section, BeamParticleSelection, Tags, Application


def run(i : int, file_desc : Master.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    events = Master.Data(file_desc, n_events, start)
    mask = BeamParticleSelection.PiBeamSelection(events, args["data"])
    if args["data"] is False:
        tags = Tags.GenerateTrueBeamParticleTags(events)
    else:
        tags = None
    return {"mask" : mask, "tags" : tags}


@Master.timer
def main(args):
    cross_section.PlotStyler.SetPlotStyle(extend_colors = True)
    outdir = args.out + "beam_norm/"
    os.makedirs(outdir, exist_ok = True)

    outputs = cross_section.ApplicationProcessing(list(args.ntuple_files.keys()), outdir, args, run, True)

    n_data = ak.sum(outputs["data"]["mask"])
    n_mc = ak.sum(outputs["mc"]["mask"])
    norm = round(n_data / n_mc, 3)

    with Plots.PlotBook(outdir + "plots.pdf") as book:
        Plots.PlotTags(outputs["mc"]["tags"], "True particle ID")
        book.Save()

    Master.SaveConfiguration({"norm" : norm, "mc" : int(n_mc), "data" : int(n_data)}, outdir + "norm.json")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Computes normalisation for beam pion analysis.", formatter_class = argparse.RawDescriptionHelpFormatter)

    Application.ApplicationArguments.Config(parser, True)
    Application.ApplicationArguments.Processing(parser)
    Application.ApplicationArguments.Output(parser)
    Application.ApplicationArguments.Regen(parser)

    args = parser.parse_args()

    args = Application.ApplicationArguments.ResolveArgs(args)
    print(vars(args))
    main(args)