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

from python.analysis import application
from python.analysis import common
from python.analysis import io
from python.analysis import ntuple
from python.analysis import plotting
from python.analysis import styles
from python.analysis import tags
from python.analysis.selection import beam


def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    events = ntuple.Data(file_desc, n_events, start)
    mask = beam.PiBeamSelection(events, args["data"])
    if args["data"] is False:
        event_tags = tags.GenerateTrueBeamParticleTags(events)
    else:
        event_tags = None
    return {"mask" : mask, "tags" : event_tags}


@common.timer
def main(args):
    styles.PlotStyler.SetPlotStyle(extend_colors = True)
    outdir = args.out + "beam_norm/"
    os.makedirs(outdir, exist_ok = True)

    outputs = application.ApplicationProcessing(list(args.ntuple_files.keys()), outdir, args, run, True)

    n_data = ak.sum(outputs["data"]["mask"])
    n_mc = ak.sum(outputs["mc"]["mask"])
    norm = round(n_data / n_mc, 3)

    with plotting.PlotBook(outdir + "plots.pdf") as book:
        plotting.plots.PlotTags(outputs["mc"]["tags"], "True particle ID")
        book.Save()

    io.SaveConfiguration({"norm" : norm, "mc" : int(n_mc), "data" : int(n_data)}, outdir + "norm.json")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Computes normalisation for beam pion analysis.", formatter_class = argparse.RawDescriptionHelpFormatter)

    application.ApplicationArguments.Config(parser, True)
    application.ApplicationArguments.Processing(parser)
    application.ApplicationArguments.Output(parser)
    application.ApplicationArguments.Regen(parser)

    args = parser.parse_args()

    args = application.ApplicationArguments.ResolveArgs(args)
    print(vars(args))
    main(args)