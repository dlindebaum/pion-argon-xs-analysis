#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:36

Author: Shyam Bhuller

Description: Computes fits for the beam quality selection and stores them in a json file to be used with the full selection.
"""
import argparse
import os

from rich import print

from python.analysis import application
from python.analysis import common
from python.analysis import ntuple
from python.analysis import vector
from python.analysis.selection import beam
from python.analysis.beam import quality

def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    events = ntuple.Data(file_desc, n_events, start)

    #? should this be made configurable i.e. pass config and apply all selections before the beam quality cuts if it is in the list?

    func_args = "mc_arguments" if args["data"] is False else "data_arguments"

    if ("DxyCut" in args["beam_selection"]["selections"]) or ("DzCut" in args["beam_selection"]["selections"]) or ("CosThetaCut" in args["beam_selection"]["selections"]):
        for s in args["beam_selection"]["selections"]:
            if s in ["DxyCut", "DzCut", "CosThetaCut"]:
                break # only apply cuts before beam quality
            else:
                print(f"{s=}")
                mask = args["beam_selection"]["selections"][s](events, **args["beam_selection"][func_args][s])
                events.Filter([mask], [mask])
    else:
        # do default selections prior to beam quality
        mask = beam.PiBeamSelection(events, args["data"])
        events.Filter([mask], [mask])

        mask = beam.PandoraTagCut(events)
        events.Filter([mask], [mask])

        mask = beam.CaloSizeCut(events)
        events.Filter([mask], [mask])

        mask = beam.HasFinalStatePFOsCut(events)
        events.Filter([mask], [mask])

    #* fit gaussians to the start positions
    start_pos, end_pos = beam.GetTruncatedPos(events, args["beam_quality_truncate"])
    beam_dir = vector.normalize(vector.sub(end_pos, start_pos))
    return {"start_pos" : start_pos, "end_pos" : end_pos, "beam_dir": beam_dir}

@common.timer
def main(args):
    outdir = args.out + "beam_quality/"
    os.makedirs(outdir, exist_ok = True)

    outputs = application.processing(list(args.ntuple_files.keys()), outdir, args, run, True)

    fit_values = {s : quality.Fits(args, outputs[s], outdir, s) for s in outputs}

    if "data" not in outputs.keys():
        outputs["data"] = None
        fit_values["data"] = None
    
    quality.MakePlots(outputs["mc"], fit_values["mc"], outputs["data"], fit_values["data"], outdir, args.beam_quality_truncate)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Computes Guassian fit paramters needed for the beam quality cuts in the beam particle selection.", formatter_class = argparse.RawDescriptionHelpFormatter)

    application.ApplicationArguments.config(parser, True)
    application.ApplicationArguments.processing(parser)
    application.ApplicationArguments.output(parser)
    application.ApplicationArguments.regen(parser)

    args = parser.parse_args()

    args = application.ApplicationArguments.resolve_args(args)
    print(vars(args))
    main(args)