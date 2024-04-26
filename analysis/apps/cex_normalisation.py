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


def run(file : str, data : bool, ntuple_type : Master.Ntuple_Type, args : cross_section.argparse.Namespace):
    events = Master.Data(file, nTuple_type = ntuple_type, target_momentum = args.pmom)

    mask = BeamParticleSelection.PiBeamSelection(events, data)
    if data is False:
        tags = Tags.GenerateTrueBeamParticleTags(events)
    else:
        tags = None

    return mask, tags


@Master.timer
def main(args):
    cross_section.SetPlotStyle(extend_colors = True)
    out = args.out + "beam_norm/"
    os.makedirs(out, exist_ok = True)

    mc_mask, mc_tags = run(args.mc_file, False, args.ntuple_type, args)
    data_mask, _ = run(args.data_file, True, args.ntuple_type, args)

    norm = round(ak.sum(data_mask) / ak.sum(mc_mask), 3)

    with Plots.PlotBook(out + "plots.pdf") as book:
        Plots.PlotTags(mc_tags, "True particle ID")
        book.Save()

    Master.SaveConfiguration({"norm" : norm}, out + "norm.json")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Computes normalisation for beam pion analysis.", formatter_class = argparse.RawDescriptionHelpFormatter)

    # cross_section.ApplicationArguments.Ntuples(parser, data = True)
    cross_section.ApplicationArguments.Config(parser, True)    
    cross_section.ApplicationArguments.Output(parser)

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)
    print(vars(args))
    main(args)