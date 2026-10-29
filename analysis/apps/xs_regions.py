#!/usr/bin/env python3
"""
Created on: 25/09/2026 16:40

Author: Shyam Bhuller

Description: Plots regions information for different combination of region definitions.
"""
import argparse
import os

from python.analysis import application, common, ntuple, plotting, regions, samples

from rich import print


def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    mc = ntuple.Data(file_desc, n_events, start)

    ri = {}
    for r, v in samples.regions.items():
        print(r)
        reco_regions, true_regions = regions.RegionSelection(mc, args, True, v, None, True) # should we generate permutations of the true process as well?
        ri[r] = {"reco_regions" : reco_regions, "true_regions" : true_regions}

    return ri

@common.timer
def main(args : argparse.Namespace):
    plotting.styles.PlotStyler.SetPlotStyle(True)
    out = args.out + "region_plots/"
    os.makedirs(out, exist_ok = True)

    output_mc = application.processing(["mc"], out, args, run, True)["mc"]

    print(f"{output_mc=}")

    regions.RecoRegionSelection(output_mc, out)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Analyses MC ntuples in order to visualise the region identifications.")

    application.ApplicationArguments.config(parser, True)
    application.ApplicationArguments.output(parser)
    application.ApplicationArguments.regen(parser)
    application.ApplicationArguments.processing(parser)

    args = parser.parse_args()
    args = application.ApplicationArguments.resolve_args(args)

    print(vars(args))
    main(args)