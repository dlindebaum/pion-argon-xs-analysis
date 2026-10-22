#!/usr/bin/env python3
"""
Created on: 25/09/2026 16:40

Author: Shyam Bhuller

Description: Plots regions information for different combination of region definitions.
"""
import argparse
import os

from python.analysis import application
from python.analysis import common
from python.analysis import ntuple
from python.analysis import regions
from python.analysis import samples
from python.analysis import styles

from apps.cex_analysis_input import RegionSelection

from rich import print


def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    mc = ntuple.Data(file_desc, n_events, start)

    ri = {}
    for r, v in samples.region.regions.items():
        print(r)
        reco_regions, true_regions = RegionSelection(mc, args, True, v, None, True) # should we generate permutations of the true process as well?
        ri[r] = {"reco_regions" : reco_regions, "true_regions" : true_regions}

    return ri

@common.timer
def main(args : argparse.Namespace):
    styles.PlotStyler.SetPlotStyle(True)
    out = args.out + "region_plots/"
    os.makedirs(out, exist_ok = True)

    output_mc = application.ApplicationProcessing(["mc"], out, args, run, True)["mc"]

    print(f"{output_mc=}")

    regions.RecoRegionSelection(output_mc, out)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Analyses MC ntuples in order to visualise the region identifications.")

    application.ApplicationArguments.Config(parser, True)
    application.ApplicationArguments.Output(parser)
    application.ApplicationArguments.Regen(parser)
    application.ApplicationArguments.Processing(parser)

    args = parser.parse_args()
    args = application.ApplicationArguments.ResolveArgs(args)

    print(vars(args))
    main(args)