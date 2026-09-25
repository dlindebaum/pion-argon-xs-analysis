#!/usr/bin/env python3
"""
Created on: 25/09/2026 16:40

Author: Shyam Bhuller

Description: Plots regions information for different combination of region definitions.
"""
import argparse
import os

import awkward as ak
import numpy as np
import pandas as pd

from alive_progress import alive_bar

from python.analysis import cross_section, Master, Plots, Tags, RegionDefinitions, Application, NtupleProcessing

from apps.cex_analysis_input import RegionSelection

from rich import print

@Master.timer
def GetTotalPionInelMasks(mc : Master.Data) -> ak.Array:
    """ Returns the mask which selects true inelastic pi+.

    Args:
        mc (Master.Data): mc events.

    Returns:
        ak.Array: mask.
    """
    particle_tags = Tags.GenerateTrueBeamParticleTags(mc)
    return particle_tags["$\\pi^{+}$:inel"].mask


def run(i : int, file_desc : Master.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    mc = Master.Data(file_desc, n_events, start)

    ri = {}
    for r, v in RegionDefinitions.regions.items():
        print(r)
        reco_regions, true_regions = RegionSelection(mc, args, True, v, None, True) # should we generate permutations of the true process as well?
        ri[r] = {"reco_regions" : reco_regions, "true_regions" : true_regions}

    return ri

@Master.timer
def RecoRegionSelection(region_selections : dict[dict], out : str):
    """ Study which computes a correlation matrix ofthe event faction for reco regions and true regions.
        Saved to file to be used in toy simulation.

    Args:
        mc (Master.Data): mc events.
        args (argparse.Namespace): application arguments.
    """
    os.makedirs(out + "reco_regions/", exist_ok = True)
    pdf = Plots.PlotBook(out + "reco_regions/reco_regions_study")
    pe = {}
    pe_index_labels = None
    for r in region_selections:
        print(r)
        reco_regions = region_selections[r]["reco_regions"]
        true_regions = region_selections[r]["true_regions"]

        process_keys = [k for k in true_regions if k != "uncategorised"]
        if pe_index_labels is None:
            tags = Tags.ExclusiveProcessTags({k : None for k in process_keys})
            pe_index_labels = tags.name_simple.values

        counts = np.array(cross_section.CountInRegions(true_regions, reco_regions))
        Plots.plt.figure(figsize = [3 * 6.4, 3 * 4.8])
        Plots.PlotConfusionMatrix(counts, list(reco_regions.keys()), list(true_regions.keys()), y_label = "True process", x_label = "Reco region", title = cross_section.remove_(r), newFigure = False)
        pdf.Save()

        (a,b)=counts.shape
        diff = a-b
        if a>b:
            padding=((0,0),(0,diff))
        else:
            padding=((0,-diff),(0,0))
        counts = np.pad(counts, padding, mode='constant', constant_values = 0)


        pe[cross_section.remove_(r)] = ((np.diag(counts) / np.sum(counts, 0)) * (np.diag(counts) / np.sum(counts, 1)))[:len(true_regions) - 1] # -1 to exlcude the uncategorised species. (purity x efficiency not needed)

        reco_regions.pop("uncategorised")
        counts = cross_section.CountInRegions(true_regions, reco_regions)

        fractions_df = counts / np.sum(counts, axis = 1)[:, np.newaxis]
        fractions_df = pd.DataFrame(np.array(fractions_df).T, columns = true_regions, index = reco_regions) # columns are the true regions, so index over those to get the fractions
        fractions_df.to_hdf(out + f"reco_regions/{r}_reco_region_fractions.hdf5", "df")
    pdf.close()
    print(pe)
    print(pe_index_labels)
    pd.DataFrame(pe, index = pe_index_labels).style.format(precision = 2).to_latex(out + "reco_regions/pe.tex")
    return

@Master.timer
def main(args : argparse.Namespace):
    cross_section.PlotStyler.SetPlotStyle(True)
    out = args.out + "region_plots/"
    cross_section.os.makedirs(out, exist_ok = True)

    output_mc = NtupleProcessing.ApplicationProcessing(["mc"], out, args, run, True)["mc"]

    print(f"{output_mc=}")

    RecoRegionSelection(output_mc, out)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Analyses MC ntuples in order to visualise the region identifications.")

    Application.ApplicationArguments.Config(parser, True)
    Application.ApplicationArguments.Output(parser)
    Application.ApplicationArguments.Regen(parser)
    Application.ApplicationArguments.Processing(parser)

    args = parser.parse_args()
    args = Application.ApplicationArguments.ResolveArgs(args)

    print(vars(args))
    main(args)