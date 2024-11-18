#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:47

Author: Shyam Bhuller

Description: Selection studies for the charge exchange analysis.
"""
import argparse
import os

from rich import print as rprint
from python.analysis import (
    Master, BeamParticleSelection, EventSelection, PFOSelection, Plots,
    shower_merging, Processing, Tags, cross_section, EnergyTools, Utils)
import python.analysis.SelectionTools as st

import awkward as ak
import numpy as np
import pandas as pd

x_label = {
    "track_score_all" : "Track score", 
    "TrackScoreCut" : "Track score",
    "NHitsCut" : "Number of hits",
    "PiPlusSelection" : "Median $dE/dX$ (MeV/cm)",
    "BeamParticleDistanceCut" : "$d$ (cm)",
    "BeamParticleIPCut" : "$b$ (cm)",
    "Chi2ProtonSelection" : "$(\chi^{2}/ndf)_{p}$",
    "PiBeamSelection" : "True particle ID",
    "APA3Cut" : "Beam end position $z$ (cm)",
    "TrueFiducialCut" : "True beam end position $z$ (cm)",
    "PandoraTagCut" : "Pandora tag",
    "DxyCut" : "$\delta_{xy}$",
    "DzCut" : "$\delta_{z}$",
    "CosThetaCut" : "$\cos(\\theta)$",
    "MichelScoreCut" : "Michel score",
    "MedianDEdXCut" : "Median $dE/dX$ (MeV/cm)",
    "BeamScraperCut" : "$r_{inst}$",
    "TrackLengthSelection" : "l (cm)"
}
y_scale = {
    "track_score_all" : "log",
    "TrackScoreCut" : "linear",
    "NHitsCut" : "linear",
    "PiPlusSelection" : "log",
    "BeamParticleDistanceCut" : "linear",
    "BeamParticleIPCut" : "linear",
    "Chi2ProtonSelection" : "linear",
    "PiBeamSelection" : None,
    "APA3Cut" : "log",
    "TrueFiducialCut" : "linear",
    "PandoraTagCut" : None,
    "DxyCut" : "log",
    "DzCut" : "log",
    "CosThetaCut" : "log",
    "MichelScoreCut" : "log",
    "MedianDEdXCut" : "log",
    "BeamScraperCut" : "linear",
    "TrackLengthSelection" : "linear"
}
x_range = {
    "track_score_all" : None,
    "TrackScoreCut" : [0, 1],
    "NHitsCut" : [0, 500],
    "PiPlusSelection" : [0, 10],
    "BeamParticleDistanceCut" : [0, 150],
    "BeamParticleIPCut" : [0, 100],
    "Chi2ProtonSelection" : [0, 600],
    "PiBeamSelection" : None,
    "PandoraTagCut" : None,
    "APA3Cut" : [0, 700],
    "TrueFiducialCut" : [0, 700],
    "DxyCut" : [0, 5],
    "DzCut" : [-5, 5],
    "CosThetaCut" : [0.9, 1],
    "MichelScoreCut" : [0, 1],
    "MedianDEdXCut" : [1.5, 3],
    "BeamScraperCut" : [0, 5],
    "TrackLengthSelection" : [0, 400],

}
nbins = {
    "track_score_all" : 50,
    "TrackScoreCut" : 50,
    "NHitsCut" : 50,
    "PiPlusSelection" : 50,
    "BeamParticleDistanceCut" : 50,
    "BeamParticleIPCut" : 50,
    "Chi2ProtonSelection" : 50,
    "PiBeamSelection" : 50,
    "APA3Cut" : 50,
    "TrueFiducialCut" : 50,
    "PandoraTagCut" : 50,
    "DxyCut" : 50,
    "DzCut" : 50,
    "CosThetaCut" : 50,
    "MichelScoreCut" : 50,
    "MedianDEdXCut" : 50,
    "BeamScraperCut" : 50,
    "TrackLengthSelection" : 50
}
ncols = {
    "track_score_all" : 2,
    "TrackScoreCut" : 2,
    "NHitsCut" : 2,
    "PiPlusSelection" : 2,
    "BeamParticleDistanceCut" : 2,
    "BeamParticleIPCut" : 2,
    "Chi2ProtonSelection" : 2,
    "PiBeamSelection" : 2,
    "APA3Cut" : 2,
    "TrueFiducialCut" : 2,
    "PandoraTagCut" : 2,
    "DxyCut" : 2,
    "DzCut" : 2,
    "CosThetaCut" : 2,
    "MichelScoreCut" : 2,
    "MedianDEdXCut" : 2,
    "BeamScraperCut" : 2,
    "TrackLengthSelection" : 2
}
truncate = {
    "track_score_all" : True,
    "TrackScoreCut" : True,
    "NHitsCut" : False,
    "PiPlusSelection" : False,
    "BeamParticleDistanceCut" : True,
    "BeamParticleIPCut" : True,
    "Chi2ProtonSelection" : True,
    "PiBeamSelection" : False,
    "APA3Cut" : False,
    "TrueFiducialCut" : False,
    "PandoraTagCut" : False,
    "DxyCut" : False,
    "DzCut" : False,
    "CosThetaCut" : False,
    "MichelScoreCut" : False,
    "MedianDEdXCut" : False,
    "BeamScraperCut" : False,
    "BeamScraperCut" : False,
    "TrackLengthSelection" : False
}


def AnalyseBeamFiducialCut(events : Master.Data, beam_instrumentation : bool, functions : dict, args : dict):
    output = {}
    masks = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueBeamParticleTags(events) if beam_instrumentation is False else None)
    output["no_selection"] = st.MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))
    if "TrueFiducialCut" in args:
        masks = {k : functions[k](events, **v) for k, v in args.items() if k in ["APA3Cut", "TrueFiducialCut"]}
        for a in ["APA3Cut", "TrueFiducialCut"]:
            mask, property = functions[a](events, **args[a], return_property = True)
            cut_table.add_mask(mask, a)
            cut_values = args[a]["cut"] if "cut" in args[a] else None
            operations = args[a]["op"] if "op" in args[a] else None
            output[a] = st.MakeOutput(property, Tags.GenerateTrueBeamParticleTags(events), cut_values, operations)

            events.Filter([mask], [mask])
            output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* true particle population
    output["final_tags"] = st.MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))

    df = cut_table.get_table(init_data_name = "no selection", pfos = False, percent_remain = False, relative_percent = False, ave_per_event = False)

    return output, df, masks


def AnalyseBeamSelection(events : Master.Data, beam_instrumentation : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame, dict]:
    """ Manually applies the beam selection while storing the value being cut on, cut values and truth tags in order to do plotting
        and produce performance tables.

    Args:
        events (Master.Data): events to look at
        beam_instrumentation (bool): use beam instrumentation for beam particle selection
        beam_quality_fits (dict): fit values for the beam quality selection

    Returns:
        dict: output data.
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueBeamParticleTags(events) if beam_instrumentation is False else None)

    output["no_selection"] = st.MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))

    masks = {k : functions[k](events, **v) for k, v in args.items()}

    #* pi+ beam selection
    mask = BeamParticleSelection.PiBeamSelection(events, **args["PiBeamSelection"])
    cut_table.add_mask(mask, "PiBeamSelection")
    counts = Tags.GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i].mask)
    output["PiBeamSelection"] = st.MakeOutput(counts, Tags.GenerateTrueBeamParticleTags(events), None, None)
    events.Filter([mask], [mask])
    output["PiBeamSelection"]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    for a in args:
        if a in ["PiBeamSelection", "TrueFiducialCut"]: continue
        if ("TrueFiducialCut" in args) and (a == "APA3Cut"): continue # should have applied fiducial cuts first
        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = st.MakeOutput(property, Tags.GenerateTrueBeamParticleTags(events), cut_values, operations)

        if (a == "BeamScraperCut") and (beam_instrumentation == False):
            if beam_instrumentation == False:
                scraper_id = EnergyTools.IsScraper(events, args[a]["fits"])
                output[a + "_scraper_tag"] = st.MakeOutput(None, Tags.BeamScraperTag(scraper_id), None, None)

        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)
        
    #* true particle population
    output["final_tags"] = st.MakeOutput(None, Tags.GenerateTrueBeamParticleTags(events), None, None, EventSelection.GenerateTrueFinalStateTags(events))

    df = cut_table.get_table(init_data_name = "no selection", pfos = False, percent_remain = False, relative_percent = False, ave_per_event = False)
    return output, df, masks

def run(i, file, n_events, start, selected_events, args) -> dict:
    events = Master.Data(file, n_events, start, args["nTuple_type"], args["pmom"])

    output = {
        "name" : file,
        "fiducial" : None,
        "beam" : None,
        "null_pfo" : None
    }

    if args["data"] == True:
        selection_args = "data_arguments"
    else:
        selection_args = "mc_arguments"

    if "beam_selection" in args:
        print("beam particle selection")

        if "TrueFiducialCut" in args["beam_selection"]["selections"]:
            output_fd, table_fd, fd_masks = AnalyseBeamFiducialCut(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args])
            output["fiducial"] = {"data" : output_fd, "table" : table_fd, "masks" : fd_masks}

        output_beam, table_beam, beam_masks = AnalyseBeamSelection(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args]) # events are cut after this
        output["beam"] = {"data" : output_beam, "table" : table_beam, "masks" : beam_masks}

    if "valid_pfo_selection" in args:
        print("PFO pre-selection (valid PFO cuts)")
        good_PFO_mask = PFOSelection.GoodShowerSelection(events)
        good_PFO_cut_table = Master.CutTable.CutHandler(events, tags = None)
        good_PFO_cut_table.add_mask(good_PFO_mask, "GoodShowerSelection")
        events.Filter([good_PFO_mask])
        output["null_pfo"] = {"table" : good_PFO_cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False), "masks" : {"ValidPFOSelection" : good_PFO_mask}}
    return output

@Master.timer
def BeamPionSelection(
        events : Master.Data,
        args : argparse.Namespace | dict, is_mc : bool) -> Master.Data:
    """
    Apply beam pion selection to ntuples. Loads stored masks if
    present (cex_beam_selection_studies already run), else applies the
    fiducial, beam, and valid PFO selections without generating plots
    or saving masks.

    Args:
        events (Master.Data): analysis ntuple
        args (argparse.Namespace): analysis configuration
        is_mc (bool): is the ntuple mc or data?

    Returns:
        Master.Data: selected events.
    """
    args_c = Utils.args_to_dict(args)

    events_copy = events.Filter(returnCopy = True)
    if is_mc:
        selection_args = "mc_arguments"
        sample = "mc"
    else:
        selection_args = "data_arguments"
        sample = "data"

    if "beam_selection_masks" in args_c:
        masks = args_c["beam_selection_masks"][sample]
        if ("fiducial" in masks) and (len(masks["fiducial"]) > 0):
            mask = st.CombineMasks(masks["fiducial"][events.filename])
            events_copy.Filter([mask], [mask])
        mask = st.CombineMasks(masks["beam"][events.filename])
        events_copy.Filter([mask], [mask])
    else:
        for s in args_c["beam_selection"]["selections"]:
            mask = args_c["beam_selection"]["selections"][s](events_copy, **args_c["beam_selection"][selection_args][s])
            events_copy.Filter([mask], [mask])
            print(events_copy.cutTable.get_table())

    if "valid_pfo_selection" in args_c:
        if args_c["valid_pfo_selection"] is True:
            if "beam_selection_masks" in args:
                events_copy.Filter([args_c["beam_selection_masks"][sample]['null_pfo'][events.filename]['ValidPFOSelection']]) # apply PFO preselection here
            else:
                events_copy.Filter(PFOSelection.GoodShowerSelection(events))
    return events_copy

def MakeBeamSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, book_name = "beam"):
    """ Beam particle selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """
    norm = False if output_data is None else norm
    with Plots.PlotBook(outDir + f"{book_name}.pdf") as pdf:

        for p in output_mc:
            if p in x_label:
                if p == "PiBeamSelection":
                    Plots.PlotTags(output_mc["PiBeamSelection"]["tags"], "True particle ID")
                    pdf.Save()
                elif p == "PandoraTagCut":
                    if output_data is None:
                        Plots.PlotBar(output_mc[p]["value"], xlabel = x_label[p])
                    else:
                        Plots.PlotBarComparision(output_mc[p]["value"], output_data[p]["value"], label_1 = "MC", label_2 = "Data", xlabel = x_label[p], fraction = True, barlabel = True)
                    pdf.Save()

                elif p == "APA3Cut":
                    for l in ["linear", "log"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = l, x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = st.CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "TrueFiducialCut":
                    for l in ["linear", "log"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = None, norm = norm, y_scale = l, x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = st.CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "BeamScraperCut":
                    for t in ["BeamScraperCut", "BeamScraperCut_scraper_tag"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[t]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = st.CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                else:
                    Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                    if p == "CosThetaCut":
                        arrow_l = 0.02
                    else:
                        arrow_l = st.CalculateArrowLength(output_mc[p]["value"], x_range[p])
                    Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = arrow_l, arrow_loc = 0.7, color = "k")
                    pdf.Save()

        Plots.PlotTags(output_mc["final_tags"]["tags"], "True particle ID")
        pdf.Save()
    Plots.plt.close("all")
    return

@Master.timer
def main(args):
    shower_merging.SetPlotStyle(extend_colors = True)
    outdir = args.out + "beam_selection/"
    cross_section.os.makedirs(outdir, exist_ok = True)

    output_mc = st.MergeSelectionMasks(st.MergeOutputs(Processing.ApplicationProcessing(["mc"], outdir, args, run, False, "output_mc")["mc"]))

    output_data = None
    if "data" in args.ntuple_files:
        if len(args.ntuple_files["data"]) > 0:
            if args.mc_only is False:
                output_data = st.MergeSelectionMasks(st.MergeOutputs(Processing.ApplicationProcessing(["data"], outdir, args, run, False, "output_data")["data"]))

    # tables
    st.MakeTables(output_mc, args.out + "tables_mc/", "mc")
    if output_data is not None: st.MakeTables(output_data, args.out + "tables_data/", "data")

    # save masks used in selection
    st.SaveMasks(output_mc, args.out + "masks_mc/")
    if output_data is not None: st.SaveMasks(output_data, args.out + "masks_data/")

    # output directories
    os.makedirs(outdir + "plots/", exist_ok = True)

    # plots
    if output_mc["fiducial"]:
        MakeBeamSelectionPlots(output_mc["fiducial"]["data"], output_data["fiducial"]["data"] if output_data else None, outdir + "plots/", norm = args.norm, book_name = "fiducial")

    if output_mc["beam"]: #* this is assuming you apply the same cuts as Data and MC (which is implictly assumed for now)
        MakeBeamSelectionPlots(output_mc["beam"]["data"], output_data["beam"]["data"] if output_data else None, outdir + "plots/", norm = args.norm, book_name = "beam")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Config(parser, required = True)
    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Regen(parser)

    parser.add_argument("--mc", dest = "mc_only", action = "store_true", help = "Only analyse the MC file.")

    args = parser.parse_args()

    args = cross_section.ApplicationArguments.ResolveArgs(args)

    rprint(vars(args))
    main(args)