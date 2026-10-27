#!/usr/bin/env python3
"""
Created on: 19/05/2023 13:47

Author: Shyam Bhuller

Description: Selection studies for the charge exchange analysis.
"""
import argparse
import os

from rich import print

from python.analysis import application, common, cuts, ntuple, plotting, selection

def run(i, file : ntuple.FileDescriptor, n_events, start, selected_events, args) -> dict:
    events = ntuple.Data(file, n_events, start)

    output = {
        "name" : file.file,
        "fiducial" : None,
        "beam" : None,
        "null_pfo" : None,
        "pi" : None,
        "photon" : None,
        "loose_pi" : None,
        "loose_photon" : None,
        "pi0" : None,
        "regions" : None
    }

    if args["data"] == True:
        selection_args = "data_arguments"
    else:
        selection_args = "mc_arguments"

    if "beam_selection" in args:
        print("beam particle selection")

        if "TrueFiducialCut" in args["beam_selection"]["selections"]:
            output_fd, table_fd, fd_masks = selection.processing.AnalyseBeamFiducialCut(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args])
            output["fiducial"] = {"data" : output_fd, "table" : table_fd, "masks" : fd_masks}

        output_beam, table_beam, beam_masks = selection.processing.AnalyseBeamSelection(events, args["data"], args["beam_selection"]["selections"], args["beam_selection"][selection_args]) # events are cut after this
        output["beam"] = {"data" : output_beam, "table" : table_beam, "masks" : beam_masks}

    if "valid_pfo_selection" in args:
        print("PFO pre-selection")
        good_PFO_mask = selection.pfo.GoodShowerSelection(events)
        good_PFO_cut_table = cuts.CutHandler(events, particle_tags = None)
        good_PFO_cut_table.add_mask(good_PFO_mask, "GoodShowerSelection")
        events.Filter([good_PFO_mask])
        output["null_pfo"] = {"table" : good_PFO_cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False), "masks" : {"ValidPFOSelection" : good_PFO_mask}}

    # this selection only needs to be done for shower merging ntuple because the PDSPAnalyser ntuples only store daughter PFOs, not the beam as well.
    if events.nTuple_type == ntuple.Ntuple_Type.SHOWER_MERGING:
        #* beam particle daughter selection 
        mask = selection.pfo.BeamDaughterCut(events)
        events.Filter([mask])

    if "piplus_selection" in args:
        print("pion selection")
        output_pip, table_pip = selection.processing.AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["piplus_selection"]["selections"], args["piplus_selection"][selection_args])        
        pip_masks = selection.processing.CreatePFOMasks(events, args["piplus_selection"], selection_args)
        output["pi"] = {"data" : output_pip, "table" : table_pip, "masks" : pip_masks}

    if "loose_pion_selection" in args:
        print("loose pion selection")
        output_loose_pip, table_loose_pip = selection.processing.AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["loose_pion_selection"]["selections"], args["loose_pion_selection"][selection_args])
        loose_pip_masks = selection.processing.CreatePFOMasks(events, args["loose_pion_selection"], selection_args)
        output["loose_pi"] = {"data" : output_loose_pip, "table" : table_loose_pip, "masks" : loose_pip_masks}

    if "loose_photon_selection" in args:
        print("loose photon selection")
        output_loose_photon, table_loose_photon = selection.processing.AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["loose_photon_selection"]["selections"], args["loose_photon_selection"][selection_args])
        loose_photon_masks = selection.processing.CreatePFOMasks(events, args["loose_photon_selection"], selection_args)
        output["loose_photon"] = {"data" : output_loose_photon, "table" : table_loose_photon, "masks" : loose_photon_masks}

    if "photon_selection" in args:
        print("photon selection")
        output_photon, table_photon = selection.processing.AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["photon_selection"]["selections"], args["photon_selection"][selection_args])
        photon_masks = selection.processing.CreatePFOMasks(events, args["photon_selection"], selection_args)
        output["photon"] = {"data" : output_photon, "table" : table_photon, "masks" : photon_masks}

        photon_selection_mask = None
        for m in photon_masks:
            if photon_selection_mask is None:
                photon_selection_mask = photon_masks[m]
            else:
                photon_selection_mask = photon_selection_mask & photon_masks[m]

        if "pi0_selection" in args:
            print("pi0 selection")
            output_pi0, table_pi0 = selection.processing.AnalysePi0Selection(events.Filter(returnCopy = True), args["data"], args["pi0_selection"]["selections"], args["pi0_selection"][selection_args], photon_selection_mask)
            pi0_masks = selection.processing.CreatePFOMasks(events, args["pi0_selection"], selection_args, {"photon_mask" : photon_selection_mask})
            output["pi0"] = {"data" : output_pi0, "table" : table_pi0, "masks" : pi0_masks}

        print("regions")
        truth_regions, reco_regions = selection.processing.AnalyseRegions(events, photon_selection_mask, args["data"], args["shower_correction"]["correction"], args["shower_correction"]["correction_params"])

        regions  = {
            "truth_regions" : truth_regions,
            "reco_regions"  : reco_regions
        }
        output["regions"] = regions
    return output


@common.timer
def main(args):
    plotting.styles.SetPlotStyle(extend_colors = True)
    outdir = args.out + "selection/"
    os.makedirs(outdir, exist_ok = True)

    output_mc = selection.processing.MergeSelectionMasks(selection.processing.merge_multiproccess_output(application.ApplicationProcessing(["mc"], outdir, args, run, False, "output_mc")["mc"]))

    output_data = None
    if "data" in args.ntuple_files:
        if len(args.ntuple_files["data"]) > 0:
            if args.mc_only is False:
                output_data = selection.processing.MergeSelectionMasks(selection.processing.merge_multiproccess_output(application.ApplicationProcessing(["data"], outdir, args, run, False, "output_data")["data"]))

    # tables
    selection.plots.MakeTables(output_mc, args.out + "tables_mc/", "mc")
    if output_data is not None: selection.plots.MakeTables(output_data, args.out + "tables_data/", "data")

    # save masks used in selection
    selection.processing.SaveMasks(output_mc, args.out + "masks_mc/")
    if output_data is not None: selection.processing.SaveMasks(output_data, args.out + "masks_data/")

    # output directories
    os.makedirs(outdir + "plots/", exist_ok = True)

    # plots
    if output_mc["fiducial"]:
        selection.plots.MakeBeamSelectionPlots(output_mc["fiducial"]["data"], output_data["fiducial"]["data"] if output_data else None, outdir + "plots/", norm = args.norm, book_name = "fiducial")

    if output_mc["beam"]: #* this is assuming you apply the same cuts as Data and MC (which is implictly assumed for now)
        selection.plots.MakeBeamSelectionPlots(output_mc["beam"]["data"], output_data["beam"]["data"] if output_data else None, outdir + "plots/", norm = args.norm, book_name = "beam")

    for i in ["pi", "photon", "loose_pi", "loose_photon"]:
        if output_mc[i]:
            selection.plots.MakePFOSelectionPlots(output_mc[i]["data"], output_data[i]["data"] if output_data else None, outdir + "plots/", norm = args.norm, book_name = i)

    if output_mc["loose_pi"]:
        selection.plots.MakePFOSelectionPlotsConsdensed(
            output_mc["pi"]["data"],
            output_mc["loose_pi"]["data"],
            output_data["pi"]["data"] if output_data else None,
            output_data["loose_pi"]["data"] if output_data else None,
            outdir + "plots/",
            norm = args.norm,
            book_name = "pi_both"
            )

    if output_mc["loose_photon"]:
        selection.plots.MakePFOSelectionPlotsConsdensed(
            output_mc["photon"]["data"],
            output_mc["loose_photon"]["data"],
            output_data["photon"]["data"] if output_data else None,
            output_data["loose_photon"]["data"] if output_data else None,
            outdir + "plots/",
            norm = args.norm,
            book_name = "photon_both"
            )

    if output_mc["pi0"]:
        selection.plots.MakePi0SelectionPlots(output_mc["pi0"]["data"], output_data["pi0"]["data"] if output_data else None, outdir + "plots/", norm = args.norm, nbins = 50)
    if output_mc["regions"]:
        selection.plots.MakeRegionPlots(output_mc["regions"], output_data["regions"] if output_data else None, outdir + "plots/")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)

    application.ApplicationArguments.Config(parser, required = True)
    application.ApplicationArguments.Processing(parser)
    application.ApplicationArguments.Output(parser)
    application.ApplicationArguments.Regen(parser)

    parser.add_argument("--mc", dest = "mc_only", action = "store_true", help = "Only analyse the MC file.")

    args = parser.parse_args()

    args = application.ApplicationArguments.ResolveArgs(args)

    print(vars(args))
    main(args)