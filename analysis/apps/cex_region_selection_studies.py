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
    shower_merging, Processing, Tags, cross_section, EnergyTools)
import python.analysis.SelectionTools as st
import apps.cex_beam_selection_studies as beam_selection
from apps.cex_beam_selection_studies import x_label, y_scale, x_range, nbins, ncols, truncate
from python.analysis import Utils, SelectionTools, RegionIdentification

import awkward as ak
import numpy as np
import pandas as pd


def AnalysePFOSelection(events : Master.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse PFO selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTagsInterestingPFOs(events))

    for a in args:
        if a in ["NHitsCut", "BeamParticleIPCut"]:
            output[a + "_completeness"] = st.MakeOutput(events.trueParticlesBT.completeness, [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = st.MakeOutput(property, Tags.GenerateTrueParticleTagsInterestingPFOs(events), cut_values, operations)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = st.MakeOutput(None, Tags.GenerateTrueParticleTagsInterestingPFOs(events), None, None)
    
    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePiPlusSelection(events : Master.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse the daughter pi+ selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTagsPiPlus(events))

    for a in args:
        if a == "NHitsCut":
            output[a + "_completeness"] = st.MakeOutput(events.trueParticlesBT.completeness, [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = st.MakeOutput(property, Tags.GenerateTrueParticleTagsPiPlus(events), cut_values, operations)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = st.MakeOutput(None, Tags.GenerateTrueParticleTagsPiPlus(events), None, None)
    
    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePhotonCandidateSelection(events : Master.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse the photon candidate selection.

    Args:
        events (Master.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GenerateTrueParticleTagsPi0Shower(events) if data is False else None)

    for a in args:
        if a in ["NHitsCut", "BeamParticleIPCut"]:
            output[a + "_completeness"] = st.MakeOutput(events.trueParticlesBT.completeness, [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = st.MakeOutput(property, Tags.GenerateTrueParticleTagsPi0Shower(events), cut_values, operations)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = st.MakeOutput(None, Tags.GenerateTrueParticleTagsPi0Shower(events), None, None)

    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePi0Selection(events : Master.Data, data : bool, functions : dict, args : dict, photon_mask : ak.Array) -> tuple[dict, pd.DataFrame]:
    """ Analyse the pi0 selection.

    Args:
        events (Master.Data): events to look at
        data (bool): is the ntuple file data or MC
        correction (callable, optional): shower energy correction. Defaults to None.
        correction_params (dict, optional): shower energy correction parameters. Defaults to None.

    Returns:
        dict: output data
    """    
    def null_tag():
        tag = Tags.Tags()
        tag["null"] = Tags.Tag(mask = (events.eventNum < -1))
        return tag

    output = {}
    photonCandidates = ak.Array(photon_mask)
    cut_table = Master.CutTable.CutHandler(events, tags = Tags.GeneratePi0Tags(events, photon_mask) if data is False else None)

    for a in args:
        if a == "Pi0MassSelection":
            mask, property = functions[a](events, **args[a], return_property = True, photon_mask = photonCandidates)
            output["mass_event_tag"] = st.MakeOutput(None, EventSelection.GenerateTrueFinalStateTags(events), None, None)
        else:
            mask, property = functions[a](events, **args[a], return_property = True, photon_mask = photonCandidates)
        if a != "NPhotonCandidateSelection":
            mask = ak.flatten(mask)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
        output[a] = st.MakeOutput(property, tags, cut_values, operations)
        events.Filter([mask], [mask])
        photonCandidates = photonCandidates[mask]
        output[a]["fs_tags"] = EventSelection.GenerateTrueFinalStateTags(events)

    #* final counts
    output["event_tag"] = st.MakeOutput(None, EventSelection.GenerateTrueFinalStateTags(events), None, None)
    tags = null_tag() if data else Tags.GeneratePi0Tags(events, photonCandidates)
    output["final_tags"] = st.MakeOutput(None, tags, None, None)

    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, pfos = False)
    print(df)
    return output, df


def make_truth_regions(evts, is_data):
    return (EventSelection.create_regions(
            evts.trueParticles.nPi0,
            evts.trueParticles.nPiPlus + evts.trueParticles.nPiMinus)
        if  (not is_data) else None)

def AnalyseRegions(events : Master.Data, photon_mask : ak.Array, is_data : bool, correction : callable = None, correction_params : dict = None) -> tuple[dict, dict]:
    """ Create masks which desribe the truth and reco regions for various exlusive cross sections.

    Args:
        events (Master.Data): events to look at
        photon_mask (ak.Array): mask for pi0 photon shower candidates
        is_data (bool): is this a data ntuple?
        correction (callable, optional): shower energy correction. Defaults to None.
        correction_params (str, optional): shower energy correction parameters file. Defaults to None.

    Returns:
        tuple[dict, dict]: truth and reco regions
    """
    truth_regions = make_truth_regions(events, is_data)

    if correction_params is None:
        params = None
    else:
        params = cross_section.LoadConfiguration(correction_params)

    reco_pi0_counts = EventSelection.count_pi0_candidates(
        events, exactly_two_photons = True, photon_mask = photon_mask,
        correction = EnergyTools.EnergyCorrection.shower_energy_correction[correction],
        correction_params = params)
    reco_pi_plus_counts_mom_cut = EventSelection.count_charged_pi_candidates(events, energy_cut = None)
    reco_regions = EventSelection.create_regions(reco_pi0_counts, reco_pi_plus_counts_mom_cut)
    return truth_regions, reco_regions


def CreatePFOMasks(sample : Master.Data, selections : dict, args_type : str, extra_args : dict = None) -> dict[np.array]:
    """ Create PFO masks to save to file.

    Args:
        mc (Master.Data): sample.
        selections (dict): PFO selections.
        args_type (str): use Data or MC arguments.
        extra_args (dict, optional): any additional arguments to add. Defaults to None.

    Returns:
        masks: dictionary of masks.
    """
    masks = {}
    for n, c, v in zip(selections["selections"].keys(), selections["selections"].values(), selections[args_type].values()):
        if extra_args is not None:
            v = {**v, **extra_args}
        masks[n] = c(sample, **v)
    return masks

def run(i, file, n_events, start, selected_events, args) -> dict:
    events = Master.Data(file, n_events, start, args["nTuple_type"], args["pmom"])

    output = {
        "name" : file,
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

    events = beam_selection.BeamPionSelection(events, args, (not args["data"]))

    # this selection only needs to be done for shower merging ntuple because the PDSPAnalyser ntuples only store daughter PFOs, not the beam as well.
    if events.nTuple_type == Master.Ntuple_Type.SHOWER_MERGING:
        #* beam particle daughter selection 
        mask = PFOSelection.BeamDaughterCut(events)
        events.Filter([mask])

    if "piplus_selection" in args:
        print("pion selection")
        output_pip, table_pip = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["piplus_selection"]["selections"], args["piplus_selection"][selection_args])        
        pip_masks = CreatePFOMasks(events, args["piplus_selection"], selection_args)
        output["pi"] = {"data" : output_pip, "table" : table_pip, "masks" : pip_masks}

    if "loose_pion_selection" in args:
        print("loose pion selection")
        output_loose_pip, table_loose_pip = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["loose_pion_selection"]["selections"], args["loose_pion_selection"][selection_args])
        loose_pip_masks = CreatePFOMasks(events, args["loose_pion_selection"], selection_args)
        output["loose_pi"] = {"data" : output_loose_pip, "table" : table_loose_pip, "masks" : loose_pip_masks}

    if "loose_photon_selection" in args:
        print("loose photon selection")
        output_loose_photon, table_loose_photon = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["loose_photon_selection"]["selections"], args["loose_photon_selection"][selection_args])
        loose_photon_masks = CreatePFOMasks(events, args["loose_photon_selection"], selection_args)
        output["loose_photon"] = {"data" : output_loose_photon, "table" : table_loose_photon, "masks" : loose_photon_masks}

    if "photon_selection" in args:
        print("photon selection")
        output_photon, table_photon = AnalysePFOSelection(events.Filter(returnCopy = True), args["data"], args["photon_selection"]["selections"], args["photon_selection"][selection_args])
        photon_masks = CreatePFOMasks(events, args["photon_selection"], selection_args)
        output["photon"] = {"data" : output_photon, "table" : table_photon, "masks" : photon_masks}

        photon_selection_mask = None
        for m in photon_masks:
            if photon_selection_mask is None:
                photon_selection_mask = photon_masks[m]
            else:
                photon_selection_mask = photon_selection_mask & photon_masks[m]

        if "pi0_selection" in args:
            print("pi0 selection")
            output_pi0, table_pi0 = AnalysePi0Selection(events.Filter(returnCopy = True), args["data"], args["pi0_selection"]["selections"], args["pi0_selection"][selection_args], photon_selection_mask)
            pi0_masks = CreatePFOMasks(events, args["pi0_selection"], selection_args, {"photon_mask" : photon_selection_mask})
            output["pi0"] = {"data" : output_pi0, "table" : table_pi0, "masks" : pi0_masks}

    print("regions")
    truth_regions, reco_regions = AnalyseRegions(events, photon_selection_mask, args["data"], args["shower_correction"]["correction"], args["shower_correction"]["correction_params"])

    regions  = {
        "truth_regions"       : truth_regions,
        "reco_regions"        : reco_regions
    }
    output["regions"] = regions
    return output


def MakePFOSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, book_name : str):
    norm = False if output_data is None else norm

    with Plots.PlotBook(outDir + book_name) as pdf:

        for p in output_mc:
            if p in x_label:
                Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = st.CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.5, color = "C6")
                Plots.plt.ylim(bottom = 0)
                pdf.Save()
                if f"{p}_completeness" in output_mc:
                    Plots.PlotHist2DImshowMarginal(ak.ravel(output_mc[p]["value"]), ak.ravel(output_mc[f"{p}_completeness"]["value"]), ylabel = "Completeness", xlabel = x_label[p], x_range = x_range[p], bins = nbins[p], norm = "column", c_scale = "linear")
                    Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = st.CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.1, color = "C6")
                    pdf.Save()
        Plots.PlotTags(output_mc["final_tags"]["tags"], xlabel = "True particle ID")
        pdf.Save()
    Plots.plt.close("all")
    return


def MakePFOSelectionPlotsConsdensed(output_mc : dict, output_mc_loose : dict, output_data : dict, output_data_loose : dict, outDir : str, norm : float, book_name : str):
    norm = False if output_data is None else norm

    with Plots.PlotBook(outDir + book_name) as pdf:

        for p in output_mc_loose:
            if p in x_label:
                Plots.PlotTagged(output_mc_loose[p]["value"], output_mc_loose[p]["tags"], data2 = output_data_loose[p]["value"] if output_data_loose else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                
                for c, mc in zip(["C6", "magenta"], [output_mc, output_mc_loose]):
                    if p == "PiPlusSelection":
                        sf = 0.5
                    else:
                        sf = 1
                    Plots.DrawMultiCutPosition(mc[p]["cuts"], face = mc[p]["op"], arrow_length = sf * st.CalculateArrowLength(mc[p]["value"], x_range[p]), arrow_loc = 0.5, color = c)
                Plots.plt.ylim(bottom = 0)                
                pdf.Save()
                if f"{p}_completeness" in output_mc:
                    Plots.PlotHist2DImshowMarginal(ak.ravel(mc[p]["value"]), ak.ravel(mc[f"{p}_completeness"]["value"]), ylabel = "Completeness", xlabel = x_label[p], x_range = x_range[p], bins = nbins[p], norm = "column", c_scale = "linear")

                    for c, mc in zip(["C6", "magenta"], [output_mc, output_mc_loose]):
                        Plots.DrawMultiCutPosition(mc[p]["cuts"], face = mc[p]["op"], arrow_length = st.CalculateArrowLength(mc[p]["value"], x_range[p]), arrow_loc = 0.1, color = c)

                    pdf.Save()

        for mc in [output_mc, output_mc_loose]:
            Plots.PlotTags(mc["final_tags"]["tags"], xlabel = "True particle ID")
            pdf.Save()
    Plots.plt.close("all")
    return


def MakePi0SelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, nbins : int):
    """ Pi0 selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """
    norm = False if output_data is None else norm

    with Plots.PlotBook(outDir + "pi0.pdf") as pdf:
        if "NPhotonCandidateSelection" in output_mc:
            if output_data is not None:
                scale = ak.count(output_data["NPhotonCandidateSelection"]["value"]) / ak.count(output_mc["NPhotonCandidateSelection"]["value"])

                n_photons_scaled = []
                u, c = np.unique(output_mc["NPhotonCandidateSelection"]["value"], return_counts = True)
                for i, j in zip(u, c):
                    n_photons_scaled.extend([i]* int(scale * j))

                Plots.PlotBarComparision(n_photons_scaled, output_data["NPhotonCandidateSelection"]["value"], xlabel = "Number of $\pi^{0}$ photon candidates", label_1 = "MC", label_2 = "Data", fraction = True, barlabel = False)
            else:
                Plots.PlotBar(output_mc["NPhotonCandidateSelection"]["value"], xlabel = "Number of $\pi^{0}$ photon candidates")
            pdf.Save()

        if "Pi0MassSelection" in output_mc:
            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["Pi0MassSelection"]["tags"], data2 = output_data["Pi0MassSelection"]["value"] if output_data else None, bins = nbins, x_label = "$m_{\gamma\gamma}$ (MeV)", x_range = [0, 500], norm = norm, ncols = 1)
            Plots.DrawMultiCutPosition(output_mc["Pi0MassSelection"]["cuts"], face = output_mc["Pi0MassSelection"]["op"], arrow_length = 50, color = "C6")
            pdf.Save()

            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["mass_event_tag"]["tags"], data2 = output_data["Pi0MassSelection"]["value"] if output_data else None, bins = nbins, x_label = "$m_{\gamma\gamma}$ (MeV)", x_range = [0, 500], norm = norm, ncols = 1)
            Plots.DrawMultiCutPosition(output_mc["Pi0MassSelection"]["cuts"], face = output_mc["Pi0MassSelection"]["op"], arrow_length = 50, color = "C6")
            pdf.Save()

        if "Pi0OpeningAngleSelection" in output_mc:
            Plots.PlotTagged(output_mc["Pi0OpeningAngleSelection"]["value"], output_mc["Pi0OpeningAngleSelection"]["tags"], data2 = output_data["Pi0OpeningAngleSelection"]["value"] if output_data else None, bins = nbins, x_label = "$\phi$ (rad)", norm = norm, ncols = 1)

            Plots.DrawMultiCutPosition((np.array(output_mc["Pi0OpeningAngleSelection"]["cuts"]) * np.pi / 180).tolist(), face = output_mc["Pi0OpeningAngleSelection"]["op"], arrow_length = 0.25, color = "C6")
            Plots.plt.ylim(bottom = 0)
            pdf.Save()
    Plots.plt.close("all")
    return


def MakeRegionPlots(outputs_mc_masks : dict, outputs_data_masks : dict, outDir : str):
    """ Correlation matrices for truth and reco region selection.

    Args:
        outputs_mc_masks (dict): mc masks for each region
        outputs_data_masks (dict): data masks for each region
        outDir (str): output directory
    """
    with Plots.PlotBook(outDir + "regions.pdf") as pdf:
        # Visualise the regions
        Plots.plot_region_data(outputs_mc_masks["truth_regions"], compare_max=0, title="truth regions")
        pdf.Save()
        Plots.plot_region_data(outputs_mc_masks["reco_regions"], compare_max=0, title="reco regions")
        pdf.Save()
        # Compare the regions
        Plots.compare_truth_reco_regions(outputs_mc_masks["reco_regions"], outputs_mc_masks["truth_regions"], title="")
        pdf.Save()

        if outputs_data_masks is not None:
            Plots.plot_region_data(outputs_data_masks["reco_regions"], compare_max=0, title="reco regions")
            pdf.Save()
    Plots.plt.close("all")
    return

#TODO update to make sure this can run even if masks aren't stored
@Master.timer
def RegionSelection(
        events : Master.Data,
        args : argparse.Namespace | dict, is_mc : bool,
        region_type : str = None,
        removed : bool = False) -> dict[np.ndarray]:
    """ Get reco and true regions (if possible) for ntuple.

    Args:
        events (Master.Data): events after beam pion selection
        args (argparse.Namespace): application arguements
        is_mc (bool): if ntuple is MC or Data.

    Returns:
        tuple[dict, dict]: regions
    """

    args_c = Utils.args_to_dict(args)

    if is_mc:
        key = "mc"
    else:
        key = "data"

    selection_masks = args_c["region_selection_masks"][key]

    counts = {}
    for obj in selection_masks:
        if obj in ["beam", "null_pfo", "fiducial"]: continue
        counts[f"n_{obj}"] = SelectionTools.GetPFOCounts(selection_masks[obj][events.filename])
    if region_type is None:
        region_def = args_c["region_identification"]
    else:
        region_def = RegionIdentification.regions[region_type]
    reco_regions = RegionIdentification.CreateRegionIdentification(region_def, **counts, removed = removed)


    if is_mc:
        events_copy = events.Filter(returnCopy = True)
        
        if "fiducial" in selection_masks and (len(selection_masks["fiducial"]) > 0):
            mask = SelectionTools.CombineMasks(selection_masks["fiducial"][events_copy.filename])
            events_copy.Filter([mask], [mask])

        # is_pip = events_copy.trueParticles.pdg[:, 0] == 211

        mask = SelectionTools.CombineMasks(selection_masks["beam"][events_copy.filename])

        n_pi_true, n_pi0_true = GetTruePionCounts(events_copy, args_c["pi_KE_lim"])
        n_pi_true = n_pi_true[mask]
        n_pi0_true = n_pi0_true[mask]
        # is_pip = is_pip[mask]
        true_regions = RegionIdentification.TrueRegions(n_pi0_true, n_pi_true)
        for k in true_regions:
            true_regions[k] = true_regions[k]# & (is_pip)
        for k in reco_regions:
            reco_regions[k] = reco_regions[k]# & (is_pip)
        return reco_regions, true_regions
    else:
        return reco_regions

@Master.timer
def main(args):
    shower_merging.SetPlotStyle(extend_colors = True)
    outdir = args.out + "region_selection/"
    os.makedirs(outdir, exist_ok = True)

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

    for i in ["pi", "photon", "loose_pi", "loose_photon"]:
        if output_mc[i]:
            MakePFOSelectionPlots(output_mc[i]["data"], output_data[i]["data"] if output_data else None, outdir + "plots/", norm = args.norm, book_name = i)

    if output_mc["loose_pi"]:
        MakePFOSelectionPlotsConsdensed(
            output_mc["pi"]["data"],
            output_mc["loose_pi"]["data"],
            output_data["pi"]["data"] if output_data else None,
            output_data["loose_pi"]["data"] if output_data else None,
            outdir + "plots/",
            norm = args.norm,
            book_name = "pi_both"
            )

    if output_mc["loose_photon"]:
        MakePFOSelectionPlotsConsdensed(
            output_mc["photon"]["data"],
            output_mc["loose_photon"]["data"],
            output_data["photon"]["data"] if output_data else None,
            output_data["loose_photon"]["data"] if output_data else None,
            outdir + "plots/",
            norm = args.norm,
            book_name = "photon_both"
            )

    if output_mc["pi0"]:
        MakePi0SelectionPlots(output_mc["pi0"]["data"], output_data["pi0"]["data"] if output_data else None, outdir + "plots/", norm = args.norm, nbins = 50)
    if output_mc["regions"]:
        MakeRegionPlots(output_mc["regions"], output_data["regions"] if output_data else None, outdir + "plots/")
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