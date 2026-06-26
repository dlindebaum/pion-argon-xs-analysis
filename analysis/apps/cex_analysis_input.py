#!/usr/bin/env python3
"""
Created on: 22/01/2024 20:41

Author: Shyam Bhuller

Description: Create analysis input files from Ntuples.
"""
import awkward as ak
import numpy as np

from python.analysis import cross_section, SelectionTools, PFOSelection, SampleDefinition, Application

from rich import print


def args_to_dict(args : cross_section.argparse.Namespace | dict) -> dict:
    if type(args) == cross_section.argparse.Namespace:
        args_c = vars(args)
    else:
        args_c = args

    return args_c


@cross_section.timer
def BeamPionSelection(events : cross_section.Data, args : cross_section.argparse.Namespace | dict, is_mc : bool) -> cross_section.Data:
    """ Apply beam pion selection to ntuples.

    Args:
        events (cross_section.Data): analysis ntuple
        args (cross_section.argparse.Namespace): analysis configuration
        is_mc (bool): is the ntuple mc or data?

    Returns:
        cross_section.Data: selected events.
    """

    args_c = args_to_dict(args)

    events_copy = events.Filter(returnCopy = True)
    if is_mc:
        selection_args = "mc_arguments"
        sample = "mc"
    else:
        selection_args = "data_arguments"
        sample = "data"

    if "selection_masks" in args:
        masks = args_c["selection_masks"][sample]
        if ("fiducial" in masks) and (len(masks["fiducial"]) > 0):
            mask = SelectionTools.CombineMasks(masks["fiducial"][events.filename])
            events_copy.Filter([mask], [mask])
        mask = SelectionTools.CombineMasks(masks["beam"][events.filename])
        events_copy.Filter([mask], [mask])
    else:
        for s in args_c["beam_selection"]["selections"]:
            mask = args_c["beam_selection"]["selections"][s](events_copy, **args_c["beam_selection"][selection_args][s])
            events_copy.Filter([mask], [mask])

    if "valid_pfo_selection" in args_c:
        if args_c["valid_pfo_selection"] is True:
            if "selection_masks" in args:
                events_copy.Filter([args_c["selection_masks"][sample]['null_pfo'][events.filename]['ValidPFOSelection']]) # apply PFO preselection here
            else:
                events_copy.Filter(PFOSelection.GoodShowerSelection(events))
    return events_copy


@cross_section.timer
def RegionSelection(events : cross_section.Data, args : cross_section.argparse.Namespace | dict, is_mc : bool, region_type : SampleDefinition.SampleDefinition = None, process_type : SampleDefinition.SampleDefinition = None, removed : bool = False) -> dict[np.ndarray]:
    """ Get reco and true regions (if possible) for ntuple.

    Args:
        events (Master.Data): events before beam pion selection.
        args (argparse.Namespace): application arguements
        is_mc (bool): if ntuple is MC or Data.

    Returns:
        tuple[dict, dict]: regions
    """

    args_c = args_to_dict(args)

    if is_mc:
        key = "mc"
    else:
        key = "data"

    selection_masks = args_c["selection_masks"][key]

    events_copy = events.Filter(returnCopy = True)
    
    if "fiducial" in selection_masks and (len(selection_masks["fiducial"]) > 0):
        mask = SelectionTools.CombineMasks(selection_masks["fiducial"][events_copy.filename])
        events_copy.Filter([mask], [mask])

    mask = SelectionTools.CombineMasks(selection_masks["beam"][events_copy.filename])
    events_copy.Filter([mask], [mask])

    # counts = {}
    # for obj in selection_masks:
    #     if obj in ["beam", "null_pfo", "fiducial"]: continue
    #     counts[f"n_{obj}"] = SelectionTools.GetPFOCounts(selection_masks[obj][events.filename])

    if region_type is None:
        region_def = args_c["region_definitions"]
    else:
        region_def = region_type

    reco_regions = region_def.CreateDefinitions(region_def.criteria_list.get_criteria_values(events_copy, selection_masks, **args["region_args"]), uncategorised = removed)
    
    if is_mc:
        if process_type is None:
            process_def = args_c["process_definitions"]
        else:
            process_def = process_type

        true_regions = process_def.CreateDefinitions(process_def.criteria_list.get_criteria_values(events_copy, **args["process_args"]), uncategorised = removed)

        for k in true_regions:
            true_regions[k] = true_regions[k]
        for k in reco_regions:
            reco_regions[k] = reco_regions[k]
        return reco_regions, true_regions
    else:
        return reco_regions


def CreateAnalysisInput(sample : cross_section.Data, args : cross_section.argparse.Namespace | dict, is_mc : bool) -> cross_section.AnalysisInput:
    """ Create analysis input from ntuple sample.

    Args:
        sample (cross_section.Data): sample
        args (cross_section.argparse.Namespace): analysis configurations
        is_mc (bool): is the sample mc?

    Returns:
        cross_section.AnalysisInput: analysis input.
    """
    args_c = args_to_dict(args)

    if type(sample) == cross_section.Toy:
        ai = cross_section.AnalysisInput.CreateAnalysisInputToy(sample)
    elif type(sample) == cross_section.Data:
        sample_selected = BeamPionSelection(sample, args_c, is_mc)
        if is_mc:
            reco_regions, true_regions = RegionSelection(sample, args_c, True, removed = True)
            reweight_params = [args_c["beam_reweight"]["params"][k]["value"] for k in args_c["beam_reweight"]["params"]]
        else:
            reco_regions = RegionSelection(sample, args_c, False, removed = True)
            true_regions = None
            reweight_params = None
        ai = cross_section.AnalysisInput.CreateAnalysisInputNtuple(sample_selected, args_c["upstream_loss_correction_params"]["value"], reco_regions, true_regions, reweight_params, args_c["beam_reweight"]["strength"], args_c["fiducial_volume"], args_c["upstream_loss_response"])
    else:
        raise Exception(f"object type {type(sample)} not a valid sample")
    return ai


def CreateAnalysisInputMCTrueBeam(mc : cross_section.Data, args : cross_section.argparse.Namespace | dict, uncategorised : bool = False):
    args_c = args_to_dict(args)

    masks = [mc.trueParticles.pdg[:, 0] == 211]
    #! mc true beam does not encorperate fiducial cuts in truth, as this loss in efficiency needs to be corrected for the final cross section measurement
    #! if a particle interacted outside the fiducial region, it was still incident on slices within the fiducial region
    mc_true_beam = mc.Filter(masks, masks, True)

    process_def = args_c["process_definitions"]
    true_regions = process_def.CreateDefinitions(process_def.criteria_list.get_criteria_values(mc_true_beam, **args["process_args"]), uncategorised = uncategorised)

    return cross_section.AnalysisInput.CreateAnalysisInputNtuple(mc_true_beam, args_c["upstream_loss_correction_params"]["value"], None, true_regions, [args["beam_reweight"]["params"][k]["value"] for k in args_c["beam_reweight"]["params"]], args_c["beam_reweight"]["strength"], upstream_loss_func = args_c["upstream_loss_response"])


def run(i : int, file_desc : cross_section.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    events = cross_section.Data(file_desc, n_events, start)

    analysis_input_s = CreateAnalysisInput(events, args, not args["data"])
    if args["data"] == False:
        analysis_input_cheated = CreateAnalysisInputMCTrueBeam(events, args) # truth beam (reco regions won't work)
    else:
        analysis_input_cheated = None
    return {"selected" : analysis_input_s, "cheated" : analysis_input_cheated}


def main(args):
    out = args.out + "analysis_input/"
    cross_section.os.makedirs(out, exist_ok = True)

    output_mc = cross_section.RunProcess(args.ntuple_files["mc"], False, args, run, False)
    output_data = cross_section.RunProcess(args.ntuple_files["data"], True, args, run, False)

    ais = {
        "mc_selected" : cross_section.AnalysisInput.Concatenate([mc["selected"] for mc in output_mc]),
        "mc_cheated" : cross_section.AnalysisInput.Concatenate([mc["cheated"] for mc in output_mc]),
        "data_selected" : cross_section.AnalysisInput.Concatenate([data["selected"] for data in output_data])
    }
    for name, ai in ais.items():
        print(f"Writing analysis input file for {name}")
        ai.ToFile(f"{out}analysis_input_{name}.dill")

        if args.root is True:
            ai.ToSplitROOTFiles(out, name)
    return


if __name__ == "__main__":

    parser = cross_section.argparse.ArgumentParser("Create analysis input files from Ntuples.")
    Application.ApplicationArguments.Config(parser)
    Application.ApplicationArguments.Output(parser)
    Application.ApplicationArguments.Processing(parser)
    parser.add_argument("-R", "--ROOT", dest = "root", action="store_true", help = "Saves the output to ROOT files in addition to the dill files.")

    args = Application.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)