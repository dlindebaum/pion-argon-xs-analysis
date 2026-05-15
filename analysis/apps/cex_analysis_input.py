#!/usr/bin/env python3
"""
Created on: 22/01/2024 20:41

Author: Shyam Bhuller

Description: Create analysis input files from Ntuples.
"""
import awkward as ak
import numpy as np

from python.analysis import cross_section, SelectionTools, PFOSelection, RegionDefinitions, ProcessDefinitions

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
def RegionSelection(events : cross_section.Data, args : cross_section.argparse.Namespace | dict, is_mc : bool, region_type : str = None, process_type : str = None, removed : bool = False) -> dict[np.ndarray]:
    """ Get reco and true regions (if possible) for ntuple.

    Args:
        events (Master.Data): events after beam pion selection
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

    counts = {}
    for obj in selection_masks:
        if obj in ["beam", "null_pfo", "fiducial"]: continue
        counts[f"n_{obj}"] = SelectionTools.GetPFOCounts(selection_masks[obj][events.filename])

    if region_type is None:
        region_def = args_c["region_definitions"]
    else:
        region_def = RegionDefinitions.regions[region_type]

    reco_regions = region_def.CreateDefinitions(counts, uncategorised = removed)
    

    if is_mc:
        if process_type is None:
            process_def = args_c["process_definitions"]
        else:
            process_def = ProcessDefinitions.processes[process_type]

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

        true_regions = process_def.CreateDefinitions({"n_pi" : n_pi_true, "n_pi0" : n_pi0_true}, uncategorised = removed)
        for k in true_regions:
            true_regions[k] = true_regions[k]# & (is_pip)
        for k in reco_regions:
            reco_regions[k] = reco_regions[k]# & (is_pip)
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
            reco_regions, true_regions = RegionSelection(sample, args_c, True)
            reweight_params = [args_c["beam_reweight"]["params"][k]["value"] for k in args_c["beam_reweight"]["params"]]
        else:
            reco_regions = RegionSelection(sample, args_c, False)
            true_regions = None
            reweight_params = None
        ai = cross_section.AnalysisInput.CreateAnalysisInputNtuple(sample_selected, args_c["upstream_loss_correction_params"]["value"], reco_regions, true_regions, reweight_params, args_c["beam_reweight"]["strength"], args_c["fiducial_volume"], args_c["upstream_loss_response"])
    else:
        raise Exception(f"object type {type(sample)} not a valid sample")
    return ai


def GetTruePionCounts(events : cross_section.Data, ke_lim : float = 0):
    n_pi_true = (events.trueParticles.number != 1) & (abs(events.trueParticles.pdg) == 211) & (events.trueParticles.mother == 1)

    ke = cross_section.KE(cross_section.vector.magnitude(events.trueParticles.momentum), cross_section.Particle.from_pdgid(211).mass)

    n_pi_true = ak.sum(n_pi_true & (ke > ke_lim), axis = -1)
    n_pi0_true = events.trueParticles.nPi0

    return n_pi_true, n_pi0_true


def CreateAnalysisInputMCTrueBeam(mc : cross_section.Data, args : cross_section.argparse.Namespace | dict, uncategorised : bool = False):
    args_c = args_to_dict(args)

    is_pip = mc.trueParticles.pdg[:, 0] == 211
    masks = [is_pip]

    #! mc true beam does not encorperate fiducial cuts in truth, as this loss in efficiency needs to be corrected for the final cross section measurement
    #! if a particle interacted outside the fiducial region, it was still incident on slices within the fiducial region
    # if "fiducial" in args.selection_masks["mc"]:
    #     if "TrueFiducialCut" in args.selection_masks["mc"]["fiducial"]:
    #         masks.insert(0, args.selection_masks["mc"]["fiducial"]["TrueFiducialCut"])
    mc_true_beam = mc.Filter(masks, masks, True)

    process_defs = args_c["process_definitions"]
    n_pi_true, n_pi0_true = GetTruePionCounts(mc_true_beam, args_c["pi_KE_lim"])
    true_regions = process_defs.CreateDefinitions({"n_pi" : n_pi_true, "n_pi0" : n_pi0_true}, uncategorised = uncategorised)

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
    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Processing(parser)
    parser.add_argument("-R", "--ROOT", dest = "root", action="store_true", help = "Saves the output to ROOT files in addition to the dill files.")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)