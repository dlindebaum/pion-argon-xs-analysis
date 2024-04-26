#!/usr/bin/env python3
"""
Created on: 22/01/2024 20:41

Author: Shyam Bhuller

Description: Create analysis input files from Ntuples.
"""
import numpy as np

from python.analysis import cross_section, SelectionTools, RegionIdentification

from rich import print

@cross_section.timer
def BeamPionSelection(events : cross_section.Data, args : cross_section.argparse.Namespace, is_mc : bool) -> cross_section.Data:
    """ Apply beam pion selection to ntuples.

    Args:
        events (cross_section.Data): analysis ntuple
        args (cross_section.argparse.Namespace): analysis configuration
        is_mc (bool): is the ntuple mc or data?

    Returns:
        cross_section.Data: selected events.
    """
    events_copy = events.Filter(returnCopy = True)
    if is_mc:
        selection_args = "mc_arguments"
        sample = "mc"
    else:
        selection_args = "data_arguments"
        sample = "data"


    if "selection_masks" in args:
        if "fiducial" in args.selection_masks[sample]:
            mask = SelectionTools.CombineMasks(args.selection_masks[sample]["fiducial"])
            events_copy.Filter([mask], [mask])

        mask = SelectionTools.CombineMasks(args.selection_masks[sample]["beam"])
        events_copy.Filter([mask], [mask])
    else:
        for s in args.beam_selection["selections"]:
            mask = args.beam_selection["selections"][s](events_copy, **args.beam_selection[selection_args][s])
            events_copy.Filter([mask], [mask])
            print(events_copy.cutTable.get_table())

    if hasattr(args, "valid_pfo_selection"):
        if args.valid_pfo_selection is True:
            events_copy.Filter([args.selection_masks[sample]['null_pfo']['ValidPFOSelection']]) # apply PFO preselection here
    return events_copy


@cross_section.timer
def RegionSelection(events : cross_section.Data, args : cross_section.argparse.Namespace, is_mc : bool, region_type : str = None, removed : bool = False) -> dict[np.ndarray]:
    """ Get reco and true regions (if possible) for ntuple.

    Args:
        events (Master.Data): events after beam pion selection
        args (argparse.Namespace): application arguements
        is_mc (bool): if ntuple is MC or Data.

    Returns:
        tuple[dict, dict]: regions
    """
    if is_mc:
        key = "mc"
    else:
        key = "data"
    counts = {}
    for obj in args.selection_masks[key]:
        if obj in ["beam", "null_pfo", "fiducial"]: continue
        counts[f"n_{obj}"] = SelectionTools.GetPFOCounts(args.selection_masks[key][obj])
    if region_type is None:
        region_def = args.region_identification
    else:
        region_def = RegionIdentification.regions[region_type]
    reco_regions = RegionIdentification.CreateRegionIdentification(region_def, **counts, removed = removed)


    if is_mc:
        events_copy = events.Filter(returnCopy = True)
        
        if "fiducial" in args.selection_masks["mc"]:
            mask = SelectionTools.CombineMasks(args.selection_masks["mc"]["fiducial"])
            events_copy.Filter([mask], [mask])

        n_pi_true = events_copy.trueParticles.nPiMinus + events_copy.trueParticles.nPiPlus
        n_pi0_true = events_copy.trueParticles.nPi0

        is_pip = events_copy.trueParticles.pdg[:, 0] == 211

        mask = SelectionTools.CombineMasks(args.selection_masks["mc"]["beam"])
        n_pi_true = n_pi_true[mask]
        n_pi0_true = n_pi0_true[mask]
        is_pip = is_pip[mask]
        true_regions = RegionIdentification.TrueRegions(n_pi0_true, n_pi_true)
        for k in true_regions:
            true_regions[k] = true_regions[k] & (is_pip)
        for k in reco_regions:
            reco_regions[k] = reco_regions[k] & (is_pip)
        return reco_regions, true_regions
    else:
        return reco_regions


def CreateAnalysisInput(sample : cross_section.Data, args : cross_section.argparse.Namespace, is_mc : bool) -> cross_section.AnalysisInput:
    """ Create analysis input from ntuple sample

    Args:
        sample (cross_section.Data): sample
        args (cross_section.argparse.Namespace): analysis configurations
        is_mc (bool): is the sample mc?

    Returns:
        cross_section.AnalysisInput: analysis input.
    """
    if type(sample) == cross_section.Toy:
        ai = cross_section.AnalysisInput.CreateAnalysisInputToy(sample)
    elif type(sample) == cross_section.Data:
        sample_selected = BeamPionSelection(sample, args, is_mc)
        if is_mc:
            reco_regions, true_regions = RegionSelection(sample, args, True)
            reweight_params = [args.beam_reweight["params"][k]["value"] for k in args.beam_reweight["params"]]
        else:
            reco_regions = RegionSelection(sample, args, False)
            true_regions = None
            reweight_params = None
        ai = cross_section.AnalysisInput.CreateAnalysisInputNtuple(sample_selected, args.upstream_loss_correction_params["value"], reco_regions, true_regions, reweight_params, args.beam_reweight["strength"], args.fiducial_volume, args.upstream_loss_response)
    else:
        raise Exception(f"object type {type(sample)} not a valid sample")
    return ai


def CreateAnalysisInputMCTrueBeam(mc : cross_section.Data, args : cross_section.argparse.Namespace):
    is_pip = mc.trueParticles.pdg[:, 0] == 211
    masks = [is_pip]

    #! mc true beam does not encorperate fiducial cuts in truth, as this loss in efficiency needs to be corrected for the final cross section measurement
    #! if a particle interacted outside the fiducial region, it was still incident on slices within the fiducial region
    # if "fiducial" in args.selection_masks["mc"]:
    #     if "TrueFiducialCut" in args.selection_masks["mc"]["fiducial"]:
    #         masks.insert(0, args.selection_masks["mc"]["fiducial"]["TrueFiducialCut"])
    mc_true_beam = mc.Filter(masks, masks, True)

    #! this is just a placeholder to populate reco regions
    n_pi =  cross_section.EventSelection.SelectionTools.GetPFOCounts(args.selection_masks["mc"]["pi"])
    n_pi0 = cross_section.EventSelection.SelectionTools.GetPFOCounts(args.selection_masks["mc"]["pi0"])
    reco_regions = RegionIdentification.TrueRegions(n_pi0, n_pi)

    n_pi_true = mc_true_beam.trueParticles.nPiMinus + mc_true_beam.trueParticles.nPiPlus
    n_pi0_true = mc_true_beam.trueParticles.nPi0
    true_regions = RegionIdentification.TrueRegions(n_pi0_true, n_pi_true)

    return cross_section.AnalysisInput.CreateAnalysisInputNtuple(mc_true_beam, args.upstream_loss_correction_params["value"], reco_regions, true_regions, [args.beam_reweight["params"][k]["value"] for k in args.beam_reweight["params"]], args.beam_reweight["strength"], upstream_loss_func = args.upstream_loss_response)


def main(args):
    out = args.out + "analysis_input/"
    cross_section.os.makedirs(out, exist_ok = True)
    mc = cross_section.Data(args.mc_file, nTuple_type = args.ntuple_type, target_momentum = args.pmom)
    analysis_input_mc_s = CreateAnalysisInput(mc, args, True) # beam particle selection
    analysis_input_mc = CreateAnalysisInputMCTrueBeam(mc, args) # truth beam (reco regions won't work)

    analysis_input_mc.ToFile(f"{out}analysis_input_mc_cheated.dill")
    analysis_input_mc_s.ToFile(f"{out}analysis_input_mc_selected.dill")

    if args.data_file is not None:
        data = cross_section.Data(args.data_file, nTuple_type = args.ntuple_type)
        analysis_input_data_s = CreateAnalysisInput(data, args, False)
        analysis_input_data_s.ToFile(f"{out}analysis_input_data_selected.dill")
    return


if __name__ == "__main__":

    parser = cross_section.argparse.ArgumentParser("Create analysis input files from Ntuples.")
    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser)

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)