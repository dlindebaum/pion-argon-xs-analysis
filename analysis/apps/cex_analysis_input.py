
#!/usr/bin/env python3
"""
Created on: 22/01/2024 20:41

Author: Shyam Bhuller

Description: Create analysis input files from Ntuples.
"""
import argparse
import awkward as ak
import numpy as np
import warnings

from python.analysis import (
    Master, cross_section, SelectionTools, RegionIdentification,
    PFOSelection, Processing, EnergyTools, Utils)
from python.gnn.DataPreparation import make_evt_ids
from python.analysis.AnalysisInputs import AnalysisInput, AnalysisInputGNN
from apps.cex_beam_selection_studies import BeamPionSelection
from apps.cex_region_selection_studies import RegionSelection
from apps.cex_gnn_predictions import get_gnn_results, get_truth_regions

from rich import print

def CreateAnalysisInput(
        sample : Master.Data,
        args : argparse.Namespace | dict,
        is_mc : bool) -> AnalysisInput:
    """ Create analysis input from ntuple sample

    Args:
        sample (Master.Data): sample
        args (argparse.Namespace): analysis configurations
        is_mc (bool): is the sample mc?

    Returns:
        AnalysisInputs.AnalysisInput: analysis input.
    """
    args_c = Utils.args_to_dict(args)

    if type(sample) == cross_section.Toy:
        ai = AnalysisInput.CreateAnalysisInputToy(sample)
    elif type(sample) == Master.Data:
        sample_selected = BeamPionSelection(sample, args_c, is_mc)
        if is_mc:
            reco_regions, true_regions = RegionSelection(sample, args_c, True)
            reweight_params = [args_c["beam_reweight"]["params"][k]["value"] for k in args_c["beam_reweight"]["params"]]
        else:
            reco_regions = RegionSelection(sample, args_c, False)
            true_regions = None
            reweight_params = None
        ai = AnalysisInput.CreateAnalysisInputNtuple(
            sample_selected,
            args_c["upstream_loss_correction_params"]["value"],
            reco_regions,
            true_regions,
            reweight_params,
            args_c["beam_reweight"]["strength"],
            args_c["fiducial_volume"],
            args_c["upstream_loss_response"])
    else:
        raise Exception(f"object type {type(sample)} not a valid sample")
    return ai


def CreateGNNAnalysisInput(
        sample : Master.Data,
        args : argparse.Namespace | dict,
        is_mc : bool) -> AnalysisInput:
    """ Create analysis input from ntuple sample

    Args:
        sample (Master.Data): sample
        args (argparse.Namespace): analysis configurations
        is_mc (bool): is the sample mc?

    Returns:
        AnalysisInputs.AnalysisInput: analysis input.
    """
    args_c = Utils.args_to_dict(args)

    if args_c["train_sample"]:
        return None

    if type(sample) == cross_section.Toy:
        raise NotImplementedError("Not implemented GNN toys")
    elif type(sample) == Master.Data:
        sample_selected = BeamPionSelection(sample, args_c, is_mc)
        gnn_predictions, ids = get_gnn_results(sample, args_c, is_mc)
        # Redundant, checked by get_gnn_results, but very bad if wrong
        if not np.all(make_evt_ids(sample_selected) == ids):
            raise Exception("Cannot match predictions to event IDs, "
                            + f"file: {sample_selected.filename}")
        if is_mc:
            true_regions = get_truth_regions(sample, args_c)
            reweight_params = [args_c["beam_reweight"]["params"][k]["value"] for k in args_c["beam_reweight"]["params"]]
        else:
            true_regions = None
            reweight_params = None
        ai = AnalysisInputGNN.CreateAnalysisInputNtuple(
            sample_selected,
            args_c["upstream_loss_correction_params"]["value"],
            gnn_predictions,
            args_c["gnn_region_labels"],
            true_regions,
            reweight_params,
            args_c["beam_reweight"]["strength"],
            args_c["fiducial_volume"],
            args_c["upstream_loss_response"])
    else:
        raise Exception(f"object type {type(sample)} not a valid sample")
    return ai


def GetTruePionCounts(events : Master.Data, ke_lim : float = 0):
    n_pi_true = (events.trueParticles.number != 1) & (abs(events.trueParticles.pdg) == 211) & (events.trueParticles.mother == 1)

    ke = EnergyTools.KE(cross_section.vector.magnitude(events.trueParticles.momentum), cross_section.Particle.from_pdgid(211).mass)

    n_pi_true = ak.sum(n_pi_true & (ke > ke_lim), axis = -1)
    n_pi0_true = events.trueParticles.nPi0

    return n_pi_true, n_pi0_true


def CreateAnalysisInputMCTrueBeam(
        mc : Master.Data,
        args : argparse.Namespace | dict):
    args_c = Utils.args_to_dict(args)

    is_pip = mc.trueParticles.pdg[:, 0] == 211
    masks = [is_pip]

    #! mc true beam does not encorperate fiducial cuts in truth, as this loss in efficiency needs to be corrected for the final cross section measurement
    #! if a particle interacted outside the fiducial region, it was still incident on slices within the fiducial region
    # if "fiducial" in args.selection_masks["mc"]:
    #     if "TrueFiducialCut" in args.selection_masks["mc"]["fiducial"]:
    #         masks.insert(0, args.selection_masks["mc"]["fiducial"]["TrueFiducialCut"])
    mc_true_beam = mc.Filter(masks, masks, True)

    #! this is just a placeholder to populate reco regions
    n_pi =  cross_section.EventSelection.SelectionTools.GetPFOCounts(args_c["selection_masks"]["mc"]["pi"][mc.filename])
    n_pi0 = cross_section.EventSelection.SelectionTools.GetPFOCounts(args_c["selection_masks"]["mc"]["pi0"][mc.filename])
    reco_regions = RegionIdentification.TrueRegions(n_pi0, n_pi)

    n_pi_true, n_pi0_true = GetTruePionCounts(mc_true_beam, args_c["pi_KE_lim"])
    true_regions = RegionIdentification.TrueRegions(n_pi0_true, n_pi_true)

    return AnalysisInput.CreateAnalysisInputNtuple(
        mc_true_beam,
        args_c["upstream_loss_correction_params"]["value"],
        reco_regions,
        true_regions,
        [args_c["beam_reweight"]["params"][k]["value"]
         for k in args_c["beam_reweight"]["params"]],
        args_c["beam_reweight"]["strength"],
        upstream_loss_func = args_c["upstream_loss_response"])

def run(i : int, file : str, n_events : int, start : int, selected_events, args : dict):
    events = Master.Data(file, n_events, start, args["nTuple_type"], args["pmom"])
    if args["gnn_do_predict"]:
        analysis_input_s = CreateGNNAnalysisInput(events, args, not args["data"])    
        if args["data"] == False:
            # analysis_input_cheated = CreateAnalysisInputMCTrueBeam(events, args) # truth beam (reco regions won't work)
            warnings.warn("Not generated true beam graphs, does not have cheat beam information")
            analysis_input_cheated = None
        else:
            analysis_input_cheated = None
    else:
        analysis_input_s = CreateAnalysisInput(events, args, not args["data"])
        if args["data"] == False:
            analysis_input_cheated = CreateAnalysisInputMCTrueBeam(events, args) # truth beam (reco regions won't work)
        else:
            analysis_input_cheated = None
    return {"selected" : analysis_input_s, "cheated" : analysis_input_cheated}


def main(args):
    out = args.out + "analysis_input/"
    cross_section.os.makedirs(out, exist_ok = True)

    args.batches = None
    args.events = None
    args.threads = 1

    output_mc = Processing.RunProcess(args.ntuple_files["mc"], False, args, run, False)
    output_data = Processing.RunProcess(args.ntuple_files["data"], True, args, run, False)

    if args.gnn_do_predict:
        ai_type = AnalysisInputGNN
    else:
        ai_type= AnalysisInput
    ai_mc_selected = ai_type.Concatenate([mc["selected"] for mc in output_mc])
    cheat_out = [mc["cheated"] for mc in output_mc if mc["cheated"] is not None]
    if len(cheat_out) != 0:
        ai_mc_cheated = ai_type.Concatenate(cheat_out)
        ai_mc_cheated.ToFile(f"{out}analysis_input_mc_cheated.dill")
    
    ai_data_selected = ai_type.Concatenate([data["selected"] for data in output_data])

    ai_mc_selected.ToFile(f"{out}analysis_input_mc_selected.dill")
    ai_data_selected.ToFile(f"{out}analysis_input_data_selected.dill")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Create analysis input files from Ntuples.")
    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser)

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)
