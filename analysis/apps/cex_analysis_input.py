#!/usr/bin/env python3
"""
Created on: 22/01/2024 20:41

Author: Shyam Bhuller

Description: Create analysis input files from Ntuples.
"""

from python.analysis import cross_section

from apps import cex_analyse

from rich import print


def CreateAnalysisInputMCTrueBeam(mc : cross_section.Data, args : cross_section.argparse.Namespace):
    is_pip = mc.trueParticles.pdg[:, 0] == 211
    mc_true_beam = mc.Filter([is_pip], [is_pip], True)

    n_pi =  cross_section.EventSelection.SelectionTools.GetPFOCounts(args.selection_masks["mc"]["pi"])
    n_pi0 = cross_section.EventSelection.SelectionTools.GetPFOCounts(args.selection_masks["mc"]["pi0"])
    reco_regions = cross_section.EventSelection.create_regions_new(n_pi0, n_pi)


    n_pi_true = mc_true_beam.trueParticles.nPiMinus + mc_true_beam.trueParticles.nPiPlus
    n_pi0_true = mc_true_beam.trueParticles.nPi0
    true_regions = cross_section.EventSelection.create_regions_new(n_pi0_true, n_pi_true)

    return cross_section.AnalysisInput.CreateAnalysisInputNtuple(mc_true_beam, args.upstream_loss_correction_params["value"], reco_regions, true_regions, args.beam_reweight_params)


def CreateAnalysisInputs(args):
    out = args.out + "analysis_input/"
    cross_section.os.makedirs(out, exist_ok = True)
    mc = cross_section.Data(args.mc_file, nTuple_type = args.ntuple_type)
    analysis_input_mc_s = cex_analyse.CreateAnalysisInput(mc, args, True) # beam particle selection
    analysis_input_mc = CreateAnalysisInputMCTrueBeam(mc, args) # truth beam (reco regions won't work)

    analysis_input_mc.ToFile(f"{out}analysis_input_{args.p_mom}GeV_mc_cheated.dill")
    analysis_input_mc_s.ToFile(f"{out}analysis_input_{args.p_mom}GeV_mc_selected.dill")

    if args.data_file is not None:
        data = cross_section.Data(args.data_file, nTuple_type = args.ntuple_type)
        analysis_input_data_s = cex_analyse.CreateAnalysisInput(data, args, False)
        analysis_input_data_s.ToFile(f"{out}analysis_input_{args.p_mom}GeV_data_selected.dill")
    return


def main(args):
    CreateAnalysisInputs(args)
    return


if __name__ == "__main__":

    parser = cross_section.argparse.ArgumentParser("Create analysis input files from Ntuples.")
    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser)

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)