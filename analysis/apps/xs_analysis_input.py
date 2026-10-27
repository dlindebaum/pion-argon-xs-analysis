#!/usr/bin/env python3
"""
Created on: 22/01/2024 20:41

Author: Shyam Bhuller

Description: Create analysis input files from Ntuples.
"""
import argparse
import os


from rich import print

from python.analysis import application
from python.analysis import ntuple
from python.analysis.cross_section import analysis_input


def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    events = ntuple.Data(file_desc, n_events, start)

    analysis_input_s = analysis_input.CreateAnalysisInput(events, args, not args["data"])
    if args["data"] == False:
        analysis_input_cheated = analysis_input.CreateAnalysisInputMCTrueBeam(events, args) # truth beam (reco regions won't work)
    else:
        analysis_input_cheated = None
    return {"selected" : analysis_input_s, "cheated" : analysis_input_cheated}


def main(args):
    out = args.out + "analysis_input/"
    os.makedirs(out, exist_ok = True)

    output_mc = ntuple.RunProcess(args.ntuple_files["mc"], False, args, run, False)
    output_data = ntuple.RunProcess(args.ntuple_files["data"], True, args, run, False)

    ais = {
        "mc_selected" : analysis_input.AnalysisInput.Concatenate([mc["selected"] for mc in output_mc]),
        "mc_cheated" : analysis_input.AnalysisInput.Concatenate([mc["cheated"] for mc in output_mc]),
        "data_selected" : analysis_input.AnalysisInput.Concatenate([data["selected"] for data in output_data])
    }
    for name, ai in ais.items():
        print(f"Writing analysis input file for {name}")
        ai.ToFile(f"{out}analysis_input_{name}.dill")

        if args.root is True:
            ai.ToSplitROOTFiles(out, name)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Create analysis input files from Ntuples.")
    application.ApplicationArguments.Config(parser)
    application.ApplicationArguments.Output(parser)
    application.ApplicationArguments.Processing(parser)
    parser.add_argument("-R", "--ROOT", dest = "root", action="store_true", help = "Saves the output to ROOT files in addition to the dill files.")

    args = application.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)