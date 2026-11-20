#!/usr/bin/env python3
"""
Created on: 22/01/2024 20:41

Author: Shyam Bhuller

Description: Create analysis input files from Ntuples.
"""
import argparse

from rich import print

from python.analysis import application
from python.analysis import ntuple
from python.analysis.cross_section import analysis_input


def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    """ Create selected and truth-beam analysis inputs for one ntuple chunk.

        Args:
            i (int): Index of the chunk being processed.
            file_desc (ntuple.FileDescriptor): Descriptor for the input ntuple file.
            n_events (int): Maximum number of events to read.
            start (int): First event index to read.
            selected_events (Any): Optional event subset supplied by the processing runner.
            args (dict): Resolved analysis configuration, including the data/MC flag.

        Returns:
            dict: Selected and, for MC, truth-beam analysis inputs for the chunk.
    """
    events = ntuple.Data(file_desc, n_events, start)

    analysis_input_s = analysis_input.create_analysis_input(events, args, not args["data"])
    if args["data"] == False:
        analysis_input_cheated = analysis_input.create_analysis_input_mc_true_beam(events, args) # truth beam (reco regions won't work)
    else:
        analysis_input_cheated = None
    return {"selected" : analysis_input_s, "cheated" : analysis_input_cheated}


def merge_analysis_inputs(output : analysis_input.AnalysisInput) -> analysis_input.AnalysisInput:
    """ Merge analysis inputs from the processsing output.

    Args:
        output (analysis_input.AnalysisInput): Processing outputs.

    Returns:
        analysis_input.AnalysisInput: Analysis input.
    """
    if isinstance(output, list):
        return analysis_input.AnalysisInput.concatenate(output)
    else:
        return output

@application.main("analysis_input")
def main(args):
    out = args.out + "analysis_input/"
    outputs = application.processing(list(args.ntuple_files.keys()), "analysis_input", args, run, True, False)

    print(outputs["mc"]["selected"])

    ais = {
        "mc_selected" : merge_analysis_inputs(outputs["mc"]["selected"]),
        "mc_cheated" : merge_analysis_inputs(outputs["mc"]["cheated"]),
        "data_selected" : merge_analysis_inputs(outputs["data"]["selected"])
    }
    hashes = {}
    for name, ai in ais.items():
        print(f"Writing analysis input file for {name}")
        fname = f"analysis_input_{name}.dill"
        hashes[fname] = ai.to_file(f"{out}{fname}")

        if args.root is True: #! remove this option
            hashes[fname] = ai.to_split_root_files(out, fname)

    return hashes

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Create analysis input files from Ntuples.")
    application.ApplicationArguments.config(parser)
    application.ApplicationArguments.output(parser)
    application.ApplicationArguments.processing(parser)
    application.ApplicationArguments.regen(parser)
    parser.add_argument("-R", "--ROOT", dest = "root", action="store_true", help = "Saves the output to ROOT files in addition to the dill files.")

    args = application.ApplicationArguments.resolve_args(parser.parse_args())
    print(vars(args))
    main(args)