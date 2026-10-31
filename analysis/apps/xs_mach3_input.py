#!/usr/bin/env python3
"""
Created on: 21/05/2026 12:26

Author: Shyam Bhuller

Description: Converts Analysis input files to MaCh3 input files. 
"""
import os

import numpy as np

from python.analysis import application, io
from python.analysis.cross_section import analysis_input

def main(args : application.argparse.Namespace):
    out = args.out + "mach3_input/"
    os.makedirs(out, exist_ok = True)

    mc = analysis_input.AnalysisInput.FromFile(args.analysis_input["mc"])
    data = analysis_input.AnalysisInput.FromFile(args.analysis_input["data"])

    # MaCh3 file requires MC samples with information and the Data histograms.
    for n, m in data.regions.items():
        new_sample = mc.SelectSample(mc.regions[n])

        bins = np.arange(
            min(args.mach3["KE_int_binning"]["range"]),
            max(args.mach3["KE_int_binning"]["range"]) + args.mach3["KE_int_binning"]["bin_width"],
            args.mach3["KE_int_binning"]["bin_width"]
            )
        hists = {f"{n}_DataHist" : np.histogram(np.array(data.KE_int_reco[m]), bins)}

        fw = io.IO(f"{out}pdsp_R{n}.root")
        fw.WriteData(vars(new_sample), hists, True)

    return

if __name__ == "__main__":
    parser = application.argparse.ArgumentParser("Create analysis input files from Ntuples.")
    application.ApplicationArguments.Config(parser)
    application.ApplicationArguments.Output(parser)

    args = application.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)