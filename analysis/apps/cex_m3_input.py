#!/usr/bin/env python3
"""
Created on: 21/05/2026 12:26

Author: Shyam Bhuller

Description: Converts Analysis input files to MaCh3 input files. 
"""

import numpy as np

from python.analysis import cross_section

def main(args : cross_section.argparse.Namespace):
    out = args.out + "mach3_input/"
    cross_section.os.makedirs(out, exist_ok = True)

    mc = cross_section.AnalysisInput.FromFile(args.analysis_input["mc"])
    data = cross_section.AnalysisInput.FromFile(args.analysis_input["data"])

    # MaCh3 file requires MC samples with information and the Data histograms.
    for n, m in data.regions.items():
        new_sample = mc.SelectSample(mc.regions[n])
        hists = {f"{n}_DataHist" : np.histogram(np.array(data.KE_int_reco[m]), np.arange(0, 2450, 50))}

        fw = cross_section.IO(f"{out}pdsp_R{n}.root")
        fw.WriteData(vars(new_sample), hists, True)

    return

if __name__ == "__main__":
    parser = cross_section.argparse.ArgumentParser("Create analysis input files from Ntuples.")
    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser)

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)