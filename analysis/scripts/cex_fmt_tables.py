#!/usr/bin/env python3

"""
Created on: 20/04/2024 22:55

Author: Shyam Bhuller

Description: Make nice tables from analysis.
"""

import argparse
import os

import pandas as pd

from rich import print

from python.analysis.Master import ReadHDF5
from python.analysis.cross_section import ApplicationArguments

selection_map = {
    "no selection" : "No selection",
    "APA3Cut" : "Reco beam end $z$ position",
    "TrueFiducialCut" : "True beam end $z$ position",
    "PiBeamSelection" : " Beam trigger",
    "PandoraTagCut" : "Pandora tag",
    "CaloSizeCut" : "Has calorimetry",
    "HasFinalStatePFOsCut" : "has daughter PFOs",
    "DxyCut" : "$\delta_{xy}$",
    "CosThetaCut" : "$\cos(\\theta)$",
    "MichelScoreCut" : "Michel score",
    "MedianDEdXCut" : "Median $dE/dX$",
    "BeamScraperCut" : "$r_{inst}$",

    "GoodShowerSelection" : "Well reconstructed PFOs",

    "Chi2ProtonSelection" : "$\chi^{2}_{p}/ndf$",
    "TrackScoreCut" : "track score",
    "NHitsCut" : "Number of hits",
    "BeamParticleDistanceCut" : "$d$",
    "BeamParticleIPCut" : "$b$",
    "PiPlusSelection" : "Median $dE/dX$",

    "NPhotonCandidateSelection" : "Number of photons",
    "Pi0MassSelection" : "$m_{\gamma\gamma}$",
    "Pi0OpeningAngleSelection" : "$\phi$",
}

units_map = {
    "no selection" : "No selection",
    "APA3Cut" : "cm",
    "TrueFiducialCut" : "cm",
    "PiBeamSelection" : "",
    "PandoraTagCut" : "",
    "CaloSizeCut" : "",
    "HasFinalStatePFOsCut" : "",
    "DxyCut" : "",
    "CosThetaCut" : "",
    "MichelScoreCut" : "",
    "MedianDEdXCut" : "MeV/cm",
    "BeamScraperCut" : "",

    "GoodShowerSelection" : "",

    "Chi2ProtonSelection" : "",
    "TrackScoreCut" : "",
    "NHitsCut" : "",
    "BeamParticleDistanceCut" : "cm",
    "BeamParticleIPCut" : "cm",
    "PiPlusSelection" : "MeV/cm",

    "NPhotonCandidateSelection" : "",
    "Pi0MassSelection" : "MeV",
    "Pi0OpeningAngleSelection" : "deg",
}
selections = {
    "pi0" : "pi0_selection",
    "beam" : "beam_selection",
    "fiducial" : "beam_selection",
    "photon" : "photon_selection",
    "pi" : "piplus_selection",
    "loose_pi" : "loose_pion_selection",
    "loose_photon" : "loose_photon_selection"
}


signal ={
    "fiducial" : "$\\pi^{+}$:inel",
    "beam" : "$\\pi^{+}$:inel",
    "loose_pi" : "$\pi^{\pm}$",
    "pi" : "$\pi^{\pm}$",
    "loose_photon" : "$\gamma$:beam $\pi^0$",
    "photon" : "$\gamma$:beam $\pi^0$",
    "pi0" : "2 $\gamma$'s, same $\pi^{0}$",
    "null_pfo": None
}


def criteria_defs():
    defs = {}
    for k,v in selections.items():
        s_args = getattr(args, v)["mc_arguments"]
        
        c_def = {}
        for s in s_args:
            c_def[s] = {i : s_args[s][i] for i in ["cut", "op"] if i in s_args[s]}
        defs[k] = c_def

    defs["null_pfo"] = None
    return defs 


def fmt_op(op):
    if op == "==":
        return op[0]
    else:
        return op


def CreateTables(path : str, selection_name : str, signal : str = None):
    tables_mc = {}
    tables_data = {}

    col_map = {"counts" : "Counts", "purity": "Purity (\%)", "efficiency" : "Efficiency (\%)"}

    for f in col_map:
        tables_mc[col_map[f]] = ReadHDF5(f"{path}/tables_mc/{selection_name}/{selection_name}_{f}.hdf5")
        tables_data[col_map[f]] = ReadHDF5(f"{path}/tables_data/{selection_name}/{selection_name}_{f}.hdf5")
    names = tables_mc[col_map["counts"]].Name

    tables_mc[col_map["purity"]] = 100 * tables_mc[col_map["purity"]]
    tables_mc[col_map["efficiency"]] = 100 * tables_mc[col_map["efficiency"]]

    if "Remaining PFOs" in tables_mc[col_map["counts"]]:
        counts_col = "Remaining PFOs"
    else:
        counts_col = "Remaining events"

    total_counts = pd.concat({k : v[col_map["counts"]][counts_col] for k, v in {"data" : tables_data, "mc" : tables_mc}.items()}, axis = 1)
    total_counts = total_counts.set_index(names)

    if signal is not None:
        signal_tables = pd.concat({t : tables_mc[t][signal] for t in tables_mc}, axis = 1)

        signal_tables = signal_tables.set_index(names)
    else:
        signal_tables = None

    defs = criteria_defs()

    fancy_names = {}
    for n in total_counts.index:
        if defs[selection_name] is None: continue
        if n in defs[selection_name]:
            cuts = defs[selection_name][n]
        else:
            cuts = None
        
        if n not in selection_map:
            fancy_name = n
        else:
            fancy_name = selection_map[n]

        if (cuts is not None) and (len(cuts) > 0):
            if type(cuts["cut"]) == list:
                fancy_name = f'{min(cuts["cut"])} {units_map[n]} $<$ {fancy_name} $<$ {max(cuts["cut"])} {units_map[n]}'
            else:
                fancy_name = f'{fancy_name} ${fmt_op(cuts["op"])}$ {cuts["cut"]} {units_map[n]}'
        fancy_names[n] = fancy_name

    if len(fancy_names) > 0:
        total_counts = total_counts.set_index(pd.Series(list(fancy_names.values())))
        if signal_tables is not None:
            signal_tables = signal_tables.set_index(pd.Series(list(fancy_names.values())))

    return signal_tables, total_counts


def FormatTable(filename):
    with open(filename) as file:
        lines = list(file.readlines())
        new_lines = []
        for i, l in enumerate(lines):
            if i in [0, len(lines) - 1]:
                new_lines.append(l)
            elif i == 1:
                entries = l.split(" & ")
                formatted = []
                for j in range(len(entries)):
                    if j == len(entries) - 1:
                        s = entries[j].split(" \\\\\n")[0]
                        formatted.append(f"\\textbf{{{s}}}" + " \\\\\n")
                    else:
                        formatted.append(f"\\textbf{{{entries[j]}}} & ")
                formatted = "".join(formatted)
                new_lines.append(formatted)
            else:
                formatted = []
                entries = l.split(" & ")
                entries[0] = f"\\textbf{{{entries[0]}}}"
                for j in range(len(entries)):
                    if j == len(entries) - 1:
                        s = entries[j].split(" \\\\\n")[0]
                        formatted.append(f"{entries[j]}")
                    else:
                        formatted.append(f"{entries[j]} & ")                    
                formatted = "".join(formatted)
                new_lines.append(formatted)

        ruled_lines = []
        for i, l in enumerate(new_lines):
            ruled_lines.append(l)
            if i in [1, len(new_lines) - 2]:
                ruled_lines.append("\\hhline\n")
            elif i in [0, len(new_lines) - 1]:
                continue
            else:
                ruled_lines.append("\\hdashline\n")
    with open(filename, "w") as file:
        file.writelines(ruled_lines)
    return


def main(args : argparse.Namespace):

    fmt_tables = {}
    path = os.path.abspath(args.workdir + "/")
    for s in signal:
        fmt_tables[s] = CreateTables(path, s, signal[s])

    for f in fmt_tables:
        outp = f"{path}/formatted_tables/{f}/"
        os.makedirs(outp, exist_ok = True)
        if fmt_tables[f][0] is not None:
            fmt_tables[f][0].style.format(precision = 1).to_latex(f"{outp}/signal_table.tex")
            FormatTable(f"{outp}/signal_table.tex")
        fmt_tables[f][1].style.format(precision = 1).to_latex(f"{outp}/total_counts.tex")
        FormatTable(f"{outp}/total_counts.tex") 
    
    return


if __name__ == "__main__":

    analysis_options = ["normalisation", "beam_quality", "beam_scraper", "selection", "photon_correction", "reweight", "upstream_correction", "toy_parameters", "analysis_input", "analyse"]

    parser = argparse.ArgumentParser()
    ApplicationArguments.Config(parser)
    parser.add_argument("-w", "--workdir", type = str, default = ".", help = "analysis working directory")

    args = ApplicationArguments.ResolveArgs(parser.parse_args())

    print(vars(args))
    main(args)