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

from python.analysis.Master import ReadHDF5, LoadConfiguration
from python.analysis.cross_section import ApplicationArguments
from python.analysis import Utils

selection_map = {
    "no selection" : "No selection",
    "APA3Cut" : "Reco beam end $z$ position",
    "TrueFiducialCut" : "True beam end $z$ position",
    "PiBeamSelection" : "Beam trigger",
    "PandoraTagCut" : "Pandora tag",
    "CaloSizeCut" : "Has calorimetry",
    "HasFinalStatePFOsCut" : "Has daughter PFOs",
    "DxyCut" : "$\delta_{xy}$",
    "CosThetaCut" : "$\cos(\\theta)$",
    "MichelScoreCut" : "Michel score",
    "MedianDEdXCut" : "Median $dE/dX$",
    "BeamScraperCut" : "$r_{inst}$",

    "GoodShowerSelection" : "Well reconstructed PFOs",

    "Chi2ProtonSelection" : "$(\chi^{2}/ndf)_{p}$",
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


def selection_names_criteria(df : pd.DataFrame):
    defs = criteria_defs()
    fancy_names = {}
    for n in df.index:
        if defs["beam"] is None: continue
        if n in defs["beam"]:
            cuts = defs["beam"][n]
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
    return fancy_names


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

    total_counts = pd.concat(objs = {k : v[col_map["counts"]][counts_col] for k, v in {"Data" : tables_data, "MC" : tables_mc}.items()}, axis = 1)
    total_counts = total_counts.set_index(names)

    if signal is not None:
        signal_tables = pd.concat({t : tables_mc[t][signal] for t in tables_mc}, axis = 1)

        signal_tables = signal_tables.set_index(names)
    else:
        signal_tables = None

    fancy_names = selection_names_criteria(total_counts)

    if len(fancy_names) > 0:
        total_counts = total_counts.set_index(pd.Series(list(fancy_names.values())))
        if signal_tables is not None:
            signal_tables = signal_tables.set_index(pd.Series(list(fancy_names.values())))

    return signal_tables, total_counts


def FormatTable(filename, bf_column : bool = True):
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
                if bf_column:
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


def bq_xy(values : dict):
    t = {}
    for i in ["mu", "sigma"]:
        for j in ["x", "y"]:
            t[f"$\{i}_{{{j}}}$"] = Utils.round_value_to_error(values[f"{i}_{j}"], values[f"{i}_err_{j}"])

    return t


def bq_angle(values : dict):
    t = {}
    for j in ["x", "y", "z"]:
        v = values[f"mu_dir_{j}"]
        e = values[f"mu_dir_err_{j}"]

        t[f"$\mu_{{\hat{{n}}_{{{j}}}}}$"] = Utils.round_value_to_error(values[f"mu_dir_{j}"], values[f"mu_dir_err_{j}"])
    return t


def brw_table(brw : dict):
    table = {}
    for i, p in enumerate(brw):
        table[f"$p_{{{i}}}$"] = Utils.round_value_to_error(brw[p]["value"], brw[p]["error"])

    table = pd.DataFrame(table, index = [0])
    return table

def upl_table(upl : dict):
    table = {}
    for i in range(args.upstream_loss_response.n_params):
        table[f"$p_{{{i}}}$"] = Utils.round_value_to_error(upl["value"][f"p{i}"], upl["error"][f"p{i}"])
    table = pd.DataFrame(table, index = [0])
    return table

def copy_table(source : str, dest : str, new_name : str = None, bf_cols : bool = True):
    name = source.split("/")[-1]

    if new_name: name = new_name.split(".tex")[0] + ".tex"

    os.makedirs(dest, exist_ok = True)
    with open(source) as f:
        with open(f"{dest}{name}", "w") as of:
            of.writelines(f.readlines())
    FormatTable(f"{dest}{name}", bf_cols)
    return


def brw_selection(path):
    selection_data = ReadHDF5(f"{path}/beam_reweight/selection_data.hdf5")
    selection_mc = ReadHDF5(f"{path}/beam_reweight/selection_mc.hdf5")

    selection_table = pd.concat(objs = [selection_data.rename(columns = {"Counts" : "Data"}), selection_mc.rename(columns = {"Counts" : "MC"})], axis = 1)
    selection_table = selection_table.rename(index = {"TrueFiducialCut" : "Fiducial region", "HasFinalStatePFOsCut" : "Inverted preselection"})
    fancy_names = selection_names_criteria(selection_table)

    selection_table = selection_table.drop(index = ["CaloSizeCut", "PandoraTagCut"])

    selection_table = selection_table.rename(index = fancy_names)
    return selection_table


def main(args : argparse.Namespace):
    path = os.path.abspath(args.workdir + "/")
    out = f"{path}/formatted_tables/"

    os.makedirs(f"{out}upstream/", exist_ok = True)
    upl_table(LoadConfiguration(f"{path}/upstream_loss/fit_parameters.json")).style.hide(axis = "index").to_latex(f"{out}upstream/fit.tex")
    FormatTable(f"{out}upstream/fit.tex", False)

    os.makedirs(f"{out}beam_reweight/", exist_ok = True)
    brw_selection(path).style.to_latex(f"{out}beam_reweight/selection.tex")
    FormatTable(f"{out}beam_reweight/selection.tex")

    brw = brw_table(LoadConfiguration(f"{path}/beam_reweight/gaussian.json"))
    brw.style.hide(axis = "index").to_latex(f"{out}/beam_reweight/fit.tex")
    FormatTable(f"{out}/beam_reweight/fit.tex", False)

    fmt_tables = {}
    for s in signal:
        fmt_tables[s] = CreateTables(path, s, signal[s])

    for f in fmt_tables:
        outp = f"{out}{f}/"
        os.makedirs(outp, exist_ok = True)
        if fmt_tables[f][0] is not None:
            fmt_tables[f][0].style.format(precision = 1).to_latex(f"{outp}/signal_table.tex")
            FormatTable(f"{outp}/signal_table.tex")
        fmt_tables[f][1].style.format(precision = 1).to_latex(f"{outp}/total_counts.tex")
        FormatTable(f"{outp}/total_counts.tex") 


    bq = {i : LoadConfiguration(f"{path}/beam_quality/{i}_beam_quality_fit_values.json") for i in ["mc", "data"]}
    
    outp = f"{out}beam_quality/"
    os.makedirs(outp, exist_ok = True)
    t_xy = pd.DataFrame([bq_xy(bq[m]) for m in ["mc", "data"]], index = ["MC", "Data"])
    t_angle = pd.DataFrame([bq_angle(bq[m]) for m in ["mc", "data"]], index = ["MC", "Data"])
    t_xy.style.to_latex(f"{outp}/xy.tex")
    FormatTable(f"{outp}/xy.tex")
    t_angle.style.to_latex(f"{outp}/angle.tex")
    FormatTable(f"{outp}/angle.tex")

    for f in Utils.ls_recursive(f"{path}/shower_energy_correction/"):
        if ".tex" in f:
            copy_table(f, f"{out}/shower_correction/", bf_cols = False)


    copy_table(f"{path}/toy_parameters/reco_regions/pe.tex", f"{out}/reco_regions/")

    for f in Utils.ls_recursive(f"{path}/toy_parameters/smearing/"):
        if (".tex" in f):
            copy_table(f, f"{out}/smearing/{f.split('/')[-2]}/", bf_cols = False)

    copy_table(f"{path}/toy_parameters/beam_profile/fit.tex", f"{out}/beam_profile/", bf_cols = False)

    copy_table(f"{path}/toy_parameters/reco_regions/pe.tex", f"{out}/reco_regions/")


    copy_table(f"{path}/measurement/pdsp/fit_results_NP.tex", f"{out}/data_fit/", bf_cols = False)
    copy_table(f"{path}/measurement/pdsp/fit_results_POI.tex", f"{out}/data_fit/", bf_cols = False)
    copy_table(f"{path}/measurement/pdsp/regions.tex", f"{out}/")

    for f in Utils.ls_recursive(f'{path}'):
        if out in f: continue
        if "table_processes_fmt.tex" in f:
            copy_table(f, f"{out}", "normalisation_test_summary.tex")
        if "pulls_no_np.tex" in f:
            copy_table(f, f"{out}", "pulls_no_np.tex")
        if "pulls_np.tex" in f:
            copy_table(f, f"{out}", "pulls_np.tex")

    if os.path.isdir(f"{path}/systematics/track_length/"):
        copy_table(f"{path}/systematics/track_length/fit_params.tex", f"{out}/track_length_resolution/", bf_cols = False)

    if os.path.isdir(f"{path}/systematics/combined/"):
        for f in Utils.ls_recursive(f"{path}/systematics/combined/"):
            if ".tex" in f:
                copy_table(f, f"{out}/systematics/", bf_cols = True)

    return


if __name__ == "__main__":

    analysis_options = ["normalisation", "beam_quality", "beam_scraper", "selection", "photon_correction", "reweight", "upstream_correction", "toy_parameters", "analysis_input", "analyse"]

    parser = argparse.ArgumentParser()
    ApplicationArguments.Config(parser)
    parser.add_argument("-w", "--workdir", type = str, default = ".", help = "analysis working directory")

    args = ApplicationArguments.ResolveArgs(parser.parse_args())

    print(vars(args))
    main(args)