#!/usr/bin/env python3
"""
Created on: 02/12/2023 13:14

Author: Shyam Bhuller

Description: reweighting parameters for beam momentum.
"""
import argparse
import os

import numpy as np
import pandas as pd

from rich import print

from python.analysis import application
from python.analysis import common
from python.analysis import fitting
from python.analysis import io
from python.analysis import ntuple
from python.analysis import plotting
from python.analysis import selection
from python.analysis import tags
from python.analysis.beam import reweight
from python.analysis.plotting import plots

def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:

    sample = "data" if args["data"] else "mc"

    selections = args["selection_masks"][sample]

    if "fiducial" in selections:
        if len(selections["fiducial"]) > 0:
            fiducial_mask = selection.CombineMasks(selections["fiducial"][file_desc.file])
        else:
            fiducial_mask = None
    else:
        fiducial_mask = None

    invert = ["HasFinalStatePFOsCut"] # invert preselection

    sideband_selection = {}
    for m in selections["beam"][file_desc.file]:
        if m in invert:
            sideband_selection[m] = ~selections["beam"][file_desc.file][m]
        else:
            sideband_selection[m] = selections["beam"][file_desc.file][m]

    table = {}
    mask = None
    for s in sideband_selection:
        if mask is None:
            mask = sideband_selection[s]
        else:
            mask = mask & sideband_selection[s]
        table[s] = sum(mask)

    print(table)
    sideband_selection = selection.CombineMasks(sideband_selection)

    events = ntuple.Data(file_desc, n_events, start)

    mask = selection.CombineMasks(selections["beam"][file_desc.file])

    if fiducial_mask is not None:
        masks = [fiducial_mask, mask]
    else:
        masks = [mask]

    analysis_sample = events.Filter(masks, masks, returnCopy = True)

    if fiducial_mask is not None:
        masks = [fiducial_mask, sideband_selection]
    else:
        masks = [sideband_selection]

    sideband_sample = events.Filter(masks, masks, returnCopy = True)

    output = {
        "sideband" : {
            "p_inst" : sideband_sample.recoParticles.beam_inst_P,
            "tags" : tags.generate_true_beam_particle_tags(sideband_sample)
        },
        "analysis" : {
            "p_inst" : analysis_sample.recoParticles.beam_inst_P,
            "tags" : tags.generate_true_beam_particle_tags(analysis_sample)
        },
        "table" : table
    }

    return output


@common.timer
def main(args : argparse.Namespace):
    plotting.styles.PlotStyler.SetPlotStyle(extend_colors = True, dpi = 100)
    out = args.out + "beam_reweight/"
    os.makedirs(out, exist_ok = True)

    outputs = application.processing(list(args.ntuple_files.keys()), out, args, run, True)

    for o in outputs:
        for t in outputs[o]["table"]:
            if type(outputs[o]["table"][t]) == list:
                outputs[o]["table"][t] = sum(outputs[o]["table"][t])

    output_mc = outputs["mc"]
    output_data = outputs["data"]

    table_data = pd.DataFrame(output_data["table"], index = ["Counts"]).T
    table_mc = pd.DataFrame(output_mc["table"], index = ["Counts"]).T

    print(table_data)
    print(table_mc)

    table_data.to_hdf(out + "selection_data.hdf5", key = "df")
    table_mc.to_hdf(out + "selection_mc.hdf5", key = "df")
    os.makedirs(out + "plots/", exist_ok = True)

    with plotting.PlotBook(out + "plots/" + "reweight_fits.pdf", True) as book:
        results = reweight.reweight(output_mc["sideband"]["p_inst"], output_data["sideband"]["p_inst"], args.beam_momentum, 20, np.array([0.75, 1.25]), book = book)

    for r in results:
        with plotting.PlotBook(out + "plots/" + f"reweight_results_{r}.pdf", True) as book:
            reweight.reweight_results(output_mc["sideband"], output_data["sideband"], args, 25, results, r, book = book)
            reweight.reweight_results(output_mc["analysis"], output_data["analysis"], args, 50, results, r, book = book)
            reweight_params = {f"p{i}" : {"value" : results[r][0][i], "error" : results[r][1][i]} for i in range(getattr(fitting, r).n_params)}
            io.write_json(reweight_params, out + r + ".json")
        plots.plt.close("all")

    reweight_params = io.read_json(out + "gaussian" + ".json")


    chi2_table = {}
    test_range = [1700, 2300]
    for s, b in zip(["sideband", "analysis"], [25, 50]):
        print(s)
        mc_mom = np.array(output_mc[s]["p_inst"])
        data_mom = np.array(output_data[s]["p_inst"])

        mc_weights = reweight.calculate_weights(mc_mom, "gaussian", [reweight_params[k]["value"] for k in reweight_params], 100)
        mc_weights_truncated = reweight.calculate_weights(mc_mom, "gaussian", [reweight_params[k]["value"] for k in reweight_params], args.beam_reweight["strength"])

        mc_counts, bins = np.histogram(mc_mom, b, test_range)
        data_counts, _ = np.histogram(data_mom, bins, test_range)

        mc_weight_counts, bins = np.histogram(mc_mom, bins, test_range, weights = mc_weights)
        mc_weight_counts_truncated, bins = np.histogram(mc_mom, bins, test_range, weights = mc_weights_truncated)

        chi2_table[s] = {}
        for k, v in zip(["unweighted", "weighted", f"weighted $w < {args.beam_reweight['strength']}$"], [mc_counts, mc_weight_counts, mc_weight_counts_truncated]):
            chi2_table[s][k] = (b - 1) * common.weighted_chi_sqr(data_counts.astype(float), v.astype(float), v.astype(float))

    table = pd.DataFrame(chi2_table)
    table.to_hdf(out + "chi2_reweight.hdf5", key = "df")
    table.to_latex(out + "chi2_reweight.tex")

    return

if __name__ == "__main__":
    args = argparse.ArgumentParser("Calculates reweighting parameters for beam momentum.")
    application.ApplicationArguments.config(args, True)
    application.ApplicationArguments.output(args)
    application.ApplicationArguments.regen(args)
    application.ApplicationArguments.processing(args)

    args = application.ApplicationArguments.resolve_args(args.parse_args())
    print(vars(args))
    main(args)