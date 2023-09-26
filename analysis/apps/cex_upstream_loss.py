#!/usr/bin/env python3
"""
Created on: 26/09/2023 11:55

Author: Shyam Bhuller

Description: Computes the upstream energy loss for beam particles after beam particle selection, then writes the fitted parameters to file.
"""
import argparse
import json
import os

import awkward as ak
import numpy as np
import pandas as pd

from python.analysis import cross_section, Master, Plots
from rich import print

cv_method = {
    "None" : None,
    "gaussian" : cross_section.Fitting.gaussian,
    "double_gaussian" : cross_section.Fitting.double_gaussian,
    "studen_t" : cross_section.Fitting.student_t,
    "crystal_ball" : cross_section.Fitting.crystal_ball,
}


def NumericalCV(bins : np.array, KE_reco_inst : np.array, KE_true_ff : np.array) -> tuple[np.array, np.array]:
    """ Central value in reco bin using the arithmetic mean.

    Args:
        bins (np.array): bin edges
        KE_reco_inst (np.array): reco KE at instrumentation
        KE_true_ff (np.array): true front facing KE

    Returns:
        tuple[np.array, np.array]: arithmetc mean in each bin, error in the mean
    """
    binned_data = {"KE_inst": [], "KEff_true" : [], "KE_first_true" : []}
    for i in range(len(bins)-1):
        mask = (KE_reco_inst > bins[i]) & (KE_reco_inst < bins[i + 1])
        mask = mask & (KE_true_ff > 0)

        binned_data["KE_inst"].append( KE_reco_inst[mask] )
        binned_data["KEff_true"].append( KE_true_ff[mask] )
    binned_data = {i : ak.Array(binned_data[i]) for i in binned_data}

    print(ak.num(binned_data["KE_inst"]))
    residual_energy = binned_data["KE_inst"] - binned_data["KEff_true"]

    mean_residual_energy = ak.mean(residual_energy, axis = -1)
    mean_error_residual_energy = ak.std(residual_energy, axis = -1) / np.sqrt(ak.num(residual_energy))
    return mean_residual_energy, mean_error_residual_energy


def CentralValueEstimation(bins : np.array, KE_reco_inst : np.array, KE_true_ff : np.array, cv_function : cross_section.Fitting.FitFunction = None) -> tuple[np.array, np.array]:
    """ Estiamte upstream loss using a reponse function to correct the reco KE at the instrumentaiton to get the KE at the front face of the TPC.
        estiamtes the central value of residuals in bins of KE_reco_inst using a fitting function.

    Args:
        bins (np.array): reco KE bins
        KE_reco_inst (np.array): reco KE at instrumentation
        KE_true_ff (np.array): true front facing KE
        cv_function (Fitting.FitFunction, optional): function to fit residuals to in order to get the central value. Defaults to None.

    Returns:
        tuple[np.array, np.array]: central values in each bin, error in central values
    """
    if cv_function is None:
        cv = NumericalCV(bins, KE_reco_inst, KE_true_ff)
    else:
        df = pd.DataFrame({"KE_inst" : KE_reco_inst, "true_ffKE" : KE_true_ff})
        df["residual"] = df.KE_inst - df.true_ffKE
        cv = cross_section.Fitting.ExtractCentralValues_df(df, "KE_inst", "residual", [-250, 250], [cv_function], bins, 50, rms_err = False)
    return cv


def main(args : argparse.Namespace):
    cross_section.SetPlotStyle(False, 100)
    mc = Master.Data(args.mc_file, -1, 0, args.ntuple_type)
    for s in args.beam_selection["selections"]:
        mask = args.beam_selection["selections"][s](mc, **args.beam_selection["mc_arguments"][s])
        mc.Filter([mask], [mask])
        print(mc.cutTable.get_table())

    bins = ak.Array(args.UPSTREAM_ENERGY_LOSS["bins"])
    x = (bins[1:] + bins[:-1]) / 2

    os.makedirs(args.out, exist_ok = True)
    with Plots.PdfPages(args.out + "cex_upstream_loss_plots.pdf") as pdf:
        reco_KE_inst = cross_section.KE(mc.recoParticles.beam_inst_P, cross_section.Particle.from_pdgid(211).mass)
        cv = CentralValueEstimation(bins, reco_KE_inst, mc.trueParticles.beam_KE_front_face, cv_method[args.cv_function])
        pdf.savefig()
        
        Plots.plt.figure()
        params = cross_section.Fitting.Fit(x, cv[0], cv[1], cross_section.Fitting.poly2d, maxfev = int(5E5), plot = True, xlabel = "$KE^{reco}_{inst}$(MeV)", ylabel = "$\mu(KE^{reco}_{inst} - KE^{true}_{ff})$(MeV)")
        pdf.savefig()

        reco_KE_ff =  reco_KE_inst - cross_section.UpstreamEnergyLoss(reco_KE_inst, params[0])
        Plots.PlotHistComparison([reco_KE_ff, mc.trueParticles.beam_KE_front_face], labels = ["$KE^{reco}_{ff}$", "$KE^{true}_{ff}$"], x_range = [bins[0], bins[-1]], xlabel = "Kinetic energy (MeV)")
        pdf.savefig()

    params_dict = {
        "value" : {f"p{i}" : params[0][i] for i in range(len(params[0]))},
        "error" : {f"p{i}" : params[1][i] for i in range(len(params[1]))}
    }
    print(f"fitted parameters : {params_dict}")
    with open(args.out + "fit_parameters.json", "w") as f:
        json.dump(params_dict, f)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Computes the upstream energy loss for beam particles after beam particle selection, then writes the fitted parameters to file.", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser)
    parser.add_argument("--cv_function", dest = "cv_function", type = str, default = "gaussian", help = "method to extract central value, possible options are ['None', 'gaussian', 'student_t', 'double_gaussian', 'crystal_ball']")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())
    args.out = args.out + "upstream_loss/"
    print(vars(args))
    main(args)