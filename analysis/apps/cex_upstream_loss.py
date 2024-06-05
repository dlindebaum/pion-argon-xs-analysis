#!/usr/bin/env python3
"""
Created on: 26/09/2023 11:55

Author: Shyam Bhuller

Description: Computes the upstream energy loss for beam particles after beam particle selection, then writes the fitted parameters to file.
"""
import argparse
import os

import awkward as ak
import numpy as np
import pandas as pd

from apps.cex_analysis_input import BeamPionSelection
from python.analysis import cross_section, Master, Plots
from rich import print

cv_method = {
    "None" : None,
    "gaussian" : cross_section.Fitting.gaussian,
    "double_gaussian" : cross_section.Fitting.double_gaussian,
    "student_t" : cross_section.Fitting.student_t,
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


def CentralValueEstimation(bins : np.ndarray, KE_reco_inst : np.ndarray, KE_true_ff : np.ndarray, cv_function : cross_section.Fitting.FitFunction = None, weights : np.ndarray = None) -> tuple[np.array, np.array]:
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
        cv = cross_section.Fitting.ExtractCentralValues_df(df, "KE_inst", "residual", [-250, 250], [cv_function], bins, 50, rms_err = False, weights = weights)
    return cv


def run(i : int, file : str, n_events : int, start : int, selected_events, args : dict):
    mc = Master.Data(file, n_events, start, args["nTuple_type"], args["pmom"])
    
    mc = BeamPionSelection(mc, args, True)
    if args["no_reweight"] is False:
        mc_weights = cross_section.RatioWeights(mc.recoParticles.beam_inst_P, "gaussian", [args["beam_reweight"]["params"][k]["value"] for k in args["beam_reweight"]["params"]], args["beam_reweight"]["strength"])
    else:
        mc_weights = None
    
    return {"p_inst" : mc.recoParticles.beam_inst_P, "KE_ff" : mc.trueParticles.beam_KE_front_face, "weights" : mc_weights}


def main(args : argparse.Namespace):
    cross_section.SetPlotStyle(False, dpi = 100)

    args.batches = None
    args.events = None
    args.threads = 1

    os.makedirs(args.out + "upstream_loss/", exist_ok = True)

    if os.path.isfile(args.out + "upstream_loss/output_mc.dill"):
        output_mc = cross_section.LoadObject(args.out + "upstream_loss/output_mc.dill")
    else:
        output_mc = cross_section.RunProcess(args.ntuple_files["mc"], False, args, run)
        cross_section.SaveObject(args.out + "upstream_loss/output_mc.dill", output_mc)

    if all(v is None for v in output_mc["weights"]):
        output_mc["weights"] = None

    bins = ak.Array(args.upstream_loss_bins)
    x = (bins[1:] + bins[:-1]) / 2

    os.makedirs(args.out + "upstream_loss/", exist_ok = True)
    with Plots.PlotBook(args.out + "upstream_loss/" + "cex_upstream_loss_plots.pdf") as pdf:
        reco_KE_inst = cross_section.KE(output_mc["p_inst"], cross_section.Particle.from_pdgid(211).mass)
        cv = CentralValueEstimation(bins, reco_KE_inst, output_mc["KE_ff"], cv_method[args.upstream_loss_cv_function], output_mc["weights"])
        pdf.Save()

        Plots.plt.figure()
        params = cross_section.Fitting.Fit(x, cv[0], cv[1], args.upstream_loss_response, maxfev = int(5E5), plot = True, xlabel = "$KE^{reco}_{inst}$(MeV)", ylabel = "$\mu(KE^{reco}_{inst} - KE^{true}_{init})$(MeV)", loc = "upper center")
        pdf.Save()

        params_dict = {
            "value" : {f"p{i}" : params[0][i] for i in range(len(params[0]))},
            "error" : {f"p{i}" : params[1][i] for i in range(len(params[1]))}
        }

        reco_KE_ff =  reco_KE_inst - cross_section.UpstreamEnergyLoss(reco_KE_inst, params_dict["value"], args.upstream_loss_response)

        Plots.PlotHistComparison([reco_KE_ff, output_mc["KE_ff"]], labels = ["$KE^{reco}_{init}$", "$KE^{true}_{init}$"], x_range = [bins[0], bins[-1]], xlabel = "Kinetic energy (MeV)", weights = [output_mc["weights"], output_mc["weights"]])
        pdf.Save()
    Plots.plt.close("all")
    print(f"fitted parameters : {params_dict}")
    cross_section.SaveConfiguration(params_dict, args.out + "upstream_loss/" + "fit_parameters.json")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Computes the upstream energy loss for beam particles after beam particle selection, then writes the fitted parameters to file.", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser)
    parser.add_argument("--cv_function", dest = "upstream_loss_cv_function", type = str, default = "gaussian", help = "method to extract central value, possible options are ['None', 'gaussian', 'student_t', 'double_gaussian', 'crystal_ball']")
    parser.add_argument("--no_reweight", dest = "no_reweight", action = "store_true", help = "perform correction without reweighted MC")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)