#!/usr/bin/env python3
"""
Created on: 26/09/2023 11:55

Author: Shyam Bhuller

Description: Computes the upstream energy loss for beam particles after beam particle selection, then writes the fitted parameters to file.
"""
import argparse
import os

import awkward as ak

from particle import Particle

from python.analysis import application
from python.analysis import fitting
from python.analysis import io
from python.analysis import kinematics
from python.analysis import ntuple
from python.analysis import plotting
from python.analysis.beam import reweight, upstream
from python.analysis.cross_section.analysis_input import BeamPionSelection

from rich import print

def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    mc = ntuple.Data(file_desc, n_events, start)
    
    mc = BeamPionSelection(mc, args, True)
    if args["no_reweight"] is False:
        mc_weights = reweight.RatioWeights(mc.recoParticles.beam_inst_P, "gaussian", [args["beam_reweight"]["params"][k]["value"] for k in args["beam_reweight"]["params"]], args["beam_reweight"]["strength"])
    else:
        mc_weights = None
    
    return {"p_inst" : mc.recoParticles.beam_inst_P, "KE_ff" : mc.trueParticles.beam_KE_front_face, "weights" : mc_weights}


def main(args : argparse.Namespace):
    plotting.styles.PlotStyler.SetPlotStyle(False)
    outdir = args.out + "upstream_loss/"
    os.makedirs(outdir, exist_ok = True)

    output_mc = application.ApplicationProcessing(["mc"], outdir, args, run, True)["mc"]

    if all(v is None for v in output_mc["weights"]):
        output_mc["weights"] = None

    bins = ak.Array(args.upstream_loss_bins)
    x = (bins[1:] + bins[:-1]) / 2

    with plotting.PlotBook(outdir + "xs_upstream_loss_plots.pdf") as pdf:
        reco_KE_inst = kinematics.KE(output_mc["p_inst"], Particle.from_pdgid(211).mass)
        with plotting.styles.PlotStyler().Update(font_scale = 1.3):
            cv = upstream.CentralValueEstimation(bins, reco_KE_inst, output_mc["KE_ff"], upstream.cv_method[args.upstream_loss_cv_function], output_mc["weights"])
        pdf.Save()

        plotting.plots.plt.figure()
        params = fitting.Fit(x, cv[0], cv[1], args.upstream_loss_response, maxfev = int(5E5), plot = True, xlabel = "$KE^{reco}_{inst}$ (MeV)", ylabel = "$\\mu(\Delta E_{upstream})$ (MeV)", loc = "upper center")
        pdf.Save()

        params_dict = {
            "value" : {f"p{i}" : params[0][i] for i in range(len(params[0]))},
            "error" : {f"p{i}" : params[1][i] for i in range(len(params[1]))}
        }

        reco_KE_ff =  reco_KE_inst - upstream.UpstreamEnergyLoss(reco_KE_inst, params_dict["value"], args.upstream_loss_response)

        plotting.plots.PlotHistComparison([reco_KE_ff, output_mc["KE_ff"]], labels = ["$KE^{reco}_{init}$", "$KE^{true}_{init}$"], x_range = [bins[0], bins[-1]], xlabel = "Kinetic energy (MeV)", weights = [output_mc["weights"], output_mc["weights"]])
        pdf.Save()
    plotting.plots.plt.close("all")
    print(f"fitted parameters : {params_dict}")
    io.SaveConfiguration(params_dict, outdir + "fit_parameters.json")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Computes the upstream energy loss for beam particles after beam particle selection, then writes the fitted parameters to file.", formatter_class = argparse.RawDescriptionHelpFormatter)

    application.ApplicationArguments.Config(parser, True)
    application.ApplicationArguments.Output(parser)
    application.ApplicationArguments.Regen(parser)
    application.ApplicationArguments.Processing(parser)
    parser.add_argument("--cv_function", dest = "upstream_loss_cv_function", type = str, default = "gaussian", help = "method to extract central value, possible options are ['None', 'gaussian', 'student_t', 'double_gaussian', 'crystal_ball']")
    parser.add_argument("--no_reweight", dest = "no_reweight", action = "store_true", help = "perform correction without reweighted MC")

    args = application.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)