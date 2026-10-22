#!/usr/bin/env python3
"""
Created on: 13/06/2023 11:19

Author: Shyam Bhuller

Description: Produces plots and fits for the beam scraper study, producing thresholds which dictate what events are beam scrapers.
"""
import argparse
import os

from particle import Particle
from rich import print

from python.analysis import application
from python.analysis import common
from python.analysis import kinematics
from python.analysis import io
from python.analysis import ntuple
from python.analysis import plots
from python.analysis import styles
from python.analysis.beam import scraper

def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    mc =  ntuple.Data(file_desc, n_events, start)
    for s in args["beam_selection"]["selections"]:
        if s == "BeamScraperCut": break
        mask = args["beam_selection"]["selections"][s](mc, **args["beam_selection"]["mc_arguments"][s])
        mc.Filter([mask], [mask])

    beam_inst_KE = kinematics.KE(mc.recoParticles.beam_inst_P, Particle.from_pdgid(211).mass) # get kinetic energy from beam instrumentation
    true_ffKE = mc.trueParticles.beam_KE_front_face

    beam_inst_x = mc.recoParticles.beam_inst_pos.x
    beam_inst_y = mc.recoParticles.beam_inst_pos.y

    delta_KE_upstream = beam_inst_KE - true_ffKE

    return {"beam_inst_KE" : beam_inst_KE, "true_ffKE" : true_ffKE, "beam_inst_x" : beam_inst_x, "beam_inst_y" : beam_inst_y, "delta_KE_upstream" : delta_KE_upstream}

@common.timer
def main(args : argparse.Namespace):

    styles.PlotStyler.SetPlotStyle(True)
    outdir = args.out + "beam_scraper/"
    os.makedirs(outdir, exist_ok = True)

    output_mc = application.ApplicationProcessing(["mc"], outdir, args, run, True)["mc"]

    residual_range = [-300, 300] # range of residual for plotsdefine 
    bins = 50

    with plots.PlotBook(outdir + "beam_scraper_fits.pdf") as pdf:
        plots.Plot(args.beam_scraper_energy_range, args.beam_scraper_energy_range, color = "red")
        plots.PlotHist2D(output_mc["beam_inst_KE"], output_mc["true_ffKE"], xlabel = "$KE^{reco}_{inst}$ (MeV)", ylabel = "$KE^{true}_{ff}$ (MeV)", x_range = args.beam_scraper_energy_range, y_range = args.beam_scraper_energy_range, newFigure = False)
        pdf.Save()

        if (args.beam_scraper_energy_bins is None):
            print("Warning: 'energy_bins' for the beam scraper fit was not defined. Beam scraper fit will not be completed (Hint: check the pdf output to help figure out the best values for the energy bins)")
            exit()

        plots.PlotHist2D(output_mc["beam_inst_KE"], output_mc["delta_KE_upstream"], xlabel = "$KE^{reco}_{inst}$ (MeV)", ylabel = "$\\Delta E_{upstream}$ (MeV)", x_range = args.beam_scraper_energy_range, y_range = residual_range)
        for i in args.beam_scraper_energy_bins: plots.plt.axvline(i, color = "red")
        pdf.Save()

        scraper_thresholds = scraper.GetScraperFits(args.beam_scraper_energy_bins, output_mc["beam_inst_KE"], output_mc["delta_KE_upstream"], bins, residual_range)
        pdf.Save()
        print(scraper_thresholds)

        position_means = scraper.BeamScraperPlots(args.beam_scraper_energy_bins, output_mc, scraper_thresholds)
        print(position_means)
        pdf.Save()

    json_dict = {}
    for i, k in enumerate(scraper_thresholds):
        json_dict[str(i)] = {**{"bins" : k}, **scraper_thresholds[k], **position_means[k]}

    name = outdir + "mc_beam_scraper_fit_values.json"
    io.SaveConfiguration(json_dict, name)
    print(f"fit values written to {name}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Calculates parameters required to idenfify beam scrapers and apply the selection.")

    application.ApplicationArguments.Config(parser, required = True)
    application.ApplicationArguments.Processing(parser)
    application.ApplicationArguments.Output(parser)
    application.ApplicationArguments.Regen(parser)

    parser.add_argument("--energy_range", dest = "beam_scraper_energy_range", type = float, nargs = 2, help = "energy range to study (MeV).")
    parser.add_argument("--energy_bins", dest = "beam_scraper_energy_bins", type = float, nargs = 5, help = "kinetic energy bin edges (currently allows only 4 bins to be made) (MeV)")

    args = parser.parse_args()
    args = application.ApplicationArguments.ResolveArgs(args)

    print(vars(args))
    main(args)