#!/usr/bin/env python3
"""
Created on: 06/10/2026 15:33

Author: Shyam Bhuller

Description: Extracts cross sections from the analysis inputs. Prefits and Postfit cross sections are made.
#! Data fit not implemented yet
"""
import os

import numpy as np

from rich import print

from python.analysis import Application, cross_section, Plots, ProcessDefinitions
from python.analysis.MaCh3FitOutput import MaCh3FitOutput



def extract_cross_sections(analysis_input : cross_section.AnalysisInput, energy_slices : cross_section.Slices, process_definitions : ProcessDefinitions.SampleDefinition, fiducial_volume : tuple[float], posterior_tensors : np.ndarray, exclude_fiducial_volume : bool = False):
    dEdX = cross_section.BetheBloch.mean_dEdX(energy_slices.pos - energy_slices.width/2, cross_section.Particle.from_pdgid(211))

    args = {
        "slices" : energy_slices,
        "processes" : analysis_input.exclusive_process,
        "posterior_tensors" : posterior_tensors
    }

    # cross sections can be calculated by either considering the fiducial volume or not
    if exclude_fiducial_volume:
        args = args | {
            "KE_init" : analysis_input.KE_ff_true,
            "KE_end" : analysis_input.KE_int_true,
            "outside_fv" : analysis_input.get_outside_mask(None),
        }
    else:
        args = args | {
            "KE_init" : analysis_input.KE_init_true,
            "KE_end" : analysis_input.KE_end_true,
            "outside_fv" : analysis_input.get_outside_mask(fiducial_volume),
        }


    counts = {k : v for k, v in zip(
        ["valid", "invalid"],
        cross_section.EnergySlice.counting_experiment_tensor_process(**args)
        )
    }

    interaction_tensor = {k : cross_section.EnergySlice.calculate_interaction_tensor(v) for k, v in counts.items()}
    interaction_tensor["all"] = np.add(*list(interaction_tensor.values()))
    n_inc = cross_section.EnergySlice.incident(np.sum(interaction_tensor["valid"], 0), np.sum(interaction_tensor["valid"], 1), 0)

    n_end = np.sum(interaction_tensor["valid"], 1)
    n_end_all = np.sum(interaction_tensor["all"], 1)
    n_int_all = {k : np.sum(v + counts["invalid"][k], 1) for k, v in counts["valid"].items()}

    cross_sections = {"total_inelastic" : cross_section.EnergySlice.total_cross_section(n_inc[1:-1], n_end[1:-1], dEdX, energy_slices.width)}
    for k, v in n_int_all.items():
        if k in process_definitions.signal:
            cross_sections[k] = cross_section.EnergySlice.exclusive_cross_section(n_inc[1:-1], n_end[1:-1], n_end_all[1:-1], v[1:-1], dEdX, energy_slices.width)
    return cross_sections


def main(args : Application.argparse.Namespace):
    cross_section.PlotStyler.SetPlotStyle(dark = True)

    ais = {k : cross_section.LoadObject(v) for k, v in args.analysis_input.items() if "mc" in k}

    prefit_cross_sections = extract_cross_sections(ais["mc_cheated"], args.energy_slices, args.process_definitions, args.fiducial_volume, None, True)

    if args.mach3["fit_output"] is not None:
        postfit_tensors = MaCh3FitOutput(args.mach3["fit_output"], args.process_definitions, args.energy_slices).posterior_tensors()
        postfit_cross_sections = extract_cross_sections(ais["mc_cheated"], args.energy_slices, args.process_definitions, args.fiducial_volume, postfit_tensors, True)
    else:
        postfit_cross_sections = None

    outdir = args.out + "xs_measurement/"
    os.makedirs(outdir, exist_ok = True)

    with Plots.PlotBook(f"{outdir}cross_section_plots") as book:
        if postfit_cross_sections is None:
            for k, v in prefit_cross_sections.items():
                cross_section.PlotCrossSection({"prefit" : v}, args.energy_slices, k, [1000, 2000])
                book.Save()
        else:
            for k, v in postfit_cross_sections.items():
                cross_section.PlotCrossSectionSteps(v, k, prefit_cross_sections[k])
                book.Save()

                cross_section.PlotCrossSectionMinMax(v, prefit_cross_sections[k], k, args.energy_slices)
                book.Save()

                cross_section.PlotCrossSectionCL(v, prefit_cross_sections[k], k, args.energy_slices)
                book.Save()

    return

if __name__ == "__main__":

    parser = Application.argparse.ArgumentParser(description = "Computes normalisation for beam pion analysis.", formatter_class = Application.argparse.RawDescriptionHelpFormatter)

    Application.ApplicationArguments.Config(parser, True)
    # Application.ApplicationArguments.Processing(parser) do we need multiprocessing? Likely not.
    Application.ApplicationArguments.Output(parser)

    args = Application.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)