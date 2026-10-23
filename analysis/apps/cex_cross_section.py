#!/usr/bin/env python3
"""
Created on: 06/10/2026 15:33

Author: Shyam Bhuller

Description: Extracts cross sections from the analysis inputs. Prefits and Postfit cross sections are made.
#! Data fit not implemented yet
"""
import os

import numpy as np

from particle import Particle
from rich import print

from python.analysis import application
from python.analysis import bethe_bloch
from python.analysis import io
from python.analysis import plotting
from python.analysis import samples
from python.analysis import slices
from python.analysis.cross_section import analysis_input, fit_output, plots

def extract_cross_sections(analysis_input : analysis_input.AnalysisInput, energy_slices : slices.Slices, process_definitions : samples.SampleDefinition, fiducial_volume : tuple[float], posterior_tensors : np.ndarray, exclude_fiducial_volume : bool = False):
    dEdX = bethe_bloch.mean_dEdX(energy_slices.pos - energy_slices.width/2, Particle.from_pdgid(211))

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
        slices.energy_slice.counting_experiment_tensor_process(**args)
        )
    }

    interaction_tensor = {k : slices.energy_slice.calculate_interaction_tensor(v) for k, v in counts.items()} # dict[proc_name: np.array.shape(E_init, E_end)], by default it will not create interaction tensors for processes labelled "Impurities".
    interaction_tensor["all"] = np.add(*list(interaction_tensor.values()))
    n_inc = slices.energy_slice.incident(np.sum(interaction_tensor["valid"], 0), np.sum(interaction_tensor["valid"], 1), 0)

    n_end = np.sum(interaction_tensor["valid"], 1)
    n_end_all = np.sum(interaction_tensor["all"], 1)
    n_int_all = {k : np.sum(v + counts["invalid"][k], 1) for k, v in counts["valid"].items()}

    cross_sections = {"total_inelastic" : slices.energy_slice.total_cross_section(n_inc[1:-1], n_end[1:-1], dEdX, energy_slices.width)}
    for k, v in n_int_all.items():
        if k in process_definitions.signal:
            cross_sections[k] = slices.energy_slice.exclusive_cross_section_from_total(n_end_all[1:-1], v[1:-1], *cross_sections["total_inelastic"])
            # cross_sections[k] = slices.energy_slice.exclusive_cross_section(n_inc[1:-1], n_end[1:-1], n_end_all[1:-1], v[1:-1], dEdX, energy_slices.width)
    return cross_sections


def main(args : application.argparse.Namespace):
    plotting.styles.PlotStyler.SetPlotStyle(dark = True)

    ais = {k : io.LoadObject(v) for k, v in args.analysis_input.items() if "mc" in k}

    prefit_cross_sections = extract_cross_sections(ais["mc_cheated"], args.energy_slices, args.process_definitions, args.fiducial_volume, None, True)

    if args.mach3["fit_output"] is not None:
        postfit_tensors = fit_output.MaCh3FitOutput(args.mach3["fit_output"], args.process_definitions, args.energy_slices).posterior_tensors()
        postfit_cross_sections = extract_cross_sections(ais["mc_cheated"], args.energy_slices, args.process_definitions, args.fiducial_volume, postfit_tensors, True)
    else:
        postfit_cross_sections = None

    outdir = args.out + "xs_measurement/"
    os.makedirs(outdir, exist_ok = True)

    with plotting.PlotBook(f"{outdir}cross_section_plots") as book:
        if postfit_cross_sections is None:
            for k, v in prefit_cross_sections.items():
                plots.PlotCrossSection({"prefit" : v}, args.energy_slices, k, [1000, 2000])
                book.Save()
        else:
            for k, v in postfit_cross_sections.items():
                plots.PlotCrossSectionSteps(v, k, prefit_cross_sections[k])
                book.Save()

                plots.PlotCrossSectionMinMax(v, prefit_cross_sections[k], k, args.energy_slices)
                book.Save()

                plots.PlotCrossSectionCL(v, prefit_cross_sections[k], k, args.energy_slices)
                book.Save()
    return


if __name__ == "__main__":

    parser = application.argparse.ArgumentParser(description = "Computes normalisation for beam pion analysis.", formatter_class = application.argparse.RawDescriptionHelpFormatter)

    application.ApplicationArguments.Config(parser, True)
    # application.ApplicationArguments.Processing(parser) do we need multiprocessing? Likely not.
    application.ApplicationArguments.Output(parser)

    args = application.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)