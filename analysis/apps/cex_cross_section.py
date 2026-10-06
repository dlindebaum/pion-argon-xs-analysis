"""
Created on: 06/10/2026 15:33

Author: Shyam Bhuller

Description: Extracts cross sections from the analysis inputs. Prefits and Postfit cross sections are made.
#! Data fit not implemented yet
"""
import awkward as ak
import numpy as np
import scipy.stats as stats

from python.analysis import Application, cross_section, Master, Plots, ProcessDefinitions
from python.analysis.MaCh3FitOutput import MaCh3FitOutput

def PlotCrossSection(xs : dict[tuple], energy_slices, cross_section_name : str, xs_sim_range : tuple[float] = None):
    geant_xs = cross_section.GeantCrossSections(energy_range=xs_sim_range)

    x = energy_slices.pos - energy_slices.width / 2

    Plots.plt.figure()
    geant_xs.Plot(cross_section_name, simplified_pion_production=True)

    for label, (y, yerr) in xs.items():
        Plots.Plot(
            x,
            y,
            xerr=energy_slices.width / 2,
            yerr=yerr,
            newFigure=False,
            marker="o",
            label=label,
        )
    Plots.plt.legend()
    return


def PlotCrossSectionPosteriorRange(xs_tensor, xs_prefit, plot_name, energy_slices):
    xs_central, xs_err = xs_tensor

    max_xs = ak.max(xs_central, 1)
    max_xs_err = ak.ravel(
        ak.Array(xs_err[ak.local_index(xs_err, axis=1) == ak.argmax(xs_central, 1, keepdims=True)])
    )
    min_xs = ak.min(xs_central, 1)
    min_xs_err = ak.ravel(
        ak.Array(xs_err[ak.local_index(xs_err, axis=1) == ak.argmin(xs_central, 1, keepdims=True)])
    )

    series = {
        "prefit" : xs_prefit,
        "minimum" : (min_xs, min_xs_err),
        "maximum" : (max_xs, max_xs_err),
    }
    PlotCrossSection(series, energy_slices, plot_name, [1000, 2000])

    return min_xs, max_xs, min_xs_err, max_xs_err


def PlotCrossSectionCL(xs_tensor, xs_prefit, plot_name, energy_slices, confidence_level : float = 0.68):
    xs_central, xs_err = xs_tensor # neglect the stat error propagation.

    mean_xs = np.mean(xs_central, axis=1)
    ci = stats.t.interval(confidence_level, df=len(xs_central)-1, loc=mean_xs, scale=np.std(xs_central, axis=1, ddof=1) / np.sqrt(len(xs_central)))

    series = {
        "prefit" : xs_prefit,
        "postfit" : (mean_xs, abs(mean_xs - np.abs(ci))),
    }
    PlotCrossSection(series, energy_slices, plot_name, [1000, 2000])
    return


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
    cross_section.PlotStyler.SetPlotStyle(extend_colors = True, dark = True)

    ais = {k : cross_section.LoadObject(v) for k, v in args.analysis_input.items() if "mc" in k}

    prefit_cross_sections = extract_cross_sections(ais["mc_cheated"], args.energy_slices, args.process_definitions, args.fiducial_volume, None)

    return

if __name__ == "__main__":

    parser = Application.argparse.argparse.ArgumentParser(description = "Computes normalisation for beam pion analysis.", formatter_class = Application.argparse.parser.RawDescriptionHelpFormatter)

    Application.ApplicationArguments.Config(parser, True)
    # Application.ApplicationArguments.Processing(parser) do we need multiprocessing? Likely not.
    Application.ApplicationArguments.Output(parser)

    Application.ApplicationArguments()
    main()