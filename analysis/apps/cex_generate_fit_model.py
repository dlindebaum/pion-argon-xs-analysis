#!/usr/bin/env python3
"""
Created on: 21/09/2026 15:21

Author: Shyam Bhuller

Description: Script that creates the systematics.yaml file for MaCh3 that defines the binning.
"""

from python.analysis import Application, cross_section, ProcessDefinitions
import numpy as np

ENERGY_BINS = {
    "Under": ("0.", "1500"),
    "0": ("1500", "1700"),
    "1": ("1700", "1900"),
    "2": ("1900", "2100"),
    "Over": ("2100", "999999"),
}

MODES = {
    "Abs": 0,
    "CEx": 1,
    "Pip": 2,
    "Decay": 3,
    "Esc": 4,
    "Impure": 999,
}

END_Z = {
    "Abs": (30, 220),
    "CEx": (30, 220),
    "Pip": (30, 220),
    "Decay": (30, 220),
    "Esc": (220, 999),
    "Impure": (30, 999),
}


def make_systematic(proc_info : tuple, init_bin : tuple, end_bin : tuple):
    init_low, init_high = init_bin[1]
    end_low, end_high = end_bin[1]
    z_low, z_high = proc_info[1]["z_region"]
    mode = proc_info[1]["mode"]

    name = f"True{proc_info[0]}-Init_{init_bin[0]}-End_{end_bin[0]}"

    return f"""  - Systematic:
      Names:
        FancyName: {name}
        ParameterName: {name}

      SampleNames: ["PDSP"]
      Mode: [{mode}]
      Error: 0.1
      FlatPrior: true
      ParameterBounds: [0, 4]
      ParameterGroup: Fit
      KinematicCuts:
        - TrueKEInt: [{init_low}, {init_high}]
        - TrueKEIni: [{end_low}, {end_high}]
        - TrueEndZ: [{z_low}, {z_high}]
      ParameterValues:
        Generator: 1.
        PreFitValue: 1.
      Type: Norm
      StepScale:
        MCMC: 0.01
"""

def slices_to_model_bins(energy_slices : cross_section.Slices) -> dict[tuple]:
    model_bins = energy_slices.edges_all[::-1]
    model_bins = np.where(model_bins == energy_slices.underflow_pos, 0, model_bins)
    model_bins = np.where(model_bins == energy_slices.overflow_pos, 999999, model_bins)

    bins = {}
    for i in range(len(model_bins)-1):
        if i == 0:
            k = "Under"
        elif i == len(model_bins)-2:
            k = "Over"
        else:
            k = str(i)
        bins[k] = (model_bins[i], model_bins[i+1])
    return bins


def get_process_info(process : ProcessDefinitions.SampleDefinition, fiducial_volume : list[float]) -> dict:
    model_process_info = {}

    for k, v in process.definitions.items():
        definitions =  process.convert_process_definitions_str(str(v[0]))
        if "beam_escapes" in definitions:
            if definitions["beam_escapes"]["op"] == "==" and definitions["beam_escapes"]["value"] == 0:
                fv = fiducial_volume
            elif definitions["beam_escapes"]["op"] == "==" and definitions["beam_escapes"]["value"] == 1:
                fv = [max(fiducial_volume), 999]
            else:
                fv = [min(fiducial_volume), 999]
        else:
            fv = (0, 999)
        model_process_info[process.short_name[k]] = {"mode" : process.mode[k], "z_region" : fv}
    return model_process_info


def main(args : Application.argparse.Namespace):

    model_bins = slices_to_model_bins(args.energy_slices)

    process_info = get_process_info(args.process_definitions(), args.fiducial_volume)


    with open(f"{args.out}/systematics.yaml", "w") as f:
        f.write("Systematics:\n\n")

        for init_bin in model_bins.items():
            for end_bin in model_bins.items():
                for proc_info in process_info.items():
                    f.write(make_systematic(proc_info, init_bin, end_bin))
    return


if __name__ == "__main__":

    parser = Application.argparse.ArgumentParser("Creates systematics yaml file for use with MaCh3.")
    Application.ApplicationArguments.Config(parser)
    Application.ApplicationArguments.Output(parser)

    args = Application.ApplicationArguments.ResolveArgs(parser.parse_args())
    print(vars(args))
    main(args)