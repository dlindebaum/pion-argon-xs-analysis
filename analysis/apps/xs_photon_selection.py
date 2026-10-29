#!/usr/bin/env python3
"""
Created on: 23/05/2023 12:58

Author: Shyam Bhuller

Description: Applies beam particle selection, photon shower candidate selection and writes out shower energies.
"""
import argparse
import os
import warnings

import awkward as ak
import numpy as np
import pandas as pd
from rich import print

from python.analysis import application
from python.analysis import common
from python.analysis import ntuple
from python.analysis import selection
from python.analysis import tags
from python.analysis import io
from python.analysis import photon_energy_correction as pec
from python.analysis import plotting

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # hide pesky pandas warnings (performance is actually ok)

def run(i : int, file_desc : ntuple.FileDescriptor, n_events : int, start : int, selected_events, args : dict) -> dict:
    output = {}

    events = ntuple.Data(file_desc, n_events, start)

    for s, a in zip(args["beam_selection"]["selections"].values(), args["beam_selection"]["mc_arguments"].values()):
        print(s)
        mask = s(events, **a)
        events.filter([mask], [mask])
    photon_masks = {}
    if args["valid_pfo_selection"] is True:
        for k, s, a in zip(args["photon_selection"]["selections"].keys(), args["photon_selection"]["selections"].values(), args["photon_selection"]["mc_arguments"].values()):
            photon_masks[k] = s(events, **a)
    photon_mask = selection.combine_masks(photon_masks)
    events.filter([photon_mask])
    print("making pairs")
    pairs = selection.event.NPhotonCandidateSelection(events, photon_mask[photon_mask], 2)

    shower_pairs = ntuple.ShowerPairs(events, shower_pair_mask = photon_mask[photon_mask] & pairs)

    params = ["angle", "sub_energy", "lead_energy", "mass"]

    for p in params:
        for t in ["reco", "true"]:
            output[f"shower_pairs_{t}_{p}"] = ak.flatten(getattr(shower_pairs, f"{t}_{p}")[pairs])

    params = {
        "reco" : ["shower_direction", "shower_start_pos", "shower_length", "n_hits", "n_hits_collection", "shower_energy"],
        "true" : ["direction", "shower_start_pos", "energy"]
    }

    for (k, param), particleData in zip(params.items(), {"reco" : events.recoParticles, "true" : events.true_particles_bt}.values()):
        for p in param:
            if hasattr(particleData, p):
                v = getattr(particleData, p)
                # if v is None: continue
                if v is not None:
                    if hasattr(v, "x"):
                        for i in ["x", "y", "z"]:
                            output[f"{k}_{p}_{i}"] = ak.flatten(v[i])
                    else:
                        output[f"{k}_{p}"] = ak.flatten(v)

    output["true_mother"] = ak.flatten(events.true_particles_bt.mother_pdg)
    pfo_tags = tags.generate_true_particle_tags_pi0_shower(events)
    output["pi0_photon"] = ak.flatten(pfo_tags["$\\gamma$:beam $\\pi^0$"].mask | pfo_tags["$\\gamma$:other $\\pi^0$"].mask)

    pi0_tags = tags.generate_pi0_tags(events, photon_mask[photon_mask] & pairs)
    for t in pi0_tags:
        pi0_tags[t].mask = pi0_tags[t].mask[pairs]
    output["pi0_tags"] = pi0_tags
    
    fs_tags = selection.event.generate_true_final_state_tags(events)
    for t in fs_tags:
        fs_tags[t].mask = fs_tags[t].mask[pairs]
    output["final_state_tags"] = fs_tags

    return output

@common.timer
def main(args):
    plotting.styles.PlotStyler.set_plot_style(False)
    out = args.out + "shower_energy_correction/"

    if (not os.path.isfile(out + "photon_energies.hdf5")) or args.regen:
        output = ntuple.run_process(args.ntuple_files["mc"], False, args, run)

        output_photons = pd.DataFrame({i : output[i] for i in output if "shower_pairs" not in i and "tags" not in i})
        output_pairs = pd.DataFrame({i : output[i] for i in output if "shower_pairs" in i and "tags" not in i})
        output_tags = pd.DataFrame({i : output[i] for i in output if "tags" in i})

        print(output_photons)
        print(output_pairs)
        print(output_tags)

        os.makedirs(out, exist_ok = True)
        output_photons.to_hdf(out + "photon_energies.hdf5", "all_photons")
        output_pairs.to_hdf(out + "photon_energies.hdf5", "photon_pairs")
        output_tags.to_hdf(out + "photon_energies.hdf5", "tags")

    df = io.read_hdf5(out + "photon_energies.hdf5")["all_photons"]
    df["residual"] = df.reco_shower_energy - df.true_energy
    df["fractional_error"] = (df.reco_shower_energy / df.true_energy) - 1

    with plotting.PlotBook(out + "plots.pdf") as book:
        #* initial plots
        pec.photon_selection(df, book)

        #* linear correction
        if args.shower_correction["energy_range"] is None:
            print("Note: no energy range was defined for the shower correction, will use an auto-defined range instead")
            energy_range = [
                np.percentile(np.concatenate([df.true_energy, df.reco_shower_energy]), 0),
                np.percentile(np.concatenate([df.true_energy, df.reco_shower_energy]), 100)
                ]
        else:
            energy_range = sorted(args.shower_correction["energy_range"])

        print(f"{energy_range=}")

        bins = np.linspace(*energy_range, 11)
        print(f"{bins=}")
        linear_correction = pec.linear_fit(df, bins, energy_range, book)
        pec.linear_fit_performance(df, linear_correction, book)


        #* response correction
        bins = np.array(pec.create_bins_df(df.reco_shower_energy, int(len(df.reco_shower_energy)//11), energy_range), dtype = int)
        #* response correction
        bins = np.array(pec.create_bins_df(df.reco_shower_energy, int(len(df.reco_shower_energy)//11), energy_range), dtype = int)

        central_values = pec.calculate_central_values(df, bins, book)
        response_params = pec.response_fits(central_values, bins, book)
        central_values = pec.calculate_central_values(df, bins, book)
        response_params = pec.response_fits(central_values, bins, book)

        tab = pec.method_comparison(df, linear_correction, response_params, bins, energy_range, book)
        tab = pec.method_comparison(df, linear_correction, response_params, bins, energy_range, book)
    
    tab.T.style.format(precision = 3).to_latex(out + "table.tex")

    params = {p :
        {
            "value" : {f"p{i}" : response_params[p]["value"][i] for i in range(len(response_params[p]["value"]))},
            "error" : {f"p{i}" : response_params[p]["error"][i] for i in range(len(response_params[p]["error"]))}
        }
    for p in response_params}

    for name, p in params.items():
        sf = [len(f'{p["error"][f"p{i}"]:.1g}') - 1 for i in range(len(p["value"]))]
        table = pd.DataFrame({f"$p_{{{i}}}$" : f'{p["value"][f"p{i}"]:.{sf[i]}f} $\\pm$ {p["error"][f"p{i}"]:.1g}' for i in range(len(p["value"]))}, index = [0])
        table.style.hide(axis = "index").to_latex(out + name + ".tex")
        io.write_json(p, out + name + ".json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection and saves properties of photon shower candidate PFOs to hdf5 file (MC only)", formatter_class = argparse.RawDescriptionHelpFormatter)

    application.ApplicationArguments.processing(parser)
    application.ApplicationArguments.output(parser)
    application.ApplicationArguments.config(parser)
    application.ApplicationArguments.regen(parser)

    args = parser.parse_args()
    args = application.ApplicationArguments.resolve_args(args)

    print(vars(args))
    main(args)