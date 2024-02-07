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
import pandas as pd
from rich import print

from python.analysis import Master, Processing, cross_section, EventSelection, Tags, SelectionTools

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # hide pesky pandas warnings (performance is actually ok)


#! replace with selection tools equivalent
def CreatePFOMasks(masks : dict[ak.Array]) -> ak.Array:
    """ combine a dicitonary of masks into a single mask.

    Args:
        masks (dict[ak.Array]): masks.

    Returns:
        ak.Array: combined mask.
    """
    mask = None
    for m in masks:
        if mask is None:
            mask = masks[m]
        else:
            mask = mask & masks[m]
    return mask


def run(i, file, n_events, start, selected_events, args):
    output = {}

    events = Master.Data(file, n_events, start, args["ntuple_type"])

    if "selection_masks" in args:
        mask = SelectionTools.CombineMasks(args["selection_masks"]["mc"]["beam"])
        events.Filter([mask], [mask])

        events.Filter([args["selection_masks"]["mc"]["null_pfo"]["ValidPFOSelection"]])
        photon_mask = CreatePFOMasks(args["selection_masks"]["mc"]["photon"])
        events.Filter([photon_mask])
    else:
        for s, a in zip(args["beam_selection"]["selections"].values(), args["beam_selection"]["mc_arguments"].values()):
            mask = s(events, **a)
            events.Filter([mask], [mask])
        photon_masks = {}
        if args["valid_pfo_selection"] is True:
            for k, s, a in zip(args["photon_selection"]["selections"].keys(), args["photon_selection"]["selections"].values(), args["photon_selection"]["mc_arguments"].values()):
                photon_masks[k] = s(events, **a)
        photon_mask = CreatePFOMasks(photon_masks)
        events.Filter([photon_mask])
    print("making pairs")
    pairs = EventSelection.NPhotonCandidateSelection(events, photon_mask[photon_mask], 2)

    shower_pairs = Master.ShowerPairs(events, shower_pair_mask = photon_mask[photon_mask] & pairs)

    params = ["angle", "sub_energy", "lead_energy", "mass"]

    for p in params:
        for t in ["reco", "true"]:
            output[f"shower_pairs_{t}_{p}"] = ak.flatten(getattr(shower_pairs, f"{t}_{p}")[pairs])

    # reco_params = ["shower_direction", "shower_start_pos", "shower_length", "n_hits", "n_hits_collection", "shower_energy"]

    # true_params = ["direction", "shower_start_pos", "energy"]

    params = {
        "reco" : ["shower_direction", "shower_start_pos", "shower_length", "n_hits", "n_hits_collection", "shower_energy"],
        "true" : ["direction", "shower_start_pos", "energy"]
    }

    for (k, param), particleData in zip(params.items(), {"reco" : events.recoParticles, "true" : events.trueParticlesBT}.values()):
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

    output["true_mother"] = ak.flatten(events.trueParticlesBT.motherPdg)
    pfo_tags = Tags.GenerateTrueParticleTagsPi0Shower(events)
    output["pi0_photon"] = ak.flatten(pfo_tags["$\\gamma$:beam $\\pi^0$"].mask | pfo_tags["$\\gamma$:other $\\pi^0$"].mask)

    pi0_tags = Tags.GeneratePi0Tags(events, photon_mask[photon_mask] & pairs)
    for t in pi0_tags:
        pi0_tags[t].mask = pi0_tags[t].mask[pairs]
    output["pi0_tags"] = pi0_tags
    
    fs_tags = EventSelection.GenerateTrueFinalStateTags(events)
    for t in fs_tags:
        fs_tags[t].mask = fs_tags[t].mask[pairs]
    output["final_state_tags"] = fs_tags

    return output

def main(args):
    outputs = Processing.mutliprocess(run, [args.mc_file], args.batches, args.events, vars(args), args.threads)

    output = {}
    for o in outputs:
        for k, v in o.items():
            if k not in output:
                output[k] = v
            else:
                if "tag" in k:
                    for tag in v:
                        output[k][tag].mask = ak.concatenate([output[k][tag].mask, v[tag].mask])
                else:
                    output[k] = ak.concatenate([output[k], v])
    print(output)
    output_photons = pd.DataFrame({i : output[i] for i in output if "shower_pairs" not in i and "tags" not in i})
    output_pairs = pd.DataFrame({i : output[i] for i in output if "shower_pairs" in i and "tags" not in i})
    output_tags = pd.DataFrame({i : output[i] for i in output if "tags" in i})

    print(output_photons)
    print(output_pairs)
    print(output_tags)

    os.makedirs(args.out, exist_ok = True)
    output_photons.to_hdf(args.out + "photon_energies.hdf5", "all_photons")
    output_pairs.to_hdf(args.out + "photon_energies.hdf5", "photon_pairs")
    output_tags.to_hdf(args.out + "photon_energies.hdf5", "tags")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection and saves properties of photon shower candidate PFOs to hdf5 file (MC only)", formatter_class = argparse.RawDescriptionHelpFormatter)

    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)
    cross_section.ApplicationArguments.Config(parser)

    args = parser.parse_args()
    args = cross_section.ApplicationArguments.ResolveArgs(args)

    print(vars(args))
    main(args)