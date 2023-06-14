#!/usr/bin/env python3
"""
Created on: 23/05/2023 12:58

Author: Shyam Bhuller

Description: Applies beam particle selection, photon shower candidate selection and writes out shower energies.
"""
import argparse
import json
import os

import awkward as ak
import pandas as pd
from rich import print

from python.analysis import Master, Processing, BeamParticleSelection, PFOSelection, cross_section, EventSelection, Tags

def run(i, file, n_events, start, selected_events, args):
    output = {}

    events = Master.Data(file, n_events, start, args["ntuple_type"])

    with open(args["mc_beam_quality_fit"], "r") as f:
        fit_values = json.load(f)


    mask = BeamParticleSelection.CreateDefaultSelection(events, False, fit_values, verbose = True, return_table = False)
    events.Filter([mask], [mask])

    mask = PFOSelection.InitialPi0PhotonSelection(events, verbose = True, return_table = False)
    events.Filter([mask])

    pairs = EventSelection.NPhotonCandidateSelection(events, mask[mask], 2)

    shower_pairs = Master.ShowerPairs(events, shower_pair_mask = mask[mask] & pairs)

    output["shower_pairs_reco_angle"] = ak.flatten(shower_pairs.reco_angle[pairs])
    output["shower_pairs_reco_lead_energy"] = ak.flatten(shower_pairs.reco_lead_energy[pairs])
    output["shower_pairs_reco_sub_energy"] = ak.flatten(shower_pairs.reco_sub_energy[pairs])
    
    output["shower_pairs_true_angle"] = ak.flatten(shower_pairs.true_angle[pairs])
    output["shower_pairs_true_lead_energy"] = ak.flatten(shower_pairs.true_lead_energy[pairs])
    output["shower_pairs_true_sub_energy"] = ak.flatten(shower_pairs.true_sub_energy[pairs])

    output["reco_energy"] = ak.flatten(events.recoParticles.energy)
    output["true_energy"] = ak.flatten(events.trueParticlesBT.energy)
    output["true_mother"] = ak.flatten(events.trueParticlesBT.motherPdg)

    pi0_tags = Tags.GeneratePi0Tags(events, mask[mask] & pairs)
    for t in pi0_tags:
        pi0_tags[t].mask = pi0_tags[t].mask[pairs]
    output["pi0_tags"] = pi0_tags
    
    fs_tags = Tags.GenerateTrueFinalStateTags(events)
    for t in fs_tags:
        fs_tags[t].mask = fs_tags[t].mask[pairs]
    output["final_state_tags"] = fs_tags

    return output

def main(args):
    outputs = Processing.mutliprocess(run, [args.file], args.batches, args.events, vars(args), args.threads)

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

    cross_section.ApplicationArguments.SingleNtuple(parser, define_sample = False)
    cross_section.ApplicationArguments.BeamQualityCuts(parser, data = False)
    cross_section.ApplicationArguments.BeamSelection(parser)
    cross_section.ApplicationArguments.Processing(parser)
    cross_section.ApplicationArguments.Output(parser)

    args = parser.parse_args()

    cross_section.ApplicationArguments.ResolveArgs(args)

    print(vars(args))
    main(args)