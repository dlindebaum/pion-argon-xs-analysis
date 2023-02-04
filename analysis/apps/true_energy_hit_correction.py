#!/usr/bin/env python3
import os

import awkward as ak
import numpy as np
import pandas as pd

import argparse

import apps.prod4a_merge_study as merge_study
from python.analysis import Master

def GetSharedHitsMask(particleData : Master.ParticleData, e, a, b):
    #! returns the mask for particle b only
    total_mask = None
    for c1, t1 in zip(particleData.channel[e][a], particleData.peakTime[e][a]):
        mask = (c1 == particleData.channel[e][b]) & (t1 == particleData.peakTime[e][b])
        if total_mask is None:
            total_mask = mask
        else:
            total_mask = total_mask | mask
    return total_mask


def GetSharedEnergy(particleData : Master.ParticleData, e, a, b):
    mask_a = GetSharedHitsMask(particleData, e, b, a)
    mask_b = GetSharedHitsMask(particleData, e, a, b)

    energy_a = particleData.hit_energy[e][a][mask_a]
    energy_b = particleData.hit_energy[e][b][mask_b]

    sum_a = ak.sum(energy_a[energy_a > 0])
    sum_b = ak.sum(energy_b[energy_b > 0])
    return sum_a, sum_b


def TrueParticleIndices(events, event, particle_number):
    index = ak.local_index(events.trueParticlesBT.number[event])
    return index[events.trueParticlesBT.number[event] == particle_number]


def GetAllSharedEnergy(events : Master.Data, event : int, a : int):
    mask = None
    partner_energy = None
    true_particles = events.trueParticlesBT.GetUniqueParticleNumbers(events.trueParticlesBT.number)

    for i in range(ak.count(true_particles[event])):
        index = TrueParticleIndices(events, event, true_particles[event][i])[0]

        if events.trueParticlesBT.number[event][a] == events.trueParticlesBT.number[event][index]: continue

        m = GetSharedHitsMask(events.trueParticlesBT, event, index, a)

        if mask is None:
            mask = m
            partner_energy = ak.unflatten(ak.where(m, events.trueParticlesBT.energy[event][index], 0), 1, -1)
        else:
            mask = mask | m
            partner_energy = ak.concatenate([partner_energy, ak.unflatten(ak.where(m, events.trueParticlesBT.energy[event][index], 0), 1, -1)], -1)
    print(f"number of shared hits: {ak.count(mask[mask])}")

    true_energy = events.trueParticlesBT.energy[event][a]
    weights = true_energy / (true_energy + ak.sum(partner_energy, -1))

    hit_energy = events.trueParticlesBT.hit_energy[event][a]
    mask = mask & (hit_energy > 0)    
    
    excess_energy = ak.sum(hit_energy[mask] * (1 - weights[mask]))
    return excess_energy

@Master.timer
def GetEnergy(events : Master.Data, showers : ak.Array, all_energy : bool = False):
    data = {
        "true_energy_0" : [],
        "true_energy_1" : [],
        
        "true_energy_hits_0" : [],
        "true_energy_hits_1" : [],
        
        "shared_energy_0" : [],
        "shared_energy_1" : [],
    }
    for i in range(ak.num(showers, 0)):
        if all_energy:
            e_0 = GetAllSharedEnergy(events, i, showers[i][0])
            e_1 = GetAllSharedEnergy(events, i, showers[i][0])
        else:
            e_0, e_1 = GetSharedEnergy(events.trueParticlesBT, i, showers[i][0], showers[i][1])
        # if e1 <= 0 or e2 <= 0: continue
        true_energy = [events.trueParticlesBT.energy[i][showers[i][j]] for j in range(2)]
        true_energy_hits = [events.trueParticlesBT.energyByHits[i][showers[i][j]] for j in range(2)]

        data["true_energy_0"].append(true_energy[0])
        data["true_energy_1"].append(true_energy[1])
        data["true_energy_hits_0"].append(true_energy_hits[0])
        data["true_energy_hits_1"].append(true_energy_hits[1])
        data["shared_energy_0"].append(e_0)
        data["shared_energy_1"].append(e_1)
    return data 


def main(args):
    events = Master.Data(args.file, nEvents = args.nEvents[0], start = args.nEvents[1])
    merge_study.EventSelection(events)
    start_showers, _ = merge_study.SplitSample(events)

    mask = np.logical_or(*start_showers)
    start_shower_indices = ak.local_index(events.recoParticles.number)[mask]
    print(start_shower_indices)

    df = pd.DataFrame(GetEnergy(events, start_shower_indices, args.all_energy))
    df.to_csv(args.csv, mode = "a", header = not os.path.exists(args.csv))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "run the hit correction study", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", type = str, help = "ROOT file to open.")
    parser.add_argument("-e", "--events", dest = "nEvents", type = int, nargs = 2, default = [-1, 0], help = "number of events to analyse and number to skip (-1 is all)")
    parser.add_argument("-o", "--out-csv", dest = "csv", type = str, default = None, help = "output csv filename")
    parser.add_argument("-a", "--all-energy", dest = "all_energy", action = "store_true", help = "get all shared energy or just shared energy between shower pairs")
    args = parser.parse_args()
    print(vars(args))

    main(args)