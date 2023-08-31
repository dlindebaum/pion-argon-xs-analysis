#!/usr/bin/env python3
import os

import awkward as ak
import numpy as np
import pandas as pd

import argparse

from python.analysis import Master, shower_merging

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

@Master.timer
def GetAllSharedEnergy(events : Master.Data, event : int, a : int, true_particles_index : ak.Array):
    mask = None
    partner_energy = None

    for i in range(ak.count(true_particles_index)):
        index = true_particles_index[i]

        if events.trueParticlesBT.number[event][a] == events.trueParticlesBT.number[event][index]: continue

        m = GetSharedHitsMask(events.trueParticlesBT, event, index, a)

        if mask is None:
            mask = m
            partner_energy = ak.unflatten(ak.where(m, events.trueParticlesBT.energy[event][index], 0), 1, -1)
        else:
            mask = mask | m
            partner_energy = ak.concatenate([partner_energy, ak.unflatten(ak.where(m, events.trueParticlesBT.energy[event][index], 0), 1, -1)], -1)
    n_shared = ak.count(mask[mask])
    print(f"number of shared hits: {n_shared}")

    true_energy = events.trueParticlesBT.energy[event][a]
    weights = true_energy / (true_energy + ak.sum(partner_energy, -1))

    hit_energy = events.trueParticlesBT.hit_energy[event][a]
    mask = mask & (hit_energy > 0)
    
    excess_energy = ak.sum(hit_energy[mask] * (1 - weights[mask]))
    return excess_energy, n_shared

@Master.timer
def GetEnergy(events : Master.Data, showers : ak.Array, all_energy : bool = False):
    data = {
        "event": [],
        "run" : [],
        "subrun" : [],

        "true_energy_0" : [],
        "true_energy_1" : [],
        
        "true_energy_hits_0" : [],
        "true_energy_hits_1" : [],

        "shared_energy_0" : [],
        "shared_energy_1" : [],
        
        "n_shared_hits_0" : [],
        "n_shared_hits_1" : [],
    }
    true_particles = events.trueParticlesBT.GetUniqueParticleNumbers(events.trueParticlesBT.number)
    for i in range(ak.num(showers, 0)):
        if all_energy:
            true_particles_index = [TrueParticleIndices(events, i, true_particles[i][j])[0] for j in range(ak.count(true_particles[i]))]

            e_0, n_0 = GetAllSharedEnergy(events, i, showers[i][0], true_particles_index)
            e_1, n_1 = GetAllSharedEnergy(events, i, showers[i][1], true_particles_index)
            data["n_shared_hits_0"].append(n_0)
            data["n_shared_hits_1"].append(n_1)

        else:
            mask = GetSharedHitsMask(events.trueParticlesBT, i, showers[i][0], showers[i][1])
            data["n_shared_hits_0"].append(ak.count(mask[mask]))
            data["n_shared_hits_1"].append(ak.count(mask[mask]))
            e_0, e_1 = GetSharedEnergy(events.trueParticlesBT, i, showers[i][0], showers[i][1])
        true_energy = [events.trueParticlesBT.energy[i][showers[i][j]] for j in range(2)]
        true_energy_hits = [events.trueParticlesBT.energyByHits[i][showers[i][j]] for j in range(2)]

        data["event"].append(events.eventNum[i])
        data["run"].append(events.run[i])
        data["subrun"].append(events.subRun[i])
        data["true_energy_0"].append(true_energy[0])
        data["true_energy_1"].append(true_energy[1])
        data["true_energy_hits_0"].append(true_energy_hits[0])
        data["true_energy_hits_1"].append(true_energy_hits[1])
        data["shared_energy_0"].append(e_0)
        data["shared_energy_1"].append(e_1)
    return data 


def main(args):
    events = Master.Data(args.file, nEvents = args.nEvents[0], start = args.nEvents[1])
    shower_merging.EventSelection(events)
    start_showers, _ = shower_merging.SplitSample(events)

    mask = np.logical_or(*start_showers)
    start_shower_indices = ak.local_index(events.recoParticles.number)[mask]
    print(start_shower_indices)

    df = pd.DataFrame(GetEnergy(events, start_shower_indices, args.all_energy))
    if args.csv != None:
        df.to_csv(args.csv, mode = "a", header = not os.path.exists(args.csv))
    else:
        df.head()
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