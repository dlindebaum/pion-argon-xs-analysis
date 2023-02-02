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

@Master.timer
def GetEnergy(events : Master.Data, showers : ak.Array):
    data = {
        "true_energy_0" : [],
        "true_energy_1" : [],
        
        "true_energy_hits_0" : [],
        "true_energy_hits_1" : [],
        
        "shared_energy_0" : [],
        "shared_energy_1" : [],
    }
    for i in range(ak.num(showers, 0)):
        e1, e2 = GetSharedEnergy(events.trueParticlesBT, i, showers[i][0], showers[i][1])
        # if e1 <= 0 or e2 <= 0: continue
        true_energy = [events.trueParticlesBT.energy[i][showers[i][j]] for j in range(2)]
        true_energy_hits = [events.trueParticlesBT.energyByHits[i][showers[i][j]] for j in range(2)]

        data["true_energy_0"].append(true_energy[0])
        data["true_energy_1"].append(true_energy[1])
        data["true_energy_hits_0"].append(true_energy_hits[0])
        data["true_energy_hits_1"].append(true_energy_hits[1])
        data["shared_energy_0"].append(e1)
        data["shared_energy_1"].append(e2)
    return data 


def main(args):
    events = Master.Data(args.file, nEvents = args.nEvents[0], start = args.nEvents[1])
    merge_study.EventSelection(events)
    start_showers, _ = merge_study.SplitSample(events)

    mask = np.logical_or(*start_showers)
    start_shower_indices = ak.local_index(events.recoParticles.number)[mask]
    print(start_shower_indices)

    df = pd.DataFrame(GetEnergy(events, start_shower_indices))
    df.to_csv(args.csv, mode = "a", header = not os.path.exists(args.csv))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "run the hit correction study", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-e", "--events", dest="nEvents", type=int, nargs=2, default=[-1, 0], help="number of events to analyse and number to skip (-1 is all)")
    parser.add_argument("-o", "--out-csv", dest="csv", type=str, default=None, help="output csv filename (will default to whatever type of data is produced)")
    args = parser.parse_args()
    print(vars(args))

    main(args)