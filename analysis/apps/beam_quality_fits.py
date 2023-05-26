#!/usr/bin/env python3
import argparse
import json
import os

import awkward as ak
import numpy as np

from rich import print
from python.analysis import Master, BeamParticleSelection, vector
from scipy.optimize import curve_fit


def gaussian(x : np.array, a : float, x0 : float, sigma : float) -> np.array:
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fit_gaussian(data : ak.Array, bins : int, range : list = None):
    if range is None:
        range = [min(data), max(data)]
    y, bins_edges = np.histogram(np.array(data), bins = bins, range = range)
    bin_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
    return curve_fit(gaussian, bin_centers, y, p0 = (0, ak.mean(data), ak.std(data)))


def fit_vector(v : ak.Record, bins : int):
    mu = {}
    sigma = {}
    for i in ["x", "y", "z"]:
        popt, _ = fit_gaussian(v[i], bins = bins)
        mu[i] = popt[1]
        sigma[i] =popt[2]

    print(mu)
    print(sigma)
    return mu, sigma


def main(args):
    events = Master.Data(args.file, nTuple_type = args.ntuple_type)

    #* apply the following cuts before fitting (following the order in BeamParticleSelection)
    mask = BeamParticleSelection.CaloSizeCut(events)
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.PiBeamSelection(events)
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.PandoraTagCut(events)
    events.Filter([mask], [mask])

    mask = BeamParticleSelection.MichelScoreCut(events)
    events.Filter([mask], [mask])

    #* fit gaussians to the start positions
    mu, sigma = fit_vector(events.recoParticles.beam_startPos, 100)

    #* fit gaussians to beam directions
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))
    mu_dir, sigma_dir = fit_vector(beam_dir, 50)

    #* convert to dictionary undestood by the BeamQualityCut function
    fit_values = {
        "mu_x"        : mu["x"],
        "mu_y"        : mu["y"],
        "mu_z"        : mu["z"],
        "sigma_x"     : sigma["x"],
        "sigma_y"     : sigma["y"],
        "sigma_z"     : sigma["z"],
        "mu_dir_x"    : mu_dir["x"],
        "mu_dir_y"    : mu_dir["y"],
        "mu_dir_z"    : mu_dir["z"],
        "sigma_dir_x" : sigma_dir["x"],
        "sigma_dir_y" : sigma_dir["y"],
        "sigma_dir_z" : sigma_dir["z"]
    }
    print(fit_values)

    #* write to json file
    os.makedirs(args.out, exist_ok = True)
    name = args.out + args.file.split("/")[-1].split(".")[0] + "_fit_values.json"
    with open(name, "w") as f:
        json.dump(fit_values, f)

    print(f"fit values written to {name}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Applies beam particle selection, PFO selection, produces tables and basic plots.", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", help = "NTuple file to study.")
    parser.add_argument("-T", "--ntuple-type", dest = "ntuple_type", type = Master.Ntuple_Type, help = f"type of ntuple I am looking at {Master.Ntuple_Type._member_map_}.", required = True)

    parser.add_argument("-o", "--out", dest = "out", type = str, default = "./", help = "directory to save plots")

    args = parser.parse_args()

    if args.out[-1] != "/": args.out += "/"

    print(vars(args))
    main(args)