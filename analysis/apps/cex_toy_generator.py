#!/usr/bin/env python3
"""
Created on: 08/08/2023 16:18

Author: Shyam Bhuller

Description: Generates toy events for inelastic pion interactions.
"""
import argparse

from collections import Counter

import numpy as np
import pandas as pd
import rich.progress as progress
import scipy.interpolate as interpolate

from rich import print

from python.analysis.Master import timer
from python.analysis.cross_section import ApplicationArguments, BetheBloch, GeantCrossSections, Particle

@timer
def GeneratePDFs(l : float) -> np.array:
    """ Creates PDF for a given cross section channel and slice thickness (i.e. step size)

    Args:
        channel (str): cross section channel
        l (float): slice thickness

    Returns:
        np.array: _description_
    """
    xs_sim = GeantCrossSections()

    pdfs = {}
    for v in vars(xs_sim):
        if v in ["KE", "file"]: continue
        pdfs[v] = interpolate.interp1d(xs_sim.KE, P_int(getattr(xs_sim, v), l), fill_value = "extrapolate")
    return pdfs


def P_int(sigma, l) -> np.array:
    return 1 - np.exp(-1E-27 * sigma * 6.02214076e23 * BetheBloch.rho * l / BetheBloch.A)


def GenerateStackedPDFs(l : float) -> dict:
    xs_sim = GeantCrossSections()

    pdfs = {}
    pdfs["total_inelastic"] = interpolate.interp1d(xs_sim.KE, P_int(xs_sim.total_inelastic, l), fill_value = "extrapolate")

    area = {}
    for v in vars(xs_sim):
        if v in ["KE", "file", "total_inelastic"]: continue
        area[v] = np.trapz(P_int(getattr(xs_sim, v), l))

    y = np.zeros(len(xs_sim.KE))
    for v, _ in sorted(area.items(), key=lambda x: x[1]):
        y = y + P_int(getattr(xs_sim, v), l)
        pdfs[v] = interpolate.interp1d(xs_sim.KE, y, fill_value = "extrapolate")
    return pdfs


def GenerateIntialKEs(n, particle : Particle, cv : float, width : float, profile : str) -> np.array:
    central_KE = (cv**2 + particle.mass**2)**0.5 - particle.mass
    if profile == "gaussian":
        return np.random.normal(central_KE, width, n)
    elif profile == "uniform":
        return np.random.uniform(central_KE - width/2, central_KE + width/2, n)
    else:
        raise ValueError(f"{profile} not a valid beam profile")


@timer
def Simulate(KE_init : np.array, stepsize, particle : Particle, pdfs : dict) -> tuple[np.array, np.array, np.array]:
    survived = ~np.zeros(len(KE_init), dtype = bool)
    i = np.zeros(len(KE_init), dtype = int)
    KE_int = KE_init

    # process = np.zeros(len(KE_init), dtype = object)

    inclusive_process = np.repeat(["decay"], len(KE_init)).astype(object)
    exclusive_process = np.repeat([""], len(KE_init)).astype(object)


    with progress.Progress(progress.SpinnerColumn(), *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()) as p:
        task = p.add_task("Propagating...", total = len(survived))
        previous = sum(survived)
        current = sum(survived)
        while any(survived):

            U = np.random.uniform(0, 1, len(KE_int))
            inelastic = U < pdfs["total_inelastic"](KE_int)
            inclusive_process[inelastic] = "total_inelastic"
            for pdf in reversed(pdfs):
                if pdf == "total_inelastic": continue
                exclusive = inelastic & (U < pdfs[pdf](KE_int))
                exclusive_process[exclusive] = pdf

            survived = survived & (survived != (inelastic)) # interacts inelasticly do does not survive
            survived = survived & (survived != (KE_int < 1)) # particle is pretty much stationary so decays

            i = i + survived # distance travelled in step numbers
            KE_int = KE_int - survived * stepsize * BetheBloch.meandEdX(KE_int, particle)

            current = sum(survived)
            p.update(task, advance = previous - current)

            previous = current
        p.finished
    z_int = i * stepsize
    return KE_int, z_int, inclusive_process, exclusive_process


def CountProcesses(proc : np.array):
    for p in Counter(proc):
        print(p, sum(proc == p))


def main(args : argparse.Namespace):
    particle = Particle.from_pdgid(211)

    # pdfs = GeneratePDFs(args.step)
    pdfs = GenerateStackedPDFs(args.step)
    print(f"{pdfs=}")

    KE_init = GenerateIntialKEs(args.events, particle, args.momentum, args.width, args.beam_profile)

    KE_int, z_int, inclusive_process, exclusive_process  = Simulate(KE_init, args.step, particle, pdfs)

    df = pd.DataFrame({
        "KE_init" : KE_init,
        "KE_int" : KE_int,
        "z_int" : z_int,
        "inclusive_process" : inclusive_process,
        "exclusive_process" : exclusive_process
    })

    CountProcesses(inclusive_process)
    CountProcesses(exclusive_process)

    print(df)
    df.to_hdf(args.out + ".hdf5", key = "df")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generates toy events for inelastic pion interactions.")
    parser.add_argument("-e", "--events", type = float, help = "number of events to generate.", required = True)
    parser.add_argument("-s", "--step", type = float, help = "step size of propagation (cm).", required = True)
    parser.add_argument("-p", "--momentum", type = float, help = "initial momentum to generate particles at (MeV).", required = True)
    parser.add_argument("-b", "--beam-profile", type = str, choices = ["gaussian", "uniform"], help = "what kind of pdf is the beam spread modelled by", required = True)
    parser.add_argument("-w", "--width", type = float, help = "width of beam spread.", required = True)

    ApplicationArguments.Output(parser)

    args = parser.parse_args()

    args.events = int(args.events)
    if args.out is None:
        args.out = "toy_mc"

    print(vars(args))
    main(args)