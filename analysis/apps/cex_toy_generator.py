#!/usr/bin/env python3
"""
Created on: 08/08/2023 16:18

Author: Shyam Bhuller

Description: Generates toy events for inelastic pion interactions.
"""
import argparse
import os

from collections import Counter

import numpy as np
import pandas as pd

from pathos.pools import ProcessPool
from rich import print
from scipy.interpolate import interp1d

from python.analysis.Master import timer
from python.analysis.cross_section import ApplicationArguments, BetheBloch, GeantCrossSections, Particle


def ComputeEnergyLoss(inital_KE : float, stepsize : float) -> interp1d:
    """ Calculate the mean dEdX profile for a given initial kinetic energy and position step size.
        Then produce a function to map kinetic energy to dEdX given the outputs, and allow for interpolation.

    Args:
        inital_KE (float): Initial kinetic energy
        stepsize (float): position step size (cm)

    Returns:
        interp1d: interpolated map of KE and dEdX
    """
    e = inital_KE
    KE = []
    dEdX = []
    counter = 0
    while e >= 0:
        KE.append(e)
        dEdX.append(BetheBloch.meandEdX(e, Particle.from_pdgid(211)))
        e = e - stepsize * dEdX[-1]
        counter += 1
    KE.append(0)
    dEdX.append(np.inf)
    return interp1d(KE, dEdX, fill_value = 0, bounds_error = False) # if outside the interpolation range, return 0


def P_int(sigma : np.array, l : float) -> np.array:
    """ Formula to calcualte the interaction probability for the thin slab approximation. 

    Args:
        sigma (np.array): cross section as a function of energy
        l (float): slab thickness

    Returns:
        np.array: interaction proabability
    """
    return 1 - np.exp(-1E-27 * sigma * 6.02214076e23 * BetheBloch.rho * l / BetheBloch.A)


def GenerateStackedPDFs(l : float, path = os.environ.get('PYTHONPATH', '').split(os.pathsep)[0] + "/data/g4_xs.root") -> dict:
    """ Creates a PDF of the inclusive cross section and a stacked PDF of the exclusive process,
        such that rejection sampling can be used to allocate an exclusive process for a given inclusive process.

    Args:
        l (float): slice thickness

    Returns:
        dict: dictionaty of pdfs
    """
    xs_sim = GeantCrossSections(file = path)

    pdfs = {}
    pdfs["total_inelastic"] = interp1d(xs_sim.KE, P_int(xs_sim.total_inelastic, l), fill_value = "extrapolate") # total inelastic pdf

    # sort the exclusice channels based on area under curve
    area = {}
    for v in vars(xs_sim):
        if v in ["KE", "file", "total_inelastic"]: continue
        area[v] = np.trapz(P_int(getattr(xs_sim, v), l))

    # stack the pdfs in ascending order
    y = np.zeros(len(xs_sim.KE))
    for v, _ in sorted(area.items(), key=lambda x: x[1]):
        y = y + P_int(getattr(xs_sim, v), l)
        pdfs[v] = interp1d(xs_sim.KE, y, fill_value = "extrapolate")
    return pdfs


def GenerateIntialKEs(n : int, particle : Particle, cv : float, width : float, profile : str) -> np.array:
    """ Create a distribution of initial kinetic energy based on a profile i.e. pdf.
        Currently supported profiles are "uniform" and "gaussian".


    Args:
        n (int): number of particles/events.
        particle (Particle): particle type
        cv (float): central value of profile
        width (float): width of profile
        profile (str): profile type
    Returns:
        np.array: initial kinetic energies.
    """
    central_KE = (cv**2 + particle.mass**2)**0.5 - particle.mass
    if profile == "gaussian":
        return np.random.normal(central_KE, width, n)
    elif profile == "uniform":
        return np.random.uniform(central_KE - width/2, central_KE + width/2, n)
    else:
        raise ValueError(f"{profile} not a valid beam profile")

@timer
def Simulate(KE_init : np.array, stepsize : float, pdfs : dict) -> tuple:
    """ Generates interacting kinetic energies and position based on the initial kinetic energy with the bethe bloch formula,
        the total inelastic pdf is used to decide when a particle interacts (using rejection sampling) and then each particle which
        interacts is assosiated a particular exclusive interaction process, using rejection sampling. Particles which reach zero KE are considered
        to be particle decays. The precision of these calculations depends on the step size, which is essentially the slab thickness.

    Args:
        KE_init (np.array): initial Kinetic energy distribution
        stepsize (float): position step size or "slab thickness"
        particle (Particle): particle type
        pdfs (dict): cross section pdfs

    Returns:
        tuple: interacting kinetic energy, interacting position, whether it interacted or decay, exclusive process
    """
    interpolated_energy_loss = ComputeEnergyLoss(2*max(KE_init), stepsize/2) # precompute the energy loss and create a function to interpolate between them

    survived = ~np.zeros(len(KE_init), dtype = bool) # keep track of which particles survived
    i = np.zeros(len(KE_init), dtype = int) # slab number i.e. position each particle interacted at
    KE_int = KE_init # set interacting kinetic energy to the initial at the start

    # setup arrays which tag each particle based on interaction and interaction type
    inclusive_process = np.repeat(["decay"], len(KE_init)).astype(object) 
    exclusive_process = np.repeat([""], len(KE_init)).astype(object)


    while any(survived):
        U = np.random.uniform(0, 1, len(KE_int)) # sample from a uniform distribution
        inelastic = U < pdfs["total_inelastic"](KE_int) # did the partcile interact?
        inclusive_process[inelastic] = "total_inelastic" # add label to all particles which did interact
        for pdf in reversed(pdfs): # do this in reverse order, such that the least probable exlcusive process is checked last
            if pdf == "total_inelastic": continue # only exclusive processes
            exclusive = inelastic & (U < pdfs[pdf](KE_int))
            exclusive_process[exclusive] = pdf

        survived = survived & (survived != (inelastic)) # interacts inelasticly do does not survive
        survived = survived & (survived != (KE_int < 1)) # particle is pretty much stationary so decays

        i = i + survived # distance travelled in step numbers
        KE_int = KE_int - survived * stepsize * interpolated_energy_loss(KE_int) # update KE if it survived

    z_int = i * stepsize # convert interaction index to interacting position

    return pd.DataFrame({
        "KE_int" : KE_int,
        "z_int" : z_int,
        "inclusive_process" : inclusive_process,
        "exclusive_process" : exclusive_process
    })


def CountProcesses(proc : np.array):
    """ Count unique processes for particles

    Args:
        proc (np.array): array of processes
    """
    for p in Counter(proc):
        print(p, sum(proc == p))


def CreateMasks(toy : pd.DataFrame) -> dict:
    """ Creates masks for each inclusive process and exclusive process based on particle tags from the toy mc.

    Args:
        toy (pd.DataFrame): toy mc to look at

    Returns:
        dict: masks of processes
    """
    masks = {}
    for c in np.unique(toy.exclusive_process)[1:]:
        masks[c] = toy.exclusive_process == c
    for c in np.unique(toy.inclusive_process)[1:]:
        masks[c] = toy.inclusive_process == c
    return masks


def main(args : argparse.Namespace):
    particle = Particle.from_pdgid(211)

    pdfs = GenerateStackedPDFs(args.step)

    KE_init = GenerateIntialKEs(args.events, particle, args.momentum, args.width, args.beam_profile)

    nodes = os.cpu_count() if args.events > 1E4 else 1
    pools = ProcessPool(nodes = nodes)
    KE_init_split = np.array_split(KE_init, nodes)
    sim_args = (KE_init_split, [args.step]*nodes, [pdfs]*nodes)

    df = pd.concat(pools.map(Simulate, *sim_args), ignore_index = True)

    masks = CreateMasks(df)
    for m in masks:
        df[m] = masks[m]

    CountProcesses(df.inclusive_process)
    CountProcesses(df.exclusive_process)

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