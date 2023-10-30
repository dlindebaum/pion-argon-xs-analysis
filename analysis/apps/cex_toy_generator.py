#!/usr/bin/env python3
"""
Created on: 08/08/2023 16:18

Author: Shyam Bhuller

Description: Generates toy events for inelastic pion interactions.
"""
import argparse
import os
import warnings

from collections import Counter
from math import ceil

import numpy as np
import pandas as pd
import tables

from pathos.pools import ProcessPool
from rich import print
from scipy.interpolate import interp1d

from python.analysis.Master import timer
from python.analysis import Fitting
from python.analysis.cross_section import ApplicationArguments, BetheBloch, GeantCrossSections, Particle, LoadConfiguration, LoadSelectionFile

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # supress annoying pandas warnings

global verbose

def vprint(*args, **kwags):
    if verbose is True:
        print(*args, **kwags)

def ReadHDF5File(file : str):
    """ Reads a HDF5 file and unpacks the contents into pandas dataframes.

    Args:
        file (str): file path.

    Returns:
        pd.DataFrame : if hdf5 file only has 1 key
        dict : if hdf5 file contains more than 1 key
    """
    keys = []
    with tables.open_file(file, driver = "H5FD_CORE") as hdf5file:
        for c in hdf5file.root: 
            keys.append(c._v_pathname[1:])
    if len(keys) == 1:
        return pd.read_hdf(file)
    else:
        return {k : pd.read_hdf(file, k) for k in keys}


def ResolveConfig(config : dict) -> argparse.Namespace:
    """ Parse Toy configuration file.

    Args:
        args (argparse.Namespace): application arguments

    Returns:
        argparse.Namespace: parsed configuration
    """
    args = argparse.Namespace()
    for k, v in config.items():
        if k == "smearing_params":
            args.smearing_params = {}
            for i in v:
                args.smearing_params[i] = LoadConfiguration(v[i])
                args.smearing_params[i]["function"] = getattr(Fitting, args.smearing_params[i]["function"])
        elif k == "reco_region_fractions":
            args.reco_region_fractions = ReadHDF5File(v)
        elif k == "beam_selection_efficiencies":
            #* that complex unpacking with pytables
            args.beam_selection_efficiencies = ReadHDF5File(v)
        elif k == "mean_track_score_kde":
            args.mean_track_score_kde = LoadSelectionFile(v)
        else:
            setattr(args, k, v)
    return args


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


def GenerateStackedPDFs(l : float, path = os.environ.get('PYTHONPATH', '').split(os.pathsep)[0] + "/data/g4_xs.root", scale_factors : dict = None) -> dict:
    """ Creates a PDF of the inclusive cross section and a stacked PDF of the exclusive process,
        such that rejection sampling can be used to allocate an exclusive process for a given inclusive process.

    Args:
        l (float): slice thickness

    Returns:
        dict: dictionaty of pdfs
    """
    xs_sim = GeantCrossSections(file = path)

    exclusive_processes = [k for k in vars(xs_sim) if k not in ["KE", "file", "total_inelastic"]]

    if scale_factors is not None:
        sum_xs = np.zeros(len(xs_sim.KE))
        for k, v in scale_factors.items():
            sum_xs = sum_xs + v * getattr(xs_sim, k)
        ratio = sum_xs / xs_sim.total_inelastic # the amount per data point to scale each exclusive process by, such that the total inelastic cross section remains unchanged
    else:
        ratio = 1

    if scale_factors is not None:
        factors = {k : v/ratio for k, v in scale_factors.items()}
    else:
        factors = {k : 1 for k in exclusive_processes}

    pdfs = {}
    pdfs["total_inelastic"] = interp1d(xs_sim.KE, P_int(xs_sim.total_inelastic, l), fill_value = "extrapolate") # total inelastic pdf

    # sort the exclusice channels based on area under curve
    area = {}
    for v in exclusive_processes:
        area[v] = np.trapz(P_int(factors[v] * getattr(xs_sim, v), l))

    # stack the pdfs in ascending order
    y = np.zeros(len(xs_sim.KE))
    for v, _ in sorted(area.items(), key=lambda x: x[1]):
        y = y + P_int(factors[v] * getattr(xs_sim, v), l)
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

    KE_int[KE_int < 0] = 0

    return pd.DataFrame({
        "KE_init" : KE_init,
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
        vprint(p, sum(proc == p))


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

@timer
def Smearing(n : int, resolutions : dict) -> pd.DataFrame:
    """ Produces a dataframe of smearings for each kinematic output of the toy.
        Emulates detector effects.

    Args:
        n (int): number of events
        resolutions (dict): smearing parameters

    Returns:
        pd.DataFrame: smearing terms
    """
    return pd.DataFrame({
        "KE_int_smearing" : Fitting.RejectionSampling(n, min(resolutions["KE_int"]["range"]), max(resolutions["KE_int"]["range"]), resolutions["KE_int"]["function"], resolutions["KE_int"]["values"]),
        "KE_init_smearing" : Fitting.RejectionSampling(n, min(resolutions["KE_init"]["range"]), max(resolutions["KE_init"]["range"]), resolutions["KE_init"]["function"], resolutions["KE_init"]["values"]),
        "z_int_smearing" : Fitting.RejectionSampling(n, min(resolutions["z_int"]["range"]), max(resolutions["z_int"]["range"]), resolutions["z_int"]["function"], resolutions["z_int"]["values"]),
    })

@timer
def BeamSelectionEfficiency(toy : pd.DataFrame, key : str, beam_selection_params : dict):
    """ Produces a mask of events which would have passed the beam selection,
        based on the selection efficiency of a particular observable.

    Args:
        toy (pd.DataFrame): toy
        key (str): observable
        beam_selection_params (dict): selection efficiencies

    Returns:
        np.array: mask of events which pass the selection.
    """
    probability_index = np.digitize(toy[key].values, beam_selection_params["bins"][key])
    probability_index = np.clip(probability_index, 0, len(beam_selection_params["efficiency"][key]) - 1)
    probability = beam_selection_params["efficiency"][key][probability_index]

    U = np.random.uniform(0, 1, len(toy))
    return pd.DataFrame({"beam_selection_mask" : U < probability}).reset_index(drop = True)

@timer
def GenerateRecoRegions(exclusive_process : pd.Series, fractions : pd.DataFrame) -> pd.DataFrame:
    """ Returns a Dataframe of masks which represent the reco and true regions of each event.
        Truth regions are remade because they differ slighly from the exclusive process.

    Args:
        exclusive_process (pd.Series): exclusive processes.
        fractions (pd.DataFrame): fractions of reco regions in true regions.

    Returns:
        pd.DataFrame: region masks.
    """
    exclusive_processes = np.unique(exclusive_process.values)

    keys = list(fractions.columns)

    toy_reco_regions = np.array(["-"]*len(exclusive_process))
    for i in exclusive_processes:
        if i == "": continue
        if i in ["quasielastic", "double_charge_exchange"]: # can't distinguish these two processes in reco
            sample_from = "single_pion_production"
            # this is single pion production
        else:
            sample_from = i
        toy_reco_regions = np.where(exclusive_process == i, np.random.choice(keys, len(exclusive_process), p = fractions[sample_from]), toy_reco_regions)

    toy_reco_region_masks = {}
    for c in keys:
        if c == "-": continue
        toy_reco_region_masks[c] = toy_reco_regions == c

    toy_true_region_masks = {}
    for i in exclusive_processes:
        if i == "": continue
        if i in ["quasielastic", "double_charge_exchange"]:
            sample_from = "single_pion_production"
            # this is single pion production
        else:
            sample_from = i
        if sample_from in toy_true_region_masks:
            toy_true_region_masks[sample_from] = toy_true_region_masks[sample_from] | (exclusive_process.values == i)
        else:
            toy_true_region_masks[sample_from] = exclusive_process.values == i

    regions_df = {}
    for i in toy_reco_region_masks:
        regions_df[f"reco_regions_{i}"] = toy_reco_region_masks[i]

    for i in toy_true_region_masks:
        regions_df[f"truth_regions_{i}"] = toy_true_region_masks[i]

    regions_df = pd.DataFrame(regions_df)

    return regions_df


def GenerateMeanTrackScores(kde : "stats.gaussian_kde", n : int) -> np.array:
    values = kde.resample(n)[0]
    mask = (values > 1) | (values < 0)
    counter = 0
    while any(mask):
        values = values[~mask]
        values = np.concatenate([values, kde.resample(sum(mask))[0]])
        mask = (values > 1) | (values < 0)
    # values = np.where(values > 1, 1, values)
    # values = np.where(values < 0, 0, values)
    return values

@timer
def MeanTrackScore(exclusive_process : pd.Series, kdes : dict) -> np.array:
    scores = np.array([None]*len(exclusive_process))
    exclusive_processes, counts = np.unique(exclusive_process.values, return_counts = True)

    for i, c in zip(exclusive_processes, counts):
        if i == "": continue
        if i in ["quasielastic", "double_charge_exchange"]:
            sample_from = "single_pion_production"
            # this is single pion production
        else:
            sample_from = i
        scores = np.where(exclusive_process == i, GenerateMeanTrackScores(kdes[sample_from], len(exclusive_process)), scores)
    return pd.DataFrame({"mean_track_score" : scores}).reset_index(drop = True)


@timer
def main(args : argparse.Namespace):
    global verbose
    if hasattr(args, "verbose"):
        verbose = args.verbose
    else:
        verbose = True

    if not verbose:
        warnings.filterwarnings("ignore") # bad but removes clutter in the terminal output

    particle = Particle.from_pdgid(211)

    pdfs = GenerateStackedPDFs(args.step, scale_factors = args.pdf_scale_factors)

    KE_init = GenerateIntialKEs(args.events, particle, args.p_init, args.beam_width, args.beam_profile)

    if args.events < 1E5:
        nodes = 1
    else:
        nodes = ceil(args.events / 1E5)

    if nodes == 1:
        df = Simulate(KE_init, args.step, pdfs) # don't need to do multiprocessing
    else:
        KE_init_split = np.array_split(KE_init, nodes)
        batches = ceil(nodes / (os.cpu_count()-1))

        df = []
        for i in range(batches):
            KE_init_batches = KE_init_split[(os.cpu_count()-1) * i:(os.cpu_count()-1) * (i+1)]
            cpus = len(KE_init_batches)
            vprint(f"starting batch : {i}, cpus : {cpus}")
            pools = ProcessPool(nodes = cpus)
            pools.restart()
            sim_args = (KE_init_batches, [args.step]*cpus, [pdfs]*cpus)
            df.extend(pools.imap(Simulate, *sim_args))
            pools.close()
        
        vprint("Done! Creating dataframe...")
        df = pd.concat(df, ignore_index = True)

    masks = CreateMasks(df)
    for m in masks:
        df[m] = masks[m]

    CountProcesses(df.inclusive_process)
    CountProcesses(df.exclusive_process)

    smearings = Smearing(args.events, args.smearing_params)
    beam_selection_mask = BeamSelectionEfficiency(df, "z_int", args.beam_selection_efficiencies)
    region_masks = GenerateRecoRegions(df.exclusive_process, args.reco_region_fractions)
    scores = MeanTrackScore(df.exclusive_process, args.mean_track_score_kde)

    df = pd.concat([df, smearings, beam_selection_mask, region_masks, scores], axis = 1)

    # allow the app to be run in other scripts or notebooks.
    if __name__ == "__main__":
        vprint(df)

        if hasattr(args, "df_format"):
            fmt = args.df_format
        else:
            fmt = "f" # fixed format, fastest read/writes

        df.to_hdf(args.out + ".hdf5", key = "df", format = fmt)
    else:
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generates toy events for inelastic pion interactions.")
    parser.add_argument("-c", "--config", type = str, default = None, help = "Config json file.")
    ApplicationArguments.Output(parser)

    args = parser.parse_args()
    config = LoadConfiguration(args.config)
    args = argparse.Namespace(**vars(args), **vars(ResolveConfig(config)))
    args.events = int(args.events)
    if args.out is None:
        args.out = "toy_mc"
    args.out = args.out.split(".")[0]

    print(vars(args))
    main(args)