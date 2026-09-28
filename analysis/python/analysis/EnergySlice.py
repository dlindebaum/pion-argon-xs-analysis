"""
Created on: 23/09/2026 15:54

Author: Shyam Bhuller

Description: Functions to calculate cross section using the Energy slice method.
"""
from collections.abc import Iterable
import warnings

import awkward as ak
import numpy as np

from particle import Particle

from python.analysis import BetheBloch
from python.analysis.Slices import Slices
from python.analysis.Utils import nandiv, quadsum, nanlog, deprecation_warning


def process_multiple_array(input : np.ndarray, function : callable) -> np.ndarray:
    """ Call function on multiple arrays.

    Args:
        input (np.ndarray): input arrays.
        function (callable): function to call.

    Raises:
        ValueError: At least one array must be passed.
        TypeError: only arrays should be passed.

    Returns:
        list[any]: outputs.
    """
    out = []
    if len(input) == 0:
        raise ValueError("At least one energy array must be provided.")

    for i in input:
        if isinstance(i, Iterable) and not isinstance(i, str):
            out.append(function(i))
        else:
            raise TypeError("inputs passed should be an array of values.")

    if len(out) == 1:
        return out[0]
    else:
        return out


def convert_energy_to_slice(slices : Slices, *energy : np.ndarray) -> np.ndarray | tuple[np.ndarray]:
    """ Converts energy distributions to slice number distributions.

    Args:
        slices (Slices): Energy slices.
        energy (np.ndarray): Energy distribution, multiple can be passed.

    Returns:
        np.ndarray | tuple[np.ndarray]: Slice distributions, equal to the number of energy distributions passed. 
    """
    func = lambda x : slices(x).num
    return process_multiple_array(energy, func)


def count(slices : Slices, *slice : np.ndarray) -> tuple[np.ndarray]:
    """ Produce counts of each slice.

    Args:
        slices (Slices): Energy slices.
        slice (np.ndarray): slice distribution, multiple can be passed.

    Returns:
        tuple[np.ndarray]: Slice distributions, equal to the number of slice distributions passed.
    """
    slice_bins = np.arange(slices.underflow_num - 0.5, slices.overflow_num + 1.5)

    func = lambda x: np.histogram(np.array(x), slice_bins)[0]

    return process_multiple_array(slice, func)


def incident(n_init : np.ndarray, n_end : np.ndarray) -> np.ndarray:
    """ Calculates the incident counts for each slice.

    Args:
        n_init (np.ndarray): Initial counts for each slice.
        n_end (np.ndarray): End counts for each slice

    Returns:
        np.ndarray: Incident counts for each slice.
    """
    c_init = np.cumsum(n_init)
    c_end = np.cumsum(n_end)

    return c_init - n_init - c_end + n_end


def complete_slice(init_slice : np.ndarray, end_slice : np.ndarray) -> np.ndarray:
    """ Whether a particle is incident on at least one slice.

    Args:
        init_slice (np.ndarray): Initial slice.
        end_slice (np.ndarray): End slice.

    Returns:
        np.ndarray: Flag to indicate particles with at least one incident slice.
    """
    return (init_slice != end_slice)


def counting_experiment_exclusive(energy_slices : Slices, KE_init : np.ndarray, KE_end : np.ndarray, mask : np.ndarray, outside_fv : np.ndarray) -> np.ndarray:
    """ Perform counting experiment to get the exclusive interacing slices for a particular subset of interactions. 

    Args:
        energy_slices (Slices): Energy slices.
        KE_init (np.ndarray): Initial kinetic energy.
        KE_end (np.ndarray): End kinetic energy.
        mask (np.ndarray): Mask of particles to include in the count.
        outside_fv (np.ndarray): Mask that excludes events that end outside the bounds of the fiducial volume.

    Returns:
        np.ndarray: Exclusive interacting counts.
    """
    selected = mask & ~outside_fv
    s_init, s_int = convert_energy_to_slice(energy_slices, KE_init[selected], KE_end[selected])
    complete_slice = complete_slice(s_init, s_int)
    return count(energy_slices, s_int[complete_slice])


def counting_experiment(KE_init : np.ndarray, KE_end : np.ndarray, slices : Slices, outside_fv : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Perform counting experiment to get initial, end and incident counts.

    Args:
        KE_init (np.ndarray): Initial kinetic energy
        KE_end (np.ndarray): end kinetic energy
        slices (Slices): Energy slices
        outside_fv (np.ndarray): Mask that excludes events that end outside the bounds of the fiducial volume.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: initial counts, end counts and incident counts
    """
    if slices.reversed is False:
        raise Exception("Energy slices should be in reverse order.")

    init_slice, end_slice = convert_energy_to_slice(slices, KE_init, KE_end)

    init_slice = init_slice[~outside_fv]
    end_slice = end_slice[~outside_fv]

    # particles that end and start in the same slice do not count towards the sample counted (they are not incident on any slice)
    # the incident calculation accounts for this, but not the histogramming for init and end.
    valid_slices = complete_slice(init_slice, end_slice)
    init_slice = init_slice[valid_slices]
    end_slice = end_slice[valid_slices]

    init_counts, end_counts = count(slices, init_slice, end_slice)
    inc_counts = incident(init_counts, end_counts)
    return init_counts, end_counts, inc_counts


def slice_dEdX(energy_slices : Slices, particle : Particle) -> np.ndarray:
    """ Computes the mean dEdX between energy slices.

    Args:
        energy_slices (Slices): energy slices
        particle (Particle): particle

    Returns:
        np.ndarray: mean dEdX
    """
    return BetheBloch.mean_dEdX(energy_slices.edges - energy_slices.width/2, particle)


def total_cross_section(n_incident : np.ndarray, n_end : np.ndarray, dEdX : np.ndarray, dE : float) -> tuple[np.ndarray, np.ndarray]:
    """ Calculate total inelastic cross section in mb.

    Args:
        n_incident (np.ndarray): incident counts.
        n_end (np.ndarray): end counts.
        dEdX (np.ndarray): mean slice dEdX.
        dE (float): energy slice width.

    Returns:
        tuple[np.ndarray, np.ndarray]: Total inelastic cross section and statistical undertainty in mb.
    """
    slice_width = dE/dEdX

    xs = np.log(n_incident / (n_incident - n_end)) # calculate a dimensionless cross section

    v_incident = n_incident # poisson uncertainty
    v_interact = n_end * (1- (n_end/n_incident)) # binomial uncertainty

    xs_e = (1/n_incident) * (1/(n_incident - n_end)) * (n_end**2 * v_incident + n_incident**2 * v_interact)**0.5

    NA = 6.02214076e23
    factor = 10**27 * BetheBloch.Constants.A  / (BetheBloch.Constants.rho * NA * slice_width)

    return factor * xs, abs(factor * xs_e)


def exclusive_cross_section(n_incident : np.ndarray, n_end : np.ndarray, n_int : np.ndarray, dEdX : np.ndarray, dE : float) -> tuple[np.ndarray, np.ndarray]:
    xs, xs_err = total_cross_section(n_incident, n_end, dEdX, dE)

    ratio = n_int / n_end
    ratio_err = ratio * (1/n_end + 1/n_int)**0.5

    return ratio * xs,  quadsum([ratio_err, xs_err], 0)

#! Deprecated.
def NIncident(n_initial : np.ndarray, n_end : np.ndarray) -> np.ndarray:
    """
        Calculate number of incident particles.
        Function is legacy and should not be used in new implementations.

    Args:
        n_initial (np.ndarray): initial particle counts
        n_end (np.ndarray): interaction counts

    Returns:
        np.ndarray: incident counts
    """
    deprecation_warning()
    n_survived_all = np.cumsum(n_initial - n_end)
    n_incident = n_survived_all + n_end
    return n_incident

#! Deprecated.
def SliceNumbers(int_energy : ak.Array, init_energy : ak.Array, outside_tpc : ak.Array, energy_slices : Slices) -> tuple[np.ndarray, np.ndarray]:
    """
        Convert energies from physical units to slice numbers.
        Function is legacy and should not be used in new implementations.

    Args:
        int_energy (ak.Array): interaction energy
        init_energy (ak.Array): initial energy
        outside_tpc (ak.Array): mask of particles which interact outside the fiducial volume
        energy_slices (Slices): energy slices

    Returns:
        tuple[np.ndarray, np.ndarray]: initial slice numbers and interacitng slice numbers
    """
    deprecation_warning()
    init_slice = energy_slices(init_energy).num + 1 # equivilant to ceil
    int_slice = energy_slices(int_energy).num

    # removes instances where the particle incident energy and interacting energy are in the same slice (Yinrui calls this an incomplete slice)
    # i.e. this happens if the particle interacting in its first slice, must be an artifact of the energy slicing because a particle that starts and interacts in a slice is thus not incident on any slice.
    bad_slices = (int_slice < init_slice) | outside_tpc
    init_slice = ak.where(bad_slices, -1, init_slice)
    int_slice = ak.where(bad_slices, -1, int_slice)
    return init_slice, int_slice

#! Deprecated
def CountingExperiment(int_energy : ak.Array, init_energy : ak.Array, outside_tpc : ak.Array, process : ak.Array, energy_slices : Slices, interact_only : bool = False, weights : np.ndarray = None) -> tuple[np.ndarray]:
    """
        Creates the interacting and incident histograms.
        Function is legacy and should not be used in new implementations.

    Args:
        int_energy (ak.Array): interacting enrgy
        init_energy (ak.Array): initial energy
        outside_tpc (ak.Array): mask of particles which interact outside the fiducial volume
        process (ak.Array): mask of events for exclusive interactions
        energy_slices (Slices): energy slices
        interact_only (bool, optional): only return exclusive interaction histogram. Defaults to False.
        weights (np.ndarray, optional): event weights. Defaults to None.

    Returns:
        np.ndarray | tuple[np.ndarray]: exclusive interaction histogram and/or initial histogram, incident histogram and interaction histogram 
    """
    deprecation_warning()
    init_slice, int_slice = SliceNumbers(int_energy, init_energy, outside_tpc, energy_slices)

    slice_bins = np.arange(-1 - 0.5, energy_slices.max_num + 1.5)

    exclusive_weights = weights[process] if weights is not None else None

    n_interact_exclusive = np.histogram(np.array(int_slice[process]), slice_bins, weights = exclusive_weights)[0]
    if interact_only == False:
        n_initial = np.histogram(np.array(init_slice), slice_bins, weights = weights)[0]
        n_interact_inelastic = np.histogram(np.array(int_slice), slice_bins, weights = weights)[0]

        n_incident = NIncident(n_initial, n_interact_inelastic)

        return n_initial, n_interact_inelastic, n_interact_exclusive, n_incident
    else:
        return n_interact_exclusive

#! Deprecated
def CrossSection(n_int_ex : np.ndarray, n_int : np.ndarray, n_inc : np.ndarray, dEdX : np.ndarray, dE : float, n_int_ex_err : np.ndarray = None, n_int_err : np.ndarray = None, n_inc_err : np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """ Compute exclusive cross sections. If interactions errors are not provided, staticial uncertainties are used (poisson for incident, binomial for interactions).
        Function is legacy and should not be used in new implementations.

    Args:
        n_int_ex (np.ndarray): exclusive interactions
        n_int (np.ndarray): interactions
        n_inc (np.ndarray): incident counts
        dEdX (np.ndarray): slice dEdX
        dE (float): energy slice width
        n_int_ex_err (np.ndarray, optional): exclusive interaction errors. Defaults to None.
        n_int_err (np.ndarray, optional): interaction errors. Defaults to None.
        n_inc_err (np.ndarray, optional): incident count errors. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    deprecation_warning()
    NA = 6.02214076e23
    factor = np.array(dEdX) * 10**27 * BetheBloch.Constants.A  / (BetheBloch.Constants.rho * NA * dE)

    n_interact_ratio = nandiv(n_int_ex, n_int)
    n_survived = n_inc - n_int

    if n_inc_err is not None:
        var_inc_inclusive = n_inc_err**2
    else:
        var_inc_inclusive = n_inc # poisson variance

    if n_int_err is not None:
        var_int = n_int_err**2
    else:
        var_int = n_int * (1 - nandiv(n_int, n_inc)) # binomial uncertainty

    if n_int_ex_err is not None:
        var_int_ex = n_int_ex_err**2
    else:
        var_int_ex = n_int_ex * (1 - nandiv(n_int_ex, n_inc)) # binomial uncertainty


    xs = factor * n_interact_ratio * nanlog(nandiv(n_inc, n_inc - n_int))

    diff_n_int_ex = nandiv(xs, n_int_ex)
    diff_n_inc = factor * n_interact_ratio * (nandiv(1, n_inc) - nandiv(1, n_survived))
    diff_n_int = factor * n_interact_ratio * nandiv(1, n_survived) - nandiv(xs, n_int)

    xs_err = ((diff_n_int_ex**2 * var_int_ex) + (diff_n_inc**2 * var_inc_inclusive) + (diff_n_int**2 * var_int))**0.5
    return np.array(xs, dtype = float), np.array(xs_err, dtype = float)
