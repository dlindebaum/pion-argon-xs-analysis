"""
Created on: 23/09/2026 15:49

Author: Shyam Bhuller

Description: Functions to calculate cross section using the Spatial slice method. Note these are deprecated and likely don't work.
"""

import warnings

import awkward as ak
import numpy as np

from python.analysis import BetheBloch
from python.analysis.Slices import Slices
from python.analysis.Utils import nandiv


def deprecation_warning():
    return warnings.warn("ThinSlice functions are deprecrated!", DeprecationWarning)


def CountingExperiment(endPos : ak.Array, channel : ak.Array, slices : Slices) -> tuple[ak.Array, ak.Array]:
    """ Creates the interacting and incident histograms.

    Args:
        endPos (ak.Array): end position of particle or "interaction vertex"
        channel (ak.Array): mask which selects particles which interact in the channel you are interested in
        slices (Slices): spatial slices

    Returns:
        tuple[ak.Array, ak.Array]: n_interact and n_incident histograms
    """
    deprecation_warning()

    end_slice_pos = slices.pos_to_num(endPos)
    slice_nums = slices.num

    n_interact = np.histogram(end_slice_pos[channel], slice_nums)[0]

    total_interact = np.histogram(end_slice_pos, slice_nums)[0]
    n_incident = np.cumsum(total_interact[::-1])[::-1]
    return n_interact, n_incident


def MeanSliceEnergy(energy : ak.Array, endPos : ak.Array, slices : Slices) -> tuple[ak.Array, ak.Array]:
    """ Compute the average energy in a spatial slice.

    Args:
        energy (ak.Array): particle energies over its lifetime in the tpc
        endPos (ak.Array): end position of particle or "interaction vertex"
        slices (Slices): spatial slices

    Returns:
        tuple[ak.Array, ak.Array]: means slice energy, error in the mean slice energy
    """
    deprecation_warning()
    beam_traj_slice = slices.pos_to_num(endPos)
    slice_nums = slices.num
    
    counts = np.histogram(ak.ravel(beam_traj_slice), slice_nums)[0] # histogram of positions will give the counts

    weights = ak.ravel(np.nan_to_num(energy, 0))

    sum_energy = np.histogram(ak.ravel(beam_traj_slice), slice_nums, weights = weights)[0] # total energy in each bin if you weight by energy
    sum_energy_sqr = np.histogram(ak.ravel(beam_traj_slice), slice_nums, weights = weights**2)[0] # same as above

    mean_energy = sum_energy / counts

    std_energy = np.divide(sum_energy_sqr, counts) - mean_energy**2
    error_mean_energy = np.sqrt(np.divide(std_energy, counts))

    return mean_energy, error_mean_energy


def total_cross_section(n_incident : np.ndarray, n_interact : np.ndarray, slice_width : float) -> tuple[np.ndarray, np.ndarray]:
    """ Returns cross section in mb.

    Args:
        n_incident (np.ndarray): incident histogram
        n_interact (np.ndarray): interacting histogram
        slice_width (float): spatial width of thin slice

    Returns:
        tuple[np.ndarray, np.ndarray]: cross section, statistical uncertainty
    """
    deprecation_warning()
    xs = np.log(n_incident / (n_incident - n_interact)) # calculate a dimensionless cross section

    v_incident = n_incident # poisson uncertainty
    v_interact = n_interact*(1- (n_interact/n_incident)) # binomial uncertainty

    xs_e = (1/n_incident) * (1/(n_incident - n_interact)) * (n_interact**2 * v_incident + n_incident**2 * v_interact)**0.5

    NA = 6.02214076e23
    factor = 10**27 * BetheBloch.A  / (BetheBloch.rho * NA * slice_width)

    return factor * xs, abs(factor * xs_e)


def CrossSection(n_int_exclusive : np.ndarray, n_int_inclusive : np.ndarray, n_inc_inclusive : np.ndarray, slice_width : float) -> tuple[np.ndarray, np.ndarray]:
    """ Cross section of exclusive process.

    Args:
        n_int_exclusive (np.ndarray): exclusive interactions
        n_int_inclusive (np.ndarray): interactions
        n_inc_inclusive (np.ndarray): incident counts
        slice_width (float): slice width

    Returns:
        tuple[np.ndarray, np.ndarray]: cross section and error
    """
    deprecation_warning()
    NA = 6.02214076e23
    factor = 10**27 * BetheBloch.A  / (BetheBloch.rho * NA * slice_width)

    n_interact_ratio = nandiv(n_int_exclusive, n_int_inclusive)
    n_survived_inclusive = n_inc_inclusive - n_int_inclusive

    var_inc_inclusive = n_inc_inclusive # poisson variance
    var_int_inclusive = n_int_inclusive * (1 - nandiv(n_int_inclusive, n_inc_inclusive)) # binomial uncertainty
    var_int_exclusive = n_int_exclusive * (1 - nandiv(n_int_exclusive, n_inc_inclusive)) # binomial uncertainty

    xs = factor * n_interact_ratio * np.log(nandiv(n_inc_inclusive, n_inc_inclusive - n_int_inclusive))

    diff_n_int_exclusive = nandiv(xs, n_int_exclusive)
    diff_n_inc_inclusive = factor * n_interact_ratio * (nandiv(1, n_inc_inclusive) - nandiv(1, n_survived_inclusive))
    diff_n_int_inclusive = factor * n_interact_ratio * nandiv(1, n_survived_inclusive) - nandiv(xs, n_int_inclusive)

    xs_err = ((diff_n_int_exclusive**2 * var_int_exclusive) + (diff_n_inc_inclusive**2 * var_inc_inclusive) + (diff_n_int_inclusive**2 * var_int_inclusive))**0.5
    return xs, xs_err
