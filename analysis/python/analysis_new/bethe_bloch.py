"""
Created on: 23/09/2026 16:00

Author: Shyam Bhuller

Description: Calculations involving the Bethe-Bloch formula.
"""
from dataclasses import dataclass

import awkward as ak
import numpy as np

from particle import Particle
from scipy.interpolate import interp1d

@dataclass
class Constants:
    rho = 1.39 # [g/cm3] density of LAr
    K = 0.307075 # [MeV cm2 / mol]
    Z = 18 # LAr atomic number
    A = 39.948 # [g/mol] LAr atomic mass
    I = 188E-6 # [MeV] mean excitation energy
    me = Particle.from_pdgid(11).mass # [MeV] electron mass

@dataclass
class DensityCorrectionParameters:
    C  : float = 5.2146
    y0 : float = 0.2
    y1 : float = 3
    a  : float = 0.19559
    k  : float = 3

    def density_correction(self, beta : float | ak.Array, gamma : float | ak.Array) -> float | ak.Array:
        """ Correction to account for the fact a particles electric field flattens and spreads as the energy increases.

        Args:
            beta (float | ak.Array): velocity
            gamma (float | ak.Array): relativistic factor

        Returns:
            (float | ak.Array): density correction value
        """
        y = np.log10(beta * gamma)

        delta_0 = 2 * np.log(10) * y - self.C
        delta_1 = delta_0 + self.a * (self.y1 - y)**self.k

        if hasattr(y, "__iter__"):
            delta = ak.where(y >= self.y1, delta_0, 0) 
            delta = ak.where((self.y0 <= y) & (y < self.y1), delta_1, delta)
        else:
            if y >= self.y1:
                delta = delta_0
            elif y < self.y0:
                delta = 0
            else:
                delta = delta_1

        return delta


def mean_dEdX(KE : float | ak.Array, particle : Particle) -> float | ak.Array:
    """ Calculate the mean dEdX for a particle with given kinetic energy.

    Args:
        KE (float | ak.Array): particle kinetic energy
        particle (Particle): particle type

    Returns:
        float | ak.Array: mean dEdX
    """
    gamma = (KE / particle.mass) + 1
    beta = (1 - (1/gamma)**2)**0.5

    w_max = 2 * Constants.me * (beta * gamma)**2 / (1 + (2 * Constants.me * (gamma/particle.mass)) + (Constants.me/particle.mass)**2)
    N = np.divide((Constants.rho * Constants.K * Constants.Z * (particle.charge)**2), (Constants.A * (beta**2)))
    B = 0.5 * np.log(2 * Constants.me * (gamma**2) * (beta**2) * w_max / ((Constants.I) **2))
    C = beta**2

    delta = DensityCorrectionParameters()
    D = 0.5 * delta.density_correction(beta, gamma)

    dEdX = N * (B - C - D)

    dEdX = np.nan_to_num(dEdX)
    if hasattr(KE, "__iter__"):
        dEdX = ak.where(dEdX < 0, 0, dEdX) # handle when np.log is -infinity i.e. when KE = 0
    else:
        if dEdX < 0: dEdX = 0
    return dEdX


def interp_KE_to_mean_dEdX(inital_KE : float, stepsize : float, particle : Particle = Particle.from_pdgid(211)) -> interp1d:
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
    while e >= 0:
        KE.append(e)
        dEdX.append(mean_dEdX(e, particle))
        e = e - stepsize * dEdX[-1]
        if dEdX[-1] <= 0: break # sometines bethebloch produces an unphysical value when KE is too small, so stop
    KE.append(0)
    dEdX.append(np.inf)
    return interp1d(KE, dEdX, fill_value = 0, bounds_error = False) # if outside the interpolation range, return 0


def interp_range_to_KE(KE_init : float, precision = 0.05) -> interp1d:
    """ Create an interpolation object for the range of a particle and its kinetic energy

    Args:
        KE_init (float): kinetic energy
        precision (float, optional): position step. Defaults to 0.05.

    Returns:
        interp1d: interpolated map of range and KE
    """
    KE = [KE_init]
    track_length = [0]
    count = 0
    while KE[-1] > 0:
        KE.append(KE[-1] - precision * mean_dEdX(KE[-1], Particle.from_pdgid(-13)))
        count += 1
        track_length.append(count * precision)
    track_length = np.array(track_length)

    return interp1d(max(track_length) - track_length, KE, fill_value = 0, bounds_error = False)


def KE_end(KE_init : ak.Array, track_length : ak.Array, n : int) -> ak.Array:
    """ Compute the interacting energy from the particles initial kinetic energy and track length.

    Args:
        KE_init (ak.Array): initial kinetic energies
        track_length (ak.Array): track lengths
        n (int): number of iterations, higher results in more accurate values but takes exponentially longer.

    Returns:
        ak.Array: interacting KEs
    """
    interpolated_energy_loss = interp_KE_to_mean_dEdX(2*max(KE_init), 1) # precompute the energy loss and create a function to interpolate between them
    steps = track_length/n

    KE_int = KE_init
    for i in range(n):
        KE_int = KE_int - interpolated_energy_loss(KE_int)*steps
    KE_int = ak.where(KE_int < 0, 0, KE_int)
    return KE_int


def range_from_KE(KE_init : np.ndarray, particle : Particle, precision : float = 1) -> ak.Array:
    """ Compute the range of particles from the  initial kinetic energy.

    Args:
        KE_init (np.ndarray): initial kinetic energies
        particle (Particle): particle type
        precision (float, optional): position step. Defaults to 1.

    Returns:
        ak.Array: ranges
    """
    interpolated_energy_loss = interp_KE_to_mean_dEdX(2*max(KE_init), precision/2, particle) # precompute the energy loss and create a function to interpolate between them
    KE = np.array(KE_init)
    n = np.zeros(len(KE_init))
    while any(KE > 0):
        KE = KE - precision * interpolated_energy_loss(KE)
        n = n + (KE > 0)
    return n * precision
