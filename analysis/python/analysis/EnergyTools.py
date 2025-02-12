"""
Created on: 02/06/2023 10:37

Author: Shyam Bhuller

Description: Library for code used in the cross section analysis. Refer to the README to see which apps correspond to the cross section analysis.
"""
import awkward as ak
import numpy as np

from particle import Particle
from scipy.interpolate import interp1d

from python.analysis import SelectionTools, Fitting, vector
from python.analysis.Master import LoadConfiguration, LoadObject, SaveObject, SaveConfiguration, ReadHDF5, Data, Ntuple_Type, timer, IO
# from python.analysis.Utils import *

def KE(p, m):
    return (p**2 + m**2)**0.5 - m

def IsScraper(mc : Data, beam_scraper_args : dict) -> ak.Array:
    # get kinetic energy from beam instrumentation
    beam_inst_KE = KE(mc.recoParticles.beam_inst_P,
                      Particle.from_pdgid(211).mass)
    true_ffKE = mc.trueParticles.beam_KE_front_face

    delta_KE_upstream = beam_inst_KE - true_ffKE

    scraper_ids = {}
    for k, v in beam_scraper_args.items():
        scraper_ids[k] = ((beam_inst_KE > min(v["bins"]))
                          & (beam_inst_KE < max(v["bins"])))
        threshold = v["mu_e_res"] + 3 * abs(v["sigma_e_res"])
        
        scraper_ids[k] = scraper_ids[k] & (delta_KE_upstream > threshold)
    scraper_ids = SelectionTools.CombineMasks(
        scraper_ids, operator = "or")
    return scraper_ids


class EnergyCorrection:
    @staticmethod
    def LinearCorrection(x, p0):
        return x / p0

    class ResponseFit(Fitting.FitFunction):
        n_params = 3

        @staticmethod
        def func(x, p0, p1, p2):
            return p0 * np.log(x - p1) + p2

        @staticmethod
        def p0(x, y):
            return None

    @staticmethod
    def ResponseCorrection(x, p0, p1, p2):
        return x / (EnergyCorrection.ResponseFit.func(x, p0, p1, p2) + 1)

    shower_energy_correction = {
        "linear" : LinearCorrection,
        "response": ResponseCorrection,
        None : None
    }


class BetheBloch:
    rho = 1.39 # [g/cm3] density of LAr
    K = 0.307075 # [MeV cm2 / mol]
    Z = 18 # LAr atomic number
    A = 39.948 # [g/mol] LAr atomic mass
    I = 188E-6 # [MeV] mean excitation energy
    me = Particle.from_pdgid(11).mass # [MeV] electron mass

    # density correction parameters
    C = 5.2146
    y0 = 0.2
    y1 = 3
    a = 0.19559
    k = 3

    @staticmethod
    def densityCorrection(beta : float | ak.Array, gamma : float | ak.Array) -> float | ak.Array:
        """ Correction to account for th fact a particles electric field flattens and spreads as the energy increases.

        Args:
            beta (float | ak.Array): velocity
            gamma (float | ak.Array): relativistic factor

        Returns:
            (float | ak.Array): density correction value
        """
        y = np.log10(beta * gamma)

        delta_0 = 2 * np.log(10) * y - BetheBloch.C
        delta_1 = delta_0 + BetheBloch.a * (BetheBloch.y1 - y)**BetheBloch.k

        if hasattr(y, "__iter__"):
            delta = ak.where(y >= BetheBloch.y1, delta_0, 0) 
            delta = ak.where((BetheBloch.y0 <= y) & (y < BetheBloch.y1), delta_1, delta)
        else:
            if y >= BetheBloch.y1:
                delta = delta_0
            elif y < BetheBloch.y0:
                delta = 0
            else:
                delta = delta_1

        return delta

    @staticmethod
    def meandEdX(KE : float | ak.Array, particle : Particle) -> float | ak.Array:
        """ Calculate the mean dEdX for a particle with given kinetic energy.

        Args:
            KE (float | ak.Array): particle kinetic energy
            particle (Particle): particle type

        Returns:
            float | ak.Array: mean dEdX
        """
        gamma = (KE / particle.mass) + 1
        beta = (1 - (1/gamma)**2)**0.5

        w_max = 2 * BetheBloch.me * (beta * gamma)**2 / (1 + (2 * BetheBloch.me * (gamma/particle.mass)) + (BetheBloch.me/particle.mass)**2)
        N = np.divide((BetheBloch.rho * BetheBloch.K * BetheBloch.Z * (particle.charge)**2), (BetheBloch.A * (beta**2)))
        A = 0.5 * np.log(2 * BetheBloch.me * (gamma**2) * (beta**2) * w_max / ((BetheBloch.I) **2))
        B = beta**2
        C = 0.5 * BetheBloch.densityCorrection(beta, gamma)

        dEdX = N * (A - B - C)

        dEdX = np.nan_to_num(dEdX)
        if hasattr(KE, "__iter__"):
            dEdX = ak.where(dEdX < 0, 0, dEdX) # handle when np.log is -infinity i.e. when KE = 0
        else:
            if dEdX < 0: dEdX = 0
        return dEdX

    @staticmethod
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
            dEdX.append(BetheBloch.meandEdX(e, particle))
            e = e - stepsize * dEdX[-1]
            if dEdX[-1] <= 0: break # sometines bethebloch produces an unphysical value when KE is too small, so stop
        KE.append(0)
        dEdX.append(np.inf)
        return interp1d(KE, dEdX, fill_value = 0, bounds_error = False) # if outside the interpolation range, return 0

    @staticmethod
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
            KE.append(KE[-1] - precision * BetheBloch.meandEdX(KE[-1], Particle.from_pdgid(-13)))
            count += 1
            track_length.append(count * precision)
        track_length = np.array(track_length)

        return interp1d(max(track_length) - track_length, KE, fill_value = 0, bounds_error = False)


    @staticmethod
    def InteractingKE(KE_init : ak.Array, track_length : ak.Array, n : int) -> ak.Array:
        """ Compute the interacting energy from the particles initial kinetic energy and track length.

        Args:
            KE_init (ak.Array): initial kinetic energies
            track_length (ak.Array): track lengths
            n (int): number of iterations, higher results in more accurate values but takes exponentially longer.

        Returns:
            ak.Array: interacting KEs
        """
        interpolated_energy_loss = BetheBloch.interp_KE_to_mean_dEdX(2*max(KE_init), 1) # precompute the energy loss and create a function to interpolate between them
        steps = track_length/n

        KE_int = KE_init
        for i in range(n):
            KE_int = KE_int - interpolated_energy_loss(KE_int)*steps
        KE_int = ak.where(KE_int < 0, 0, KE_int)
        return KE_int

    @staticmethod
    def RangeFromKE(KE_init : np.ndarray, particle : Particle, precision : float = 1) -> ak.Array:
        """ Compute the range of particles from the  initial kinetic energy.

        Args:
            KE_init (np.ndarray): initial kinetic energies
            particle (Particle): particle type
            precision (float, optional): position step. Defaults to 1.

        Returns:
            ak.Array: ranges
        """
        interpolated_energy_loss = BetheBloch.interp_KE_to_mean_dEdX(2*max(KE_init), precision/2, particle) # precompute the energy loss and create a function to interpolate between them
        KE = np.array(KE_init)
        n = np.zeros(len(KE_init))
        while any(KE > 0):
            KE = KE - precision * interpolated_energy_loss(KE)
            n = n + (KE > 0)
        return n * precision


def UpstreamEnergyLoss(KE_inst : ak.Array, params : np.ndarray, function : Fitting.FitFunction = Fitting.poly2d) -> ak.Array:
    """ compute the upstream loss based on a repsonse function and it's fit parameters.

    Args:
        KE_inst (ak.Array): kinetic energy measured by the beam instrumentation
        function (Fitting.FitFunction): repsonse function, defaults to Fitting.poly2d. 
        params (np.ndarray): function paramters

    Returns:
        ak.Array: upstream energy loss
    """
    return function.func(KE_inst, **params)

@timer
def RecoDepositedEnergy(events : Data, ff_KE : ak.Array, method : str) -> ak.Array:
    """ Calcuales the energy deposited by the beam particle in the TPC, either using calorimetric information or the bethe bloch formula (spatial information).

    Args:
        events (Data): events to look at
        ff_KE (ak.Array): front facing kinetic energy
        method (str): method to calcualte the deposited energy, either "calo" or "bb"

    Returns:
        ak.Array: depotisted energy
    """
    reco_pitch = vector.dist(events.recoParticles.beam_calo_pos[:, :-1], events.recoParticles.beam_calo_pos[:, 1:]) # distance between reconstructed calorimetry points
    
    if method == "calo":
        dE = ak.sum(events.recoParticles.beam_dEdX[:, :-1] * reco_pitch, -1)
    elif method == "bb":
        KE_int_bb = BetheBloch.InteractingKE(ff_KE, events.recoParticles.beam_track_length, 50)
        dE = ff_KE - KE_int_bb
    else:
        raise Exception(f"{method} not a valid method, pick 'calo' or 'bb'")
    return dE

def RecoDepositedEnergyFiducial(events : Data, ff_KE : ak.Array, fid_end : float) -> ak.Array:
    """ Calcuales the energy deposited by the beam particle in the TPC, either using calorimetric information or the bethe bloch formula (spatial information).

    Args:
        events (Data): events to look at
        ff_KE (ak.Array): front facing kinetic energy
        fid_end (float): Fiducial z end position in cm

    Returns:
        ak.Array: depotisted energy,
        ak.Array: Track lengths
    """
    valid_mask = events.recoParticles.beam_calo_pos.z < fid_end
    valid_points = events.recoParticles.beam_calo_pos[valid_mask]
    track_lens = ak.sum(vector.dist(valid_points[..., 1:], valid_points[..., :-1]), axis=-1)
    # invalids = events.recoParticles.beam_calo_pos.z >= fid_end
    # assert vector.dist(valid_points[..., -1], events.recoParticles.beam_calo_pos[invalids][..., 0]) < 5

    # reco_pitch = vector.dist(events.recoParticles.beam_calo_pos[:, :-1], events.recoParticles.beam_calo_pos[:, 1:]) # distance between reconstructed calorimetry points
    KE_int_bb = BetheBloch.InteractingKE(ff_KE, track_lens, 50)
    dE = ff_KE - KE_int_bb
    return dE

def TrueInitialEnergyFiducial(events, fiducial_start):
    index_before_tpc = ak.argmax(
        events.trueParticles.beam_traj_pos.z > fiducial_start,
        axis=-1, keepdims = True) - 1
    index_before_tpc = ak.where(index_before_tpc < 0, 0, index_before_tpc)
    not_in_tpc = ak.sum(
        events.trueParticles.beam_traj_pos.z > fiducial_start,
        axis=-1, keepdims = True) == 0
    # Force to match interacting = initial energy, which means the
    #   event will be removed
    # In other words the last recorded pre-interaction energy before
    #   the TPC was entered.
    index_before_tpc = ak.where(not_in_tpc, -2, index_before_tpc)
    return ak.flatten(events.trueParticles.beam_traj_KE[index_before_tpc])

def TrueInitialEnergyFiducialInterpolate(events, fiducial_start):
    hit_counts = ak.count(
        events.trueParticles.beam_traj_pos.z,
        axis=-1, keepdims=True) - 1
    ind_before_fiducial = ak.sum(
        events.trueParticles.beam_traj_pos.z <= fiducial_start,
        axis=-1, keepdims=True) - 1
    ind_after_fiducial = ak.where(
        ind_before_fiducial == hit_counts,
        0, ind_before_fiducial + 1)
    before_tpc_energy = events.trueParticles.beam_traj_KE[ind_before_fiducial]
    beam_z = events.trueParticles.beam_traj_pos.z
    extra_lengths = (
        ((fiducial_start-beam_z[ind_before_fiducial])
         /(beam_z[ind_after_fiducial]-beam_z[ind_before_fiducial]))
        * vector.dist(events.trueParticles.beam_traj_pos[ind_after_fiducial],
                      events.trueParticles.beam_traj_pos[ind_before_fiducial])
    )
    fid_start_energy = ak.where(
        ind_before_fiducial[..., 0] > 0,
        EnergyTools.BetheBloch.InteractingKE(
            before_tpc_energy[..., 0], extra_lengths[..., 0], n=5),
        before_tpc_energy[..., 0])
    return ak.where(ind_before_fiducial[..., 0] == hit_counts[..., 0],
                                 events.trueParticles.beam_traj_KE[..., -2],
                                 fid_start_energy)

def TrueEndEnergyFiducial(events, fiducial_end):
    index_before_tpc_end = ak.sum(
        events.trueParticles.beam_traj_pos.z < fiducial_end,
        axis=-1, keepdims = True) - 1
    neg_ind_before_end = index_before_tpc_end - ak.count(
        events.trueParticles.beam_traj_pos.z, axis=-1)
    in_tpc_interaction = neg_ind_before_end == -1
    neg_ind_before_end = ak.where(
        in_tpc_interaction, -2, neg_ind_before_end)
    return (
        ak.flatten(events.trueParticles.beam_traj_KE[neg_ind_before_end]),
        ak.flatten(in_tpc_interaction))
