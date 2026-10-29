"""
Created on: 20/02/2026 12:12

Author: Shyam Bhuller

Description: True process defintions.
"""
from dataclasses import dataclass

import awkward as ak

from particle import Particle

from .common import *
from python.analysis import ntuple, kinematics, vector

@dataclass
class process_criteria_geant(CriteriaList):
    """ Criteria defined on the true particle counts.
        Distinguishes charged pions to categorise double charge exchange and quasi-elastic.
    """
    pi_inelastic : Criteria # is pi inelatstic interaction
    n_pi_plus : Criteria # number of pi pluses
    n_pi_minus : Criteria # number of charged pi minuses
    n_pi0 : Criteria # numbre of pi0's

    @staticmethod
    def get_criteria_values(events : ntuple.Data, pi_KE_lim : float = 0):

        n_pip_true, n_pim_true, n_pi0_true = get_true_pion_counts_geant(events, pi_KE_lim) # counts number of pions in events
        pi_inel = events.true_particles.true_beam_end_process == "pi+Inelastic" # used to distinguish interactions from decays/non-pions
        return {"pi_inelastic" : pi_inel, "n_pi_plus" : n_pip_true, "n_pi_minus" : n_pim_true, "n_pi0" : n_pi0_true}


@dataclass
class process_criteria(CriteriaList):
    """ Criteria defined on the true particle counts.
        Uses only pions in the final state to categorise events.
    """
    pi_inelastic : Criteria # is pi inelatstic interaction
    n_pi : Criteria # number of charged pi's
    n_pi0 : Criteria # numbre of pi0's

    @staticmethod
    def get_criteria_values(events : ntuple.Data, pi_KE_lim : float = 0):
        n_pi_true, n_pi0_true = get_true_pion_counts(events, pi_KE_lim)
        pi_inel = events.true_particles.true_beam_end_process == "pi+Inelastic"
        return {"pi_inelastic" : pi_inel, "n_pi" : n_pi_true, "n_pi0" : n_pi0_true}


@dataclass 
class process_criteria_exp(CriteriaList):
    """ Expanded criteria for true process definitions
    """
    pi_beam : Criteria # beam particle is pion
    beam_escapes : Criteria # beam particle escapes fiducial volume
    beam_decay : Criteria # beam particle decays
    pi_inelastic : Criteria # is pi inelatstic interaction
    n_pi : Criteria # number of charged pi's
    n_pi0 : Criteria # numbre of pi0's

    @staticmethod
    def get_criteria_values(events : ntuple.Data, pi_KE_lim : float = 0, fiducial_volume : list[float] = [0, 700]):

        cvs = process_criteria.get_criteria_values(events, pi_KE_lim)

        pi_beam = events.true_particles.pdg[:, 0] == 211
        decay = (events.true_particles.true_beam_end_process == "Decay")
        escapes = events.true_particles.beam_traj_pos.z[:, -1] >= max(fiducial_volume)

        return cvs | {"pi_beam" : pi_beam , "beam_escapes" : escapes, "beam_decay" : decay}


class geant_process_energy_limit(SampleDefinition):
    """" Interaction definitions compatible with GEANT4 simulation file, including charged pion KE limit."""
    criteria_list = process_criteria_geant

    definitions = {
        "total_inelastic" : [
            criteria_list(Criteria("==", 1), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0))
        ],
        "absorption" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 0), Criteria("==", 0))
        ],
        "charge_exchange" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 0), Criteria("==", 1))
        ],
        "quasielastic" : [
            criteria_list(Criteria("==", 1), Criteria("==", 1), Criteria("==", 0), Criteria("==", 0))
        ],
        "double_charge_exchange" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0))
        ],
        "pion_production" : [
            criteria_list(Criteria("==", 1), Criteria(">", 1), Criteria(">=", 0), Criteria(">=", 0)),
            criteria_list(Criteria("==", 1), Criteria(">=", 0), Criteria(">", 1), Criteria(">=", 0)),
            criteria_list(Criteria("==", 1), Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 1)),
            criteria_list(Criteria("==", 1), Criteria("==", 1), Criteria("==", 1), Criteria("==", 1)),
            criteria_list(Criteria("==", 1), Criteria("==", 1), Criteria("==", 1), Criteria("==", 0)),
            criteria_list(Criteria("==", 1), Criteria("==", 1), Criteria("==", 0), Criteria("==", 1)),
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 1), Criteria("==", 1))
        ],
    }


class four_signal_process(SampleDefinition):
    """ Signal defintions where pion production is split into single and 'multi' pion production."""
    criteria_list = process_criteria
    definitions = {
        "absorption" : [
            criteria_list(Criteria(">=", 0), Criteria("==", 0), Criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(Criteria(">=", 0), Criteria("==", 0), Criteria("==", 1)),
        ],
        "single_pion_production" : [
            criteria_list(Criteria(">=", 0), Criteria("==", 1), Criteria("==", 0)),
        ],
        "pion_production" : [
            criteria_list(Criteria(">=", 0), Criteria(">", 1), Criteria(">=", 0)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 1)),
            criteria_list(Criteria(">=", 0), Criteria("==", 1), Criteria("==", 1))
        ]
    }


class three_signal_process(SampleDefinition):
    """ Signal defintion more widely used in PDSP analyses"""
    criteria_list = process_criteria
    definitions = {
        "absorption" : [
            criteria_list(Criteria(">=", 0), Criteria("==", 0), Criteria("=s=", 0)),
        ],
        "charge_exchange" : [
            criteria_list(Criteria(">=", 0), Criteria("==", 0), Criteria("==", 1)),
        ],
        "pion_production" : [
            criteria_list(Criteria(">=", 0), Criteria(">=", 1), Criteria(">=", 0)),
            criteria_list(Criteria(">=", 0), Criteria("==", 0), Criteria(">", 1)),
        ]
    }

class three_signal_process_bkg(SampleDefinition):
    """ Signal defintion more widely used in PDSP analyses, accounting for impurities arising from non pi+ inelastic interactions."""
    criteria_list = process_criteria
    definitions = {
        "absorption" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 1)),
        ],
        "pion_production" : [
            criteria_list(Criteria("==", 1), Criteria(">=", 1), Criteria(">=", 0)),
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria(">", 1)),
        ],
        "impurities" : [
            criteria_list(Criteria("==", 0), Criteria(">=", 0), Criteria(">=", 0)),
        ]
    }

class three_signal_process_bkg_fd(SampleDefinition):
    """ Signal defintion more widely used in PDSP analyses, accounting for impurities arising from non pi+ inelastic interactions and also any particles which escape the fiducial volume."""
    criteria_list = process_criteria_exp
    definitions = {
        "absorption" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 1)),
        ],
        "pion_production" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 0), Criteria("==", 1), Criteria(">=", 1), Criteria(">=", 0)),
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria(">", 1)),
        ],
        "decay" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria(">=", 0), Criteria(">=", 0))
        ],
        "escaping" : [
            criteria_list(Criteria("==", 1), Criteria("==", 1), Criteria(">=", 0) ,Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0))
        ],
        "impurities" : [
            criteria_list(Criteria("==", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 0), Criteria(">=", 0), Criteria(">=", 0)),
        ],
    }
    mode = {
        "absorption" : 0,
        "charge_exchange" : 1,
        "pion_production" : 2,
        "decay" : 3,
        "escaping" : 4,
        "impurities" : 999,
    }
    short_name = {
        "absorption" : "Abs",
        "charge_exchange" : "CEx",
        "pion_production" : "Pip",
        "decay" : "Decay",
        "escaping" : "Esc",
        "impurities" : "Impure",
    }
    signal = ["absorption", "charge_exchange", "pion_production"] # The processes we actually want to measure cross sections for.


def get_true_pion_counts(events : ntuple.Data, ke_lim : float = 0):
    n_pi_true = (events.true_particles.number != 1) & (abs(events.true_particles.pdg) == 211) & (events.true_particles.mother == 1)

    ke = kinematics.KE(vector.magnitude(events.true_particles.momentum), Particle.from_pdgid(211).mass)

    n_pi_true = ak.sum(n_pi_true & (ke > ke_lim), axis = -1)
    n_pi0_true = events.true_particles.n_pi0

    return n_pi_true, n_pi0_true


def get_true_pion_counts_geant(events, KE_lim : float = 0) -> tuple[ak.Array, ak.Array]:
    KE = kinematics.KE(vector.magnitude(events.true_particles.momentum), Particle.from_pdgid(211).mass)

    n_pip_true = (events.true_particles.number != 1) & (events.true_particles.pdg == 211) & (events.true_particles.mother == 1)
    n_pim_true = (events.true_particles.number != 1) & (events.true_particles.pdg == -211) & (events.true_particles.mother == 1)

    n_pip_true = ak.sum(n_pip_true & (KE > KE_lim), axis = -1)
    n_pim_true = ak.sum(n_pim_true & (KE > KE_lim), axis = -1)
    n_pi0_true = events.true_particles.n_pi0

    return n_pip_true, n_pim_true, n_pi0_true

processes = {
    "four_signal_process" : four_signal_process,
    "three_signal_process" : three_signal_process,
    "three_signal_process_bkg" : three_signal_process_bkg,
    "three_signal_process_bkg_fd" : three_signal_process_bkg_fd,
}
