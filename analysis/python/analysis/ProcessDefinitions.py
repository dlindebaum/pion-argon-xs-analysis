"""
Created on: 20/02/2026 12:12

Author: Shyam Bhuller

Description: True process defintions.
"""
from python.analysis.SampleDefinition import SampleDefinition, CriteriaList, criteria, dataclass
from python.analysis import cross_section

import awkward as ak

@dataclass
class process_criteria(CriteriaList):
    """ Criteria defined on the true particle counts.
        Uses only pions in the final state to categorise events.
    """
    pi_inelastic : criteria # is pi inelatstic interaction
    n_pi : criteria # number of charged pi'ss
    n_pi0 : criteria # numbre of pi0's

    @staticmethod
    def get_criteria_values(events : cross_section.Data, pi_KE_lim : float = 0):
        n_pi_true, n_pi0_true = GetTruePionCounts(events, pi_KE_lim)
        pi_inel = events.trueParticles.true_beam_endProcess == "pi+Inelastic"
        return {"pi_inelastic" : pi_inel, "n_pi" : n_pi_true, "n_pi0" : n_pi0_true}


@dataclass 
class process_criteria_exp(CriteriaList):
    """ Expanded criteria for true process definitions
    """
    pi_beam : criteria # beam particle is pion
    beam_escapes : criteria # beam particle escapes fiducial volume
    beam_decay : criteria # beam particle decays
    pi_inelastic : criteria # is pi inelatstic interaction
    n_pi : criteria # number of charged pi's
    n_pi0 : criteria # numbre of pi0's

    @staticmethod
    def get_criteria_values(events : cross_section.Data, pi_KE_lim : float = 0, fiducial_volume : list[float] = [0, 700]):

        cvs = process_criteria.get_criteria_values(events, pi_KE_lim)

        pi_beam = events.trueParticles.pdg[:, 0] == 211
        decay = (events.trueParticles.true_beam_endProcess == "Decay")
        escapes = events.trueParticles.beam_traj_pos.z[:, -1] >= max(fiducial_volume)

        return cvs | {"pi_beam" : pi_beam , "beam_escapes" : escapes, "beam_decay" : decay}


class four_signal_process(SampleDefinition):
    """ Signal defintions where pion production is split into single and 'multi' pion production."""
    criteria_list = process_criteria
    definitions = {
        "absorption" : [
            criteria_list(criteria(">=", 0), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(criteria(">=", 0), criteria("==", 0), criteria("==", 1)),
        ],
        "single_pion_production" : [
            criteria_list(criteria(">=", 0), criteria("==", 1), criteria("==", 0)),
        ],
        "pion_production" : [
            criteria_list(criteria(">=", 0), criteria(">", 1), criteria(">=", 0)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">", 1)),
            criteria_list(criteria(">=", 0), criteria("==", 1), criteria("==", 1))
        ]
    }


class three_signal_process(SampleDefinition):
    """ Signal defintion more widely used in PDSP analyses"""
    criteria_list = process_criteria
    definitions = {
        "absorption" : [
            criteria_list(criteria(">=", 0), criteria("==", 0), criteria("=s=", 0)),
        ],
        "charge_exchange" : [
            criteria_list(criteria(">=", 0), criteria("==", 0), criteria("==", 1)),
        ],
        "pion_production" : [
            criteria_list(criteria(">=", 0), criteria(">=", 1), criteria(">=", 0)),
            criteria_list(criteria(">=", 0), criteria("==", 0), criteria(">", 1)),
        ]
    }

class three_signal_process_bkg(SampleDefinition):
    """ Signal defintion more widely used in PDSP analyses, accounting for impurities arising from non pi+ inelastic interactions."""
    criteria_list = process_criteria
    definitions = {
        "absorption" : [
            criteria_list(criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(criteria("==", 1), criteria("==", 0), criteria("==", 1)),
        ],
        "pion_production" : [
            criteria_list(criteria("==", 1), criteria(">=", 1), criteria(">=", 0)),
            criteria_list(criteria("==", 1), criteria("==", 0), criteria(">", 1)),
        ],
        "impurities" : [
            criteria_list(criteria("==", 0), criteria(">=", 0), criteria(">=", 0)),
        ]
    }

class three_signal_process_bkg_fd(SampleDefinition):
    """ Signal defintion more widely used in PDSP analyses, accounting for impurities arising from non pi+ inelastic interactions and also any particles which escape the fiducial volume."""
    criteria_list = process_criteria_exp
    definitions = {
        "absorption" : [
            process_criteria_exp(criteria("==", 1), criteria("==", 0), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(criteria("==", 1), criteria("==", 0), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 1)),
        ],
        "pion_production" : [
            criteria_list(criteria("==", 1), criteria("==", 0), criteria("==", 0), criteria("==", 1), criteria(">=", 1), criteria(">=", 0)),
            criteria_list(criteria("==", 1), criteria("==", 0), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria(">", 1)),
        ],
        "impurities" : [
            criteria_list(criteria("==", 0), criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria(">=", 0), criteria(">=", 0)),
        ],
        "decay" : [
            process_criteria_exp(criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria(">=", 0), criteria(">=", 0))
        ],
        "escaping" : [
            process_criteria_exp(criteria("==", 1), criteria("==", 1), criteria(">=", 0) ,criteria(">=", 0), criteria(">=", 0), criteria(">=", 0))
        ]

    }


def GetTruePionCounts(events : cross_section.Data, ke_lim : float = 0):
    n_pi_true = (events.trueParticles.number != 1) & (abs(events.trueParticles.pdg) == 211) & (events.trueParticles.mother == 1)

    ke = cross_section.KE(cross_section.vector.magnitude(events.trueParticles.momentum), cross_section.Particle.from_pdgid(211).mass)

    n_pi_true = ak.sum(n_pi_true & (ke > ke_lim), axis = -1)
    n_pi0_true = events.trueParticles.nPi0

    return n_pi_true, n_pi0_true



processes = {
    "four_signal_process" : four_signal_process,
    "three_signal_process" : three_signal_process,
    "three_signal_process_bkg" : three_signal_process_bkg,
    "three_signal_process_bkg_fd" : three_signal_process_bkg_fd,
}