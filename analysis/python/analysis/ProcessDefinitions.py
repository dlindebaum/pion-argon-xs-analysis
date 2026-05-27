"""
Created on: 20/02/2026 12:12

Author: Shyam Bhuller

Description: True process defintions.
"""
from python.analysis.SampleDefinition import SampleDefinition, CriteriaList, criteria, dataclass

@dataclass
class process_criteria(CriteriaList):
    """ Criteria defined on the true particle counts.
        Uses only pions in the final state to categorise events. 
    """
    pi_inelastic : criteria # is pi inelatstic interaction
    n_pi : criteria # number of charged pi's
    n_pi0 : criteria # numbre of pi0's


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
            process_criteria(criteria(">=", 0), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            process_criteria(criteria(">=", 0), criteria("==", 0), criteria("==", 1)),
        ],
        "pion_production" : [
            process_criteria(criteria(">=", 0), criteria(">=", 1), criteria(">=", 0)),
            process_criteria(criteria(">=", 0), criteria("==", 0), criteria(">", 1)),
        ]
    }

class three_signal_process_bkg(SampleDefinition):
    """ Signal defintion more widely used in PDSP analyses"""
    criteria_list = process_criteria
    definitions = {
        "absorption" : [
            process_criteria(criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            process_criteria(criteria("==", 1), criteria("==", 0), criteria("==", 1)),
        ],
        "pion_production" : [
            process_criteria(criteria("==", 1), criteria(">=", 1), criteria(">=", 0)),
            process_criteria(criteria("==", 1), criteria("==", 0), criteria(">", 1)),
        ]
    }

processes = {
    "four_signal_process" : four_signal_process,
    "three_signal_process" : three_signal_process,
    "three_signal_process_bkg" : three_signal_process_bkg,
}