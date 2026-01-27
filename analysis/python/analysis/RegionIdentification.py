"""
Created on: 29/03/2024 23:09

Author: Shyam Bhuller

Description: Code to create regions from PFO selections
"""
import operator
from collections import namedtuple
from dataclasses import dataclass

import awkward as ak
import numpy as np

from python.analysis import SelectionTools

criteria = namedtuple("criteria", ["op", "value"])

@dataclass
class process_criteria:
    loose_pi : criteria
    loose_photon : criteria
    pi : criteria
    photon : criteria
    pi0 : criteria


def CreateRegionIdentification(region_defs : dict[list], n_loose_pi : ak.Array, n_loose_photon : ak.Array, n_pi : ak.Array, n_photon : ak.Array, n_pi0 : ak.Array, removed : bool = False):
    counts = {
        "loose_pi" : n_loose_pi,
        "loose_photon" : n_loose_photon,
        "pi" : n_pi,
        "photon" : n_photon,
        "pi0" : n_pi0,
    }
    ops = {
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge
    }

    regions = {}
    for n, p_c in region_defs.items():
        defs = [ak.all([ops[vars(c)[i].op](counts[i], vars(c)[i].value) for i in counts], 0) for c in p_c]
        regions[n] = ak.any(defs, 0)

    if removed:
        regions["uncategorised"] = ~SelectionTools.CombineMasks(regions, "or")

    return regions

regions = {
    "high_purity" : {
        "absorption" : [
            process_criteria(criteria("==", 0), criteria("==", 0), criteria("==", 0), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            process_criteria(criteria("==", 0), criteria("==", 2), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
        ],
        "single_pion_production" : [
            process_criteria(criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 1), criteria(">=", 0), criteria(">=", 0)),
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 0), criteria("==", 2), criteria("==", 1)),
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 2), criteria(">=", 0)),
            process_criteria(criteria(">=", 0), criteria(">", 2), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 1), criteria(">=", 0)),
        ],
    },
    "high_efficiency" : {
        "absorption" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
            process_criteria(criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0)),
        ],
        "single_pion_production" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            process_criteria(criteria(">", 0), criteria(">", 0), criteria(">", 0), criteria(">", 0), criteria(">=", 0)),
            process_criteria(criteria(">", 1), criteria(">=", 0), criteria(">", 1), criteria("==", 0), criteria("==", 0)),
            process_criteria(criteria(">=", 0), criteria(">", 1), criteria("==", 0), criteria(">", 0), criteria("==", 0)),
            process_criteria(criteria(">", 0), criteria("<", 3), criteria("==", 0), criteria(">", 0), criteria("==", 0)),
        ],
    },
    "moderate_efficiency" : {
        "absorption" : [
            process_criteria(criteria("==", 0), criteria("==", 0), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
        ],
        "charge_exchange" : [
            process_criteria(criteria("==", 0), criteria("==", 2), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
        ], 
        "pion_production" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 1), criteria(">=", 0), criteria(">=", 0)),
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 0), criteria(">=", 2), criteria(">=", 1)),
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 2), criteria(">=", 0)),
            process_criteria(criteria(">=", 0), criteria(">", 2), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
        ],
    },
    "jakes" : {
        "absorption" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria("==", 0), criteria(">=", 0)),
        ],
        "charge_exchange" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria(">=", 1), criteria(">=", 0)),
        ],
        "other" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 1), criteria(">=", 0), criteria(">=", 0)),
        ],
    },
    "default" : {
        "absorption" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria(">=", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria(">=", 0), criteria("==", 1)),
        ],
        "single_pion_production" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria(">=", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 1), criteria(">=", 0), criteria(">=", 0)),
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 1)),
            process_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria(">=", 0), criteria("==", 1)),
        ],
    }
}

def TrueRegions(pi0_counts, pi_charged_counts):
    regions_dict = {
        "absorption": np.logical_and(pi0_counts==0, pi_charged_counts==0),
        "charge_exchange": np.logical_and(pi0_counts==1, pi_charged_counts==0),
        "single_pion_production": np.logical_and(pi0_counts==0, pi_charged_counts==1),
        "pion_production": ((pi0_counts >= 0) & (pi_charged_counts > 1)) | ((pi0_counts > 1) & (pi_charged_counts >= 0)) | ((pi0_counts == 1) & (pi_charged_counts == 1)),
    }
    return regions_dict
