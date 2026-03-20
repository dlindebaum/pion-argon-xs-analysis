"""
Created on: 20/02/2026 12:15

Author: Shyam Bhuller

Description: Reconstructed region defintions.
"""
from python.analysis.SampleDefinition import SampleDefinition, CriteriaList, criteria, dataclass

@dataclass
class region_criteria(CriteriaList):
    n_loose_pi : criteria
    n_loose_photon : criteria
    n_pi : criteria
    n_photon : criteria
    n_pi0 : criteria


class high_purity_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(criteria("==", 0), criteria("==", 0), criteria("==", 0), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            region_criteria(criteria("==", 0), criteria("==", 2), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
        ],
        "single_pion_production" : [
            region_criteria(criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 1), criteria(">=", 0), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 0), criteria("==", 2), criteria("==", 1)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 2), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">", 2), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 1), criteria(">=", 0)),
        ],        
    }


class high_efficiency_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
            region_criteria(criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0)),
        ],
        "single_pion_production" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            region_criteria(criteria(">", 0), criteria(">", 0), criteria(">", 0), criteria(">", 0), criteria(">=", 0)),
            region_criteria(criteria(">", 1), criteria(">=", 0), criteria(">", 1), criteria("==", 0), criteria("==", 0)),
            region_criteria(criteria(">=", 0), criteria(">", 1), criteria("==", 0), criteria(">", 0), criteria("==", 0)),
            region_criteria(criteria(">", 0), criteria("<", 3), criteria("==", 0), criteria(">", 0), criteria("==", 0)),
        ],        
    }


class moderate_efficiency_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(criteria("==", 0), criteria("==", 0), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
        ],
        "charge_exchange" : [
            region_criteria(criteria("==", 0), criteria("==", 2), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
            region_criteria(criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0)),
        ],
        "single_pion_production" : [
            region_criteria(criteria(">", 0), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 1), criteria(">=", 0), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 0), criteria("==", 2), criteria("==", 1)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 2), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">", 2), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 1), criteria(">=", 0)),
        ],        
    }

class moderate_efficiency_three_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(criteria("==", 0), criteria("==", 0), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
        ],
        "charge_exchange" : [
            region_criteria(criteria("==", 0), criteria("==", 2), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
            region_criteria(criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0)),
        ],
        "pion_production" : [
            region_criteria(criteria(">", 0), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 1), criteria(">=", 0), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 0), criteria("==", 2), criteria("==", 1)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 2), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">", 2), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 1), criteria(">=", 0)),
        ],        
    }

class pdsp_1GeV_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria("==", 0), criteria(">=", 0)),
        ],
        "charge_exchange" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria(">=", 1), criteria(">=", 0)),
        ],
        "other" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 1), criteria(">=", 0), criteria(">=", 0)),
        ],        
    }


class default(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria(">=", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria(">=", 0), criteria("==", 1)),
        ],
        "single_pion_production" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria(">=", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">", 1), criteria(">=", 0), criteria(">=", 0)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 1)),
            region_criteria(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria(">=", 0), criteria("==", 1)),
        ],        
    }

regions = {
    "default" : default,
    "pdsp_1GeV_regions" : pdsp_1GeV_regions,
    "moderate_efficiency_three_regions" : moderate_efficiency_three_regions,
    "moderate_efficiency_regions" : moderate_efficiency_regions,
    "high_efficiency_regions" : high_efficiency_regions,
    "high_purity_regions" : high_purity_regions
}