"""
Created on: 20/02/2026 12:15

Author: Shyam Bhuller

Description: Reconstructed region defintions.
"""
from dataclasses import dataclass

from .common import *

from python.analysis import ntuple, selection

@dataclass
class region_criteria(CriteriaList):
    n_loose_pi : Criteria
    n_loose_photon : Criteria
    n_pi : Criteria
    n_photon : Criteria
    n_pi0 : Criteria

    @staticmethod
    def get_criteria_values(events : ntuple.Data, selection_masks: dict, **kwargs) -> dict:
        counts = {}
        for obj in selection_masks:
            if obj in ["beam", "null_pfo", "fiducial"]: continue
            counts[f"n_{obj}"] = selection.pfo_counts(selection_masks[obj][events.filename])
        return counts

@dataclass
class region_criteria_esc(region_criteria):
    escapes : Criteria

    @staticmethod
    def get_criteria_values(events : ntuple.Data, selection_masks: dict, **kwargs) -> dict:
        values = region_criteria.get_criteria_values(events, selection_masks, **kwargs)
        values["escapes"] = events.recoParticles.beam_end_pos_sce.z >= max(kwargs["fiducial_volume"])
        return values


class high_purity_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            criteria_list(Criteria("==", 0), Criteria("==", 0), Criteria("==", 0), Criteria("==", 0), Criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(Criteria("==", 0), Criteria("==", 2), Criteria("==", 0), Criteria("==", 2), Criteria("==", 1)),
        ],
        "single_pion_production" : [
            criteria_list(Criteria("==", 1), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 0)),
        ],
        "pion_production" : [
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 1), Criteria(">=", 0), Criteria(">=", 0)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 0), Criteria("==", 2), Criteria("==", 1)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 2), Criteria(">=", 0)),
            criteria_list(Criteria(">=", 0), Criteria(">", 2), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 1), Criteria("==", 1), Criteria(">=", 0)),
        ],        
    }


class high_efficiency_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 0), Criteria("==", 0), Criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 0), Criteria("==", 2), Criteria("==", 1)),
            criteria_list(Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0)),
        ],
        "single_pion_production" : [
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 0)),
        ],
        "pion_production" : [
            criteria_list(Criteria(">", 0), Criteria(">", 0), Criteria(">", 0), Criteria(">", 0), Criteria(">=", 0)),
            criteria_list(Criteria(">", 1), Criteria(">=", 0), Criteria(">", 1), Criteria("==", 0), Criteria("==", 0)),
            criteria_list(Criteria(">=", 0), Criteria(">", 1), Criteria("==", 0), Criteria(">", 0), Criteria("==", 0)),
            criteria_list(Criteria(">", 0), Criteria("<", 3), Criteria("==", 0), Criteria(">", 0), Criteria("==", 0)),
        ],        
    }


class moderate_efficiency_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            criteria_list(Criteria("==", 0), Criteria("==", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0)),
        ],
        "charge_exchange" : [
            criteria_list(Criteria("==", 0), Criteria("==", 2), Criteria("==", 0), Criteria("==", 2), Criteria("==", 1)),
            criteria_list(Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0)),
        ],
        "single_pion_production" : [
            criteria_list(Criteria(">", 0), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 0)),
        ],
        "pion_production" : [
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 1), Criteria(">=", 0), Criteria(">=", 0)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 0), Criteria("==", 2), Criteria("==", 1)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 2), Criteria(">=", 0)),
            criteria_list(Criteria(">=", 0), Criteria(">", 2), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 1), Criteria("==", 1), Criteria(">=", 0)),
        ],        
    }

class moderate_efficiency_three_regions_esc(SampleDefinition):
    criteria_list = region_criteria_esc
    definitions = {
        "absorption" : [
            criteria_list(Criteria("==", 0), Criteria("==", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria("==",0)),
        ],
        "charge_exchange" : [
            criteria_list(Criteria("==", 0), Criteria("==", 2), Criteria("==", 0), Criteria("==", 2), Criteria("==", 1), Criteria("==",0)),
            criteria_list(Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==",0)),
        ],
        "pion_production" : [
            criteria_list(Criteria(">", 0), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 0), Criteria("==",0)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 1), Criteria(">=", 0), Criteria(">=", 0), Criteria("==",0)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 0), Criteria("==", 2), Criteria("==", 1), Criteria("==",0)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 2), Criteria(">=", 0), Criteria("==",0)),
            criteria_list(Criteria(">=", 0), Criteria(">", 2), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria("==",0)),
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 1), Criteria("==", 1), Criteria(">=", 0), Criteria("==",0)),
        ],
        "escaping" : [
            criteria_list(Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria("==",1)),
        ],
    }

class moderate_efficiency_three_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(Criteria("==", 0), Criteria("==", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0)),
        ],
        "charge_exchange" : [
            region_criteria(Criteria("==", 0), Criteria("==", 2), Criteria("==", 0), Criteria("==", 2), Criteria("==", 1)),
            region_criteria(Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0)),
        ],
        "pion_production" : [
            region_criteria(Criteria(">", 0), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 0)),
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 1), Criteria(">=", 0), Criteria(">=", 0)),
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 0), Criteria("==", 2), Criteria("==", 1)),
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 2), Criteria(">=", 0)),
            region_criteria(Criteria(">=", 0), Criteria(">", 2), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0)),
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 1), Criteria("==", 1), Criteria(">=", 0)),
        ],        
    }

class moderate_efficiency_three_regions_loose_pi(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(Criteria("==", 0), Criteria("==", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0)),
        ],
        "charge_exchange" : [
            region_criteria(Criteria("==", 0), Criteria("==", 2), Criteria("==", 0), Criteria("==", 2), Criteria("==", 1)),
            region_criteria(Criteria("==", 0), Criteria("==", 1), Criteria("==", 0), Criteria("==", 1), Criteria("==", 0)),
        ],
        "pion_production" : [
            region_criteria(Criteria(">", 0), Criteria("==", 0), Criteria(">=", 0), Criteria("==", 0), Criteria("==", 0)),
        ],        
    }

class pdsp_1GeV_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 0), Criteria("==", 0), Criteria(">=", 0)),
        ],
        "charge_exchange" : [
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 0), Criteria(">=", 1), Criteria(">=", 0)),
        ],
        "other" : [
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 1), Criteria(">=", 0), Criteria(">=", 0)),
        ],        
    }


class default(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 0), Criteria(">=", 0), Criteria("==", 0)),
        ],
        "charge_exchange" : [
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 0), Criteria(">=", 0), Criteria("==", 1)),
        ],
        "single_pion_production" : [
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 1), Criteria(">=", 0), Criteria("==", 0)),
        ],
        "pion_production" : [
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 1), Criteria(">=", 0), Criteria(">=", 0)),
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">=", 0), Criteria(">", 1)),
            region_criteria(Criteria(">=", 0), Criteria(">=", 0), Criteria("==", 1), Criteria(">=", 0), Criteria("==", 1)),
        ],        
    }

regions = {
    "default" : default,
    "pdsp_1GeV_regions" : pdsp_1GeV_regions,
    "moderate_efficiency_three_regions_loose_pi" : moderate_efficiency_three_regions_loose_pi,
    "moderate_efficiency_regions" : moderate_efficiency_regions,
    "high_efficiency_regions" : high_efficiency_regions,
    "high_purity_regions" : high_purity_regions,
    "moderate_efficiency_three_regions" : moderate_efficiency_three_regions,
    "moderate_efficiency_three_regions_esc": moderate_efficiency_three_regions_esc
}
