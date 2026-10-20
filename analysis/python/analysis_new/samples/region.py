"""
Created on: 20/02/2026 12:15

Author: Shyam Bhuller

Description: Reconstructed region defintions.
"""
from python.analysis.SampleDefinition import SampleDefinition, CriteriaList, criteria, dataclass
from python.analysis.SelectionTools import GetPFOCounts
from python.analysis.Master import Data


@dataclass
class region_criteria(CriteriaList):
    n_loose_pi : criteria
    n_loose_photon : criteria
    n_pi : criteria
    n_photon : criteria
    n_pi0 : criteria

    @staticmethod
    def get_criteria_values(events : Data, selection_masks: dict, **kwargs) -> dict:
        counts = {}
        for obj in selection_masks:
            if obj in ["beam", "null_pfo", "fiducial"]: continue
            counts[f"n_{obj}"] = GetPFOCounts(selection_masks[obj][events.filename])
        return counts

@dataclass
class region_criteria_esc(region_criteria):
    escapes : criteria

    @staticmethod
    def get_criteria_values(events : Data, selection_masks: dict, **kwargs) -> dict:
        values = region_criteria.get_criteria_values(events, selection_masks, **kwargs)
        values["escapes"] = events.recoParticles.beam_endPos_SCE.z >= max(kwargs["fiducial_volume"])
        return values


class high_purity_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            criteria_list(criteria("==", 0), criteria("==", 0), criteria("==", 0), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(criteria("==", 0), criteria("==", 2), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
        ],
        "single_pion_production" : [
            criteria_list(criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">", 1), criteria(">=", 0), criteria(">=", 0)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">", 0), criteria("==", 2), criteria("==", 1)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 2), criteria(">=", 0)),
            criteria_list(criteria(">=", 0), criteria(">", 2), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 1), criteria(">=", 0)),
        ],        
    }


class high_efficiency_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria("==", 0), criteria("==", 0)),
        ],
        "charge_exchange" : [
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
            criteria_list(criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0)),
        ],
        "single_pion_production" : [
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            criteria_list(criteria(">", 0), criteria(">", 0), criteria(">", 0), criteria(">", 0), criteria(">=", 0)),
            criteria_list(criteria(">", 1), criteria(">=", 0), criteria(">", 1), criteria("==", 0), criteria("==", 0)),
            criteria_list(criteria(">=", 0), criteria(">", 1), criteria("==", 0), criteria(">", 0), criteria("==", 0)),
            criteria_list(criteria(">", 0), criteria("<", 3), criteria("==", 0), criteria(">", 0), criteria("==", 0)),
        ],        
    }


class moderate_efficiency_regions(SampleDefinition):
    criteria_list = region_criteria
    definitions = {
        "absorption" : [
            criteria_list(criteria("==", 0), criteria("==", 0), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
        ],
        "charge_exchange" : [
            criteria_list(criteria("==", 0), criteria("==", 2), criteria("==", 0), criteria("==", 2), criteria("==", 1)),
            criteria_list(criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0)),
        ],
        "single_pion_production" : [
            criteria_list(criteria(">", 0), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0)),
        ],
        "pion_production" : [
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">", 1), criteria(">=", 0), criteria(">=", 0)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">", 0), criteria("==", 2), criteria("==", 1)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 2), criteria(">=", 0)),
            criteria_list(criteria(">=", 0), criteria(">", 2), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 1), criteria(">=", 0)),
        ],        
    }

class moderate_efficiency_three_regions_esc(SampleDefinition):
    criteria_list = region_criteria_esc
    definitions = {
        "absorption" : [
            criteria_list(criteria("==", 0), criteria("==", 0), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria("==",0)),
        ],
        "charge_exchange" : [
            criteria_list(criteria("==", 0), criteria("==", 2), criteria("==", 0), criteria("==", 2), criteria("==", 1), criteria("==",0)),
            criteria_list(criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==",0)),
        ],
        "pion_production" : [
            criteria_list(criteria(">", 0), criteria("==", 0), criteria("==", 1), criteria("==", 0), criteria("==", 0), criteria("==",0)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">", 1), criteria(">=", 0), criteria(">=", 0), criteria("==",0)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">", 0), criteria("==", 2), criteria("==", 1), criteria("==",0)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">", 2), criteria(">=", 0), criteria("==",0)),
            criteria_list(criteria(">=", 0), criteria(">", 2), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria("==",0)),
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria("==", 1), criteria("==", 1), criteria(">=", 0), criteria("==",0)),
        ],
        "escaping" : [
            criteria_list(criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria(">=", 0), criteria("==",1)),
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

class moderate_efficiency_three_regions_loose_pi(SampleDefinition):
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
            region_criteria(criteria(">", 0), criteria("==", 0), criteria(">=", 0), criteria("==", 0), criteria("==", 0)),
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
    "moderate_efficiency_three_regions_loose_pi" : moderate_efficiency_three_regions_loose_pi,
    "moderate_efficiency_regions" : moderate_efficiency_regions,
    "high_efficiency_regions" : high_efficiency_regions,
    "high_purity_regions" : high_purity_regions,
    "moderate_efficiency_three_regions" : moderate_efficiency_three_regions,
    "moderate_efficiency_three_regions_esc": moderate_efficiency_three_regions_esc
}