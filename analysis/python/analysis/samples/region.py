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
    """Store reconstructed counts of loose pions, loose photons, pions, photons, and neutral pions.
    """
    n_loose_pi : Criteria
    n_loose_photon : Criteria
    n_pi : Criteria
    n_photon : Criteria
    n_pi0 : Criteria

    @staticmethod
    def get_criteria_values(events : ntuple.Data, selection_masks: dict, **kwargs) -> dict:
        """Count selected PFOs in each particle-selection mask, excluding beam and preselection masks.

            Args:
                events (ntuple.Data): Ntuple used to select the masks matching its source file.
                selection_masks (dict): Named particle-selection masks, indexed by source filename.
                **kwargs (Any): Additional criteria options; unused by this criteria set.

            Returns:
                dict: Particle-count values keyed by names such as ``n_pi`` and ``n_photon``.
        """
        counts = {}
        for obj in selection_masks:
            if obj in ["beam", "null_pfo", "fiducial"]: continue
            counts[f"n_{obj}"] = selection.pfo_counts(selection_masks[obj][events.filename])
        return counts

@dataclass
class region_criteria_esc(region_criteria):
    """Extend particle-count criteria with a flag for reconstructed endpoints at or beyond the fiducial z maximum.
    """
    escapes : Criteria

    @staticmethod
    def get_criteria_values(events : ntuple.Data, selection_masks: dict, **kwargs) -> dict:
        """Add an ``escapes`` flag when the reconstructed beam endpoint reaches the fiducial z maximum.

            Args:
                events (ntuple.Data): Ntuple containing reconstructed endpoint positions.
                selection_masks (dict): Named particle-selection masks used for particle counts.
                **kwargs (Any): Must include ``fiducial_volume`` with the upper z limit.

            Returns:
                dict: Particle counts from ``region_criteria`` plus the boolean ``escapes`` mask.
        """
        values = region_criteria.get_criteria_values(events, selection_masks, **kwargs)
        values["escapes"] = events.recoParticles.beam_end_pos_sce.z >= max(kwargs["fiducial_volume"])
        return values


class high_purity_regions(SampleDefinition):
    """Define exclusive reconstructed regions with strict particle-count criteria for purity.
    """
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
    """Define reconstructed interaction regions with relaxed count criteria to increase efficiency.
    """
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
    """Define moderate-efficiency absorption, charge-exchange, single-pion, and pion-production regions.
    """
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
    """Define moderate-efficiency absorption, charge-exchange, pion-production, and escape regions.
    """
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
    """Define three interaction regions using moderate-efficiency particle-count criteria.
    """
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
    """Define three moderate-efficiency regions with a relaxed pion-count requirement for pion production.
    """
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
    """Define absorption, charge-exchange, and other regions for the PDSP 1 GeV selection.
    """
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
    """Define the default absorption, charge-exchange, single-pion, and pion-production regions.
    """
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
