"""
Created on: 20/02/2026 10:40

Author: Shyam Bhuller

Description: Module to define categories for samples based on particle counts. 
"""
import operator
from abc import ABC
from collections import namedtuple
from dataclasses import dataclass

import awkward as ak
import numpy as np

from python.analysis import SelectionTools

criteria = namedtuple("criteria", ["op", "value"]) # criteria is defined using an operator and value (particle count)

@dataclass
class CriteriaList(ABC):
    """ A list of Criteria """
    pass

class SampleDefinition:
    """ Class that is used to define Samples based on provided criteria.
        Members:
        criteria_list (CritieriaList): The criteria used to define an events in the sample.
        definitions (dict[list[CriteriaList]]): The categories that the samples are idenified as. This should be based on the CriteriaList. 
    """
    criteria_list : CriteriaList
    definitions : dict[list[CriteriaList]]

    def CreateDefinitions(self, particle_counts : dict[ak.Array], uncategorised = False) -> dict[ak.Array]:
        """ Based on the definitions and criteria list, generate masks to categorise events when provided the particle counts.

        Args:
            particle_counts (dict[ak.Array]): particle counts. keys should be the different particle types, and must match what is used by criteria_list.
            uncategorised (bool, optional): Create a new category for particles that do not fall into the pre-defined categories . Defaults to False.

        Returns:
            dict[ak.Array]: Dictionary of masks for each category.
        """
        particle_list = list(self.criteria_list.__dataclass_fields__.keys())
        for c in particle_list:
            if c not in particle_counts:
                raise Exception(f"particle_counts does not contain the required particles for the criteria: {particle_list}")
        for c in particle_counts:
            if c not in particle_counts:
                raise Exception(f"particle_counts contains extra particle types not defined in the criteria: {particle_list}")
        ops = {
            "==": operator.eq,
            "!=": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge
        }

        masks = {}
        for n, p_c in self.definitions.items():
            defs = [ak.all([ops[vars(c)[i].op](particle_counts[i], vars(c)[i].value) for i in particle_counts], 0) for c in p_c]
            masks[n] = ak.any(defs, 0)

        if uncategorised:
            masks["uncategorised"] = ~SelectionTools.CombineMasks(masks, "or")

        return masks
