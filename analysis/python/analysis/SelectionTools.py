"""
Created on: 08/03/2023 13:39

Author: Shyam Bhuller

Description: Generic tools to use when making selections on Data.
"""
from functools import wraps

import awkward as ak
import numpy as np
import pandas as pd

from python.analysis.Master import Data


def CountMask(m : ak.Array) -> tuple:
    """ Counts the total number of entries in a boolean mask,
        and the number which are True.

    Args:
        m (ak.Array): boolean mask.

    Returns:
        tuple: (number of entries, number of entries which are true).
    """
    return ak.count(m), ak.count(m[m])


def CountsWrapper(f):
    """ Wrapper for selection functions which checks the number of entries that pass a cut.

    Args:
        f (function): selection function.
    Returns:
        any: output of f.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        m = f(*args, **kwargs)
        c = CountMask(m)
        print(f"number of entries before|after {f.__name__}: {c[0]}|{c[1]}")
        return m
    return wrapper


def CombineSelections(events : Data, selection : list, levels : int, verbose : bool = False, return_table : bool = False) -> ak.Array:
    """ Combines multiple beam particle selections.

    Args:
        events (Data): events to study
        selection (list): list of selections
        levels (int): what kind of data is the selection being made on, 0 is event, 1 is PFO.
        verbose (bool, optional): verbose output. Defaults to False.
        return_table (bool, optional): return performance table. Defaults to False.

    Returns:
        ak.Array: boolean mask
        ak.Array, pd.Dataframe: boolean mask and performance table
    """
    if levels == 0:
        total = ak.count(events.eventNum)
    else:
        total = ak.count(events.recoParticles.number)

    if verbose or return_table:
        table = {
            "no selection" : [total, 100]*2
        }

    mask = None
    for s in selection:
        new_mask = s(events)
        if not hasattr(mask, "__iter__"):
            mask = new_mask
        else:
            mask = np.logical_and(mask, new_mask)

        if return_table or verbose:
            successive_counts = ak.count(mask[mask])
            single_counts = ak.count(new_mask[new_mask])
            table[s.__name__] = [single_counts, 100 * single_counts/ table["no selection"][0], successive_counts, 100 * successive_counts / table["no selection"][0]]

    if return_table or verbose:
        table = pd.DataFrame(table, index = ["number of events which pass the cut", "single efficiency", "number of events after successive cuts", "successive efficiency"]).T
        relative_efficiency = np.append([np.nan], 100 * table["number of events after successive cuts"].values[1:] / table["number of events after successive cuts"].values[:-1])
        table["relative efficiency"] = relative_efficiency
    if verbose:
        print(table)

    if return_table:
        return mask, table
    else:
        return mask
