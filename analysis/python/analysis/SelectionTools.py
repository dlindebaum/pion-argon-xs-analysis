"""
Created on: 08/03/2023 13:39

Author: Shyam Bhuller

Description: Generic tools to use when making selections on Data.
"""
from functools import wraps

import operator
import awkward as ak
import numpy as np
import pandas as pd

from python.analysis.Master import Data


def CountMask(m: ak.Array) -> tuple:
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


def CombineSelections(events : Data, selection: list, levels: int, args: list = None, verbose: bool = False, return_table: bool = False) -> ak.Array:
    """ Combines multiple beam particle selections.

    Args:
        events (Data): events to study
        selection (list): list of selections
        levels (int): what kind of data is the selection being made on, 0 is event, 1 is PFO.
        args (list): Additional keyword arguments to be passed to the selection functions
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
            "no selection": [total, 100]*2
        }

    if args is None:
        args = [{}] * len(selection)

    mask = None
    for s, kwargs in zip(selection, args):
        new_mask = s(events, **kwargs)
        if not hasattr(mask, "__iter__"):
            mask = new_mask
        else:
            mask = np.logical_and(mask, new_mask)

        if return_table or verbose:
            successive_counts = ak.count(mask[mask])
            single_counts = ak.count(new_mask[new_mask])
            table[s.__name__] = [single_counts, 100 * single_counts / table["no selection"]
                                 [0], successive_counts, 100 * successive_counts / table["no selection"][0]]

    if return_table or verbose:
        table = pd.DataFrame(table, index=["number of events which pass the cut", "single efficiency",
                             "number of events after successive cuts", "successive efficiency"]).T
        relative_efficiency = np.append([np.nan], 100 * table["number of events after successive cuts"].values[1:]
                                        / table["number of events after successive cuts"].values[:-1])
        table["relative efficiency"] = relative_efficiency
    if verbose:
        print(table)

    if return_table:
        return mask, table
    else:
        return mask


def del_prop(obj, property_name: str) -> None:
    """
    Deletes a properties from the supplied `RecoPaticleData` type
    object.

    Requires the `obj` to have a property
    ``_RecoPaticleData__{property_name}``.

    Parameters
    ----------
    obj : RecoPaticleData
        Object from which to remove the property.
    property_name : str
        Property to be deleted (should match the name of the property).
    """
    del(obj.__dict__["_RecoPaticleData__" + property_name])
    return


def np_to_ak_indicies(indicies: np.ndarray) -> ak.Array:
    """
    Takes a numpy array of indicies for slicing and converts them to a
    format compatible with awkward arrays which selected PFOs in an
    event base on the index, rather than events themselves based on the
    index.

    Parameters
    ----------
    indicies: np.ndarray
        Array of indicies for conversion.

    Returns
    -------
    ak_indicies : ak.Array
        Array of indicies which selected PFOs when slicing an awkward
        array.
    """
    # 1. Expands the dimensions to ensure you hit one index per event
    # 2. Convert to list - this is necessary to ensure the final
    #    awkward array has variable size. Without variable size arrays,
    #    it tries to gather the event of the index, not the PFO at the
    #    index in the event.
    # 3. Convert to awkward array
    return ak.Array(np.expand_dims(indicies, 1).tolist())

def insert_values_to_func_str(func_str, values):
    for i in range(len(values)):
        func_str = func_str.replace(f"_{i}_", f"{values[i]}")
    return func_str

def cuts_to_func(values, *operations, func_str=None):
    ops = {
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge}

    if func_str is None:
        if len(operations) != 1:
            raise ValueError("operations must be sepcified if func_str is not specified")
        else:
            operations = operations[0]
        def cut_func(property_to_cut):
            def next_cut(index):
                curr_cut = ops[operations[index]](property_to_cut, values[index])
                if index <= 0:
                    return curr_cut
                else:
                    return np.logical_and(curr_cut, next_cut(index-1))
            return next_cut(len(values)-1)
        return cut_func
    else:
        formatted_func = insert_values_to_func_str(func_str, values)
        return lambda x: eval(formatted_func)

def cuts_to_str(values, *operations, func_str=None, name_format=False):
    str_ini = ", " if name_format else ""
    if func_str is None:
        if len(operations) != 1:
            raise ValueError("operations must be sepcified if func_str is not specified")
        else:
            operations = operations[0]
        
        if len(values) == 1:
            str_ini = "" if name_format else "x"
            return str_ini + f" {operations[0]} {values[0]}"
        elif len(values) == 2:
            if (">" in operations[0]) and ("<" in operations[1]):
                return str_ini + f"{values[0]} {operations[0].replace('>','<')} x {operations[1]} {values[1]}"
            elif (">" in operations[1]) and ("<" in operations[0]):
                return str_ini + f"{values[1]} {operations[1].replace('>','<')} x {operations[0]} {values[0]}"
        return str_ini + " and ".join([f"(x {op} {val})" for val, op in zip(values, operations)])

    else:
        return str_ini + insert_values_to_func_str(func_str, values)
