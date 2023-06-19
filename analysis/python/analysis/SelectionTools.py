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


def _insert_values_to_func_str(func_str, values):
    for i in range(len(values)):
        func_str = func_str.replace(f"_{i}_", f"{values[i]}")
    return func_str


def _count_and_insert_params_to_func_str(func_str):
    split_func = func_str.split("_")
    params = [ord(p) for i, p in enumerate(split_func) if (len(p) == 1) and (p not in split_func[:i]) and p.islower()]
    params.sort()
    params = [chr(p) for p in params]
    for i,p in enumerate(params):
        func_str = func_str.replace(f"_{p}_", f"x[{i}]")
    return func_str


def cuts_to_func(values, *operations, func_str=None):
    """
    Generates a cut function to generate a mask when called on some
    property.

    A cut function is defined through some set of `values` and
    corresponding `operations`, or a `func_str` to define a more
    complex function.
    
    For example, to test whether my property `x` satisfies
    `3 <= x < 8` (equivalent to  `x >= 3 and x < 8`), we run the
    following:
    ```
    cut_func = cuts_to_func([3, 8], [">=", "<"])
    cut_func(x)
    ```
    At index 0 of values and operations, we see `3` and `">="`, so we
    require `x >= 3`. At index 1, we see `8` and `"<"`, so we also
    require `x < 8`.
    
    Parameters
    ----------
    values : list
        List of cut values to be used.
    *operations : list
        List of strings containing the operations which should be
        applied with the value at the equivalent index in `values`.
        Must have the same length as `values`. Allowed strings are: 
        {"==", "!=", "<", "<=", ">", ">="}.
    func_str : str, optional
        Custom cut function in python as a string. The property to be
        cut must be name "x". Values in `values` are referenced by
        calling the corresponding index of the value as "_{index}_".
        E.g. to require `x == values[0]`, the string should be
        `"x == _0_"`. Default is None.
    
    Returns
    -------
    function
        Function which returns a boolean mask when applied to a
        property.
    """
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
        if not isinstance(values, list):
            values = [values]
        if not isinstance(operations, list):
            operations = [operations]
        if len(operations) != len(values):
            raise ValueError("values and operations must have equivalent lengths")
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
        formatted_func = _insert_values_to_func_str(func_str, values)
        return lambda x: eval(formatted_func)


def multi_cuts_to_func(values, *operations, func_str=None):
    """
    Generates a cut function to generate a mask when called on some
    properties.

    A cut function is defined through some set of `values` and
    corresponding `operations`, or a `func_str` to define a more
    complex function. The generated function must be passed a list of
    properties to be cut upon, with the same length as `values`.
    
    For example, to test whether my properties `x` and `y` satisfy
    `x == 0 and y > 1`, we run the following:
    ```
    cut_func = cuts_to_func([0, 1], ["==", ">"])
    cut_func([x, y])
    ```
    At index 0 of values and operations, we see `x` and `"=="`, and we
    pass `x` at index 0, so we require `x == 0`. At index 1, we see `1`
    and `">"`, and pass `y` so we also require `y > 1`.
    
    Parameters
    ----------
    values : list
        List of cut values to be used.
    *operations : list
        List of strings containing the operations which should be
        applied with the value at the equivalent index in `values`.
        Must have the same length as `values`. Allowed strings are: 
        {"==", "!=", "<", "<=", ">", ">="}.
    func_str : str, optional
        Custom cut function in python as a string. Values in `values`
        are referenced by calling the corresponding index of the value
        as "_{index}_". Properties to be cut are label by some 
        owercase letters surrounded by underscores, i.e. "_a_".
        Multiple properties are passed as a list to the resulting
        function. Values are taken in alphabetical order (so `_a_`
        points to `x[0]` and `_b_` to `x[2]` if `x` is the input list).
        E.g. to require `x[1] < values[1] and x[0] == values[0]`, the
        string would be: `"_b_ < _1_ and _a_ == _0_"`. Default is None.
    
    Returns
    -------
    function
        Function which returns a boolean mask when applied to a
        property.
    """
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
        if not isinstance(values, list):
            values = [values]
        if not isinstance(operations, list):
            operations = [operations]
        if len(operations) != len(values):
            raise ValueError("values and operations must have equivalent lengths")
        def cut_func(properties_to_cut):
            def next_cut(index):
                curr_cut = ops[operations[index]](properties_to_cut, values[index])
                if index <= 0:
                    return curr_cut
                else:
                    return np.logical_and(curr_cut, next_cut(index-1))
            return next_cut(len(values)-1)
        return cut_func
    else:
        formatted_func = _insert_values_to_func_str(func_str, values)
        formatted_func = _count_and_insert_params_to_func_str(formatted_func)
        return lambda x: eval(formatted_func)


def cuts_to_str(values, *operations, func_str=None, name_format=False):
    """
    Generates a string respresentation of some cut.

    A cut function is defined through some set of `values` and
    corresponding `operations`, or a `func_str` to define a more
    complex function.
    
    For example, to show we have the cut `"3 <= x < 8"` (equivalent to
    `"(x >= 3) and (x < 8)"`), we run the following:
    ```
    cuts_to_str([3, 8], [">=", "<"])
    ```
    Which returns:
    ```
    "3 <= x < 8"
    ```
    At index 0 of values and operations, we see `3` and `">="`, so we
    require `x >= 3`. At index 1, we see `8` and `"<"`, so we also
    require `x < 8`.
    
    Parameters
    ----------
    values : list
        List of cut values to be used.
    *operations : list
        List of strings containing the operations which should be
        applied with the value at the equivalent index in `values`.
        Must have the same length as `values`. Allowed strings are: 
        {"==", "!=", "<", "<=", ">", ">="}.
    func_str : str, optional
        Custom cut function in python as a string. Values in `values`
        are referenced by calling the corresponding index of the value
        as "_{index}_". E.g. to require `x == values[0]`, the string
        should be `"x == _0_"`. Default is None.
    name_format : bool, optional
        If True, the string will include a ", " before the cut display,
        unless only one cut is present. If only 1 cut is present, the
        leading "x" is skipped, i.e. "x > 5" -> " > 5". This purpose is
        to auotmatically format the result if used in the following
        way: `"property x" + cuts_to_str(values, operations)`. Default
        is False.
    
    Returns
    -------
    str
        A string repesntation of the passed function.
    """
    str_ini = ", " if name_format else ""
    if func_str is None:
        if len(operations) != 1:
            raise ValueError("operations must be sepcified if func_str is not specified")
        else:
            operations = operations[0]
        if not isinstance(values, list):
            values = [values]
        if not isinstance(operations, list):
            operations = [operations]
        if len(operations) != len(values):
            raise ValueError("values and operations must have equivalent lengths")
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
        return str_ini + _insert_values_to_func_str(func_str, values)


def cuts_to_str_multi(param_names, values, *operations, func_str=None):
    """
    Generates a string respresentation of some cut.

    A cut function is defined through some set of `values` and
    corresponding `operations`, or a `func_str` to define a more
    complex function. A set of `param_names` define the names given to
    the properties.
    

    For example, to show we have the cut on properties `x` and `y`
    satisfing `x == 0 and y > 1`, we run the following:
    ```
    cuts_to_str_multi(['x', 'y'], [0, 1], ["==", ">"])
    ```
    Which returns:
    ```
    "(x == 0) and (y > 1)"
    ```

    At index 0 of values and operations, we see `x` and `"=="`, and we
    pass `x` at index 0, so we require `x == 0`. At index 1, we see `1`
    and `">"`, and pass `y` so we also require `y > 1`.
    
    Parameters
    ----------
    param_names : list
        List of property names.
    values : list
        List of cut values to be used.
    *operations : list
        List of strings containing the operations which should be
        applied with the value at the equivalent index in `values`.
        Must have the same length as `values`. Allowed strings are: 
        {"==", "!=", "<", "<=", ">", ">="}.
    func_str : str, optional
        Custom cut function in python as a string. Values in `values`
        are referenced by calling the corresponding index of the value
        as "_{index}_". Properties to be cut are label by some 
        owercase letters surrounded by underscores, i.e. "_a_".
        Multiple properties are passed as a list to the resulting
        function. Values are taken in alphabetical order (so `_a_`
        points to `param_names[0]` and `_b_` to `param_names[2]`). E.g.
        to require
        `(param_names[1] < values[1]) and (param_names[0] == values[0])`
        , the string would be: `"_b_ < _1_ and _a_ == _0_"`.
        Alternatively, write the paramter names directly into the
        string. Default is None.
    
    Returns
    -------
    str
        A string repesntation of the passed function.
    """
    if func_str is None:
        if len(operations) != 1:
            raise ValueError("operations must be sepcified if func_str is not specified")
        else:
            operations = operations[0]
        if not isinstance(values, list):
            values = [values]
        if not isinstance(operations, list):
            operations = [operations]
        if len(operations) != len(values):
            raise ValueError("values and operations must have equivalent lengths")
        return " and ".join([f"({name} {op} {val})" for name, val, op in zip(param_names, values, operations)])

    else:
        formatted_func =  _insert_values_to_func_str(func_str, values)
        formatted_func = _count_and_insert_params_to_func_str(formatted_func)
        for i, name in enumerate(param_names):
            formatted_func.replace(f"x[{i}]", name)
        return formatted_func