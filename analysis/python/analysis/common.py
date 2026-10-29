"""
Created on: 21/01/2024 17:14

Author: Shyam Bhuller

Description: random functions which are sometimes useful
"""
import argparse
import awkward as ak
import numpy as np
import os
import time
import warnings

import dill

import sys

from IPython.core.magic import register_line_magic
from IPython import get_ipython

def register_line_magic_safe(func):
    if get_ipython() is not None:
        return register_line_magic(func)
    return func

@register_line_magic_safe
def init_notebook(line):
    ipython = get_ipython()
    ipython.run_line_magic("matplotlib", "inline") # %matplotlib inline
    ipython.run_line_magic("load_ext", "autoreload") # %load_ext autoreload
    ipython.run_line_magic("autoreload", "2") # %autoreload 2

    #* magic to add python path to notebook environment
    for pypath in sys.path:
        if pypath.split("/")[-1] == "analysis": break
    print(pypath)
    ipython.run_line_magic("env", "PYTHONPATH $pypath") # %env PYTHONPATH $pypath
    return


def efficiency(selected_count : np.array, total_count : np.array) -> tuple[np.array, np.array]:
    """ Calcualtes selection efficiency and binomial error.

    Args:
        selected_count (np.array): number of selected events
        total_count (np.array): number of total events

    Returns:
        tuple[np.array, np.array]: efficiency, error
    """
    p = selected_count / total_count
    p = np.nan_to_num(p)
    error = (p * (1 - p) / total_count)**0.5
    return p, error


def timer(func):
    """ Decorator which times a function.

    Args:
        func (function): function to time
    """
    def wrapper_function(*args, **kwargs) -> object:
        """ Times funcions, returns outputs
        Returns:
            any: func output
        """
        s = time.time()
        out = func(*args,  **kwargs)
        print(f'{func.__name__!r} executed in {(time.time()-s):.4f}s')
        return out
    return wrapper_function


def dimension(shape : ak.Array | np.ndarray):
    if type(shape) != np.ndarray:
        return len(str(shape.type).split(" * ")) - 1
    else:
        return 0

def match_dimension(mask_shape : ak.Array | np.ndarray, var_shape : ak.Array | np.ndarray):
    target_dim = dimension(mask_shape)

    counter = 0
    new_shape = var_shape
    current_dim = dimension(new_shape)
    while current_dim != target_dim:
        if counter > 10:
            break
        new_shape = ak.num(new_shape, -1)
        current_dim = dimension(new_shape)
        counter += 1
    return new_shape

def args_to_dict(args : argparse.Namespace | dict) -> dict:
    if type(args) == argparse.Namespace:
        args_c = vars(args)
    else:
        args_c = args

    return args_c


def dill_copy(obj : any):
    return dill.loads(dill.dumps(obj))


def bin_centers(bins : np.ndarray) -> np.ndarray:
    return (bins[1:] + bins[:-1]) / 2


def fpower(num : np.ndarray, exp, real : bool = True):
    y = num.astype("complex")**exp
    if real: y = y.real
    return y

def nandiv(num, den) -> np.ndarray:
    return np.divide(num, np.where(den == 0, np.nan, den))


def nanlog(x) -> np.ndarray:
    return np.log(np.where(x < 0, np.nan, x))


def weighted_chi_sqr(observed, expected, uncertainties) -> np.ndarray:
    u = np.array(uncertainties)
    u[u == 0] = np.nan
    return np.nansum((observed - expected)**2 / u**2) / (len(observed)-1)


def quadsum(x : np.ndarray | list, axis : int = None) -> np.ndarray:
    return np.sqrt(np.sum(np.array(x)**2, axis))


def nanquadsum(x : np.ndarray | list, axis : int = None) -> np.ndarray:
    return np.sqrt(np.nansum(np.array(x)**2, axis))


def remove_(str) -> str:
    return str.replace("_", " ")


def ls_recursive(path : os.PathLike):
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn]


def sf(x):
    try:
        v = -int(np.ceil(np.log10(abs(x))))+1
    except (OverflowError, ValueError):
        v = 0
    return v


def round_value_to_error(v, e) -> str:
    fe = sf(e)
    fe = fe if fe > 0 else 0

    fv = sf(v)
    fv = fv if fv > 0 else 0
    f = max(fe, fv)
    return f"${round(v, f):.{f}f} \\pm {round(e, f):.{f}f}$"


def deprecation_warning():
    return warnings.warn("This function is legacy and should not be used in new implementations.", DeprecationWarning)


def dict_inverse(d : dict) -> dict:
    return {v : k for k, v in d.items()}


def clip_jagged(array, min, max):
    """ Clips a jagged (awkward) array, if the type of aray is an awkward array, returns as an awkward array, otherwise it returns a list of nested arrays

    Args:
        array: array

    Returns:
        clipped array
    """
    orig_type = type(array)
    if orig_type != ak.Array:
        array = ak.Array(array)

    array = ak.where(array < min, min, array)
    array = ak.where(array > max, max, array)

    if orig_type != ak.Array:
        array = list(array)
    return array
