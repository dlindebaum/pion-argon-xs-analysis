"""
Created on: 21/01/2024 17:14

Author: Shyam Bhuller

Description: Random functions which are sometimes useful.
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
    """Register an IPython line magic only when running inside IPython.

        Args:
            func (Any): Line-magic function to register.

        Returns:
            Any: The registered magic in IPython, or the original function otherwise.
    """
    if get_ipython() is not None:
        return register_line_magic(func)
    return func

@register_line_magic_safe
def init_notebook(line):
    """Enable inline plotting and autoreload, then expose the analysis path in a notebook.

        Args:
            line (Any): IPython magic input; unused.
    """
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
            selected_count (np.array): Number of selected events.
            total_count (np.array): Number of total events.

        Returns:
            tuple[np.array, np.array]: Efficiency, error.
    """
    p = selected_count / total_count
    p = np.nan_to_num(p)
    error = (p * (1 - p) / total_count)**0.5
    return p, error


def timer(func):
    """Wrap a function to print its execution time and return its result.

        Args:
            func (function): Function whose execution duration should be reported.

        Returns:
            Any: A wrapper that returns the decorated function's result.
    """
    def wrapper_function(*args, **kwargs) -> object:
        """Run the wrapped function, print its elapsed time, and return its output.

            Args:
                *args (Any): Positional arguments forwarded to the wrapped function.
                **kwargs (Any): Keyword arguments forwarded to the wrapped function.

            Returns:
                object: The wrapped function's return value.
        """
        s = time.time()
        out = func(*args,  **kwargs)
        print(f'{func.__name__!r} executed in {(time.time()-s):.4f}s')
        return out
    return wrapper_function


def dimension(shape : ak.Array | np.ndarray):
    """Return zero for NumPy arrays or count Awkward type dimensions.

        Args:
            shape (ak.Array | np.ndarray): Array whose dimensions are inspected.

        Returns:
            Any: Zero for NumPy arrays; otherwise the number of dimensions in the Awkward type.
    """
    if type(shape) != np.ndarray:
        return len(str(shape.type).split(" * ")) - 1
    else:
        return 0

def match_dimension(mask_shape : ak.Array | np.ndarray, var_shape : ak.Array | np.ndarray):
    """Apply ``ak.num`` repeatedly until an array's dimension matches the mask's.

        Args:
            mask_shape (ak.Array | np.ndarray): Array whose dimensionality is the target.
            var_shape (ak.Array | np.ndarray): Array to reduce with repeated ``ak.num`` calls.

        Returns:
            Any: The reduced array, or the last attempted shape after ten reductions.
    """
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
    """Convert an argparse namespace to its attribute dictionary, or pass a dictionary through.

        Args:
            args (argparse.Namespace | dict): Parsed arguments or an existing dictionary.

        Returns:
            dict: The namespace attributes, or the input dictionary unchanged.
    """
    if type(args) == argparse.Namespace:
        args_c = vars(args)
    else:
        args_c = args

    return args_c


def dill_copy(obj : any):
    """Create a deep copy by serializing and deserializing an object with dill.

        Args:
            obj (any): Object to copy.

        Returns:
            Any: A deserialized copy of the object.
    """
    return dill.loads(dill.dumps(obj))


def bin_centers(bins : np.ndarray) -> np.ndarray:
    """Calculate the midpoint of each adjacent pair of bin edges.

        Args:
            bins (np.ndarray): Ordered bin-edge values.

        Returns:
            np.ndarray: Midpoint of each consecutive edge pair.
    """
    return (bins[1:] + bins[:-1]) / 2


def fpower(num : np.ndarray, exp, real : bool = True):
    """Raise array values to a power using complex arithmetic, optionally keeping real parts.

        Args:
            num (np.ndarray): Values to exponentiate.
            exp (Any): Exponent applied elementwise.
            real (bool): If true, discard imaginary components from the result.

        Returns:
            Any: Powered values, real-valued when ``real`` is true.
    """
    y = num.astype("complex")**exp
    if real: y = y.real
    return y

def nandiv(num, den) -> np.ndarray:
    """Divide elementwise, treating zero denominators as NaN.

        Args:
            num (Any): Numerator values.
            den (Any): Denominator values; zeros are replaced with NaN.

        Returns:
            np.ndarray: Elementwise quotient with NaN where the denominator is zero.
    """
    return np.divide(num, np.where(den == 0, np.nan, den))


def nanlog(x) -> np.ndarray:
    """Take the natural logarithm, replacing negative inputs with NaN.

        Args:
            x (Any): Values to transform.

        Returns:
            np.ndarray: Elementwise natural logarithm, with negative inputs mapped to NaN.
    """
    return np.log(np.where(x < 0, np.nan, x))


def weighted_chi_sqr(observed, expected, uncertainties) -> np.ndarray:
    """Calculate a reduced chi-square using uncertainties and ``n - 1`` degrees of freedom.

        Args:
            observed (Any): Measured values.
            expected (Any): Model values corresponding to ``observed``.
            uncertainties (Any): Standard uncertainties; zero entries are treated as NaN.

        Returns:
            np.ndarray: Sum of squared uncertainty-normalized residuals divided by ``len(observed) - 1``.
    """
    u = np.array(uncertainties)
    u[u == 0] = np.nan
    return np.nansum((observed - expected)**2 / u**2) / (len(observed)-1)


def quadsum(x : np.ndarray | list, axis : int = None) -> np.ndarray:
    """Compute the square root of the sum of squared values along an axis.

        Args:
            x (np.ndarray | list): Values whose quadrature sum is calculated.
            axis (int): Axis along which to sum; ``None`` sums all values.

        Returns:
            np.ndarray: Quadrature sum.
    """
    return np.sqrt(np.sum(np.array(x)**2, axis))


def nanquadsum(x : np.ndarray | list, axis : int = None) -> np.ndarray:
    """Compute a quadrature sum while ignoring NaN values.

        Args:
            x (np.ndarray | list): Values whose quadrature sum is calculated.
            axis (int): Axis along which to sum; ``None`` sums all values.

        Returns:
            np.ndarray: Quadrature sum excluding NaN values.
    """
    return np.sqrt(np.nansum(np.array(x)**2, axis))


def remove_(str) -> str:
    """Replace underscores in a string with spaces.

        Args:
            str (Any): Text to convert.

        Returns:
            str: Text with every underscore replaced by a space.
    """
    return str.replace("_", " ")


def ls_recursive(path : os.PathLike):
    """List file paths recursively beneath a directory, expanding ``~`` in the path.

        Args:
            path (os.PathLike): Root directory to traverse.

        Returns:
            Any: Joined paths for every file found below the root.
    """
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn]


def sf(x):
    """Return decimal places calculated from the value's base-10 magnitude.

        Args:
            x (Any): Value whose order of magnitude determines the precision.

        Returns:
            Any: Decimal places from the computed magnitude, or zero when it is invalid.
    """
    try:
        v = -int(np.ceil(np.log10(abs(x))))+1
    except (OverflowError, ValueError):
        v = 0
    return v


def round_value_to_error(v, e) -> str:
    """Format a value and its uncertainty with matching, magnitude-based decimal precision.

        Args:
            v (Any): Central value to format.
            e (Any): Uncertainty that sets the displayed precision.

        Returns:
            str: LaTeX-formatted value and uncertainty separated by ``\\pm``.
    """
    fe = sf(e)
    fe = fe if fe > 0 else 0

    fv = sf(v)
    fv = fv if fv > 0 else 0
    f = max(fe, fv)
    return f"${round(v, f):.{f}f} \\pm {round(e, f):.{f}f}$"


def deprecation_warning():
    """Emit a deprecation warning for legacy code.
    """
    return warnings.warn("This function is legacy and should not be used in new implementations.", DeprecationWarning)


def dict_inverse(d : dict) -> dict:
    """Reverse a dictionary's keys and values.

        Args:
            d (dict): Mapping to invert; values are expected to be unique.

        Returns:
            dict: Mapping from each original value to its original key.
    """
    return {v : k for k, v in d.items()}


def clip_jagged(array, min, max):
    """ Clips a jagged (awkward) array, if the type of aray is an awkward array, returns as an awkward array, otherwise it returns a list of nested arrays.

        Args:
            array (Any): Nested values to clip elementwise.
            min (Any): Lower clipping bound.
            max (Any): Upper clipping bound.

        Returns:
            Result: Clipped values, returned as an Awkward array when the input was one and as a list otherwise.
    """
    orig_type = type(array)
    if orig_type != ak.Array:
        array = ak.Array(array)

    array = ak.where(array < min, min, array)
    array = ak.where(array > max, max, array)

    if orig_type != ak.Array:
        array = list(array)
    return array
