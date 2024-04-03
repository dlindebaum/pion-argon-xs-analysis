"""
Created on: 21/01/2024 17:14

Author: Shyam Bhuller

Description: random functions which are sometimes useful
"""
import numpy as np
import os

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
    return np.nansum((observed - expected)**2 / u**2) / len(observed)


def quadsum(x : np.ndarray | list, axis : int = None) -> np.ndarray:
    return np.sqrt(np.sum(np.array(x)**2, axis))


def nanquadsum(x : np.ndarray | list, axis : int = None) -> np.ndarray:
    return np.sqrt(np.nansum(np.array(x)**2, axis))


def remove_(str) -> str:
    return str.replace("_", " ")


def ls_recursive(path : os.PathLike):
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn]