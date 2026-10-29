"""Upstream beam corrections and studies."""
import awkward as ak
import numpy as np
import pandas as pd

from python.analysis import fitting

def upstream_energy_loss(KE_inst : ak.Array, params : np.ndarray, function : fitting.FitFunction = fitting.poly2d) -> ak.Array:
    """ Compute the upstream loss based on a repsonse function and it's fit parameters.

    Args:
        KE_inst (ak.Array): kinetic energy measured by the beam instrumentation
        function (fitting.FitFunction): repsonse function, defaults to fitting.poly2d. 
        params (np.ndarray): function paramters

    Returns:
        ak.Array: upstream energy loss
    """
    return function.func(KE_inst, **params)

central_value_method = {
    "None" : None,
    "gaussian" : fitting.gaussian,
    "double_gaussian" : fitting.double_gaussian,
    "student_t" : fitting.student_t,
    "crystal_ball" : fitting.crystal_ball,
}


def numerical_cv(bins : np.array, KE_reco_inst : np.array, KE_true_ff : np.array) -> tuple[np.array, np.array]:
    """ Central value in reco bin using the arithmetic mean.

    Args:
        bins (np.array): bin edges
        KE_reco_inst (np.array): reco KE at instrumentation
        KE_true_ff (np.array): true front facing KE

    Returns:
        tuple[np.array, np.array]: arithmetc mean in each bin, error in the mean
    """
    binned_data = {"KE_inst": [], "KEff_true" : [], "KE_first_true" : []}
    for i in range(len(bins)-1):
        mask = (KE_reco_inst > bins[i]) & (KE_reco_inst < bins[i + 1])
        mask = mask & (KE_true_ff > 0)

        binned_data["KE_inst"].append( KE_reco_inst[mask] )
        binned_data["KEff_true"].append( KE_true_ff[mask] )
    binned_data = {i : ak.Array(binned_data[i]) for i in binned_data}

    print(ak.num(binned_data["KE_inst"]))
    residual_energy = binned_data["KE_inst"] - binned_data["KEff_true"]

    mean_residual_energy = ak.mean(residual_energy, axis = -1)
    mean_error_residual_energy = ak.std(residual_energy, axis = -1) / np.sqrt(ak.num(residual_energy))
    return mean_residual_energy, mean_error_residual_energy


def central_value_estimation(bins : np.ndarray, KE_reco_inst : np.ndarray, KE_true_ff : np.ndarray, cv_function : fitting.FitFunction = None, weights : np.ndarray = None) -> tuple[np.array, np.array]:
    """ Estiamte upstream loss using a reponse function to correct the reco KE at the instrumentaiton to get the KE at the front face of the TPC.
        estiamtes the central value of residuals in bins of KE_reco_inst using a fitting function.

    Args:
        bins (np.array): reco KE bins
        KE_reco_inst (np.array): reco KE at instrumentation
        KE_true_ff (np.array): true front facing KE
        cv_function (fitting.FitFunction, optional): function to fit residuals to in order to get the central value. Defaults to None.

    Returns:
        tuple[np.array, np.array]: central values in each bin, error in central values
    """
    if cv_function is None:
        cv = numerical_cv(bins, KE_reco_inst, KE_true_ff)
    else:
        df = pd.DataFrame({"KE_inst" : KE_reco_inst, "true_ffKE" : KE_true_ff})
        df["residual"] = df.KE_inst - df.true_ffKE
        cv = fitting.extract_central_values_df(df, "KE_inst", "residual", [-250, 250], [cv_function], bins, 50, rms_err = False, weights = weights, bin_units = "(MeV)", bin_label = "$KE^{reco}_{inst}$")
    return cv