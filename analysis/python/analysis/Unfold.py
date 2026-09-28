"""
Created on: 24/09/2026 11:19

Author: Shyam Bhuller

Description: Functions for performing Iterative Bayesian Unfolding.
"""

import numpy as np
import pandas as pd

from scipy.interpolate import UnivariateSpline

from python.analysis import EnergySlice, Plots

from python.analysis.Slices import Slices
from python.analysis.Utils import nandiv
from python.analysis import cross_section


### OVERRIDE UNFOLDING API TO RETURN COVARIANCE MATRIX ###
from pyunfold.callbacks import setup_callbacks_regularizer, Logger
from pyunfold.mix import Mixer
from pyunfold.teststat import get_ts, KS
from pyunfold.priors import setup_prior
from pyunfold.utils import cast_to_array

def iterative_unfold(data=None, data_err=None, response=None,
                     response_err=None, efficiencies=None,
                     efficiencies_err=None, prior=None, ts='ks',
                     ts_stopping=0.01, max_iter=100, cov_type='multinomial',
                     return_iterations=False, callbacks=None):
    """Performs iterative unfolding. Custom method which returns the covariance matrix.

    Parameters
    ----------
    data : array_like
        Input observed data distribution.
    data_err : array_like
        Uncertainties of the input observed data distribution. Must be the
        same shape as ``data``.
    response : array_like
        Response matrix.
    response_err : array_like
        Uncertainties of response matrix. Must be the same shape as
        ``response``.
    efficiencies : array_like
        Detection efficiencies for the cause distribution.
    efficiencies_err : array_like
        Uncertainties of detection efficiencies. Must be the same shape as
        ``efficiencies``.
    prior : array_like, optional
        Prior distribution to use in unfolding. If None, then a uniform
        (or flat) prior will be used. If array_like, then must have the same
        shape as ``efficiencies`` (default is None).
    ts : {'ks', 'chi2', 'bf', 'rmd'}
        Test statistic to use for stopping condition (default is 'ks').
        For more information about the available test statistics, see the
        `Test Statistics API documentation <api.rst#test-statistics>`__.
    ts_stopping : float, optional
        Test statistic stopping condition. At each unfolding iteration, the
        test statistic is computed between the current and previous iteration.
        Once the test statistic drops below ts_stopping, the unfolding
        procedure is stopped (default is 0.01).
    max_iter : int, optional
        Maximum number of iterations to allow (default is 100).
    cov_type : {'multinomial', 'poisson'}
        Whether to use the Multinomial or Poisson form for the covariance
        matrix (default is 'multinomial').
    return_iterations : bool, optional
        Whether to return unfolded distributions for each iteration
        (default is False).
    callbacks : list, optional
        List of ``pyunfold.callbacks.Callback`` instances to be applied during
        unfolding (default is None, which means no Callbacks are applied).

    Returns
    -------
    unfolded_result : dict
        Returned if ``return_iterations`` is False (default). Dictionary
        containing the final unfolded distribution, associated uncertainties,
        and test statistic information.

        The returned ``dict`` has the following keys:

            unfolded
                Final unfolded cause distribution
            stat_err
                Statistical uncertainties on the unfolded cause distribution
            sys_err
                Systematic uncertainties on the unfolded cause distribution
                associated with limited statistics in the response matrix
            ts_iter
                Final test statistic value
            ts_stopping
                Test statistic stopping criterion
            num_iterations
                Number of unfolding iterations
            unfolding_matrix
                Unfolding matrix

    unfolding_iters : pandas.DataFrame
        Returned if ``return_iterations`` is True. DataFrame containing the
        unfolded distribution, associated uncertainties, test statistic
        information, etc. at each iteration.
    """
    # Validate user input
    inputs = {'data': data,
              'data_err': data_err,
              'response': response,
              'response_err': response_err,
              'efficiencies': efficiencies,
              'efficiencies_err': efficiencies_err
              }
    for name in inputs:
        if inputs[name] is None:
            raise ValueError('The input for {} must not be None.'.format(name))
        elif np.amin(inputs[name]) < 0:
            raise ValueError('The items in {} must be non-negative.'.format(name))

    data, data_err = cast_to_array(data, data_err)
    response, response_err = cast_to_array(response, response_err)
    efficiencies, efficiencies_err = cast_to_array(efficiencies,
                                                   efficiencies_err)

    num_causes = len(efficiencies)

    # Setup prior
    prior = setup_prior(prior=prior, num_causes=num_causes)

    # Define first prior counts distribution
    n_c = np.sum(data) * prior

    # Setup Mixer
    mixer = Mixer(data=data,
                  data_err=data_err,
                  efficiencies=efficiencies,
                  efficiencies_err=efficiencies_err,
                  response=response,
                  response_err=response_err,
                  cov_type=cov_type)

    # Setup test statistic
    ts_obj = get_ts(ts)
    ts_func = ts_obj(tol=ts_stopping,
                     num_causes=num_causes,
                     TestRange=[0, 1e2],
                     verbose=False)

    unfolding_iters = _unfold_custom(prior=n_c,
                                     mixer=mixer,
                                     ts_func=ts_func,
                                     max_iter=max_iter,
                                     callbacks=callbacks)

    if return_iterations:
        return unfolding_iters
    else:
        unfolded_result = dict(unfolding_iters.iloc[-1])
        return unfolded_result


def _unfold_custom(prior=None, mixer=None, ts_func=None, max_iter=100,
            callbacks=None):
    """Perform iterative unfolding. Custom version of the method which just returns the covariance method in addition to the regular output

    Parameters
    ----------
    prior : array_like
        Initial cause distribution.
    mixer : pyunfold.Mix.Mixer
        Mixer to perform the unfolding.
    ts_func : pyunfold.Utils.TestStat
        Test statistic object.
    max_iter : int, optional
        Maximum allowed number of iterations to perform.
    callbacks : list, optional
        List of ``pyunfold.callbacks.Callback`` instances to be applied during
        unfolding (default is None, which means no Callbacks are applied).

    Returns
    -------
    unfolding_iters : pandas.DataFrame
        DataFrame containing the unfolded result for each iteration.
        Each row in unfolding_result corresponds to an iteration.
    """
    # Set up callbacks, regularizer Callbacks are treated separately
    callbacks, regularizer = setup_callbacks_regularizer(callbacks)
    callbacks.on_unfolding_begin()

    current_n_c = prior.copy()
    iteration = 0
    unfolding_iters = []
    while not ts_func.pass_tol() and iteration < max_iter:
        callbacks.on_iteration_begin(iteration=iteration)

        # Perform unfolding for this iteration
        unfolded_n_c = mixer.smear(current_n_c)
        iteration += 1
        status = {'unfolded': unfolded_n_c,
                  'stat_err': mixer.get_stat_err(),
                  'sys_err': mixer.get_MC_err(),
                  'num_iterations': iteration,
                  'unfolding_matrix': mixer.Mij,
                  'covariance_matrix': mixer.get_cov()}

        if regularizer:
            # Will want the nonregularized distribution for the final iteration
            unfolded_nonregularized = status['unfolded'].copy()
            regularizer.on_iteration_end(iteration=iteration, status=status)

        ts_iter = ts_func.calc(status['unfolded'], current_n_c)
        status['ts_iter'] = ts_iter
        status['ts_stopping'] = ts_func.tol

        callbacks.on_iteration_end(iteration=iteration, status=status)
        unfolding_iters.append(status)

        # Updated current distribution for next iteration of unfolding
        current_n_c = status['unfolded'].copy()

    # Convert unfolding_iters list of dictionaries to a pandas DataFrame
    unfolding_iters = pd.DataFrame.from_records(unfolding_iters)

    # Replace final folded iteration with un-regularized distribution
    if regularizer:
        last_iteration_index = unfolding_iters.index[-1]
        unfolding_iters.at[last_iteration_index, 'unfolded'] = unfolded_nonregularized

    callbacks.on_unfolding_end(status=status)

    return unfolding_iters
 ##############################################################

def CorrelationMarix(observed : np.ndarray, true : np.ndarray, bins : np.ndarray, remove_overflow : bool = True) -> np.ndarray:
    """ Caclulate Correlation matrix of observed and true parameters.

    Args:
        observed (np.ndarray): observed data (reco).
        true (np.ndarray): true data (truth).
        bins (np.ndarray): bins.
        remove_overflow (bool, optional): remove the first bins which are interpreted as overflow. Defaults to True.

    Returns:
        np.ndarray: Correlation matrix.
    """
    corr = np.histogram2d(np.array(observed), np.array(true), bins = bins)[0]
    if remove_overflow is True:
        corr = corr[1:, 1:]
    return corr


def ResponseMatrix(observed : np.ndarray, true : np.ndarray, bins : np.ndarray, efficiencies : np.ndarray = None, remove_overflow : bool = False) -> tuple[np.ndarray, np.ndarray]:
    """ Caclulate Correlation matrix of observed and true parameters.

    Args:
        observed (np.ndarray): observed data (reco).
        true (np.ndarray): true data (truth).
        bins (np.ndarray): bins.
        efficiencies (np.ndarray, optional): selection efficiency. Defaults to None.
        remove_overflow (bool, optional): remove the first bins which are interpreted as overflow. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: response matrix and the statistical error in the response matrix
    """
    if efficiencies is None:
        efficiencies = np.ones(len(bins) - 1 - int(remove_overflow))

    response_hist = Unfold.CorrelationMarix(observed, true, bins, remove_overflow)
    response_hist_err = np.sqrt(response_hist)

    column_sums = response_hist.sum(axis=0)
    if remove_overflow is True:
        normalization_factor = nandiv(efficiencies[1:], column_sums)
    else:
        normalization_factor = nandiv(efficiencies, column_sums)
    response = response_hist * normalization_factor
    response_err = response_hist_err * normalization_factor
    
    response = np.nan_to_num(response)
    response_err = np.nan_to_num(response_err)

    return response, response_err

 #? move to Plots?
def PlotMatrix(matrix : np.ndarray, energy_slices : Slices, title : str = None, c_label : str = None, text : bool = False, text_colour = "k", cmap = "plasma"):
    """ Plot numpy matrix.

    Args:
        matrix (np.ndarray): matrix
        title (str, optional): plot title. Defaults to None.
        c_label (str, optional): colourbar label. Defaults to None.
    """
    x = energy_slices.pos_overflow - energy_slices.width/2
    #* cause = true, effect = reco
    Plots.plt.figure()
    Plots.plt.imshow(np.flip(matrix), origin = "lower", cmap = cmap, vmin = np.nanmin(matrix), vmax = np.nanmax(matrix))
    Plots.plt.xlabel("True $KE$ (MeV)")
    Plots.plt.ylabel("Reco $KE$ (MeV)")
    Plots.plt.grid(False)
    Plots.plt.colorbar(label = c_label)
    Plots.plt.title(title, pad = 10)
    Plots.plt.xticks(np.linspace(0, len(x) - 1, len(x)), np.array(x[::-1], dtype = int), rotation = 30)
    Plots.plt.yticks(np.linspace(0, len(x) - 1, len(x)), np.array(x[::-1], dtype = int), rotation = 30)
    Plots.plt.tight_layout(pad = 1)

    if text:
        for (i, j), z in np.ndenumerate(np.flip(matrix)):
            Plots.plt.gca().text(j, i, f"{z:.1g}", ha='center', va='center', fontsize = 10, color = text_colour)
    return


def CalculateResponseMatrices(template : cross_section.AnalysisInput, process : str, energy_slice : Slices, regions : bool = False, book : Plots.PlotBook = None, efficiencies : dict[np.ndarray] = None) -> dict[np.ndarray]:
    """ Calculate response matrix of energy histograms from analysis input.

    Args:
        template (AnalysisInput): template analysis input
        process (str): exclusive process
        energy_slice (Slices): energy slices
        book (Plots.PlotBook, optional): plot book. Defaults to None.
        efficiencies (dict[np.ndarray], optional): selection efficiencies. Defaults to None.

    Returns:
        dict[np.ndarray]: response matrices with errors for each histogram
    """
    slice_bins = np.arange(-1 - 0.5, energy_slice.max_num + 1.5)

    outside_tpc_mask = template.outside_fv_reco | template.outside_fv_true

    true_slices = EnergySlice.SliceNumbers(template.KE_int_true, template.KE_init_true, outside_tpc_mask, energy_slice)
    reco_slices = EnergySlice.SliceNumbers(template.KE_int_reco, template.KE_init_reco, outside_tpc_mask, energy_slice)

    if regions:
        channel = {i : (template.exclusive_process[i])[~outside_tpc_mask] for i in template.regions}
    else:
        channel = template.exclusive_process[process][~outside_tpc_mask]

    slice_pairs = {
        "init" : [reco_slices[0], true_slices[0]],
        "int" : [reco_slices[1], true_slices[1]]
    }

    if regions:
        for i in channel:
            slice_pairs[i] = [reco_slices[1][channel[i]], true_slices[1][channel[i]]]
    else:
        slice_pairs["int_ex"] = [reco_slices[1][channel], true_slices[1][channel]]

    corr = {}
    resp = {}

    labels = {"init" : "$N_{init}$", "int" : "$N_{int}$", "int_ex" : "$N_{int, ex}$", "absorption": "$N_{int,abs}$", "charge_exchange" : "$N_{int,cex}$", "single_pion_production" : "$N_{int,spip}$", "pion_production" : "$N_{int,pip}$"}


    for k, v in slice_pairs.items():
        corr[k] = Unfold.CorrelationMarix(*v, bins = slice_bins, remove_overflow = False)
        resp[k] = Unfold.ResponseMatrix(*v, bins = slice_bins, efficiencies = None if efficiencies is None else efficiencies[k][0], remove_overflow = False)

    if book is not None:
        for k in resp:
            Unfold.PlotMatrix(corr[k], energy_slice, title = f"Response marix: {labels[k]}", c_label = "Counts")
            book.Save()
            Unfold.PlotMatrix(resp[k][0], energy_slice, title = f"Normalised response matrix: {labels[k]}", c_label = "$P(E_{i}|C_{j})$")
            book.Save()

    return resp


def Unfold(observed : dict[np.ndarray], observed_err : dict[np.ndarray], response_matrices : dict[np.ndarray], priors : dict[np.ndarray] = None, ts_stop = 0.01, max_iter = 100, ts = "ks", regularizers : dict[UnivariateSpline] = None, verbose : bool = False, efficiencies : dict[np.ndarray] = None, covariance : str = "multinomial") -> dict[dict]:
    """ Run iterative bayesian unfolding for each histogram.

    Args:
        observed (dict[np.ndarray]): observed data
        observed_err (dict[np.ndarray]): observed data error
        response_matrices (dict[np.ndarray]): repsonse matrices
        priors (dict[np.ndarray], optional): pior distributions. Defaults to None.
        ts_stop (float, optional): tolerance of test statistic. Defaults to 0.01.
        max_iter (int, optional): maximum number of iterations. Defaults to 100.
        ts (str, optional): test statistic type. Defaults to "ks".
        regularizers (dict[UnivariateSpline], optional): splines to regularise the priors. Defaults to None.
        verbose (bool, optional): verbose printout. Defaults to False.
        efficiencies (dict[np.ndarray], optional): selection efficiencies. Defaults to None.

    Returns:
        dict: unfolding results for each histogram.
    """
    def make_cb(key):
        cb = []
        if verbose: cb.append(Logger())
        if regularizers is not None:
            cb = cb + [regularizers[key]]
        return cb
    results = {}

    for k, v in response_matrices.items():
        if verbose: print(k)
        n = observed[k]
        n_e = observed_err[k]


        cb = make_cb(k)

        if efficiencies is not None:
            efficiency = efficiencies[k][0]
            efficiency_err = efficiencies[k][1]
        else:
            efficiency = np.ones_like(n, dtype = float) #! for the toy, assume perfect selection efficiency, so 1 +- 0
            efficiency_err = np.zeros_like(n, dtype = float) #? not exactly zero, make this very small so the systematic uncertainty from the response matrix can still be calculated?

        if priors is None:
            p = n/sum(n)
        else:
            p = priors[k] / sum(priors[k])

        results[k] = iterative_unfold(n, n_e, v[0], v[1], efficiency, efficiency_err, callbacks = cb, prior = p, ts_stopping = ts_stop, max_iter = max_iter, ts = ts, cov_type = covariance)
    return results


def PlotUnfoldingResults(obs : np.ndarray, obs_err : np.ndarray, true : np.ndarray, results : dict, energy_slices : Slices, title : str, book : Plots.PlotBook = Plots.PlotBook.null()):
    """ Plot unfolded histogram in comparison to observed and true.

    Args:
        obs (np.ndarray): observation
        true (np.ndarray): truth
        results (dict): unfolding results
        energy_bins (np.ndarray): energy bins
        label (str): x label (units of MeV are automatically applied)
        book (Plots.PlotBook, optional): plot book. Defaults to Plots.PlotBook.null().
    """
    if "num_iterations" in results:
        label = f"Data unfolded, {results['num_iterations']} iterations"
    else:
        label = "Data unfolded"

    print(title)
    # print(f"original: {weighted_chi_sqr(obs/sum(obs), true/sum(true), np.sqrt(true/sum(true)))}")
    # print(f"unfolded: {weighted_chi_sqr(results['unfolded']/sum(results['unfolded']), true/sum(true), np.sqrt(true/sum(true)))}")
    # print(f"original: {weighted_chi_sqr(obs/sum(obs), true/sum(true), obs_err/sum(obs))}")
    # print(f"unfolded: {weighted_chi_sqr(results['unfolded']/sum(results['unfolded']), true/sum(true), results['stat_err']/sum(results['unfolded']))}")
    # print(f"original: {ks_2samp(obs, true)}")
    # print(f"unfolded: {ks_2samp(results['unfolded'], true)}")
    ks = KS(num_causes = len(obs))
    print(f"original: {ks.calc(obs, true)}")
    print(f"unfolded: {ks.calc(results['unfolded'], true)}")


    cross_section.PlotXSHists(energy_slices, true, None, True, 1/sum(true), label = "MC true (initial prior)", ylabel = "Fractional counts", color = "C1", newFigure = False)
    cross_section.PlotXSHists(energy_slices, obs, obs_err, True, 1/sum(obs), label = "Data reco", ylabel = "Fractional counts", color = "k")
    cross_section.PlotXSHists(energy_slices, results["unfolded"], results["stat_err"], True, 1 / sum(results["unfolded"]), label =  label, color = "C4", ylabel = "Fractional counts", newFigure = False, title = title)
    Plots.plt.legend(loc = "upper left")
    book.Save() 
    if "unfolding_matrix" in results:
        Unfold.PlotMatrix(results["unfolding_matrix"], energy_slices, title = "Unfolded matrix: " + label, c_label = "$P(C_{j}|E_{i})$", text = True, text_colour = "red")
        book.Save()

        Unfold.PlotMatrix((results["unfolding_matrix"].T * obs).T, energy_slices, "Migrations : " + label, c_label = "Counts")
        book.Save()

    if "covariance_matrix" in results:
        Unfold.PlotMatrix(results["covariance_matrix"], energy_slices, title = "Covariance matrix: " + label, c_label = "Counts")
        book.Save()

        Unfold.PlotMatrix(np.corrcoef(results["covariance_matrix"]), energy_slices, title = "Correlation matrix: " + label, c_label = "Correlation coefficient", text = True, text_colour = "k", cmap = "coolwarm")
        book.Save()

    return