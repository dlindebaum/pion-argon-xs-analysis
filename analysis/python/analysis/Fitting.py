"""
Created on: 27/07/2023 14:26

Author: Shyam Bhuller

Description: Code for fitting functions to data using scipy's curve fit.
"""
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.special import gamma, erf
from scipy.stats import ks_2samp

from python.analysis import Plots

class FitFunction(ABC):
    """ An abstract class for defining function pass to Fit(). Contains the following:
        n_params: number of parameters which describe the function e.g. a normalised gaussian has two
        func: implementation of the function itself
        p0: Initial values for the fitting, can be set to return None
        bounds: the allowed range of each fit parameter, to turn off, return (-np.inf, np.inf)
        mu: the mean value of the function, only needed for PDFs
        sigma: the rms value of the function, only needed for PDFs
        var: variance for the function, only needed for PDFs
    """
    @property
    @abstractmethod
    def n_params(self):
        """ number of fit parameters
        """
        pass

    @staticmethod
    @abstractmethod
    def func():
        pass

    @staticmethod
    @abstractmethod
    def bounds(x, y):
        pass

    @staticmethod
    @abstractmethod
    def p0():
        pass

    @staticmethod
    @abstractmethod
    def mu():
        pass
    
    @staticmethod
    @abstractmethod
    def var():
        pass

    def __init__(self) -> None:
        pass


class gaussian(FitFunction):
    n_params = 3

    @staticmethod
    def func(x, A, mu, sigma):
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def bounds(x, y):
        return [(0, min(x), 0.001), (np.inf, max(x), max(x) - min(x))]

    @staticmethod
    def p0(x, y):
        return [max(y), np.median(x), np.std(x)]
    
    @staticmethod
    def mu(A, mu, sigma):
        return mu

    @staticmethod
    def var(A, mu, sigma):
        return sigma**2


class student_t(FitFunction):
    n_params = 4
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def func(x, A, mu, nu, l):
        t = (x - mu)/ l
        return (A**2 / l) * (gamma((nu + 1)/2) / (np.sqrt(nu * np.pi) * gamma(nu/2))) * (1 + t**2/nu)**(-(nu + 1)/2)

    @staticmethod
    def bounds(x, y):
        return [(0, min(x), 0.01, 0.001),
                (np.inf, max(x), 10, max(x) - min(x))]

    @staticmethod
    def p0(x, y):
        return [max(y), np.median(x), 2, np.std(x)]

    @staticmethod
    def mu(A, mu, nu, l):
        return mu

    @staticmethod
    def var(A, mu, nu, l):
        if nu > 2:
            return nu / (nu - 2)
        elif (nu <= 2) and nu > 1:
            return np.Inf
        else:
            return np.NaN


class double_gaussian(FitFunction):
    n_params = 6

    def __init__(self) -> None:
        pass

    @staticmethod
    def func(x, A_1, mu_1, sigma_1, A_2, mu_2, sigma_2):
        return A_1 * np.exp(-(x - mu_1)**2 / (2 * sigma_1**2)) + A_2 * np.exp(-(x - mu_2)**2 / (2 * sigma_2**2))

    @staticmethod
    def bounds(x, y):
        return [(0     , min(x), 0.001 , 0     , min(x), 0.001),
                (np.inf, max(x) , max(x) - min(x), np.inf, max(x) , max(x) - min(x))]

    @staticmethod
    def p0(x, y):
        return [max(y), np.mean(x), np.std(x), max(y), np.mean(x), np.std(x)]

    @staticmethod
    def mu(A_1, mu_1, sigma_1, A_2, mu_2, sigma_2):
        w_1 = A_1 / (A_1 + A_2)
        w_2 = A_2 / (A_1 + A_2)
        return (w_1 * mu_1) + (w_2 * mu_2)

    @staticmethod
    def var(A_1, mu_1, sigma_1, A_2, mu_2, sigma_2):
        w_1 = A_1 / (A_1 + A_2)
        w_2 = A_2 / (A_1 + A_2)

        return w_1 * (sigma_1**2) + w_2 * (sigma_2**2) + (w_1 * (mu_1**2) + w_2 * (mu_2**2) - (w_1*mu_1 + w_2*mu_2)**2)


class crystal_ball(FitFunction):
    n_params = 5

    def __init__(self) -> None:
        pass

    @staticmethod
    def func(x, S, mu, sigma, alpha, n):
        t = (x - mu) / sigma

        a_alpha = abs(alpha)
        n_alpha = n / a_alpha

        A = (n_alpha)**n * np.exp(-a_alpha**2 / 2)

        B = n_alpha - a_alpha

        C = (n_alpha) * (1/(n - 1)) * np.exp(-a_alpha**2/2)

        D = np.sqrt(np.pi / 2) *(1 + erf(a_alpha/np.sqrt(2)))

        N = 1 / (C + D) # should be 1 / sigma * (C + D), but I dont want to normalise the function

        y = np.where(t > -alpha, np.exp(-t**2 / 2), A * (B - t)**-n)
        return S * N * y

    @staticmethod
    def bounds(x, y):
        return [(0, min(x), 0.001, 1, 2),
                (np.inf, max(x), max(x) - min(x), 3, 10)]

    @staticmethod
    def p0(x,y):
        return [max(y), np.mean(x), np.std(x), 1, 2]

    @staticmethod
    def mu(S, mu, sigma, alpha, n):
        return mu

    @staticmethod
    def var(S, mu, sigma, alpha, n):
        return sigma


def Fit(x : np.array, y_obs : np.array, y_err : np.array, func : FitFunction, method = "trf", maxfev = int(10E4), plot : bool = False, xlabel : str = "", ylabel : str = "", ylim : list = None) -> tuple[np.array, np.array]:
    """ Implementation of scipy's curve fit, with some constraints, checks to handle nan data and optional plotting.

    Args:
        x (np.array): x data
        y_obs (np.array): y data
        y_err (np.array): error in y
        func (FitFunction): function to fit to data
        method (str, optional): fit method, see scipy's documentation. Defaults to "trf".
        maxfev (_type_, optional): max number of iterations, see scipy's documentation. Defaults to int(10E4).
        plot (bool, optional): make a plot of the data and the fitted function. Defaults to False.
        xlabel (str, optional): plot x label. Defaults to "".
        ylabel (str, optional): plot y label. Defaults to "".
        ylim (list, optional): y limit of plot. Defaults to None.

    Returns:
        tuple[np.array, np.array]: fit parameters and errors in the parameters
    """

    y_obs = np.array(y_obs, dtype = float) # ensure the input is an array of numpy floats
    y_err = np.array(y_err, dtype = float)

    mask = ~np.isnan(y_obs) # remove nans

    x = np.array(x[mask], dtype = float)
    y_obs = y_obs[mask]
    y_err = y_err[mask]

    popt, pcov = curve_fit(func.func, x, y_obs, sigma = y_err, maxfev = maxfev, p0 = func.p0(x, y_obs), bounds = func.bounds(x, y_obs), method = method)
    perr = np.sqrt(np.diag(pcov))

    if plot is True:
        y_pred = func.func(x, *popt) # y values predicted from the fit
        y_pred_min = func.func(x, *(popt - perr)) # y values predicted from the lower limit of the fit
        y_pred_max = func.func(x, *(popt + perr)) # y values predicted from the upper limit of the fit
        y_pred_err = (abs(y_pred - y_pred_min) + abs(y_pred - y_pred_max)) / 2 # error in the predicted fit value, taken to be the average deviation from the lower and upper limits

        chisqr = np.nansum(((y_obs - y_pred)/y_pred_err)**2)
        ndf = len(y_obs) - len(popt)

        #* main plotting
        x_interp = np.linspace(min(x), max(x), 1000)
        Plots.Plot(x_interp, func.func(x_interp, *popt), newFigure = False, x_scale = "linear", xlabel = xlabel, ylabel = ylabel, color = "#1f77b4", zorder = 11, label = "fit")
        plt.fill_between(x_interp, func.func(x_interp, *(popt + perr)), func.func(x_interp, *(popt - perr)), color = "#7f7f7f", alpha = 0.5, zorder = 10, label = "$1\sigma$ error region")
        Plots.Plot(x, y_obs, yerr = y_err, marker = "x", linestyle = "", color = "#d62728", label = "sample points", newFigure = False)
        if ylim:
            plt.ylim(*sorted(ylim))

        main_legend = plt.legend(loc = "upper left")
        main_legend.set_zorder(12)

        #* add fit metrics to the plot in a second legend
        plt.gca().add_artist(main_legend)
        text = ""
        for j in range(len(popt)):
            text += f"\np{j}: ${popt[j]:.2g}\pm${perr[j]:.2g}"
        text += "\n$\chi^{2}/ndf$ : " + f"{chisqr/ndf:.2g}"
        legend = plt.gca().legend(handlelength = 0, labels = [text[1:]], loc = "upper right")
        legend.set_zorder(12)
        for l in legend.legendHandles:
            l.set_visible(False)

    return popt, perr


def create_bins_df(value : pd.Series, n_entries, v_range : list = None):
    sorted_value = value.sort_values()
    n_bins = len(sorted_value) // n_entries

    bins = []
    for i in range(n_bins + 1):
        mi = sorted_value.values[i * n_entries]
        bins.append(mi)
    if v_range:
        bins[0] = min(v_range)
        bins[-1] = max(v_range)
    return np.array(bins)


def ExtractCentralValues_df(df : pd.DataFrame, bin_variable : str, variable : str, v_range : list, funcs, data_bins : list, hist_bins : int, log : bool = False, rms_err : bool = True):
    """ Estimate a central value in each reco energy bin based on some FitFunction or collection of FitFunctions.

    Args:
        bin_variable (str): variable to bin in
        variable (str): variable to fit to
        v_range (list): variable range
        funcs (FitFunction): functions to try fit
        reco_bins (list): reco energy bins
        hist_bins (int): number of bins for variable histograms
        log (bool, optional): verbose printout. Defaults to False.
    """
    def print_log(x):
        if log: print(x)

    cv = []
    cv_err = []
    fig_handles = None
    fig_labels = None
    for i in Plots.MultiPlot(len(data_bins) - 1):
        if i == len(data_bins): continue
        print_log(i)
        binned_data = df[(df[bin_variable] > data_bins[i]) & (df[bin_variable] < data_bins[i+1])]
    
        y, edges = np.histogram(binned_data[variable], bins = hist_bins, range = [min(v_range), max(v_range)])
        x = (edges[1:] + edges[:-1]) / 2
        x_interp = np.linspace(min(x), max(x), hist_bins*5)

        best_f = None
        best_popt = None
        best_perr = None
        k_best = None
        p_best = None

        for f in funcs:
            function = f()
            popt = None
            pcov = None
            perr = None
            try:
                popt, pcov = curve_fit(function.func, x, y, p0 = function.p0(x, y), method = "dogbox", bounds = function.bounds(x, y), maxfev = 500000)
                perr = np.sqrt(np.diag(pcov))
                print_log(popt)
                print_log(perr)
                print_log(pcov)
            except Exception as e:
                print_log("could not fit, reason:")
                print_log(e)
                pass
            y_pred = function.func(x, *popt) if popt is not None else None
            if y_pred is not None:
                k, p = ks_2samp(y, y_pred)
            else:
                k = 1
                p = 0

            if p_best is None or p > p_best : # larger p value suggests a better fit
                p_best = p
                k_best = k
                best_popt = popt
                best_perr = perr
                best_f = f

        mean = None
        mean_error = None
        if best_popt is not None:
            function = best_f()
            mean = function.mu(*best_popt)
            if rms_err:
                mean_error = np.sqrt(abs(function.var(*best_popt))/len(binned_data[variable]))
            else:
                mean_error = mean - function.mu(*(best_popt + best_perr))
            y_pred = function.func(x, *best_popt)
            y_pred_interp = function.func(x_interp, *best_popt)
            k, p = ks_2samp(y, y_pred)

            Plots.Plot(x_interp, y_pred_interp, marker = "", color = "black", newFigure = False, label = "fit")
            plt.axvline(mean, color = "black", linestyle = "--", label = "central value")
        Plots.PlotHist(binned_data[variable], bins = hist_bins, newFigure = False, title = f"bin : {[data_bins[i], data_bins[i+1]]}", range = [min(v_range), max(v_range)])

        plt.axvline(np.mean(binned_data[variable]), linestyle = "--", color = "C1", label = "mean")

        if not fig_handles: fig_handles, fig_labels = plt.gca().get_legend_handles_labels()

        if best_popt is not None:
            text = ""
            for j in range(len(best_popt)):
                text += f"\np{j}: ${best_popt[j]:.2f}\pm${best_perr[j]:.2f}"
            text += f"\nks : {k_best:.2f}, p : {p_best:.2f}"
            legend = plt.gca().legend(handlelength = 0, labels = [text[1:]], title = best_f.__name__.replace("_", " "))
            for l in legend.legendHandles:
                l.set_visible(False)

        cv.append(mean)
        cv_err.append(abs(mean_error) if mean_error is not None else mean_error)
    
    plt.gcf().legend(fig_handles, fig_labels, loc = "lower right", ncols = 3)
    plt.gcf().supxlabel(variable.replace("_", " "))
    plt.tight_layout()
    return np.array(cv), np.array(cv_err)
