"""
Created on: 27/07/2023 14:26

Author: Shyam Bhuller

Description: Code for fitting functions to data using scipy's curve fit.
#TODO Move function and rejection sampling code to it's own module
"""
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.special import gamma, erf
from scipy.stats import ks_2samp, chi2, norm, lognorm


from python.analysis import Plots, Utils

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

    @abstractmethod
    def __new__(cls, x) -> np.array:
        return cls.func(x)

    @staticmethod
    @abstractmethod
    def func():
        pass

    @staticmethod
    @abstractmethod
    def bounds(x, y):
        return (-np.inf, np.inf)

    @staticmethod
    @abstractmethod
    def p0(x, y):
        return None

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

    def __new__(cls, x, p0, p1, p2) -> np.array:
        return cls.func(x, p0, p1, p2)

    @staticmethod
    def func(x, p0, p1, p2):
        return p0 * np.exp(-(x - p1) ** 2 / (2 * p2 ** 2))

    @staticmethod
    def bounds(x, y):
        return [(0, min(x), 0.001), (np.inf, max(x), max(x) - min(x))]

    @staticmethod
    def p0(x, y):
        return [max(y), np.median(x), np.std(x)]
    
    @staticmethod
    def mu(p0, p1, p2):
        return p1

    @staticmethod
    def var(p0, p1, p2):
        return p2**2


class student_t(FitFunction):
    n_params = 4

    def __new__(cls, x, p0, p1, p2, p3) -> np.array:
        return cls.func(x, p0, p1, p2, p3)

    @staticmethod
    def func(x, p0, p1, p2, p3):
        t = (x - p1)/ p3
        return (p0**2 / p3) * (gamma((p2 + 1)/2) / (np.sqrt(p2 * np.pi) * gamma(p2/2))) * (1 + t**2/p2)**(-(p2 + 1)/2)

    @staticmethod
    def bounds(x, y):
        return [(0, min(x), 0.01, 0.001),
                (np.inf, max(x), 10, max(x) - min(x))]

    @staticmethod
    def p0(x, y):
        return [max(y), np.median(x), 2, np.std(x)]

    @staticmethod
    def mu(p0, p1, p2, p3):
        return p1

    @staticmethod
    def var(p0, p1, p2, p3):
        if p2 > 2:
            return p2 / (p2 - 2)
        elif (p2 <= 2) and p2 > 1:
            return np.Inf
        else:
            return np.NaN


class double_gaussian(FitFunction):
    n_params = 6

    def __new__(cls, x, p0, p1, p2, p3, p4, p5) -> np.array:
        return cls.func(x, p0, p1, p2, p3, p4, p5)

    @staticmethod
    def func(x, p0, p1, p2, p3, p4, p5):
        return p0 * np.exp(-(x - p1)**2 / (2 * p2**2)) + p3 * np.exp(-(x - p4)**2 / (2 * p5**2))

    @staticmethod
    def bounds(x, y):
        return [(0     , min(x), 0.001 , 0     , min(x), 0.001),
                (np.inf, max(x) , max(x) - min(x), np.inf, max(x) , max(x) - min(x))]

    @staticmethod
    def p0(x, y):
        return [max(y), np.mean(x) + np.std(x)/2, np.std(x), max(y), np.mean(x) - np.std(x)/2, np.std(x)]

    @staticmethod
    def mu(p0, p1, p2, p3, p4, p5):
        w_1 = p0 / (p0 + p3)
        w_2 = p3 / (p0 + p3)
        return (w_1 * p1) + (w_2 * p4)

    @staticmethod
    def var(p0, p1, p2, p3, p4, p5):
        w_1 = p0 / (p0 + p3)
        w_2 = p3 / (p0 + p3)

        return w_1 * (p2**2) + w_2 * (p5**2) + (w_1 * (p1**2) + w_2 * (p4**2) - (w_1*p1 + w_2*p4)**2)


class crystal_ball(FitFunction):
    n_params = 5

    def __new__(cls, x, p0, p1, p2, p3, p4) -> np.array:
        return cls.func(x, p0, p1, p2, p3, p4)

    @staticmethod
    def func(x, p0, p1, p2, p3, p4):
        t = (x - p1) / p2

        a_alpha = abs(p3)
        n_alpha = p4 / a_alpha

        A = (n_alpha)**p4 * np.exp(-a_alpha**2 / 2)

        B = n_alpha - a_alpha

        C = (n_alpha) * (1/(p4 - 1)) * np.exp(-a_alpha**2/2)

        D = np.sqrt(np.pi / 2) *(1 + erf(a_alpha/np.sqrt(2)))

        N = 1 / (C + D) # should be 1 / sigma * (C + D), but I dont want to normalise the function

        y = np.where(t > -p3, np.exp(-t**2 / 2), A * Utils.fpower(B - t, -p4))
        return p0 * N * y

    @staticmethod
    def bounds(x, y):
        return [(0, min(x), 0.001, 1, 2),
                (np.inf, max(x), max(x) - min(x), 3, 10)]

    @staticmethod
    def p0(x,y):
        return [max(y), np.mean(x), np.std(x), 1, 2]

    @staticmethod
    def mu(p0, p1, p2, p3, p4):
        return p1

    @staticmethod
    def var(p0, p1, p2, p3, p4):
        return p2


class poly2d(FitFunction):
    n_params = 3

    def __new__(cls, x, p0, p1, p2) -> np.array:
        return cls.func(x, p0, p1, p2)

    @staticmethod
    def func(x, p0, p1, p2):
        return p0 + (p1 * x) + p2 * (x**2)

    @staticmethod
    def p0(x, y):
        return None

    @staticmethod
    def bounds(x, y):
        return ([-np.inf, -np.inf, -np.inf], [np.inf]*3)


class exp(FitFunction):
    n_params = 3

    def __new__(cls, x, p0, p1, p2) -> np.array:
        return cls.func(x, p0, p1, p2)

    @staticmethod
    def func(x, p0, p1, p2):
        return p0 + (p1 * np.exp(p2 * x))

    @staticmethod
    def p0(x, y):
        return [0, 1, 1E-3]

    @staticmethod
    def bounds(x, y):
        return ([-np.inf]*3, [np.inf]*3)



class poly3d(FitFunction):
    n_params = 4

    def __new__(cls, x, p0, p1, p2, p3) -> np.array:
        return cls.func(x, p0, p1, p2, p3)

    @staticmethod
    def func(x, p0, p1, p2, p3):
        return p0 + (p1 * x) + (p2 * (x**2)) + (p3 * (x**3))

    @staticmethod
    def p0(x, y):
        return None

    @staticmethod
    def bounds(x, y):
        return ([-np.inf]*4, [np.inf]*4)


class double_crystal_ball(FitFunction):
    n_params = 7

    def __new__(cls, x, p0, p1, p2, p3, p4, p5, p6) -> np.array:
        return cls.func(x, p0, p1, p2, p3, p4, p5, p6)

    @staticmethod
    def ExponentNormalisation(alpha, n):
        A = (n/abs(alpha))**n * np.exp(-0.5 * abs(alpha)**2)
        B = (n/abs(alpha)) - abs(alpha)
        return A, B

    @staticmethod
    def func(x, p0, p1, p2, p3, p4, p5, p6):
        z = (x - p1) / p2
        A, B = double_crystal_ball.ExponentNormalisation(p3, p4)
        C, D = double_crystal_ball.ExponentNormalisation(p5, p6)
        E1 = A * Utils.fpower(B - z, -p4)
        E2 = C * Utils.fpower(D + z, -p6)
        y = np.exp(-0.5*(z**2))
        y = np.where(z < -p3, E1, y)
        y = np.where(z > p5, E2, y)
        return p0 * y

    @staticmethod
    def bounds(x, y):
        # note p4 and p6 can be unbounded, but negative values have weird distribution shapes
        return [(0, min(x), 0.001, 0.001, 0, 0.001, 0), (np.inf, max(x), max(x) - min(x), np.inf, 10, np.inf, 10)]

    @staticmethod
    def p0(x, y):
        return [max(y), np.median(x), np.std(x), 1, 1, 1, 1]

    @staticmethod
    def mu(p0, p1, p2, p3, p4, p5, p6):
        return p1
    
    @staticmethod
    def var(p0, p1, p2, p3, p4, p5, p6):
        return p2


class line(FitFunction):
    n_params = 2

    def __new__(cls, x, p0, p1) -> np.array:
        return cls.func(x, p0, p1)

    def func(x, p0, p1):
        return (p0 * x) + p1


class asym(FitFunction):
    n_params = 3

    def __new__(cls, x, p0, p1, p2) -> np.array:
        return cls.func(x, p0, p1, p2)

    def func(x, p0, p1, p2):
        return 1/(p0*(x**p2) + p1)


class lognormal_gaussian_exp(FitFunction):
    n_params = 8

    def __new__(cls, x, p0, p1, p2, p3, p4, p5, p6, p7) -> np.array:
        return cls.func(x, p0, p1, p2, p3, p4, p5, p6, p7)

    def func(x, p0, p1, p2, p3, p4, p5, p6, p7):
        lognormal_component = lognorm.pdf(x, s = p2, scale = p1)
        gaussian_component = norm.pdf(x, loc = p4, scale = p5)
        exponential_component = np.exp(-p7 * x)
        return p0 * lognormal_component + p3 * gaussian_component + p6 * exponential_component # Adjust weights as needed

    def bounds(x, y):
        lims = np.array([
            (0, 1),
            (min(x), max(x)),
            (0.001, np.inf),

            (0, 1),
            (min(x), max(x)),
            (0.001, np.inf),

            (-np.inf, np.inf),
            (-np.inf, np.inf),

        ])
        return (lims[:, 0], lims[:, 1])    


def RejectionSampling(num : int, low : float, high : float, func : FitFunction, params : dict, scale_param : str = "p0", rng : np.random.default_rng = None) -> np.array:
    """ Performs Rejection sampling for a given function which describes a pdf.

    Args:
        num (int): number of samples to generate.
        low (float): minimum range.
        high (float): maximum range.
        func (FitFunction): function which desribes a pdf, by default should be a function whose amplitude or scale parameter is p0.
        params (dict): parameters of the function.
        scale_param (str, optional): name of amplitude/scale parameter. Defaults to "p0".

    Returns:
        np.array: sampled values.
    """

    if rng is None:
        rng = np.random.default_rng()

    pdf_params = {i : params[i] for i in params}
    pdf_params[scale_param] = 1 # fix ampltiude parameter to 1

    x = np.array([])
    while len(x) < num:
        u = rng.uniform(low, high, num) # generate a random range of values
        v = rng.uniform(0, 1, num) # generate a random probability
        keep = v <= func(u, **pdf_params) # reject if v > probability of observing u
        x = np.concatenate([x, u[keep]]) # concatenate x
    return x[:num] #? is there a way to generate only the desired number rather than truncating x?


def Fit(x : np.array, y_obs : np.array, y_err : np.array, func : FitFunction, method = "trf", maxfev = int(10E4), plot : bool = False, xlabel : str = "", ylabel : str = "", ylim : list = None, plot_style : str = "scatter", title : str = "", plot_range : list = None, return_chi_sqr : bool = False, loc = "upper right") -> tuple[np.array, np.array]:
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
        plot_style (str, optional): plot style of the oberved data points, either "hist" or "scatter". Defaults to "scatter".
        title (str, optional): plot title.s Defaults to "".
        plot_range (list, optional): plot range. Defaults to None.
        return_chi_sqr (bool, optional): additionally returns chi_sqr ndf and p value in a tuple (chi_sqr, ndf, p), defaults to False

    Returns:
        tuple[np.array, np.array]: fit parameters and errors in the parameters
    """

    y_obs = np.array(y_obs, dtype = float) # ensure the input is an array of numpy floats

    mask = ~np.isnan(y_obs) # remove nans

    x = np.array(x[mask], dtype = float)
    y_obs = y_obs[mask]

    if y_err is not None:
        y_err = np.array(y_err, dtype = float)
        y_err = y_err[mask]

    popt, pcov = curve_fit(func.func, x, y_obs, sigma = y_err, maxfev = maxfev, p0 = func.p0(x, y_obs), bounds = func.bounds(x, y_obs), method = method, absolute_sigma = True)
    perr = np.sqrt(np.diag(pcov))

    y_pred = func.func(x, *popt) # y values predicted from the fit
    chisqr = abs(np.nansum(((y_obs - y_pred))**2/y_pred)) # abs in case predictions are negative
    ndf = len(y_obs) - len(popt)
    p_value = 1 - chi2.cdf(chisqr, ndf)

    if plot is True:
        #* main plotting
        x_interp = np.linspace(min(x), max(x), 1000)
        Plots.Plot(x_interp, func.func(x_interp, *popt), newFigure = False, x_scale = "linear", ylabel = ylabel, color = "#1f77b4", zorder = 11, label = "fit", title = title)
        
        p_min = popt - perr
        p_max = popt + perr

        plt.fill_between(x_interp, func.func(x_interp, *p_max), func.func(x_interp, *p_min), color = "#7f7f7f", alpha = 0.5, zorder = 10, label = "$1\sigma$ error region")

        if plot_style == "hist":
            marker = ""
            colour = "black"
            label = "observed uncertainty"
            widths = (x[1] - x[0])/2
            Plots.PlotHist(x - widths, x - widths, weights = y_obs, color = "#d62728", label = "observed", newFigure = False, range = plot_range)
        else:
            marker = "x"
            colour = "#d62728"
            label = "observed"

        Plots.Plot(x, y_obs, yerr = y_err, marker = marker, linestyle = "", color = colour, xlabel = xlabel, label = label, newFigure = False)
        if ylim:
            plt.ylim(*sorted(ylim))

        main_legend = plt.legend(loc = "upper left")
        main_legend.set_zorder(12)

        #* add fit metrics to the plot in a second legend
        plt.gca().add_artist(main_legend)
        text = ""
        for j in range(len(popt)):
            text += f"\n$p_{{{j}}}$: ${popt[j]:.2g}\pm${perr[j]:.2g}"
        text += "\n$\chi^{2}/ndf$ : " + f"{chisqr/ndf:.2g}, p : " + f"{p_value:.1g}"
        legend = plt.gca().legend(handlelength = 0, labels = [text[1:]], loc = loc, title = Utils.remove_(func.__name__))
        legend.set_zorder(12)
        for l in legend.legendHandles:
            l.set_visible(False)

    if return_chi_sqr == True:
        return popt, perr, (chisqr, ndf, p_value)
    else:
        return popt, perr


def ExtractCentralValues_df(df : pd.DataFrame, bin_variable : str, variable : str, v_range : list, funcs, data_bins : list, hist_bins : int, log : bool = False, rms_err : bool = False, weights : np.ndarray = None):
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
        mask = (df[bin_variable] > data_bins[i]) & (df[bin_variable] < data_bins[i+1])
        binned_data = df[mask]
        if weights is None:
            binned_weights = None
        else:
            binned_weights = np.array(weights)[mask]

        y, edges = np.histogram(binned_data[variable], bins = hist_bins, range = [min(v_range), max(v_range)], weights = binned_weights)
        x = (edges[1:] + edges[:-1]) / 2
        x_interp = np.linspace(min(x), max(x), hist_bins*5)

        best_f = None
        best_popt = None
        best_perr = None
        k_best = None
        p_best = None

        for f in funcs:
            popt = None
            pcov = None
            perr = None
            try:
                popt, pcov = curve_fit(f, x, y, p0 = f.p0(x, y), method = "dogbox", bounds = f.bounds(x, y), maxfev = 500000)
                perr = np.sqrt(np.diag(pcov))
                print_log(popt)
                print_log(perr)
                print_log(pcov)
            except Exception as e:
                print_log("could not fit, reason:")
                print_log(e)
                pass
            y_pred = f.func(x, *popt) if popt is not None else None
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
            mean = best_f.mu(*best_popt)
            if rms_err:
                mean_error = np.sqrt(abs(best_f.var(*best_popt))/len(binned_data[variable]))
            else:
                mean_error = mean - best_f.mu(*(best_popt + best_perr))
            y_pred = best_f.func(x, *best_popt)
            y_pred_interp = best_f.func(x_interp, *best_popt)
            k, p = ks_2samp(y, y_pred)

            Plots.Plot(x_interp, y_pred_interp, marker = "", color = "black", newFigure = False, label = "fit")
            plt.axvline(mean, color = "black", linestyle = "--", label = "central value")
        Plots.PlotHist(binned_data[variable], bins = hist_bins, newFigure = False, title = f"bin : {[data_bins[i], data_bins[i+1]]}", range = [min(v_range), max(v_range)], weights = binned_weights)

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
