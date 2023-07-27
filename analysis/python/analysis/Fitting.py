"""
Created on: 27/07/2023 14:26

Author: Shyam Bhuller

Description: Code for fitting functions to data using scipy's curve fit.
"""
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.special import gamma, erf

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

        #* add fit metrics to the plot in a second legend
        plt.gca().add_artist(main_legend)
        text = ""
        for j in range(len(popt)):
            text += f"\np{j}: ${popt[j]:.2f}\pm${perr[j]:.2g}"
        text += "\n$\chi^{2}/ndf$ : " + f"{chisqr/ndf:.2g}"
        legend = plt.gca().legend(handlelength = 0, labels = [text[1:]], loc = "upper right")
        for l in legend.legendHandles:
            l.set_visible(False)

    return popt, perr
