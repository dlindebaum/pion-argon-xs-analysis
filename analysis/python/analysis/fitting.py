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

from python.analysis import common
from python.analysis.plotting import plots

class FitFunction(ABC):
    """ An abstract class for defining function pass to Fit(). Contains the following:

        n_params: Number of parameters which describe the function e.g. a normalised gaussian has two.
        func: Implementation of the function itself.
        p0: Initial values for the fitting, can be set to return None.
        bounds: The allowed range of each fit parameter, to turn off, return (-np.inf, np.inf).
        mu: The mean value of the function, only needed for PDFs.
        sigma: The rms value of the function, only needed for PDFs.
        var: Variance for the function, only needed for PDFs.
    """
    @property
    @abstractmethod
    def n_params(self):
        """ Number of fit parameters.
        """
        pass

    @abstractmethod
    def __new__(cls, x) -> np.array:
        """Evaluate the model at the supplied x values.

            Args:
                x (Any): Independent-variable values at which to evaluate the model.

            Returns:
                np.ndarray: Model values returned by the subclass implementation at x.
        """
        return cls.func(x)

    @staticmethod
    @abstractmethod
    def func():
        """Define the subclass hook that evaluates a fit model.
        """
        pass

    @staticmethod
    @abstractmethod
    def bounds(x, y):
        """Provide unconstrained bounds by default; subclasses can set model-specific limits.

            Args:
                x (Any): Independent-variable values for the observed data.
                y (Any): Observed dependent-variable values corresponding to x.

            Returns:
                tuple: Lower and upper parameter bounds; both are unbounded by default.
        """
        return (-np.inf, np.inf)

    @staticmethod
    @abstractmethod
    def p0(x, y):
        """Return initial fit-parameter estimates, or None when no estimate is provided.

            Args:
                x (Any): Independent-variable values for the observed data.
                y (Any): Observed dependent-variable values corresponding to x.
        """
        return None

    @staticmethod
    @abstractmethod
    def mu():
        """Define the subclass hook that returns a model's mean.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def var():
        """Define the subclass hook that returns a model's variance.
        """
        pass

    def __init__(self) -> None:
        """Initialize the stateless base class for fit-function models.
        """
        pass


class gaussian(FitFunction):
    """Evaluate a Gaussian peak parameterized by amplitude, mean, and width.
    """
    n_params = 3

    def __new__(cls, x, p0, p1, p2) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the Gaussian.
                p0 (Any): Gaussian amplitude.
                p1 (Any): Gaussian mean.
                p2 (Any): Gaussian width (standard deviation).

            Returns:
                np.ndarray: Gaussian model values evaluated at x.
        """
        return cls.func(x, p0, p1, p2)

    @staticmethod
    def func(x, p0, p1, p2):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the Gaussian.
                p0 (Any): Gaussian amplitude.
                p1 (Any): Gaussian mean.
                p2 (Any): Gaussian width (standard deviation).

            Returns:
                np.ndarray: Gaussian model values evaluated at x.
        """
        return p0 * np.exp(-(x - p1) ** 2 / (2 * p2 ** 2))

    @staticmethod
    def bounds(x, y):
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values used to estimate parameter bounds.

            Returns:
                tuple: Lower and upper bounds for the amplitude, mean, and width.
        """
        return [(0, min(x), 0.001), (np.inf, max(x), max(x) - min(x))]

    @staticmethod
    def p0(x, y):
        """Estimate initial fit parameters from x and y, or return None when no estimate is supplied.

            Args:
                x (Any): Independent-variable values in the fit sample.
                y (Any): Observed values used to initialize the fit.

            Returns:
                list: Initial amplitude, median x value, and x standard deviation.
        """
        return [max(y), np.median(x), np.std(x)]
    
    @staticmethod
    def mu(p0, p1, p2):
        """Return the mean derived from this model's fit parameters.

            Args:
                p0 (Any): Gaussian amplitude (unused in the mean calculation).
                p1 (Any): Gaussian mean.
                p2 (Any): Gaussian width (unused in the mean calculation).

            Returns:
                Any: The Gaussian mean parameter p1.
        """
        return p1

    @staticmethod
    def var(p0, p1, p2):
        """Return the variance estimate computed from this model's fit parameters.

            Args:
                p0 (Any): Gaussian amplitude (unused in the variance calculation).
                p1 (Any): Gaussian mean (unused in the variance calculation).
                p2 (Any): Gaussian width.

            Returns:
                Any: The square of the Gaussian width, p2.
        """
        return p2**2


class student_t(FitFunction):
    """Evaluate a Student's t density with amplitude, location, degrees of freedom, and scale parameters.
    """
    n_params = 4

    def __new__(cls, x, p0, p1, p2, p3) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the Student's t model.
                p0 (Any): Amplitude scale used in the Student's t density.
                p1 (Any): Location parameter.
                p2 (Any): Degrees of freedom.
                p3 (Any): Scale parameter.

            Returns:
                np.ndarray: Student's t model values evaluated at x.
        """
        return cls.func(x, p0, p1, p2, p3)

    @staticmethod
    def func(x, p0, p1, p2, p3):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the Student's t model.
                p0 (Any): Amplitude scale used in the Student's t density.
                p1 (Any): Location parameter.
                p2 (Any): Degrees of freedom.
                p3 (Any): Scale parameter.

            Returns:
                np.ndarray: Student's t model values evaluated at x.
        """
        t = (x - p1)/ p3
        return (p0**2 / p3) * (gamma((p2 + 1)/2) / (np.sqrt(p2 * np.pi) * gamma(p2/2))) * (1 + t**2/p2)**(-(p2 + 1)/2)

    @staticmethod
    def bounds(x, y):
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values used to estimate parameter bounds.

            Returns:
                tuple: Lower and upper bounds for amplitude, location, degrees of freedom, and scale.
        """
        return [(0, min(x), 0.01, 0.001),
                (np.inf, max(x), 10, max(x) - min(x))]

    @staticmethod
    def p0(x, y):
        """Estimate initial fit parameters from x and y, or return None when no estimate is supplied.

            Args:
                x (Any): Independent-variable values in the fit sample.
                y (Any): Observed values used to initialize the fit.

            Returns:
                list: Initial amplitude, median x value, two degrees of freedom, and x standard deviation.
        """
        return [max(y), np.median(x), 2, np.std(x)]

    @staticmethod
    def mu(p0, p1, p2, p3):
        """Return the mean derived from this model's fit parameters.

            Args:
                p0 (Any): Amplitude scale (unused in the returned location).
                p1 (Any): Location parameter returned as the mean.
                p2 (Any): Degrees of freedom (unused in the returned location).
                p3 (Any): Scale parameter (unused in the returned location).

            Returns:
                Any: The location parameter p1.
        """
        return p1

    @staticmethod
    def var(p0, p1, p2, p3):
        """Return p2/(p2 - 2) for p2 > 2, infinity for 1 < p2 <= 2, or NaN otherwise.

            Args:
                p0 (Any): Amplitude scale (not used in the variance calculation).
                p1 (Any): Location parameter (not used in the variance calculation).
                p2 (Any): Degrees of freedom that determine the returned value.
                p3 (Any): Scale parameter (not used in the variance calculation).

            Returns:
                Any: P2/(p2 - 2) for p2 > 2, infinity for 1 < p2 <= 2, otherwise NaN.
        """
        if p2 > 2:
            return p2 / (p2 - 2)
        elif (p2 <= 2) and p2 > 1:
            return np.Inf
        else:
            return np.NaN


class double_gaussian(FitFunction):
    """Evaluate the sum of two Gaussian components with independent amplitudes, means, and widths.
    """
    n_params = 6

    def __new__(cls, x, p0, p1, p2, p3, p4, p5) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate both Gaussian components.
                p0 (Any): Amplitude of the first Gaussian.
                p1 (Any): Mean of the first Gaussian.
                p2 (Any): Width of the first Gaussian.
                p3 (Any): Amplitude of the second Gaussian.
                p4 (Any): Mean of the second Gaussian.
                p5 (Any): Width of the second Gaussian.

            Returns:
                np.ndarray: Sum of the two Gaussian components evaluated at x.
        """
        return cls.func(x, p0, p1, p2, p3, p4, p5)

    @staticmethod
    def func(x, p0, p1, p2, p3, p4, p5):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate both Gaussian components.
                p0 (Any): Amplitude of the first Gaussian.
                p1 (Any): Mean of the first Gaussian.
                p2 (Any): Width of the first Gaussian.
                p3 (Any): Amplitude of the second Gaussian.
                p4 (Any): Mean of the second Gaussian.
                p5 (Any): Width of the second Gaussian.

            Returns:
                np.ndarray: Sum of the two Gaussian components evaluated at x.
        """
        return p0 * np.exp(-(x - p1)**2 / (2 * p2**2)) + p3 * np.exp(-(x - p4)**2 / (2 * p5**2))

    @staticmethod
    def bounds(x, y):
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values used to estimate parameter bounds.

            Returns:
                tuple: Lower and upper bounds for both components' amplitudes, means, and widths.
        """
        return [(0     , min(x), 0.001 , 0     , min(x), 0.001),
                (np.inf, max(x) , max(x) - min(x), np.inf, max(x) , max(x) - min(x))]

    @staticmethod
    def p0(x, y):
        """Estimate initial fit parameters from x and y, or return None when no estimate is supplied.

            Args:
                x (Any): Independent-variable values in the fit sample.
                y (Any): Observed values used to initialize the fit.

            Returns:
                list: Initial amplitudes, means offset around the x mean, and widths for both components.
        """
        return [max(y), np.mean(x) + np.std(x)/2, np.std(x), max(y), np.mean(x) - np.std(x)/2, np.std(x)]

    @staticmethod
    def mu(p0, p1, p2, p3, p4, p5):
        """Return the mean derived from this model's fit parameters.

            Args:
                p0 (Any): First Gaussian amplitude, used as its mixture weight.
                p1 (Any): First Gaussian mean.
                p2 (Any): First Gaussian width (unused in the mean calculation).
                p3 (Any): Second Gaussian amplitude, used as its mixture weight.
                p4 (Any): Second Gaussian mean.
                p5 (Any): Second Gaussian width (unused in the mean calculation).

            Returns:
                Any: Amplitude-weighted mean of the two Gaussian components.
        """
        w_1 = p0 / (p0 + p3)
        w_2 = p3 / (p0 + p3)
        return (w_1 * p1) + (w_2 * p4)

    @staticmethod
    def var(p0, p1, p2, p3, p4, p5):
        """Return the variance estimate computed from this model's fit parameters.

            Args:
                p0 (Any): First Gaussian amplitude, used as its mixture weight.
                p1 (Any): First Gaussian mean.
                p2 (Any): First Gaussian width.
                p3 (Any): Second Gaussian amplitude, used as its mixture weight.
                p4 (Any): Second Gaussian mean.
                p5 (Any): Second Gaussian width.

            Returns:
                Any: Amplitude-weighted variance of the two-component Gaussian mixture.
        """
        w_1 = p0 / (p0 + p3)
        w_2 = p3 / (p0 + p3)

        return w_1 * (p2**2) + w_2 * (p5**2) + (w_1 * (p1**2) + w_2 * (p4**2) - (w_1*p1 + w_2*p4)**2)


class crystal_ball(FitFunction):
    """Evaluate a Gaussian core joined to a power-law tail.
    """
    n_params = 5

    def __new__(cls, x, p0, p1, p2, p3, p4) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the Crystal Ball model.
                p0 (Any): Model amplitude.
                p1 (Any): Gaussian-core mean.
                p2 (Any): Gaussian-core width.
                p3 (Any): Signed transition threshold for the power-law tail.
                p4 (Any): Power-law tail exponent parameter.

            Returns:
                np.ndarray: Crystal Ball model values evaluated at x.
        """
        return cls.func(x, p0, p1, p2, p3, p4)

    @staticmethod
    def func(x, p0, p1, p2, p3, p4):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the Crystal Ball model.
                p0 (Any): Model amplitude.
                p1 (Any): Gaussian-core mean.
                p2 (Any): Gaussian-core width.
                p3 (Any): Signed transition threshold for the power-law tail.
                p4 (Any): Power-law tail exponent parameter.

            Returns:
                np.ndarray: Crystal Ball model values evaluated at x.
        """
        t = (x - p1) / p2

        a_alpha = abs(p3)
        n_alpha = p4 / a_alpha

        A = (n_alpha)**p4 * np.exp(-a_alpha**2 / 2)

        B = n_alpha - a_alpha

        C = (n_alpha) * (1/(p4 - 1)) * np.exp(-a_alpha**2/2)

        D = np.sqrt(np.pi / 2) *(1 + erf(a_alpha/np.sqrt(2)))

        N = 1 / (C + D) # should be 1 / sigma * (C + D), but I dont want to normalise the function

        y = np.where(t > -p3, np.exp(-t**2 / 2), A * common.fpower(B - t, -p4))
        return p0 * N * y

    @staticmethod
    def bounds(x, y):
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values used to estimate parameter bounds.

            Returns:
                tuple: Lower and upper bounds for amplitude, mean, width, tail threshold, and tail exponent.
        """
        return [(0, min(x), 0.001, 1, 2),
                (np.inf, max(x), max(x) - min(x), 3, 10)]

    @staticmethod
    def p0(x,y):
        """Estimate initial fit parameters from x and y, or return None when no estimate is supplied.

            Args:
                x (Any): Independent-variable values in the fit sample.
                y (Any): Observed values used to initialize the fit.

            Returns:
                list: Initial amplitude, x mean, x standard deviation, and tail parameters 1 and 2.
        """
        return [max(y), np.mean(x), np.std(x), 1, 2]

    @staticmethod
    def mu(p0, p1, p2, p3, p4):
        """Return the mean derived from this model's fit parameters.

            Args:
                p0 (Any): Model amplitude (unused in the returned mean).
                p1 (Any): Gaussian-core mean returned by this method.
                p2 (Any): Gaussian-core width (unused in the returned mean).
                p3 (Any): Tail threshold (unused in the returned mean).
                p4 (Any): Tail exponent (unused in the returned mean).

            Returns:
                Any: The Gaussian-core mean parameter p1.
        """
        return p1

    @staticmethod
    def var(p0, p1, p2, p3, p4):
        """Return the fitted width parameter p2 unchanged.

            Args:
                p0 (Any): Model amplitude (unused in the returned width).
                p1 (Any): Gaussian-core mean (unused in the returned width).
                p2 (Any): Gaussian-core width.
                p3 (Any): Tail threshold (unused in the returned width).
                p4 (Any): Tail exponent (unused in the returned width).

            Returns:
                Any: The width parameter p2, returned without squaring.
        """
        return p2


class poly2d(FitFunction):
    """Evaluate a quadratic polynomial in x.
    """
    n_params = 3

    def __new__(cls, x, p0, p1, p2) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the polynomial.
                p0 (Any): Constant coefficient.
                p1 (Any): Linear coefficient.
                p2 (Any): Quadratic coefficient.

            Returns:
                np.ndarray: Quadratic polynomial values evaluated at x.
        """
        return cls.func(x, p0, p1, p2)

    @staticmethod
    def func(x, p0, p1, p2):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the polynomial.
                p0 (Any): Constant coefficient.
                p1 (Any): Linear coefficient.
                p2 (Any): Quadratic coefficient.

            Returns:
                np.ndarray: Quadratic polynomial values evaluated at x.
        """
        return p0 + (p1 * x) + p2 * (x**2)

    @staticmethod
    def p0(x, y):
        """Estimate initial fit parameters from x and y, or return None when no estimate is supplied.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values (unused; this model has no default limits).
        """
        return None

    @staticmethod
    def bounds(x, y):
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values (unused; this model has no default limits).

            Returns:
                tuple: Unbounded lower and upper limits for the three polynomial coefficients.
        """
        return ([-np.inf, -np.inf, -np.inf], [np.inf]*3)


class exp(FitFunction):
    """Evaluate a constant plus an exponential term in x.
    """
    n_params = 3

    def __new__(cls, x, p0, p1, p2) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the exponential model.
                p0 (Any): Constant offset.
                p1 (Any): Exponential amplitude.
                p2 (Any): Exponent coefficient multiplying x.

            Returns:
                np.ndarray: Exponential model values evaluated at x.
        """
        return cls.func(x, p0, p1, p2)

    @staticmethod
    def func(x, p0, p1, p2):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the exponential model.
                p0 (Any): Constant offset.
                p1 (Any): Exponential amplitude.
                p2 (Any): Exponent coefficient multiplying x.

            Returns:
                np.ndarray: Exponential model values evaluated at x.
        """
        return p0 + (p1 * np.exp(p2 * x))

    @staticmethod
    def p0(x, y):
        """Estimate initial fit parameters from x and y, or return None when no estimate is supplied.

            Args:
                x (Any): Independent-variable values in the fit sample (unused by the fixed defaults).
                y (Any): Observed values in the fit sample (unused by the fixed defaults).

            Returns:
                list: Fixed initial parameters [0, 1, 0.001].
        """
        return [0, 1, 1E-3]

    @staticmethod
    def bounds(x, y):
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values (unused; this model has no default limits).

            Returns:
                tuple: Unbounded lower and upper limits for all three parameters.
        """
        return ([-np.inf]*3, [np.inf]*3)


class exp_alt(FitFunction):
    """Evaluate a shifted exponential plus a constant.
    """
    n_params = 3

    def __new__(cls, x, p0, p1, p2) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the shifted exponential.
                p0 (Any): Constant offset.
                p1 (Any): Additive shift applied to x inside the exponent.
                p2 (Any): Exponent coefficient.

            Returns:
                np.ndarray: Shifted exponential model values evaluated at x.
        """
        return cls.func(x, p0, p1, p2)

    @staticmethod
    def func(x, p0, p1, p2):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the shifted exponential.
                p0 (Any): Constant offset.
                p1 (Any): Additive shift applied to x inside the exponent.
                p2 (Any): Exponent coefficient.

            Returns:
                np.ndarray: Shifted exponential model values evaluated at x.
        """
        return p0 + np.exp(p2 * (x + p1))

    @staticmethod
    def p0(x, y):
        """Estimate initial fit parameters from x and y, or return None when no estimate is supplied.

            Args:
                x (Any): Independent-variable values in the fit sample (unused by the fixed defaults).
                y (Any): Observed values in the fit sample (unused by the fixed defaults).

            Returns:
                list: Fixed initial parameters [0, 1, 0.00001].
        """
        return [0, 1, 1E-5]

    @staticmethod
    def bounds(x, y):
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values (unused; this model has no default limits).

            Returns:
                tuple: Unbounded lower and upper limits for all three parameters.
        """
        return ([-np.inf]*3, [np.inf]*3)



class poly3d(FitFunction):
    """Evaluate a cubic polynomial in x.
    """
    n_params = 4

    def __new__(cls, x, p0, p1, p2, p3) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the polynomial.
                p0 (Any): Constant coefficient.
                p1 (Any): Linear coefficient.
                p2 (Any): Quadratic coefficient.
                p3 (Any): Cubic coefficient.

            Returns:
                np.ndarray: Cubic polynomial values evaluated at x.
        """
        return cls.func(x, p0, p1, p2, p3)

    @staticmethod
    def func(x, p0, p1, p2, p3):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the polynomial.
                p0 (Any): Constant coefficient.
                p1 (Any): Linear coefficient.
                p2 (Any): Quadratic coefficient.
                p3 (Any): Cubic coefficient.

            Returns:
                np.ndarray: Cubic polynomial values evaluated at x.
        """
        return p0 + (p1 * x) + (p2 * (x**2)) + (p3 * (x**3))

    @staticmethod
    def p0(x, y):
        """Estimate initial fit parameters from x and y, or return None when no estimate is supplied.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values (unused; this model has no default limits).
        """
        return None

    @staticmethod
    def bounds(x, y):
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values (unused; this model has no default limits).

            Returns:
                tuple: Unbounded lower and upper limits for the four polynomial coefficients.
        """
        return ([-np.inf]*4, [np.inf]*4)


class double_crystal_ball(FitFunction):
    """Evaluate a Gaussian core with separate power-law tails on both sides.
    """
    n_params = 7

    def __new__(cls, x, p0, p1, p2, p3, p4, p5, p6) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the model.
                p0 (Any): Gaussian-core amplitude.
                p1 (Any): Gaussian-core mean.
                p2 (Any): Gaussian-core width.
                p3 (Any): Lower-tail transition threshold.
                p4 (Any): Lower-tail power exponent.
                p5 (Any): Upper-tail transition threshold.
                p6 (Any): Upper-tail power exponent.

            Returns:
                np.ndarray: Double Crystal Ball model values evaluated at x.
        """
        return cls.func(x, p0, p1, p2, p3, p4, p5, p6)

    @staticmethod
    def ExponentNormalisation(alpha, n):
        """Calculate the normalization coefficients for one power-law tail.

            Args:
                alpha (Any): Tail transition parameter used in the power-law coefficients.
                n (Any): Tail exponent used in the power-law coefficients.

            Returns:
                tuple: Coefficients A and B used to evaluate a power-law tail.
        """
        A = (n/abs(alpha))**n * np.exp(-0.5 * abs(alpha)**2)
        B = (n/abs(alpha)) - abs(alpha)
        return A, B

    @staticmethod
    def func(x, p0, p1, p2, p3, p4, p5, p6):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the model.
                p0 (Any): Gaussian-core amplitude.
                p1 (Any): Gaussian-core mean.
                p2 (Any): Gaussian-core width.
                p3 (Any): Lower-tail transition threshold.
                p4 (Any): Lower-tail power exponent.
                p5 (Any): Upper-tail transition threshold.
                p6 (Any): Upper-tail power exponent.

            Returns:
                np.ndarray: Gaussian core with lower and upper power-law tails evaluated at x.
        """
        z = (x - p1) / p2
        A, B = double_crystal_ball.ExponentNormalisation(p3, p4)
        C, D = double_crystal_ball.ExponentNormalisation(p5, p6)
        E1 = A * common.fpower(B - z, -p4)
        E2 = C * common.fpower(D + z, -p6)
        y = np.exp(-0.5*(z**2))
        y = np.where(z < -p3, E1, y)
        y = np.where(z > p5, E2, y)
        return p0 * y

    @staticmethod
    def bounds(x, y):
        # note p4 and p6 can be unbounded, but negative values have weird distribution shapes
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values used to estimate parameter bounds.

            Returns:
                list: Lower and upper bounds for amplitude, mean, width, and both tail parameter pairs.
        """
        return [(0, min(x), 0.001, 0.001, 0, 0.001, 0), (np.inf, max(x), max(x) - min(x), np.inf, 10, np.inf, 10)]

    @staticmethod
    def p0(x, y):
        """Estimate initial fit parameters from x and y, or return None when no estimate is supplied.

            Args:
                x (Any): Independent-variable values in the fit sample.
                y (Any): Observed values used to initialize the fit.

            Returns:
                list: Initial amplitude, median x value, x standard deviation, and unit tail parameters.
        """
        return [max(y), np.median(x), np.std(x), 1, 1, 1, 1]

    @staticmethod
    def mu(p0, p1, p2, p3, p4, p5, p6):
        """Return the mean derived from this model's fit parameters.

            Args:
                p0 (Any): Gaussian-core amplitude (unused in the returned mean).
                p1 (Any): Gaussian-core mean returned by this method.
                p2 (Any): Gaussian-core width (unused in the returned mean).
                p3 (Any): Lower-tail threshold (unused in the returned mean).
                p4 (Any): Lower-tail exponent (unused in the returned mean).
                p5 (Any): Upper-tail threshold (unused in the returned mean).
                p6 (Any): Upper-tail exponent (unused in the returned mean).

            Returns:
                Any: The Gaussian-core mean parameter p1.
        """
        return p1
    
    @staticmethod
    def var(p0, p1, p2, p3, p4, p5, p6):
        """Return the fitted width parameter p2 unchanged.

            Args:
                p0 (Any): Gaussian-core amplitude (unused in the returned width).
                p1 (Any): Gaussian-core mean (unused in the returned width).
                p2 (Any): Gaussian-core width.
                p3 (Any): Lower-tail threshold (unused in the returned width).
                p4 (Any): Lower-tail exponent (unused in the returned width).
                p5 (Any): Upper-tail threshold (unused in the returned width).
                p6 (Any): Upper-tail exponent (unused in the returned width).

            Returns:
                Any: The width parameter p2, returned without squaring.
        """
        return p2


class line(FitFunction):
    """Evaluate a straight line from its slope and intercept parameters.
    """
    n_params = 2

    def __new__(cls, x, p0, p1) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the line.
                p0 (Any): Slope.
                p1 (Any): Vertical intercept.

            Returns:
                np.ndarray: Line values evaluated at x.
        """
        return cls.func(x, p0, p1)

    def func(x, p0, p1):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the line.
                p0 (Any): Slope.
                p1 (Any): Vertical intercept.

            Returns:
                np.ndarray: Line values evaluated at x.
        """
        return (p0 * x) + p1


class asym(FitFunction):
    """Evaluate the reciprocal power-law function defined by the fit parameters.
    """
    n_params = 3

    def __new__(cls, x, p0, p1, p2) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the reciprocal power law.
                p0 (Any): Coefficient multiplying x raised to p2 in the denominator.
                p1 (Any): Additive constant in the denominator.
                p2 (Any): Power applied to x.

            Returns:
                np.ndarray: Reciprocal power-law values evaluated at x.
        """
        return cls.func(x, p0, p1, p2)

    def func(x, p0, p1, p2):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the reciprocal power law.
                p0 (Any): Coefficient multiplying x raised to p2 in the denominator.
                p1 (Any): Additive constant in the denominator.
                p2 (Any): Power applied to x.

            Returns:
                np.ndarray: Reciprocal power-law values evaluated at x.
        """
        return 1/(p0*(x**p2) + p1)


class lognormal_gaussian_exp(FitFunction):
    """Evaluate a weighted sum of log-normal, Gaussian, and exponential components.
    """
    n_params = 8

    def __new__(cls, x, p0, p1, p2, p3, p4, p5, p6, p7) -> np.array:
        """Evaluate the model at x using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the component sum.
                p0 (Any): Log-normal component amplitude.
                p1 (Any): Log-normal scale parameter.
                p2 (Any): Log-normal shape parameter.
                p3 (Any): Gaussian component amplitude.
                p4 (Any): Gaussian location parameter.
                p5 (Any): Gaussian scale parameter.
                p6 (Any): Exponential component amplitude.
                p7 (Any): Exponential decay coefficient.

            Returns:
                np.ndarray: Weighted log-normal, Gaussian, and exponential values evaluated at x.
        """
        return cls.func(x, p0, p1, p2, p3, p4, p5, p6, p7)

    def func(x, p0, p1, p2, p3, p4, p5, p6, p7):
        """Evaluate the model at each x value using the supplied fit parameters.

            Args:
                x (Any): Independent-variable values at which to evaluate the component sum.
                p0 (Any): Log-normal component amplitude.
                p1 (Any): Log-normal scale parameter.
                p2 (Any): Log-normal shape parameter.
                p3 (Any): Gaussian component amplitude.
                p4 (Any): Gaussian location parameter.
                p5 (Any): Gaussian scale parameter.
                p6 (Any): Exponential component amplitude.
                p7 (Any): Exponential decay coefficient.

            Returns:
                np.ndarray: Weighted log-normal, Gaussian, and exponential values evaluated at x.
        """
        lognormal_component = lognorm.pdf(x, s = p2, scale = p1)
        gaussian_component = norm.pdf(x, loc = p4, scale = p5)
        exponential_component = np.exp(-p7 * x)
        return p0 * lognormal_component + p3 * gaussian_component + p6 * exponential_component # Adjust weights as needed

    def bounds(x, y):
        """Return the lower and upper fit-parameter bounds for this model.

            Args:
                x (Any): Independent-variable values covered by the fit.
                y (Any): Observed values used to constrain the component parameters.

            Returns:
                tuple: Lower and upper bounds for the eight component parameters.
        """
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


def rejection_sampling(num : int, low : float, high : float, func : FitFunction, params : dict, scale_param : str = "p0", rng : np.random.default_rng = None) -> np.array:
    """ Performs Rejection sampling for a given function which describes a pdf.

        Args:
            num (int): Number of samples to generate.
            low (float): Minimum range.
            high (float): Maximum range.
            func (FitFunction): Function which desribes a pdf, by default should be a function whose amplitude or scale parameter is p0.
            params (dict): Parameters of the function.
            scale_param (str): Name of amplitude/scale parameter. Defaults to "p0".
            rng (np.random.default_rng): Random generator used to draw candidate values; a default generator is created if omitted.

        Returns:
            np.array: Sampled values.
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


def fit(x : np.array, y_obs : np.array, y_err : np.array, func : FitFunction, method = "trf", maxfev = int(10E4), plot : bool = False, xlabel : str = "", ylabel : str = "", ylim : list = None, plot_style : str = "scatter", title : str = "", plot_range : list = None, return_chi_sqr : bool = False, loc = "upper right") -> tuple[np.array, np.array]:
    """ Implementation of scipy's curve fit, with some constraints, checks to handle nan data and optional plotting.

        Args:
            x (np.array): X data.
            y_obs (np.array): Y data.
            y_err (np.array): Error in y.
            func (FitFunction): Function to fit to data.
            method (str, optional): Fit method, see scipy's documentation. Defaults to "trf".
            maxfev (_type_, optional): Max number of iterations, see scipy's documentation. Defaults to int(10E4).
            plot (bool): Make a plot of the data and the fitted function. Defaults to False.
            xlabel (str): Plot x label. Defaults to "".
            ylabel (str): Plot y label. Defaults to "".
            ylim (list): Y limit of plot. Defaults to None.
            plot_style (str): Plot style of the oberved data points, either "hist" or "scatter". Defaults to "scatter".
            title (str): Plot title.s Defaults to "".
            plot_range (list): Plot range. Defaults to None.
            return_chi_sqr (bool): Additionally returns chi_sqr ndf and p value in a tuple (chi_sqr, ndf, p), defaults to False.
            loc (Any): Matplotlib legend location for the fit-metrics legend when plotting is enabled.

        Returns:
            tuple[np.array, np.array]: Fit parameters and errors in the parameters.
    """

    y_obs = np.array(y_obs, dtype = float) # ensure the input is an array of numpy floats

    mask = ~np.isnan(y_obs) # remove nans

    x = np.array(x[mask], dtype = float)
    y_obs = y_obs[mask]

    if y_err is not None:
        y_err = np.array(y_err, dtype = float)
        y_err = y_err[mask]

    popt, pcov = curve_fit(func.func, x, y_obs, sigma = y_err, maxfev = maxfev, p0 = func.p0(x, y_obs), bounds = func.bounds(x, y_obs), method = method, absolute_sigma = True)
    print(pcov)
    perr = np.sqrt(np.diag(pcov))

    y_pred = func.func(x, *popt) # y values predicted from the fit
    chisqr = abs(np.nansum(((y_obs - y_pred))**2/y_pred)) # abs in case predictions are negative
    ndf = len(y_obs) - len(popt)
    p_value = 1 - chi2.cdf(chisqr, ndf)

    if plot is True:
        #* main plotting
        x_interp = np.linspace(min(x), max(x), 1000)
        plots.plot(x_interp, func.func(x_interp, *popt), new_figure = False, x_scale = "linear", ylabel = ylabel, color = "#1f77b4", zorder = 11, label = "fit", title = title)
        
        p_min = popt - perr #(perr * chisqr/ndf)
        p_max = popt + perr #(perr * chisqr/ndf)

        plt.fill_between(x_interp, func.func(x_interp, *p_max), func.func(x_interp, *p_min), color = "#7f7f7f", alpha = 0.5, zorder = 10, label = "$1\\sigma$ error region")

        if plot_style == "hist":
            marker = ""
            colour = "black"
            label = "observed\nuncertainty"
            widths = (x[1] - x[0])/2
            # plots.plot_hist(x - widths, x - widths, weights = y_obs, color = "#d62728", label = "observed", new_figure = False, range = plot_range)
            plots.plot(x, y_obs, style = "bar", linestyle = "", color = "#d62728", xlabel = xlabel, label = "observed", new_figure = False)

            # automatically calculate cap size to match bin width
            w = x[1] - x[0] # bin width
            ax = plots.plt.gca()
            t = ax.transData.transform # tranform units to distance on page
            ppd = 72 / ax.figure.dpi # points per distance (same point used in font sizes)
            capsize =  0.5 * (ppd * (t([w, 0]) - t([0, 0])))[0] # factor of 0.5 because cap size is in units of half-pints 
        else:
            marker = "x"
            colour = "#d62728"
            label = "observed"
            capsize = 3

        plots.plot(x, y_obs, yerr = y_err, marker = marker, linestyle = "", color = colour, xlabel = xlabel, label = label, new_figure = False, capsize = capsize / 2)
        if ylim:
            plt.ylim(*sorted(ylim))
        # if plot_range:
        #     plt.xlim(plot_range)

        main_legend = plt.legend(loc = "upper left")
        main_legend.set_zorder(12)

        #* add fit metrics to the plot in a second legend
        plt.gca().add_artist(main_legend)
        text = ""
        for j in range(len(popt)):
            text += f"\n$p_{{{j}}}$: ${popt[j]:.2g}\\pm${perr[j]:.2g}"
        text += "\n$\\chi^{2}/ndf$ : " + f"{chisqr/ndf:.2g}, p : " + f"{p_value:.1g}"
        legend = plt.gca().legend(handlelength = 0, labels = [text[1:]], loc = loc, title = common.remove_(func.__name__))
        legend.set_zorder(12)
        for l in legend.legend_handles:
            l.set_visible(False)

    if return_chi_sqr == True:
        return popt, perr, (chisqr, ndf, p_value)
    else:
        return popt, perr


def extract_central_values_df(df : pd.DataFrame, bin_variable : str, variable : str, v_range : list, funcs, data_bins : list, hist_bins : int, log : bool = False, rms_err : bool = False, weights : np.ndarray = None, outer_legend : bool = False, bin_label : str = "bin", bin_units : str = ""):
    """ Estimate a central value in each reco energy bin based on some FitFunction or collection of FitFunctions.

        Args:
            df (pd.DataFrame): Input table containing the binning and fitted variables.
            bin_variable (str): Variable to bin in.
            variable (str): Variable to fit to.
            v_range (list): Variable range.
            funcs (FitFunction): Functions to try fit.
            data_bins (list): Edges defining the bins of bin_variable.
            hist_bins (int): Number of bins for variable histograms.
            log (bool): Verbose printout. Defaults to False.
            rms_err (bool): Whether to estimate the central-value error from the fitted RMS.
            weights (np.ndarray): Optional per-row weights applied when histogramming the fitted variable.
            outer_legend (bool): Whether to place a shared legend outside the individual subplots.
            bin_label (str): Label used to identify the binning variable in plot titles.
            bin_units (str): Units appended to the displayed bin range.

        Returns:
            tuple: Arrays of fitted central values and their estimated errors, one per data bin.
    """
    def print_log(x):
        """Print a message only when logging is enabled.

            Args:
                x (Any): Message or value to print when logging is enabled.
        """
        if log: print(x)

    cv = []
    cv_err = []
    fig_handles = None
    fig_labels = None
    for i in plots.multi_plot(len(data_bins) - 1, orientation = "vertical"):
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
                k, p = ks_2samp(y[y > 0], y_pred[y > 0]) # ignore zero entries in bins to prevent overestimating the test statistic
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
                mean_error = np.sqrt(abs(best_f.var(*best_popt)))/2
            else:
                mean_error = mean - best_f.mu(*(best_popt + best_perr))
            y_pred = best_f.func(x, *best_popt)
            y_pred_interp = best_f.func(x_interp, *best_popt)

            plots.plot(x_interp, y_pred_interp, marker = "", color = "black", new_figure = False, label = "fit")
            plt.axvline(mean, color = "black", linestyle = "--", label = "central value")
        # plots.plot_hist(binned_data[variable], bins = hist_bins, new_figure = False, title = f"{bin_label} : {[data_bins[i], data_bins[i+1]]} {bin_units}", range = [min(v_range), max(v_range)], weights = binned_weights, color = "C0")
        plots.plot(x, y, yerr = np.sqrt(y), new_figure = False, color = "C0", style = "bar", title = f"{bin_label} : {[data_bins[i], data_bins[i+1]]} {bin_units}")

        plt.axvline(np.mean(binned_data[variable]), linestyle = "--", color = "C1", label = "mean")

        if not fig_handles: fig_handles, fig_labels = plt.gca().get_legend_handles_labels()

        if best_popt is not None:
            text = ""
            for j in range(len(best_popt)):
                text += f"\np{j}: ${best_popt[j]:.2f}\\pm${best_perr[j]:.2f}"
            text += f"\nks : {k_best:.2f}, p : {p_best:.2f}"
            legend = plt.gca().legend(handlelength = 0, labels = [text[1:]], title = best_f.__name__.replace("_", " "))
            for l in legend.legend_handles:
                l.set_visible(False)

        cv.append(mean)
        cv_err.append(abs(mean_error) if mean_error is not None else mean_error)
    
    if outer_legend:
        plt.gcf().legend(fig_handles, fig_labels, loc = "lower right", ncols = 3)
    plt.gcf().supxlabel(variable.replace("_", " ").capitalize())
    plt.tight_layout()
    return np.array(cv), np.array(cv_err)
