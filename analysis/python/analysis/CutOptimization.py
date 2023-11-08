"""
Created on: 12/07/2022 18:46

Author: Shyam Bhuller

Description: Code to optimize a a set of cuts made on data.
"""
from abc import ABC, abstractmethod
from enum import Enum

import awkward as ak
import numpy as np
from tabulate import tabulate

from python.analysis import Plots


class Operator(str, Enum):
    """ Enum for types of cuts.
    """
    LESS = "<"
    GREATER = ">"
    EQUAL = "=="
    NOT_EQUAL = "!="


class Cuts():
    """ Struct for defining cuts.
    Attributes:
        variable (str): variable to cut
        operator (Operator): condition of the cut
        value (Any): cut value

    Methods:
        cut: cuts data provided
    """
    def __init__(self, variable : str, operator : Operator, value) -> None:
        self.variable = variable
        self.operator = operator
        self.value = value
        pass

    def cut(self, data):
        """ Cuts data.

        Args:
            data (ak.Array): data to cut

        Returns:
            ak.Array: cut data
        """
        if self.operator == Operator.LESS:
            return data < self.value
        elif self.operator == Operator.GREATER:
            return data > self.value
        elif self.operator == Operator.EQUAL:
            return data == self.value
        elif self.operator == Operator.NOT_EQUAL:
            return data != self.value
        else:
            Exception(f"{self.operator} is not a valid Operator")

    def mask(self):
        """ makes mask of cuts.
        Returns:
            ak.Array: mask
        """
        if self.operator == Operator.LESS:
            return self.variable < self.value
        elif self.operator == Operator.GREATER:
            return self.variable > self.value
        elif self.operator == Operator.EQUAL:
            return self.variable == self.value
        elif self.operator == Operator.NOT_EQUAL:
            return self.variable != self.value
        else:
            Exception(f"{self.operator} is not a valid Operator")

    def __repr__(self) -> str:
        """ Pretty printing of cut.
        """
        return f"{self.variable} {self.operator.value} {self.value}"


def MaxSRootBRatio(cuts : np.array, metrics : dict) -> float:
    """ Select a cut which maximises signal/root background ratio.

    Args:
        cuts (np.array): list of cuts
        metrics (dict): metrics used to define the criteria

    Returns:
        float: Cut that matches criteria
    """
    return cuts[np.argmax(metrics["s/rootb"])]


def Fit(x : np.array, y : np.array):
    """ Least squares fit of 1D data.

    Args:
        x (np.array): x data
        y (np.array): y data

    Returns:
        tuple: polynomial coefficients and normalization used when fitting
    """
    #* calculate for normalization
    mx = np.mean(x)
    sx = np.std(x)
    my = np.mean(y)
    sy = np.std(y)
    coefficients = np.polyfit( (x-mx)/sx, (y-my)/sy, 3)[::-1] # reverse list as coefficients are returned in descending order of powers
    return coefficients, mx, my, sx, sy


def EqualSRootB(cuts : np.array, metrics : dict, target : float) -> float:
    """ Select a cut at specific signal/root background ratio.
    Args:
        cuts (np.array): list of cuts
        metrics (dict): metrics used to define the criteria
        target (float) : value of s/rootb to match evaluate cut at.

    Returns:
        float: Cut that matches criteria
    """
    c, mx, my, sx, sy = Fit(metrics["s/rootb"], cuts)
    value = 0
    for i in range(len(c)):
        value += c[i] * ((target-mx)/sx)**i
    return abs((value * sy) + my)


def EqualSignalBackgroundEfficiency(cuts : np.array, metrics : dict) -> float:
    """ Select a cut where signal efficiency and background rejection are closest or equal.
        TODO fit a function to se and br as a fuction of cut and then get exact value at which they are the same
    Args:
        cuts (np.array): list of cuts
        metrics (dict): metrics used to define the criteria

    Returns:
        float: Cut that matches criteria
    """
    return cuts[np.argmin(np.abs(metrics["signal efficiency"] - metrics["background rejection"]))]


def MaxPurity(cuts : np.array, metrics : dict) -> float:
    """ Select cut at the highest purity.

    Args:
        cuts (np.array): list of cuts
        metrics (dict): metrics used to define the criteria

    Returns:
        float: Cut that matches criteria
    """
    return cuts[np.argmax(metrics["purity"])]


def SetPurity(cuts : np.array, metrics : dict, target : float) -> float:
    """ Select cut at specific purity.

    Args:
        cuts (np.array): list of cuts
        metrics (dict): metrics used to define the criteria
        target (float): value of purity to evaluate cut at

    Returns:
        float: Cut that matches criteria
    """
    c, mx, my, sx, sy = Fit(metrics["purity"], cuts)
    value = 0
    for i in range(len(c)):
        value += c[i] * ((target-mx)/sx)**i
    return abs((value * sy) + my)


def Metrics(nEntires):
    return {
        "s/b" : np.empty(shape=nEntires),
        "s/rootb" : np.empty(shape=nEntires),
        "signal efficiency" : np.empty(shape=nEntires),
        "background rejection" : np.empty(shape=nEntires),
        "purity" : np.empty(shape=nEntires),
    }


class CutOptimization(ABC):
    """ Class which takes a data object and will produce a set of cuts which finds a
        value which optimizes some working point e.g. signal/background
    Attributes:
        quantities: a class which satifies the following criteria:
            - contains attributes which resemble data
            - contains a selectionVariables attribute which lists which attributes are the selection variables
            - contains boolean masks which defines a signal and background sample
        initial_cuts (list[Cut]): list of initial cuts to try
        debug (bool): enable for verbose output
    Abstract Methods:
        Optimize: method which defines how you want to optimize the cuts see examples below
    Methods:
        __init__:
        PrintSignalMetrics: Print various metrics for data.
        InitialChecks: Calculate initial signal and bacgkround counts and print metrics.
        CreateMask: Create a mask of events which pass the cuts on each variable, or skip one.
        EvaluateCuts: Calculate signal metrics after applying a set of cuts.
        NMinus1Study: optimizes nth cut value.


    """
    def __init__(self, _quantities, _initial_cuts : list, _debug : bool = False):
        self.quantities = _quantities
        self.initial_cuts = _initial_cuts
        self.debug = _debug
        pass

    @abstractmethod
    def Optimize(self):
        self.InitialChecks()
        pass


    def PrintSignalMetrics(self, signal : ak.Array, background : ak.Array, initial_signal : ak.Array = [], initial_background : ak.Array = []):
        """ Print various metrics for data.

        Args:
            signal (ak.Array): signal mask
            background (ak.Array): background mask
            initial_signal (ak.Array, optional): initial signal mask
            initial_background (ak.Array, optional): initial background mask

        Returns:
            tuple: calculated metrics
        """
        s = ak.count(signal)
        b = ak.count(background)
        if b == 0:
            sb = -1
            srb = -1
        else:
            sb = s/b
            srb = s/np.sqrt(b)
        if s != 0 or b != 0:
            p = s/(s+b)
        else:
            p = -1
        if self.debug: print(f"signal: {s} | background: {b} | s/b {sb:.3f} | s/rootb {srb:.3f} | purity: {p:.3f}")
        if ak.count(initial_signal) > 0 and ak.count(initial_background) > 0:
            si = ak.count(initial_signal)
            bi = ak.count(initial_background)
            se = s/si
            be = b/bi
            e = (s+b) / (si+bi)
            return s, b, sb, srb, p, se, be, e
        else:
            return s, b, sb, srb, p


    def InitialChecks(self):
        """ Calculate initial signal and bacgkround counts and print metrics.

        Args:
            quantities (any): quantities class to look at
            initial_cuts (list): list of initial cuts

        Returns:
            tuple: Initial metrics.
        """
        if len(self.initial_cuts) != len(self.quantities.selectionVariables):
            raise Exception("lenth of initial_cuts must be equal to number of cut variables")

        initial_signal = ak.flatten(self.quantities.signal)[ak.flatten(self.quantities.signal)]
        initial_background = ak.flatten(self.quantities.background)[ak.flatten(self.quantities.background)]
        return self.PrintSignalMetrics(initial_signal, initial_background)


    def CreateMask(self, cuts : list, skip : str = None) -> ak.Array:
        """ Create a mask of events which pass the cuts on each variable, or skip one.

        Args:
            quantities (any): quantities class
            cuts (list): list of Cuts
            skip (int, optional): index to skip. Defaults to -1 (skip none).

        Returns:
            ak.Array: mask
        """
        mask = [] # mask of entries passing the cuts
        for c in cuts:
            if c.variable == skip: continue  # dont cut on nth variable
            if (len(mask) == 0):
                mask = c.cut(ak.flatten(getattr(self.quantities, c.variable)))
            else:
                mask = np.logical_and(mask, c.cut(ak.flatten(getattr(self.quantities, c.variable))))
        if self.debug: print(f"number of PFOs after cuts: {ak.count(mask[mask])}")
        return mask


    def EvaluateCuts(self, cuts : list):
        """ Calculate signal metrics after applying a set of cuts.

        Args:
            cuts (list): cuts to apply

        Returns:
            tuple: signal metrics
        """
        mask = self.CreateMask(cuts)
        si = ak.flatten(self.quantities.signal)
        bi = ak.flatten(self.quantities.background)
        s = si[mask]
        b = bi[mask]
        return self.PrintSignalMetrics(s[s], b[b], si[si], bi[bi])


    def NMinus1Study(self, icuts : list, n : int, stepSize : int, criteria = MaxSRootBRatio, args : list = []):
        """ Finds the best cut for the nth variable after applying the existing cuts to the n-1th variables.

        Args:
            icuts (list): initial cuts
            n (int): nth variable
            stepSize (int): number of new cuts to try
            criteria (_type_, optional): which working point to optimize. Defaults to MaxSRootBRatio.
            args (list, optional): arguements for criteria. Defaults to [].

        Returns:
            any: best cut value
        """
        mcuts = list(icuts)
        nm1 = icuts[n]

        #* first apply all but one cut
        mask = self.CreateMask(mcuts, nm1.variable)

        #* get data that passes n-1 cuts for nth variable 
        sig = ak.flatten(self.quantities.signal)[mask]
        bkg = ak.flatten(self.quantities.background)[mask]
        ns_i = ak.count(sig[sig])
        nb_i = ak.count(bkg[bkg])
        var = ak.flatten(getattr(self.quantities, nm1.variable))[mask]

        if len(var) == 0:
            print(f"cuts {mcuts}, results in no events being selected!")
            return icuts[n].value
        if ns_i == 0:
            print(f"cuts {mcuts}, results in no signal events being selected, doing nothing...")
            return icuts[n].value
        if nb_i == 0:
            print(f"cuts {mcuts}, results in no background events being selected, doing nothing...")
            return icuts[n].value

        #* cut variable at various points, calculate s/b, s/rootb
        cuts = np.linspace(ak.min(var), ak.max(var), stepSize)
        cut_metrics = Metrics(len(cuts))
        for i in range(len(cuts)):
            mask = var < cuts[i]
            s = sig[mask][sig[mask]]
            b = bkg[mask][bkg[mask]]
            m = self.PrintSignalMetrics(s, b)
            cut_metrics["signal efficiency"][i] = m[0] / ns_i
            cut_metrics["background rejection"][i] = 1 - (m[1] / nb_i)
            cut_metrics["s/b"][i] = m[2]
            cut_metrics["s/rootb"][i] = m[3]
            cut_metrics["purity"][i] = m[4]
            string = f"signal: {m[0]} | background: {m[1]} |"
            if self.debug:
                for key, metric in cut_metrics.items():
                    string += f"{key} : {metric[i]:.3f} | "
                print(string[:-1])
        # if self.debug:
        #     Plots.PlotHistComparison([var[bkg], var[sig]], labels=["background", "signal"], bins=25, xlabel=nm1.variable, density=False)
        #     Plots.Plot(cuts[cut_metrics["s/b"]>0], cut_metrics["s/b"][cut_metrics["s/b"]>0], nm1.variable, "$s/b$")
        #     Plots.Plot(cuts[cut_metrics["s/rootb"]>0], cut_metrics["s/rootb"][cut_metrics["s/rootb"]>0], nm1.variable, "$s/\\sqrt{b}$")
        #     Plots.Plot(cuts, cut_metrics["signal efficiency"], nm1.variable, label="signal efficiency")
        #     Plots.Plot(cuts, cut_metrics["background rejection"], nm1.variable, label="background rejection", newFigure=False)
        #     Plots.Plot(cut_metrics["signal efficiency"], cut_metrics["background rejection"], "signal efficiency", "background rejection")
        #* criteria to pick a starting shower should be something like, highest s/b or se == br
        best_cut = criteria(cuts, cut_metrics, *args)
        if self.debug: print(f"best cut for {nm1.variable} is {best_cut}")
        
        return best_cut


class OptimizeSingleCut(CutOptimization):
    def Optimize(self : CutOptimization, stepSize : int = 10, criteria = MaxSRootBRatio, args = []):
        """ Apply n-1 cuts and iteratively optimize a single cut. 

        Args:
            stepSize (int, optional): number of uniformly distributed cuts to try. Defaults to 10.
            plot (bool, optional): make n-1 plots (for debugging). Defaults to False.
            criteria (any, optional): function to decide best cut value. Defaults to MaxSRootBRatio.
            args (list, optional): arguments for criteria. Defaults to [].

        Returns:
            tuple: final set of cuts and metrics
        """
        final_cuts = []
        final_metrics = []
        nm1 = 0
        while nm1 < len(self.initial_cuts):
            nm1_cut = self.initial_cuts[nm1]
            if self.debug: print(f"look at : {nm1_cut}")
            best_cut = self.NMinus1Study(list(self.initial_cuts), nm1, stepSize, criteria, args)

            modified_cuts = list(self.initial_cuts)
            modified_cuts[nm1].value = best_cut # this is the new best cut for this variable
            final_cuts.append([c.value for c in modified_cuts])
            final_metrics.append(list(self.EvaluateCuts(modified_cuts)))
            nm1 += 1

        return final_cuts, final_metrics


class OptimizeAllCuts(CutOptimization):
    def Optimize(self : CutOptimization, startPoint : int = 0, stepSize : int = 10, criteria = MaxSRootBRatio, args = []):
        """ Optimise cuts for variables by scanning for an optimal set of cuts which all satisfy a certain criteria.
            Can be run recusrively to converge to an "optimal" set of cuts. 

        Args:
            startPoint (int, optional): which quantity to start with, number matches index of initial_cuts. Defaults to 0.
            stepSize (int, optional): when scanning n-1 plots how many values should be tried. Defaults to 10.
            plotFinal (bool, optional): plot quantities with the final cuts applied. Defaults to False.

        Returns:
            list : list of final cuts.
        """
        
        final_cuts = list(self.initial_cuts)
        nm1 = -1
        while nm1 != startPoint:
            if nm1 == -1: nm1 = startPoint # define the variable to produce n-1 plot

            best_cut = self.NMinus1Study(final_cuts, nm1, stepSize, criteria, args)

            final_cuts[nm1].value = best_cut # this is the new best cut for this variable
            nm1 = (nm1 + 1)%len(final_cuts)
            print(f"next look at : {final_cuts[nm1]}")

        print(f"final cuts are:")
        print(tabulate([self.quantities.selectionVariables] + [c.value for c in final_cuts], tablefmt="fancy_grid"))
        #* cut one final time to look at the results
        final_metrics = list(self.EvaluateCuts(final_cuts))

        return final_cuts, final_metrics
