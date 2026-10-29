import os

import numpy as np
import uproot

from scipy.interpolate import interp1d

from python.analysis import common
from python.analysis.plotting import plots

GEANT_XS = os.environ["PYTHONPATH"] + "/data/g4_xs_pi_KE_100.root"
# GEANT_XS = os.environ["PYTHONPATH"] + "/data/g4_xs.root"

class GeantCrossSections:
    """ Object for accessing Geant 4 cross sections from the root file generated with Geant4Reweight tools.
    """
    labels = {"abs_KE;1" : "absorption", "inel_KE;1" : "quasielastic", "cex_KE;1" : "charge_exchange", "dcex_KE;1" : "double_charge_exchange", "prod_KE;1" : "pion_production", "total_inel_KE;1" : "total_inelastic"}

    def __init__(self, file : str = GEANT_XS, energy_range : list = None, n_cascades : int = None) -> None:
        with uproot.open(file) as ufile: # open root file
            self.KE = ufile["abs_KE;1"].all_members["fX"] # load kinetic energy from one channel (shared for all cross section channels)

            if energy_range:
                self.KE = self.KE[(self.KE <= max(energy_range)) & (self.KE >= min(energy_range))]

            for k in ufile.keys():
                if "KE" in k:
                    g = ufile[k]
                    if energy_range:
                        mask = (g.all_members["fX"] <= max(energy_range)) & (g.all_members["fX"] >= min(energy_range))
                        xs = g.all_members["fY"][mask]
                    else:
                        xs = g.all_members["fY"]
                    s = "_frac" if "frac" in k else "" 
                    setattr(self, self.labels[k.replace("_frac", "")] + s, xs[0:len(self.KE)]) # assign class variables for each cross section channel

            self.exclusive_processes = list(self.labels.values())
            self.exclusive_processes.remove("total_inelastic")
            self.n_cascades = n_cascades
        pass


    def stat_error(self, xs : str) -> np.ndarray:
        """ Statisitical error of the simulation, done using binomial uncertainties. Only works if n_cascades is known.

        Args:
            xs (str): cross section process

        Returns:
            np.ndarray: statistical error
        """
        if (self.n_cascades is None) or (not hasattr(self, xs + "_frac")):
            return 0 * getattr(self, xs)
        else:
            return getattr(self, xs) * np.sqrt(getattr(self, xs + "_frac") / self.n_cascades)


    def __plot_all(self, title : str = None):
        """ Plot all cross section channels.
        """
        for k in self.labels.values():
            plots.plot(self.KE, getattr(self, k), label = common.remove_(k), new_figure = False, xlabel = "$KE$ (MeV)", ylabel = "$\\sigma$ (mb)", title = title)
            # plots.plt.fill_between(self.KE, getattr(self, k) - self.stat_error(k), getattr(self, k) + self.stat_error(k), color = plots.plt.gca()._get_lines.get_next_color())


    def plot(self, xs : str, color : str = None, label : str = None, title : str = None, simplified_pion_production : bool = False):
        """ Plot cross sections. To be used in conjunction with other plots for comparisons.

        Args:
            xs (str): cross section channel to plot, if given all, will plot all cross section channels
            color (str, optional): colour of single plot. Defaults to None.
            label (str, optional): label of plot, if None, the channel name is used. Defaults to None.
            title (str, optional): title of plot, set to the channel name if label is provided. Defaults to None.
        """
        if xs == "all":
            self.__plot_all(title = title)
        else:
            if label is None:
                label = common.remove_(xs)
            else:
                if title is None:
                    title = common.remove_(xs).capitalize()
            if xs == "single_pion_production":
                y = self.quasielastic + self.double_charge_exchange
            elif xs == "pion_production" and simplified_pion_production is True:
                y = self.quasielastic + self.double_charge_exchange + self.pion_production
            else:
                y = getattr(self, xs)
            plots.plot(self.KE, y, label = label, title = title, new_figure = False, xlabel = "$KE$ (MeV)", ylabel = "$\\sigma$  (mb)", color = color)
            # plots.plt.fill_between(self.KE, getattr(self, xs) - self.stat_error(xs), getattr(self, xs) + self.stat_error(xs), color = plots.plt.gca()._get_lines.get_next_color())


    def get_interpolated_curve(self, process : str) -> interp1d:
        """ returns interpolated cross section curve as function of KE.

        Args:
            process (str): cross section process

        Returns:
            interp1d: _description_
        """
        if process == "single_pion_production":
            sigma = self.quasielastic + self.double_charge_exchange
        else:
            sigma = getattr(self, process)
        return interp1d(self.KE, sigma, fill_value = "extrapolate")
