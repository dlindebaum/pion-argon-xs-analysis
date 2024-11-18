"""
Created on: 02/06/2023 10:37

Author: Shyam Bhuller

Description: Library for code used in the cross section analysis. Refer to the README to see which apps correspond to the cross section analysis.
"""
from collections import namedtuple

import awkward as ak
import numpy as np

from particle import Particle

from python.analysis import EnergyTools as etools
from python.analysis.Utils import *

class Slices:
    """
    Describes slices of a variable, equivilant to a list of bin edges
    but has more functionality. 

    Slice : a Single slice, has properies number (integer) and
        "position" in the parameter space of the value you want to
        slice up. 
    """
    Slice = namedtuple("Slice", "num pos")
    def __init__(self, width, _min, _max, reversed : bool = False):
        self.width = width
        self.min = _min
        self.max = _max
        self.reversed = reversed
        
        self.max_num = max(self.num)
        self.min_num = min(self.num)
        self.max_pos = max(self.pos)
        self.min_pos = min(self.pos)


    def __conversion__(self, x):
        """ convert a value to its slice number.

        Args:
            x: value, array of float

        Returns:
            slice: slice number/s
        """
        if self.reversed:
            numerator = self.max - x
        else:
            numerator = x
        c = np.floor(numerator // self.width)
        if hasattr(c, "__iter__"):
            return ak.values_astype(c, int)
        else:
            return int(c)


    def __create_slice__(self, i) -> Slice:
        """ using the slice number, create the Slice object.

        Args:
            i (int): slice number/s

        Returns:
            Slice: slice
        """
        if self.reversed:
            p = self.max - i * self.width
        else:
            p = i * self.width
        return self.Slice(i, p)


    def __call__(self, x):
        """ get the slice number for a set of values

        Args:
            x: values

        Returns:
            array or int: slice numbers
        """
        return self.__create_slice__(self.__conversion__(x))


    def __getitem__(self, i : int) -> Slice:
        """ Creates slices from slice numbers.

        Args:
            i (int): slice number

        Raises:
            StopIteration

        Returns:
            Slice: ith slice
        """
        if i * self.width > (self.max - self.min):
            raise StopIteration
        else:
            if self.reversed:
                return self.__create_slice__(i + self.__conversion__(self.max))
            else:
                return self.__create_slice__(i + self.__conversion__(self.min))

    @property
    def num(self) -> np.ndarray:
        """ Return all slice numbers.

        Returns:
            np.ndarray: slice numbers
        """
        return np.array([ s.num for s in self], dtype = int)

    @property
    def pos(self) -> np.ndarray:
        """ Return all slice positions.

        Returns:
            np.ndarray: slice positions
        """
        return np.array([ s.pos for s in self])

    @property
    def pos_overflow(self):
        return np.insert(self.pos, 0, self.max_pos + self.width)

    @property
    def pos_bins(self):
        return np.sort(self.pos_overflow)


    def pos_to_num(self, pos):
        """ Convert slice positions to numbers

        Args:
            pos: positions

        Returns:
            array or int: slice numbers
        """
        slice_num = self.__conversion__(pos)
        if hasattr(pos, "__iter__"):
            slice_num = ak.where(slice_num > max(self.num), max(self.num), slice_num)
            slice_num = ak.where(slice_num < 0, min(self.num), slice_num)
        else:
            if pos > max(self.pos): 
                slice_num = max(self.num) # above range go into overflow bin
            if pos < 0:
                slice_num = min(self.num) # below range go into the underflow bin
        return slice_num

class ThinSlice:
    """Methods for implementing the thin slice measurement method."""
    @staticmethod
    def CountingExperiment(
            endPos : ak.Array,
            channel : ak.Array,
            slices : Slices) -> tuple[ak.Array, ak.Array]:
        """ Creates the interacting and incident histograms.

        Args:
            endPos (ak.Array): end position of particle or "interaction
                vertex".
            channel (ak.Array): mask which selects particles which
                interact in the channel you are interested in.
            slices (Slices): spatial slices.

        Returns:
            tuple[ak.Array, ak.Array]: n_interact and n_incident histograms
        """
        end_slice_pos = slices.pos_to_num(endPos)
        slice_nums = slices.num

        n_interact = np.histogram(end_slice_pos[channel], slice_nums)[0]

        total_interact = np.histogram(end_slice_pos, slice_nums)[0]
        n_incident = np.cumsum(total_interact[::-1])[::-1]
        return n_interact, n_incident

    @staticmethod
    def MeanSliceEnergy(
            energy : ak.Array,
            endPos : ak.Array,
            slices : Slices) -> tuple[ak.Array, ak.Array]:
        """ Compute the average energy in a spatial slice.

        Args:
            energy (ak.Array): particle energies over its lifetime in the tpc
            endPos (ak.Array): end position of particle or "interaction vertex"
            slices (Slices): spatial slices

        Returns:
            tuple[ak.Array, ak.Array]: means slice energy, error in the mean slice energy
        """
        beam_traj_slice = slices.pos_to_num(endPos)
        slice_nums = slices.num
        
        # histogram of positions will give the counts
        counts = np.histogram(ak.ravel(beam_traj_slice), slice_nums)[0]

        weights = ak.ravel(np.nan_to_num(energy, 0))

        # total energy in each bin if you weight by energy
        sum_energy = np.histogram(
            ak.ravel(beam_traj_slice), slice_nums, weights = weights)[0]
        sum_energy_sqr = np.histogram(
            ak.ravel(beam_traj_slice), slice_nums, weights = weights**2)[0] # same as above

        mean_energy = sum_energy / counts

        std_energy = np.divide(sum_energy_sqr, counts) - mean_energy**2
        error_mean_energy = np.sqrt(np.divide(std_energy, counts))

        return mean_energy, error_mean_energy

    @staticmethod 
    def TotalCrossSection(
            n_incident : np.ndarray,
            n_interact : np.ndarray,
            slice_width : float) -> tuple[np.ndarray, np.ndarray]:
        """ Returns cross section in mb.

        Args:
            n_incident (np.ndarray): incident histogram
            n_interact (np.ndarray): interacting histogram
            slice_width (float): spatial width of thin slice

        Returns:
            tuple[np.ndarray, np.ndarray]: cross section, statistical uncertainty
        """
        xs = np.log(n_incident / (n_incident - n_interact)) # calculate a dimensionless cross section

        v_incident = n_incident # poisson uncertainty
        v_interact = n_interact*(1- (n_interact/n_incident)) # binomial uncertainty

        xs_e = ((1/n_incident) * (1/(n_incident - n_interact))
                * (n_interact**2 * v_incident + n_incident**2 * v_interact)**0.5)

        NA = 6.02214076e23
        factor = (10**27 * etools.BetheBloch.A
                  / (etools.BetheBloch.rho * NA * slice_width))

        return factor * xs, abs(factor * xs_e)


    def CrossSection(
            n_int_exclusive : np.ndarray,
            n_int_inclusive : np.ndarray,
            n_inc_inclusive : np.ndarray,
            slice_width : float) -> tuple[np.ndarray, np.ndarray]:
        """ Cross section of exclusive process.

        Args:
            n_int_exclusive (np.ndarray): exclusive interactions
            n_int_inclusive (np.ndarray): interactions
            n_inc_inclusive (np.ndarray): incident counts
            slice_width (float): slice width

        Returns:
            tuple[np.ndarray, np.ndarray]: cross section and error
        """
        NA = 6.02214076e23
        factor = (10**27 * etools.BetheBloch.A
                  / (etools.BetheBloch.rho * NA * slice_width))

        n_interact_ratio = nandiv(n_int_exclusive, n_int_inclusive)
        n_survived_inclusive = n_inc_inclusive - n_int_inclusive

        var_inc_inclusive = n_inc_inclusive # poisson variance
        # binomial uncertainties
        var_int_inclusive = (
            n_int_inclusive * (1 - nandiv(n_int_inclusive, n_inc_inclusive)))
        var_int_exclusive = (
            n_int_exclusive * (1 - nandiv(n_int_exclusive, n_inc_inclusive)))

        xs = (
            factor * n_interact_ratio
            * np.log(nandiv(n_inc_inclusive, n_inc_inclusive - n_int_inclusive)))

        diff_n_int_exclusive = nandiv(xs, n_int_exclusive)
        diff_n_inc_inclusive = (
            factor * n_interact_ratio
            * (nandiv(1, n_inc_inclusive) - nandiv(1, n_survived_inclusive)))
        diff_n_int_inclusive = (
            factor * n_interact_ratio
            * nandiv(1, n_survived_inclusive) - nandiv(xs, n_int_inclusive))

        xs_err = ((diff_n_int_exclusive**2 * var_int_exclusive)
                  + (diff_n_inc_inclusive**2 * var_inc_inclusive)
                  + (diff_n_int_inclusive**2 * var_int_inclusive))**0.5
        return xs, xs_err


class EnergySlice:
    """ Methods for implementing the energy slice measurement method.
    """
    @staticmethod
    def TrunacteSlices(slice_array : ak.Array, energy_slices : Slices) -> ak.Array:
        """
        Method for truncating slice numbers due to the fact energy
        slices should be in reverse order vs kinetic energy.

        Args:
            slice_array (ak.Array): slices to truncate
            energy_slices (Slices): energy slices

        Returns:
            ak.Array: truncated slices
        """
        # set minimum to -1 (underflow i.e. energy > plim)
        slice_array = ak.where(slice_array < 0, -1, slice_array)
        # set maxslice (overflow i.e. energy < dE)
        slice_array = ak.where(slice_array > energy_slices.max_num,
                               energy_slices.max_num, slice_array)
        return slice_array


    @staticmethod
    def NIncident(n_initial : np.ndarray, n_end : np.ndarray) -> np.ndarray:
        """ Calculate number of incident particles

        Args:
            n_initial (np.ndarray): initial particle counts
            n_end (np.ndarray): interaction counts

        Returns:
            np.ndarray: incident counts
        """
        n_survived_all = np.cumsum(n_initial - n_end)
        n_incident = n_survived_all + n_end
        return n_incident

    @staticmethod
    def SliceNumbers(int_energy : ak.Array, init_energy : ak.Array, outside_tpc : ak.Array, energy_slices : Slices) -> tuple[np.ndarray, np.ndarray]:
        """ Convert energies from physical units to slice numbers.

        Args:
            int_energy (ak.Array): interaction energy
            init_energy (ak.Array): initial energy
            outside_tpc (ak.Array): mask of particles which interact outside the fiducial volume
            energy_slices (Slices): energy slices

        Returns:
            tuple[np.ndarray, np.ndarray]: initial slice numbers and interacitng slice numbers
        """
        init_slice = energy_slices(init_energy).num + 1 # equivilant to ceil
        int_slice = energy_slices(int_energy).num

        init_slice = EnergySlice.TrunacteSlices(init_slice, energy_slices)
        int_slice = EnergySlice.TrunacteSlices(int_slice, energy_slices)

        # removes instances where the particle incident energy and interacting energy are in the same slice (Yinrui calls this an incomplete slice)
        # i.e. this happens if the particle interacting in its first slice, must be an artifact of the energy slicing but not sure why.
        bad_slices = (int_slice < init_slice) | outside_tpc
        init_slice = ak.where(bad_slices, -1, init_slice)
        int_slice = ak.where(bad_slices, -1, int_slice)
        return init_slice, int_slice

    @staticmethod
    def CountingExperiment(
            int_energy : ak.Array,
            init_energy : ak.Array,
            outside_tpc : ak.Array,
            process : ak.Array,
            energy_slices : Slices,
            interact_only : bool = False,
            weights : np.ndarray = None,
            return_int_binning : bool = False) -> tuple[np.ndarray]:
        """ Creates the interacting and incident histograms.

        Args:
            int_energy (ak.Array): interacting enrgy
            init_energy (ak.Array): initial energy
            outside_tpc (ak.Array): mask of particles which interact
                outside the fiducial volume
            process (ak.Array, optional): mask of events for exclusive
                interactions. Ignored if return_int_binning is True.
            energy_slices (Slices): energy slices
            interact_only (bool, optional): only return exclusive
                interaction histogram. Defaults to False.
            weights (np.ndarray, optional): event weights. Defaults to
                None.
            return_int_binning (bool, optional): If true, replace the
                exclusive interaction histogram with an array of bin
                indicies for each event, indicating which interaction
                bin the event falls into. Defaults to False.

        Returns:
            np.ndarray | tuple[np.ndarray]: exclusive interaction
                histogram and/or initial histogram, incident histogram
                and interaction histogram 
        """
        init_slice, int_slice = EnergySlice.SliceNumbers(
            int_energy, init_energy, outside_tpc, energy_slices)

        slice_bins = np.arange(-1 - 0.5, energy_slices.max_num + 1.5)

        exclusive_weights = weights[process] if weights is not None else None
        if return_int_binning:
            exclusive_return = np.digitize(np.array(int_slice), slice_bins[1:-1])
        else:
            exclusive_return = np.histogram(
                np.array(int_slice[process]), slice_bins,
                weights = exclusive_weights)[0]
        if interact_only == False:
            n_initial = np.histogram(
                np.array(init_slice), slice_bins, weights = weights)[0]
            n_interact_inelastic = np.histogram(
                np.array(int_slice), slice_bins, weights = weights)[0]

            n_incident = EnergySlice.NIncident(n_initial, n_interact_inelastic)
            return n_initial, n_interact_inelastic, exclusive_return, n_incident
        else:
            return exclusive_return

    @staticmethod
    def CountingExperimentOld(int_energy : ak.Array, ff_energy : ak.Array, outside_tpc : ak.Array, channel : ak.Array, energy_slices : Slices) -> tuple[np.ndarray, np.ndarray]:
        """ (Legacy) Creates the interacting and incident histograms.

        Args:
            int_energy (ak.Array): interacting enrgy
            ff_energy (ak.Array): front facing energy
            outside_tpc (ak.Array): mask which selects particles decaying outside the tpc
            channel (ak.Array): mask which selects particles which interact in the channel you are interested in
            energy_slices (Slices): energy slices

        Returns:
            tuple[np.ndarray, np.ndarray]: n_interact and n_incident histograms
        """
        true_init_slice = energy_slices(ff_energy).num + 1 # equivilant to ceil
        true_int_slice = energy_slices(int_energy).num

        true_init_slice = EnergySlice.TrunacteSlices(true_init_slice, energy_slices)
        true_int_slice = EnergySlice.TrunacteSlices(true_int_slice, energy_slices)

        # just in case we encounter an instance where E_int > E_ini (unphysical)
        bad_slices = true_int_slice < true_init_slice
        true_init_slice = ak.where(bad_slices < 0, -1, true_init_slice)
        true_int_slice = ak.where(bad_slices, -1, true_int_slice)

        n_incident = np.zeros(energy_slices.max_num + 1)
        n_interact = np.zeros(energy_slices.max_num + 1)

        true_int_slice_in_tpc = true_int_slice[~outside_tpc]
        true_init_slice_in_tpc = true_init_slice[~outside_tpc]

        #! slowest but most explict version
        # n_incident = np.zeros(max_slice + 1)
        # for i in range(len(n_incident)):
        #     for p in range(len(true_int_slice_in_tpc)):
        #         if (true_init_slice_in_tpc[p] <= i) and (true_int_slice_in_tpc[p] >= i):
        #             n_incident[i] += 1
        #! faster, order log(n) because it skips checking for empty entries
        # true_init_slice_in_tpc = ak.where(true_init_slice_in_tpc == -1, 0, true_init_slice_in_tpc) #! done because -n index in python means you add to the last nth bin
        # for p in range(len(true_int_slice_in_tpc)):
        #     n_incident[true_init_slice_in_tpc[p] : true_int_slice_in_tpc[p] + 1] += 1
        # print(n_incident)

        #! fastest, vectorised version of the first but c++ loops are faster. 
        n_incident = np.array([
            ak.sum(ak.where((true_init_slice_in_tpc <= i)
                            & (true_int_slice_in_tpc > i), 1, 0))
            for i in range(energy_slices.max_num + 1)])
    
        n_interact = np.histogram(
            np.array(true_int_slice_in_tpc[channel[~outside_tpc]]),
            range(-1, energy_slices.max_num + 1))[0]
        # Shift the underflow bin to the location of the overflow bin
        #   in n_incident i.e. merge them.
        n_interact = np.roll(n_interact, -1)
        return n_interact, n_incident + n_interact

    @staticmethod
    def Slice_dEdX(energy_slices : Slices, particle : Particle) -> np.ndarray:
        """ Computes the mean dEdX between energy slices.

        Args:
            energy_slices (Slices): energy slices
            particle (Particle): particle

        Returns:
            np.ndarray: mean dEdX
        """
        return etools.BetheBloch.meandEdX(
            energy_slices.pos - energy_slices.width/2, particle)

    @staticmethod
    def TotalCrossSection(n_interact : np.ndarray, n_incident : np.ndarray, dEdX : np.ndarray, dE : float) -> tuple[np.ndarray, np.ndarray]:
        """ Compute cross section using ThinSlice.CrossSection, by passing an effective spatial slice width.

        Args:
            n_interact (np.ndarray): interacting histogram
            n_incident (np.ndarray): incident histogram
            dEdX (np.ndarray): mean slice dEdX
            dE (float): energy slice width

        Returns:
            tuple[np.ndarray, np.ndarray]: Cross section and statistical uncertainty.
        """
        return ThinSlice.TotalCrossSection(n_incident, n_interact, dE/dEdX)

    @staticmethod
    def CrossSection(n_int_ex : np.ndarray, n_int : np.ndarray, n_inc : np.ndarray, dEdX : np.ndarray, dE : float, n_int_ex_err : np.ndarray = None, n_int_err : np.ndarray = None, n_inc_err : np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """ Compute exclusive cross sections. If interactions errors are not provided, staticial uncertainties are used (poisson for incident, binomial for interactions).

        Args:
            n_int_ex (np.ndarray): exclusive interactions
            n_int (np.ndarray): interactions
            n_inc (np.ndarray): incident counts
            dEdX (np.ndarray): slice dEdX
            dE (float): energy slice width
            n_int_ex_err (np.ndarray, optional): exclusive interaction errors. Defaults to None.
            n_int_err (np.ndarray, optional): interaction errors. Defaults to None.
            n_inc_err (np.ndarray, optional): incident count errors. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        NA = 6.02214076e23
        factor = (np.array(dEdX) * 10**27 * etools.BetheBloch.A
                  / (etools.BetheBloch.rho * NA * dE))

        n_interact_ratio = nandiv(n_int_ex, n_int)
        n_survived = n_inc - n_int

        if n_inc_err is not None:
            var_inc_inclusive = n_inc_err**2
        else:
            var_inc_inclusive = n_inc # poisson variance
    
        if n_int_err is not None:
            var_int = n_int_err**2
        else:
            var_int = n_int * (1 - nandiv(n_int, n_inc)) # binomial uncertainty
    
        if n_int_ex_err is not None:
            var_int_ex = n_int_ex_err**2
        else:
            var_int_ex = n_int_ex * (1 - nandiv(n_int_ex, n_inc)) # binomial uncertainty


        xs = factor * n_interact_ratio * nanlog(nandiv(n_inc, n_inc - n_int))

        diff_n_int_ex = nandiv(xs, n_int_ex)
        diff_n_inc = factor * n_interact_ratio * (nandiv(1, n_inc) - nandiv(1, n_survived))
        diff_n_int = factor * n_interact_ratio * nandiv(1, n_survived) - nandiv(xs, n_int)

        xs_err = ((diff_n_int_ex**2 * var_int_ex)
                  + (diff_n_inc**2 * var_inc_inclusive)
                  + (diff_n_int**2 * var_int))**0.5
        return np.array(xs, dtype = float), np.array(xs_err, dtype = float)