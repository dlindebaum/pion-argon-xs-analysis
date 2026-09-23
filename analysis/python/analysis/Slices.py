"""
Created on: 23/09/2026 15:44

Author: Shyam Bhuller

Description: Class for handling creation of slices used for the cross section calculation.
"""

from collections import namedtuple

import awkward as ak
import numpy as np

class Slices:
    Slice = namedtuple("Slice", "num pos")
    def __init__(self, edges : np.ndarray[int]):
        self.edges = np.array(edges)
        self.min = min(edges)
        self.max = max(edges)


        if all((edges[1:] - edges[:-1]) > 0):
            # increasing, not reversed
            self.reversed = False
        elif all((edges[1:] - edges[:-1]) < 0):
            # decreasing, reversed
            self.reversed = True
        else:
            raise Exception(f"edges: {edges}\n is not an ordered list (either ascending or descending), or slice widths are zero")

        self.max_num = max(self.num)
        self.min_num = min(self.num)

        self.max_pos = max(self.pos)
        self.min_pos = min(self.pos)

        self.overflow_num = self.max_num + 1 # overflow slice number
        self.underflow_num = -1 # underflow slice number


    def __conversion__(self, x):
        """ convert a value to its slice number.

        Args:
            x: value, array of float

        Returns:
            slice: slice number/s
        """
        if hasattr(x, "__iter__"):
            if self.reversed:
                n = ak.sum(ak.unflatten(x, 1) <= self.edges, 1) - 1

            else:
                n = ak.sum(ak.unflatten(x, 1) >= self.edges, 1) - 1
        else:
            if self.reversed:
                n = sum(x <= self.edges) - 1
            else:
                n = sum(x >= self.edges) - 1
        return n


    def __create_slice__(self, i) -> Slice:
        """ using the slice number, create the Slice object.

        Args:
            i (int): slice number/s

        Returns:
            Slice: slice
        """
        if hasattr(i, "__iter__"):
            if self.reversed:
                j = ak.where(i >= len(self.edges), i - 1, i) # create a dummy index to truncate indexes beyond the range
                p = ak.where(i >= len(self.edges), -np.inf, self.edges[j]) # get the slice edges, and underflow goes to -inf
                p = ak.where(i < 0, np.inf, p) # overflow goes to inf
            else:
                p = ak.where(i == 0, -np.inf, self.edges[i-1]) # dont need to account for the +ve inf case (overflow)
        else:
            if self.reversed:
                if i >= len(self.edges):
                    p = -np.inf
                elif i < 0:
                    p = np.inf
                else:
                    p = self.edges[i]
            else:
                if i == 0:
                    p = -np.inf
                elif i > len(self.edges): # Given the way __conversion__ works, this condition will never be satisfied...
                    p = np.inf
                else:
                    p = self.edges[i - 1]

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
        #! it is possible to return slice -1 but not a slice value exceedin the maximum slice

        Args:
            i (int): slice number

        Raises:
            StopIteration

        Returns:
            Slice: ith slice
        """
        if i >= len(self.edges) - 1:
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
    def num_all(self) -> np.ndarray:
        """ Return all slice numbers.

        Returns:
            np.ndarray: slice numbers
        """
        return np.array([self.underflow_num] + [s.num for s in self] + [self.overflow_num], dtype = int)

    @property
    def pos(self) -> np.ndarray:
        """ Return all slice positions. Within the valid slice range.

        Returns:
            np.ndarray: slice positions
        """
        return np.array([s.pos for s in self])

    @property
    def width(self) -> np.ndarray:
        """ Return widths of each slice.

        Returns:
            np.ndarray: slice widths
        """
        if reversed:
            return self.edges[:-1] - self.edges[1:]
        else:
            return self.edges[1:] - self.edges[:-1] 

    @property
    def pos_overflow(self) -> np.ndarray:
        widths = abs(self.edges[1:] - self.edges[:-1])
        i = 0 if self.reversed else -1
        ov = self.edges[i] + widths[i]
        if self.reversed:
            return np.insert(self.pos, i, ov)
        else:
            return np.append(self.pos, ov)

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
