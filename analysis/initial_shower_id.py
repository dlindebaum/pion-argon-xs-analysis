"""
Created on: 27/07/2022

Author: Dennis Lindebaum

Description: Fetch and calculate properties for determining
    leading photon pair from a pion.
Produces a pickle file containing all desired properties.
"""

from xml.sax.handler import property_interning_dict
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import uproot
import pickle
import Master
import vector


class Cut():
    """
    We want to have a system to work out the priority order in which to
    load data such that we minimise the amount of time we must keep some data loaded.
    This class keeps track of the function and assiciated properties to enable this
    """
    def __init__(self, operation, *props, cut=None):
        self.op = operation
        self.props = props
        if cut is None:
            self.cut_arg = ()
        else:
            self.cut_arg = cut
        return
    
    def get_filter(self, loaded_properties):
        return self.op(*(p for p in self.props), *self.cut_arg)

    def __call__(self, loaded_properties):
        return self.get_filter(loaded_properties)


def load_and_apply_selection(file, selections, batch_size = "1GB"):
    props = []

    # Not allow to unpack in python comprehension
    for cut in selections:
        props += [*cut.props]
    props = np.unique(props)

    for batch in file.iterate(props, step_size = batch_size):
        pass



class Filter():
    aliases = {}

    # @classmethod
    def _alias_decorator(*alias):
        func_name = None
        def wrapper(func):
            nonlocal func_name
            func_name = func.__name__
            return func
        for a in alias:
            Filter.aliases.update({func_name:a})
        return wrapper

    # @classmethod
    def _lookup_alias(self, alias):
        if alias not in self.__dict__.keys():
            try:
                alias = self.aliases[alias]
            except:
                raise NameError(f"Alias {alias} not found.")
        return self.__dict__[alias]

    # @_alias_decorator('>', 'g', 'greater')
    @staticmethod
    def greater_cut(data, cut):
        return data > cut
    
    # @_alias_decorator('>=', 'ge', 'greaterequal')
    @staticmethod
    def greater_equal_cut(data, cut):
        return data >= cut

    # @_alias_decorator('<', 'l', 'less')
    @staticmethod
    def less_cut(data, cut):
        return data < cut
    
    # @_alias_decorator('<=', 'le', 'lessequal')
    @staticmethod
    def less_equal_cut(data, cut):
        return data <= cut
    
    # @_alias_decorator('=', '==', 'e', 'eq', 'equal')
    @staticmethod
    def equal_cut(data_1, data_2):
        return data_1 == data_2
    

    def __init__(self):
        # We don't want to load our properties multiple times
        self.props = []
        self.loaded_props = []
        self.cuts = []

    def add_simple_filter(self, operation, prop, cut):
        if type(operation) is str:
            operation = self._lookup_alias(operation)
        
        prop_ind = self._get_prop_ind(prop)
        self.cuts += [Cut(operation, prop_ind, cut=cut)]
        return
        
    def add_complex_filter(self, operation, *prop):
        prop_inds = []
        for p in prop:
            prop_inds += [self._get_prop_ind(p)]
        self.cuts += [Cut(operation, *prop_inds)]
        return
    
    def _get_prop_ind(self, prop):
        if prop in self.props:
                return self.props.index(prop)
        else:
            self.props += [prop]
            self.loaded_props = [None]
            return len(self.props) - 1

    @property
    def connections_matrix(self):
        """
        Generates a connections matrix:
        Columns = num. properties
        Rows    = num. functions
        1 when the functions requires the property, 0 otherwise.
        """
        connections = np.zeros((len(self.props), len(self.cuts)))
        for i in range (len(self.cuts)):
            connections[:,i][self.cuts[i].prop_inds] = 1
        return connections

    def gen_filter(self):
        """
        TODO Need to compare speed with that of looping through each event
        Could be a way to easily get the true mask out at the same time...
        Need to have  away of appling th ecut on an arbitary piece of data.

        See https://uproot.readthedocs.io/en/latest/basic.html#iterating-over-intervals-of-entries
        Final method will be: iterate over some number of events (8086? 8096 * O(100) * O(10) * O(100) => O(100MB) )
        Even better can use: step_size="1GB" for example.

        Then we can simply load and apply all cuts very easily.

        To deal with multiple files, we load all files with uproot.concatenate().
        This shouldn't load everything until we actually iterate over.

        OLD:
        To deal with multiple files (i.e. when we load the whole 6GeV beam) can initially
        create a lazy uproot array which loads every file:
        https://uproot.readthedocs.io/en/latest/basic.html#reading-on-demand-with-lazy-arrays
        Then simply iterate throuhg, as in this method, and we're sorted.

        Need to confirm that we can create a lazy mask though...
        """
        applied_funcs = []
        for f in range(len(self.cuts)):
            func_ind = self.props_handler(applied_funcs)
            applied_funcs += [func_ind]
            if f == 0:
                self.filter = self.cuts[func_ind].get_filter(self.loaded_props)
            else:
                self.filter = np.logical_and(self.filter, self.cuts[func_ind].get_filter(self.loaded_props))

        return self.filter

    def props_handler(self, cut_func_inds):
        prop_weights = np.sum(self.connections_matrix, axis=1)
        weighted_connections = self.connections_matrix * prop_weights
        func_weights = np.sum(weighted_connections, axis=0)

        return func_weights.argmax()



class BeamCut(Cut):
    def __init__(self, events):
        has_beam = events["beamNum"].array() != -999
        beam_particle = events["beamNum"].array() == events["reco_PFP_ID"].array()
        beam_daughters = events["beamNum"].array() == events["reco_PFP_Mother"].array()
        self.filter = np.logical_and(has_beam, np.logical_or(beam_particle, beam_daughters))
        return

def beam_cut(beam, pfp_id, pfp_mother):
    has_beam = beam != -999
    beam_particle = beam == pfp_id
    beam_daughters = beam == pfp_mother
    return np.logical_and(has_beam, np.logical_or(beam_particle, beam_daughters))

class HitsCut(Cut):
    def __init__(self, events):
        self.filter = events["reco_daughter_PFP_nHits_collection"].array() > 50
        return

def closest_approach(dir1, dir2, start1, start2):
    """
    Find the closest approach between two showers.

    Parameters:
        Format: ak.zip({'x':ak.array, 'y':ak.array, 'z':ak.array})
            dir1    : direction of first shower
            dir2    : direction of second shower
            start1  : initial point of first shower
            start2  : initial point of second shower

    Returns:
        Format: ak.array
            d       : Magnitude of the sortest direction between the two showers
    """
    # x_1 - x_2 + lambda_1 v_1 - lambda_2 v_2 = d/sin(theta) v_1 x v_2
    cross = vector.normalize( vector.cross(dir1, dir2) )
    rel_start = vector.sub(start1, start2)

    # Separation between the lines
    d = vector.dot(rel_start, cross)

    return d

def shared_vertex(distance, dir1, dir2, start1, start2):
    """
    Find the closest approach between two showers.

    Parameters:
        Format: ak.array
            distance : Cloest separation between two showers
        Format: ak.zip({'x':ak.array, 'y':ak.array, 'z':ak.array})
            dir1     : direction of first shower
            dir2     : direction of second shower
            start1   : initial point of first shower
            start2   : initial point of second shower

    Returns:
        Format: ak.zip({'x':ak.array, 'y':ak.array, 'z':ak.array})
            mid      : Midpoint of the line of shortest separation betwee, the two showers
    """
    # Is there a better way to deal with having to recalculate cross and relstart?
    # Maybe have the midp