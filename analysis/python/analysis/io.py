"""
Created on: 30/10/2026 16:05

Author: Shyam Bhuller

Description: Functions and classes for reading from / writing to differnt file types.
"""
import json
import warnings

import dill
import awkward as ak
import pandas as pd
import tables
import uproot


def read_dill(file : str):
    """ Opens and serialises object saved as a dill file. May be remaned to a more general method if dill files are used more commonly.

    Args:
        file (str): dill file

    Returns:
        any: loaded object
    """
    with open(file, "rb") as f:
        obj = dill.load(f)
    return obj


def write_dill(file : str, obj : dict):
    """ Saves object to file.

    Args:
        file (str): file name
        obj (dict): object to save
    """
    with open(file, "wb") as f:
        dill.dump(obj, f)
    return

def read_json(file : str) -> dict:
    """ Loads a json file.

    Args:
        file (str): file path

    Returns:
        dict: unpacked json 
    """
    with open(file, "rb") as f:
        config = json.load(f)
    return config


def write_json(params : any, file : str):
    """ Saves an object to a json file. params should be an object serialisable by json (no numpy arrays etc.).

    Args:
        params (any): correction parameters.
        file (str): file path.
    """
    with open(file, "w") as f:
        json.dump(params, f, indent = 2)
    return


def write_hdf5(dictionary : dict, file : str):
    """ Write dictionary to a HDF5 file.

    Args:
        dictionary (dict): dictionary to save
        file (str): file
    """
    with pd.HDFStore(file) as hdf:
        for k, v in dictionary.items():
            hdf.put(k, v)
    return


def read_hdf5(file : str):
    """ Reads a HDF5 file and unpacks the contents into pandas dataframes.

    Args:
        file (str): file path.

    Returns:
        pd.DataFrame : if hdf5 file only has 1 key
        dict : if hdf5 file contains more than 1 key
    """
    keys = []
    with tables.open_file(file, driver = "H5FD_CORE") as hdf5file:
        for c in hdf5file.root: 
            keys.append(c._v_pathname[1:])
    if len(keys) == 1:
        return pd.read_hdf(file)
    else:
        return {k : pd.read_hdf(file, k) for k in keys}


class RootIO:
    """ Interface to ROOT file using uproot
    Attributes:
        filename (str): ROOT filename
        n_events (int): number of events to read. Defaults to -1 (all events)
        start (int): starting point to read from. Defaults to None.
    Methods:
        Get (ak.Array): Load ntuple from root file as awkward array.
        list_ntuples : list all NTuple entries produced by the analyser.
    """

    def __init__(self, _filename: str, _n_events: int = -1, _start: int = None):
        self.filename = _filename
        self.start = _start
        self.n_events = _n_events

    @staticmethod
    def convert_types(d : dict[dict | ak.Array]) -> dict:
        """ Convert the types from a nested dictionary of awkward arrays being written to a ROOT file.
            Optional types (data where there was/is a null entry) is converted to a standard Awkward array type.

        Args:
            d (dict): Nested dictionary of awkward arrays. 

        Returns:
            dict: converted dictionary. Dictionary is flattened to prevent overwrites in TTree entries.
        """
        d_new = {}
        for k, v in d.items():
            if v is None:
                print(f"Info: data entry {k} is None, this entry will be dropped.")
                continue
            if type(v) == dict:
                for k1, v1 in RootIO.convert_types(v).items():
                    d_new[f"{k}_{k1}"] = v1
            elif isinstance(v.type.content, ak.types.OptionType):
                d_new[k] = ak.Array(v.layout.project())
            else:
                d_new[k] = v
        return d_new


    def write_data(self, vectors : dict = None, histograms : dict = None, overwrite : bool = False):
        """ Write TTree to a new root file.

        Args:
            vectors (dict, optional): Dictionary of vectors. Vectors must be the same size, can be a nested dictionary of np or Awkward arrays. Defaults to None.
            histograms (dict, optional): Dictionary of Histogram like objects (e.g. output of np.histogram). Can be variable in size. Defaults to None.
            overwrite (bool, optional): Overwrites output file. Defaults to False.
        """
        split = self.filename.split("/")
        path = "/".join(split[:-1] + [split[-1].split(".")[0] + ".root"])

        io_obj = uproot.recreate if overwrite is True else uproot.create
        with io_obj(path) as file: # create instead of recreate so we dont accidently overwrite our precious ntuple files.
            # vector like inputs
            if vectors is not None:
                file.mktree("FlatTree_VARS", self.convert_types(vectors))
    
            # histogram like inputs
            if histograms is not None:
                for k, v in histograms.items():
                    file[k] = v

        print(f"TTree Written to {path}")
        return


    def read_data(self, item: str) -> ak.Array:
        """ Load ntuple from root file as awkward array.

        Args:
            item (str): ntuple name in root file

        Returns:
            ak.Array: ntuple loaded
        """
        with uproot.open(self.filename) as file:
            try:
                if self.n_events > 0:
                    for batch in file["pduneana/beamana"].iterate(entry_start = self.start, entry_stop = self.start + self.n_events, step_size = self.n_events, filter_name = item):
                        if len(batch) > 0:
                            return batch[item]
                        else:
                            raise uproot.KeyInFileError(item)
                else:
                    return file["pduneana/beamana"][item].array()
            except uproot.KeyInFileError:
                # print(f"{item} not found in file, moving on...")
                return None


    def get(self, item):
        if type(item) is list:
            for i in item:
                out = self.read_data(i)
                if out is not None:
                    return out
            if out is None:
                warnings.warn(f"ntuple names {item} could not be found in the file.")
        else:
            return self.read_data(item)


    def list_ntuples(self, search: str = "", return_keys=False):
        """ list all NTuple entries produced by the analyser.

        Args:
            search (str, optional): filter NTuple by name (not case sensitive). Defaults to "".
        """
        with uproot.open(self.filename) as file:
            names = file["pduneana/beamana"].keys()
            keys = [x for x in names if search.casefold() in x.casefold()]
            if return_keys:
                return keys
            else:
                print(keys)
                return