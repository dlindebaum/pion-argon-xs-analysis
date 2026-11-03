import numpy as np
import pandas as pd

from python.analysis import io, regions, tags
from python.analysis.plotting import plots

class Toy:
    """Access toy-event columns and lazily derived region masks from a dataframe.

        Args:
            file (str): HDF5 file to load as the toy dataframe.
            df (str): Dataframe to use directly when no file is supplied.
    """
    def __init__(self, file : str = None, df : str = None) -> None:
        """Load a toy dataframe from HDF5 or store the supplied dataframe.

            Args:
                file (str): HDF5 file to load as the toy dataframe.
                df (str): Dataframe to use directly when no file is supplied.
        """
        if file is not None:
            self.df = io.read_hdf5(file)
        elif df is not None:
            self.df = df
        else:
            return
    
        self.exclusive_processes = np.unique(self.df.exclusive_process)
        self.exclusive_processes = self.exclusive_processes[self.exclusive_processes != ""]

    @staticmethod
    def get_region(toy : pd.DataFrame, region : str) -> pd.DataFrame:
        """ get region/process masks from data frame.

            Args:
                toy (pd.DataFrame): Toy
                region (str): region regex, either "truth_region_" or "reco_region_"

            Returns:
                pd.DataFrame: dataframe of masks
        """
        regions = toy.filter(regex = region)
        new_col_names = {}
        for i in regions:
            new_col_names[i] = i.split(region)[1]
        regions = regions.rename(columns = new_col_names)
        return regions


    def get_correlation_matrix(self) -> np.ndarray:
        """ Compute the confusion matrix for the reco/truth regions.

            Returns:
                np.ndarray: confusion matrix
        """
        reco_regions = Toy.get_region(self.df, "reco_regions_")
        true_regions = Toy.get_region(self.df, "truth_regions_")
        return regions.count_in_regions(true_regions, reco_regions)


    def set_property(self, name : str, value : any):
        """Cache a value in a name-mangled attribute and return the cached value.

            Args:
                name (str): Property name used to form the private cache attribute.
                value (any): Value to cache if that attribute has not been set.

            Returns:
                Any: The cached value; an existing value is retained on later calls.
        """
        hidden_name = f"_{type(self).__name__}__{name}"
        if not hasattr(self, hidden_name):
            setattr(self, hidden_name, value)
        return getattr(self, hidden_name)


    def get_region_names(self, name : str) -> list[str]:
        """ Get names of each region as labelled in the dataframe column.

            Args:
                name (str): column regex

            Returns:
                list[str]: names
        """
        labels = self.df.filter(regex = name).columns
        return [s.split(name)[-1] for s in labels]

    @property
    def outside_tpc(self):
        """Return and cache a mask for true interaction positions outside the 0--700 cm z-range.

            Returns:
                Any: Boolean mask from the ``z_int`` column.
        """
        return self.set_property("outside_tpc", (self.df.z_int < 0) | (self.df.z_int > 700))

    @property
    def outside_tpc_smeared(self):
        """Return and cache a mask for smeared interaction positions outside the 0--700 cm z-range.

            Returns:
                Any: Boolean mask from the ``z_int_smeared`` column.
        """
        return self.set_property("outside_tpc_smeared", (self.df.z_int_smeared < 0) | (self.df.z_int_smeared > 700))

    @property
    def truth_regions(self):
        """Return and cache truth-region mask columns as a dataframe.

            Returns:
                Any: Dataframe of columns whose names start with ``truth_regions_``, with that prefix removed.
        """
        return self.set_property("truth_regions", self.get_region(self.df, "truth_regions_"))

    @property
    def reco_regions(self):
        """Return and cache reconstructed-region mask columns as a dataframe.

            Returns:
                Any: Dataframe of columns whose names start with ``reco_regions_``, with that prefix removed.
        """
        return self.set_property("reco_regions", self.get_region(self.df, "reco_regions_"))

    @property
    def reco_region_labels(self):
        """List reconstructed-region labels parsed from the toy dataframe's column names.

            Returns:
                Any: Suffixes of columns beginning with ``reco_regions_``.
        """
        return self.get_region_names("reco_regions_")

    @property
    def truth_region_labels(self):
        """List truth-region labels parsed from the toy dataframe's column names.

            Returns:
                Any: Suffixes of columns beginning with ``truth_regions_``.
        """
        return self.get_region_names("truth_regions_")

    @staticmethod
    def plot_observables_in_regions(observable : pd.Series, reco_regions : pd.DataFrame, true_regions : pd.DataFrame, label : str, norm : bool = False, stacked : bool = False, histtype = "step"):
        """ Plot an observable from the toy in each region for each process.

            Args:
                observable (pd.Series): observable to plot
                reco_regions (pd.DataFrame): reco regions
                true_regions (pd.DataFrame): true regions
                label (str): x label
                norm (bool): normalise plots. Defaults to False.
                stacked (bool): stack histograms. Defaults to False.
                histtype (str, optional): histogram style. Defaults to "step".
        """
        for _, r in plots.iter_multi_plot(reco_regions.columns):
            tmp_regions = {t : true_regions[t].values & reco_regions[r].values & (observable > 0) for t in true_regions.columns} # filter the reco events for this region only
            plots.plot_tagged(observable, tags.exclusive_process_tags(tmp_regions), bins = 50, new_figure = False, title = f"reco region : {r}", reverse_sort = False, stacked = stacked, histtype = histtype, x_label = label, ncols = 1, norm = norm)
        return
