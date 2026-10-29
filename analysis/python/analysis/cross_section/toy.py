import numpy as np
import pandas as pd

from python.analysis import io, regions, tags
from python.analysis.plotting import plots

class Toy:
    def __init__(self, file : str = None, df : str = None) -> None:
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

        Args:
            toy (pd.DataFrame): Toy

        Returns:
            np.ndarray: confusion matrix
        """
        reco_regions = Toy.get_region(self.df, "reco_regions_")
        true_regions = Toy.get_region(self.df, "truth_regions_")
        return regions.count_in_regions(true_regions, reco_regions)


    def set_property(self, name : str, value : any):
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
        return self.set_property("outside_tpc", (self.df.z_int < 0) | (self.df.z_int > 700))

    @property
    def outside_tpc_smeared(self):
        return self.set_property("outside_tpc_smeared", (self.df.z_int_smeared < 0) | (self.df.z_int_smeared > 700))

    @property
    def truth_regions(self):
        return self.set_property("truth_regions", self.get_region(self.df, "truth_regions_"))

    @property
    def reco_regions(self):
        return self.set_property("reco_regions", self.get_region(self.df, "reco_regions_"))

    @property
    def reco_region_labels(self):
        return self.get_region_names("reco_regions_")

    @property
    def truth_region_labels(self):
        return self.get_region_names("truth_regions_")

    @staticmethod
    def plot_observables_in_regions(observable : pd.Series, reco_regions : pd.DataFrame, true_regions : pd.DataFrame, label : str, norm : bool = False, stacked : bool = False, histtype = "step"):
        """ Plot an observable from the toy in each region for each process.

        Args:
            observable (pd.Series): observable to plot
            reco_regions (pd.DataFrame): reco regions
            true_regions (pd.DataFrame): true regions
            label (str): x label
            norm (bool, optional): normalise plots. Defaults to False.
            stacked (bool, optional): stack histograms. Defaults to False.
            histtype (str, optional): histogram style. Defaults to "step".
        """
        for _, r in plots.iter_multi_plot(reco_regions.columns):
            tmp_regions = {t : true_regions[t].values & reco_regions[r].values & (observable > 0) for t in true_regions.columns} # filter the reco events for this region only
            plots.plot_tagged(observable, tags.exclusive_process_tags(tmp_regions), bins = 50, new_figure = False, title = f"reco region : {r}", reverse_sort = False, stacked = stacked, histtype = histtype, x_label = label, ncols = 1, norm = norm)
        return
