import numpy as np
import pandas as pd

import Plots
import Tags

from Master import ReadHDF5
from utils import CountInRegions

class Toy:
    def __init__(self, file : str = None, df : str = None) -> None:
        if file is not None:
            self.df = ReadHDF5(file)
        elif df is not None:
            self.df = df
        else:
            return
    
        self.exclusive_processes = np.unique(self.df.exclusive_process)
        self.exclusive_processes = self.exclusive_processes[self.exclusive_processes != ""]

    @staticmethod
    def GetRegion(toy : pd.DataFrame, region : str) -> pd.DataFrame:
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


    def GetCorrelationMatrix(self) -> np.ndarray:
        """ Compute the confusion matrix for the reco/truth regions.

        Args:
            toy (pd.DataFrame): Toy

        Returns:
            np.ndarray: confusion matrix
        """
        reco_regions = Toy.GetRegion(self.df, "reco_regions_")
        true_regions = Toy.GetRegion(self.df, "truth_regions_")
        return CountInRegions(true_regions, reco_regions)


    def SetProperty(self, name : str, value : any):
        hidden_name = f"_{type(self).__name__}__{name}"
        if not hasattr(self, hidden_name):
            setattr(self, hidden_name, value)
        return getattr(self, hidden_name)


    def GetRegionNames(self, name : str) -> list[str]:
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
        return self.SetProperty("outside_tpc", (self.df.z_int < 0) | (self.df.z_int > 700))

    @property
    def outside_tpc_smeared(self):
        return self.SetProperty("outside_tpc_smeared", (self.df.z_int_smeared < 0) | (self.df.z_int_smeared > 700))

    @property
    def truth_regions(self):
        return self.SetProperty("truth_regions", self.GetRegion(self.df, "truth_regions_"))

    @property
    def reco_regions(self):
        return self.SetProperty("reco_regions", self.GetRegion(self.df, "reco_regions_"))

    @property
    def reco_region_labels(self):
        return self.GetRegionNames("reco_regions_")

    @property
    def truth_region_labels(self):
        return self.GetRegionNames("truth_regions_")

    @staticmethod
    def PlotObservablesInRegions(observable : pd.Series, reco_regions : pd.DataFrame, true_regions : pd.DataFrame, label : str, norm : bool = False, stacked : bool = False, histtype = "step"):
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
        for _, r in Plots.IterMultiPlot(reco_regions.columns):
            tmp_regions = {t : true_regions[t].values & reco_regions[r].values & (observable > 0) for t in true_regions.columns} # filter the reco events for this region only
            Plots.PlotTagged(observable, Tags.ExclusiveProcessTags(tmp_regions), bins = 50, newFigure = False, title = f"reco region : {r}", reverse_sort = False, stacked = stacked, histtype = histtype, x_label = label, ncols = 1, norm = norm)
        return
