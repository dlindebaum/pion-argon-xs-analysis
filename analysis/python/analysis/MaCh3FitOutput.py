"""
Created on: 02/10/2026 16:20

Author: Shyam Bhuller

Description: Class to read and process MaCh3Fit input data.
"""
import numpy as np

from python.analysis import Master, Slices, Utils, ProcessDefinitions

class MaCh3FitOutput:
    def __init__(self, file : str, process_definitions : ProcessDefinitions.SampleDefinition, energy_slices : Slices):
        self.process_definitions = process_definitions
        self.energy_slices = energy_slices
        with Master.uproot.open(file) as fit_output:

            self.param_labels = fit_output["CovarianceFolder"]["xsec_cov_Corr"].axis("x").labels()

            posterior_outputs = []
            for k in fit_output:
                if "posteriors" in k: 
                    posterior_outputs.append(k)
            key = f"posteriors;{max([int(k.split(";")[1]) for k in posterior_outputs])}"
            self.posteriors = {k : v.array() for k, v in zip(self.param_labels, fit_output[key])}
        pass


    def __fmt_bin__(self, str_bin : str, slices : Slices) -> int:
        """ Convert back from the binning scheme in MaCh3 to Slices.

        Args:
            str_bin (str): Bin label.
            slices (Slices): Energy slices.

        Returns:
            int: Energy Slices.
        """
        if str_bin == "Under":
            int_bin = 0
        elif str_bin == "Over":
            int_bin = len(slices.edges)
        else:
            int_bin = int(str_bin) + 1
        return int_bin


    def posterior_tensors(self) -> dict[np.ndarray]:
        """ Produce posterior tensors for each process.
            Dimensions are n,n,m, where n is the number of slices, m is the number of MCMC steps.

        Returns:
            dict[np.ndarray]: Dictionary of posterior tensors.
        """
        posterior_tensor = {}
        for k, v in self.posteriors.items():
            proc, *bins = k.split("-")
            proc = proc.replace("True", "")
            proc = Utils.dict_reverse(self.process_definitions.short_name)[proc]

            index = [self.__fmt_bin__(i.split("_")[-1], self.energy_slices) for i in bins]
            if proc in posterior_tensor:
                posterior_tensor[proc][*index] = v
            else:
                shape = [len(self.energy_slices.edges) + 1]*2 + [len(v)]
                posterior_tensor[proc] = np.ones(shape)
                posterior_tensor[proc][*index] = v
        return posterior_tensor
