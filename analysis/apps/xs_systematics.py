#!/usr/bin/env python3
"""
Created on: 09/04/2024 13:53

Author: Shyam Bhuller

Description: App to calculate systematic uncertainties for cross section measurement.
"""
import argparse
from abc import ABC, abstractmethod
import os
import warnings

import pandas as pd
import numpy as np

from particle import Particle
from rich import print
import pyhf
from rich.rule import Rule

from python.analysis import application
from python.analysis import bethe_bloch
from python.analysis import common
from python.analysis import fitting
from python.analysis import kinematics
from python.analysis import io
from python.analysis import ntuple
from python.analysis import plotting
from python.analysis import processing
from python.analysis import selection
from python.analysis import tags
from python.analysis import toy
from python.legacy import RegionFit

from python.analysis.cross_section import analysis_input, plotting, geant
from apps import xs_toy_generator, xs_analyse, xs_fit_studies, xs_analysis_input

systematics = ["mc_stat", "upstream", "beam_reweight", "shower_energy", "track_length", "beam_res", "fit_inaccuracy", "theory", "all"]
systematics_label = {"mc_stat" : "MC stat", "fit_inaccuracy" : "Fit inaccuracy", "upstream" : "Upstream", "beam_reweight" : "Reweight", "shower_energy" : "Shower energy", "track_length" : "Track length", "beam_res" : "Beam momentum", "theory" : "Theory"}

exclusive_proc = ["absorption", "charge_exchange", "single_pion_production", "pion_production"]

pd.set_option('future.no_silent_downcasting', True)

def save_systematic_error(systematic : dict, fractional : dict, out : str):
    """ Serialize systematic errors and their fractional values to a dill file.

        Args:
            systematic (dict): Absolute systematic uncertainties by source/process.
            fractional (dict): Fractional uncertainties corresponding to the systematics.
            out (str): Destination file path.
    """
    return io.write_dill(out, {"systematic" : systematic, "fractional": fractional})


def save_tables(tables : pd.DataFrame, outdir : str, precision : int):
    """ Save systematic tables in HDF5 and formatted LaTeX files.

        Args:
            tables (pd.DataFrame): Mapping of table names to per-process data frames.
            outdir (str): Directory in which to write the tables.
            precision (int): Number of decimal places used in LaTeX formatting.
    """
    io.write_hdf5(tables, f"{outdir}tables.hdf5")
    for t in tables:
        tables[t].style.format(precision = precision).hide(axis = 0).to_latex(f"{outdir}table_{t}.tex")
    return


class MCMethod(ABC):
    """ Base workflow for systematic variations evaluated with MC templates.

        Args:
            args (argparse.Namespace): Analysis and fit configuration.
            model (pyhf.Model): Region-fit model used for the analysis.
            data_config (dict): Toy-data configuration for experiments.
    """
    def __init__(self, args : argparse.Namespace, model : pyhf.Model, data_config : dict) -> None:
        """ Store independent copies of the configuration and fit model.

            Args:
                args (argparse.Namespace): Analysis and fit configuration.
                model (pyhf.Model): Region-fit model used for the analysis.
                data_config (dict): Toy-data configuration for experiments.
        """
        self.args = common.dill_copy(args)
        self.model = model
        self.data_config = data_config
        pass

    def analyse(self, analysis_input : analysis_input.AnalysisInput, book : plotting.PlotBook = plotting.PlotBook.null()):
        """ Fit regions, subtract backgrounds, unfold, and calculate cross sections.

            Args:
                analysis_input (analysis_input.AnalysisInput): Event-level input to analyze.
                book (plotting.PlotBook): Optional destination for diagnostic plots.

            Returns:
                dict: Cross-section estimates and uncertainties by process.
        """

        region_fit_result = xs_analyse.region_fit(analysis_input, self.args.energy_slices, self.args.fit["mean_track_score"], self.model, mc_stat_unc = self.args.fit["mc_stat_unc"], single_bin = self.args.fit["single_bin"])

        _, histograms_reco_obs, histograms_reco_obs_err = xs_analyse.background_subtraction(analysis_input, self.args.signal_process, self.args.energy_slices, region_fit_result, self.args.fit["single_bin"], self.args.fit["regions"], self.args.toy_template, args.bkg_sub_mc_stat, book)


        if self.args.fit["regions"]:
            histograms_reco_obs = {**histograms_reco_obs, **histograms_reco_obs["int_ex"]}
            histograms_reco_obs.pop("int_ex")
            histograms_reco_obs_err = {**histograms_reco_obs_err, **histograms_reco_obs_err["int_ex"]}
            histograms_reco_obs_err.pop("int_ex")

        unfolding_result = xs_analyse.unfolding(histograms_reco_obs, histograms_reco_obs_err, self.args.toy_template, self.args.unfolding, self.args.signal_process, len(analysis_input.KE_init_reco)/len(self.args.toy_template.KE_init_reco), self.args.energy_slices, self.args.fit["regions"], self.args.fit["mc_stat_unc"], None, book)

        if book is not None:
            for p in xs_analyse.process_labels:
                hist_true = analysis_input.create_histograms(self.args.energy_slices, p, False, None)["int_ex"]
                plotting.plots.plot(self.args.energy_slices.pos_overflow, unfolding_result[p]["unfolded"], yerr = common.quadsum([unfolding_result[p]["stat_err"], unfolding_result[p]["sys_err"]], 0), xlabel = f"$N_{{int, {xs_analyse.process_labels[p]}}}$ (MeV)", ylabel = "Counts", label = "unfolded", style = "step", color = "C6")
                plotting.plots.plot(self.args.energy_slices.pos_overflow, hist_true, label = "true", style = "step", color = "C0", new_figure = False)
                book.save()

        xs = xs_analyse.xs_unfold(unfolding_result, self.args.energy_slices, True, True, self.args.fit["regions"])
        return xs

    @abstractmethod
    def run_experiment(self, config : dict) -> tuple[dict, dict]:
        """ Run one systematic experiment using the supplied toy configuration.

            Args:
                config (dict): Configuration for the generated toy data.

            Returns:
                tuple[dict, dict]: Cross sections and their corresponding truth values.
        """
        pass


    def evaluate(self, n : int, **kwargs):
        """ Run the systematic experiment repeatedly with shared keyword arguments.

            Args:
                n (int): Number of experiments to run.
                **kwargs (Any): Arguments forwarded to ``run_experiment``.

            Returns:
                list: Results from each experiment.
        """
        xs = []
        for i in range(n):
            print(Rule(f"Experiment : {i + 1}"))
            xs.append(self.run_experiment(**kwargs))
        return xs


    def calculate_sys_cov(self, results : list[dict], book : plotting.PlotBook = plotting.PlotBook.null()):
        """ Calculate cross-section uncertainties from experiment covariance matrices.

            Args:
                results (list[dict]): Per-experiment cross-section estimates and errors.
                book (plotting.PlotBook): Optional destination for covariance plots.

            Returns:
                dict: Per-process standard deviations derived from covariance diagonals.
        """
        values = {k : [] for k in results[0]}
        for i in results:
            for k in i:
                values[k].append(i[k][0])

        cov = {k : np.cov(values[k], rowvar = False) for k in values}

        if book:
            for k in cov:
                plotting.plots.plt.figure()
                plotting.plots.plt.imshow(cov[k])
                plotting.plots.plt.grid(False)
                plotting.plots.plt.colorbar()
                plotting.plots.plt.title(common.remove_(k).capitalize())

                x = self.args.energy_slices.pos_overflow[1:-1] - self.args.energy_slices.width/2
                plotting.plots.plt.xticks(np.linspace(0, len(x) - 1, len(x)), np.array(x[::-1], dtype = int), rotation = 30)
                plotting.plots.plt.yticks(np.linspace(0, len(x) - 1, len(x)), np.array(x[::-1], dtype = int), rotation = 30)
                plotting.plots.plt.xlabel("KE (MeV)")
                plotting.plots.plt.ylabel("KE (MeV)")

                for (i, j), z in np.ndenumerate(cov[k]):
                    plotting.plots.plt.gca().text(j, i, f"{z:.1g}", ha='center', va='center', fontsize = 10, color = "red")
                book.save()
        return {k : np.sqrt(np.diag(cov[k])) for k in cov}


    def tables(self, xs_nominal : dict, sys : dict, name : str) -> dict[pd.DataFrame]:
        """ Build per-process tables combining data statistics and one systematic.

            Args:
                xs_nominal (dict): Nominal cross sections and data-statistical uncertainties.
                sys (dict): Absolute systematic uncertainties by process.
                name (str): Label for the systematic uncertainty column.

            Returns:
                dict[pd.DataFrame]: Formatted uncertainty tables keyed by process.
        """
        x = self.args.energy_slices.pos_overflow[1:-1] - (self.args.energy_slices.width/2)

        KEs = pd.Series(np.array(x, dtype = int), name = "$KE$ (MeV)")

        data_stat_err = pd.DataFrame({p : xs_nominal["pdsp"][p][1] for p in xs_nominal["pdsp"]})

        err = pd.DataFrame({p : sys[p] for p in xs_nominal["pdsp"]})

        tables = {}
        for p in xs_nominal["pdsp"]:
            d = data_stat_err[p] / xs_nominal["pdsp"][p][0]
            d.name = "Data stat"

            e = err[p] / xs_nominal["pdsp"][p][0]
            e.name = name
        
            t = pd.Series(common.quadsum([d, e], 0))
            t.name = "Total"
        
            table = pd.concat([KEs, t, d, e], axis = 1).sort_values(by = ["$KE$ (MeV)"])

            avg = table.mean().astype(object)
            avg["$KE$ (MeV)"] = "average"
            tables[p] = pd.concat([table, pd.DataFrame(avg).T]).reset_index(drop = True)

        return tables


class DataAnalysis(ABC):
    """ Base workflow that creates varied analysis inputs and compares fitted results.

        Args:
            args (argparse.Namespace): Analysis and variation configuration.
    """
    def __init__(self, args : argparse.Namespace) -> None:
        """ Store a copy of the analysis configuration.

            Args:
                args (argparse.Namespace): Analysis and variation configuration.
        """
        self.args = common.dill_copy(args)
        pass

    @property
    @abstractmethod
    def name(self):
        """ Return the directory-name prefix identifying this systematic variation.

            Returns:
                str: Name used to identify varied input directories.
        """
        pass

    @classmethod
    def create_new_ais():
        """ Define the subclass hook for creating varied analysis inputs.
        """
        pass


    def __run_analysis(self, new_input):
        """ Run the cross-section analysis on a directory of varied inputs.

            Args:
                new_input (str): Directory containing the varied analysis-input files.

            Returns:
                dict: Fitted cross sections for the PDSP sample.
        """
        args_copy = common.dill_copy(self.args)
        inp = {}
        for f in common.ls_recursive(new_input):
            inp[f.split("/")[-1].split(".")[0].split("analysis_input_")[-1].split("_selected")[0]] = os.path.abspath(f)
        args_copy.analysis_input = inp
        args_copy.pdsp = True
        args_copy.toy_template = None
        args_copy.out = ""
        args_copy.all = False
        xs = xs_analyse.analyse(args_copy, False)["pdsp"]
        return xs


    def run_analysis(self, path):
        """ Analyze the low and high systematic-variation input directories.

            Args:
                path (str): Parent directory containing the variation subdirectories.

            Returns:
                dict: Low/high fitted cross sections keyed by variation.
        """
        xs = {i : self.__run_analysis(f"{path}{self.name}_{i}/") for i in ["low", "high"]}
        return xs


    def plot_results(self, xs_nominal : dict, result : dict, book : plotting.PlotBook = plotting.PlotBook.null()):
        """ Plot low/high varied cross sections against the nominal result.

            Args:
                xs_nominal (dict): Nominal cross sections keyed by sample and process.
                result (dict): Low/high systematic-variation cross sections.
                book (plotting.PlotBook): Destination for comparison plots.
        """
        for p in xs_nominal["pdsp"]:
            plotting.plot_xs_cpmparison({"low" : result["low"][p], "nominal" : xs_nominal["pdsp"][p], "high" : result["high"][p]}, self.args.energy_slices, p, cv_only = True, marker_size = 12)
            book.save()
        return

    @staticmethod
    def calculate_sys_error(xs):
        """ Estimate symmetric uncertainty as half the difference between variations.

            Args:
                xs (dict): Low/high cross-section results by process.

            Returns:
                dict: Absolute symmetric uncertainty by process.
        """
        return {p : abs((xs["high"][p][0] - xs["low"][p][0]) / 2) for p in xs["low"]}


    @staticmethod
    def calculate_sys_error_asym(xs_nominal : dict, xs : dict):
        """ Calculate low/high deviations from the nominal cross-section values.

            Args:
                xs_nominal (dict): Nominal cross sections keyed by process.
                xs (dict): Low/high varied cross sections keyed by variation and process.

            Returns:
                dict: Low/high absolute deviations from nominal by process.
        """
        errs = {}
        for p in xs_nominal["pdsp"]:
            e_low = xs_nominal["pdsp"][p][0] - xs["low"][p][0]
            e_high = xs_nominal["pdsp"][p][0] - xs["high"][p][0]

            err_high = np.max(abs(np.array([np.where(e_low < 0, e_low, 0), np.where(e_high < 0, e_high, 0)])), 0)

            err_low = np.max(np.array([np.where(e_low >= 0, e_low, 0), np.where(e_high >= 0, e_high, 0)]), 0)
            errs[p] = {"low" : err_low, "high" : err_high}
        return errs


    def data_analysis_tables(self, xs_nominal : dict, err_asym : dict, name : str):
        """ Create per-process tables with data-statistical and asymmetric errors.

            Args:
                xs_nominal (dict): Nominal cross sections and data-statistical uncertainties.
                err_asym (dict): Low/high absolute systematic deviations by process.
                name (str): Label for the systematic columns.

            Returns:
                dict[pd.DataFrame]: Formatted error tables keyed by process.
        """
        x = self.args.energy_slices.pos_overflow[1:-1] - (self.args.energy_slices.width/2)

        KEs = pd.Series(np.array(x, dtype = int), name = "$KE$ (MeV)")

        data_stat_err = pd.DataFrame({p : xs_nominal["pdsp"][p][1] for p in xs_nominal["pdsp"]})

        err_low = pd.DataFrame({p : err_asym[p]["low"] for p in xs_nominal["pdsp"]})
        err_high = pd.DataFrame({p : err_asym[p]["high"] for p in xs_nominal["pdsp"]})

        tables = {}
        for p in xs_nominal["pdsp"]:
            d = data_stat_err[p] / xs_nominal["pdsp"][p][0]
            d.name = "Data stat"

            l = err_low[p] / xs_nominal["pdsp"][p][0]
            l.name = name + " low"
            h = err_high[p] / xs_nominal["pdsp"][p][0]
            h.name = name + " high"
        
            t = pd.Series(common.quadsum([d, l, h], 0))
            t.name = "Total"
        
            table = pd.concat([KEs, t, d, l, h], axis = 1).sort_values(by = ["$KE$ (MeV)"])

            avg = table.mean().astype(object)
            avg["$KE$ (MeV)"] = "average"
            tables[p] = pd.concat([table, pd.DataFrame(avg).T]).reset_index(drop = True)

        return tables


class NuisanceParameters:
    """ Compare fits with nuisance parameters fixed and profiled to estimate MC statistics.

        Args:
            args (argparse.Namespace): Analysis configuration used to run the fixed- and profiled-nuisance fits.
    """
    def __init__(self, args : argparse.Namespace) -> None:
        """ Store a copy of the analysis configuration.

            Args:
                args (argparse.Namespace): Analysis and fit configuration.
        """
        self.args = common.dill_copy(args)
        pass


    def __run_analysis(self, np : bool = False):
        """ Run the analysis with nuisance parameters fixed or profiled.

            Args:
                np (bool): Whether to profile nuisance parameters and include their MC-statistical effects.

            Returns:
                dict: Fitted cross sections for the PDSP sample.
        """
        args_copy = common.dill_copy(self.args)
        args_copy.fit["mc_stat_unc"] = True
        args_copy.fit["fix_np"] = not np
        args_copy.bkg_sub_mc_stat = np
        args_copy.unfolding["mc_stat_unc"] = np

        # print(f'{args_copy.fit["fix_np"]=}')
        # print(f"{args_copy.bkg_sub_mc_stat=}")
        # print(f'{args_copy.unfolding["mc_stat_unc"]=}')

        args_copy.pdsp = True
        args_copy.toy_template = None
        args_copy.out = ""
        args_copy.all = False

        xs = xs_analyse.analyse(args_copy, False)["pdsp"]
        return xs


    def run_experiment(self):
        """ Run both fixed- and profiled-nuisance fits.

            Returns:
                dict: Cross-section results for the two nuisance-parameter configurations.
        """
        xs = self.__run_analysis(False)
        xs_np = self.__run_analysis(True)
        return {"no_np" : xs, "np" : xs_np}


    def calculate_sys_error(self, result : dict):
        """ Derive the MC-statistical uncertainty from the two fit uncertainties.

            Args:
                result (dict): Results from ``run_experiment``.

            Returns:
                dict: MC-statistical uncertainty by process.
        """
        np_sys = {}
        for p in result["no_np"]:
            np_sys[p] = np.sqrt(result["np"][p][1]**2 - result["no_np"][p][1]**2)
        return np_sys


    def plot_xs_mc_stat(self, result, book : plotting.PlotBook = plotting.PlotBook.null()):
        """ Plot cross sections with data-only and data-plus-MC statistical errors.

            Args:
                result (dict): Results from ``run_experiment``.
                book (plotting.PlotBook): Destination for comparison plots.
        """
        for p in result["np"]:
            plotting.plot_xs_cpmparison({"Data stat + MC stat" : result["np"][p], "Data stat" : result["no_np"][p]}, self.args.energy_slices, process = p)
            book.save()


    def mc_stat_tables(self, result : dict):
        """ Build per-process tables of data, MC, and total statistical errors.

            Args:
                result (dict): Results from ``run_experiment``.

            Returns:
                dict: Formatted statistical-error tables keyed by process.
        """
        x = self.args.energy_slices.pos_overflow[1:-1] - (self.args.energy_slices.width/2)

        KEs = pd.Series(np.array(x, dtype = int), name = "$KE$ (MeV)")

        mc_stat_err = self.calculate_sys_error(result)
        data_stat_err = {p : result["no_np"][p][1] for p in result["no_np"]}

        tables = {}
        for p in data_stat_err:
            d = pd.DataFrame(data_stat_err)[p] / result["no_np"][p][0]
            d.name = "Data stat"
            m = pd.DataFrame(mc_stat_err)[p] / result["no_np"][p][0]
            m.name = "MC stat"

            t = pd.Series(common.quadsum([d, m], 0))
            t.name = "Total"
        
            table = pd.concat([KEs, t, d, m], axis = 1).sort_values(by = ["$KE$ (MeV)"])

            avg = table.mean().astype(object)
            avg["$KE$ (MeV)"] = "average"
            tables[p] = pd.concat([table, pd.DataFrame(avg).T]).reset_index(drop = True)
        return tables


class UpstreamCorrectionSystematic(DataAnalysis):
    """ Create input samples with upstream-energy-loss parameters shifted by one sigma.
    """
    name = "upstream_loss_1_sigma"

    def create_new_ais(self, outdir : str):
        """ Generate low/high analysis inputs using shifted upstream-loss parameters.

            Args:
                outdir (str): Parent directory for the varied analysis inputs.
        """
        upl = {
            "low" : {f"p{i}" : self.args.upstream_loss_correction_params["value"][f"p{i}"] - self.args.upstream_loss_correction_params["error"][f"p{i}"] for i in range(self.args.upstream_loss_response.n_params)},
            "high" : {f"p{i}" : self.args.upstream_loss_correction_params["value"][f"p{i}"] + self.args.upstream_loss_correction_params["error"][f"p{i}"] for i in range(self.args.upstream_loss_response.n_params)}
        }

        args_copy = common.dill_copy(self.args)
        for k, v in upl.items():
            args_copy.upstream_loss_correction_params["value"] = v
            args_copy.out = f"{outdir}{self.name}_{k}/"
            xs_analysis_input.main(args_copy)
        return


class BeamReweightSystematic(DataAnalysis):
    """ Create input samples with beam-reweight parameters shifted by one sigma.
    """
    name = "beam_reweight_1_sigma"

    def create_new_ais(self, outdir : str):
        """ Generate low/high analysis inputs using shifted beam-reweight parameters.

            Args:
                outdir (str): Parent directory for the varied analysis inputs.
        """
        cfg = {
            "low" : {k : v["value"] - v["error"] for k, v in self.args.beam_reweight["params"].items()},
            "high" : {k : v["value"] + v["error"] for k, v in self.args.beam_reweight["params"].items()}
        }

        args_copy = common.dill_copy(self.args)
        for k, v in cfg.items():
            for p in args_copy.beam_reweight["params"]:
                args_copy.beam_reweight["params"][p]["value"] = v[p]
            print(args_copy.beam_reweight["params"])
            args_copy.out = f"{outdir}{self.name}_{k}/"
            xs_analysis_input.main(args_copy)
        return


class ShowerEnergyCorrectionSystematic(DataAnalysis):
    """ Recompute pi0 selections after shifting shower-energy correction parameters.

        Args:
            args (argparse.Namespace): Analysis and shower-energy-correction configuration for the varied selections.
    """
    name = "shower_energy_correction_1_sigma"

    def __init__(self, args: argparse.Namespace) -> None:
        """ Configure single-threaded processing for shower-selection variations.

            Args:
                args (argparse.Namespace): Analysis and selection configuration.
        """
        super().__init__(args)
        self.args.batches = None
        self.args.events = None
        self.args.threads = 1


    @staticmethod
    def __run(i : int, file : str, n_events : int, start : int, selected_events, args : dict):
        """ Calculate low/high pi0-selection masks for one ntuple chunk.

            Args:
                i (int): Index of the chunk being processed.
                file (str): Input ntuple file path.
                n_events (int): Maximum number of events to read.
                start (int): First event index to read.
                selected_events (Any): Optional event subset supplied by the processing runner.
                args (dict): Per-chunk selection and correction configuration.

            Returns:
                dict: Low/high selection masks and the source file name.
        """
        print(f"starting process {i}")
        sample = "data" if args["data"] is True else "mc"

        events =  ntuple.Data(file, n_events, start, args["ntuple_type"], args["pmom"])

        for s, a in zip(args["beam_selection"]["selections"].values(), args["beam_selection"][f"{sample}_arguments"].values()):
            mask = s(events, **a)
            events.filter([mask], [mask])
        photon_masks = {}
        if args["valid_pfo_selection"] is True:
            for k, s, a in zip(args["photon_selection"]["selections"].keys(), args["photon_selection"]["selections"].values(), args["photon_selection"][f"{sample}_arguments"].values()):
                photon_masks[k] = s(events, **a)
        photon_candidates = selection.combine_masks(photon_masks)

        output = {}
        for name, sign in zip(["low", "high"], [1, -1]):
            masks = {}
            selection_args_copy = common.dill_copy(args["pi0_selection"][f"{sample}_arguments"].values())
            for (s, f), a in zip(args["pi0_selection"]["selections"].items(), selection_args_copy):
                if s == "Pi0MassSelection":
                    a["correction_params"]["value"] = {f"p{i}" : a["correction_params"]["value"][f"p{i}"] + sign * a["correction_params"]["error"][f"p{i}"] for i in range(len(a["correction_params"]["error"]))}
                mask = f(events, **a, photon_mask = photon_candidates)
                masks[s] = mask
            output[name] = masks
        output["name"] = file

        print(f"finished process {i}")
        return output


    def __merge(self, output : dict) -> dict:
        """ Merge processing chunks separately for each input ntuple file.

            Args:
                output (dict): Per-chunk selection masks and source file names.

            Returns:
                dict: Merged low/high selections for each source file.
        """
        names = np.unique([o["name"] for o in output])

        split_output = {n : [] for n in names}
        for n in names:
            for o in output:
                if o["name"] == n:
                    split_output[n].append(o)

        merged_output = []
        for s in split_output:
            o = processing.merge_outputs(split_output[s])
            if type(o["name"]) == list:
                o["name"] = o["name"][0] 
            merged_output.append(o)
        return merged_output


    def create_alt_pi0_selections(self):
        """ Generate and merge shifted pi0-selection masks for data and MC.

            Returns:
                dict: Per-file low/high selection masks for data and MC.
        """

        self.args.cpus = 1
        processing_args = ntuple.calculate_batches(self.args)
        for k, v in processing_args.items():
            setattr(self.args, k, v)

        print("running MC")
        output_mc = self.__merge(ntuple.run_process(self.args.ntuple_files["mc"], False, self.args, self.__run, False))
        print("running Data")
        output_data = self.__merge(ntuple.run_process(self.args.ntuple_files["data"], True, self.args, self.__run, False))
        return {"mc" : output_mc, "data" : output_data}


    def create_new_ais(self, outdir : str):
        """ Save shifted pi0 masks and build low/high analysis-input samples.

            Args:
                outdir (str): Parent directory for masks and varied analysis inputs.
        """

        os.makedirs(outdir, exist_ok = True)

        masks = self.create_alt_pi0_selections()
        io.write_dill(f"{outdir}pi0_selection_masks.dill", masks)

        args_copy = {}
        for i in ["low", "high"]:
            a = common.dill_copy(self.args)
            a.selection_masks["mc"]["pi0"] = {j["name"] : j[i] for j in masks["mc"]}

            a.selection_masks["data"]["pi0"] = {j["name"] : j[i] for j in masks["data"]}
            args_copy[i] = a

        for k, v in args_copy.items():
            v.out = f"{outdir}{self.name}_{k}/"
            xs_analysis_input.main(v)
        return


class BeamMomentumResolutionSystematic(MCMethod):
    """ Evaluate the effect of smearing reconstructed beam momentum.

        Args:
            args (argparse.Namespace): Analysis and fit configuration, including the toy template to smear.
            model (pyhf.Model): Region-fit model used to analyze the smeared sample.
            data_config (dict): Toy-data configuration passed to the MC systematic workflow.
    """
    name = "beam_momentum_resolution"

    def __init__(self, args: argparse.Namespace, model: pyhf.Model, data_config: dict) -> None:
        """ Initialize the MC workflow and cache original reconstructed momentum.

            Args:
                args (argparse.Namespace): Analysis and fit configuration.
                model (pyhf.Model): Region-fit model used for the analysis.
                data_config (dict): Toy-data configuration for experiments.
        """
        super().__init__(args, model, data_config)
        self.P_reco_original = np.sqrt((self.args.toy_template.KE_init_reco + Particle.from_pdgid(211).mass)**2 - (Particle.from_pdgid(211).mass)**2)


    def run_experiment(self, analysis_input_data : analysis_input.AnalysisInput, resolution : float) -> tuple[dict, dict]:
        """ Smear beam momentum and reanalyze the supplied sample.

            Args:
                analysis_input_data (analysis_input.AnalysisInput): Event sample to fit.
                resolution (float): Relative Gaussian beam-momentum resolution.

            Returns:
                tuple[dict, dict]: Cross-section estimates and uncertainties.
        """
        P_reco_smeared = self.P_reco_original * (1 + np.random.normal(0, resolution, len(self.P_reco_original)))

        KE_init_reco = kinematics.KE(P_reco_smeared, Particle.from_pdgid(211).mass)
        KE_int_reco = bethe_bloch.KE_end(KE_init_reco, self.args.toy_template.track_length_reco, 10)

        self.args.toy_template.KE_ff_reco = KE_init_reco
        self.args.toy_template.KE_init_reco = KE_init_reco
        self.args.toy_template.KE_int_reco = KE_int_reco

        if self.args.fit["single_bin"] == False:
            self.model = RegionFit.CreateModel(self.args.toy_template, self.args.energy_slices, self.args.fit["mean_track_score"], False, self.args.toy_template.weights, self.args.fit["mc_stat_unc"], True, self.args.fit["single_bin"])
        xs = self.analyse(analysis_input_data, None)
        return xs
    

class TrackLengthResolutionSystematic(MCMethod):
    """ Evaluate the effect of smearing reconstructed track length.

        Args:
            args (argparse.Namespace): Analysis and fit configuration, including the toy template to smear.
            model (pyhf.Model): Region-fit model used to analyze the smeared sample.
            data_config (dict): Toy-data configuration passed to the MC systematic workflow.
    """
    name = "track_length_resolution"

    def __init__(self, args: argparse.Namespace, model: pyhf.Model, data_config: dict) -> None:
        """ Initialize the MC workflow and cache original reconstructed track lengths.

            Args:
                args (argparse.Namespace): Analysis and fit configuration.
                model (pyhf.Model): Region-fit model used for the analysis.
                data_config (dict): Toy-data configuration for experiments.
        """
        super().__init__(args, model, data_config)
        self.track_length_original = self.args.toy_template.track_length_reco


    def run_experiment(self, analysis_input_data : analysis_input.AnalysisInput, resolution : float) -> tuple[dict, dict]:
        """ Smear track lengths and reanalyze the supplied sample.

            Args:
                analysis_input_data (analysis_input.AnalysisInput): Event sample to fit.
                resolution (float): Relative Gaussian track-length resolution.

            Returns:
                tuple[dict, dict]: Cross-section estimates and uncertainties.
        """
        track_length_smeared = self.track_length_original * (1 + np.random.normal(0, resolution, len(self.track_length_original)))

        KE_int_reco = bethe_bloch.KE_end(self.args.toy_template.KE_init_reco, track_length_smeared, 10)
        self.args.toy_template.KE_int_reco = KE_int_reco

        if self.args.fit["single_bin"] == False:
            self.model = RegionFit.CreateModel(self.args.toy_template, self.args.energy_slices, self.args.fit["mean_track_score"], False, self.args.toy_template.weights, self.args.fit["mc_stat_unc"], True, self.args.fit["single_bin"])
        xs = self.analyse(analysis_input_data, None)
        return xs
    

    def calculate_resolution(self, book : plotting.PlotBook = plotting.PlotBook.null()):
        """ Fit the MC fractional track-length residuals and estimate their width.

            Args:
                book (plotting.PlotBook): Destination for the residual-fit plot.

            Returns:
                float: Fitted central 68% width used as the relative resolution.
        """
        ai_mc = analysis_input.AnalysisInput.from_file(self.args.analysis_input["mc"])

        r = np.array(common.nandiv(ai_mc.track_length_reco - ai_mc.track_length_true, ai_mc.track_length_reco))

        y, edges = np.histogram(r, 150, [-0.5, 0.5])
        x = common.bin_centers(edges)

        p = fitting.fit(x, y,np.sqrt(y), fitting.double_crystal_ball, method = "dogbox", plot = True, xlabel = "$\\theta$", ylabel = "Counts", plot_style = "scatter")
        book.save()

        y_interp = fitting.double_crystal_ball(x, *p[0])

        w = []
        for v in [-0.5, 0.5]:
            for i in np.linspace(0, v, 10000):
                if fitting.double_crystal_ball(i, *p[0]) <= (max(y_interp) / 2):
                    print(i)
                    break
            w.append(i)

        self.resolution = max(w) - min(w)
        self.popt = p
        return self.resolution


class TheoryShape(MCMethod):
    """ Evaluate cross-section changes caused by energy-dependent theory reweighting.

        Args:
            args (argparse.Namespace): Analysis and fit configuration for the theory-shape experiments.
            model (pyhf.Model): Region-fit model used to analyze the reweighted samples.
            data_config (dict): Toy-data configuration passed to the MC systematic workflow.
    """
    def __init__(self, args: argparse.Namespace, model: pyhf.Model, data_config: dict) -> None:
        """ Initialize the MC workflow for theory-shape experiments.

            Args:
                args (argparse.Namespace): Analysis and fit configuration.
                model (pyhf.Model): Region-fit model used for the analysis.
                data_config (dict): Toy-data configuration for experiments.
        """
        super().__init__(args, model, data_config)


    def generate_KE_shape_weights(self, bins : int, shape_lims : list):
        """ Generate event weights with random shape variations over true interaction energy.

            Args:
                bins (int): Number of true-energy bins used to define the weights.
                shape_lims (list): Minimum and maximum random weight before normalization.

            Returns:
                np.ndarray: Normalized per-event shape weights.
        """
        KE_int = self.args.toy_template.KE_int_true
        hist, edges = np.histogram(KE_int[~np.isnan(KE_int)], bins = bins)
        weights = np.random.uniform(min(shape_lims), max(shape_lims), len(hist))
        scale = sum(hist) / sum(hist * weights)
        weights = scale * weights

        ind = np.digitize(KE_int, edges[1:-1])
        return weights[ind]


    def run_experiment(self, analysis_input_data : analysis_input.AnalysisInput) -> tuple[dict, dict]:
        """ Reweight a toy sample by true interaction energy and reanalyze it.

            Args:
                analysis_input_data (analysis_input.AnalysisInput): Event sample to fit.

            Returns:
                tuple[dict, dict]: Cross-section estimates and uncertainties.
        """
        weights = self.generate_KE_shape_weights(50, [0.8, 1.2])
        self.args.toy_template.weights = weights

        xs = self.analyse(analysis_input_data, None)
        return xs


class NormalisationSystematic(MCMethod):
    """ Measure fit bias under normalization variations in generated toy samples.
    """
    def run_experiment(self, config : dict) -> tuple[dict, dict]:
        """ Generate and fit one normalization-varied toy sample.

            Args:
                config (dict): Toy-generation configuration with process scale factors.

            Returns:
                tuple[dict, dict]: And true cross sections for the experiment.
        """
        x = self.args.energy_slices.pos[:-1] - self.args.energy_slices.width / 2

        ai = analysis_input.AnalysisInput.create_analysis_input_toy(toy.Toy(df = xs_toy_generator.run(config)))

        xs = self.analyse(ai, None)
        xs_sim_mod = xs_toy_generator.modify_geant_xs(scale_factors = config["pdf_scale_factors"])

        if self.args.fit["regions"]:
            xs_true = {k : xs_sim_mod.get_interpolated_curve(k)(x) for k in exclusive_proc}
        else:
            xs_true = xs_sim_mod.get_interpolated_curve(self.args.signal_process)(x)

        return xs, xs_true


    def evaluate(self, norms = [0.8, 1.2], repeats : int = 1):
        """ Run normalization tests for each process at each requested scale.

            Args:
                norms (Sequence[float]): Normalization values to test.
                repeats (int): Number of independent toys generated per scale.

            Returns:
                dict: Measured and true cross sections grouped by process and scale.
        """
        cvs = {}
        true_cvs = {}
        for target in exclusive_proc:
            print(Rule(f"normalisation systematic: {target}"))
            scales = {k : 1 for k in ['absorption', 'quasielastic', 'charge_exchange', 'double_charge_exchange', 'pion_production']}
            xs = {i : [] for i in norms}
            xs_true = {}
            for i in norms:
                for j in range(repeats):
                    if i == 1:
                        config = {k : v for k,v in self.data_config.items()}
                    else:
                        if target == "single_pion_production":
                            scales["quasielastic"] = i
                            scales["double_charge_exchange"] = i
                        else:
                            scales[target] = i
                        config = xs_fit_studies.create_config_normalisation(scales, self.data_config)

                    if repeats != 1:
                        config["seed"] = j + 1

                    output = self.run_experiment(config)
                    xs[i].append(output[0])
                    xs_true[i] = output[1]
            cvs[target] = xs
            true_cvs[target] = xs_true
        return {"cv" : cvs, "true_cv" : true_cvs}

    @staticmethod
    def plot_normalisation_test_results(results : dict, args : argparse.Namespace, xs_nominal : dict, book : plotting.PlotBook = plotting.PlotBook.null()):
        """ Plot measured and true cross sections for each normalization test.

            Args:
                results (dict): Measured and true test results by process and scale.
                args (argparse.Namespace): Analysis configuration and energy binning.
                xs_nominal (dict): Nominal cross sections used for comparison.
                book (plotting.PlotBook): Destination for normalization-test plots.
        """
        xs_sim = geant.GeantCrossSections()
        scale_factors = {"absorption" : 1, "charge_exchange" : 1, "pion_production" : 1, "double_charge_exchange" : 1, "quasielastic" : 1}
        for r in results["cv"]:
            for _, p in plotting.plots.iter_multi_plot(results["cv"][r][list(results["cv"][r].keys())[0]]):
                for n in results["cv"][r]:
                    mod_norm = {k : v for k, v in scale_factors.items()}
                    if r == "single_pion_production":
                        mod_norm["double_charge_exchange"] = n
                        mod_norm["quasielastic"] = n
                    else:
                        mod_norm[r] = n
                    mod_sim = xs_toy_generator.modify_geant_xs(scale_factors = mod_norm, modified_PDFs = None)

                    if p == "single_pion_production":
                        gxs = getattr(mod_sim, "double_charge_exchange") + getattr(mod_sim, "quasielastic")
                    else:
                        gxs = getattr(mod_sim, p)

                    plotting.plots.plot(mod_sim.KE, gxs, new_figure = False, label = f"true, $\\mathcal{{N}} = {n}$", title = f"process : {common.remove_(p)}", ylabel = "$\\sigma$ (mb)", xlabel = "$KE$ (MeV)")
                    plotting.plots.plot(args.energy_slices.pos[:-1] - args.energy_slices.width/2, results["cv"][r][n][p][0], yerr = results["cv"][r][n][p][1], xerr = args.energy_slices.width/2, marker = "x", linestyle = "", label = f"measured, $\\mathcal{{N}} = {n}$", new_figure = False)
                    plotting.plots.plt.xlim(args.energy_slices.min_pos - args.energy_slices.width, args.energy_slices.max_pos + args.energy_slices.width)
                    # plotting.plots.plt.ylim(0, 1.5 * max(results["cv"][r][n][p][0]))
                if p == "single_pion_production":
                    gxs = getattr(xs_sim, "double_charge_exchange") + getattr(xs_sim, "quasielastic")
                else:
                    gxs = getattr(xs_sim, p)
                plotting.plots.plot(mod_sim.KE, gxs, new_figure = False, label = f"true, $\\mathcal{{N}} = 1$")
                plotting.plots.plot(args.energy_slices.pos[:-1] - args.energy_slices.width/2, xs_nominal[p][0], yerr = xs_nominal[p][1], xerr = args.energy_slices.width/2, marker = "x", linestyle = "", label = f"measured, $\\mathcal{{N}} = 1$", new_figure = False)
            plotting.plots.plt.suptitle(f"normalisation test : {common.remove_(r)}")
            plotting.plots.plt.tight_layout()
            book.save()
        return

    @staticmethod
    def calculate_sys_err(results):
        """ Calculate low/high deviations between measured and true varied cross sections.

            Args:
                results (dict): Normalization-test measured and truth results.

            Returns:
                dict: Low/high absolute systematic errors by varied process.
        """
        def sys_err(r, tr):
            """ Combine absolute process deviations in quadrature.

                Args:
                    r (dict): Measured cross sections by process.
                    tr (dict): True cross-section values by process.

                Returns:
                    np.ndarray: Quadrature sum of process deviations per energy bin.
            """
            return common.quadsum([abs(r[p][0] - tr[p]) for p in r], 0)

        sys_err_low = {}
        sys_err_high = {}
        for r in results["cv"]:
            norms = list(results["cv"][r].keys())
            sys_err_low[r] = sys_err(results["cv"][r][min(norms)], results["true_cv"][r][min(norms)])
            sys_err_high[r] = sys_err(results["cv"][r][max(norms)], results["true_cv"][r][max(norms)])
        return {"low" : sys_err_low, "high" : sys_err_high}

    @staticmethod
    def total_sys_qs(sys_err : dict):
        """ Combine low/high normalization errors across targets in quadrature.

            Args:
                sys_err (dict): Low/high systematic errors grouped by target process.

            Returns:
                dict: Total low/high errors by exclusive signal process.
        """
        norm_sys_qs = {}
        for p in exclusive_proc:
            norm_sys_qs[p] = [
                common.quadsum([sys_err["low"][r][p] for r in sys_err["low"]], 0),
                common.quadsum([sys_err["high"][r][p] for r in sys_err["high"]], 0)
            ]
        return norm_sys_qs

    @staticmethod
    def total_sys_max(sys_err : dict):
        """ Select the largest low/high normalization error across targets.

            Args:
                sys_err (dict): Low/high systematic errors grouped by target process.

            Returns:
                dict: Maximum low/high errors by exclusive signal process.
        """
        norm_sys_max = {}
        for p in exclusive_proc:
            norm_sys_max[p] = [
                np.max([sys_err["low"][r][p] for r in sys_err["low"][p]], 0),
                np.max([sys_err["high"][r][p] for r in sys_err["high"][p]], 0)
            ]
        return norm_sys_max
    
    @staticmethod
    def average_results(results : dict):
        """ Replace repeated measured cross sections with their mean and mean errors.

            Args:
                results (dict): Mutable normalization-test results containing repetitions.
        """
        cv_avg = {}
        for t in results["cv"]:
            t_avg = {}
            for n in results["cv"][t]:
                cv = {}
                cv_err = {}
                for rep in results["cv"][t][n]:
                    for p in rep:
                        if p not in cv:
                            cv[p] = [rep[p][0]]
                            cv_err[p] = [rep[p][1]]
                        else:
                            cv[p].append(rep[p][0])
                            cv_err[p].append(rep[p][1])

                t_avg[n] = {k : [np.mean(np.array(v), 0), np.mean(np.array(cv_err[k]), 0)] for k, v in cv.items()}
            cv_avg[t] = t_avg

        results["cv"] = cv_avg

    @staticmethod
    def calculate_fractional_error(sys_err_avg : dict, xs_nominal : dict):
        """ Divide low/high absolute systematic errors by nominal cross sections.

            Args:
                sys_err_avg (dict): Low/high absolute uncertainties by process.
                xs_nominal (dict): Nominal cross-section values and uncertainties.

            Returns:
                dict: Low/high fractional systematic errors by process.
        """
        frac_low = {}
        frac_high = {}
        for i in xs_nominal:
            frac_low[i] = sys_err_avg["low"][i] / xs_nominal[i][0]
            frac_high[i] = sys_err_avg["high"][i] / xs_nominal[i][0]
        return {"low" : frac_low, "high" : frac_high}

    
    def create_tables(self, xs_nominal, sys):
        """ Create per-process tables for nominal and fit-inaccuracy errors.

            Args:
                xs_nominal (dict): Nominal cross sections and statistical uncertainties.
                sys (dict): Absolute and fractional systematic uncertainties.

            Returns:
                dict: Formatted uncertainty tables keyed by process.
        """
        x = args.energy_slices.pos_overflow[1:-1] - (args.energy_slices.width/2)

        KEs = pd.Series(np.array(x, dtype = int), name = "$KE$ (MeV)")

        data_stat_err = pd.DataFrame({p : xs_nominal["pdsp"][p][1] for p in xs_nominal["pdsp"]})

        process_keys = list(xs_nominal["pdsp"].keys())
        process_tags = tags.exclusive_process_tags({k : None for k in process_keys})

        tables = {}
        for p in xs_nominal["pdsp"]:
            d = data_stat_err[p] / xs_nominal["pdsp"][p][0]
            d.name = "Data stat"
            ls = pd.Series(sys["fractional"]["low"][p])
            ls.name = "Fit inaccuracy low"

            hs = pd.Series(sys["fractional"]["high"][p])
            hs.name = "Fit inaccuracy high"

            t = pd.Series(common.quadsum([d, ls, hs], 0))
            t.name = "Total"

            table = pd.concat([KEs, t, d, ls, hs], axis = 1).sort_values(by = ["$KE$ (MeV)"])

            avg = table.mean().astype(object)
            avg["$KE$ (MeV)"] = "average"
            tables[p] = pd.concat([table, pd.DataFrame(avg).T]).reset_index(drop = True)
        return tables


def create_systematics_table(out : str, data_only : dict, args : argparse.Namespace) -> dict[pd.DataFrame]:
    """ Combine saved systematic errors and central values into per-process tables.

        Args:
            out (str): Root directory containing serialized systematic results.
            data_only (dict): Central cross sections and data-statistical uncertainties.
            args (argparse.Namespace): Analysis configuration, including energy slices.

        Returns:
            dict[pd.DataFrame]: Combined uncertainty table for each process.
    """
    sys_err = {}
    for f in common.ls_recursive(out):
        if "sys.dill" in f:
            sys_err[f.split("/")[-2]] = io.read_dill(f)["systematic"]

    tables_dict = {p : {} for p in exclusive_proc}
    for k, v in data_only["pdsp"].items():
        tables_dict[k]["Data stat"] = v[1]
    for s in sys_err:
        t = pd.DataFrame(sys_err[s])
        if "low" in t.index:
            for i in ["low", "high"]:
                for x, v in t.loc[i].items():
                    tables_dict[x][systematics_label[s] + f" {i}"] = v
                    
        elif "low" in t.columns:
            for i in ["low", "high"]:
                for x, v in t[i].items():
                    tables_dict[x][systematics_label[s] + f" {i}"] = v
        else:
            for x in t.columns:
                tables_dict[x][systematics_label[s]] = t[x].values

    x = args.energy_slices.pos[:-1] - args.energy_slices.width/2

    fmt_tables = {}
    for t in tables_dict:
        table = pd.DataFrame(tables_dict[t]).T

        for s in systematics_label.values():
            filtered = table.filter(regex = s, axis = "index")
            if len(filtered) > 1:
                table = table.drop(index = filtered.index)
                mask = filtered.where(filtered == 0, False) == filtered.where(filtered != 0, False)
                mask = mask.values
                zero = mask[0] | mask[1]
                n = np.where(zero, 1, 0.5)
                table = pd.concat([table, pd.DataFrame({s : n * (filtered.loc[f"{s} low"] + filtered.loc[f"{s} high"])}).T])

        total_err = {}
        for i in table:
            total_err[i] = common.quadsum(table[i], 0)    
        total_err = pd.DataFrame(total_err, index = ["Total uncertainty (mb)"])

        total_sys_err = {}
        for i in table:
            total_sys_err[i] = common.quadsum(table[i].drop(index = "Data stat"), 0)
        total_sys_err = pd.DataFrame(total_sys_err, index = ["Total systematic uncertainty (mb)"])

        names = {c : c + " (mb)" for c in table.index}
        table = table.rename(index = names)

        fmt_tables[t] = pd.concat([pd.DataFrame({"KE (MeV)" : x, "Central value (mb)" : data_only["pdsp"][t][0]}).T, table, total_sys_err, total_err])
    return fmt_tables


def save_systematic_tables(systematic_tables : dict[pd.DataFrame], out : str):
    """ Save combined systematic tables to HDF5 and formatted LaTeX files.

        Args:
            systematic_tables (dict[pd.DataFrame]): Per-process combined uncertainty tables.
            out (str): Output directory for the table files.
    """
    for k, v in systematic_tables.items():
        v.to_hdf(out + f"systematic_table_{k}.hdf5", key = "df")
        v.style.format(
            precision = 2
              ).format(
            precision = 0, subset = (v.index[0], v.select_dtypes(float).columns)
              ).hide(
            axis = "columns"
              ).to_latex(out + f"systematic_table_{k}.tex")
    return


def plot_sys_hist(systematic_table : dict[pd.DataFrame], book : plotting.PlotBook = plotting.PlotBook.null()):
    """ Plot absolute and cumulative relative uncertainties for each process.

        Args:
            systematic_table (dict[pd.DataFrame]): Per-process combined uncertainty tables.
            book (plotting.PlotBook): Destination for uncertainty plots.
    """
    for t in systematic_table:
        plotting.plots.plt.figure()
        c = 0
        for i  in systematic_table[t].T:
            if i in ["KE (MeV)", "Central value (mb)"] : continue
            if i in ["Total uncertainty (mb)", "Total systematic uncertainty (mb)"]:
                color = "k"
            else:
                color = f"C{c}"
                c+=1
            if i  == "Total uncertainty (mb)":
                linestyle = "dashdot"
            else:
                linestyle = "-"
            plotting.plots.plot(systematic_table[t].loc["KE (MeV)"], systematic_table[t].loc[i], new_figure = False, xlabel = "KE (MeV)", ylabel = "Uncertainty (mb)", label = i.split(" (mb)")[0], title = common.remove_(t).capitalize(), style = "step", linestyle = linestyle, color = color)
            plotting.plots.plt.legend(ncols = 2)
        plotting.plots.plt.ylim(0, 1.5 * max(plotting.plots.plt.gca().get_ylim()))
        book.save()

    table_filtered = {k : v[~v.index.isin(["KE (MeV)", "Central value (mb)", "Total uncertainty (mb)", "Total systematic uncertainty (mb)"])] for k, v in systematic_table.items()}
    order = {k : v.sum(axis = 1).values.argsort() for k, v in table_filtered.items()}
    total = {k : v.sum(axis = 0) for k, v in table_filtered.items()}


    for t in systematic_table:
        plotting.plots.plt.figure()
        prev = 0
        for e, i in enumerate(reversed(order[t])):
            v = table_filtered[t].iloc[i]
            prev = prev + v.T.values
            plotting.plots.plot(systematic_table[t].loc["KE (MeV)"].values, prev, new_figure = False, xlabel = "KE (MeV)", ylabel = "Relative uncertainty (mb)", label = v.name.split(" (mb)")[0], title = common.remove_(t).capitalize(), style = "bar", linestyle = "-", color = f"C{i}", zorder = 100-e)
        plotting.plots.plt.legend(ncols = 2)
        plotting.plots.plt.ylim(0, 1.5 * max(plotting.plots.plt.gca().get_ylim()))
        book.save()

    for t in systematic_table:
        plotting.plots.plt.figure()
        prev = 0
        for e, i in enumerate(reversed(order[t])):
            v = table_filtered[t].iloc[i]
            prev = prev + v.T.values
            plotting.plots.plot(systematic_table[t].loc["KE (MeV)"].values, prev/total[t], new_figure = False, xlabel = "KE (MeV)", ylabel = "Relative uncertainty", label = v.name.split(" (mb)")[0], title = common.remove_(t).capitalize(), style = "bar", linestyle = "-", color = f"C{i}", zorder = 100-e)
        plotting.plots.plt.legend(ncols = 2)
        plotting.plots.plt.ylim(0, 1.5 * max(plotting.plots.plt.gca().get_ylim()))
        book.save()

    return


def final_plots(cv, systematics_table : dict[pd.DataFrame], energy_slices, book : plotting.PlotBook = plotting.PlotBook.null(), alt_xs : bool = False):
    """ Plot final cross sections and calculate goodness-of-fit results.

        Args:
            cv (dict): Central cross sections and statistical errors by process.
            systematics_table (dict[pd.DataFrame]): Combined systematic tables by process.
            energy_slices (slices.Slices): Energy binning used in the cross-section result.
            book (plotting.PlotBook): Destination for final comparison plots.
            alt_xs (bool): Whether to also compare against the alternative Geant prediction.

        Returns:
            Nominal: And alternative goodness-of-fit tables.
    """
    goodness_of_fit = {}
    if alt_xs:
        xs_alt = geant.GeantCrossSections(geant.GEANT_XS, energy_range = [energy_slices.min_pos - energy_slices.width, energy_slices.max_pos])
    goodness_of_fit_alt = {}
    for p in cv:
        xs = {
            "ProtoDUNE SP: Data Stat + Sys Error" : cv[p],
            "" : [cv[p][0], systematics_table[p].loc["Total uncertainty (mb)"]]
        }
        plotting.plot_xs_cpmparison(xs, energy_slices, p, simulation_label = "Geant4 v10.6", colors = {k : f"C0" for k in xs}, chi2 = False)
        goodness_of_fit[p] = plotting.hyp_test_xs(cv[p][0], systematics_table[p].loc["Total uncertainty (mb)"], p, energy_slices)
        if alt_xs:
            xs_alt.plot(p, "red", label = "Geant4 v10.6, $\\pi^{\\pm}$:$2^{nd}$ $KE > 100 MeV$")
            goodness_of_fit_alt[p] = plotting.hyp_test_xs(cv[p][0], systematics_table[p].loc["Total uncertainty (mb)"], p, energy_slices, geant.GEANT_XS)
        book.save()
    return pd.DataFrame(goodness_of_fit), pd.DataFrame(goodness_of_fit_alt)


def can_run(systematic : str):
    """ Check whether a systematic was selected to run and has not been skipped.

        Args:
            systematic (str): Systematic name to test.

        Returns:
            bool: Whether the systematic is selected for execution.
    """
    run = ((systematic not in args.skip) and (systematic in args.run)) or ("all" in args.run)
    if run is True:
        print(Rule(common.remove_(systematic)))
    return run


def can_regen(dir : str):
    """ Determine whether results in a directory should be regenerated.

        Args:
            dir (str): Systematic output directory to inspect.

        Returns:
            bool: Whether calculations should be rerun instead of loading saved results.
    """
    if os.path.exists(dir) and (args.regen is False):
        if len(common.ls_recursive(dir)) > 0:
            for f in common.ls_recursive(dir):
                if ("results.dill" in f):
                    return False
            return True
        else:
            return True
    else:
        return True

@common.timer
def main(args : argparse.Namespace):
    """ Run selected systematic studies and optionally combine and plot their results.

        Args:
            args (argparse.Namespace): Resolved analysis, systematic, and output configuration.
    """
    plotting.styles.PlotStyler.set_plot_style(dark = True)
    out = args.out + "systematics/"
    os.makedirs(out, exist_ok = True)

    print(f"{args.skip=}")
    print(f"{args.run=}")

    if args.toy_template:
        args.toy_template = analysis_input.AnalysisInput.create_analysis_input_toy(toy.Toy(file = args.toy_template))
        model = RegionFit.CreateModel(args.toy_template, args.energy_slices, args.fit["mean_track_score"], False, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])
    
        toy_nominal = toy.Toy(df = xs_toy_generator.run(args.toy_data_config))
        analysis_input_nominal = analysis_input.AnalysisInput.create_analysis_input_toy(toy_nominal)


    if ("all" not in args.skip) or ("all" in args.run):
        if can_run("mc_stat"):
            outdir = out + "mc_stat/"
            os.makedirs(outdir, exist_ok = True)

            mc_stat = NuisanceParameters(args)
            if can_regen(outdir):
                result = mc_stat.run_experiment()
                # sys = mc_stat.calculate_sys_error(result)
                io.write_dill(f"{outdir}results.dill", result)
                # SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = io.read_dill(f"{outdir}results.dill")
                # sys = io.read_dill(f"{outdir}sys.dill")

            sys = mc_stat.calculate_sys_error(result)
            save_systematic_error(sys, None, f"{outdir}sys.dill")
            with plotting.PlotBook(f"{outdir}plotting.plots.pdf") as book:
                mc_stat.plot_xs_mc_stat(result, book)
            tables = mc_stat.mc_stat_tables(result)
            save_tables(tables, outdir, 2)

        if can_run("shower_energy"):
            outdir = out + "shower_energy/"
            sc = ShowerEnergyCorrectionSystematic(args)
            if can_regen(outdir):
                sc.create_new_ais(outdir)
                result = sc.run_analysis(outdir)
                # sys = sc.calculate_sys_error_asym(args.cv, result)
                io.write_dill(f"{outdir}results.dill", result)
                # SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = io.read_dill(f"{outdir}results.dill")
                # sys = io.read_dill(f"{outdir}sys.dill")["systematic"]

            sys = sc.calculate_sys_error_asym(args.cv, result)
            save_systematic_error(sys, None, f"{outdir}sys.dill")
            with plotting.PlotBook(f"{outdir}plotting.plots.pdf") as book:
                sc.plot_results(args.cv, result, book)
            tables = sc.data_analysis_tables(args.cv, sys, "Shower energy")
            save_tables(tables, outdir, 2)

        if can_run("upstream"):
            outdir = out + "upstream/"
            upl = UpstreamCorrectionSystematic(args)

            if can_regen(outdir):
                upl.create_new_ais(outdir)
                result = upl.run_analysis(outdir)
                # sys = upl.calculate_sys_error_asym(args.cv, result)

                io.write_dill(f"{outdir}results.dill", result)
                # SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = io.read_dill(f"{outdir}results.dill")
                # sys = io.read_dill(f"{outdir}sys.dill")

            sys = upl.calculate_sys_error_asym(args.cv, result)
            save_systematic_error(sys, None, f"{outdir}sys.dill")
            with plotting.PlotBook(f"{outdir}plotting.plots.pdf") as book:
                upl.plot_results(args.cv, result, book)
            tables = upl.data_analysis_tables(args.cv, sys, "Upstream")
            save_tables(tables, outdir, 2)

        if can_run("beam_reweight"):
            outdir = out + "beam_reweight/"
            brw = BeamReweightSystematic(args)

            if can_regen(outdir):
                brw.create_new_ais(outdir)
                result = brw.run_analysis(outdir)
                # sys = brw.calculate_sys_error_asym(args.cv, result)

                io.write_dill(f"{outdir}results.dill", result)
                # SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = io.read_dill(f"{outdir}results.dill")
                # sys = io.read_dill(f"{outdir}sys.dill")["systematic"]

            sys = brw.calculate_sys_error_asym(args.cv, result)
            save_systematic_error(sys, None, f"{outdir}sys.dill")
            with plotting.PlotBook(f"{outdir}plotting.plots.pdf") as book:
                brw.plot_results(args.cv, result, book)
            tables = brw.data_analysis_tables(args.cv, sys, "Reweight")
            save_tables(tables, outdir, 2)

        if can_run("track_length"):
            outdir = out + "track_length/"
            os.makedirs(outdir, exist_ok = True)
            trk = TrackLengthResolutionSystematic(args, model, args.toy_data_config)

            with plotting.PlotBook(outdir + "trk_res.pdf") as book:
                trk.calculate_resolution(book)
            table = {}
            for i in range(len(trk.popt[0])):
                table[f"$p_{{{i}}}$"] = common.round_value_to_error(trk.popt[0][i], trk.popt[1][i])
            pd.DataFrame(table, index = [0]).style.hide(axis = "index").to_latex(outdir + "fit_params.tex")

            if can_regen(outdir):
                result = trk.evaluate(args.n_throws, analysis_input_data = analysis_input_nominal, resolution = trk.resolution)
                io.write_dill(f"{outdir}results.dill", result)
            else:
                result = io.read_dill(f"{outdir}results.dill")

            if len(result) > 1:
                with plotting.PlotBook(f"{outdir}cov_mat") as book:
                    sys = trk.calculate_sys_cov(result, book)
                save_systematic_error(sys, None, f"{outdir}sys.dill")

                tables = trk.tables(args.cv, sys, "Track length")
                save_tables(tables, outdir, 3)
            else:
                print("only one experiment was ran, run more experiments to produce the systematic errors.")


        if can_run("beam_res"):
            outdir = out + "beam_res/"
            os.makedirs(outdir, exist_ok = True)
            # resolution = 2.5/100
            resolution = 1.2/100
            bm = BeamMomentumResolutionSystematic(args, model, args.toy_data_config)

            if can_regen(outdir):
                result = bm.evaluate(args.n_throws, analysis_input_data = analysis_input_nominal, resolution = resolution)
                io.write_dill(f"{outdir}results.dill", result)
            else:
                result = io.read_dill(f"{outdir}results.dill")

            if len(result) > 1:
                with plotting.PlotBook(f"{outdir}cov_mat") as book:
                    sys = bm.calculate_sys_cov(result, book)
                save_systematic_error(sys, None, f"{outdir}sys.dill")

                tables = bm.tables(args.cv, sys, "Beam momentum")
                save_tables(tables, outdir, 3)
            else:
                print("only one experiment was ran, run more experiments to produce the systematic errors.")

        if can_run("theory"):
            outdir = out + "theory/"
            os.makedirs(outdir, exist_ok = True)

            ts = TheoryShape(args, model, args.toy_data_config)

            if can_regen(outdir):
                result = ts.evaluate(args.n_throws, analysis_input_data = analysis_input_nominal)
                print(f"saving: {outdir}results.dill")
                io.write_dill(f"{outdir}results.dill", result)
            else:
                result = io.read_dill(f"{outdir}results.dill")

            if len(result) > 1:
                with plotting.PlotBook(f"{outdir}cov_mat") as book:
                    sys = ts.calculate_sys_cov(result, book)
                save_systematic_error(sys, None, f"{outdir}sys.dill")

                tables = ts.tables(args.cv, sys, "Theory")
                save_tables(tables, outdir, 3)
            else:
                print("only one experiment was ran, run more experiments to produce the systematic errors.")

        if can_run("fit_inaccuracy"):
            outdir = out + "fit_inaccuracy/"
            os.makedirs(outdir, exist_ok = True)
            
            norm_sys = NormalisationSystematic(args, model, args.toy_data_config)
            
            xs_nominal = norm_sys.analyse(analysis_input_nominal, None)

            if can_regen(outdir):
                if not os.path.isfile(f"{outdir}results.dill"):    
                    result = norm_sys.evaluate([0.8, 1.2], args.fit_inaccuracy_repeats)
                    io.write_dill(f"{outdir}results.dill", result)

                result = io.read_dill(f"{outdir}results.dill")
            else:
                result = io.read_dill(f"{outdir}results.dill")

            if len(result) > 1:
                NormalisationSystematic.average_results(result)
                print(result)
                sys_err = NormalisationSystematic.calculate_sys_err(result)
                frac_err = NormalisationSystematic.calculate_fractional_error(sys_err, xs_nominal)
                save_systematic_error(sys_err, frac_err, f"{outdir}sys.dill")
                sys = io.read_dill(f"{outdir}sys.dill")

                with plotting.PlotBook(f"{outdir}plots", True) as book:
                    #! use styler properly
                    plotting.styles.PlotStyler.set_plot_style(dark = False, extend_colors = True)
                    NormalisationSystematic.plot_normalisation_test_results(result, args, xs_nominal, book)
                    plotting.styles.PlotStyler.set_plot_style(dark = True, extend_colors = False)

                tables = norm_sys.create_tables(args.cv, sys)
                save_tables(tables, outdir, 2)
            else:
                print("only one experiment was ran, run more experiments to produce the systematic errors.")


    if args.plot:
        outdir = out + "combined/"
        os.makedirs(outdir, exist_ok = True)
        tables = create_systematics_table(out, args.cv, args)
        save_systematic_tables(tables, outdir)

        with plotting.PlotBook(outdir + "systematic_plotting.plots.pdf") as book:
            with plotting.plots.matplotlib.rc_context({"axes.prop_cycle" : plotting.plots.plt.cycler("color", plotting.plots.matplotlib.colormaps["tab20"].colors)}):
                plot_sys_hist(tables, book)
        with plotting.PlotBook(outdir + "final_plotting.plots.pdf", watermark = "DUNE: Work in Progress") as book:
            table, table_alt = final_plots(args.cv["pdsp"], tables, args.energy_slices, book, alt_xs = False)

        process_tags = tags.exclusive_process_tags({k : None for k in table.columns})
        table = table.rename(index = {"w_chi2" : "$\\chi^{2}/ndf$", "p" : "$p$"}, columns={t : tags[t].name_simple.capitalize() for t in tags})
        table.style.format("{:.3g}").to_latex(outdir + "goodness_of_fit.tex")

        if len(table_alt) > 0:
            table_alt = table_alt.rename(index = {"w_chi2" : "$\\chi^{2}/ndf$", "p" : "$p$"}, columns={t : tags[t].name_simple.capitalize() for t in tags})
            table_alt.style.format("{:.3g}").to_latex(outdir + "goodness_of_fit_alt.tex")
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Estimate Systematics for the cross section analysis")
    application.ApplicationArguments.config(parser)
    application.ApplicationArguments.output(parser, "systematic/")

    parser.add_argument("--cv", "-v", dest = "cv", type = str, default = None, help = "plot systematics with central value measurement", required = True)

    parser.add_argument("--toy_template", "-t", dest = "toy_template", type = str, help = "toy template hdf5 file", required = False)
    parser.add_argument("--toy_data_config", "-d", dest = "toy_data_config", type = str, help = "json config for toy data", required = False)

    parser.add_argument("--skip", type = str, nargs = "+", default = [], choices = systematics)
    parser.add_argument("--run", type = str, nargs = "+", default = [], choices = systematics)

    parser.add_argument("--regen", "-r", dest = "regen", action = "store_true", help = "fully rerun systematic tests if results already exist")

    parser.add_argument("--toy_throws", dest = "n_throws", type = int, default = 50, help = "number of throws for systematics using the toy (excluding fit inaccuracy)")
    parser.add_argument("--fit_inaccuracy_repeats", dest = "fit_inaccuracy_repeats", type = int, default = 5, help = "number of repeats for fit inaccuracy systematic")

    parser.add_argument("--plot", "-p", dest = "plot", action = "store_true", default = None, help = "plot systematics with central value measurement")

    args = application.ApplicationArguments.resolve_args(parser.parse_args())

    if ("all" in args.run) or ("fit_inaccuracy" in args.run) or ("track_length" in args.run) or ("beam_res" in args.run) or ("theory" in args.run):
        if not args.toy_template:
            raise Exception("--toy_template must be specified")
        if not args.toy_data_config:
            raise Exception("--toy_data_config must be specified")        
    if args.toy_data_config:
        args.toy_data_config = common.read_json(args.toy_data_config)

    args.cv = io.read_dill(args.cv)

    print(vars(args))
    print(warnings.warn("App is deprecated and likely no longer works!"))
    main(args)