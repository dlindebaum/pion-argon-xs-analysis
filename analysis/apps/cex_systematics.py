#!/usr/bin/env python3
"""
Created on: 09/04/2024 13:53

Author: Shyam Bhuller

Description: app to calculate systematic uncertainties for cross section measurement
"""
from abc import ABC, abstractmethod
from argparse import Namespace

import pandas as pd
import numpy as np

from rich import print
from rich.rule import Rule

from python.analysis import cross_section, Plots
from python.analysis.Master import DictToHDF5
from python.analysis.Utils import dill_copy, quadsum
from apps import cex_toy_generator, cex_analyse, cex_fit_studies, cex_analysis_input, cex_beam_reweight, cex_upstream_loss


exclusive_proc = ["absorption", "charge_exchange", "single_pion_production", "pion_production"]


def SaveSystematicError(systematic : dict, fractional : dict, out : str):
    return cross_section.SaveObject(out, {"systematic" : systematic, "fractional": fractional})


class MCMethod(ABC):
    def __init__(self, args : cross_section.argparse.Namespace, model : cross_section.pyhf.Model, data_config : dict) -> None:
        self.args = args
        self.model = model
        self.data_config = data_config
        pass

    @cross_section.timer
    def Analyse(self, analysis_input : cross_section.AnalysisInput, book : Plots.PlotBook = Plots.PlotBook.null):

        region_fit_result = cex_analyse.RegionFit(analysis_input, self.args.energy_slices, self.args.fit["mean_track_score"], self.model, mc_stat_unc = self.args.fit["mc_stat_unc"], single_bin = self.args.fit["single_bin"])

        _, histograms_reco_obs, histograms_reco_obs_err = cex_analyse.BackgroundSubtraction(analysis_input, self.args.signal_process, self.args.energy_slices, region_fit_result, self.args.fit["single_bin"], self.args.fit["regions"], self.args.toy_template, self.args.bkg_sub_err, book)


        if self.args.fit["regions"]:
            histograms_reco_obs = {**histograms_reco_obs, **histograms_reco_obs["int_ex"]}
            histograms_reco_obs.pop("int_ex")
            histograms_reco_obs_err = {**histograms_reco_obs_err, **histograms_reco_obs_err["int_ex"]}
            histograms_reco_obs_err.pop("int_ex")

        unfolding_result = cex_analyse.Unfolding(histograms_reco_obs, histograms_reco_obs_err, self.args.toy_template, self.args.unfolding, self.args.signal_process, len(analysis_input.KE_init_reco)/len(self.args.toy_template.KE_init_reco), self.args.energy_slices, self.args.fit["regions"], self.args.fit["mc_stat_unc"], None, book)

        if book is not None:
            for p in cex_analyse.process_labels:
                hist_true = analysis_input.CreateHistograms(self.args.energy_slices, p, False, None)["int_ex"]
                Plots.Plot(self.args.energy_slices.pos_overflow, unfolding_result[p]["unfolded"], yerr = quadsum([unfolding_result[p]["stat_err"], unfolding_result[p]["sys_err"]], 0), xlabel = f"$N_{{int, {cex_analyse.process_labels[p]}}}$ (MeV)", ylabel = "Counts", label = "unfolded", style = "step", color = "C6")
                Plots.Plot(self.args.energy_slices.pos_overflow, hist_true, label = "true", style = "step", color = "C0", newFigure = False)
                book.Save()

        xs = cex_analyse.XSUnfold(unfolding_result, self.args.energy_slices, True, True, self.args.fit["regions"])
        return xs

    def RunExperiment(self, config : dict, out : str = None) -> tuple[dict, dict]:
        x = self.args.energy_slices.pos[:-1] - self.args.energy_slices.width / 2

        ai = cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(df = cex_toy_generator.run(config)))

        if out:
            book = Plots.PlotBook(out)
        else:
            book = None

        xs = self.Analyse(ai, book)
        if out:
            book.close()
            Plots.plt.close("all")
        xs_sim_mod = cex_toy_generator.ModifyGeantXS(scale_factors = config["pdf_scale_factors"])

        if self.args.fit["regions"]:
            xs_true = {k : xs_sim_mod.GetInterpolatedCurve(k)(x) for k in exclusive_proc}
        else:
            xs_true = xs_sim_mod.GetInterpolatedCurve(self.args.signal_process)(x)

        return xs, xs_true


class DataAnalysis(ABC):
    def __init__(self, args : cross_section.argparse.Namespace) -> None:
        self.args = dill_copy(args)
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @classmethod
    def CreateNewAIs():
        pass


    def __run_analysis(self, new_input):
        args_copy = dill_copy(self.args)
        inp = {}
        for f in cross_section.ls_recursive(new_input):
            inp[f.split("/")[-1].split(".")[0].split("analysis_input_")[-1].split("_selected")[0]] = cross_section.os.path.abspath(f)
        args_copy.analysis_input = inp
        args_copy.pdsp = True
        args_copy.toy_template = None
        args_copy.out = ""
        args_copy.all = False
        xs = cex_analyse.Analyse(args_copy, False)["pdsp"]
        return xs


    def RunAnalysis(self, path):
        xs = {i : self.__run_analysis(f"{path}{self.name}_{i}/") for i in ["low", "high"]}
        return xs


    def PlotResults(self, xs_nominal : dict, result : dict, book : Plots.PlotBook = Plots.PlotBook.null):
        for p in xs_nominal["pdsp"]:
            cross_section.PlotXSComparison({"low" : result["low"][p], "nominal" : xs_nominal["pdsp"][p], "high" : result["high"][p]}, self.args.energy_slices, p, cv_only = True, marker_size = 12)
            book.Save()
        return

    @staticmethod
    def CalculateSysError(xs):
        return {p : abs((xs["high"][p][0] - xs["low"][p][0]) / 2) for p in xs["low"]}


    @staticmethod
    def CalculateSysErrorAsym(xs_nominal : dict, xs : dict):
        errs = {}
        for p in xs_nominal["pdsp"]:
            e_low = xs_nominal["pdsp"][p][0] - xs["low"][p][0]
            e_high = xs_nominal["pdsp"][p][0] - xs["high"][p][0]

            err_high = np.max(abs(np.array([np.where(e_low < 0, e_low, 0), np.where(e_high < 0, e_high, 0)])), 0)

            err_low = np.max(np.array([np.where(e_low >= 0, e_low, 0), np.where(e_high >= 0, e_high, 0)]), 0)
            errs[p] = {"low" : err_low, "high" : err_high}
        return errs


    def DataAnalysisTables(self, xs_nominal : dict, err_asym : dict, name : str):
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
        
            t = pd.Series(quadsum([d, l, h], 0))
            t.name = "Total"
        
            table = pd.concat([KEs, t, d, l, h], axis = 1).sort_values(by = ["$KE$ (MeV)"])

            avg = table.mean()
            avg["$KE$ (MeV)"] = "average"
            tables[p] = pd.concat([table, pd.DataFrame(avg).T]).reset_index(drop = True)

        return tables


class NuisanceParameters:
    def __init__(self, args : cross_section.argparse.Namespace) -> None:
        self.args = dill_copy(args)
        pass


    def __run_analysis(self, np : bool = False):
        args_copy = dill_copy(self.args)
        args_copy.bkg_sub_err = False
        args_copy.fit["mc_stat_unc"] = True
        args_copy.fit["fix_np"] = not np
        args_copy.pdsp = True
        args_copy.toy_template = None
        args_copy.out = ""
        args_copy.all = False

        xs = cex_analyse.Analyse(args_copy, False)["pdsp"]
        return xs


    def RunExperiment(self):
        xs = self.__run_analysis(False)
        xs_np = self.__run_analysis(True)
        return {"no_np" : xs, "np" : xs_np}


    def CalculateSysError(self, result : dict):
        np_sys = {}
        for p in result["no_np"]:
            np_sys[p] = np.sqrt(result["np"][p][1]**2 - result["no_np"][p][1]**2)
        return np_sys


    def PlotXSMCStat(self, result, book : Plots.PlotBook = Plots.PlotBook.null):
        for p in result["np"]:
            cross_section.PlotXSComparison({"Data stat + MC stat" : result["np"][p], "Data stat" : result["no_np"][p]}, self.args.energy_slices, process = p)
            book.Save()


    def MCStatTables(self, result : dict):
        x = self.args.energy_slices.pos_overflow[1:-1] - (self.args.energy_slices.width/2)

        KEs = pd.Series(np.array(x, dtype = int), name = "$KE$ (MeV)")

        mc_stat_err = self.CalculateSysError(result)
        data_stat_err = {p : result["no_np"][p][1] for p in result["no_np"]}

        tables = {}
        for p in data_stat_err:
            d = pd.DataFrame(data_stat_err)[p] / result["no_np"][p][0]
            d.name = "Data stat"
            m = pd.DataFrame(mc_stat_err)[p] / result["no_np"][p][0]
            m.name = "MC stat"

            t = pd.Series(quadsum([d, m], 0))
            t.name = "Total"
        
            table = pd.concat([KEs, t, d, m], axis = 1).sort_values(by = ["$KE$ (MeV)"])

            avg = table.mean()
            avg["$KE$ (MeV)"] = "average"
            tables[p] = pd.concat([table, pd.DataFrame(avg).T]).reset_index(drop = True)
        return tables


class BkgSubSystematic:
    def __init__(self, args : cross_section.argparse.Namespace) -> None:
        self.args = dill_copy(args)
        pass


    def __run_analysis(self, bkg_sub : bool = False):
        args_copy = dill_copy(self.args)
        args_copy.bkg_sub_err = bkg_sub
        args_copy.pdsp = True
        args_copy.toy_template = None
        args_copy.out = ""
        args_copy.all = False

        xs = cex_analyse.Analyse(args_copy, False)["pdsp"]
        return xs


    def RunExperiment(self):
        xs = self.__run_analysis(False)
        xs_np = self.__run_analysis(True)
        return {"no_bkg" : xs, "bkg" : xs_np}


    def CalculateSysError(self, result : dict):
        np_sys = {}
        for p in result["no_bkg"]:
            np_sys[p] = np.sqrt(result["bkg"][p][1]**2 - result["no_bkg"][p][1]**2)
        return np_sys


class UpstreamCorrectionSystematic(DataAnalysis):
    name = "upstream_loss_1_sigma"

    def CreateNewAIs(self, outdir : str):
        upl = {
            "low" : {f"p{i}" : self.args.upstream_loss_correction_params["value"][f"p{i}"] - self.args.upstream_loss_correction_params["error"][f"p{i}"] for i in range(self.args.upstream_loss_response.n_params)},
            "high" : {f"p{i}" : self.args.upstream_loss_correction_params["value"][f"p{i}"] + self.args.upstream_loss_correction_params["error"][f"p{i}"] for i in range(self.args.upstream_loss_response.n_params)}
        }

        args_copy = dill_copy(self.args)
        for k, v in upl.items():
            args_copy.upstream_loss_correction_params["value"] = v
            args_copy.out = f"{outdir}{self.name}_{k}/"
            cex_analysis_input.main(args_copy)
        return


class BeamReweightSystematic(DataAnalysis):
    name = "beam_reweight_1_sigma"

    def CreateNewAIs(self, outdir : str):
        cfg = {
            "low" : {k : v["value"] - v["error"] for k, v in self.args.beam_reweight["params"].items()},
            "high" : {k : v["value"] + v["error"] for k, v in self.args.beam_reweight["params"].items()}
        }

        args_copy = dill_copy(self.args)
        for k, v in cfg.items():
            for p in args_copy.beam_reweight["params"]:
                args_copy.beam_reweight["params"][p]["value"] = v[p]
            print(args_copy.beam_reweight["params"])
            args_copy.out = f"{outdir}{self.name}_{k}/"
            cex_analysis_input.main(args_copy)
        return


class ShowerEnergyCorrectionSystematic(DataAnalysis):
    name = "shower_energy_correction_1_sigma"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.LoadSamples()

    def LoadSamples(self):
        self.mc = cross_section.Data(self.args.mc_file, nTuple_type = self.args.ntuple_type, target_momentum = self.args.pmom)
        self.data = cross_section.Data(self.args.data_file, nTuple_type = self.args.ntuple_type, target_momentum = 1)
        return

    def CreateAltPi0Selections(self):
        samples = {
            "mc" : cex_analysis_input.BeamPionSelection(self.mc, self.args, True),
            "data" : cex_analysis_input.BeamPionSelection(self.data, self.args, False)
        }

        photon_candidates = {s : cex_analysis_input.SelectionTools.CombineMasks(self.args.selection_masks[s]["photon"]) for s in samples}

        masks_sample = {}
        for sample in ["mc", "data"]:
            masks_sys = {}
            for name, sign in zip(["low", "high"], [1, -1]):
                masks = {}
                selection_args_copy = cross_section.dill_copy(self.args.pi0_selection[f"{sample}_arguments"].values())
                for (s, f), a in zip(self.args.pi0_selection["selections"].items(), selection_args_copy):
                    if s == "Pi0MassSelection":
                        a["correction_params"]["value"] = {f"p{i}" : a["correction_params"]["value"][f"p{i}"] + sign * a["correction_params"]["error"][f"p{i}"] for i in range(len(a["correction_params"]["error"]))}
                    mask = f(samples[sample], **a, photon_mask = photon_candidates[sample])
                    masks[s] = mask
                masks_sys[name] = masks
            masks_sample[sample] = masks_sys
        return masks_sample


    def CreateNewAIs(self, outdir : str):

        masks = self.CreateAltPi0Selections()
        cross_section.SaveObject(f"{outdir}pi0_selection_masks.dill", masks)

        args_copy = {}
        for i in ["low", "high"]:
            a = cross_section.dill_copy(self.args)
            a.selection_masks["mc"]["pi0"] = masks["mc"][i]
            a.selection_masks["data"]["pi0"] = masks["data"][i]
            args_copy[i] = a

        for k, v in args_copy.items():
            v.out = f"{outdir}{self.name}_{k}/"
            cex_analysis_input.main(v)
        return


class TrackLengthResolutionSystematic(DataAnalysis):
    name = "track_length_1_sigma"

    def CalculateResolution(self):
        ai_mc = cross_section.AnalysisInput.FromFile(self.args.analysis_input["mc"])

        r = np.array(cross_section.nandiv(ai_mc.track_length_reco - ai_mc.track_length_true, ai_mc.track_length_reco))

        y, edges = np.histogram(r, 150, [-0.5, 0.5])
        x = cross_section.bin_centers(edges)

        p = cross_section.Fitting.Fit(x, y,np.sqrt(y), cross_section.Fitting.double_crystal_ball, method = "dogbox", plot = True, xlabel = "$(l^{reco} - l^{true}) / l^{reco}$", ylabel = "Counts", plot_style = "scatter")

        y_interp = cross_section.Fitting.double_crystal_ball(x, *p[0])

        w = []
        for v in [-0.5, 0.5]:
            for i in np.linspace(0, v, 10000):
                if cross_section.Fitting.double_crystal_ball(i, *p[0]) <= (max(y_interp) / 2):
                    print(i)
                    break
            w.append(i)

        self.resolution = max(w) - min(w)
        return

    def CreateNewAIs(self, outdir : str):
        cross_section.os.makedirs(f"{outdir}{self.name}_high/", exist_ok = True)
        cross_section.os.makedirs(f"{outdir}{self.name}_low/", exist_ok = True)
        
        self.CalculateResolution()
        for i in self.args.analysis_input:
            ai = cross_section.AnalysisInput.FromFile(self.args.analysis_input[i])
            if i == "data":
                cross_section.SaveObject(f"{outdir}{self.name}_high/{i}.dill", ai)
                cross_section.SaveObject(f"{outdir}{self.name}_low/{i}.dill", ai)
            else:
                Ep = cross_section.BetheBloch.InteractingKE(ai.KE_ff_reco, ai.track_length_reco * (1+self.resolution), 50)
                Em = cross_section.BetheBloch.InteractingKE(ai.KE_ff_reco, ai.track_length_reco * (1-self.resolution), 50)
                
                ai.KE_int_reco = Ep
                cross_section.SaveObject(f"{outdir}{self.name}_high/{i}.dill", ai)
                ai.KE_int_reco = Em
                cross_section.SaveObject(f"{outdir}{self.name}_low/{i}.dill", ai)
        cross_section.SaveConfiguration({"value" : self.resolution}, f"{outdir}resolution.json")
        return


class BeamMomentumResolutionSystematic(DataAnalysis):
    name = "beam_momentum_resolution"

    def CreateNewConfigEntry(self, target_files, additional_args, output_path):
        print("outputs: " + output_path)
        new_config_entry = {}
        files = cross_section.os.listdir(output_path)
        for k, v in target_files.items():
            if v in files:
                new_config_entry[k] = cross_section.os.path.abspath(output_path + v)
        for k, v in additional_args.items():
            new_config_entry[k] = v
        return new_config_entry

    def CreateNewAIs(self, outdir : str, resolution : float):

        for k, v in zip(["low", "high"], [-1, 1]):
            args_copy = cross_section.dill_copy(self.args)
            args_copy.out = f"{outdir}/{self.name}_{k}/"
            args_copy.pmom = args_copy.pmom * (1 + (v * resolution))
            cex_beam_reweight.main(args_copy)

            output_path = args_copy.out + "beam_reweight/"

            target_files = {
                "params" : "gaussian.json", # default choice, rework reweight to include a choice in the config
            }
            additional_args = {"strength" : args_copy.beam_reweight["strength"]}

            new_config_entry = self.CreateNewConfigEntry(target_files, additional_args, output_path)

            new_config_entry["params"] = cross_section.LoadConfiguration(new_config_entry["params"])
            args_copy.beam_reweight = new_config_entry

            args_copy.no_reweight = False
            cex_upstream_loss.main(args_copy)

            output_path = args_copy.out + "upstream_loss/"
            target_files = {
            "correction_params" : "fit_parameters.json",
            }

            additional_args = {
                "cv_function" : args_copy.upstream_loss_cv_function,
                "response" : args_copy.upstream_loss_response,
                "bins" : args_copy.upstream_loss_bins,
            }

            new_config_entry = self.CreateNewConfigEntry(target_files, additional_args, output_path)

            args_copy.upstream_loss_bins = new_config_entry["bins"]
            args_copy.upstream_loss_response = new_config_entry["response"]
            args_copy.upstream_loss_cv_function = new_config_entry["cv_function"]
            args_copy.upstream_loss_correction_params = cross_section.LoadConfiguration(new_config_entry["correction_params"])

            cex_analysis_input.main(args_copy)
        return


class NormalisationSystematic(MCMethod):
    def Evaluate(self, norms = [0.8, 1.2], repeats : int = 1):
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
                        config = cex_fit_studies.CreateConfigNormalisation(scales, self.data_config)

                    if repeats != 1:
                        config["seed"] = j + 1

                    output = self.RunExperiment(config, None)
                    xs[i].append(output[0])
                    xs_true[i] = output[1]
            cvs[target] = xs
            true_cvs[target] = xs_true
        return {"cv" : cvs, "true_cv" : true_cvs}

    @staticmethod
    def PlotNormalisationTestResults(results : dict, args : cross_section.argparse.Namespace, xs_nominal : dict, book : Plots.PlotBook = Plots.PlotBook.null):
        xs_sim = cross_section.GeantCrossSections()
        scale_factors = {"absorption" : 1, "charge_exchange" : 1, "pion_production" : 1, "double_charge_exchange" : 1, "quasielastic" : 1}
        for r in results["cv"]:
            for _, p in Plots.IterMultiPlot(results["cv"][r][list(results["cv"][r].keys())[0]]):
                for n in results["cv"][r]:
                    mod_norm = {k : v for k, v in scale_factors.items()}
                    if r == "single_pion_production":
                        mod_norm["double_charge_exchange"] = n
                        mod_norm["quasielastic"] = n
                    else:
                        mod_norm[r] = n
                    mod_sim = cex_toy_generator.ModifyGeantXS(scale_factors = mod_norm, modified_PDFs = None)

                    if p == "single_pion_production":
                        gxs = getattr(mod_sim, "double_charge_exchange") + getattr(mod_sim, "quasielastic")
                    else:
                        gxs = getattr(mod_sim, p)

                    Plots.Plot(mod_sim.KE, gxs, newFigure = False, label = f"true, $\mathcal{{N}} = {n}$", title = f"process : {cross_section.remove_(p)}", ylabel = "$\sigma$ (mb)", xlabel = "$KE$ (MeV)")
                    Plots.Plot(args.energy_slices.pos[:-1] - args.energy_slices.width/2, results["cv"][r][n][p][0], yerr = results["cv"][r][n][p][1], xerr = args.energy_slices.width/2, marker = "x", linestyle = "", label = f"measured, $\mathcal{{N}} = {n}$", newFigure = False)
                    Plots.plt.xlim(args.energy_slices.min_pos - args.energy_slices.width, args.energy_slices.max_pos + args.energy_slices.width)
                    Plots.plt.ylim(0, 1.5 * max(results["cv"][r][n][p][0]))
                if p == "single_pion_production":
                    gxs = getattr(xs_sim, "double_charge_exchange") + getattr(xs_sim, "quasielastic")
                else:
                    gxs = getattr(xs_sim, p)
                Plots.Plot(mod_sim.KE, gxs, newFigure = False, label = f"true, $\mathcal{{N}} = 1$")
                Plots.Plot(args.energy_slices.pos[:-1] - args.energy_slices.width/2, xs_nominal[p][0], yerr = xs_nominal[p][1], xerr = args.energy_slices.width/2, marker = "x", linestyle = "", label = f"measured, $\mathcal{{N}} = 1$", newFigure = False)
            Plots.plt.suptitle(f"normalisation test : {cross_section.remove_(r)}")
            Plots.plt.tight_layout()
            book.Save()
        return

    @staticmethod
    def CalculateSysErr(results):
        def sys_err(r, tr):
            return {p : abs(r[p][0] - tr[p]) for p in r}

        sys_err_low = {}
        sys_err_high = {}
        for r in results["cv"]:
            norms = list(results["cv"][r].keys())
            sys_err_low[r] = sys_err(results["cv"][r][min(norms)], results["true_cv"][r][min(norms)])
            sys_err_high[r] = sys_err(results["cv"][r][max(norms)], results["true_cv"][r][max(norms)])
        return {"low" : sys_err_low, "high" : sys_err_high}

    @staticmethod
    def TotalSysQS(sys_err : dict):
        norm_sys_qs = {}
        for p in exclusive_proc:
            norm_sys_qs[p] = [
                quadsum([sys_err["low"][r][p] for r in sys_err["low"]], 0),
                quadsum([sys_err["high"][r][p] for r in sys_err["high"]], 0)
            ]
        return norm_sys_qs

    @staticmethod
    def TotalSysMax(sys_err : dict):
        norm_sys_max = {}
        for p in exclusive_proc:
            norm_sys_max[p] = [
                np.max([sys_err["low"][r][p] for r in sys_err["low"][p]], 0),
                np.max([sys_err["high"][r][p] for r in sys_err["high"][p]], 0)
            ]
        return norm_sys_max
    
    @staticmethod
    def AverageResults(results : dict):
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
    def CalculateFractionalError(sys_err_avg : dict, xs_nominal : dict):
        frac_low = {}
        frac_high = {}
        for i in xs_nominal:
            lo = {}
            hi = {}
            for p in xs_nominal:
                lo[p] = sys_err_avg["low"][i][p] / xs_nominal[p][0]
                hi[p] = sys_err_avg["high"][i][p] / xs_nominal[p][0]
            frac_low[i] = lo
            frac_high[i] = hi

        return {"low" : frac_low, "high" : frac_high}

    
    def CreateTables(self, xs_nominal, sys):
        x = args.energy_slices.pos_overflow[1:-1] - (args.energy_slices.width/2)

        KEs = pd.Series(np.array(x, dtype = int), name = "$KE$ (MeV)")

        data_stat_err = pd.DataFrame({p : xs_nominal["pdsp"][p][1] for p in xs_nominal["pdsp"]})

        tags = cross_section.Tags.ExclusiveProcessTags(None)

        tables = {}
        for p in xs_nominal["pdsp"]:
            d = data_stat_err[p] / xs_nominal["pdsp"][p][0]
            d.name = "Data stat"
            ls = []
            hs = []
            for q in sys["fractional"]["low"]:
                l = pd.Series(sys["fractional"]["low"][p][q])
                l.name = "Model inaccuracy " + tags[q].name_simple + " low"

                h = pd.Series(sys["fractional"]["high"][p][q])
                h.name = "Model inaccuracy" + tags[q].name_simple + " high"
                ls.append(l)
                hs.append(h)
            t = pd.Series(quadsum([d, *ls, *hs], 0))
            t.name = "Total"

            table = pd.concat([KEs, t, d, *ls, *hs], axis = 1).sort_values(by = ["$KE$ (MeV)"])

            avg = table.mean()
            avg["$KE$ (MeV)"] = "average"
            tables[p] = pd.concat([table, pd.DataFrame(avg).T]).reset_index(drop = True)
        return tables


def TheoryXS(theory_sys, cv):
    theory_xs = {}
    for s in theory_sys["fractional"]["low"]:
        theory_err_s = {}
        for p in cv:
            err = np.array([abs(theory_sys["fractional"]["low"][s][p] * cv[p][0]), abs(theory_sys["fractional"]["high"][s][p] * cv[p][0])])
            theory_err_s[p] = err
        theory_xs[s] = theory_err_s

    theory_xs_T = {}
    for k in theory_xs:
        theory_xs_T[k] = {k2 : theory_xs[k2][k] for k2 in theory_xs}
    return theory_xs_T


def PlotSysHist(cv, systematics, energy_slices, book : Plots.PlotBook = Plots.PlotBook.null):
    x = energy_slices.pos[:-1] - energy_slices.width/2
    for _, p in Plots.IterMultiPlot(exclusive_proc):
        for s in systematics:
            if systematics[s][p].shape == (2, len(x)):
                err = cross_section.quadsum(systematics[s][p], 0) / 2
            else:
                err = systematics[s][p]
            Plots.Plot(x, err, label = s, title = cross_section.remove_(p), newFigure = False, style = "step", xlabel = "$KE$ (MeV)", ylabel = "Systematic error (mb)")
    book.Save()
    for _, p in Plots.IterMultiPlot(exclusive_proc):
        for s in systematics:
            if systematics[s][p].shape == (2, len(x)):
                err = cross_section.quadsum(systematics[s][p], 0) / 2
            else:
                err = systematics[s][p]
            Plots.Plot(x, cross_section.nandiv(err, cv[p][0]), label = s, title = cross_section.remove_(p), newFigure = False, style = "step", xlabel = "$KE$ (MeV)", ylabel = "Systematic fractional error")
    book.Save()

def FinalPlots(cv, systematics, energy_slices, book : Plots.PlotBook = Plots.PlotBook.null):
    for p in cv:
        xs = {
            "ProtoDUNE SP: Data Stat + Sys Error" : cv[p],
            "" : [cv[p][0], cross_section.quadsum([cv[p][1]] + [systematics[s][p] for s in systematics], 0)]
        }
        cross_section.PlotXSComparison(xs, energy_slices, p, simulation_label = "Geant4 v10.6", colors = {k : f"C0" for k in xs}, chi2 = False)
        book.Save()
    for _, p in Plots.IterMultiPlot(cv):
        xs = {
            "ProtoDUNE SP: Data Stat + Sys Error" : cv[p],
            "" : [cv[p][0], cross_section.quadsum([cv[p][1]] + [systematics[s][p] for s in systematics], 0)]
        }
        cross_section.PlotXSComparison(xs, energy_slices, p, simulation_label = "Geant4 v10.6", colors = {k : f"C0" for k in xs}, chi2 = False, newFigure = False)
    book.Save()
    return


def can_run(systematic):
    return ((systematic not in args.skip) and (systematic in args.run)) or ("all" in args.run)


def can_regen(dir):
    if cross_section.os.path.exists(dir):
        for f in cross_section.ls_recursive(dir):
            print(f)
            if ("dill" in f) and ("analysis_input" not in f) and (args.regen is True):
                return True
    else:
        return True
    return False

@cross_section.timer
def main(args : cross_section.argparse.Namespace):
    cross_section.SetPlotStyle(dark = True)
    out = args.out + "systematics/"
    cross_section.os.makedirs(out, exist_ok = True)

    print(f"{args.skip=}")
    print(f"{args.run=}")

    if ("all" not in args.skip) or ("all" in args.run):
        if can_run("bkg_sub"):
            print(Rule("bkg_sub"))
            if can_regen(out + "bkg_sub/"):
                bkg_sub = BkgSubSystematic(args)
                sys = bkg_sub.CalculateSysError(bkg_sub.RunExperiment())
                cross_section.os.makedirs(out + "bkg_sub/", exist_ok = True)
                SaveSystematicError(sys, None, out + "bkg_sub/sys.dill")
        if can_run("mc_stat"):
            print(Rule("mc_stat"))
            mc_stat = NuisanceParameters(args)

            if can_regen(out + "mc_stat/"):
                result = mc_stat.RunExperiment()
                sys = mc_stat.CalculateSysError(result)
                cross_section.os.makedirs(out + "mc_stat/", exist_ok = True)
                cross_section.SaveObject(out + "mc_stat/result.dill", result)
                SaveSystematicError(sys, None, out + "mc_stat/sys.dill")
            else:
                result = cross_section.LoadObject(out + "mc_stat/result.dill")

            with Plots.PlotBook(out + "mc_stat/plots.pdf") as book:
                mc_stat.PlotXSMCStat(result, book)
            tables = mc_stat.MCStatTables(result)
            DictToHDF5(tables, out + "mc_stat/tables.hdf5")
            for t in tables:
                tables[t].style.format(precision = 2).hide(axis = 0).to_latex(out + f"mc_stat/table_{t}.tex")


        if can_run("upstream"):
            print(Rule("upstream"))
            upl = UpstreamCorrectionSystematic(args)

            if can_regen(out + "upstream/"):
                upl.CreateNewAIs(out + "upstream/")
                xs = upl.RunAnalysis(out + "upstream/")
                sys = upl.CalculateSysErrorAsym(args.cv, xs)
                SaveSystematicError(sys, None, out + "upstream/sys.dill")
            else:
                sys = cross_section.LoadObject(out + "upstream/sys.dill")

            with Plots.PlotBook(out + "upstream/plots.pdf") as book:
                upl.PlotResults(args.cv, xs, book)
            tables = upl.DataAnalysisTables(args.cv, sys, "Upstream")
            DictToHDF5(tables, out + "upstream/tables.hdf5")
            for t in tables:
                tables[t].style.format(precision = 2).hide(axis = 0).to_latex(out + f"upstream/table_{t}.tex")

        if can_run("beam_reweight"):
            print(Rule("beam reweight"))
            brw = BeamReweightSystematic(args)

            if can_regen(out + "beam_reweight/"):
                brw.CreateNewAIs(out + "beam_reweight/")
                xs = brw.RunAnalysis(out + "beam_reweight/")
                sys = brw.CalculateSysErrorAsym(args.cv, xs)
                SaveSystematicError(sys, None, out + "beam_reweight/sys.dill")
            else:
                sys = cross_section.LoadObject(out + "beam_reweight/sys.dill")

            with Plots.PlotBook(out + "beam_reweight/plots.pdf") as book:
                brw.PlotResults(args.cv, xs, book)
            tables = brw.DataAnalysisTables(args.cv, sys, "Reweight")
            DictToHDF5(tables, out + "beam_reweight/tables.hdf5")
            for t in tables:
                tables[t].style.format(precision = 2).hide(axis = 0).to_latex(out + f"beam_reweight/table_{t}.tex")

        if can_run("track_length"):
            print(Rule("track length"))
            trk = TrackLengthResolutionSystematic(args)

            if can_regen(out + "track_length/"):
                trk.CreateNewAIs(out + "track_length/")
                xs = trk.RunAnalysis(out + "track_length/")
                sys = trk.CalculateSysErrorAsym(args.cv, xs)
                SaveSystematicError(sys, None, out + "track_length/sys.dill")
            else:
                sys = cross_section.LoadObject(out + "track_length/sys.dill")

            with Plots.PlotBook(out + "track_length/plots.pdf") as book:
                trk.PlotResults(args.cv, xs, book)
            tables = trk.DataAnalysisTables(args.cv, sys, "Track length")
            DictToHDF5(tables, out + "track_length/tables.hdf5")
            for t in tables:
                tables[t].style.format(precision = 2).hide(axis = 0).to_latex(out + f"track_length/table_{t}.tex")

        if can_run("beam_res"):
            print(Rule("beam resolution"))
            resolution = 2.5/100
            bm = BeamMomentumResolutionSystematic(args)

            if can_regen(out + "beam_res/"):
                cross_section.os.makedirs(out + "beam_res/", exist_ok = True)
                bm.CreateNewAIs(out + "beam_res/", resolution)
                xs = bm.RunAnalysis(out + "beam_res/")
                sys = bm.CalculateSysErrorAsym(args.cv, xs)
                SaveSystematicError(sys, None, out + "beam_res/sys.dill")
            else:
                sys = cross_section.LoadObject(out + "beam_res/sys.dill")

            with Plots.PlotBook(out + "beam_res/plots.pdf") as book:
                bm.PlotResults(args.cv, xs, book)
            tables = bm.DataAnalysisTables(args.cv, sys, "Beam momentum")
            DictToHDF5(tables, out + "beam_res/tables.hdf5")
            for t in tables:
                tables[t].style.format(precision = 2).hide(axis = 0).to_latex(out + f"beam_res/table_{t}.tex")


        if can_run("theory"):
            print(Rule("theory"))
            cross_section.os.makedirs(out + "theory/", exist_ok = True)

            args.toy_template = cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(file = args.toy_template))
            model = cross_section.RegionFit.CreateModel(args.toy_template, args.energy_slices, args.fit["mean_track_score"], False, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])
            
            toy_nominal = cross_section.Toy(df = cex_toy_generator.run(args.toy_data_config))
            analysis_input_nominal = cross_section.AnalysisInput.CreateAnalysisInputToy(toy_nominal)
            
            norm_sys = NormalisationSystematic(args, model, args.toy_data_config)
            
            xs_nominal = norm_sys.Analyse(analysis_input_nominal, None)

            if can_regen(out + "theory/"):
                if not cross_section.os.path.isfile(out + "theory/test_results.dill"):    
                    results = norm_sys.Evaluate([0.8, 1.2], 3)
                    cross_section.SaveObject(out + "theory/test_results.dill", results)

                results = cross_section.LoadObject(out + "theory/test_results.dill")
                NormalisationSystematic.AverageResults(results)
                sys_err = NormalisationSystematic.CalculateSysErr(results)
                frac_err = NormalisationSystematic.CalculateFractionalError(sys_err, xs_nominal)
                SaveSystematicError(sys_err, frac_err, out + "theory/sys.dill")
            else:
                results = cross_section.LoadObject(out + "theory/test_results.dill")
                NormalisationSystematic.AverageResults(results)
                sys = cross_section.LoadObject(out + "theory/sys.dill")

            with Plots.PlotBook(out + "theory/plots", True) as book:
                cross_section.SetPlotStyle(dark = False, extend_colors = True)
                NormalisationSystematic.PlotNormalisationTestResults(results, args, xs_nominal, book)
                cross_section.SetPlotStyle(dark = True, extend_colors = False)

            tables = norm_sys.CreateTables(cross_section.LoadObject(args.cv), sys)
            DictToHDF5(tables, out + "theory/tables.hdf5")
            for t in tables:
                tables[t].style.format(precision = 2).hide(axis = 0).to_latex(out + f"theory/table_{t}.tex")

    if args.plot is not None:
        label_short = {
            'absorption': "abs",
            'charge_exchange': "cex",
            'single_pion_production': "spip",
            'pion_production': "pip"
        }

        cv = cross_section.LoadObject(args.plot)["pdsp"]

        systematics = {}
        for f in cross_section.os.listdir(out):
            if cross_section.os.path.isfile(f"{out}{f}") : continue
            sys = cross_section.LoadObject(f"{out}{f}/sys.dill")
            if f == "theory":
                t = TheoryXS(sys, cv)
                systematics =  {**systematics, **{f"theory, {label_short[k]}" : t[k] for k in t}}
            if f == "upstream":
                systematics[f] = sys["systematic"]
            if f == "mc_stat":
                systematics["MC stat"] = sys["systematic"]
            if f == "beam_reweight":
                systematics["beam reweight"] = sys["systematic"]
            if f == "bkg_sub":
                systematics["theory: bkg sub"] = sys["systematic"]
        with Plots.PlotBook(out + "plots") as book:
            PlotSysHist(cv, systematics, args.energy_slices, book)

            FinalPlots(cv, systematics, args.energy_slices, book)

    return

if __name__ == "__main__":
    systematics = ["mc_stat", "theory", "upstream", "beam_reweight", "shower_energy", "bkg_sub", "track_length", "beam_res", "all"]

    parser = cross_section.argparse.ArgumentParser("Estimate Systematics for the cross section analysis")
    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser)

    parser.add_argument("--toy_template", "-t", dest = "toy_template", type = str, help = "toy template hdf5 file", required = False)
    parser.add_argument("--toy_data_config", "-d", dest = "toy_data_config", type = str, help = "json config for toy data", required = False)

    parser.add_argument("--skip", type = str, nargs = "+", default = [], choices = systematics)
    parser.add_argument("--run", type = str, nargs = "+", default = [], choices = systematics)

    parser.add_argument("--regen", "-r", dest = "regen", action = "store_true", help = "fully rerun systematic tests if results already exist")

    parser.add_argument("--cv", "-v", dest = "cv", type = str, default = None, help = "plot systematics with central value measurement")

    parser.add_argument("--plot", "-p", dest = "plot", type = str, default = None, help = "plot systematics with central value measurement")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())

    if ("all" in args.run) or ("theory" in args.run):
        if not args.toy_template:
            raise Exception("--toy_template must be specified")
        if not args.toy_data_config:
            raise Exception("--toy_data_config must be specified")        
        args.toy_data_config = cross_section.LoadConfiguration(args.toy_data_config)

    if ("all" in args.run) or ("upstream" in args.run) or ("beam_reweight" in args.run) or ("shower_energy" in args.run) or ("track_length" in args.run) or ("beam_res" in args.run):
        if not args.cv:
            raise Exception("--cv must be specified")
        args.cv = cross_section.LoadObject(args.cv)

    print(vars(args))
    main(args)