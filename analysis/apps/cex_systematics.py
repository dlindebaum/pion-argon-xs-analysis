#!/usr/bin/env python3
"""
Created on: 09/04/2024 13:53

Author: Shyam Bhuller

Description: app to calculate systematic uncertainties for cross section measurement
"""
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from rich import print
from rich.rule import Rule

from python.analysis import cross_section, Plots
from python.analysis.Master import DictToHDF5
from python.analysis.Utils import dill_copy, quadsum, round_value_to_error
from apps import cex_toy_generator, cex_analyse, cex_fit_studies, cex_analysis_input

systematics = ["mc_stat", "upstream", "beam_reweight", "shower_energy", "track_length", "beam_res", "fit_inaccuracy", "theory", "all"]
systematics_label = {"mc_stat" : "MC stat", "fit_inaccuracy" : "Fit inaccuracy", "upstream" : "Upstream", "beam_reweight" : "Reweight", "shower_energy" : "Shower energy", "track_length" : "Track length", "beam_res" : "Beam momentum", "theory" : "Theory"}

exclusive_proc = ["absorption", "charge_exchange", "single_pion_production", "pion_production"]


def SaveSystematicError(systematic : dict, fractional : dict, out : str):
    return cross_section.SaveObject(out, {"systematic" : systematic, "fractional": fractional})


def SaveTables(tables : pd.DataFrame, outdir : str, precision : int):
    DictToHDF5(tables, f"{outdir}tables.hdf5")
    for t in tables:
        tables[t].style.format(precision = precision).hide(axis = 0).to_latex(f"{outdir}table_{t}.tex")
    return


class MCMethod(ABC):
    def __init__(self, args : cross_section.argparse.Namespace, model : cross_section.pyhf.Model, data_config : dict) -> None:
        self.args = cross_section.dill_copy(args)
        self.model = model
        self.data_config = data_config
        pass

    def Analyse(self, analysis_input : cross_section.AnalysisInput, book : Plots.PlotBook = Plots.PlotBook.null):

        region_fit_result = cex_analyse.RegionFit(analysis_input, self.args.energy_slices, self.args.fit["mean_track_score"], self.model, mc_stat_unc = self.args.fit["mc_stat_unc"], single_bin = self.args.fit["single_bin"])

        _, histograms_reco_obs, histograms_reco_obs_err = cex_analyse.BackgroundSubtraction(analysis_input, self.args.signal_process, self.args.energy_slices, region_fit_result, self.args.fit["single_bin"], self.args.fit["regions"], self.args.toy_template, args.bkg_sub_mc_stat, book)


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

    @abstractmethod
    def RunExperiment(self, config : dict) -> tuple[dict, dict]:
        pass


    def Evaluate(self, n : int, **kwargs):
        xs = []
        for i in range(n):
            print(Rule(f"Experiment : {i + 1}"))
            xs.append(self.RunExperiment(**kwargs))
        return xs


    def CalculateSysCov(self, results : list[dict], book : Plots.PlotBook = Plots.PlotBook.null):
        values = {k : [] for k in results[0]}
        for i in results:
            for k in i:
                values[k].append(i[k][0])

        cov = {k : np.cov(values[k], rowvar = False) for k in values}

        if book:
            for k in cov:
                Plots.plt.figure()
                Plots.plt.imshow(cov[k])
                Plots.plt.grid(False)
                Plots.plt.colorbar()
                Plots.plt.title(cross_section.remove_(k).capitalize())

                x = self.args.energy_slices.pos_overflow[1:-1] - self.args.energy_slices.width/2
                Plots.plt.xticks(np.linspace(0, len(x) - 1, len(x)), np.array(x[::-1], dtype = int), rotation = 30)
                Plots.plt.yticks(np.linspace(0, len(x) - 1, len(x)), np.array(x[::-1], dtype = int), rotation = 30)
                Plots.plt.xlabel("KE (MeV)")
                Plots.plt.ylabel("KE (MeV)")

                for (i, j), z in np.ndenumerate(cov[k]):
                    Plots.plt.gca().text(j, i, f"{z:.1g}", ha='center', va='center', fontsize = 10, color = "red")
                book.Save()
        return {k : np.sqrt(np.diag(cov[k])) for k in cov}


    def Tables(self, xs_nominal : dict, sys : dict, name : str) -> dict[pd.DataFrame]:
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
        
            t = pd.Series(quadsum([d, e], 0))
            t.name = "Total"
        
            table = pd.concat([KEs, t, d, e], axis = 1).sort_values(by = ["$KE$ (MeV)"])

            avg = table.mean()
            avg["$KE$ (MeV)"] = "average"
            tables[p] = pd.concat([table, pd.DataFrame(avg).T]).reset_index(drop = True)

        return tables


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

    def __init__(self, args: cross_section.argparse.Namespace) -> None:
        super().__init__(args)
        self.args.batches = None
        self.args.events = None
        self.args.threads = 1


    @staticmethod
    def __run(i : int, file : str, n_events : int, start : int, selected_events, args : dict):
        print(f"starting process {i}")
        sample = "data" if args["data"] is True else "mc"

        events = cross_section.Data(file, n_events, start, args["nTuple_type"], args["pmom"])

        for s, a in zip(args["beam_selection"]["selections"].values(), args["beam_selection"][f"{sample}_arguments"].values()):
            mask = s(events, **a)
            events.Filter([mask], [mask])
        photon_masks = {}
        if args["valid_pfo_selection"] is True:
            for k, s, a in zip(args["photon_selection"]["selections"].keys(), args["photon_selection"]["selections"].values(), args["photon_selection"][f"{sample}_arguments"].values()):
                photon_masks[k] = s(events, **a)
        photon_candidates = cross_section.SelectionTools.CombineMasks(photon_masks)

        output = {}
        for name, sign in zip(["low", "high"], [1, -1]):
            masks = {}
            selection_args_copy = cross_section.dill_copy(args["pi0_selection"][f"{sample}_arguments"].values())
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
        names = np.unique([o["name"] for o in output])

        split_output = {n : [] for n in names}
        for n in names:
            for o in output:
                if o["name"] == n:
                    split_output[n].append(o)

        merged_output = []
        for s in split_output:
            o = cross_section.MergeOutputs(split_output[s])
            if type(o["name"]) == list:
                o["name"] = o["name"][0] 
            merged_output.append(o)
        return merged_output


    def CreateAltPi0Selections(self):

        self.args.cpus = 1
        processing_args = cross_section.CalculateBatches(self.args)
        for k, v in processing_args.items():
            setattr(self.args, k, v)

        print("running MC")
        output_mc = self.__merge(cross_section.RunProcess(self.args.ntuple_files["mc"], False, self.args, self.__run, False))
        print("running Data")
        output_data = self.__merge(cross_section.RunProcess(self.args.ntuple_files["data"], True, self.args, self.__run, False))
        return {"mc" : output_mc, "data" : output_data}


    def CreateNewAIs(self, outdir : str):

        cross_section.os.makedirs(outdir, exist_ok = True)

        masks = self.CreateAltPi0Selections()
        cross_section.SaveObject(f"{outdir}pi0_selection_masks.dill", masks)

        args_copy = {}
        for i in ["low", "high"]:
            a = cross_section.dill_copy(self.args)
            a.selection_masks["mc"]["pi0"] = {j["name"] : j[i] for j in masks["mc"]}

            a.selection_masks["data"]["pi0"] = {j["name"] : j[i] for j in masks["data"]}
            args_copy[i] = a

        for k, v in args_copy.items():
            v.out = f"{outdir}{self.name}_{k}/"
            cex_analysis_input.main(v)
        return


class BeamMomentumResolutionSystematic(MCMethod):
    name = "beam_momentum_resolution"

    def __init__(self, args: cross_section.argparse.Namespace, model: cross_section.pyhf.Model, data_config: dict) -> None:
        super().__init__(args, model, data_config)
        self.P_reco_original = np.sqrt((self.args.toy_template.KE_init_reco + cross_section.Particle.from_pdgid(211).mass)**2 - (cross_section.Particle.from_pdgid(211).mass)**2)


    def RunExperiment(self, analysis_input_data : cross_section.AnalysisInput, resolution : float) -> tuple[dict, dict]:
        P_reco_smeared = self.P_reco_original * (1 + np.random.normal(0, resolution, len(self.P_reco_original)))

        KE_init_reco = cross_section.KE(P_reco_smeared, cross_section.Particle.from_pdgid(211).mass)
        KE_int_reco = cross_section.BetheBloch.InteractingKE(KE_init_reco, self.args.toy_template.track_length_reco, 10)

        self.args.toy_template.KE_ff_reco = KE_init_reco
        self.args.toy_template.KE_init_reco = KE_init_reco
        self.args.toy_template.KE_int_reco = KE_int_reco

        if self.args.fit["single_bin"] == False:
            self.model = cross_section.RegionFit.CreateModel(self.args.toy_template, self.args.energy_slices, self.args.fit["mean_track_score"], False, self.args.toy_template.weights, self.args.fit["mc_stat_unc"], True, self.args.fit["single_bin"])
        xs = self.Analyse(analysis_input_data, None)
        return xs
    

class TrackLengthResolutionSystematic(MCMethod):
    name = "track_length_resolution"

    def __init__(self, args: cross_section.argparse.Namespace, model: cross_section.pyhf.Model, data_config: dict) -> None:
        super().__init__(args, model, data_config)
        self.track_length_original = self.args.toy_template.track_length_reco


    def RunExperiment(self, analysis_input_data : cross_section.AnalysisInput, resolution : float) -> tuple[dict, dict]:
        track_length_smeared = self.track_length_original * (1 + np.random.normal(0, resolution, len(self.track_length_original)))

        KE_int_reco = cross_section.BetheBloch.InteractingKE(self.args.toy_template.KE_init_reco, track_length_smeared, 10)
        self.args.toy_template.KE_int_reco = KE_int_reco

        if self.args.fit["single_bin"] == False:
            self.model = cross_section.RegionFit.CreateModel(self.args.toy_template, self.args.energy_slices, self.args.fit["mean_track_score"], False, self.args.toy_template.weights, self.args.fit["mc_stat_unc"], True, self.args.fit["single_bin"])
        xs = self.Analyse(analysis_input_data, None)
        return xs
    

    def CalculateResolution(self, book : Plots.PlotBook = Plots.PlotBook.null):
        ai_mc = cross_section.AnalysisInput.FromFile(self.args.analysis_input["mc"])

        r = np.array(cross_section.nandiv(ai_mc.track_length_reco - ai_mc.track_length_true, ai_mc.track_length_reco))

        y, edges = np.histogram(r, 150, [-0.5, 0.5])
        x = cross_section.bin_centers(edges)

        p = cross_section.Fitting.Fit(x, y,np.sqrt(y), cross_section.Fitting.double_crystal_ball, method = "dogbox", plot = True, xlabel = "$\\theta$", ylabel = "Counts", plot_style = "scatter")
        book.Save()

        y_interp = cross_section.Fitting.double_crystal_ball(x, *p[0])

        w = []
        for v in [-0.5, 0.5]:
            for i in np.linspace(0, v, 10000):
                if cross_section.Fitting.double_crystal_ball(i, *p[0]) <= (max(y_interp) / 2):
                    print(i)
                    break
            w.append(i)

        self.resolution = max(w) - min(w)
        self.popt = p
        return self.resolution


class TheoryShape(MCMethod):
    def __init__(self, args: cross_section.argparse.Namespace, model: cross_section.pyhf.Model, data_config: dict) -> None:
        super().__init__(args, model, data_config)


    def GenerateKEShapeWeights(self, bins : int, shape_lims : list):
        KE_int = self.args.toy_template.KE_int_true
        hist, edges = np.histogram(KE_int[~np.isnan(KE_int)], bins = bins)
        weights = np.random.uniform(min(shape_lims), max(shape_lims), len(hist))
        scale = sum(hist) / sum(hist * weights)
        weights = scale * weights

        ind = np.digitize(KE_int, edges[1:-1])
        return weights[ind]


    def RunExperiment(self, analysis_input_data : cross_section.AnalysisInput) -> tuple[dict, dict]:
        weights = self.GenerateKEShapeWeights(50, [0.8, 1.2])
        self.args.toy_template.weights = weights

        xs = self.Analyse(analysis_input_data, None)
        return xs


class NormalisationSystematic(MCMethod):
    def RunExperiment(self, config : dict) -> tuple[dict, dict]:
        x = self.args.energy_slices.pos[:-1] - self.args.energy_slices.width / 2

        ai = cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(df = cex_toy_generator.run(config)))

        xs = self.Analyse(ai, None)
        xs_sim_mod = cex_toy_generator.ModifyGeantXS(scale_factors = config["pdf_scale_factors"])

        if self.args.fit["regions"]:
            xs_true = {k : xs_sim_mod.GetInterpolatedCurve(k)(x) for k in exclusive_proc}
        else:
            xs_true = xs_sim_mod.GetInterpolatedCurve(self.args.signal_process)(x)

        return xs, xs_true


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

                    output = self.RunExperiment(config)
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
                    # Plots.plt.ylim(0, 1.5 * max(results["cv"][r][n][p][0]))
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
            return cross_section.quadsum([abs(r[p][0] - tr[p]) for p in r], 0)

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
            frac_low[i] = sys_err_avg["low"][i] / xs_nominal[i][0]
            frac_high[i] = sys_err_avg["high"][i] / xs_nominal[i][0]
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
            ls = pd.Series(sys["fractional"]["low"][p])
            ls.name = "Fit inaccuracy low"

            hs = pd.Series(sys["fractional"]["high"][p])
            hs.name = "Fit inaccuracy high"

            t = pd.Series(quadsum([d, ls, hs], 0))
            t.name = "Total"

            table = pd.concat([KEs, t, d, ls, hs], axis = 1).sort_values(by = ["$KE$ (MeV)"])

            avg = table.mean()
            avg["$KE$ (MeV)"] = "average"
            tables[p] = pd.concat([table, pd.DataFrame(avg).T]).reset_index(drop = True)
        return tables


def CreateSystematicTable(out : str, data_only : dict, args : cross_section.argparse.Namespace) -> dict[pd.DataFrame]:
    sys_err = {}
    for f in cross_section.ls_recursive(out):
        if "sys.dill" in f:
            sys_err[f.split("/")[-2]] = cross_section.LoadObject(f)["systematic"]

    tables_dict = {p : {} for p in exclusive_proc}
    for k, v in data_only["pdsp"].items():
        tables_dict[k]["Data stat"] = v[1]
    for s in sys_err:
        t = cross_section.pd.DataFrame(sys_err[s])
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
        table = cross_section.pd.DataFrame(tables_dict[t]).T

        for s in systematics_label.values():
            filtered = table.filter(regex = s, axis = "index")
            if len(filtered) > 1:
                table = table.drop(index = filtered.index)
                mask = filtered.where(filtered == 0, False) == filtered.where(filtered != 0, False)
                mask = mask.values
                zero = mask[0] | mask[1]
                n = np.where(zero, 1, 0.5)
                table = cross_section.pd.concat([table, cross_section.pd.DataFrame({s : n * (filtered.loc[f"{s} low"] + filtered.loc[f"{s} high"])}).T])

        total_err = {}
        for i in table:
            total_err[i] = quadsum(table[i], 0)    
        total_err = cross_section.pd.DataFrame(total_err, index = ["Total uncertainty (mb)"])

        total_sys_err = {}
        for i in table:
            total_sys_err[i] = quadsum(table[i].drop(index = "Data stat"), 0)
        total_sys_err = cross_section.pd.DataFrame(total_sys_err, index = ["Total systematic uncertainty (mb)"])

        names = {c : c + " (mb)" for c in table.index}
        table = table.rename(index = names)

        fmt_tables[t] = cross_section.pd.concat([cross_section.pd.DataFrame({"KE (MeV)" : x, "Central value (mb)" : data_only["pdsp"][t][0]}).T, table, total_sys_err, total_err])
    return fmt_tables


def SaveSystematicTables(systematic_tables : dict[pd.DataFrame], out : str):
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


def PlotSysHist(systematic_table : dict[pd.DataFrame], book : Plots.PlotBook = Plots.PlotBook.null):
    for t in systematic_table:
        Plots.plt.figure()
        for i in systematic_table[t].T:
            if i in ["KE (MeV)", "Central value (mb)"] : continue
            if i in ["Total uncertainty (mb)", "Total systematic uncertainty (mb)"]:
                color = "k"
            else:
                color = None
            if i  == "Total uncertainty (mb)":
                linestyle = "dashdot"
            else:
                linestyle = "-"
            Plots.Plot(systematic_table[t].loc["KE (MeV)"], systematic_table[t].loc[i], newFigure = False, xlabel = "KE (MeV)", ylabel = "Uncertainty (mb)", label = i.split(" (mb)")[0], title = cross_section.remove_(t).capitalize(), style = "step", linestyle = linestyle, color = color)
            Plots.plt.legend(ncols = 2)
        Plots.plt.ylim(0, 1.5 * max(Plots.plt.gca().get_ylim()))
        book.Save()

    table_filtered = {k : v[~v.index.isin(["KE (MeV)", "Central value (mb)", "Total uncertainty (mb)", "Total systematic uncertainty (mb)"])] for k, v in systematic_table.items()}
    order = {k : v.sum(axis = 1).values.argsort() for k, v in table_filtered.items()}
    total = {k : v.sum(axis = 0) for k, v in table_filtered.items()}


    for t in systematic_table:
        Plots.plt.figure()
        prev = 0
        for e, i in enumerate(reversed(order[t])):
            v = table_filtered[t].iloc[i]
            prev = prev + v.T.values
            Plots.Plot(systematic_table[t].loc["KE (MeV)"].values, prev, newFigure = False, xlabel = "KE (MeV)", ylabel = "Relative uncertainty (mb)", label = v.name.split(" (mb)")[0], title = cross_section.remove_(t).capitalize(), style = "bar", linestyle = "-", color = f"C{i}", zorder = 100-e)
        Plots.plt.legend(ncols = 2)
        Plots.plt.ylim(0, 1.5 * max(Plots.plt.gca().get_ylim()))
        book.Save()

    for t in systematic_table:
        Plots.plt.figure()
        prev = 0
        for e, i in enumerate(reversed(order[t])):
            v = table_filtered[t].iloc[i]
            prev = prev + v.T.values
            Plots.Plot(systematic_table[t].loc["KE (MeV)"].values, prev/total[t], newFigure = False, xlabel = "KE (MeV)", ylabel = "Relative uncertainty", label = v.name.split(" (mb)")[0], title = cross_section.remove_(t).capitalize(), style = "bar", linestyle = "-", color = f"C{i}", zorder = 100-e)
        Plots.plt.legend(ncols = 2)
        Plots.plt.ylim(0, 1.5 * max(Plots.plt.gca().get_ylim()))
        book.Save()

    return


def FinalPlots(cv, systematics_table : dict[pd.DataFrame], energy_slices, book : Plots.PlotBook = Plots.PlotBook.null, alt_xs : bool = False):
    goodness_of_fit = {}
    if alt_xs:
        xs_alt = cross_section.GeantCrossSections(cross_section.os.environ["PYTHONPATH"] + "/data/g4_xs_pi_KE_100.root", energy_range = [energy_slices.min_pos - energy_slices.width, energy_slices.max_pos])
    goodness_of_fit_alt = {}
    for p in cv:
        xs = {
            "ProtoDUNE SP: Data Stat + Sys Error" : cv[p],
            "" : [cv[p][0], systematics_table[p].loc["Total systematic uncertainty (mb)"]]
        }
        cross_section.PlotXSComparison(xs, energy_slices, p, simulation_label = "Geant4 v10.6", colors = {k : f"C0" for k in xs}, chi2 = False)
        goodness_of_fit[p] = cross_section.HypTestXS(cv[p][0], systematics_table[p].loc["Total systematic uncertainty (mb)"], p, energy_slices)
        if alt_xs:
            xs_alt.Plot(p, "red", label = "Geant4 v10.6, $\pi^{\pm}$:$2^{nd}$ $KE > 100 MeV$")
            goodness_of_fit_alt[p] = cross_section.HypTestXS(cv[p][0], systematics_table[p].loc["Total systematic uncertainty (mb)"], p, energy_slices, cross_section.os.environ["PYTHONPATH"] + "/data/g4_xs_pi_KE_100.root")
        book.Save()
    return pd.DataFrame(goodness_of_fit), pd.DataFrame(goodness_of_fit_alt)


def can_run(systematic : str):
    run = ((systematic not in args.skip) and (systematic in args.run)) or ("all" in args.run)
    if run is True:
        print(Rule(cross_section.remove_(systematic)))
    return run


def can_regen(dir : str):
    if cross_section.os.path.exists(dir) and (args.regen is False):
        if len(cross_section.ls_recursive(dir)) > 0:
            for f in cross_section.ls_recursive(dir):
                if ("sys.dill" in f):
                    return False
            return True
        else:
            return True
    else:
        return True

@cross_section.timer
def main(args : cross_section.argparse.Namespace):
    cross_section.PlotStyler.SetPlotStyle(dark = True)
    out = args.out + "systematics/"
    cross_section.os.makedirs(out, exist_ok = True)

    print(f"{args.skip=}")
    print(f"{args.run=}")

    if args.toy_template:
        args.toy_template = cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(file = args.toy_template))
        model = cross_section.RegionFit.CreateModel(args.toy_template, args.energy_slices, args.fit["mean_track_score"], False, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])
    
        toy_nominal = cross_section.Toy(df = cex_toy_generator.run(args.toy_data_config))
        analysis_input_nominal = cross_section.AnalysisInput.CreateAnalysisInputToy(toy_nominal)


    if ("all" not in args.skip) or ("all" in args.run):
        if can_run("mc_stat"):
            outdir = out + "mc_stat/"
            cross_section.os.makedirs(outdir, exist_ok = True)

            mc_stat = NuisanceParameters(args)
            if can_regen(outdir):
                result = mc_stat.RunExperiment()
                sys = mc_stat.CalculateSysError(result)
                cross_section.SaveObject(f"{outdir}result.dill", result)
                SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = cross_section.LoadObject(f"{outdir}result.dill")
                sys = cross_section.LoadObject(f"{outdir}sys.dill")

            with Plots.PlotBook(f"{outdir}plots.pdf") as book:
                mc_stat.PlotXSMCStat(result, book)
            tables = mc_stat.MCStatTables(result)
            SaveTables(tables, outdir, 2)

        if can_run("shower_energy"):
            outdir = out + "shower_energy/"
            sc = ShowerEnergyCorrectionSystematic(args)
            if can_regen(outdir):
                sc.CreateNewAIs(outdir)
                result = sc.RunAnalysis(outdir)
                sys = sc.CalculateSysErrorAsym(args.cv, result)
                cross_section.SaveObject(f"{outdir}result.dill", result)
                SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = cross_section.LoadObject(f"{outdir}result.dill")
                sys = cross_section.LoadObject(f"{outdir}sys.dill")["systematic"]

            with Plots.PlotBook(f"{outdir}plots.pdf") as book:
                sc.PlotResults(args.cv, result, book)
            tables = sc.DataAnalysisTables(args.cv, sys, "Shower energy")
            SaveTables(tables, outdir, 2)

        if can_run("upstream"):
            outdir = out + "upstream/"
            upl = UpstreamCorrectionSystematic(args)

            if can_regen(outdir):
                upl.CreateNewAIs(outdir)
                result = upl.RunAnalysis(outdir)
                sys = upl.CalculateSysErrorAsym(args.cv, result)

                cross_section.SaveObject(f"{outdir}result.dill", result)
                SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = cross_section.LoadObject(f"{outdir}result.dill")
                sys = cross_section.LoadObject(f"{outdir}sys.dill")

            with Plots.PlotBook(f"{outdir}plots.pdf") as book:
                upl.PlotResults(args.cv, result, book)
            tables = upl.DataAnalysisTables(args.cv, sys, "Upstream")
            SaveTables(tables, outdir, 2)

        if can_run("beam_reweight"):
            outdir = out + "beam_reweight/"
            brw = BeamReweightSystematic(args)

            if can_regen(outdir):
                brw.CreateNewAIs(outdir)
                result = brw.RunAnalysis(outdir)
                sys = brw.CalculateSysErrorAsym(args.cv, result)

                cross_section.SaveObject(f"{outdir}result.dill", result)
                SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = cross_section.LoadObject(f"{outdir}result.dill")
                sys = cross_section.LoadObject(f"{outdir}sys.dill")["systematic"]

            with Plots.PlotBook(f"{outdir}plots.pdf") as book:
                brw.PlotResults(args.cv, result, book)
            tables = brw.DataAnalysisTables(args.cv, sys, "Reweight")
            SaveTables(tables, outdir, 2)

        if can_run("track_length"):
            outdir = out + "track_length/"
            cross_section.os.makedirs(outdir, exist_ok = True)
            trk = TrackLengthResolutionSystematic(args, model, args.toy_data_config)

            with Plots.PlotBook(outdir + "trk_res.pdf") as book:
                trk.CalculateResolution(book)
            table = {}
            for i in range(len(trk.popt[0])):
                table[f"$p_{{{i}}}$"] = round_value_to_error(trk.popt[0][i], trk.popt[1][i])
            pd.DataFrame(table, index = [0]).style.hide(axis = "index").to_latex(outdir + "fit_params.tex")

            if can_regen(outdir):
                result = trk.Evaluate(1, analysis_input_data = analysis_input_nominal, resolution = trk.resolution)
                with Plots.PlotBook(f"{outdir}cov_mat") as book:
                    sys = trk.CalculateSysCov(result, book)
                cross_section.SaveObject(f"{outdir}result.dill", result)
                SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = cross_section.LoadObject(f"{outdir}result.dill")
                sys = cross_section.LoadObject(f"{outdir}sys.dill")["systematic"]

            tables = trk.Tables(args.cv, sys, "Track length")
            SaveTables(tables, outdir, 3)

        if can_run("beam_res"):
            outdir = out + "beam_res/"
            cross_section.os.makedirs(outdir, exist_ok = True)
            resolution = 2.5/100
            bm = BeamMomentumResolutionSystematic(args, model, args.toy_data_config)

            if can_regen(outdir):
                result = bm.Evaluate(1, analysis_input_data = analysis_input_nominal, resolution = resolution)
                with Plots.PlotBook(f"{outdir}cov_mat") as book:
                    sys = bm.CalculateSysCov(result, book)
                cross_section.SaveObject(f"{outdir}results.dill", result)
                SaveSystematicError(sys, None, f"{outdir}sys.dill")
            else:
                result = cross_section.LoadObject(f"{outdir}result.dill")
                sys = cross_section.LoadObject(f"{outdir}sys.dill")["systematic"]

            tables = bm.Tables(args.cv, sys, "Beam momentum")
            SaveTables(tables, outdir, 3)

        if can_run("theory"):
            outdir = out + "theory/"
            cross_section.os.makedirs(outdir, exist_ok = True)

            ts = TheoryShape(args, model, args.toy_data_config)

            if can_regen(outdir):
                result = ts.Evaluate(1, analysis_input_data = analysis_input_nominal)
                print(f"saving: {outdir}results.dill")
                cross_section.SaveObject(f"{outdir}results.dill", result)
            else:
                result = cross_section.LoadObject(f"{outdir}result.dill")
                # sys = cross_section.LoadObject(f"{outdir}sys.dill")["systematic"]

            if len(result) > 1:
                with Plots.PlotBook(f"{outdir}cov_mat") as book:
                    sys = ts.CalculateSysCov(result, book)
                SaveSystematicError(sys, None, f"{outdir}sys.dill")

                tables = ts.Tables(args.cv, sys, "Theory")
                SaveTables(tables, outdir, 3)
            else:
                print("only one experiment was ran, run more experiments to produce the systematic errors.")

        if can_run("fit_inaccuracy"):
            outdir = out + "fit_inaccuracy/"
            cross_section.os.makedirs(outdir, exist_ok = True)
            
            norm_sys = NormalisationSystematic(args, model, args.toy_data_config)
            
            xs_nominal = norm_sys.Analyse(analysis_input_nominal, None)

            if can_regen(outdir):
                if not cross_section.os.path.isfile(f"{outdir}test_results.dill"):    
                    results = norm_sys.Evaluate([0.8, 1.2], 1)
                    cross_section.SaveObject(f"{outdir}test_results.dill", results)

                results = cross_section.LoadObject(f"{outdir}test_results.dill")
                NormalisationSystematic.AverageResults(results)
                sys_err = NormalisationSystematic.CalculateSysErr(results)
                frac_err = NormalisationSystematic.CalculateFractionalError(sys_err, xs_nominal)
                SaveSystematicError(sys_err, frac_err, f"{outdir}sys.dill")
            else:
                results = cross_section.LoadObject(f"{outdir}test_results.dill")
                NormalisationSystematic.AverageResults(results)
            sys = cross_section.LoadObject(f"{outdir}sys.dill")

            with Plots.PlotBook(f"{outdir}plots", True) as book:
                #! use styler properly
                cross_section.PlotStyler.SetPlotStyle(dark = False, extend_colors = True)
                NormalisationSystematic.PlotNormalisationTestResults(results, args, xs_nominal, book)
                cross_section.PlotStyler.SetPlotStyle(dark = True, extend_colors = False)

            tables = norm_sys.CreateTables(args.cv, sys)
            SaveTables(tables, outdir, 2)

    if args.plot:
        outdir = out + "combined/"
        cross_section.os.makedirs(outdir, exist_ok = True)
        tables = CreateSystematicTable(out, args.cv, args)
        SaveSystematicTables(tables, outdir)

        with Plots.PlotBook(outdir + "plots.pdf") as book:
            with Plots.matplotlib.rc_context({"axes.prop_cycle" : Plots.plt.cycler("color", Plots.matplotlib.cm.get_cmap("tab20").colors)}):
                PlotSysHist(tables, book)
            table, table_alt = FinalPlots(args.cv["pdsp"], tables, args.energy_slices, book, alt_xs = False)

        tags = cross_section.Tags.ExclusiveProcessTags(None)
        table = table.rename(index = {"w_chi2" : "$\chi^{2}/ndf$", "p" : "$p$"}, columns={t : tags[t].name_simple.capitalize() for t in tags})
        table.style.format("{:.3g}").to_latex(outdir + "goodness_of_fit.tex")

        if len(table_alt) > 0:
            table_alt = table_alt.rename(index = {"w_chi2" : "$\chi^{2}/ndf$", "p" : "$p$"}, columns={t : tags[t].name_simple.capitalize() for t in tags})
            table_alt.style.format("{:.3g}").to_latex(outdir + "goodness_of_fit_alt.tex")
    return

if __name__ == "__main__":

    parser = cross_section.argparse.ArgumentParser("Estimate Systematics for the cross section analysis")
    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser, "systematic/")

    parser.add_argument("--cv", "-v", dest = "cv", type = str, default = None, help = "plot systematics with central value measurement", required = True)

    parser.add_argument("--toy_template", "-t", dest = "toy_template", type = str, help = "toy template hdf5 file", required = False)
    parser.add_argument("--toy_data_config", "-d", dest = "toy_data_config", type = str, help = "json config for toy data", required = False)

    parser.add_argument("--skip", type = str, nargs = "+", default = [], choices = systematics)
    parser.add_argument("--run", type = str, nargs = "+", default = [], choices = systematics)

    parser.add_argument("--regen", "-r", dest = "regen", action = "store_true", help = "fully rerun systematic tests if results already exist")


    parser.add_argument("--plot", "-p", dest = "plot", action = "store_true", default = None, help = "plot systematics with central value measurement")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())

    if ("all" in args.run) or ("fit_inaccuracy" in args.run) or ("track_length" in args.run) or ("beam_res" in args.run) or ("theory" in args.run):
        if not args.toy_template:
            raise Exception("--toy_template must be specified")
        if not args.toy_data_config:
            raise Exception("--toy_data_config must be specified")        
    if args.toy_data_config:
        args.toy_data_config = cross_section.LoadConfiguration(args.toy_data_config)

    args.cv = cross_section.LoadObject(args.cv)

    print(vars(args))
    main(args)