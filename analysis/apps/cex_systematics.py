#!/usr/bin/env python3
"""
Created on: 09/04/2024 13:53

Author: Shyam Bhuller

Description: app to calculate systematic uncertainties for cross section measurement
"""
from abc import ABC, abstractmethod

import numpy as np

from python.analysis import cross_section, Plots
from apps import cex_toy_generator, cex_analyse, cex_fit_studies

exclusive_proc = ["absorption", "charge_exchange", "single_pion_production", "pion_production"]

class MCMethod(ABC):
    def __init__(self, args : cross_section.argparse.Namespace, model : cross_section.pyhf.Model, data_config : dict) -> None:
        self.args = args
        self.model = model
        self.data_config = data_config
        pass

    @cross_section.timer
    def Analyse(self, analysis_input : cross_section.AnalysisInput, book : Plots.PlotBook = Plots.PlotBook.null):

        region_fit_result = cex_analyse.RegionFit(analysis_input, self.args.energy_slices, self.args.fit["mean_track_score"], self.model, mc_stat_unc = self.args.fit["mc_stat_unc"], single_bin = self.args.fit["single_bin"])

        _, histograms_reco_obs, histograms_reco_obs_err = cex_analyse.BackgroundSubtraction(analysis_input, self.args.signal_process, self.args.energy_slices, region_fit_result, self.args.fit["single_bin"], self.args.fit["regions"], self.args.toy_template, book)


        if self.args.fit["regions"]:
            histograms_reco_obs = {**histograms_reco_obs, **histograms_reco_obs["int_ex"]}
            histograms_reco_obs.pop("int_ex")
            histograms_reco_obs_err = {**histograms_reco_obs_err, **histograms_reco_obs_err["int_ex"]}
            histograms_reco_obs_err.pop("int_ex")

        unfolding_result = cex_analyse.Unfolding(histograms_reco_obs, histograms_reco_obs_err, self.args.toy_template, self.args.unfolding, self.args.signal_process, len(analysis_input.KE_init_reco)/len(self.args.toy_template.KE_init_reco), self.args.energy_slices, self.args.fit["regions"], None, book)

        if book is not None:
            for p in cex_analyse.process_labels:
                hist_true = analysis_input.CreateHistograms(self.args.energy_slices, p, False, None)["int_ex"]
                Plots.Plot(self.args.energy_slices.pos_overflow, unfolding_result[p]["unfolded"], yerr = cross_section.quadsum([unfolding_result[p]["stat_err"], unfolding_result[p]["sys_err"]], 0), xlabel = f"$N_{{int, {cex_analyse.process_labels[p]}}}$ (MeV)", ylabel = "Counts", label = "unfolded", style = "step", color = "C6")
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


class DataAnalysis:
    pass

class NuisanceParameters:
    pass


class NormalisationSystematic(MCMethod):
    def Evaluate(self, norms = [0.8, 1.2], repeats : int = 1):
        cvs = {}
        true_cvs = {}
        for target in exclusive_proc:
            print(f"{target=}")
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
    def PlotNormalisationTestResults(results : dict, args : cross_section.argparse.Namespace, xs_nominal : dict):
        with Plots.PlotBook("test/normalisation_systematic/raw_results", False) as book:
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
            return {p : r[p][0] - tr[p] for p in r}

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
                cross_section.quadsum([sys_err["low"][r][p] for r in sys_err["low"]], 0),
                cross_section.quadsum([sys_err["high"][r][p] for r in sys_err["high"]], 0)
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

                t_avg[n] = {k : [np.mean(np.array(v), 0), np.mean(np.array(cv_err[k]), 0) ] for k, v in cv.items()}
            cv_avg[t] = t_avg

        results["cv"] = cv_avg