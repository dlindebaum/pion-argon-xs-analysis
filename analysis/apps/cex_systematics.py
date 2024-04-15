#!/usr/bin/env python3
"""
Created on: 09/04/2024 13:53

Author: Shyam Bhuller

Description: app to calculate systematic uncertainties for cross section measurement
"""
from abc import ABC, abstractmethod

import numpy as np

from rich import print

from python.analysis import cross_section, Plots
from python.analysis.Utils import dill_copy, quadsum
from apps import cex_toy_generator, cex_analyse, cex_fit_studies, cex_analysis_input

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


class DataAnalysis:
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

    @staticmethod
    def CalculateSysError(xs):
        return {p : abs((xs["high"][p][0] - xs["low"][p][0]) / 2) for p in xs["low"]}


class NuisanceParameters:
    pass


class UpstreamCorrectionSystematic(DataAnalysis):
    name = "upstream_loss_1_sigma"

    def CreateNewAIs(self, outdir : str):
        upl = {
            "low" : {f"p{i}" : self.args.upstream_loss_correction_params["value"][f"p{i}"] - self.args.upstream_loss_correction_params["error"][f"p{i}"] for i in range(getattr(cross_section.Fitting, self.args.upstream_loss_cv_function).n_params)},
            "high" : {f"p{i}" : self.args.upstream_loss_correction_params["value"][f"p{i}"] + self.args.upstream_loss_correction_params["error"][f"p{i}"] for i in range(getattr(cross_section.Fitting, self.args.upstream_loss_cv_function).n_params)}
        }

        args_copy = dill_copy(self.args)
        for k, v in upl.items():
            args_copy.upstream_loss_correction_params["value"] = v
            args_copy.out = f"{outdir}{self.name}_{k}/"
            cex_analysis_input.main(args_copy)
        return


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


def PlotSysHist(systematics, energy_slices, book : Plots.PlotBook = Plots.PlotBook.null):
    x = energy_slices.pos[:-1] - energy_slices.width/2
    for _, p in Plots.IterMultiPlot(exclusive_proc):
        for s in systematics:
            if systematics[s][p].shape == (2, len(x)):
                err = cross_section.quadsum(systematics[s][p], 0) / 2
            else:
                err = systematics[s][p]
            Plots.Plot(x, err, label = s, title = cross_section.remove_(p), newFigure = False, style = "step", xlabel = "$KE$ (MeV)", ylabel = "Systematic error (mb)")
    book.Save()


def FinalPlots(cv, systematics, energy_slices, book : Plots.PlotBook = Plots.PlotBook.null):
    for p in cv:
        xs = {
            "ProtoDUNE SP: Stat + Sys Error" : cv[p],
            "" : [cv[p][0], cross_section.quadsum([cv[p][1]] + [systematics[s][p] for s in systematics], 0)]
        }
        # xs = {"ProtoDUNE SP: stat + sys" : [cv[p][0], cross_section.quadsum([cv[p][1]] + [systematics[s][p] for s in systematics], 0)]}
        cross_section.PlotXSComparison(xs, energy_slices, p, simulation_label = "Geant4 v10.6", colors = {k : f"C0" for k in xs}, chi2 = False)
        book.Save()
    for _, p in Plots.IterMultiPlot(cv):
        xs = {
            "ProtoDUNE SP: Stat + Sys Error" : cv[p],
            "" : [cv[p][0], cross_section.quadsum([cv[p][1]] + [systematics[s][p] for s in systematics], 0)]
        }
        # xs = {"ProtoDUNE SP: stat + sys" : [cv[p][0], cross_section.quadsum([cv[p][1]] + [systematics[s][p] for s in systematics], 0)]}
        cross_section.PlotXSComparison(xs, energy_slices, p, simulation_label = "Geant4 v10.6", colors = {k : f"C0" for k in xs}, chi2 = False, newFigure = False)
        book.Save()
    book.Save()
    return


@cross_section.timer
def main(args : cross_section.argparse.Namespace):
    cross_section.SetPlotStyle(dark = True)
    out = args.out + "systematics/"
    cross_section.os.makedirs(out, exist_ok = True)

    if ("all" not in args.skip) or ("all" in args.run):
        if ("upstream" not in args.skip) or ("upstream" in args.run):
            upl = UpstreamCorrectionSystematic(args)
            upl.CreateNewAIs(out + "upstream/")
            xs = upl.RunAnalysis(out + "upstream/")
            sys = upl.CalculateSysError(xs)
            SaveSystematicError(sys, None, out + "upstream/sys.dill")

        if ("theory" not in args.skip) or ("theory" in args.run):
            cross_section.os.makedirs(out + "theory/", exist_ok = True)

            args.toy_template = cross_section.AnalysisInput.CreateAnalysisInputToy(cross_section.Toy(file = args.toy_template))

            model = cross_section.RegionFit.CreateModel(args.toy_template, args.energy_slices, args.fit["mean_track_score"], False, None, args.fit["mc_stat_unc"], True, args.fit["single_bin"])

            toy_nominal = cross_section.Toy(df = cex_toy_generator.run(args.toy_data_config))

            analysis_input_nominal = cross_section.AnalysisInput.CreateAnalysisInputToy(toy_nominal)

            norm_sys = NormalisationSystematic(args, model, args.toy_data_config)
            
            xs_nominal = norm_sys.Analyse(analysis_input_nominal, None)
            results = norm_sys.Evaluate([0.8, 1.2], 3)
            cross_section.SaveObject(out + "theory/test_results.dill", results)

            NormalisationSystematic.AverageResults(results)

            NormalisationSystematic.PlotNormalisationTestResults(results, args, xs_nominal)

            sys_err = NormalisationSystematic.CalculateSysErr(results)
            # norm_sys_max = NormalisationSystematic.TotalSysMax(sys_err)
            frac_err = NormalisationSystematic.CalculateFractionalError(sys_err, xs_nominal)
            # norm_sys_qs = cex_systematics.NormalisationSystematic.TotalSysQS(sys_err)
            SaveSystematicError(sys_err, frac_err, out + "theory/sys.dill")

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
            sys = cross_section.LoadObject(f"{out}{f}/sys.dill")
            if f == "theory":
                t = TheoryXS(sys, cv)
                systematics =  {**systematics, **{f"theory, {label_short[k]}" : t[k] for k in t}}
            if f == "upstream":
                systematics[f] = sys["systematic"]

    return

if __name__ == "__main__":
    systematics = ["mc_stat", "theory", "upstream", "beam_reweight", "shower_energy", "all"]

    parser = cross_section.argparse.ArgumentParser("Estimate Systematics for the cross section analysis")
    cross_section.ApplicationArguments.Config(parser)
    cross_section.ApplicationArguments.Output(parser)

    parser.add_argument("--toy_template", "-t", dest = "toy_template", type = str, help = "toy template hdf5 file", required = False)
    parser.add_argument("--toy_data_config", "-d", dest = "toy_data_config", type = str, help = "json config for toy data", required = False)

    parser.add_argument("--skip", type = str, nargs = "+", default = [], choices = systematics)
    parser.add_argument("--run", type = str, nargs = "+", default = [], choices = systematics)

    parser.add_argument("--plot", "-p", dest = "plot", type = str, default = None, help = "plot systematics with central value measurement")

    args = cross_section.ApplicationArguments.ResolveArgs(parser.parse_args())

    if ("all" in args.run) or ("theory" in args.run):
        if not args.toy_template:
            raise Exception("--toy_template must be specified")
        if not args.toy_data_config:
            raise Exception("--toy_data_config must be specified")        
        args.toy_data_config = cross_section.LoadConfiguration(args.toy_data_config)
    print(vars(args))
    main(args)