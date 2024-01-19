#!/usr/bin/env python3
"""
Created on: 02/12/2023 13:14

Author: Shyam Bhuller

Description: reweighting parameters for beam momentum.
"""
import os

import numpy as np
import awkward as ak

from alive_progress import alive_bar
from iminuit import minimize
from rich import print

from python.analysis import cross_section, Plots, vector, Processing

from apps.cex_analyse import BeamPionSelection

def RecoFractionalRange(events : cross_section.Data, particle : cross_section.Particle, args):
    reco_KE_inst = cross_section.KE(events.recoParticles.beam_inst_P, particle.mass)
    reco_upstream_loss = cross_section.UpstreamEnergyLoss(reco_KE_inst, args.upstream_loss_correction_params["value"])
    reco_KE_ff = reco_KE_inst - reco_upstream_loss
    return events.recoParticles.beam_track_length / cross_section.BetheBloch.RangeFromKE(reco_KE_ff, particle)


def Weights(p_inst_true, mu, sigma, mu_0, sigma_0):
    weights = cross_section.Fitting.gaussian(p_inst_true, 1, mu, sigma) / cross_section.Fitting.gaussian(p_inst_true, 1, mu_0, sigma_0)
    return np.where(weights > 3, 3, weights)


def Chi2KE(p, args):
    mu_fit = p[0]
    sigma_fit = p[1]

    weights = Weights(args["p_inst_true"], mu_fit, sigma_fit, args["mu_0"], args["sigma_0"])
    
    mc = np.clip(args["KE_init_range"]["mc"], min(args["bins"]), max(args["bins"]))
    data = np.clip(args["KE_init_range"]["data"], min(args["bins"]), max(args["bins"]))

    hist_mc = np.histogram(mc, args["bins"], weights = weights)[0]
    hist_data = np.histogram(data, args["bins"])[0]
    
    hist_mc = hist_mc * args["norm"]

    bin_index = np.digitize(mc, args["bins"][:-1])

    s = []
    for i in np.linspace(min(bin_index), max(bin_index), len(hist_mc), True):
        w = weights[np.where(bin_index == i)[0]]
        s.append(cross_section.nandiv(sum(w**2), sum(w)))
    s = np.array(s)
    chi2 = (hist_data - hist_mc)**2 / ((s * hist_data) + hist_mc)
    chi2 = np.nansum(chi2) / len(hist_mc)-1
    return chi2


def KEs(samples, upstream_energy_loss_params, smearing : np.array = None):
    KE_samples = {}

    if smearing is None:
        smearing = np.zeros(len(samples["mc"].eventNum))

    for s in samples:
        obs = {}
        p = samples[s].recoParticles.beam_inst_P + smearing if s == "mc" else samples[s].recoParticles.beam_inst_P
        obs["reco_KE_inst"] = cross_section.KE(p, cross_section.Particle.from_pdgid(211).mass)

        reco_upstream_loss = cross_section.UpstreamEnergyLoss(obs["reco_KE_inst"], upstream_energy_loss_params)
        obs["reco_KE_ff"] = obs["reco_KE_inst"] - reco_upstream_loss
        obs["reco_KE_int"] = obs["reco_KE_ff"] - cross_section.RecoDepositedEnergy(samples[s], obs["reco_KE_ff"], "bb")
        KE_samples[s] = obs
    return KE_samples


def MakePlots(pion_sample : dict[cross_section.Data], reco_KEs : dict[dict[np.array]], tags : cross_section.Tags, args : cross_section.argparse.Namespace, weights : np.array = None, smearing : np.array = None, book : Plots.PlotBook = Plots.PlotBook.null, bins : int = 50):
    if smearing is None:
        smearing = np.zeros(len(pion_sample["mc"].eventNum))
    Plots.PlotTagged(pion_sample["mc"].recoParticles.beam_inst_P + smearing, tags(pion_sample["mc"]), data2 = pion_sample["data"].recoParticles.beam_inst_P, bins = bins, x_range = [0.75 * args.beam_momentum, 1.25 * args.beam_momentum], norm = args.norm, loc = "upper left", x_label = "$P_{inst}^{reco}$(MeV)", data_weights = weights)
    book.Save()
    Plots.PlotTagged(reco_KEs["mc"]["reco_KE_inst"], tags(pion_sample["mc"]), data2 = reco_KEs["data"]["reco_KE_inst"], bins = bins, x_range = args.KE_inst_range, norm = args.norm, loc = "upper left", x_label = "$KE_{inst}^{reco}$(MeV)", data_weights = weights)
    book.Save()
    Plots.PlotTagged(reco_KEs["mc"]["reco_KE_ff"], tags(pion_sample["mc"]), data2 = reco_KEs["data"]["reco_KE_ff"], bins = bins, x_range = args.KE_init_range, norm = args.norm, loc = "upper left", x_label = "$KE_{init}^{reco}$(MeV)", data_weights = weights)
    book.Save()    
    Plots.PlotTagged(reco_KEs["mc"]["reco_KE_int"], tags(pion_sample["mc"]), data2 = reco_KEs["data"]["reco_KE_int"], bins = bins, x_range = args.KE_int_range, norm = args.norm, loc = "upper left", x_label = "$KE_{int}^{reco}$(MeV)", data_weights = weights)
    book.Save()
    return


def run(i, file, n_events, start, selected_events, args) -> dict:
    events = cross_section.Data(file, nEvents = n_events, start = start, nTuple_type = args.ntuple_type) # load data
    events = BeamPionSelection(events, args, not args.data)
    muon = cross_section.Particle.from_pdgid(-13)

    if args.data == False:
        true_fractional_range = events.trueParticles.beam_track_length / cross_section.BetheBloch.RangeFromKE(events.trueParticles.beam_KE_front_face, muon)
        p_inst_true = vector.magnitude(events.trueParticles.momentum[:, 0])
        stopping_muon_tag = cross_section.Tags.StoppingMuonTag(events)
    else:
        true_fractional_range = None
        p_inst_true = None
        stopping_muon_tag = None

    end_z = events.recoParticles.beam_endPos_SCE.z
    reco_fractional_range = RecoFractionalRange(events, muon, args)
    range_to_KE = cross_section.BetheBloch.interp_range_to_KE(max(events.recoParticles.beam_inst_P))
    KE_init_range = range_to_KE(events.recoParticles.beam_track_length)
    p_inst_reco = events.recoParticles.beam_inst_P

    output = {
        "true_fractional_range" : true_fractional_range,
        "p_inst_true" : p_inst_true,
        "p_inst_reco" : p_inst_reco,
        "end_z" : end_z,
        "reco_fractional_range" : reco_fractional_range,
        "KE_init_range" : KE_init_range,
        "stopping_muon_tag" : stopping_muon_tag
    }
    return output

def MergeOutputs(outputs : dict):
    merged = {}
    for output in outputs:
        for k, v in output.items():
            if k not in merged:
                merged[k] = v
            else:
                if v is not None:
                    if type(v) == cross_section.Tags.Tags:
                        for tag in merged[k]:
                            merged[k][tag].mask = ak.concatenate([merged[k][tag].mask, v[tag].mask])
                    else:
                        merged[k] = ak.concatenate([merged[k], v])
    return merged


def GaussFit(value, range : list, bins : int = 20, book : Plots.PlotBook = Plots.PlotBook.null, label : str = "value (units)", title : str = "", weights : ak.Array = None) -> tuple[list]:
    if weights is None:
        w = None
    else:
        w = np.array(weights)
    y, edges = np.histogram(np.array(value), bins, range = range, weights = w)
    x = (edges[1:] + edges[:-1]) / 2
    Plots.plt.figure()
    p, p_err = cross_section.Fitting.Fit(x, y, np.sqrt(y), cross_section.Fitting.gaussian, plot = True, title = title, xlabel = label)

    print(f"{p=}")
    print(f"{p_err=}")
    book.Save()
    return p, p_err


def Smearing(mc_momenta : ak.Array, data_momenta : ak.Array, bins : int, range : list[float], book : Plots.PlotBook.null, mc_weights : ak.Array = None):
    p_mc, p_mc_err = GaussFit(mc_momenta, range, bins, book, label = "$P_{inst}^{reco}$ (MeV)", title = "MC, sideband", weights = mc_weights)
    p_data, p_data_err = GaussFit(data_momenta, range, bins, book, label = "$P_{inst}^{reco}$ (MeV)", title = "Data, sideband")

    smearing_mu = p_data[1] - p_mc[1]
    smearing_sigma = np.sqrt(p_data[2]**2 - p_mc[2]**2)

    smearing_mu_err = np.sqrt(p_data_err[1]**2 + p_mc_err[1]**2)

    smearing_sigma_err = (1/smearing_sigma) * np.sqrt((p_data[2] * p_data_err[2])**2 + (p_mc[2] * p_mc_err[2])**2)

    return [smearing_mu, smearing_mu_err], [smearing_sigma, smearing_sigma_err], [p_mc[1:], p_mc_err[1:]], [p_data[1:], p_data_err[1:]]


def SelectStoppingMuon(output : dict):
    stopping_muon_mask = output["reco_fractional_range"] > 0.9
    for k in output:
        if type(output[k]) == cross_section.Tags.Tags:
            for tag in output[k]:
                output[k][tag].mask = output[k][tag].mask[stopping_muon_mask]
        elif output[k] is None:
            continue
        else:
            output[k] = output[k][stopping_muon_mask]
    return output


def main(args : cross_section.argparse.Namespace):
    cross_section.SetPlotStyle(extend_colors = True, dpi = 100)

    os.makedirs(args.out, exist_ok = True)
    args.data = False
    with alive_bar(title = "load mc") as bar:
        output_mc = MergeOutputs(Processing.mutliprocess(run, [args.mc_file], args.batches, args.events, args, args.threads))
    print(output_mc)

    args.data = True
    with alive_bar(title = "load data") as bar:
        output_data = MergeOutputs(Processing.mutliprocess(run, [args.data_file], args.batches, args.events, args, args.threads))
    print(output_data)

    with Plots.PlotBook(args.out + "stopping_muon_selection.pdf", True) as book:
        Plots.PlotTagged(output_mc["true_fractional_range"], output_mc["stopping_muon_tag"], ncols = 1, x_range = [0.01, 1.2], x_label = "fractional track length (MC truth)", title = "stopping muons")
        Plots.DrawCutPosition(args.cut, face = ">")
        book.Save()

        Plots.PlotTagged(output_mc["reco_fractional_range"], output_mc["stopping_muon_tag"], ncols = 1, x_range = [0, 1.2], x_label = "fractional track length (MC reco)", title = "stopping muons")
        Plots.DrawCutPosition(args.cut, face = ">")
        book.Save()

        Plots.PlotHist(output_data["reco_fractional_range"], range = [0, 1.2], xlabel = "fractional track length (Data)", title = "stopping muons")
        Plots.DrawCutPosition(args.cut, face = ">")
        book.Save()

        Plots.PlotHist2DMarginal(output_data["reco_fractional_range"], output_data["end_z"], x_range = [0, 1.2], xlabel = "fractional track length (MC reco)", ylabel = "end z position (cm)")
        book.Save()

        Plots.PlotHist2DMarginal(output_mc["p_inst_true"], output_mc["true_fractional_range"], x_range = args.P_inst_range, y_range = [0, 1.2], xlabel = "$P_{inst}^{true}(MeV)$", ylabel = "true fractional range")
        book.Save()

        Plots.PlotHist2DMarginal(output_mc["p_inst_true"], output_mc["reco_fractional_range"], x_range = args.P_inst_range, y_range = [0, 1.2], xlabel = "$P_{inst}^{true}(MeV)$", ylabel = "reco fractional range")
        book.Save()


    output_mc = SelectStoppingMuon(output_mc)
    output_data = SelectStoppingMuon(output_data)

    with Plots.PlotBook(args.out + "reweight_smearing.pdf") as book:
        p_0, p_0_err = GaussFit(output_mc["p_inst_true"]/args.beam_momentum, [0.75, 1.25], 14, book, "$P_{inst}^{true}$ (MeV)", "stopping muons")
        p_0 = p_0[1:]
        p_0_err = p_0_err[1:]

        KE_ff_range_bins = np.linspace(0.5 * args.beam_momentum, 1.2 * args.beam_momentum, 15)
        norm = len(output_data["reco_fractional_range"])/len(output_mc["reco_fractional_range"])
        fit_args = {
            "mu_0" : p_0[0],
            "sigma_0" : p_0[1],
            "bins" : KE_ff_range_bins,
            "norm" : norm,
            "p_inst_true" : output_mc["p_inst_true"] / args.beam_momentum,
            "KE_init_range" : {"mc" : output_mc["KE_init_range"], "data" : output_data["KE_init_range"]},
            }
        result = minimize(Chi2KE, [p_0[0], p_0[1]], args = [fit_args], method = "simplex")
        print(result.minuit)

        p_fit = result.minuit.values
        p_fit_err = result.minuit.errors


        weights_stopping_muon = Weights(fit_args["p_inst_true"], result.x[0], result.x[1], p_0[0], p_0[1])

        hist_data = np.histogram(np.clip(fit_args["KE_init_range"]["data"], min(fit_args["bins"]), max(fit_args["bins"])), fit_args["bins"])[0]
        hist_mc = np.histogram(np.clip(fit_args["KE_init_range"]["mc"], min(fit_args["bins"]), max(fit_args["bins"])), fit_args["bins"])[0]
        hist_mc_weighted = np.histogram(np.clip(fit_args["KE_init_range"]["mc"], min(fit_args["bins"]), max(fit_args["bins"])), fit_args["bins"], weights = weights_stopping_muon)[0]
        x = (KE_ff_range_bins[1:] + KE_ff_range_bins[:-1]) / 2
        Plots.Plot(x, hist_data / sum(hist_data), yerr = np.sqrt(hist_data) / sum(hist_data), style = "step", label = "data", color = "C6")
        Plots.Plot(x, hist_mc / sum(hist_mc), yerr = np.sqrt(hist_mc) / sum(hist_mc), style = "step", label = "mc", color = "C0", newFigure = False)
        Plots.Plot(x, hist_mc_weighted / sum(hist_mc_weighted), yerr = np.sqrt(hist_mc_weighted) / sum(hist_mc_weighted), style = "step", label = "mc reweighted", color = "k", xlabel = "$KE^{reco}_{int}$ (MeV)", title = "stopping muons", newFigure = False)
        book.Save()
        Plots.PlotHist(weights_stopping_muon, xlabel = "weights")
        book.Save()

        p_range = [0.8 * args.beam_momentum, 1.2 * args.beam_momentum]
        smearing_mu_rw, smearing_sigma_rw, _, _  = Smearing(output_mc["p_inst_reco"], output_data["p_inst_reco"], 20, p_range, book, weights_stopping_muon)
        smearing_mu, smearing_sigma, _, _  = Smearing(output_mc["p_inst_reco"], output_data["p_inst_reco"], 20, p_range, book)

    p_0 = {f"p{i}" : v for i, v in enumerate(p_0)}
    p_0_err = {f"p{i}" : v for i, v in enumerate(p_0_err)}

    p_fit = {f"p{i}" : v for i, v in enumerate(p_fit)}
    p_fit_err = {f"p{i}" : v for i, v in enumerate(p_fit_err)}

    smearing_params = {"mu" : {"value" : smearing_mu[0], "error" : smearing_mu[1]}, "sigma" : {"value" : smearing_sigma[0], "error" : smearing_sigma[1]}}
    smearing_params_rw = {"mu" : {"value" : smearing_mu_rw[0], "error" : smearing_mu_rw[1]}, "sigma" : {"value" : smearing_sigma_rw[0], "error" : smearing_sigma_rw[1]}}

    fit_values = {
        "p_0" : {"values" : p_0, "error" : p_0_err},
        "p_fit" : {"values" : p_fit, "error" : p_fit_err},
        "smearing" : smearing_params,
        "smearing_rw" : smearing_params_rw
        }

    for k, v in fit_values.items():
        cross_section.SaveConfiguration(v, args.out + k + ".json")

    # smearing_stopping_muon = np.random.normal(smearing_params["mu"]["value"], smearing_params["sigma"]["value"], len(output_mc["p_inst_true"]))
    # smearing_stopping_muon_rw = np.random.normal(smearing_params_rw["mu"]["value"], smearing_params_rw["sigma"]["value"], len(output_mc["p_inst_true"]))

    # weights_stopping_muon = Weights(vector.magnitude(pion_sample["mc"].trueParticles.momentum[:, 0]), result.x[0], result.x[1], true_params[1], true_params[2])
    # reco_KE_unsmeared_sm = KEs(samples_stopping_muon_selected, args_stopping_muon.upstream_loss_correction_params["value"])
    # reco_KE_smeared_sm = KEs(samples_stopping_muon_selected, args_stopping_muon.upstream_loss_correction_params["value"], smearing_stopping_muon)
    # reco_KE_smeared_sm_rw = KEs(samples_stopping_muon_selected, args_stopping_muon.upstream_loss_correction_params["value"], smearing_stopping_muon_rw)

    return

if __name__ == "__main__":
    args = cross_section.argparse.ArgumentParser("Calculates reweighting parameters for beam momentum.")
    cross_section.ApplicationArguments.Config(args, "True")
    cross_section.ApplicationArguments.Processing(args)
    cross_section.ApplicationArguments.Output(args)
    args.add_argument("--cut", dest = "cut", type = float, default = 0.9, help = "cut value for fractional range", required = True)

    args = cross_section.ApplicationArguments.ResolveArgs(args.parse_args())
    args.out = args.out + "beam_reweight/"
    print(vars(args))
    main(args)