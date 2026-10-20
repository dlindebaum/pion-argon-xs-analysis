"""Beam reweighting utilities."""

def RatioWeights(beam_inst_P : np.ndarray, func : str, params : list, truncate : int = 10):
    weights = 1/getattr(Fitting, func)(beam_inst_P, *params)
    weights = np.where(weights > truncate, truncate, weights)
    return weights


def ReWeight(p_MC, p_Data, p_nominal : float, bins : int = 10, p_range : np.array = np.array([0.75, 1.25]), book : Plots.PlotBook = Plots.PlotBook.null()):
    p_mc, edges = np.histogram(np.array(p_MC), bins, range = p_nominal * p_range)
    p_data = np.histogram(np.array(p_Data), bins, range = p_nominal * p_range)[0]

    with Plots.RatioPlot((edges[1:] + edges[:-1]) / 2, p_data, p_mc, np.sqrt(p_data), np.sqrt(p_mc), "$P_{inst}^{reco}$ (MeV)", "Data/MC") as ratio_plot:
        Plots.Plot(ratio_plot.x, ratio_plot.y1, yerr = ratio_plot.y1_err, newFigure = False)
        Plots.Plot(ratio_plot.x, ratio_plot.y2, yerr = ratio_plot.y2_err, newFigure = False, ylabel = "Counts")
    book.Save()

    scale = sum(ratio_plot.y1) / sum(ratio_plot.y2)

    ratio = scale * np.nan_to_num(cross_section.nandiv(ratio_plot.y2, ratio_plot.y1), posinf = 0)
    ratio_err = np.nan_to_num(abs(ratio * np.sqrt(cross_section.nandiv(ratio_plot.y1_err, ratio_plot.y1)**2 + cross_section.nandiv(ratio_plot.y2_err, ratio_plot.y2)**2)))

    Plots.Plot(ratio_plot.x, ratio, yerr = ratio_err, xlabel = "$P_{inst}^{reco}$ (MeV)", ylabel = "$r$")
    book.Save()

    results = {}
    for f in [cross_section.Fitting.gaussian, cross_section.Fitting.student_t, cross_section.Fitting.poly2d, cross_section.Fitting.crystal_ball, cross_section.Fitting.double_crystal_ball, cross_section.Fitting.double_gaussian]:
        Plots.plt.figure()
        results[f.__name__] = cross_section.Fitting.Fit(ratio_plot.x[ratio > 0], ratio[ratio > 0], ratio_err[ratio > 0], f, plot = True, xlabel = "$P_{inst}^{reco}$(MeV)", ylabel = "$r$")
        book.Save()
    return results


def ReWeightResults(sideband_mc : dict, sideband_data : dict, args : cross_section.argparse.Namespace, bins : int, reweight_results : dict, reweight_func : str, book : Plots.PlotBook = Plots.PlotBook.null()):
    weights = cross_section.RatioWeights(np.array(sideband_mc["p_inst"]), reweight_func, reweight_results[reweight_func][0], args.beam_reweight["strength"])

    plot_range = [args.beam_momentum * 0.75, args.beam_momentum * 1.25]

    Plots.PlotHist(weights, range = [0, 3], xlabel = "weights", truncate = True)
    book.Save()

    Plots.PlotTagged(sideband_mc["p_inst"], sideband_mc["tags"], data2 = sideband_data["p_inst"], x_range = plot_range, norm = args.norm, data_weights = None, bins = bins, x_label = "$P_{inst}^{reco}$ (MeV)", ncols = 1)
    Plots.plt.title("nominal", pad = 15)
    book.Save()

    Plots.PlotTagged(sideband_mc["p_inst"], sideband_mc["tags"], data2 = sideband_data["p_inst"], x_range = plot_range, norm = args.norm, data_weights = weights, bins = bins, x_label = "$P_{inst}^{reco}$ (MeV)", ncols = 1)
    Plots.plt.title(f"reweighted : {reweight_func}", pad = 15)
    book.Save()
    return


def SmearingFactors(sample, weights : np.array = None):
    average = np.average(sample.recoParticles.beam_inst_P, weights=weights)
    variance = np.average((sample.recoParticles.beam_inst_P-average)**2, weights=weights)
    std = np.sqrt(variance)
    return average, std
