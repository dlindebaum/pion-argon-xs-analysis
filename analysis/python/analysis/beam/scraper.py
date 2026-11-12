"""Beam scraper utilities."""
import awkward as ak
import numpy as np

from particle import Particle

from python.analysis import bethe_bloch, common, fitting, kinematics, selection, ntuple
from python.analysis.plotting import plots


def is_scraper(mc : ntuple.Data, beam_scraper_args : dict) -> ak.Array:
    """Flag reconstructed events with unusually large upstream energy loss in configured momentum bins.

        Args:
            mc (ntuple.Data): Ntuple with reconstructed beam momentum and true front-face kinetic energy.
            beam_scraper_args (dict): Per-bin limits containing ``bins``, ``mu_e_res``, and ``sigma_e_res``.

        Returns:
            ak.Array: OR-combined mask for events above each bin's ``mu_e_res + 3 * abs(sigma_e_res)`` threshold.
    """
    beam_inst_KE = kinematics.KE(mc.recoParticles.beam_inst_momentum, Particle.from_pdgid(211).mass) # get kinetic energy from beam instrumentation
    true_ffKE = mc.true_particles.beam_KE_front_face

    delta_KE_upstream = beam_inst_KE - true_ffKE

    scraper_ids = {}
    for k, v in beam_scraper_args.items():
        scraper_ids[k] = (beam_inst_KE > min(v["bins"])) & (beam_inst_KE < max(v["bins"]))
        threshold = v["mu_e_res"] + 3 * abs(v["sigma_e_res"])
        
        scraper_ids[k] = scraper_ids[k] & (delta_KE_upstream > threshold)
    scraper_ids = selection.combine_masks(scraper_ids, operator = "or")
    return scraper_ids


def get_scraper_fits(ke_bins : list, beam_inst_KE : ak.Array, delta_KE_upstream : ak.Array, fit_bins : int, residual_range : list) -> dict:
    """ Fit gaussians to front facing kinetic energies to extract values required to define events as beam scrapers.

        Args:
            ke_bins (list): Bins of Kinetic energy to split the sample in.
            beam_inst_KE (ak.Array): Beam instrumentation kinetic energy.
            delta_KE_upstream (ak.Array): Difference in the front facing KE and beam kinetic energy.
            fit_bins (int): Number of bins to histogram the data for the fit.
            residual_range (list): Plot range.

        Returns:
            dict: Fit parameters.
    """
    plot = plots.multi_plot(len(ke_bins) - 1)

    scraper_fit = {}
    for i in range(1, len(ke_bins)):
        bin_label = "$KE^{reco}_{inst}$:" + f"[{ke_bins[i-1]},{ke_bins[i]}] (MeV)"
        e = (beam_inst_KE < ke_bins[i]) & (beam_inst_KE > ke_bins[i-1])

        data = np.array(delta_KE_upstream[e][~np.isnan(delta_KE_upstream[e])].to_list(), dtype = float)
        # data = delta_KE_upstream[e]

        y, bin_edges = np.histogram(data, bins = fit_bins, range = sorted([np.nanpercentile(data, 10), np.nanpercentile(data, 90)]))
        yerr = np.sqrt(y) # Poisson error

        print(f"{(max(y), np.nanmedian(data), np.nanstd(data))=}")

        next(plot)
        popt, perr, metrics = fitting.fit(common.bin_centers(bin_edges), y, yerr, fitting.gaussian, method = "dogbox", return_chi_sqr = True)#, plot = True, plot_style = "scatter", xlabel = "$\\Delta E_{upstream}$ (MeV)", title = bin_label, plot_range = residual_range)
        heights, _ = plots.plot_hist(data, new_figure = False, bins = fit_bins, range = residual_range, label = "observed")
        x_interp = np.linspace(min(data), max(data), 10 * fit_bins)
        y_interp = fitting.gaussian.func(x_interp, max(heights), popt[1], popt[2])
        plots.plot(x_interp, y_interp, color = "black", label = "fit", title = bin_label, xlabel = "$\\Delta E_{upstream}$ (MeV)", new_figure = False)
        plots.plt.axvline(popt[1] + 3 * abs(popt[2]), color = "black", linestyle = "--", label = "$\\mu + 3\\sigma$")
        plots.plt.xlim(*residual_range)

        main_legend = plots.plt.legend(loc = "upper left")
        main_legend.set_zorder(12)

        #* add fit metrics to the plot in a second legend
        plots.plt.gca().add_artist(main_legend)
        text = ""
        for j in range(len(popt)):
            text += f"\\np{j}: ${popt[j]:.2g}\\pm${perr[j]:.2g}"
        text += "\\n$\\chi^{2}/ndf$ : " + f"{metrics[0]/metrics[1]:.2g}, p : " + f"{metrics[2]:.1g}"
        legend = plots.plt.gca().legend(handlelength = 0, labels = [text[1:]], loc = "upper right", title = fitting.gaussian.__name__)
        legend.set_zorder(12)
        for l in legend.legend_handles:
            l.set_visible(False)

        print(f"{popt=}")
        print(f"{perr=}")
        scraper_fit[(ke_bins[i-1], ke_bins[i])] = {"mu_e_res" : popt[1], "sigma_e_res" : abs(popt[2])}
    return scraper_fit


def beam_scraper_plots(beam_inst_KE_bins : list, output_mc : dict[ak.Array], scraper_fits : dict) -> dict:
    """ Scatter plots of events to show beam scraper events as a function of the XY position at the beam instrumentation.

        Args:
            beam_inst_KE_bins (list): Beam kinetic energy bins.
            output_mc (dict[ak.Array]): MC beam-instrument positions, kinetic energies, and upstream energy differences.
            scraper_fits (dict): Per-energy-bin fitted means and widths used for the scraper threshold.

        Returns:
            dict: Per-bin beam-instrument x/y means and standard deviations.
    """

    output = {}

    for i in plots.multi_plot(len(beam_inst_KE_bins)-1):
        if i == len(beam_inst_KE_bins): continue
        bin_edges = (beam_inst_KE_bins[i], beam_inst_KE_bins[i+1])
        bin_label = "$KE^{reco}_{inst}$:" + f"[{beam_inst_KE_bins[i]},{beam_inst_KE_bins[i+1]}] (MeV)"
        e = (output_mc["beam_inst_KE"] < beam_inst_KE_bins[i+1]) & (output_mc["beam_inst_KE"] > beam_inst_KE_bins[i])
        fit_values = scraper_fits[(beam_inst_KE_bins[i], beam_inst_KE_bins[i+1])]

        is_scraper = output_mc["delta_KE_upstream"][e] > (fit_values["mu_e_res"] + 3 * fit_values["sigma_e_res"])

        plots.plot(output_mc["beam_inst_x"][e][~is_scraper], output_mc["beam_inst_y"][e][~is_scraper], new_figure = False, linestyle = "", marker = "o", markersize = 2, color = "C0", alpha = 0.5, label = "non-scraper", rasterized = True)
        plots.plot(output_mc["beam_inst_x"][e][is_scraper], output_mc["beam_inst_y"][e][is_scraper], new_figure = False, linestyle = "", marker = "o", markersize = 2, color = "C6", alpha = 0.5, label = "scraper", rasterized = True)

        mu_x = ak.mean(output_mc["beam_inst_x"][e])
        mu_y = ak.mean(output_mc["beam_inst_y"])
        print(bin_label)
        print(mu_x, mu_y)
        sigma_x = ak.std(output_mc["beam_inst_x"][e])
        sigma_y = ak.std(output_mc["beam_inst_y"])

        output[bin_edges] = {"mu_x_inst" : mu_x, "mu_y_inst" : mu_y, "sigma_x_inst" : sigma_x, "sigma_y_inst" : sigma_y}

        theta = np.linspace(0, 2*np.pi, 100)
        for j, m in enumerate([0.5, 1, 1.5, 2, 3]):
            r = m * (sigma_x**2 + sigma_y**2)**0.5
            x = r*np.cos(theta) + mu_x
            y = r*np.sin(theta) + mu_y

            plots.plot(x, y, linestyle = "--", color = f"C{7+j}", alpha = 1, label = f"{m}$r_{{inst}}$", new_figure = False)
        plots.plt.xlabel("$X^{reco}_{inst}$ (cm)")
        plots.plt.ylabel("$Y^{reco}_{inst}$ (cm)")
        plots.plt.title(bin_label)
        plots.plt.axis('scaled')
        plots.plt.legend()
    return output
