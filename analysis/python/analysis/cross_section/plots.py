import awkward as ak
import numpy as np

from scipy import stats

from . import geant

from python.analysis import common, slices
from python.analysis.plotting import plots

def plot_cross_section(xs : dict[tuple], energy_slices : slices.Slices, cross_section_name : str, xs_sim_range : tuple[float] = None):
    """ Plot a set of cross section measurements. Assumed from the same process.
        # TODO Include chi squared option.
        # TODO allow more than one process to be plot for same figure plots? (name match the process to xs key for chi2 comparison)
    Args:
        xs (dict[tuple]): cross section measurements. Struture is tuple of arrays, weere index 0 is the value, and 1 is the error.
        energy_slices (Slices): Energy slices.
        cross_section_name (str): process name to plot from GEANT4 simulation.
        xs_sim_range (tuple[float], optional): Energy range to plot for the simulation cross section (MeV). Defaults to None (full range).
    """
    geant_xs = geant.GeantCrossSections(energy_range=xs_sim_range)

    x = energy_slices.pos - energy_slices.width / 2

    plots.plt.figure()
    geant_xs.plot(cross_section_name, simplified_pion_production=True)

    for label, (y, yerr) in xs.items():
        plots.plot(
            x,
            y,
            xerr=energy_slices.width / 2,
            yerr=yerr,
            new_figure=False,
            marker="o",
            label=label,
        )
    plots.plt.legend()
    return


def plot_cross_section_steps(xs_tensor : tuple[np.ndarray], process : str, xs_prefit : tuple[np.ndarray]):
    """ Plot the cross section for each MCMC step in histograms, one for each energy bin. Compares to prefit value.

    Args:
        xs_tensor (np.ndarray): Cross section measurements at each MCMC step.
        process (str): Process name.
        xs_prefit (np.ndarray): Prefit cross section.
    """
    for i, xs in plots.iter_multi_plot(xs_tensor[0]):
        plots.plt.axvline(xs_prefit[0][i] ,color="k", linestyle="--", label = "Prefit")
        plots.plt.axvline(np.mean(xs), color="C1", linestyle="--", label = "Mean value")
        plots.plot_hist(xs, color="C0", label="MCMC steps", bins=50, new_figure=False)
        plots.plt.xlabel(f"{common.remove_(process.capitalize())} cross section (mb)")
    return


def plot_cross_section_minmax(xs_tensor : tuple[np.ndarray], xs_prefit : tuple[np.ndarray], process : str, energy_slices : slices.Slices):
    """ Plot the Minimum and Maximum cross sections at each MCMC step. Note that the absolute max and min are taken, not from just one step.

    Args:
        xs_tensor (tuple[np.ndarray]): Cross section measurements at each MCMC step.
        xs_prefit (tuple[np.ndarray]): Prefit cross section.
        process (str): Process name.
        energy_slices (Slices): Energy slices.
    """
    xs_central, xs_err = xs_tensor

    max_xs = ak.max(xs_central, 1)
    max_xs_err = ak.ravel(
        ak.Array(xs_err[ak.local_index(xs_err, axis=1) == ak.argmax(xs_central, 1, keepdims=True)])
    )
    min_xs = ak.min(xs_central, 1)
    min_xs_err = ak.ravel(
        ak.Array(xs_err[ak.local_index(xs_err, axis=1) == ak.argmin(xs_central, 1, keepdims=True)])
    )

    series = {
        "prefit" : xs_prefit,
        "minimum" : (min_xs, min_xs_err),
        "maximum" : (max_xs, max_xs_err),
    }
    plot_cross_section(series, energy_slices, process, [1000, 2000])
    return


def plot_cross_section_cl(xs_tensor : tuple[np.ndarray], xs_prefit : tuple[np.ndarray], process : str, energy_slices : slices.Slices, confidence_level : float = 0.68):
    """ Plot the mean cross section and the error by calculating the Confidence level. Compares to simulation and prefit.

    Args:
        xs_tensor (tuple[np.ndarray]): Cross section measurements at each MCMC step.
        xs_prefit (tuple[np.ndarray]): Prefit cross section.
        process (str): Process name.
        energy_slices (Slices): Energy slices.
        confidence_level (float, optional): Confidence level to evaluate the errors at. Defaults to 0.68 (1 sigma).
    """
    xs_central, xs_err = xs_tensor # neglect the stat error from propagation.

    mean_xs = np.mean(xs_central, axis=1)
    ci = stats.t.interval(confidence_level, df=len(xs_central)-1, loc=mean_xs, scale=np.std(xs_central, axis=1, ddof=1) / np.sqrt(len(xs_central)))

    series = {
        "prefit" : xs_prefit,
        "postfit" : (mean_xs, abs(mean_xs - np.abs(ci))),
    }
    plot_cross_section(series, energy_slices, process, [1000, 2000])
    return


def plot_xs_hists(energy_slices, hist_counts : np.ndarray, hist_counts_err : np.ndarray = None, overflow : bool = True, scale : float = 1, xlabel : str = "$KE$ (MeV)", ylabel : str = "Counts", label : str = None, color : str = None, new_figure : bool = True, title : str = None):
    common.deprecation_warning()
    if hist_counts_err is None:
        hist_counts_err = np.sqrt(hist_counts)

    if overflow is False:
        s = slice(1, -1)
    else:
        s = slice(0, len(energy_slices.pos_overflow))
    x = energy_slices.pos_overflow - energy_slices.width/2
    x = x[s]

    plots.plot(x, scale * hist_counts[s], yerr = scale * hist_counts_err[s], xlabel = xlabel, new_figure = new_figure, style = "step", label = label, color = color, ylabel = ylabel, title = title)
    return


def hyp_test_xs(cv, error, process, energy_slice, file = geant.GEANT_XS):
    common.deprecation_warning()
    xs_sim = geant.GeantCrossSections(file, energy_range = [energy_slice.min_pos - energy_slice.width, energy_slice.max_pos])
    sim_curve_interp = xs_sim.get_interpolated_curve(process)
    x = energy_slice.pos[:-1] - energy_slice.width/2

    w_chi_sqr = common.weighted_chi_sqr(cv, sim_curve_interp(x), error)

    p = stats.chi2.sf((len(x)-1) * w_chi_sqr, len(x) - 1)
    return {"w_chi2" : w_chi_sqr, "p" : p}

def plot_xs_cpmparison(xs : dict[np.ndarray], energy_slice, process : str = None, colors : dict[str] = None, xs_sim_color : str = "k", title : str = None, simulation_label : str = "simulation", chi2 : bool = True, new_figure : bool = True, cv_only : bool = False, marker_size : float = 6):
    common.deprecation_warning()
    if hasattr(energy_slice.width, "__iter__"):
        width = energy_slice.width[:-1][::-1]
        xs_sim = geant.GeantCrossSections(energy_range = [energy_slice.min_pos - energy_slice.width[0], energy_slice.max_pos + energy_slice.width[-1]])
    else:
        width = energy_slice.width
        xs_sim = geant.GeantCrossSections(energy_range = [energy_slice.min_pos - energy_slice.width, energy_slice.max_pos + energy_slice.width])

    if colors is None:
        colors = {k : f"C{i}" for i, k in enumerate(xs)}

    sim_curve_interp = xs_sim.get_interpolated_curve(process)
    x = energy_slice.pos[:-1] - width/2

    if new_figure is True: plots.plt.figure()
    chi_sqrs = {}
    for k, v in xs.items():
        w_chi_sqr = common.weighted_chi_sqr(v[0], sim_curve_interp(x), v[1])
        chi_sqrs[k] = w_chi_sqr
        if (chi2 is True) and (cv_only is False):
            chi2_l = ", $\\chi^{2}/ndf$ = " + f"{w_chi_sqr:.3g}"
        else:
            chi2_l = ""
        plots.plot(x, v[0], xerr = width / 2  if cv_only is False else None, yerr = v[1] if cv_only is False else None, label = k + chi2_l, color = colors[k], linestyle = "", marker = "x", new_figure = False, markersize = marker_size, capsize = marker_size/2)
    
    if process == "single_pion_production":
        plots.plot(xs_sim.KE, sim_curve_interp(xs_sim.KE), label = simulation_label, title = "Single pion production" if title is None else title.capitalize(), new_figure = False, xlabel = "$KE (MeV)$", ylabel = "$\\sigma$ (mb)", color = xs_sim_color)
    else:
        xs_sim.plot(process, label = simulation_label, color = xs_sim_color, title = title.capitalize() if type(title) is str else title)

    plots.plt.ylim(0)
    if max(plots.plt.gca().get_ylim()) > np.nanmax(sim_curve_interp(xs_sim.KE).astype(float)) * 2:
        plots.plt.ylim(0, max(sim_curve_interp(xs_sim.KE)) * 2)
    if hasattr(width, "__iter__"):
        plots.plt.xlim(energy_slice.min_pos - (0.2 * width[0]), energy_slice.max_pos + (0.2 * width[-1]))
    else:
        plots.plt.xlim(energy_slice.min_pos - (0.2 * width), energy_slice.max_pos + (0.2 * width))
    return chi_sqrs
