"""
Created on: 30/10/2026 16:07

Author: Shyam Bhuller

Description: Functions and classes related to the photon energy correction method for final state showers.
"""
import awkward as ak
import numpy as np
import pandas as pd

from python.analysis import fitting, plotting
from python.analysis.plotting import plots

def apply_linear_correction(x, p0):
    """Scale reconstructed shower energy by dividing it by a fitted correction factor.

        Args:
            x (Any): Reconstructed energy values.
            p0 (Any): Fitted linear correction factor.

        Returns:
            Any: Energy values divided by ``p0``.
    """
    return x / p0

class ResponseFit(fitting.FitFunction):
    """Fit the logarithmic response model ``p0 * log(x - p1) + p2``.
    """
    n_params = 3

    @staticmethod
    def func(x, p0, p1, p2):
        """Evaluate the logarithmic response model at ``x``.

            Args:
                x (Any): Reconstructed energy values, shifted by ``p1`` inside the logarithm.
                p0 (Any): Logarithm scale.
                p1 (Any): Horizontal offset.
                p2 (Any): Additive offset.

            Returns:
                Any: ``p0 * log(x - p1) + p2``.
        """
        return p0 * np.log(x - p1) + p2

    @staticmethod
    def p0(x, y):
        """Return no initial parameter estimate; this fit supplies no custom starting values.

            Args:
                x (Any): Independent data values; unused.
                y (Any): Dependent data values; unused.
        """
        return None

def response_correction(x, p0, p1, p2):
    """Correct reconstructed energy using the fitted logarithmic response plus unity.

        Args:
            x (Any): Reconstructed energy values.
            p0 (Any): Logarithm scale parameter.
            p1 (Any): Horizontal offset parameter.
            p2 (Any): Additive offset parameter.

        Returns:
            Any: ``x / (ResponseFit.func(x, p0, p1, p2) + 1)``.
    """
    return x / (ResponseFit.func(x, p0, p1, p2) + 1)


shower_energy_correction = {
    "linear" : apply_linear_correction,
    "response": response_correction,
    None : None
}

#! replace with selection tools equivalent
def create_pfo_masks(masks : dict[ak.Array]) -> ak.Array:
    """ Combine a dicitonary of masks into a single mask.

        Args:
            masks (dict[ak.Array]): Masks.

        Returns:
            ak.Array: Combined mask.
    """
    mask = None
    for m in masks:
        if mask is None:
            mask = masks[m]
        else:
            mask = mask & masks[m]
    return mask


def photon_selection(df : pd.DataFrame, book : plotting.PlotBook = plotting.PlotBook.null()):
    """Plot shower-energy fractional errors and return rows whose true mother is a neutral pion.

        Args:
            df (pd.DataFrame): Photon rows with truth-mother, selection, and energy columns.
            book (plotting.PlotBook): Plot book that receives the diagnostic figures.

        Returns:
            Any: The subset of ``df`` with ``true_mother == 111``.
    """
    pi0_mother = df.true_mother == 111
    counts = {"pi0_daughter" : len(pi0_mother[df.pi0_photon]), "other" : len(df.pi0_photon) - len(df.pi0_photon[df.pi0_photon])}
    print(counts)
    print({c : counts[c] / len(df) for c in counts})

    plots.plot_hist([df[pi0_mother].fractional_error, df[~pi0_mother].fractional_error], stacked = False, histtype = "step", range = [-1, 1], xlabel = "$E^{reco}$ fractional error", label = ["$\\pi^{0}$ daughter", "other"])
    book.save()

    df = df[pi0_mother]
    plots.plot_hist_2d_imshow_marginal(df.reco_shower_energy, df.fractional_error, bins = 100, xlabel = "$E^{reco}$ (MeV)", ylabel = "$E^{reco}$ fractional error", cmap = "plasma", x_range = [0, max(df.reco_shower_energy)], y_range = [-1, 1], norm = False)
    book.save()
    return df


def line(x, m):
    """Evaluate a line through the origin with slope ``m``.

        Args:
            x (Any): Independent values.
            m (Any): Line slope.

        Returns:
            Any: ``m * x``.
    """
    return m * x


def binned_dataframe(df : pd.DataFrame, bins : list, energy_range : list) -> list:
    """ Split a dataframe into a list of dataframes based on reco energy bins.

        Args:
            df (pd.DataFrame): Event data containing reconstructed shower energy and true energy.
            bins (list): Bin edges.
            energy_range (list): Range whose upper bound limits the true energy retained in each reconstructed-energy bin.

        Returns:
            list: List of binned data frames.
    """
    binned_data = []
    for i in range(1, len(bins)):
        data = df[(df.reco_shower_energy < bins[i]) & (df.reco_shower_energy > bins[i-1])]
        binned_data.append(data[data.true_energy < max(energy_range)])
    return binned_data


def linear_fit(df : pd.DataFrame, bins : np.ndarray, energy_range : list, book : plotting.PlotBook = plotting.PlotBook.null()) -> float:
    """ Perform linear fit of true energy vs reco energy, equivalent to the correction done in the microboone simulation paper.
                     #TODO implment using fitting module instead.

        Args:
            df (pd.DataFrame): Event data containing reconstructed shower energy and true energy.
            bins (np.ndarray): Reco energy bins.
            energy_range (list): Range whose upper bound limits true energy when calculating the binned mean response.
            book (plotting.PlotBook): Plot book used to save the fit and comparison figures.

        Returns:
            float: Gradient of linear fit (correction).
    """
    x = (bins[1:] + bins[:-1]) / 2
    y = np.array([d.true_energy.mean() for d in binned_dataframe(df, bins, energy_range)])

    if any(np.isnan(np.concatenate([x, y]))):
        raise Exception(f"binned x and y values contain NaNs. Consider changing the binning range. \n {x, y=}")

    popt, pcov = fitting.curve_fit(line, x, y)

    print(popt, pcov**0.5)
    perr = np.array([pcov[i][i] for i in range(len(popt))])**0.5

    plots.plot(x, y, marker = "x", linestyle = "")
    plots.plot(x, line(x, *popt), new_figure = False, label = "fit")
    plots.plt.fill_between(x, line(x, *(popt + perr)), line(x, *(popt - perr)), color = "C3", alpha = 0.5)
    plots.plot(x, x, new_figure = False, label = "$y = x$", xlabel = "$E^{reco}$ (MeV)", ylabel = "$E^{true}$ (MeV)")
    plots.plt.legend()


    plots.plot_hist_2d(df.true_energy, df.reco_shower_energy, x_range = [0, 1500], y_range = [0, 1500], cmap = "summer")
    plots.plot(x, line(x, *popt), new_figure = False, label = "fit", color = "C0")
    plots.plot(x, x, new_figure = False, color = "black", label = "$y = x$")
    plots.plot(x, y, marker = "x", linestyle = "", ylabel = "$E^{true}$ (MeV)", xlabel = "$E^{reco}$ (MeV)", new_figure = False, color = "C0")
    book.save()
    return popt[0]


def linear_fit_performance(df : pd.DataFrame, linear_correction : float, book : plotting.PlotBook = plotting.PlotBook.null()):
    """ Performance plots and metrics of linear correction.

        Args:
            df (pd.DataFrame): Events with reconstructed and true shower energies.
            linear_correction (float): Factor used to divide reconstructed energy.
            book (plotting.PlotBook): Plot book that receives the comparison figures.
    """
    corrected_energy = apply_linear_correction(df.reco_shower_energy, linear_correction)

    fe = (df.reco_shower_energy / df.true_energy) - 1
    fec = (corrected_energy / df.true_energy) - 1

    print(f"correction factor : {linear_correction}")
    print(f"mean shower energy fractional error: {np.mean(fe)} +- {np.std(fe)}")
    print(f"mean shower energy fractional error after correction: {np.mean(np.mean(fec))} +- {np.std(fec)}")

    plots.plot_hist_comparison([df.reco_shower_energy, corrected_energy], labels = ["uncorrected", "corrected"], x_range = [0, 2000], xlabel = "$E^{reco}$ (MeV)")
    plots.plot_hist_comparison([df.reco_shower_energy - df.true_energy, corrected_energy - df.true_energy], labels = ["uncorrected", "corrected"], x_range=[-500, 500], xlabel = "$E^{reco}$ residual (MeV)")
    book.save()

    plots.plot_hist_2d_comparison([df.true_energy, df.true_energy], [fe, fec], [0, 2000], [-1, 1], bins = 50, cmap = "Accent", xlabels = ["$E^{true}$ (MeV)"]*2, ylabels = ["Fractional error"]*2, titles = ["uncorrected", "corrected"])
    book.save()
    return


def iter_binned_df(df : pd.DataFrame, variable : str, v_range: list, reco_bins : list) -> pd.DataFrame:
    """Yield the selected variable values separately for each reconstructed-energy bin.

        Args:
            df (pd.DataFrame): Event table containing the requested variable and reconstructed energy.
            variable (str): Column to select from each bin.
            v_range (list): Inclusive range limits used to filter the variable values (bounds are strict).
            reco_bins (list): Reconstructed-energy bin edges (bin bounds are strict).

        Returns:
            pd.DataFrame: An iterator yielding one filtered Series per reconstructed-energy bin.
    """
    for i in range(1, len(reco_bins)):
        binned_data = df[(df.reco_shower_energy > reco_bins[i-1]) & (df.reco_shower_energy < reco_bins[i])]
        ranged_data = binned_data[(binned_data[variable] > min(v_range)) & (binned_data[variable] < max(v_range))]
        yield ranged_data[variable]


def calculate_mean(df : pd.DataFrame, variable : str, v_range: list, reco_bins : list) -> np.ndarray:
    """Calculate the mean of a variable in each reconstructed-energy bin.

        Args:
            df (pd.DataFrame): Event table containing the variable and reconstructed energy.
            variable (str): Column whose mean is calculated.
            v_range (list): Strict lower and upper limits for values included in each bin.
            reco_bins (list): Reconstructed-energy bin edges.

        Returns:
            np.ndarray: Mean value for each reconstructed-energy bin.
    """
    means = []
    for column in iter_binned_df(df, variable, v_range, reco_bins):
        means.append(column.mean())
    return np.array(means)


def calculate_sem(df : pd.DataFrame, variable : str, v_range: list, reco_bins : list) -> np.ndarray:
    """Calculate the standard error of the mean of a variable in each reconstructed-energy bin.

        Args:
            df (pd.DataFrame): Event table containing the variable and reconstructed energy.
            variable (str): Column whose standard error is calculated.
            v_range (list): Strict lower and upper limits for values included in each bin.
            reco_bins (list): Reconstructed-energy bin edges.

        Returns:
            np.ndarray: Per-bin standard deviation divided by the square root of the bin count.
    """
    sem = []
    for column in iter_binned_df(df, variable, v_range, reco_bins):
        sem.append(column.std() / np.sqrt(len(column)))
    return np.array(sem)


def create_bins_df(value : pd.Series, n_entries, v_range : list = None) -> np.ndarray:
    """Choose sorted-value edges at fixed entry intervals, optionally forcing the outer limits.

        Args:
            value (pd.Series): Values used to derive bin edges.
            n_entries (Any): Number of sorted entries between successive selected edges.
            v_range (list): Optional range whose minimum and maximum replace the first and last edges.

        Returns:
            np.ndarray: Selected bin-edge values.
    """
    sorted_value = value.sort_values()
    n_bins = len(sorted_value) // n_entries

    bins = []
    for i in range(n_bins + 1):
        mi = sorted_value.values[i * n_entries]
        bins.append(mi)
    if v_range:
        bins[0] = min(v_range)
        bins[-1] = max(v_range)
    return np.array(bins)


def calculate_central_values(df : pd.date_range, bins : np.ndarray, book : plotting.PlotBook) -> dict:
    """Extract binned fractional-error central values with Student-t, Gaussian, and mean estimates.

        Args:
            df (pd.date_range): Event data containing reconstructed energy and fractional error.
            bins (np.ndarray): Reconstructed-energy bin edges.
            book (plotting.PlotBook): Plot book used to save fit and summary figures.

        Returns:
            dict: Central-value estimates and uncertainties keyed by fit method, plus binned mean results.
    """
    central_values = {}

    with plotting.styles.PlotStyler().update(font_scale = 1.3):
        for k, v in {"student_t" : fitting.student_t, "gaussian" : fitting.gaussian}.items():
            central_values[k] = fitting.extract_central_values_df(df, "reco_shower_energy", "fractional_error", [-1, 1], [v], bins, 20, bin_label = "$E^{reco}$", bin_units = "(MeV)")
            book.save()
        # central_values["gaussian"] = fitting.extract_central_values_df(df, "reco_shower_energy", "fractional_error", [-1, 1], [fitting.gaussian], bins, 20)
        # book.save()
        central_values["mean"] = [calculate_mean(df, "fractional_error", [-1, 1], bins), calculate_sem(df, "fractional_error", [-1, 1], bins)]
        book.save()
    return central_values


def response_fits(central_values : dict, bins : np.ndarray, book : plotting.PlotBook) -> dict:
    """Fit ``ResponseFit`` to each supplied set of binned central values and save fit plots.

        Args:
            central_values (dict): Mapping of estimator names to per-bin values and uncertainties.
            bins (np.ndarray): Reconstructed-energy bin edges used to form fit x-values.
            book (plotting.PlotBook): Plot book used to save fit figures.

        Returns:
            dict: Fit parameter values and errors for each estimator.
    """
    x = (bins[1:] + bins[:-1]) / 2
    response_params = {}

    for _, cv in plots.iter_multi_plot(central_values):
        print(cv)
        print(central_values[cv][1])
        popt, perr = fitting.fit(x, central_values[cv][0], central_values[cv][1], ResponseFit, method = "lm", plot = True, xlabel = "$E^{reco}$ (MeV)", ylabel = "Fractional error", maxfev = int(1E6))
        plots.plt.ylim(-1, 1)
        plots.plt.title(cv)
        response_params[cv] = {"value" : popt, "error" : perr}

    print(response_params)
    book.save()


    for cv in central_values:
        plots.plt.figure()
        popt, _ = fitting.fit(x, central_values[cv][0], central_values[cv][1], ResponseFit, method = "lm", plot = True, xlabel = "$E^{reco}$ (MeV)", ylabel = "Fractional error", maxfev = int(1E6))
        plots.plt.ylim(-1, 1)
        book.save()
    return response_params


def method_comparison(df : pd.DataFrame, linear_correction : float, response_params : dict, bins : np.ndarray, energy_range : list, book : plotting.PlotBook):
    """Compare uncorrected and Gaussian-response shower energies using diagnostic plots and summaries.

        Args:
            df (pd.DataFrame): Events with reconstructed and true shower energies.
            linear_correction (float): Factor used to compute the linear-corrected energy; that series is omitted from the final comparisons.
            response_params (dict): Response-fit parameters; the plotted correction currently uses the ``gaussian`` entry.
            bins (np.ndarray): Reconstructed-energy bin edges for fractional-error summaries.
            energy_range (list): Energy range used in the comparison plots.
            book (plotting.PlotBook): Plot book used to save comparison figures.
    """
    energies = {"uncorrected" : df.reco_shower_energy, "linear" : df.reco_shower_energy / linear_correction}

    for p in response_params:
        energies[p] = response_correction(df.reco_shower_energy, *response_params[p]["value"])

    energies = {"uncorrected" : energies["uncorrected"], "gaussian" : energies["gaussian"]}

    plots.plot_hist_comparison(list(energies.values()), labels = list(energies.keys()), x_range = energy_range, xlabel = "$E^{reco}$ (MeV)")

    fe = {i : (energies[i]/df.true_energy) - 1 for i in energies}

    plots.plot_hist_2d_comparison([df.true_energy]*len(fe), list(fe.values()), energy_range, [-1, 1], bins = 50, cmap = "Accent", xlabels = ["$E^{true}$ (MeV)"]*len(fe), ylabels = ["Fractional error"]*len(fe), titles = list(fe.keys()))
    book.save()

    plots.plot_hist_2d_comparison([df.reco_shower_energy]*len(fe), list(fe.values()), energy_range, [-1, 1], bins = 50, cmap = "Accent", xlabels = ["Uncorrected reco shower energy (MeV)"]*len(fe), ylabels = ["Fractional error"]*len(fe), titles = list(fe.keys()))
    book.save()

    for _, (l, e) in plots.iter_multi_plot(energies.items()):
        plots.plot_hist_2d(e, df.true_energy, bins = 100, title = l, xlabel = "$E^{reco}$ (MeV)", ylabel = "$E^{true}$ (MeV)", x_range = energy_range, y_range = energy_range, cmap = "summer", new_figure = False)
        plots.plt.plot(energy_range, energy_range)
    book.save()

    tab = {}
    for l, f in fe.items():
        v = f[(f > -1) & (f < 1)]
        tab[l] = {"$\\mu$" : v.mean(), "$\\sigma$" : v.std()}
    tab = pd.DataFrame(tab)

    for i, f in plots.iter_multi_plot(fe):
        binned_data = [fe[f][(df.reco_shower_energy > bins[i + 0]) & (df.reco_shower_energy < bins[i + 1])] for i in range(len(bins)-1)]

        mean = [d.mean() for d in binned_data]
        sem = [d.std()/(len(d)**0.5) for d in binned_data]
        plots.plot((bins[1:] + bins[:-1]) / 2, mean, yerr = sem, marker = "x", capsize = 3, new_figure = False)
        plots.plot_hist_2d(df.reco_shower_energy, fe[f], bins = [bins, 50], x_range = energy_range, y_range = [-1, 1], new_figure = False, xlabel = "Uncorrected reco energy (MeV)", ylabel = "Fractional error", title = f)
    book.save()

    plots.plt.figure()
    for i, f in enumerate(fe):
        binned_data = [fe[f][(df.reco_shower_energy > bins[i + 0]) & (df.reco_shower_energy < bins[i + 1])] for i in range(len(bins)-1)]
        mean = [d.mean() for d in binned_data]
        sem = [d.std()/(len(d)**0.5) for d in binned_data]
        plots.plot((bins[1:] + bins[:-1]) / 2, mean, yerr = sem, marker = "x", capsize = 3, label = f, new_figure = False)
        plots.plt.xlabel("Uncorrected reco energy (MeV)")
        plots.plt.ylabel("Fractional error")
        plots.plt.legend()
    book.save()
    return tab
