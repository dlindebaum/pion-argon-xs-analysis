"""
Created on Fri Mar 26 12:16:29 2021

Author: Shyam Bhuller

Description: A script conatining boiler plate code for creating plots with matplotlib.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import gamma


def Save(name : str = "plot", directory : str = "", dpi = 100):
    """ Saves the last created plot to file. Run after one the functions below.

    Args:
        name (str, optional): Name of plot. Defaults to "plot".
        directory (str, optional): directory to save plot in.
    """
    plt.savefig(directory + name + ".png", dpi = dpi)
    plt.close()


def Plot(x, y, xlabel : str = "", ylabel : str = "", title : str = "", label : str = "", marker : str = "", linestyle : str = "-", newFigure : bool = True, annotation : str = None):
    """ Make scatter plot.
    """
    if newFigure is True: plt.figure()
    plt.plot(x, y, marker=marker, linestyle=linestyle, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if label != "": plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()


def PlotHist(data, bins = 100, xlabel : str = "", title : str = "", label = None, alpha : int = 1, histtype : str = "bar", sf : int = 2, density : bool = False, x_scale : str = "linear", y_scale : str = "linear", newFigure : bool = True, annotation : str = None, stacked : bool = False, color = None):
    """ Plot 1D histograms.

    Returns:
        np.arrays : bin heights and edges
    """
    if newFigure is True: plt.figure()
    height, edges, _ = plt.hist(data, bins, label = label, alpha = alpha, density = density, histtype = histtype, stacked = stacked, color = color)
    binWidth = round((edges[-1] - edges[0]) / len(edges), sf)
    # TODO: make ylabel a parameter
    if density == False:
        yl = "Number of entries (bin width=" + str(binWidth) + ")"
    else:
        yl = "Normalized number of entries (bin width=" + str(binWidth) + ")"
    plt.ylabel(yl)
    plt.xlabel(xlabel)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.title(title)
    if label is not None: plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    return height, edges


def PlotHist2D(data_x, data_y, bins : int = 100, x_range : list = [], y_range : list = [], z_range : list = [None, None], xlabel : str = "", ylabel : str = "", title : str = "", label : str = "", x_scale : str = "linear", y_scale : str = "linear", newFigure : bool = True, annotation : str = None):
    """ Plot 2D histograms.

    Returns:
        np.arrays : bin heights and edges
    """
    if newFigure is True: plt.figure()
    # clamp data_x and data_y given the x range
    if len(x_range) == 2:
        data_y = data_y[data_x >= x_range[0]] # clamp y before x
        data_x = data_x[data_x >= x_range[0]]
        
        data_y = data_y[data_x <= x_range[1]]
        data_x = data_x[data_x <= x_range[1]]
    
    # clamp data_x and data_y given the y range
    if len(y_range) == 2:
        data_x = data_x[data_y >= y_range[0]] # clamp x before y
        data_y = data_y[data_y >= y_range[0]]
        
        data_x = data_x[data_y <= y_range[1]]
        data_y = data_y[data_y <= y_range[1]]

    # plot data with a logarithmic color scale
    height, xedges, yedges, _ = plt.hist2d(data_x, data_y, bins, norm = matplotlib.colors.LogNorm(), label = label, vmin = z_range[0], vmax = z_range[1])
    plt.colorbar()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.title(title)
    if label != "": plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    return height, [xedges, yedges]


def PlotHistComparison(datas, xRange : list = [], bins : int = 100, xlabel : str = "", title : str = "", labels : list = [], alpha : int = 1, histtype : str = "step", x_scale : str = "linear", y_scale : str = "linear", sf : int = 2, density : bool = True, annotation : str = None):
    """ Plots multiple histograms on one plot

    Args:
        datas (any): list of data sets to plot
        xRange (list, optional): plot range for all data. Defaults to [].
    """
    plt.figure()
    for i in range(len(labels)):
        data = datas[i]
        if xRange and len(xRange) == 2:
            data = data[data > xRange[0]]
            data = data[data < xRange[1]]
        if i == 0:
            _, edges = PlotHist(data, bins, xlabel, title, labels[i], alpha, histtype, sf, density, newFigure=False)
        else:
            PlotHist(data, edges, xlabel, title, labels[i], alpha, histtype, sf, density, newFigure=False)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')


def PlotHist2DComparison(x : list, y : list, bins : int = 50, xlabels : list = [None] * 2, ylabels : list = [None] * 2, colourmap = "plasma", figsize = [6.4, 4.8]):
    edges = [bins, bins]
    h_range = [[np.min(x), np.max(x)], [np.min(y), np.max(y)]]

    heights = []
    for i, j in zip(x, y):
        h, edges[0], edges[1] = np.histogram2d(np.array(i), np.array(j), bins = edges, range = h_range)
        h[h == 0] = np.nan
        heights.append(h.T)

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
    extent = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]

    cmin = np.nanmin(heights)
    cmax = np.nanmax(heights)

    for i in range(len(heights)):
        im = ax[i].imshow(heights[i], origin = "lower", interpolation = "nearest", extent = extent, vmin = cmin, vmax = cmax, cmap = colourmap)
        ax[i].set_xlabel(xlabels[i])
        ax[i].set_ylabel(ylabels[i])

    fig.tight_layout()
    fig.colorbar(im, ax = ax.ravel().tolist())


def UniqueData(data):
    """ Formats data to be plotted as a bar plot based on unique values and how often they occur.
    """
    unique, counts = np.unique(data, return_counts=True)
    counts = list(counts)
    unique_labels = []
    for i in range(len(unique)):
        if str(unique[i]) != "[]":
            unique_labels.append(str(unique[i]))
        else:
            counts.pop(i)
    return unique_labels, counts


def PlotBar(data, width : float = 0.8, xlabel : str = "", title : str = "", label : str = "", alpha : float = 1, newFigure : bool = True, annotation : str = None, bar_labels : bool = False):
    """ Plot a bar graph or unique items in data.
    """
    if newFigure is True: plt.figure()

    unique, counts = UniqueData(data)
    bar = plt.bar(unique, counts, width, label=label, alpha=alpha)
    if bar_labels: plt.bar_label(bar, counts)
    plt.ylabel("Counts")
    plt.xlabel(xlabel)
    plt.title(title)
    if label != "": plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    return unique, counts


def PlotBarComparision(data_1, data_2, width : float = 0.8, xlabel : str = "", title : str = "", label_1 : str = "", label_2 : str = "", newFigure : bool = True, annotation : str = None):
    """ Plot two bar plots of the same data type side-by-side.
    """
    if newFigure is True: plt.figure()
    
    unique_1, counts_1 = UniqueData(data_1)
    unique_2, counts_2 = UniqueData(data_2)

    m = None
    if(len(unique_2) > len(unique_1)):
        m = unique_2
        unique_2 = unique_1
        unique_1 = m

        m = counts_2
        counts_2 = counts_1
        counts_1 = m

        m = label_2
        label_2 = label_1
        label_1 = m

    missing = [i for i in unique_2 if i not in unique_1]
    loc = [unique_2.index(i) for i in missing]
    for i in loc:
        counts_1.insert(i, 0)
        unique_1.insert(i, unique_2[i] )

    missing = [i for i in unique_1 if i not in unique_2]
    loc = [unique_1.index(i) for i in missing]
    for i in loc:
        counts_2.insert(i, 0)

    x = np.arange(len(unique_1))

    plt.bar(x - (width/2), counts_1, width, label = label_1)
    plt.bar(x + (width/2), counts_2, width, label = label_2)
    plt.xticks(x, unique_1)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title)
    plt.legend()
    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    return [unique_1, counts_1], [unique_2, counts_2]


def PlotStackedBar(bars, labels, xlabel : str = None, colours : list = None, alpha : float = 1, label_title : str = None, width : float = 0.8, annotation : str = None):
    """ Plot stacked bar chart.

    Args:
        bars : bar plot data for each sample, is a list of a list of two numpy arrays, first is the labels, second is the counts.
        labels : sample labels
    """
    bar = []
    for i in range(len(bars)):
        bar.extend(bars[i][0])
    bar = np.unique(bar)

    # pad empty bars and sort
    for i in range(len(bars)):
        for j in range(len(bar)):
            if(bar[j] not in bars[i][0]):
                bars[i][0] = np.append(bars[i][0], bar[j])
                bars[i][1] = np.append(bars[i][1], 0)
        bars[i] = [j[bars[i][0].argsort()] for j in bars[i]]
        bars[i][0] = [str(i) for i in bars[i][0]] # convert bar values to string for better plotting

    # stack counts
    for i in range(len(bars)):
        if i == 0: continue
        bars[i][1] += bars[i-1][1]
    plt.figure()

    if colours == None:
        for b in reversed(bars):
            bar = plt.bar(b[0], b[1], alpha = alpha, width = width)
    else:
        for b, c in zip(reversed(bars), reversed(colours)):
            bar = plt.bar(b[0], b[1], color = c, alpha = alpha, width = width)

    plt.legend(labels = labels[::-1], title = label_title)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.tight_layout()

    if annotation is not None:
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction')


def BW(x, A : float, M : float, T : float):
    """ Breit Wigner distribution.
    
    Args:
        x : COM energy (data)
        M : particle mass
        T : decay width
        A : amplitude to scale PDF to data
    """
    # see https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution for its definition
    gamma = np.sqrt(M**2 * (M**2 + T**2))  # formula is complex, so split it into multiple terms
    k = A * ((2 * 2**0.5)/np.pi) * (M * T * gamma)/((M**2 + gamma)**0.5)
    return k/((x**2 - M**2)**2 + (M*T)**2)


def Gaussian(x, A : float, mu : float, sigma : float):
    """ Gaussain distribution (not normalised).
    Args:
        x : sample data
        A : amplitude to scale
        mu : mean value
        sigma : standard deviation
    """
    return A * np.exp( -0.5 * ((x-mu) / sigma)**2 )


def ChiSqrPDF(x, ndf : int):
    """ Chi Squared PDF.
    Args:
        x : sample data
        ndf : degrees of freedom
        scale : pdf normalisation?
        poly : power term
        exponent : exponential term
    """
    scale = 1 /( np.power(2, ndf/2) * gamma(ndf/2) )
    poly = np.power(x, ndf - 2)
    exponent = np.exp(- (x**2) / 2)
    return scale * poly * exponent


def LeastSqrFit(data, nbins : int = 25, function = Gaussian, pinit : list = None, xlabel : str = "", sf : int = 3, interpolation : int = 500, capsize : float = 1):
    """ Fit a function to binned data using the least squares method, implemented in Scipy.
        Plots the fitted function and histogram with y error bars.
    Args:
        hist : height of each histogram bin
        bins : data range of each bin
        x : ceneterd value of each bin
        binWidth : width of the bins
        uncertainty : poisson uncertainty of each bin
        scale : normalisation of data for curve fitting
        popt : paramters of the fitting function which minimises the chi-qsr
        cov : covarianc matrix of least sqares fit
        ndf : number of degrees of freedom
        chi_sqr : chi squared
        x_inter : interplolated x values of the best fit curve to show the fit in a plot
        y_inter : interpolated y values
    """
    data = data[data != -999] # reject null data
    hist, bins = np.histogram(data, nbins) # bin data
    x = (bins[:-1] + bins[1:])/2  # get center of bins
    x = np.array(x, dtype=float) # convert from object to float
    binWidth = bins[1] - bins[0] # calculate bin width

    uncertainty = np.sqrt(hist) # calculate poisson uncertainty if each bin

    # normalise data
    scale = 1 / max(hist)
    uncertainty = uncertainty * scale
    hist = hist * scale
    
    popt, cov = curve_fit(function, x, hist, pinit, uncertainty) # perform least squares curve fit, get the optimal function parameters and covariance matrix

    ndf = nbins - len(popt) # degrees of freedom
    chi_sqr = np.sum( (hist - Gaussian(x, *popt) )**2 / Gaussian(x, *popt) ) # calculate chi squared

    p = quad(ChiSqrPDF, np.sqrt(chi_sqr), np.Infinity, args=(ndf)) # calculate the p value, integrate the chi-qsr function from the chi-qsr to infinity to get p(x > chi-sqr)

    print( "chi_sqaured / ndf: " + str(chi_sqr/ ndf))
    print("p value and compuational error: " + str(p))

    popt[0] = popt[0] / scale
    print("optimised parameters: " + str(popt))

    cov = np.sqrt(cov)  # get standard deviations
    print("uncertainty in optimised parameters: " + str([cov[0, 0], cov[1, 1], cov[2, 2]]))
    
    # calculate plot points for optimised curve
    x_inter = np.linspace(x[0], x[-1], interpolation)  # create x values to draw the best fit curve
    y_inter = function(x_inter, *popt)

    # plot data / fitted curve
    plt.bar(x, hist/scale, binWidth, yerr=uncertainty/scale, capsize=capsize, color="C0")
    Plot(x_inter, y_inter)
    binWidth = round(binWidth, sf)
    plt.ylabel("Number of events (bin width=" + str(binWidth) + ")")
    plt.xlabel(xlabel)
    plt.tight_layout()