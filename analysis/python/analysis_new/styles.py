class PlotStyler:
    def __init__(self, extend_colors : bool = False, custom_colors : list = None, dpi : int = 100, dark : bool = False, font_scale : float = 1, font_style : str = "sans"):
        self.args = locals()
        self.args.pop("self")
        PlotStyler.SetPlotStyle(**self.args)

    class __updater__:
        def __init__(self, parent : "PlotStyler", extend_colors : bool = None, custom_colors : list = None, dpi : int = None, dark : bool = None, font_scale : float = None, font_style : str = None):
            self.args = locals()
            self.args.pop("self")
            self.args.pop("parent")
            self.parent = parent

            for k in self.args:
                if self.args[k] is None:
                    self.args[k] = self.parent.args[k]

            pass
        def __enter__(self):
            PlotStyler.SetPlotStyle(**self.args)
            pass
        def __exit__(self, type, value, traceback):
            PlotStyler.SetPlotStyle(**self.parent.args)
            pass

    def Update(self, extend_colors : bool = None, custom_colors : list = None, dpi : int = None, dark : bool = None, font_scale : float = None, font_style : str = None):
        return self.__updater__(self, extend_colors, custom_colors, dpi, dark, font_scale, font_style)

    @staticmethod
    def SetPlotStyle(extend_colors : bool = False, custom_colors : list = None, dpi : int = 300, dark : bool = False, font_scale : float = 1, font_style : str = "sans"):
        Plots.plt.style.use("default") # first load the default to reset any previous changes made by other styles
        Plots.plt.style.use('ggplot')
        Plots.plt.rcParams.update({'patch.linewidth': 1})
        Plots.plt.rcParams.update({'font.size': font_scale * 10})
        Plots.plt.rcParams.update({"axes.titlecolor" : "#555555"})
        Plots.plt.rcParams.update({"axes.titlesize" : font_scale * 12})
        Plots.plt.rcParams['figure.dpi'] = dpi
        Plots.plt.rcParams['legend.fontsize'] = "small"
        Plots.plt.rcParams["font.family"] = font_style

        Plots.plt.rc('text.latex', preamble=r"\\usepackage{amsmath}")
        if custom_colors:
            Plots.plt.rcParams.update({"axes.prop_cycle" : Plots.plt.cycler("color", custom_colors)})
        if dark:
            l_2 = [
            Plots.matplotlib.cm.get_cmap("tab20c").colors[0],
            Plots.matplotlib.cm.get_cmap("tab20c").colors[8],
            Plots.matplotlib.cm.get_cmap("tab20b").colors[13],
            Plots.matplotlib.cm.get_cmap("tab20b").colors[0],
            Plots.matplotlib.cm.get_cmap("tab20b").colors[17],
            Plots.matplotlib.cm.get_cmap("tab20b").colors[4],
            Plots.matplotlib.cm.get_cmap("tab20c").colors[12],
            Plots.matplotlib.cm.get_cmap("tab20c").colors[16],
            ]
            Plots.plt.rcParams.update({"axes.prop_cycle" : Plots.plt.cycler("color", l_2)})
        if extend_colors:
            Plots.plt.rcParams.update({"axes.prop_cycle" : Plots.plt.cycler("color", Plots.matplotlib.cm.get_cmap("tab20").colors)})
        return


def SetPlotStyle(extend_colors : bool = False, custom_colors : list = None, dpi : int = 300, dark : bool = False, font_scale : float = 1, font_style : str = "sans"):
    Plots.plt.style.use("default") # first load the default to reset any previous changes made by other styles
    Plots.plt.style.use('ggplot')
    Plots.plt.rcParams.update({'patch.linewidth': 1})
    Plots.plt.rcParams.update({'font.size': font_scale * 10})
    Plots.plt.rcParams.update({"axes.titlecolor" : "#555555"})
    Plots.plt.rcParams.update({"axes.titlesize" : font_scale * 12})
    Plots.plt.rcParams['figure.dpi'] = dpi
    Plots.plt.rcParams['legend.fontsize'] = "small"
    Plots.plt.rcParams["font.family"] = font_style

    Plots.plt.rc('text.latex', preamble=r"\\usepackage{amsmath}")
    if custom_colors:
        Plots.plt.rcParams.update({"axes.prop_cycle" : Plots.plt.cycler("color", custom_colors)})
    if dark:
        l_2 = [
        Plots.matplotlib.cm.get_cmap("tab20c").colors[0],
        Plots.matplotlib.cm.get_cmap("tab20c").colors[8],
        Plots.matplotlib.cm.get_cmap("tab20b").colors[13],
        Plots.matplotlib.cm.get_cmap("tab20b").colors[0],
        Plots.matplotlib.cm.get_cmap("tab20b").colors[17],
        Plots.matplotlib.cm.get_cmap("tab20b").colors[4],
        Plots.matplotlib.cm.get_cmap("tab20c").colors[12],
        Plots.matplotlib.cm.get_cmap("tab20c").colors[16],
        ]
        Plots.plt.rcParams.update({"axes.prop_cycle" : Plots.plt.cycler("color", l_2)})
    if extend_colors:
        Plots.plt.rcParams.update({"axes.prop_cycle" : Plots.plt.cycler("color", Plots.matplotlib.cm.get_cmap("tab20").colors)})


