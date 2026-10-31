import matplotlib
import matplotlib.pyplot as plt

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
        plt.style.use("default") # first load the default to reset any previous changes made by other styles
        plt.style.use('ggplot')
        plt.rcParams.update({'patch.linewidth': 1})
        plt.rcParams.update({'font.size': font_scale * 10})
        plt.rcParams.update({"axes.titlecolor" : "#555555"})
        plt.rcParams.update({"axes.titlesize" : font_scale * 12})
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['legend.fontsize'] = "small"
        plt.rcParams["font.family"] = font_style

        plt.rc('text.latex', preamble=r"\\usepackage{amsmath}")
        if custom_colors:
            plt.rcParams.update({"axes.prop_cycle" : plt.cycler("color", custom_colors)})
        if dark:
            l_2 = [
            matplotlib.cm.get_cmap("tab20c").colors[0],
            matplotlib.cm.get_cmap("tab20c").colors[8],
            matplotlib.cm.get_cmap("tab20b").colors[13],
            matplotlib.cm.get_cmap("tab20b").colors[0],
            matplotlib.cm.get_cmap("tab20b").colors[17],
            matplotlib.cm.get_cmap("tab20b").colors[4],
            matplotlib.cm.get_cmap("tab20c").colors[12],
            matplotlib.cm.get_cmap("tab20c").colors[16],
            ]
            plt.rcParams.update({"axes.prop_cycle" : plt.cycler("color", l_2)})
        if extend_colors:
            plt.rcParams.update({"axes.prop_cycle" : plt.cycler("color", matplotlib.cm.get_cmap("tab20").colors)})
        return


def SetPlotStyle(extend_colors : bool = False, custom_colors : list = None, dpi : int = 300, dark : bool = False, font_scale : float = 1, font_style : str = "sans"):
    plt.style.use("default") # first load the default to reset any previous changes made by other styles
    plt.style.use('ggplot')
    plt.rcParams.update({'patch.linewidth': 1})
    plt.rcParams.update({'font.size': font_scale * 10})
    plt.rcParams.update({"axes.titlecolor" : "#555555"})
    plt.rcParams.update({"axes.titlesize" : font_scale * 12})
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['legend.fontsize'] = "small"
    plt.rcParams["font.family"] = font_style

    plt.rc('text.latex', preamble=r"\\usepackage{amsmath}")
    if custom_colors:
        plt.rcParams.update({"axes.prop_cycle" : plt.cycler("color", custom_colors)})
    if dark:
        l_2 = [
        matplotlib.cm.get_cmap("tab20c").colors[0],
        matplotlib.cm.get_cmap("tab20c").colors[8],
        matplotlib.cm.get_cmap("tab20b").colors[13],
        matplotlib.cm.get_cmap("tab20b").colors[0],
        matplotlib.cm.get_cmap("tab20b").colors[17],
        matplotlib.cm.get_cmap("tab20b").colors[4],
        matplotlib.cm.get_cmap("tab20c").colors[12],
        matplotlib.cm.get_cmap("tab20c").colors[16],
        ]
        plt.rcParams.update({"axes.prop_cycle" : plt.cycler("color", l_2)})
    if extend_colors:
        plt.rcParams.update({"axes.prop_cycle" : plt.cycler("color", matplotlib.cm.get_cmap("tab20").colors)})


