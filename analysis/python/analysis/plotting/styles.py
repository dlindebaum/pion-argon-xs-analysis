import matplotlib
import matplotlib.pyplot as plt

class PlotStyler:
    """Apply ggplot-based Matplotlib styles and optional fonts, colors, and sizing.

        Args:
            extend_colors (bool): Use the full ``tab20`` color cycle.
            custom_colors (list): Optional explicit color cycle.
            dpi (int): Figure resolution in dots per inch.
            dark (bool): Use the module's alternate dark palette.
            font_scale (float): Scale factor for configured font sizes.
            font_style (str): Matplotlib font family.
    """
    def __init__(self, extend_colors : bool = False, custom_colors : list = None, dpi : int = 100, dark : bool = False, font_scale : float = 1, font_style : str = "sans"):
        """Store the style options and apply them to Matplotlib.

            Args:
                extend_colors (bool): Use the full ``tab20`` color cycle.
                custom_colors (list): Optional explicit color cycle.
                dpi (int): Figure resolution in dots per inch.
                dark (bool): Use the module's alternate dark palette.
                font_scale (float): Scale factor for configured font sizes.
                font_style (str): Matplotlib font family.
        """
        self.args = locals()
        self.args.pop("self")
        set_plot_style(**self.args)

    class __updater__:
        """Temporarily apply style overrides and restore the parent style on exit.

            Args:
                parent ('PlotStyler'): Styler whose options provide defaults and are restored on exit.
                extend_colors (bool): Optional override for the ``tab20`` color cycle.
                custom_colors (list): Optional override for the color cycle.
                dpi (int): Optional override for figure resolution.
                dark (bool): Optional override for the alternate palette.
                font_scale (float): Optional override for font-size scaling.
                font_style (str): Optional override for Matplotlib font family.
        """
        def __init__(self, parent : "PlotStyler", extend_colors : bool = None, custom_colors : list = None, dpi : int = None, dark : bool = None, font_scale : float = None, font_style : str = None):
            """Keep provided overrides and fill unspecified options from the parent styler.

                Args:
                    parent ('PlotStyler'): Styler supplying defaults and the original style.
                    extend_colors (bool): Optional override for the ``tab20`` color cycle.
                    custom_colors (list): Optional override for the color cycle.
                    dpi (int): Optional override for figure resolution.
                    dark (bool): Optional override for the alternate palette.
                    font_scale (float): Optional override for font-size scaling.
                    font_style (str): Optional override for Matplotlib font family.
            """
            self.args = locals()
            self.args.pop("self")
            self.args.pop("parent")
            self.parent = parent

            for k in self.args:
                if self.args[k] is None:
                    self.args[k] = self.parent.args[k]

            pass
        def __enter__(self):
            """Apply the temporary plotting style when entering the context.
            """
            set_plot_style(**self.args)
            pass
        def __exit__(self, type, value, traceback):
            """Restore the parent styler's settings when leaving the context.

                Args:
                    type (Any): Exception type, if the context exited due to an exception.
                    value (Any): Exception instance, if one was raised.
                    traceback (Any): Traceback for the raised exception, if any.
            """
            set_plot_style(**self.parent.args)
            pass

    def update(self, extend_colors : bool = None, custom_colors : list = None, dpi : int = None, dark : bool = None, font_scale : float = None, font_style : str = None):
        """Create a context manager that applies style overrides for a ``with`` block.

            Args:
                extend_colors (bool): Optional override for the ``tab20`` color cycle.
                custom_colors (list): Optional override for the color cycle.
                dpi (int): Optional override for figure resolution.
                dark (bool): Optional override for the alternate palette.
                font_scale (float): Optional override for font-size scaling.
                font_style (str): Optional override for Matplotlib font family.

            Returns:
                Any: Context manager that restores the prior style on exit.
        """
        return self.__updater__(self, extend_colors, custom_colors, dpi, dark, font_scale, font_style)


def set_plot_style(extend_colors : bool = False, custom_colors : list = None, dpi : int = 300, dark : bool = False, font_scale : float = 1, font_style : str = "sans"):
    """Reset Matplotlib defaults and apply the configured ggplot style, palette, and fonts.

        Args:
            extend_colors (bool): Use the full ``tab20`` color cycle.
            custom_colors (list): Optional explicit color cycle.
            dpi (int): Figure resolution in dots per inch.
            dark (bool): Use the module's alternate dark palette.
            font_scale (float): Scale factor for configured font sizes.
            font_style (str): Matplotlib font family.
    """
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

