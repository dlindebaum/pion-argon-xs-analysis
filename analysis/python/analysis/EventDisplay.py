"""
Created on: 25/09/2022 11:14

Author: Shyam Bhuller

Description: Event display object.
"""
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from python.analysis import vector


class EventDisplay:
    """ Event display class, an object which produces figures and allows the user.
        to draw various objects on the plot (see methods).
    Attributes:
        xlim (tuple): xRange of plots.
        ylim (tuple): yRange of plots.
        zlim (tuple): zRange of plots.
        fig2D (matplotlib.figure.Figure): figure for 2D plots.
        xy (matplotlib.axes._subplots.AxesSubplot): 2D plot on the xy plane.
        xz (matplotlib.axes._subplots.AxesSubplot): 2D plot on the xz plane.
        fig3D (matplotlib.figure.Figure): figure for 3D plots.
        ax3D (mpl_toolkits.mplot3d.axes3d.Axes3D): 3D plot.

    Methods:
        DetectorBounds: Set axes range to dectector bounds.
        PFO: Plot a PFO in the event display.
        Point: Draw a single point.
        Line: Draw a line.
        Text: Draw text.
    """
    xlim = (-350, 350)
    ylim = (0, 600)
    zlim = (0, 600)
    def __init__(self, eventID : str, run : str, subrun : str, plot2D : bool = True, plot3D : bool = True):
        """ Constructor

        Args:
            eventID (str): Event number
            run (str): run number
            subrun (str): subrun number
            plot2D (bool, optional): whether to create 2D plots. Defaults to True.
            plot3D (bool, optional): whether to create 3D plots. Defaults to True.
        """
        #* set plot defaults
        plt.rcParams.update({'font.size': 14})
        title = f"event: {eventID}, run: {run}, subrun: {subrun}"
        
        #* make figures/axes
        if plot2D:
            plt.figure(1).clf() # clear existing 2D figure 
            self.fig2D, (self.xy, self.xz) = plt.subplots(nrows=2, ncols=1, figsize=(6.4*20, 4.8*20), num=1)
            self.xy.set_xlabel("x (cm)")
            self.xz.set_xlabel("x (cm)")
            self.xy.set_ylabel("y (cm)")
            self.xz.set_ylabel("z (cm)")
            self.xy.set_title(title)

        if plot3D:
            plt.figure(2).clf() # clear existing 3D figure 
            self.fig3D = plt.figure(num = 2)
            self.ax3D = self.fig3D.add_subplot(projection = "3d")
            self.ax3D.set_xlabel("x (cm)")
            self.ax3D.set_ylabel("z (cm)") # swtich y and z axes, so that the y axis is vertical
            self.ax3D.set_zlabel("y (cm)")
            self.ax3D.set_title(title, y = 1, pad = -10)
        return

    def DetectorBounds(self, x : tuple = xlim, y : tuple = ylim, z : tuple = zlim):
        """ Set axes range to dectector bounds.

        Args:
            x (tuple, optional): x bound. Defaults to xlim.
            y (tuple, optional): y bound. Defaults to ylim.
            z (tuple, optional): z bound. Defaults to zlim.
        """
        if hasattr(self, "fig2D"):
            self.xy.set_xlim(x)
            self.xy.set_ylim(y)
            self.xz.set_xlim(x)
            self.xz.set_ylim(z)
        if hasattr(self, "fig3D"):
            self.ax3D.set_xlim3d(x)
            self.ax3D.set_ylim3d(z)
            self.ax3D.set_zlim3d(y)

    def PFO(self, points : ak.Array, marker : str, colour : str, alpha : float = 1, pointSize : int = 2, startPoint : ak.Record = None, direction : ak.Record = None, pdg : int = None):
        """ Plot a PFO in the event display.

        Args:
            points (ak.Array): hit space point positions
            marker (str): style of points
            colour (str): colour of points
            pointSize (int, optional): size of points. Defaults to 2.
            startPoint (ak.Record, optional): PFO start point. Defaults to None.
            direction (ak.Record, optional): direction of point. Defaults to None.
            pdg (int, optional): pdg code of point. Defaults to None.
        """
        points = points[points.x != -999] # don't plot null space point values

        #* fiducial cut for hit space points
        x_mask = np.logical_and(points.x < self.xlim[1], points.x > self.xlim[0])
        y_mask = np.logical_and(points.y < self.ylim[1], points.y > self.ylim[0])
        z_mask = np.logical_and(points.z < self.zlim[1], points.z > self.zlim[0])
        fudicial_cut = np.logical_or(np.logical_or(x_mask, y_mask), z_mask)
        points = points[fudicial_cut]

        if hasattr(self, "fig2D"):
            self.xy.scatter(points.x, points.y, pointSize, marker = marker, color = colour, alpha = alpha)
            self.xz.scatter(points.x, points.z, pointSize, marker = marker, color = colour, alpha = alpha)
            if startPoint is not None:
                self.xy.scatter(startPoint.x, startPoint.y, pointSize * 30, marker = "x", color = colour, alpha = alpha)
                self.xz.scatter(startPoint.x, startPoint.z, pointSize * 30, marker = "x", color = colour, alpha = alpha)
        
        if hasattr(self, "fig3D"):
            self.ax3D.scatter(points.x, points.z, points.y, s = pointSize, marker = marker, color = colour, alpha = alpha)
            if startPoint is not None:
                self.ax3D.scatter(startPoint.x, startPoint.z, startPoint.y, s = pointSize * 30, marker = "x", color = colour, alpha = alpha)
        
        if direction is not None and startPoint is not None:
            self.Line(startPoint, vector.add(startPoint, vector.prod(10, direction)), colour, lineStyle="-")
        
        if pdg is not None and startPoint is not None:
            self.Text(startPoint, str(pdg))
        
        return

    def Point(self, point : ak.Record, marker : str, colour : str, pointSize : int = 2):
        """ Draw a single point.

        Args:
            point (ak.Record): point position
            marker (str): point style
            colour (str): point colour
            pointSize (int, optional): point size. Defaults to 2.
        """
        if hasattr(self, "fig2D"):
            self.xy.scatter(point.x, point.y, pointSize, marker=marker, color=colour)
            self.xz.scatter(point.x, point.z, pointSize, marker=marker, color=colour)
        
        if hasattr(self, "fig3D"):
            self.ax3D.scatter(point.x, point.z, point.y, s=pointSize, marker=marker, color=colour)
        
        return

    def Line(self, start : ak.Record, end : ak.Record, colour : str, lineStyle="-"):
        """ Draw a line.

        Args:
            start (ak.Record): start position of line
            end (ak.Record): end position of line
            colour (str): colour of line
            lineStyle (str, optional): line style. Defaults to "-".
        """
        if hasattr(self, "fig2D"):
            self.xy.plot([start.x, end.x], [start.y, end.y], lineStyle, color = colour)
            self.xz.plot([start.x, end.x], [start.z, end.z], lineStyle, color = colour)
        
        if hasattr(self, "fig3D"):
            self.ax3D.plot([start.x, end.x], [start.z, end.z], [start.y, end.y], lineStyle, color = colour)
        
        return

    def Text(self, point : ak.Record, text : str, fontsize : int = 16, colour = "black"):
        """ Draw text.

        Args:
            point (ak.Record): point to draw text (top left of bounding box)
            text (str): the text.
            fontsize (int, optional): font size. Defaults to 16.
        """
        if hasattr(self, "fig2D"):
            self.xy.text(point.x, point.y, str(text), fontsize = fontsize, clip_on = True, color = colour, path_effects = [pe.withStroke(linewidth = 1, foreground = "black")])
            self.xz.text(point.x, point.z, str(text), fontsize = fontsize, clip_on = True, color = colour, path_effects = [pe.withStroke(linewidth = 1, foreground = "black")])

        if hasattr(self, "fig3D"):
            self.ax3D.text(point.x, point.z, point.y, str(text), fontsize = fontsize, clip_on = True, color = colour, path_effects = [pe.withStroke(linewidth = 1, foreground = "black")])
            self.ax3D.set_clip_on(True) # prevents text objects from being rendered outside of the axes bounds

        return
