"""
Created on: 03/08/2022 16:23

Author: Shyam Bhuller

Description: Create Event Display for Prod4a Shower merging study 
"""

from types import SimpleNamespace
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
import awkward as ak
import numpy as np

from python.analysis import Master, vector

from apps.prod4a_merge_study import EventSelection, ShowerMergeQuantities

class EventDisplay:
    xlim = (-350, 350)
    ylim = (0, 600)
    zlim = (0, 600)
    def __init__(self, eventID : str, run : str, subrun : str, plotOrtho : bool = True, plot3D : bool = True):
        plt.rcParams.update({'font.size': 14})
        title = f"event: {eventID}, run: {run}, subrun: {subrun}"
        if plotOrtho:
            plt.figure(1).clf()
            self.fig2D, (self.xy, self.xz) = plt.subplots(nrows=2, ncols=1, figsize=(6.4*20, 4.8*20), num=1)
            self.xy.set_xlabel("x (cm)")
            self.xz.set_xlabel("x (cm)")
            self.xy.set_ylabel("y (cm)")
            self.xz.set_ylabel("z (cm)")
            self.xy.set_title(title)

        if plot3D:
            plt.figure(2).clf()
            self.fig3D = plt.figure(num=2)
            self.ax3D = Axes3D(self.fig3D)
            self.ax3D.set_xlabel("x (cm)")
            self.ax3D.set_ylabel("z (cm)")
            self.ax3D.set_zlabel("y (cm)")
            self.ax3D.set_title(title, y = 1, pad = -10)
        return

    def DetectorBounds(self, x=xlim, y=ylim, z=zlim):
        self.xy.set_xlim(x)
        self.xy.set_ylim(y)
        self.xz.set_xlim(x)
        self.xz.set_ylim(z)
        self.ax3D.set_xlim3d(x)
        self.ax3D.set_ylim3d(z)
        self.ax3D.set_zlim3d(y)

    def PlotPFO(self, points : ak.Array, marker : str, colour : str, pointSize : int = 2, startPoint : ak.Record = None, direction : ak.Record = None, pdg : int = None):
        points = points[points.x != -999] # don't plot null space point values

        x_mask = np.logical_and(points.x < self.xlim[1], points.x > self.xlim[0])
        y_mask = np.logical_and(points.y < self.ylim[1], points.y > self.ylim[0])
        z_mask = np.logical_and(points.z < self.zlim[1], points.z > self.zlim[0])
        fudicial_cut = np.logical_or(np.logical_or(x_mask, y_mask), z_mask)
        points = points[fudicial_cut]

        if self.fig2D:
            self.xy.scatter(points.x, points.y, pointSize, marker=marker, color=colour)
            self.xz.scatter(points.x, points.z, pointSize, marker=marker, color=colour)
            if startPoint is not None:
                self.xy.scatter(startPoint.x, startPoint.y, pointSize * 30, marker="x", color=colour)
                self.xz.scatter(startPoint.x, startPoint.z, pointSize * 30, marker="x", color=colour)
        if self.fig3D:
            self.ax3D.scatter(points.x, points.z, points.y, s=pointSize, marker=marker, color=colour)
            if startPoint is not None:
                self.ax3D.scatter(startPoint.x, startPoint.z, startPoint.y, s=pointSize * 30, marker="x", color=colour)
        if direction is not None and startPoint is not None:
            self.PlotLine(startPoint, vector.add(startPoint, vector.prod(10, direction)), colour, lineStyle="-")
        if pdg is not None and startPoint is not None:
            self.PlotText(startPoint, str(pdg))
        return

    def PlotPoint(self, point : ak.Record, marker : str, colour : str, pointSize : int = 2):
        if self.fig2D:
            self.xy.scatter(point.x, point.y, pointSize, marker=marker, color=colour)
            self.xz.scatter(point.x, point.z, pointSize, marker=marker, color=colour)
        if self.fig3D:
            self.ax3D.scatter(point.x, point.z, point.y, s=pointSize, marker=marker, color=colour)
        return

    def PlotLine(self, start : ak.Record, end : ak.Record, colour : str, lineStyle="-"):
        if self.fig2D:
            self.xy.plot([start.x, end.x], [start.y, end.y], lineStyle, color = colour)
            self.xz.plot([start.x, end.x], [start.z, end.z], lineStyle, color = colour)
        if self.fig3D:
            self.ax3D.plot([start.x, end.x], [start.z, end.z], [start.y, end.y], lineStyle, color = colour)
        return

    def PlotText(self, point : ak.Record, text : str, fontsize : int = 16):
        if self.fig2D:
            self.xy.text(point.x, point.y, str(text), fontsize = fontsize, clip_on = True)
            self.xz.text(point.x, point.z, str(text), fontsize = fontsize, clip_on = True)
        if self.fig3D:
            self.ax3D.text(point.x, point.z, point.y, str(text), fontsize = fontsize, clip_on = True)
            self.ax3D.set_clip_on(True)
        return


def PlotImpactParameter(eventDisplay : EventDisplay, startPoint, target, direction):
    l = np.abs(vector.dot(vector.sub(target, startPoint), direction))
    t = vector.magnitude(vector.cross(vector.sub(target, startPoint), direction))
    point = vector.add(startPoint, vector.prod(-l, direction))
    eventDisplay.PlotPoint(point, "x", "purple", 40)
    eventDisplay.PlotLine(startPoint, target, "purple", "-") # d
    eventDisplay.PlotLine(startPoint, point, "purple", "-") # l
    eventDisplay.PlotLine(point, target, "purple", "--") # impact parameter (t)
    print(f"plotted impact parameter: {vector.magnitude(vector.sub(point, target))} cm")
    print(f"calculated impact parameter: {t} cm")
    return

signal_colours = [["olivedrab", "yellowgreen", "darkolivegreen", "greenyellow", "chartreuse", "lawngreen", "honeydew"],
                  ["aquamarine", "turquoise", "lightseagreen", "mediumturquoise", "azure", "lightcyan", "paleturquoise"]]

background_colours = ["bisque", "darkorange", "burlywood", "tan", "navajowhite", "blanchedalmond", "papayawhip", "moccasin", "orange", "wheat", "peru", "sandybrown", "saddlebrown", "chocolate"]


def PlotBackgroundPFO(display : EventDisplay, eventNum : int, background : ak.Array, beam_mask : ak.Array, points : ak.Array, start : ak.Record, direction : ak.Record, pdg : ak.Array = None, i : int = -1, plotIP : bool = False):
    #* Plot background PFOs
    if beam_mask is not None:
        mask = np.logical_and(background[eventNum], beam_mask[eventNum])
    else:
        mask = background[eventNum]
    background_points = points[mask]
    background_startPoints = start[mask]
    background_direction = direction[mask]
    if pdg is not None:
        background_pdg = pdg[mask]
    else:
        background_pdg = [None]*ak.num(background_points, 0)
    if i == -1:
        for p in range(ak.num(background_points, 0)):
            display.PlotPFO(background_points[p], marker = "x", colour = background_colours[p % len(background_colours)], startPoint = background_startPoints[p], pdg = background_pdg[p])
            if plotIP: PlotImpactParameter(display, background_startPoints[p], events.recoParticles.beamVertex[i], background_direction[p])
    else:
        display.PlotPFO(background_points[i], marker = "x", colour = background_colours[i % len(background_colours)], startPoint = background_startPoints[i], pdg = background_pdg[i])
        if plotIP: PlotImpactParameter(display, background_startPoints[i], events.recoParticles.beamVertex[i], background_direction[i])
    return


def PlotSignalPFO(display : EventDisplay, eventNum : int, signal : ak.Array, points : ak.Array, start : ak.Record, direction : ak.Record, pdg : ak.Array = None, start_shower : int = 0, i : int = -1, plotIP : bool = False):
    if start_shower not in [0, 1]:
        raise Exception("start_shower must be 0 or 1")
    mask = signal[start_shower][eventNum]
    colours = signal_colours[start_shower]

    signal_points = points[mask]
    signal_startPoints = start[mask]
    signal_direction = direction[mask]
    if pdg is not None:
        signal_pdg = pdg[mask]
    else:
        signal_pdg = [None]*ak.num(signal_points, 0)
    if i == -1:
        print(f"Number of signal PFOs for start shower {start_shower}: {ak.num(signal_points, 0)}")
        for p in range(ak.num(signal_points, 0)):
            display.PlotPFO(signal_points[p], marker = "x", colour = colours[p % len(colours)], startPoint = signal_startPoints[p], direction = signal_direction[p], pdg = signal_pdg[p])
            if plotIP: PlotImpactParameter(display, signal_startPoints[p], events.recoParticles.beamVertex[i], signal_direction[p])
    else:
        display.PlotPFO(signal_points[i], marker = "x", colour = colours[i % len(colours)], startPoint = signal_startPoints[i], direction = signal_direction[i], pdg = signal_pdg[p])
        if plotIP:
            PlotImpactParameter(display, signal_startPoints[i], events.recoParticles.beamVertex[i], signal_direction[i])


def RenderEventDisplay(n):
    display = EventDisplay(events.eventNum[n], events.run[n], events.subRun[n])

    #* now Plot start showers:
    start_showers_merged = np.logical_or(*start_showers)
    points = events.recoParticles.spacePoints[start_showers_merged][n]
    startPoints = events.recoParticles.startPos[start_showers_merged][n]
    directions = events.recoParticles.direction[start_showers_merged][n]
    pdgs = events.trueParticlesBT.pdg[start_showers_merged][n]
    display.PlotPFO(points[0], marker = "x", colour = "green", startPoint = startPoints[0], direction = directions[0], pdg=pdgs[0])
    display.PlotPFO(points[1], marker = "x", colour = "blue", startPoint = startPoints[1], direction = directions[1], pdg=pdgs[1])


    points = events.recoParticles.spacePoints[to_merge][n]
    startPoints = events.recoParticles.startPos[to_merge][n]
    directions = events.recoParticles.direction[to_merge][n]
    pdgs = events.trueParticlesBT.pdg[to_merge][n]
    beam_mask = np.logical_not(events.recoParticles.number == events.recoParticles.beam_number)[to_merge][q.null]
    #* Plot background PFOs
    if showBackground: PlotBackgroundPFO(display, n, background, beam_mask, points, startPoints, directions, pdgs, i = -1, plotIP = False)

    #* Plot Signal PFOs
    if showSignal:
        PlotSignalPFO(display, n, signal, points, startPoints, directions, pdgs, 0, i = -1, plotIP = False) # green
        PlotSignalPFO(display, n, signal, points, startPoints, directions, pdgs, 1, i = -1, plotIP = False) # blue


    #* Plot BeamParticle:
    beam_mask = events.recoParticles.number == events.recoParticles.beam_number
    points = events.recoParticles.spacePoints[beam_mask][n]
    pdg = events.trueParticlesBT.pdg[beam_mask][n]
    display.PlotPFO(points, marker="o", colour="black", startPoint = events.recoParticles.beamVertex[n], pdg=pdg)

    #* Plot beam vertex
    display.PlotPoint(events.recoParticles.beamVertex[n], marker="x", colour="red", pointSize=100)

    custom_lines = [matplotlib.lines.Line2D([0], [0], color="black", lw=2),
                    matplotlib.lines.Line2D([0], [0], color="green", lw=2),
                    matplotlib.lines.Line2D([0], [0], color="lime", lw=2),
                    matplotlib.lines.Line2D([0], [0], color="blue", lw=2),
                    matplotlib.lines.Line2D([0], [0], color="cyan", lw=2),
                    matplotlib.lines.Line2D([0], [0], color="orange", lw=2),
                    matplotlib.lines.Line2D([0], [0], marker="x", color="red", markersize=15, lw=0),
                    ]

    display.ax3D.legend(custom_lines, ["beam particle", "start shower 1", "signal 1", "start shower 2", "signal 2", "background", "decay vertex"], loc="lower right")
    display.xy.legend(custom_lines, ["beam particle", "start shower 1", "signal 1", "start shower 2", "signal 2", "background", "decay vertex"])
    display.xy.grid()
    display.xz.grid()

    #* plot some information about the event:
    text =  "$E_{\pi^{+}}$: "  + str(events.trueParticlesBT.energy[n][events.recoParticles.beam_number[n] == events.recoParticles.number[n]][0])[0:4] + "GeV \n"
    text += "$E_{\pi^{0}}$: "  + str(events.trueParticles.energy[n][events.trueParticles.pdg[n] == 111][0])[0:4] + "GeV \n"
    text += "$E_{\gamma 0}$: " + str(events.trueParticlesBT.energy[start_showers_merged][n][0])[0:4] + "GeV \n"
    text += "$E_{\gamma 1}$: " + str(events.trueParticlesBT.energy[start_showers_merged][n][1])[0:4] + "GeV "
    
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    display.xz.text(0.01, 0.85, text, transform=display.xz.transAxes, fontsize=14, bbox=props)

    roi = SimpleNamespace(**{
        "x": [events.recoParticles.beamVertex[n].x-100, events.recoParticles.beamVertex[n].x+100],
        "y": [events.recoParticles.beamVertex[n].y-100, events.recoParticles.beamVertex[n].y+100],
        "z": [events.recoParticles.beamVertex[n].z-100, events.recoParticles.beamVertex[n].z+100]
    })
    display.DetectorBounds(roi.x, roi.y, roi.z)

    name = f"{events.eventNum[n]}_{events.run[n]}_{events.subRun[n]}"

    display.fig2D.set_size_inches(6.4*3, 4.8*3)
    display.fig3D.set_size_inches(6.4*3, 4.8*3)
    display.fig2D.tight_layout()
    display.fig2D.savefig(f"prod4a_merging_evd/{name}-2D.png", dpi=400)
    plt.close(display.fig2D)
    display.fig3D.savefig(f"prod4a_merging_evd/{name}-3D.png", dpi=400)
    plt.close(display.fig3D)


def main():
    global events, start_showers, to_merge, showBackground, showSignal, signal, background, q
    ##################################################################################################
    events = Master.Data("work/ROOTFiles/Prod4a_6GeV_BeamSim_00_evd.root", True)

    #* create hit space point arrays
    events.recoParticles.spacePoints = ak.zip({"x" : events.io.Get("reco_daughter_allShower_spacePointX"), 
                                            "y" : events.io.Get("reco_daughter_allShower_spacePointY"),
                                            "z" : events.io.Get("reco_daughter_allShower_spacePointZ")})
    start_showers = EventSelection(events)

    #* get boolean mask of PFP's to merge
    to_merge = np.logical_not(np.logical_or(*start_showers))

    q = ShowerMergeQuantities(events, to_merge)
    q.Evaluate(events, start_showers)

    #* get boolean mask of PFP's which are actual fragments of the starting showers
    start_shower_ID = events.trueParticlesBT.number[np.logical_or(*start_showers)]
    to_merge_ID = events.trueParticlesBT.number[to_merge]
    signal = [to_merge_ID == start_shower_ID[:, i] for i in range(2)] # signal are the PFOs which is a fragment of the ith starting shower

    #* define signal and background
    signal_all = np.logical_or(*signal)
    signal_all = signal_all[q.null]
    background = np.logical_not(signal_all) # background is all other PFOs unrelated to the pi0 decay
    signal = [signal[i][q.null] for i in range(2)]

    ##################################################################################################

    nEvents = ak.num(events.recoParticles.spacePoints.x, 0)
    eventNum = -1 # 11 and 13 identical??
    nPFO = ak.num(events.recoParticles.spacePoints.x)
    showSignal = True
    showBackground = True

    if eventNum == -1:
        for i in range(len(events.eventNum)):
            print(f"rendering {i+1} out of {len(events.eventNum)}")
            print(f"event: {events.eventNum[n]}_{events.run[n]}_{events.subRun[n]}")
            try:
                RenderEventDisplay(i)
            except:
                print(f"Something went wrong with {i+1}, you should look into it")

    else:
        try:
            RenderEventDisplay(eventNum)
        except:
            print(f"Something went wrong with {eventNum}, you should look into it")

if __name__ == "__main__":
    main()