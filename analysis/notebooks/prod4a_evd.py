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
from python.analysis.EventDisplay import EventDisplay

from apps.prod4a_merge_study import EventSelection, ShowerMergeQuantities

def PlotImpactParameter(eventDisplay : EventDisplay, startPoint, target, direction):
    l = np.abs(vector.dot(vector.sub(target, startPoint), direction))
    t = vector.magnitude(vector.cross(vector.sub(target, startPoint), direction))
    point = vector.add(startPoint, vector.prod(-l, direction))
    eventDisplay.Point(point, "x", "purple", 40)
    eventDisplay.Line(startPoint, target, "purple", "-") # d
    eventDisplay.Line(startPoint, point, "purple", "-") # l
    eventDisplay.Line(point, target, "purple", "--") # impact parameter (t)
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
            display.PFO(background_points[p], marker = "x", colour = background_colours[p % len(background_colours)], startPoint = background_startPoints[p], pdg = background_pdg[p])
            if plotIP: PlotImpactParameter(display, background_startPoints[p], events.recoParticles.beamVertex[i], background_direction[p])
    else:
        display.PFO(background_points[i], marker = "x", colour = background_colours[i % len(background_colours)], startPoint = background_startPoints[i], pdg = background_pdg[i])
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
            display.PFO(signal_points[p], marker = "x", colour = colours[p % len(colours)], startPoint = signal_startPoints[p], direction = signal_direction[p], pdg = signal_pdg[p])
            if plotIP: PlotImpactParameter(display, signal_startPoints[p], events.recoParticles.beamVertex[i], signal_direction[p])
    else:
        display.PFO(signal_points[i], marker = "x", colour = colours[i % len(colours)], startPoint = signal_startPoints[i], direction = signal_direction[i], pdg = signal_pdg[p])
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
    display.PFO(points[0], marker = "x", colour = "green", startPoint = startPoints[0], direction = directions[0])#, pdg=pdgs[0])
    display.PFO(points[1], marker = "x", colour = "blue", startPoint = startPoints[1], direction = directions[1])#, pdg=pdgs[1])


    points = events.recoParticles.spacePoints[to_merge][n]
    startPoints = events.recoParticles.startPos[to_merge][n]
    directions = events.recoParticles.direction[to_merge][n]
    pdgs = events.trueParticlesBT.pdg[to_merge][n]
    beam_mask = np.logical_not(events.recoParticles.number == events.recoParticles.beam_number)[to_merge][q.null]
    #* Plot background PFOs
    if showBackground: PlotBackgroundPFO(display, n, background, beam_mask, points, startPoints, directions, pdg=None, i = -1, plotIP = False)

    #* Plot Signal PFOs
    if showSignal:
        PlotSignalPFO(display, n, signal, points, startPoints, directions, None, 0, i = -1, plotIP = False) # green
        PlotSignalPFO(display, n, signal, points, startPoints, directions, None, 1, i = -1, plotIP = False) # blue


    #* Plot BeamParticle:
    beam_mask = events.recoParticles.number == events.recoParticles.beam_number
    points = events.recoParticles.spacePoints[beam_mask][n]
    pdg = events.trueParticlesBT.pdg[beam_mask][n]
    display.PFO(points, marker="o", colour="black", startPoint = events.recoParticles.beamVertex[n], pdg=None)

    #* Plot beam vertex
    display.Point(events.recoParticles.beamVertex[n], marker="x", colour="red", pointSize=100)

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
    text += "$E_{\gamma_{0}}$: " + str(events.trueParticlesBT.energy[start_showers_merged][n][0])[0:4] + "GeV \n"
    text += "$E_{\gamma_{1}}$: " + str(events.trueParticlesBT.energy[start_showers_merged][n][1])[0:4] + "GeV "
    
    props = dict(boxstyle='round', facecolor='grey', alpha=1)
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
    display.fig2D.savefig(f"prod4a_merging_evd/{name}-2D.png", dpi=500)
    plt.close(display.fig2D)
    display.fig3D.savefig(f"prod4a_merging_evd/{name}-3D.png", dpi=500)
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
    eventNum = 69 # 11 and 13 identical??
    nPFO = ak.num(events.recoParticles.spacePoints.x)
    showSignal = True
    showBackground = True

    if eventNum == -1:
        for i in range(len(events.eventNum)):
            print(f"rendering {i+1} out of {len(events.eventNum)}")
            print(f"event: {events.eventNum[i]}_{events.run[i]}_{events.subRun[i]}")
            try:
                RenderEventDisplay(i)
            except:
                print(f"Something went wrong with {i+1}, you should look into it")

    else:
        RenderEventDisplay(eventNum)

if __name__ == "__main__":
    main()