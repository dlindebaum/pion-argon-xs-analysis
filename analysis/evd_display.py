"""
Created on: 30/03/2022 19:32

Author: Shyam Bhuller

Description: 
"""
import argparse
import os
import Master
import Plots
from matplotlib import colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import awkward as ak
from tabulate import tabulate

def main():
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    def PrintStats(event):
        t, r, e = Master.CalculateQuantities(events)
        for i in range(ak.count(events.eventNum)):
            if i > 100: break
            print(f"subrun {events.subRun[i]}, event {events.eventNum[i]}")
            print(tabulate([["truth", *t[:, i]], ["reco", *r[:, i]], ["error", *e[:, i]]], headers=["quantity", *names], tablefmt="fancy_grid"))


    events = Master.Data(file)

    hasBeam = events.recoParticles.beam_number != -999
    t_q = events.trueParticles.CalculatePhotonPairProperties()

    # get nTuples not automatically retrieved (set as class variables so filtering is possible)
    events.recoParticles.pandoraID = events.io.Get("pandoraTag")
    events.recoParticles.spacePoints = ak.zip({"x" : events.io.Get("reco_daughter_allShower_spacePointX"), 
                                            "y" : events.io.Get("reco_daughter_allShower_spacePointY"),
                                            "z" : events.io.Get("reco_daughter_allShower_spacePointZ")})

    # apply filters before making event displays if needed
    events.ApplyBeamFilter()

    # plotting styles
    marker_size=5
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors) # change the default colour cycle to one with more variety

    nEvents = ak.count(events.eventNum)
    # make event display for each event
    for n in range(nEvents):
        print(f"plotting : {n}")
        
        # data for table
        nPFP = ak.count(events.recoParticles.nHits[n])
        nTrueParticle = ak.count(events.trueParticles.number[n])
        nTrack = ak.count(events.recoParticles.pandoraID[n][events.recoParticles.pandoraID[n] == 13])
        nShower = ak.count(events.recoParticles.pandoraID[n][events.recoParticles.pandoraID[n] == 11])
        nNull = ak.count(events.recoParticles.pandoraID[n][events.recoParticles.pandoraID[n] == -999])

        # create figure
        plt.figure(figsize=(6.4*2, 4.8*2.7), dpi=80*2) #? customisable?
        G = gridspec.GridSpec(2, 1) # create gidspace of three plots, one for each orthographic view, third one for the table
        xz = plt.subplot(G[0])
        yz = plt.subplot(G[1])

        # loop through each PFP, plot hit spacepoints (collection) in each orthographic view
        # resemles the orthographic event displays made in LArSoft
        for i in range(nPFP):
            startPos = events.recoParticles.startPos[n][i]
            if startPos.x == -999: continue # ensure the PFP has a valid start point to plot

            # don't plot spacepoints which have invalid positions
            points = events.recoParticles.spacePoints[n][i]
            points = points[points.x != -999]
            
            # xz display
            xz.scatter(points.z, points.x, marker_size, color=f"C{i}")
            plotPos = (events.recoParticles.startPos.z[n][i], events.recoParticles.startPos.x[n][i])
            xz.scatter(plotPos[0], plotPos[1], marker_size*10, marker="x", color=f"C{i}")
            xz.annotate(i, (plotPos[0]-2, plotPos[1]+1))

            # yz display
            yz.scatter(points.z, points.y, marker_size, color=f"C{i}")
            plotPos = (events.recoParticles.startPos.z[n][i], events.recoParticles.startPos.y[n][i])
            yz.scatter(plotPos[0], plotPos[1], marker_size*10, marker="x", color=f"C{i}")
            yz.annotate(i, (plotPos[0]-2, plotPos[1]+1))

        # plot true particles
        for i in range(nTrueParticle):
            startPos = events.trueParticles.startPos[n][i]
            endPos = events.trueParticles.endPos[n][i]
            xz.plot((startPos.z, endPos.z), (startPos.x, endPos.x), marker="o")
            xz.annotate(events.trueParticles.pdg[n][i], (endPos.z-2, endPos.x+1), color="red")

            yz.plot((startPos.z, endPos.z), (startPos.y, endPos.y), marker="o")
            yz.annotate(events.trueParticles.pdg[n][i], (endPos.z-2, endPos.y+1), color="red")

        # plot labels
        xz.grid()
        xz.set_xlabel("z (cm)")
        xz.set_ylabel("x (cm)")
        xz.set_title(f"subrun: {events.subRun[n]}, event: {events.eventNum[n]}")
        
        yz.grid()
        yz.set_xlabel("z (cm)")
        yz.set_ylabel("y (cm)")

        # create table
        t = [
            ["number of reconstructed PFOs", nPFP],
            ["contains beam", hasBeam[n]],
            ["beam particle", events.recoParticles.beam_number[n]],
            ["number of tracks (Pandora ID)", nTrack],
            ["number of showers (Pandora ID)", nShower],
            ["no Pandora ID", nNull],
            ["opening angle (rad)", ak.to_list(t_q[1][n])[0]],
            ["leading energy (GeV)", ak.to_list(t_q[2][n])[0]],
            ["sub leading energy (GeV)", ak.to_list(t_q[3][n])[0]],
            ["$\pi^{0}$ momentum (GeV)", ak.to_list(t_q[4][n])[0]]
        ]
        #yz.patch.set_visible(False)
        #yz.axis('tight')
        yz.table(t, bbox=[0.0,-0.5,1,.28])
        plt.tight_layout()

        os.makedirs("eventDisplays/", exist_ok=True)
        Plots.Save(f"{events.subRun[n]}_{events.eventNum[n]}", "eventDisplays/")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot quantities to study shower reconstruction")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    args = parser.parse_args() #! run in command line

    file = args.file
    main()