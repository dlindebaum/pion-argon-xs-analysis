"""
Created on: 08/02/2022 17:00

Author: Shyam Bhuller

Description: Compare results of different filters for Pi0MC. 
"""

import argparse
import os
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
# custom modules
import Plots
import Master
import vector

def Separation(shower : ak.Record, photon : ak.Record, null_shower_dir : ak.Record, typeof : str):
    """ Calculate angular or spatial separation,
        accounting for null direction vectors.

    Args:
        shower (ak.Record): shower direction/position
        photon (ak.Record): photon direction/position
        typeof (str): "Angular" or "Spatial"

    Returns:
        ak.Array: separation
    """
    if typeof == "Angular":
        s = vector.angle(shower, photon)
    if typeof == "Spatial":
        s = vector.dist(shower, photon)
    # if direction is null, set separation to masssive value i.e. it is never matched
    # otherwise assign the separation
    return ak.where(null_shower_dir == True, 1E8, s)


@Master.timer
def GetMCMatchingFilters(events : Master.Data, cut=0.25):
    """ Matches Reconstructed showers to true photons and selected the best events
        i.e. ones which have both errors of less than 0.25 radians. Only works for
        events with two reconstructed showers and two true photons per event.

    Args:
        photon_dir (ak.Record created by vector): direction of true photons
        shower_dir (ak.Record created by vector): direction of reco showers

    Returns:
        ak.Array: shower indices in order of true photon number
        ak.Array: boolean mask of showers not matched
        ak.Array: mask which indicates which pi0 decays are "good"
        ak.Array: minimum angle between showers and each true photon
        ak.Array: minimum distance between showers and each true photon
    """
    null_shower_dir = events.recoParticles.direction.x == -999 # keep track of showers which don't have a valid direction vector
    # angle of all reco showers wrt to each true photon per event i.e. error
    photon_dir = vector.normalize(events.trueParticles.momentum)[events.trueParticles.truePhotonMask]
    angle_error_0 = Separation(events.recoParticles.direction, photon_dir[:, 0], null_shower_dir, "Angular")
    angle_error_1 = Separation(events.recoParticles.direction, photon_dir[:, 1], null_shower_dir, "Angular")
    ind = ak.sort(ak.argsort(angle_error_0, -1), -1) # create array of indices to keep track of showers

    # get smallest spatial separation wrt to each true photon
    photon_pos = events.trueParticles.endPos[events.trueParticles.truePhotonMask]
    dist_error_0 = Separation(events.recoParticles.startPos, photon_pos[:, 0], null_shower_dir, "Spatial")
    dist_error_1 = Separation(events.recoParticles.startPos, photon_pos[:, 1], null_shower_dir, "Spatial")
    m_0 = ak.unflatten(ak.min(dist_error_0, -1), 1)
    m_1 = ak.unflatten(ak.min(dist_error_1, -1), 1)
    dists = ak.concatenate([m_0, m_1], -1)

    # get smallest angle wrt to each true photon
    m_0 = ak.unflatten(ak.min(angle_error_0, -1), 1)
    m_1 = ak.unflatten(ak.min(angle_error_1, -1), 1)
    angles = ak.concatenate([m_0, m_1], -1)

    first_matched_photon = ak.argmin(angles, -1, keepdims=True)
    remaining_photon = ak.where(first_matched_photon == 0, 1, 0)
    m_0 = ind[m_0 == angle_error_0]
    m_1 = ind[m_1 == angle_error_1]
    first_matched_shower = ak.where(first_matched_photon == 0, m_0, m_1)

    already_matched_0 = ak.flatten(angle_error_0[first_matched_shower]) != angle_error_0
    already_matched_1 = ak.flatten(angle_error_1[first_matched_shower]) != angle_error_1

    new_angle_error_0 = angle_error_0[already_matched_0]
    new_angle_error_1 = angle_error_1[already_matched_1]

    m_0 = ak.unflatten(ak.min(new_angle_error_0, -1), 1)
    m_1 = ak.unflatten(ak.min(new_angle_error_1, -1), 1)

    m_0 = ind[m_0 == angle_error_0]
    m_1 = ind[m_1 == angle_error_1]
    second_matched_shower = ak.where(remaining_photon == 0, m_0, m_1)

    photon_order = ak.concatenate([first_matched_photon, remaining_photon], -1)
    photon_order_sort = ak.argsort(photon_order, -1)
    matched_mask = ak.concatenate([first_matched_shower, second_matched_shower], -1)
    matched_mask = matched_mask[photon_order_sort]

    t_0 = matched_mask[:, 0] == ind
    t_1 = matched_mask[:, 1] == ind
    unmatched_mask = np.logical_not(np.logical_or(t_0, t_1))

    # get events where both reco MC angles are less than 0.25 radians
    angles_0 = vector.angle(events.recoParticles.direction[matched_mask][:, 0], photon_dir[:, 0])
    angles_1 = vector.angle(events.recoParticles.direction[matched_mask][:, 1], photon_dir[:, 1])
    selection = np.logical_and(angles_0 < cut, angles_1 < cut)

    # check how many showers had the same reco match to both true particles
    same_match = matched_mask[:, 0][selection] == matched_mask[:, 1][selection]
    same_match_percentage = 100 * ak.count(same_match[same_match]) / ak.count(same_match)
    print(f"number of events after selection: {ak.count(same_match)}")
    print(f"number of events where both photons match to the same shower: {ak.count(same_match[same_match])}")
    print(f"percantage of events where both photons match to the same shower: {same_match_percentage:.3f}")

    return matched_mask, unmatched_mask, selection, angles, dists, same_match_percentage

def SpatialStudy(events : Master.Data):
    valid = Master.Pi0MCMask(events, None)

    filtered = events.Filter([valid], [valid])
    null_shower_dir = filtered.recoParticles.direction.x == -999
    photon_pos = filtered.trueParticles.endPos[filtered.trueParticles.truePhotonMask]
    dist_error_0 = Separation(filtered.recoParticles.startPos, photon_pos[:, 0], null_shower_dir, "Spatial")
    dist_error_1 = Separation(filtered.recoParticles.startPos, photon_pos[:, 1], null_shower_dir, "Spatial")
    m_0 = ak.unflatten(ak.argmin(dist_error_0, -1), 1)
    m_1 = ak.unflatten(ak.argmin(dist_error_1, -1), 1)
    closest = ak.concatenate([m_0, m_1], -1)
    closest_start_pos = filtered.recoParticles.startPos[closest]
    d_x_0 = closest_start_pos.x[:, 0] - photon_pos.x[:, 0]
    d_x_1 = closest_start_pos.x[:, 1] - photon_pos.x[:, 1]
    d_y_0 = closest_start_pos.y[:, 0] - photon_pos.y[:, 0]
    d_y_1 = closest_start_pos.y[:, 1] - photon_pos.y[:, 1]
    d_z_0 = closest_start_pos.z[:, 0] - photon_pos.z[:, 0]
    d_z_1 = closest_start_pos.z[:, 1] - photon_pos.z[:, 1]
    Plots.PlotHistComparison([d_x_0, d_x_1], [-50, 50], bins=bins, xlabel="x separation of closest reco shower and true photon (cm)", labels=["photon 0", "photon 1"])
    if save is True: Plots.Save(outDir+"separation_x")
    Plots.PlotHistComparison([d_y_0, d_y_1], [-50, 50], bins=bins, xlabel="y separation of closest reco shower and true photon (cm)", labels=["photon 0", "photon 1"])
    if save is True: Plots.Save(outDir+"separation_y")
    Plots.PlotHistComparison([d_z_0, d_z_1], [-75, 75], bins=bins, xlabel="z separation of closest reco shower and true photon (cm)", labels=["photon 0", "photon 1"])
    if save is True: Plots.Save(outDir+"separation_z")


def CreateFilteredEvents(events : Master.Data, nDaughters=None, cut : int = 0.25):
    valid = Master.Pi0MCMask(events, nDaughters)

    filtered = events.Filter([valid], [valid])

    print(f"Number of events: {ak.num(filtered.recoParticles.direction, 0)}")
    showers, _, selection_mask, _, _, _ = GetMCMatchingFilters(filtered, cut)

    reco_filters = [showers, selection_mask]
    true_filters = [selection_mask]

    return filtered.Filter(reco_filters, true_filters)


def AnalyseMatching(events : Master.Data, nDaughters=None, cut : int = 0.25, title : str = ""):
    valid = Master.Pi0MCMask(events, nDaughters)

    filtered = events.Filter([valid], [valid])

    print(f"Number of events: {ak.num(filtered.recoParticles.direction, 0)}")
    matched, _, selection, angles, dists, percentage = GetMCMatchingFilters(filtered, cut)

    reco_filters = [matched, selection]
    true_filters = [selection]
    filtered.Filter(reco_filters, true_filters, returnCopy=False)

    reco_mc_dist = ak.ravel(vector.dist(filtered.recoParticles.startPos, filtered.trueParticles.endPos[filtered.trueParticles.truePhotonMask]))
    reco_mc_angle = ak.ravel(vector.angle(filtered.recoParticles.direction, filtered.trueParticles.direction[filtered.trueParticles.truePhotonMask]))
    
    return dists, angles, reco_mc_dist, reco_mc_angle, percentage


def Plot1D(data : ak.Array, xlabels : list, subDir : str, plot_ranges = [[]]*5, legend_loc = ["upper right"]*5):
    """ 1D histograms of data for each sample

    Args:
        data (ak.Array): list of samples data to plot
        xlabels (list): x labels
        subDir (str): subdirectiory to save in
        plot_range (list, optional): range to plot. Defaults to [].
    """
    if save is True: os.makedirs(outDir + subDir, exist_ok=True)
    for i in range(len(names)):
        Plots.PlotHistComparison(data[:, i], plot_ranges[i], bins=bins, xlabel=xlabels[i], histtype="step", labels=s_l, density=True)
        plt.legend(loc=legend_loc[i])
        if save is True: Plots.Save( names[i] , outDir + subDir)


def Pi0MomFilter(events : Master.Data, r : list = [0.5, 1]):
    if len(r) != 2: r = [0.5, 1]
    pi0 = events.trueParticles.pdg == 111
    pi0 = np.logical_and(pi0, events.trueParticles.number == 1)
    pi0_mom = vector.magntiude(events.trueParticles.momentum)[pi0]
    mask = np.logical_and(pi0_mom > r[0], pi0_mom < r[1])
    mask = ak.flatten(mask)
    return events.Filter([mask], [mask])

@Master.timer
def main():
    global x, m, ranges
    events = Master.Data(file)
    events.ApplyBeamFilter() # apply beam filter if possible
    ts = []
    rs = []
    es = []

    angles = []
    dists = []
    rmas = []
    rmds = []

    if ana == "matching":
        saveDir = outDir+"matching/"
        if save is True: os.makedirs(saveDir, exist_ok=True)
        if binned_analysis is True:
            ranges = np.arange(0.5, 7.5, 0.5)
            x = []
            m = []
            for i in range(len(ranges)-1):
                b = [ranges[i], ranges[i+1]]
                x.append(str(b))
                binned_events = Pi0MomFilter(events, b)
                d, a, rmd, rma, same_match = AnalyseMatching(binned_events, None, 0.25, f"{b} GeV")
                angles.append(a)
                dists.append(d)
                rmas.append(rma)
                rmds.append(rmd)
                m.append(same_match)

            angles = ak.Array(angles)
            dists = ak.Array(dists)
            if save is True: os.makedirs(saveDir+"binned/", exist_ok=True)
            Plots.Plot(x, m, xlabel="True pi0 momentum bins (GeV)", ylabel="Percentage of events with same matched shower")
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save is True: Plots.Save("same_match", saveDir+"binned/")
            Plots.PlotHistComparison(angles[:, :, 0], [], bins, "Angular separation between closest shower and true photon 1 (rad)", labels=x)
            if save is True: Plots.Save("angle_dist_1", saveDir+"binned/")
            Plots.PlotHistComparison(angles[:, :, 1], [], bins, "Angular separation between closest shower and true photon 2 (rad)", labels=x)
            if save is True: Plots.Save("angle_dist_2", saveDir+"binned/")
            Plots.PlotHistComparison(dists[:, :, 0], [0, 150], bins, "Spatial separation between closest shower and true photon 1 (cm)", labels=x)
            if save is True: Plots.Save("spatial_dist_1", saveDir+"binned/")
            Plots.PlotHistComparison(dists[:, :, 1], [0, 150], bins, "Spatial separation between closest shower and true photon 2 (cm)", labels=x)
            if save is True: Plots.Save("spatial_dist_2", saveDir+"binned/")
            Plots.PlotHistComparison(rmas, [], bins, "angular separation between shower and matched photon (cm)", labels=x)
            if save is True: Plots.Save("reco_mc_angle", saveDir+"binned/")
            Plots.PlotHistComparison(rmds, [0, 150], bins, "spatial separation between shower and matched photon (cm)", labels=x)
            if save is True: Plots.Save("reco_mc_dist", saveDir+"binned/")
        else:
            SpatialStudy(events)
            d, a, rmd, rma, same_match = AnalyseMatching(events)
            print(f"percentage of events with photons matched to the same shower: {same_match:.3f}")
            Plots.PlotHistComparison([ak.ravel(a[:, 0]), ak.ravel(a[:, 1])], bins=bins, xlabel="Angular separation between closest shower and true photon (rad)", labels=["photon 1", "photon_2"])
            if save is True: Plots.Save("angle_dist", outDir+"matching/")
            Plots.PlotHistComparison([ak.ravel(d[:, 0]), ak.ravel(d[:, 1])], [0, 150], bins=bins, xlabel="Spatial separation between closest shower and true photon (cm)", labels=["photon 1", "photon_2"])
            if save is True: Plots.Save("spatial_dist", outDir+"matching/")
            Plots.PlotHist(rma, bins, "angular separation between matched shower and photon (rad)")
            if save is True: Plots.Save("reco_mc_angle", outDir+"matching/")
            Plots.PlotHist(rmd[rmd < 150], bins, "spatial separation between matched shower and photon (cm)")
            if save is True: Plots.Save("reco_mc_dist", outDir+"matching/")
        

    if ana == "quantity":
        for i in range(len(filters)):
            filtered = CreateFilteredEvents(events, filters[i])
            t, r, e = Master.CalculateQuantities(filtered, names)
            ts.append(t)
            rs.append(r)
            es.append(e)

        ts = ak.Array(ts)
        rs = ak.Array(rs)
        es = ak.Array(es)
        Plot1D(ts, t_l, "truth/", t_range, t_locs)
        Plot1D(rs, r_l, "reco/", r_range, r_locs)
        Plot1D(es, e_l, "fractional_error/", e_range, e_locs)

        if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
        plt.rcParams["figure.figsize"] = (6.4*3,4.8*1)
        for j in range(len(names)):
            plt.figure()
            for i in range(len(filters)):
                plt.subplot(1, 3, i+1)
                if i == 0:
                    _, edges = Plots.PlotHist2D(ts[i][j], es[i][j], bins, x_range=t_range[j], y_range=e_range[j], xlabel=t_l[j], ylabel=e_l[j], title=s_l[i], newFigure=False)
                else:
                    Plots.PlotHist2D(ts[i][j], es[i][j], edges, x_range=t_range[j], y_range=e_range[j], xlabel=t_l[j], ylabel=e_l[j], title=s_l[i], newFigure=False)
            if save is True: Plots.Save( names[j] , outDir + "2D/")
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


if __name__ == "__main__":
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    # plot labels
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading photon energy (GeV)", "True Sub leading photon energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    e_l = ["Invariant mass fractional error", "Opening angle fractional error", "Leading shower energy fractional error", "Sub leading shower energy fractional error", "$\pi^{0}$ momentum fractional error"]
    r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Subleading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]
    # plot ranges, order is invariant mass, angle, lead energy, sub energy, pi0 momentum
    e_range = [[-10, 10]] * 5
    r_range = [[]] * 5
    r_range[0] = [0, 0.5]
    t_range = [[]] * 5
    # legend location, order is invariant mass, angle, lead energy, sub energy, pi0 momentum
    r_locs = ["upper right", "upper right", "upper right", "upper right", "upper right"]
    t_locs = ["upper left", "upper right", "upper right", "upper right", "upper right"]
    e_locs = ["upper right", "upper right", "upper right", "upper right", "upper left"]

    filters = [2, 3, -3]
    s_l = ["2 daughters", "3 daughters", "> 3 daughters"]

    parser = argparse.ArgumentParser(description="Study shower-photon matching for pi0 decays.")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms.")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="pi0_0p5GeV_100K/matching_study/", help="directory to save plots.")
    parser.add_argument("-a", "--analysis", dest="ana", type=str, choices=["matching", "quantity"], default="quantity", help="what analysis to run.")
    parser.add_argument("--binned", dest="binned_analysis", action="store_true", help="do analysis binned in true pi0 momentum.")
    #args = parser.parse_args("ROOTFiles/pi0_multi_9_3_22.root -a matching".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    ana = args.ana
    outDir = args.outDir
    binned_analysis = args.binned_analysis
    main()
