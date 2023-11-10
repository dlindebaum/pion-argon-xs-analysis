"""
Created on: 15/02/2022 17:37

Author: Shyam Bhuller

Description: A script studying pi0 decay geometry and shower merging.
"""
import argparse
import os

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# custom modules
from python.analysis import Master, Plots, vector


def Separation(shower_1 : ak.Record, shower_2 : ak.Record, null : ak.Record, typeof : str) -> ak.Array:
    """ Calculate angular or spatial separation, acocunting for null values

    Args:
        shower_1 (ak.Record): first shower
        shower_2 (ak.Record): second shower
        null (ak.Record): boolean mask of null vectors
        typeof (str): "Angular" or "Spatial"

    Returns:
        ak.Array: separation
    """
    if typeof == "Angular":
        s = vector.angle(shower_1, shower_2)
    if typeof == "Spatial":
        s = vector.dist(shower_1, shower_2)
    # if direction is null, set separation to masssive value i.e. it is never matched
    # otherwise assign the separation
    return ak.where(null == True, 1E8, s)


def AnalyzeReco(events : Master.Data, matched : ak.Array, unmatched : ak.Array):
    """ Study relationships between angles and distances between matched and unmatched showers.

    Args:
        events (Master.Event): events to study
        matched (ak.Array): indicies of matched showers
        unmatched (ak.Array): boolean mask of unmatched showers
    """
    matched_reco = events.Filter([matched], returnCopy=True).recoParticles # filter reco for matched/unmatched only
    unmatched_reco = events.Filter([unmatched], returnCopy=True).recoParticles
    null_dir = unmatched_reco.direction.x == -999 # should only be needed for unmatched sample
    valid = np.logical_not(null_dir)

    #* calculate separation of matched to unmatched
    separation_0 = Separation(unmatched_reco.shower_start_pos, matched_reco.shower_start_pos[:, 0], null_dir, "Spatial")[valid]
    separation_1 = Separation(unmatched_reco.shower_start_pos, matched_reco.shower_start_pos[:, 1], null_dir, "Spatial")[valid]
    separation = ak.concatenate([separation_0, separation_1], -1)
    minMask_dist = ak.min(separation, -1) == separation # get closest matched shower to matched to study various combinations

    #* same as above but for angular distance
    angle_0 = Separation(unmatched_reco.direction, matched_reco.direction[:, 0], null_dir, "Angular")[valid]
    angle_1 = Separation(unmatched_reco.direction, matched_reco.direction[:, 1], null_dir, "Angular")[valid]
    angle = ak.concatenate([angle_0, angle_1], -1)
    minMask_angle = ak.min(angle, -1) == angle

    #* get various combinations of distances and angles to look at
    min_separation_by_dist = separation[minMask_dist]
    min_separation_by_angle = separation[minMask_angle]
    min_angle_by_dist = angle[minMask_dist]
    min_angle_by_angle = angle[minMask_angle]

    #* ravel for plotting
    separation_0 = ak.ravel(separation_0)
    separation_1 = ak.ravel(separation_1)
    angle_0 = ak.ravel(angle_0)
    angle_1 = ak.ravel(angle_1)
    min_separation_by_dist = ak.ravel(min_separation_by_dist)
    min_separation_by_angle = ak.ravel(min_separation_by_angle)
    min_angle_by_dist = ak.ravel(min_angle_by_dist)
    min_angle_by_angle = ak.ravel(min_angle_by_angle)

    #* plots
    directory = outDir + "merging/"
    if save is True: os.makedirs(directory, exist_ok=True)
    
    Plots.PlotHistComparison([separation_0, separation_1[separation_1 < 300]], bins=bins, xlabel="Spatial separation between matched showers and shower 2 (cm)", labels=["shower 0", "shower 1"])
    if save is True: Plots.Save("spatial", directory)
    Plots.PlotHistComparison([angle_0, angle_1], bins=bins, xlabel="Angular separation between matched showers and shower 2 (rad)", labels=["shower 0", "shower 1"])
    if save is True: Plots.Save("angular", directory)
    
    plt.rcParams["figure.figsize"] = (6.4*2,4.8)
    plt.figure()
    plt.subplot(1, 2, 1)
    _, edges = Plots.PlotHist2D(separation_0, angle_0, bins, xlabel="Spatial separation between shower 0 and shower 2 (cm)", ylabel="Angular separation between shower 0 and shower 2 (rad)", newFigure=False)
    plt.subplot(1, 2, 2)
    Plots.PlotHist2D(separation_1, angle_1, edges, xlabel="Spatial separation between shower 1 and shower 2 (cm)", ylabel="Angular separation between shower 1 and shower 2 (rad)", newFigure=False)
    if save is True: Plots.Save( "spatial_vs_anglular" , directory)
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

    min_spatial_l = "Spatial separation between closest matched shower and shower 2 (cm)"
    min_angular_l = "Angular separation between closest shower and shower 2 (cm)"
    merge_dist_l = "merge by distance"
    merge_angle_l = "merge by angle"

    Plots.PlotHistComparison([min_separation_by_dist, min_separation_by_angle], bins=bins, xlabel=min_spatial_l, labels=[merge_dist_l, merge_angle_l])
    if save is True: Plots.Save("min_spatial" , directory)
    Plots.PlotHistComparison([min_angle_by_dist, min_angle_by_angle], bins=bins, xlabel=min_angular_l, labels=[merge_dist_l, merge_angle_l])
    if save is True: Plots.Save("min_angle" , directory)

    plt.rcParams["figure.figsize"] = (6.4*2,4.8)
    plt.figure()
    plt.subplot(1, 2, 1)
    _, edges = Plots.PlotHist2D(min_separation_by_dist, min_angle_by_dist, bins, xlabel=min_spatial_l, ylabel=min_angular_l, title=merge_dist_l, newFigure=False)
    plt.subplot(1, 2, 2)
    Plots.PlotHist2D(min_separation_by_angle, min_angle_by_angle, edges, xlabel=min_spatial_l, ylabel=min_angular_l, title=merge_angle_l, newFigure=False)
    if save is True: Plots.Save( "spatial_vs_anglular" , directory)
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def AnalyzeTruth(events : Master.Data):
    """ Plot distances of true particles wrt to eachother + the decay vertex to gauge size of pi0 decays

    Args:
        events (Master.Events): events to look at
    """
    pi0_vertex = events.Filter(true_filters=[ak.to_list(events.trueParticles.number == 1)], returnCopy=True).trueParticles.endPos # decay vertex is where the particle trajectory ends

    photons_true = events.Filter(true_filters=[events.trueParticles.truePhotonMask], returnCopy=True).trueParticles

    #* distance from each photon end point to the deacy vertex
    dist_to_vertex_0 = vector.dist(pi0_vertex, photons_true.endPos[:, 0]) # photon end points are the start of showers
    dist_to_vertex_1 = vector.dist(pi0_vertex, photons_true.endPos[:, 1])

    #* distance between each photon end point
    true_shower_separation = vector.dist(photons_true.endPos[:, 0], photons_true.endPos[:, 1])

    #* plots
    if save is True: os.makedirs(outDir + "truth/", exist_ok=True)
    Plots.PlotHist(ak.ravel(dist_to_vertex_0), bins, "Distance from photon 0 end point to decay vertex (cm)")
    if save is True: Plots.Save("dist_vertex_0" , outDir + "truth/")
    Plots.PlotHist(ak.ravel(dist_to_vertex_1), bins, "Distance from photon 1 end point to decay vertex (cm)")
    if save is True: Plots.Save("dist_vertex_1" , outDir + "truth/")
    Plots.PlotHist(true_shower_separation, bins, "True photon separation (cm)")
    if save is True: Plots.Save("separation" , outDir + "truth/")


def MakePlots(dist : ak.Array, angle : ak.Array, dist_label : str, angle_label : str, subdirectory : str, title : str = None):
    """ Make plots of distance, angle and distance vs angle.

    Args:
        dist (ak.Array): distance between two showers
        angle (ak.Array): angle between two showers
        dist_label (str): distance plot label
        angle_label (str): angle plot label
        subdirectory (str): output subdirectory
    """
    _dir = outDir + subdirectory
    if save is True: os.makedirs(_dir, exist_ok=True)
    Plots.PlotHist(dist, bins, dist_label, title=title)
    if save is True: Plots.Save( "distance" , _dir)
    Plots.PlotHist(angle, bins, angle_label, title=title)
    if save is True: Plots.Save( "angle" , _dir)
    Plots.PlotHist2D(dist, angle, bins, x_range=[0, 150], xlabel=dist_label, ylabel=angle_label, title=title)
    if save is True: Plots.Save( "2D" , _dir)


def MergeQuantity(matched : ak.Array, unmatched : ak.Array, mask : ak.Array, type : str) -> ak.Array:
    """ Merge a shower quantity

    Args:
        matched (ak.Array): start showers
        unmatched (ak.Array): PFOs to merge
        mask (ak.Array): mask of events to not merge
        type (str): type of quantity we are merging

    Returns:
        ak.Array: merged quantity
    """
    if type == "Vector3":
        null = {"x": 0, "y": 0, "z": 0}
    elif type == "Scalar":
        null = 0
    else:
        raise Exception(f"{type} not a mergable type")
    toMerge = ak.where(mask, unmatched, null)
    if type == "Vector3":
         toMerge = ak.where(toMerge.x != -999, toMerge, null)
    if type == "Scalar":
         toMerge = ak.where(toMerge != -999, toMerge, null)
    merge_0 = toMerge[:, :, 0]
    merge_1 = toMerge[:, :, 1]
    merge_0 = ak.sum(merge_0, -1)
    merge_1 = ak.sum(merge_1, -1)
    merge_0 = ak.unflatten(merge_0, 1, -1)
    merge_1 = ak.unflatten(merge_1, 1, -1)
    merge = ak.concatenate([merge_0, merge_1], -1)
    if type == "Vector3":
         merge = ak.where(matched.x != -999, merge, null)
    if type == "Scalar":
         merge = ak.where(matched != -999, merge, null)
    return merge

@Master.timer
def mergeShower(events : Master.Data, matched : ak.Array, unmatched : ak.Array, mergeMethod : int = 1, energyScalarSum : bool = False):
    """ Merge shower not matched to MC to the spatially closest matched shower.

    Args:
        events (Master.Event): events to study
        matched (ak.Array): matched shower indicies
        unmatched (ak.Array): boolean mask of unmatched showers
        mergeMethod (int): method 1 merges by closest angular distance, method 2 merges by closest spatial distance
        energyScalarSum (bool): False does a sum of momenta, then magnitude, True does magnitude of momenta, then sum

    Returns:
        Master.Events: events with matched reco showers after merging
    """
    events_matched = events.Filter([matched], returnCopy=True)
    unmatched_reco = events.Filter([unmatched], returnCopy=True).recoParticles # filter reco for matched/unmatched only
    null_dir = unmatched_reco.direction.x == -999 # should only be needed for unmatched sample

    if mergeMethod == 2:
        #* distance from each matched to unmatched
        separation_0 = Separation(unmatched_reco.shower_start_pos, events_matched.recoParticles.shower_start_pos[:, 0], null_dir, "Spatial")
        separation_1 = Separation(unmatched_reco.shower_start_pos, events_matched.recoParticles.shower_start_pos[:, 1], null_dir, "Spatial")
        separation_0 = ak.unflatten(separation_0, 1, -1)
        separation_1 = ak.unflatten(separation_1, 1, -1)
        separation = ak.concatenate([separation_0, separation_1], -1)
        mergeMask = ak.min(separation, -1) == separation # get boolean mask to which matched shower to merge to

    if mergeMethod == 1:        
        angle_0 = Separation(unmatched_reco.direction, events_matched.recoParticles.direction[:, 0], null_dir, "Angular")
        angle_1 = Separation(unmatched_reco.direction, events_matched.recoParticles.direction[:, 1], null_dir, "Angular")
        angle_0 = ak.unflatten(angle_0, 1, -1)
        angle_1 = ak.unflatten(angle_1, 1, -1)
        angle = ak.concatenate([angle_0, angle_1], -1)
        mergeMask = ak.min(angle, -1) == angle

    #* create Array which contains the amount of energy to merge to the showers
    #* will be zero for the shower we don't want to merge to
    momentumToMerge = MergeQuantity(events_matched.recoParticles.momentum, unmatched_reco.momentum, mergeMask, "Vector3")
    new_momentum = vector.add(events_matched.recoParticles.momentum, momentumToMerge)
    events_matched.recoParticles._RecoParticleData__momentum = new_momentum

    new_direction = vector.normalize(events_matched.recoParticles.momentum)
    new_direction = ak.where(events_matched.recoParticles.momentum.x != -999, new_direction, {"x": -999, "y": -999, "z": -999})
    events_matched.recoParticles._RecoParticleData__direction = new_direction

    if energyScalarSum is True:
        energyToMerge = MergeQuantity(events_matched.recoParticles.energy, unmatched_reco.energy, mergeMask, "Scalar")
        events_matched.recoParticles._RecoParticleData__energy = events_matched.recoParticles.energy + energyToMerge # merge energies
        events_matched.recoParticles._RecoParticleData__momentum = vector.prod(events_matched.recoParticles.energy, events_matched.recoParticles.direction)
    else:
        new_energy = vector.magnitude(events_matched.recoParticles.momentum)
        events_matched.recoParticles._RecoParticleData__energy = ak.where(events_matched.recoParticles.momentum.x != -999, new_energy, -999)

    return events_matched


def CreateFilteredEvents(events : Master.Data, nDaughters : int = None):
    """ Filter events with specific number of daughters, then match the showers to
       MC truth.

    Args:
        events (Master.Event): events to study
        nDaughters (int, optional): filter events with ndaughters. Defaults to None.

    Returns:
        Master.Event: events after filering 
    """
    valid = Master.Pi0MCMask(events, nDaughters)
    filtered = events.Filter([valid], [valid], returnCopy=True)
    print(f"Number of showers events: {ak.num(filtered.recoParticles.direction, 0)}")

    filtered.MCMatching()
    return filtered


def Plot1D(data : ak.Array, xlabels : list, subDir : str, labels : list = [""]*5, plot_ranges : list = [[]]*5):
    """ 1D histograms of data for each sample

    Args:
        data (ak.Array): list of samples data to plot
        xlabels (list): x labels
        subDir (str): subdirectiory to save in
        plot_range (list, optional): range to plot. Defaults to [].
    """
    if save is True: os.makedirs(outDir + subDir, exist_ok=True)
    for i in range(len(names)):
        Plots.PlotHistComparison(data[:, i], plot_ranges[i], bins=bins, xlabel=xlabels[i], histtype="step", labels=labels, density=True)
        if save is True: Plots.Save( names[i] , outDir + subDir)


def AnalyseQuantities(truths : np.array, recos : np.array, errors : np.array, labels : list, directory : str):
    """ Plot calculated quantities for given events

    Args:
        truths (np.array): true quantities
        recos (np.array): reconstruced quantities
        errors (np.array): fractional errors
        labels (list): plot labels for different event types
        directory (str): output directory
    """
    #! fix this
    Plot1D(ak.Array(recos), r_l, "reco/", labels, r_range)
    Plot1D(ak.Array(errors), e_l, "fractional_error/", labels, fe_range)
    if save is True: os.makedirs(directory + "2D/", exist_ok=True)
    plt.rcParams["figure.figsize"] = (6.4*2,4.8*2)
    for j in range(len(names)):
        plt.figure()
        for i in range(len(labels)):
            plt.subplot(2, 2, i+1)
            if i == 0:
                _, edges = Plots.PlotHist2D(truths[i][j], errors[i][j], bins, y_range=fe_range[j], xlabel=t_l[j], ylabel=e_l[j], title=labels[i], newFigure=False)
            else:
                Plots.PlotHist2D(truths[i][j], errors[i][j], edges, y_range=fe_range[j], xlabel=t_l[j], ylabel=e_l[j], title=labels[i], newFigure=False)
        if save is True: Plots.Save( names[j] , directory + "2D/")
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def SelectSample(events : Master.Data, nDaughters : int) -> tuple:
    """ Perform an event selection and get starting showers

    Args:
        events (Master.Data): events to look at
        nDaughters (int): select events by number of PFOs 

    Returns:
        tuple: filtered events and start showers
    """
    valid = Master.Pi0MCMask(events, nDaughters)
    filtered = events.Filter([valid], [valid], returnCopy=True)
    singleMatchedEvents = filtered.trueParticlesBT.SingleMatch
    filtered.Filter([singleMatchedEvents], [singleMatchedEvents])
    best_match, selection = filtered.MatchByAngleBT()
    filtered.Filter([selection], [selection])
    best_match = best_match[selection]
    return filtered, best_match

@Master.timer
def main():
    events = Master.Data(file)
    events.ApplyBeamFilter() # apply beam filter if possible

    if study == "performance":
        
        energy_differences = []
        cm = []
        for i in range(len(n_obj)):
            sample, target_PFPs = SelectSample(events, n_obj[i])
            cm.append(Master.ShowerMergePerformance(sample, target_PFPs))
            merged_cheat, null = sample.MergePFPCheat()
            merged_bt = sample.MergeShowerBT(target_PFPs)
            energy_differences.append(ak.ravel(merged_bt.recoParticles.energy - merged_cheat.recoParticles.energy) / 1000)
        
        if save is True: os.makedirs(outDir, exist_ok=True)
        if len(n_obj) > 1:
            Plots.PlotHistComparison(energy_differences, [], bins, "merged shower energy difference (GeV)", labels=s_l, density=False)
            if save is True: Plots.Save(outDir + "energy_difference")
            Plots.Plot(s_l, cm, xlabel="sample", ylabel="percentage of showers correctly merged")
            if save is True: Plots.Save(outDir + "correct_merge")
        else:
            Plots.PlotHist(energy_differences[0], bins, "merged shower energy difference (GeV)")
            if save is True: Plots.Save(outDir + "energy_difference")

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading photon energy (GeV)", "True Sub leading photon energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    e_l = ["Invariant mass fractional error", "Opening angle fractional error", "Leading shower energy fractional error", "Sub leading shower energy fractional error", "$\pi^{0}$ momentum fractional error"]
    r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Subleading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]
    s_l = ["2 showers", "3 showers, unmerged", "angular vector sum", "spatial vector sum", "angular scalar sum", "spatial scalar sum"]
    fe_range = [[-10, 10]] * 5
    r_range = [[]] * 5
    r_range[0] = [0, 0.5]
    t_range = [[]] * 5

    n_obj = [-2] # nPFOs
    s_l = ["all"] # sample label

    parser = argparse.ArgumentParser(description="Study em shower merging for pi0 decays")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=25, help="number of bins when plotting histograms")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="pi0_0p5GeV_100K/shower_merge/", help="directory to save plots")
    parser.add_argument("-a", "--analysis", dest="study", type=str, choices=["performance"], default="performance", help="what plots we want to study")
    #args = parser.parse_args("work/ROOTFiles/pi0_0p5GeV_100K_5_7_21.root".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    outDir = args.outDir
    study = args.study
    main()
