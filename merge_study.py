"""
Created on: 15/02/2022 17:37

Author: Shyam Bhuller

Description: A script studying pi0 decay geometry and shower merging.
TODO Move shower merging algorithm to Master.Data
TODO optimise main()
TODO calculated quantities should be in ak.Array format
"""
import argparse
import os
import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# custom modules
import Plots
import Master
import vector

def Separation(shower_1 : ak.Record, shower_2 : ak.Record, null : ak.Record, typeof : str):
    """ Calculate angular or spatial separation, acocunting for null values

    Args:
        shower_1 (ak.Record): first shower
        shower_2 (ak.Record): second shower
        null (ak.Record): boolean mask of null vectors
        typeof (str): "Angular" or "Spatial"

    Returns:
        _type_: _description_
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
    matched_reco = events.Filter([matched]).recoParticles # filter reco for matched/unmatched only
    unmatched_reco = events.Filter([unmatched]).recoParticles
    null_dir = unmatched_reco.direction.x == -999 # should only be needed for unmatched sample
    valid = np.logical_not(null_dir)

    #* calculate separation of matched to unmatched
    #separation_0 = vector.dist(unmatched_reco.startPos, matched_reco.startPos[:, 0])
    #separation_1 = vector.dist(unmatched_reco.startPos, matched_reco.startPos[:, 1])
    separation_0 = Separation(unmatched_reco.startPos, matched_reco.startPos[:, 0], null_dir, "Spatial")[valid]
    separation_1 = Separation(unmatched_reco.startPos, matched_reco.startPos[:, 1], null_dir, "Spatial")[valid]
    separation = ak.concatenate([separation_0, separation_1], -1)
    minMask_dist = ak.min(separation, -1) == separation # get closest matched shower to matched to study various combinations

    #* same as above but for angular distance
    #angle_0 = vector.angle(unmatched_reco.direction, matched_reco.direction[:, 0])
    #angle_1 = vector.angle(unmatched_reco.direction, matched_reco.direction[:, 1])
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
    #separation = ak.ravel(vector.dist(matched_reco.startPos[:, 0], matched_reco.startPos[:, 1]))
    #opening_angle = ak.ravel(vector.dist(matched_reco.direction[:, 0], matched_reco.direction[:, 1]))

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
    pi0_vertex = events.Filter(true_filters=[ak.to_list(events.trueParticles.number == 1)]).trueParticles.endPos # decay vertex is where the particle trajectory ends

    photons_true = events.Filter(true_filters=[events.trueParticles.truePhotonMask]).trueParticles

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


def MergeQuantity(matched, unmatched, mask, type):
    if type == "Vector3":
        null = {"x": 0, "y": 0, "z": 0}
    elif type == "Scalar":
        null = 0
    else:
        # print error message
        print("not a mergable type")
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
    """Merge shower not matched to MC to the spatially closest matched shower.

    Args:
        events (Master.Event): events to study
        matched (ak.Array): matched shower indicies
        unmatched (ak.Array): boolean mask of unmatched showers
        mergeMethod (int): method 1 merges by closest angular distance, method 2 merges by closest spatial distance
        energyScalarSum (bool): False does a sum of momenta, then magnitude, True does magnitude of momenta, then sum

    Returns:
        Master.Events: events with matched reco showers after merging
    """
    events_matched = events.Filter([matched])
    unmatched_reco = events.Filter([unmatched]).recoParticles # filter reco for matched/unmatched only
    null_dir = unmatched_reco.direction.x == -999 # should only be needed for unmatched sample

    if mergeMethod == 2:
        #* distance from each matched to unmatched
        #separation_0 = vector.dist(unmatched_reco.startPos, events_matched.recoParticles.startPos[:, 0])
        #separation_1 = vector.dist(unmatched_reco.startPos, events_matched.recoParticles.startPos[:, 1])
        separation_0 = Separation(unmatched_reco.startPos, events_matched.recoParticles.startPos[:, 0], null_dir, "Spatial")
        separation_1 = Separation(unmatched_reco.startPos, events_matched.recoParticles.startPos[:, 1], null_dir, "Spatial")
        separation_0 = ak.unflatten(separation_0, 1, -1)
        separation_1 = ak.unflatten(separation_1, 1, -1)
        separation = ak.concatenate([separation_0, separation_1], -1)
        mergeMask = ak.min(separation, -1) == separation # get boolean mask to which matched shower to merge to

    if mergeMethod == 1:
        #angle_0 = vector.angle(unmatched_reco.direction, events_matched.recoParticles.direction[:, 0])
        #angle_1 = vector.angle(unmatched_reco.direction, events_matched.recoParticles.direction[:, 1])        
        angle_0 = Separation(unmatched_reco.direction, events_matched.recoParticles.direction[:, 0], null_dir, "Angular")
        angle_1 = Separation(unmatched_reco.direction, events_matched.recoParticles.direction[:, 1], null_dir, "Angular")
        angle_0 = ak.unflatten(angle_0, 1, -1)
        angle_1 = ak.unflatten(angle_1, 1, -1)
        angle = ak.concatenate([angle_0, angle_1], -1)
        mergeMask = ak.min(angle, -1) == angle

    #* create Array which contains the amount of energy to merge to the showers
    #* will be zero for the shower we don't want to merge to
    momentumToMerge = MergeQuantity(events_matched.recoParticles.momentum, unmatched_reco.momentum, mergeMask, "Vector3")
    events_matched.recoParticles.momentum = vector.Add(events_matched.recoParticles.momentum, momentumToMerge)

    new_direction = vector.normalize(events_matched.recoParticles.momentum)
    events_matched.recoParticles.direction = ak.where(events_matched.recoParticles.momentum.x != -999, new_direction, {"x": -999, "y": -999, "z": -999})

    if energyScalarSum is True:
        energyToMerge = MergeQuantity(events_matched.recoParticles.energy, unmatched_reco.energy, mergeMask, "Scalar")
        events_matched.recoParticles.energy = events_matched.recoParticles.energy + energyToMerge # merge energies
        events_matched.recoParticles.momentum = vector.prod(events_matched.recoParticles.energy, events_matched.recoParticles.direction)
    else:
        new_energy = vector.magntiude(events_matched.recoParticles.momentum)
        events_matched.recoParticles.energy = ak.where(events_matched.recoParticles.momentum.x != -999, new_energy, -999)

    return events_matched


def CreateFilteredEvents(events : Master.Data, nDaughters : int = None):
    """Filter events with specific number of daughters, then match the showers to
       MC truth.

    Args:
        events (Master.Event): events to study
        nDaughters (int, optional): filter events with ndaughters. Defaults to None.

    Returns:
        Master.Event: events after filering 
    """
    valid = Master.Pi0MCMask(events, nDaughters)
    filtered = events.Filter([valid], [valid])
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
    """Plot calculated quantities for given events

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


def Plot2DTest(ind, truths, errors, labels, xlabels, ylabels, nrows, ncols, bins=25):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4*nrows,4.8*ncols))

    for i in range(len(axes.flat)):
        x = truths[i][ind]
        y = errors[i][ind]

        if len(np.unique(x)) == 1:
            x_range = [min(x)-0.01, max(x)+0.01]
        else:
            x_range = [min(x), max(x)]
        if i == 0:
            h0, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, fe_range[ind] ], density=True)
            h0[h0==0] = np.nan
            h0T = h0.T
            im = axes.flat[i].imshow(np.flip(h0T, 0), extent=[x_range[0], x_range[1], fe_range[ind][0], fe_range[ind][1]], norm=matplotlib.colors.LogNorm())#, norm=norm, cmap=cmap)
            fig.colorbar(im, ax=axes.flat[i])
        else:
            h, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], range=[x_range, fe_range[ind]], density=True)
            h = h / h0
            h[h==0] = np.nan
            im = axes.flat[i].imshow(np.flip(h.T, 0), extent=[x_range[0], x_range[1], fe_range[ind][0], fe_range[ind][1]], norm=matplotlib.colors.LogNorm())#, norm=norm, cmap=cmap)
            fig.colorbar(im, ax=axes.flat[i])
        axes.flat[i].set_aspect("auto")
        axes.flat[i].set_title(labels[i])

    # add common x and y axis labels
    fig.add_subplot(1, 1, 1, frame_on=False)
    plt.tight_layout()
    # Hiding the axis ticks and tick labels of the bigger plot
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    # Adding the x-axis and y-axis labels for the bigger plot
    plt.xlabel(xlabels, fontsize=14)
    plt.ylabel(ylabels, fontsize=14)


@Master.timer
def main():
    events = Master.Data(file)
    events.ApplyBeamFilter() # apply beam filter if possible
    events_2 = CreateFilteredEvents(events, 2)
    
    valid = Master.Pi0MCMask(events, 3)
    events_3 = events.Filter([valid], [valid])
    matched, unmatched, selection_mask = events_3.MCMatching(applyFilters=False)
    events_3 = events_3.Filter([selection_mask], [selection_mask]) # filter events based on MC matching

    # filter masks
    matched = matched[selection_mask]
    unmatched = unmatched[selection_mask]

    events_merged_a_scalar = mergeShower(events_3, matched, unmatched, 1, True)
    events_merged_s_scalar = mergeShower(events_3, matched, unmatched, 2, True)
    events_merged_a_vector = mergeShower(events_3, matched, unmatched, 1, False)
    events_merged_s_vector = mergeShower(events_3, matched, unmatched, 2, False)
    events_unmerged = events_3.Filter([matched])

    q_2 = Master.CalculateQuantities(events_2, names)
    q = Master.CalculateQuantities(events_unmerged, names)
    q_a_vector = Master.CalculateQuantities(events_merged_a_vector, names)
    q_s_vector = Master.CalculateQuantities(events_merged_s_vector, names)
    q_a_scalar = Master.CalculateQuantities(events_merged_a_scalar, names)
    q_s_scalar = Master.CalculateQuantities(events_merged_s_scalar, names)

    f_l_vector = [s_l[0], s_l[1], "angular", "spatial"]
    ts_vector = [q_2[0], q[0], q_a_vector[0], q_s_vector[0]]
    rs_vector = [q_2[1], q[1], q_a_vector[1], q_s_vector[1]]
    es_vector = [q_2[2], q[2], q_a_vector[2], q_s_vector[2]]

    f_l_scalar = [s_l[0], s_l[1], "angular", "spatial"]
    ts_scalar = [q_2[0], q[0], q_a_scalar[0], q_s_scalar[0]]
    rs_scalar = [q_2[1], q[1], q_a_scalar[1], q_s_scalar[1]]
    es_scalar = [q_2[2], q[2], q_a_scalar[2], q_s_scalar[2]]

    f_l_angle = [s_l[0], s_l[1], s_l[2], s_l[4]]
    ts_angle = [q_2[0], q[0], q_a_scalar[0], q_a_vector[0]]
    rs_angle = [q_2[1], q[1], q_a_scalar[1], q_a_vector[1]]
    es_angle = [q_2[2], q[2], q_a_scalar[2], q_a_vector[2]]

    f_l_dist = [s_l[0], s_l[1], s_l[3], s_l[5]]
    ts_dist = [q_2[0], q[0], q_s_scalar[0], q_s_vector[0]]
    rs_dist = [q_2[1], q[1], q_s_scalar[1], q_s_vector[1]]
    es_dist = [q_2[2], q[2], q_s_scalar[2], q_s_vector[2]]

    if study == "merge":
        AnalyseQuantities(ts_vector, rs_vector, es_vector, f_l_vector, outDir+"merge_comp/")
    if study == "energy":
        AnalyseQuantities(ts_angle, rs_angle, es_angle, f_l_angle, outDir+"angle/")
        AnalyseQuantities(ts_dist, rs_dist, es_dist, f_l_dist, outDir+"dist/")
    if study == "separation":
        AnalyzeReco(events_3, matched, unmatched)
        AnalyzeTruth(events_3)

if __name__ == "__main__":

    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading photon energy (GeV)", "True Sub leading photon energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    e_l = ["Invariant mass fractional error", "Opening angle fractional error", "Leading shower energy fractional error", "Sub leading shower energy fractional error", "$\pi^{0}$ momentum fractional error"]
    r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Subleading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]
    s_l = ["2 showers", "3 showers, unmerged", "angular vector sum", "spatial vector sum", "angular scalar sum", "spatial scalar sum"]
    fe_range = [[-10, 10]] * 5
    r_range = [[]] * 5
    r_range[0] = [0, 0.5]
    t_range = [[]] * 5


    parser = argparse.ArgumentParser(description="Study em shower merging for pi0 decays")
    parser.add_argument(dest="file", type=str, help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="pi0_0p5GeV_100K/shower_merge/", help="directory to save plots")
    parser.add_argument("-a", "--analysis", dest="study", type=str, choices=["separation", "merge", "energy"], default="merge", help="what plots we want to study")
    #args = parser.parse_args("-f ROOTFiles/pi0_multi_9_3_22.root -a merge".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    outDir = args.outDir
    study = args.study
    main()
