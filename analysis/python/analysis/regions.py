"""Region definitions and utilities."""
import os

from argparse import Namespace

import awkward as ak
import numpy as np
import pandas as pd

from python.analysis import common, ntuple, plots, samples, selection, tags

@common.timer
def RegionSelection(events : ntuple.Data, args : Namespace | dict, is_mc : bool, region_type : samples.SampleDefinition = None, process_type : samples.SampleDefinition = None, removed : bool = False) -> dict[np.ndarray]:
    """ Get reco and true regions (if possible) for ntuple.

    Args:
        events (Master.Data): events before beam pion selection.
        args (Namespace): application arguements
        is_mc (bool): if ntuple is MC or Data.

    Returns:
        tuple[dict, dict]: regions
    """

    args_c = common.args_to_dict(args)

    if is_mc:
        key = "mc"
    else:
        key = "data"

    selection_masks = args_c["selection_masks"][key]

    events_copy = events.Filter(returnCopy = True)
    
    if "fiducial" in selection_masks and (len(selection_masks["fiducial"]) > 0):
        mask = selection.CombineMasks(selection_masks["fiducial"][events_copy.filename])
        events_copy.Filter([mask], [mask])

    mask = selection.CombineMasks(selection_masks["beam"][events_copy.filename])
    events_copy.Filter([mask], [mask])

    # counts = {}
    # for obj in selection_masks:
    #     if obj in ["beam", "null_pfo", "fiducial"]: continue
    #     counts[f"n_{obj}"] = SelectionTools.GetPFOCounts(selection_masks[obj][events.filename])

    if region_type is None:
        region_def = args_c["region_definitions"]
    else:
        region_def = region_type

    reco_regions = region_def.CreateDefinitions(region_def.criteria_list.get_criteria_values(events_copy, selection_masks, **args["region_args"]), uncategorised = removed)
    
    if is_mc:
        if process_type is None:
            process_def = args_c["process_definitions"]
        else:
            process_def = process_type

        true_regions = process_def.CreateDefinitions(process_def.criteria_list.get_criteria_values(events_copy, **args["process_args"]), uncategorised = removed)

        for k in true_regions:
            true_regions[k] = true_regions[k]
        for k in reco_regions:
            reco_regions[k] = reco_regions[k]
        return reco_regions, true_regions
    else:
        return reco_regions


def CountInRegions(true_regions : dict, reco_regions : dict, selection_efficincy : np.ndarray = None) -> np.ndarray:
    """ Computes the counts of each combination of reco and true regions.

    Args:
        true_regions (dict): true region masks
        reco_regions (dict): reco region masks
        return_counts (bool, optional): return matrix of counts. Defaults to False.

    Returns:
        np.ndarray: counts.
    """
    counts = []
    for t in true_regions:
        true_counts = []
        for r in reco_regions:
            mask = true_regions[t] & reco_regions[r]
            if selection_efficincy is not None: mask = mask & selection_efficincy
            true_counts.append(ak.sum(mask)) # true counts for each reco region
        counts.append(true_counts)
    return counts

@common.timer
def GetTotalPionInelMasks(mc : ntuple.Data) -> ak.Array:
    """ Returns the mask which selects true inelastic pi+.

    Args:
        mc (Master.Data): mc events.

    Returns:
        ak.Array: mask.
    """
    particle_tags = tags.GenerateTrueBeamParticleTags(mc)
    return particle_tags["$\\pi^{+}$:inel"].mask

@common.timer
def RecoRegionSelection(region_selections : dict[dict], out : str):
    """ Study which computes a correlation matrix ofthe event faction for reco regions and true regions.
        Saved to file to be used in toy simulation.

    Args:
        mc (Master.Data): mc events.
        args (Namespace): application arguments.
    """
    os.makedirs(out + "reco_regions/", exist_ok = True)
    pdf = plots.PlotBook(out + "reco_regions/reco_regions_study")
    pe = {}
    pe_index_labels = None
    for r in region_selections:
        print(r)
        reco_regions = region_selections[r]["reco_regions"]
        true_regions = region_selections[r]["true_regions"]

        process_keys = [k for k in true_regions if k != "uncategorised"]
        if pe_index_labels is None:
            tags = tags.ExclusiveProcessTags({k : None for k in process_keys})
            pe_index_labels = tags.name_simple.values

        counts = np.array(CountInRegions(true_regions, reco_regions))
        plots.plt.figure(figsize = [3 * 6.4, 3 * 4.8])
        plots.PlotConfusionMatrix(counts, list(reco_regions.keys()), list(true_regions.keys()), y_label = "True process", x_label = "Reco region", title = common.remove_(r), newFigure = False)
        pdf.Save()

        (a,b)=counts.shape
        diff = a-b
        if a>b:
            padding=((0,0),(0,diff))
        else:
            padding=((0,-diff),(0,0))
        counts = np.pad(counts, padding, mode='constant', constant_values = 0)


        pe[common.remove_(r)] = ((np.diag(counts) / np.sum(counts, 0)) * (np.diag(counts) / np.sum(counts, 1)))[:len(true_regions) - 1] # -1 to exlcude the uncategorised species. (purity x efficiency not needed)

        reco_regions.pop("uncategorised")
        counts = CountInRegions(true_regions, reco_regions)

        fractions_df = counts / np.sum(counts, axis = 1)[:, np.newaxis]
        fractions_df = pd.DataFrame(np.array(fractions_df).T, columns = true_regions, index = reco_regions) # columns are the true regions, so index over those to get the fractions
        fractions_df.to_hdf(out + f"reco_regions/{r}_reco_region_fractions.hdf5", "df")
    pdf.close()
    print(pe)
    print(pe_index_labels)
    pd.DataFrame(pe, index = pe_index_labels).style.format(precision = 2).to_latex(out + "reco_regions/pe.tex")
    return
