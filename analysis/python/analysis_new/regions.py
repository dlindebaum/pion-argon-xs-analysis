"""Region definitions and utilities."""

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
