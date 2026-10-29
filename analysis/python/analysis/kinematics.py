"""Kinematic helpers."""
import awkward as ak

from python.analysis import bethe_bloch, selection, vector

def KE(p, m):
    return (p**2 + m**2)**0.5 - m

def track_pitch(tracks : ak.Array) -> ak.Array:
    return vector.dist(tracks[:, :-1], tracks[:, 1:])


def track_length(tracks : ak.Array = None, pitches : ak.Array = None) -> ak.Array:
    if (tracks is None) & (pitches is not None):
        p = pitches
    elif (tracks is not None) & (pitches is None):
        p = track_pitch(tracks)
    else:
        raise Exception("Track length requires either tracks or pitches to be supplied.")
    return ak.fill_none(ak.sum(p, -1), 0)


def truncate_track(tracks : ak.Array, x_trunc = None, y_trunc = None, z_trunc = None):
    trunc = {"x" : x_trunc, "y" : y_trunc, "z" : z_trunc}

    masks = {}
    for k, v in trunc.items():
        if v is None: continue
        masks[k] = (selection.cuts_to_func(trunc[k], "<")(tracks[k]))
    mask = selection.CombineMasks(masks, "and")

    index = ak.local_index(mask)
    max_ind = ak.argmin(~mask == False, -1) 
    max_ind = ak.where(max_ind == 0, ak.num(mask), max_ind)
    truncated_tracks = tracks[index <= max_ind]
    return truncated_tracks

def reco_end_energy(tracks : ak.Array, KE_init: ak.Array, dEdX : ak.Array | None, method : str) -> ak.Array:
    """ Calculates the energy deposited by the beam particle in the TPC, either using calorimetric information or the bethe bloch formula (track information).

    Args:
        tracks (ak.Array): Track trajectory points.
        KE_init (ak.Array): Initial kinetic energy.
        dEdX (ak.Array | None): dEdX for each hit deposition, note this is only used for calo method.
        method (str): method to calcualte the deposited energy, either "calo" or "track".

    Returns:
        ak.Array: deposited energy
    """
    pitches = track_pitch(tracks)

    if method == "calo":
        dE = ak.sum(dEdX[:, :-1] * pitches, -1)
        KE_end = KE_init - dE
    elif method == "track":
        track_length = track_length(pitches=pitches)
        KE_end = bethe_bloch.KE_end(KE_init, track_length, 50)
    else:
        raise Exception(f"{method} not a valid method, pick 'calo' or 'track'")
    return KE_end
