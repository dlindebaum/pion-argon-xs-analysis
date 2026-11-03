"""
Created on: 30/10/2026 16:06

Author: Shyam Bhuller

Description: Functions to help calculate kinematic properties, generally related to particle tracks.
"""
import awkward as ak

from python.analysis import bethe_bloch, selection, vector

def KE(p, m):
    """ Calculate kinetic energy from momentum and rest mass.

        Args:
            p (Any): Particle momentum.
            m (Any): Particle rest mass.

        Returns:
            Any: Kinetic energy, calculated as ``sqrt(p**2 + m**2) - m``..
    """
    return (p**2 + m**2)**0.5 - m


def track_pitch(tracks : ak.Array) -> ak.Array:
    """ Calculate the distances between consecutive points on each track.

        Args:
            tracks (ak.Array): Track trajectories represented as sequences of three-dimensional points.

        Returns:
            ak.Array: Distance between each adjacent pair of points..
    """
    return vector.dist(tracks[:, :-1], tracks[:, 1:])


def track_length(tracks : ak.Array = None, pitches : ak.Array = None) -> ak.Array:
    """ Calculate the total length of each track.

               Supply either track points or precomputed consecutive-point pitches, but
               not both.

        Args:
            tracks (ak.Array): Track trajectories used to calculate pitches. Defaults to None.
            pitches (ak.Array): Precomputed distances between adjacent track points. Defaults to None.

        Returns:
            ak.Array: Sum of pitches for each track, with missing sums filled by 0..
    """
    if (tracks is None) & (pitches is not None):
        p = pitches
    elif (tracks is not None) & (pitches is None):
        p = track_pitch(tracks)
    else:
        raise Exception("Track length requires either tracks or pitches to be supplied.")
    return ak.fill_none(ak.sum(p, -1), 0)


def truncate_track(tracks : ak.Array, x_trunc = None, y_trunc = None, z_trunc = None):
    """ Truncate tracks using optional coordinate thresholds.

        For each supplied threshold, points are tested against the corresponding
        coordinate using a ``<`` comparison. The coordinate masks are combined
        with logical AND, and points through the resulting per-track truncation
        index are retained.

        Args:
            tracks (ak.Array): Track trajectories with ``x``, ``y``, and ``z`` coordinate fields.
            x_trunc (Any | None): Upper threshold for the ``x`` coordinate. Defaults to None.
            y_trunc (Any | None): Upper threshold for the ``y`` coordinate. Defaults to None.
            z_trunc (Any | None): Upper threshold for the ``z`` coordinate. Defaults to None.

        Returns:
            Any: Track points retained after applying the supplied thresholds..
    """
    trunc = {"x" : x_trunc, "y" : y_trunc, "z" : z_trunc}

    masks = {}
    for k, v in trunc.items():
        if v is None: continue
        masks[k] = (selection.cuts_to_func(trunc[k], "<")(tracks[k]))
    mask = selection.combine_masks(masks, "and")

    index = ak.local_index(mask)
    max_ind = ak.argmin(~mask == False, -1) 
    max_ind = ak.where(max_ind == 0, ak.num(mask), max_ind)
    truncated_tracks = tracks[index <= max_ind]
    return truncated_tracks

def reco_end_energy(tracks : ak.Array, KE_init: ak.Array, dEdX : ak.Array | None, method : str) -> ak.Array:
    """ Calculate the beam particle's reconstructed kinetic energy at track end.

        The calorimetric method subtracts the summed ``dEdX * pitch`` energy loss
        from the initial kinetic energy. The track method estimates the remaining
        energy from the track length using the Bethe-Bloch calculation.

        Args:
            tracks (ak.Array): Track trajectory points.
            KE_init (ak.Array): Initial kinetic energy of the beam particle.
            dEdX (ak.Array | None): Energy-loss-per-distance values for trajectory points; required when ``method`` is ``"calo"``.
            method (str): Energy estimator to use: ``"calo"`` or ``"track"``.

        Returns:
            ak.Array: Reconstructed kinetic energy at the end of the track..
    """
    pitches = track_pitch(tracks)

    if method == "calo":
        dE = ak.sum(dEdX[:, :-1] * pitches, -1)
        KE_end = KE_init - dE
    elif method == "track":
        tl = track_length(pitches=pitches)
        KE_end = bethe_bloch.KE_end(KE_init, tl, 50)
    else:
        raise Exception(f"{method} not a valid method, pick 'calo' or 'track'")
    return KE_end
