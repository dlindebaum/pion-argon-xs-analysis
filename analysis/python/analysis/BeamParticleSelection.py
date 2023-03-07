"""
Created on: 03/03/2023 14:30

Author: Shyam Bhuller

Description: Contains cuts for Beam Particle Selection.
TODO Cleanup beam quality cut code
? Should the cuts be configurable?
? should this be kept in a class?
"""
from functools import wraps

import awkward as ak
import numpy as np
import pandas as pd

import python.analysis.vector as vector
from python.analysis.Master import Data


def CountMask(m : ak.Array) -> tuple:
    """ Counts the total number of entries in a boolean mask,
        and the number which are True.

    Args:
        m (ak.Array): boolean mask.

    Returns:
        tuple: (number of entries, number of entries which are true).
    """
    return ak.count(m), ak.count(m[m])


def CountEventsWrapper(f):
    """ Wrapper for selection functions which checks the number of entries that pass a cut.

    Args:
        f (function): selection function.
    Returns:
        any: output of f.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        m = f(*args, **kwargs)
        c = CountMask(m)
        print(f"number of entries before|after {f.__name__}: {c[0]}|{c[1]}")
        return m
    return wrapper

#! This is a pi+ beam selection, should this be here?
@CountEventsWrapper
def PiBeamSelection(events : Data) -> ak.Array:
    """ Pi+ beam particle selection. For MC only.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    pdg = events.io.Get("reco_beam_PFP_true_byHits_pdg")
    return (pdg == 211) | (pdg == -13) # return both 211 and -13 as you can't distinguish between pi+ and mu+ in data

@CountEventsWrapper
def PandoraTagCut(events : Data) -> ak.Array:
    """ Cut on Pandora slice tag, selects track like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    tag = events.recoParticles.pandoraTag[events.recoParticles.beam_number == events.recoParticles.number]
    tag = ak.flatten(ak.fill_none(ak.pad_none(tag, 1), -999))
    return tag == 13

@CountEventsWrapper
def CaloSizeCut(events : Data) -> ak.Array:
    """ Cut which checks the beam particle has a calorimetry object (required for median dEdX cut).

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    calo_wire = events.recoParticles.beam_caloWire
    calo_wire = calo_wire[calo_wire != -999] # Analyser fills the empty entry with a -999
    return ak.num(calo_wire, 1) > 0 

@CountEventsWrapper
def BeamQualityCut(events : Data) -> ak.Array:
    """ Cut on beam particle start position and trajectory, 
        Selects beam particles with values consistent to the beam plug.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    # beam quality cut
    beam_startX_data = -28.3483;
    beam_startY_data = 424.553;
    beam_startZ_data = 3.19841;

    beam_startX_rms_data = 4.63594;
    beam_startY_rms_data = 5.21649;
    beam_startZ_rms_data = 1.2887;

    beam_startX_mc = -30.7834;
    beam_startY_mc = 422.422;
    beam_startZ_mc = 0.113008;

    beam_startX_rms_mc = 4.97391;
    beam_startY_rms_mc = 4.47824;
    beam_startZ_rms_mc = 0.214533;

    beam_angleX_data = 100.464;
    beam_angleY_data = 103.442;
    beam_angleZ_data = 17.6633;

    beam_angleX_mc = 101.579;
    beam_angleY_mc = 101.212;
    beam_angleZ_mc = 16.5822;

    # beam XY parameters
    meanX_data = -31.3139;
    meanY_data = 422.116;

    rmsX_data = 3.79366;
    rmsY_data = 3.48005;

    meanX_mc = -29.1637;
    meanY_mc = 421.76;

    rmsX_mc = 4.50311;
    rmsY_mc = 3.83908;

    # range of acceptable deltas

    #! these are set like this so that the comparisons are not done for dx and dy, instead a comparison with sqrt(x**2 + y**2) is done (dxy)
    dx_min = 3
    dx_max = -3
    dy_min = 3
    dy_max = -3
    #!
    dz_min = -3
    dz_max = 3
    dxy_min = -1
    dxy_max = 3
    costh_min = 0.95
    costh_max = 2
    
    has_angle_cut = True;

    # do only MC for now.
    beam_dx = (events.recoParticles.beam_startPos.x - beam_startX_mc) / beam_startX_rms_mc
    beam_dy = (events.recoParticles.beam_startPos.y - beam_startY_mc) / beam_startY_rms_mc
    beam_dz = (events.recoParticles.beam_startPos.z - beam_startZ_mc) / beam_startZ_rms_mc
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5

    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))

    beam_dir_mc = vector.vector(
        np.cos(beam_angleX_mc * np.pi / 180),
        np.cos(beam_angleY_mc * np.pi / 180),
        np.cos(beam_angleZ_mc * np.pi / 180),
        )
    beam_dir_mc = vector.normalize(beam_dir_mc)

    beam_costh = vector.dot(beam_dir, beam_dir_mc)


    beam_quality_mask = events.eventNum > 0 # mask which is all trues

    def cut(x, xmin, xmax):
        return ((x > xmin) & (x < xmax))

    if dx_min < dx_max:
        beam_quality_mask = beam_quality_mask & cut(beam_dx, dx_min, dx_max) #* should be the same as the logic below

    if dy_min < dy_max:
        beam_quality_mask = beam_quality_mask & cut(beam_dy, dy_min, dy_max) #* should be the same as the logic below

    if dz_min < dz_max:
        beam_quality_mask = beam_quality_mask & cut(beam_dz, dz_min, dz_max) #* should be the same as the logic below

    if dxy_min < dxy_max:
        beam_quality_mask = beam_quality_mask & cut(beam_dxy, dxy_min, dxy_max) #* should be the same as the logic below

    if has_angle_cut and (costh_min < costh_max):
        beam_quality_mask = beam_quality_mask & cut(beam_costh, costh_min, costh_max) #* should be the same as the logic below
    return beam_quality_mask

@CountEventsWrapper
def APA3Cut(events : Data) -> ak.Array:
    """ Cuts on beam end z position to select beam particles which end in APA3.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    # APA3 cut
    return events.recoParticles.beam_endPos.z < 220 # cm

@CountEventsWrapper
def MichelScoreCut(events : Data) -> ak.Array:
    """ Cut on michel score to remove muon like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    score = events.recoParticles.beam_michelScore / events.recoParticles.beam_nHits
    return score < 0.55

@CountEventsWrapper
def MedianDEdXCut(events : Data) -> ak.Array:
    """ cut on median dEdX to exlude proton background.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    beam_dEdX = events.recoParticles.beam_dEdX

    # awkward has no median function and numpy median won't work on jagged arrays, so we do it ourselves
    beam_dEdX_sorted = ak.sort(beam_dEdX, -1) # first sort in ascending order
    count = ak.num(beam_dEdX, 1) # get the number of entries per beam dEdX

    # calculate the median assuming the arrray length is odd
    med_odd = count // 2 # median is middle entry
    select = ak.local_index(beam_dEdX_sorted) == med_odd
    median_odd = beam_dEdX_sorted[select]

    # calculate the median assuming the arrray length is even
    med_even = (med_odd - 1) * (count > 1) # need the middle - 1 value
    select_even = ak.local_index(beam_dEdX_sorted) == med_even
    median_even = (beam_dEdX_sorted[select] + beam_dEdX_sorted[select_even]) / 2 # median is average of middle value and middle - 1 value

    median = ak.flatten(ak.fill_none(ak.pad_none(ak.where(count % 2, median_odd, median_even), 1), -999)) # pick which median is the correct one

    return median < 2.4


def CombineSelections(events : Data, selection : list, verbose : bool = False, return_table : bool = False) -> ak.Array:
    """ Combines multiple beam particle selections.

    Args:
        events (Data): events to study.
        selection (list): list of beam particle selection functions (must return a boolean mask).

    Returns:
        ak.Array: boolean mask of combined selection.
    """
    if verbose or return_table:
        table = {
            "no selection" : [ak.num(events.eventNum, 0), 100]*2
        }

    mask = None
    for s in selection:
        new_mask = s(events)
        if not hasattr(mask, "__iter__"):
            mask = new_mask
        else:
            mask = mask & new_mask

        if return_table or verbose:
            successive_counts = ak.num(mask[mask], 0)
            single_counts = ak.num(new_mask[new_mask], 0)
            table[s.__name__] = [single_counts, 100 * single_counts/ table["no selection"][0], successive_counts, 100 * successive_counts / table["no selection"][0]]

    if return_table or verbose:
        table = pd.DataFrame(table, index = ["number of events which pass the cut", "single efficiency", "number of events afer successive cuts", "successive efficiency"]).T
        relative_efficiency = np.append([np.nan], 100 * table["number of events afer successive cuts"].values[1:] / table["number of events afer successive cuts"].values[:-1])
        table["relative efficiency"] = relative_efficiency
    if verbose:
        print(table)

    if return_table:
        return mask, table
    else:
        return mask


def ApplyDefaultSelection(events : Data, verbose : bool = True, return_table : bool = True) -> ak.Array:
    """ Create boolean mask for default MC beam particle selection
        (includes pi+ beam selection for now as well).

    Args:
        events (Data): events to study

    Returns:
        ak.Array: boolean mask
    """
    selection = [
        PiBeamSelection, #* pi+ beam selection
        PandoraTagCut,
        CaloSizeCut,
        BeamQualityCut,
        APA3Cut,
        MichelScoreCut,
        MedianDEdXCut
    ]
    return CombineSelections(events, selection, verbose, return_table)