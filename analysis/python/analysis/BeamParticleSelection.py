"""
Created on: 03/03/2023 14:30

Author: Shyam Bhuller

Description: Contains cuts for Beam Particle Selection.
TODO Cleanup beam quality cut code
? Should the cuts be configurable?
? should this be kept in a class?
"""
import awkward as ak
import numpy as np

from python.analysis import vector
from python.analysis.Master import Data
from python.analysis.SelectionTools import *


#! This is a pi+ beam selection, should this be here?
@CountsWrapper
def PiBeamSelection(events: Data) -> ak.Array:
    """ Pi+ beam particle selection. For MC only.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    pdg = events.io.Get("reco_beam_PFP_true_byHits_pdg")
    # return both 211 and -13 as you can't distinguish between pi+ and mu+ in data
    return (pdg == 211) | (pdg == -13)


@CountsWrapper
def PandoraTagCut(events: Data) -> ak.Array:
    """ Cut on Pandora slice tag, selects track like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    tag = events.recoParticles.pandoraTag[events.recoParticles.beam_number ==
                                          events.recoParticles.number]
    tag = ak.flatten(ak.fill_none(ak.pad_none(tag, 1), -999))
    return tag == 13


@CountsWrapper
def CaloSizeCut(events: Data) -> ak.Array:
    """ Cut which checks the beam particle has a calorimetry object (required for median dEdX cut).

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    calo_wire = events.recoParticles.beam_caloWire
    # Analyser fills the empty entry with a -999
    calo_wire = calo_wire[calo_wire != -999]
    return ak.num(calo_wire, 1) > 0


@CountsWrapper
def BeamQualityCut(events: Data) -> ak.Array:
    """ Cut on beam particle start position and trajectory, 
        Selects beam particles with values consistent to the beam plug.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    # beam quality cut
    beam_startX_data = -28.3483
    beam_startY_data = 424.553
    beam_startZ_data = 3.19841

    beam_startX_rms_data = 4.63594
    beam_startY_rms_data = 5.21649
    beam_startZ_rms_data = 1.2887

    beam_startX_mc = -30.7834
    beam_startY_mc = 422.422
    beam_startZ_mc = 0.113008

    beam_startX_rms_mc = 4.97391
    beam_startY_rms_mc = 4.47824
    beam_startZ_rms_mc = 0.214533

    beam_angleX_data = 100.464
    beam_angleY_data = 103.442
    beam_angleZ_data = 17.6633

    beam_angleX_mc = 101.579
    beam_angleY_mc = 101.212
    beam_angleZ_mc = 16.5822

    # beam XY parameters
    meanX_data = -31.3139
    meanY_data = 422.116

    rmsX_data = 3.79366
    rmsY_data = 3.48005

    meanX_mc = -29.1637
    meanY_mc = 421.76

    rmsX_mc = 4.50311
    rmsY_mc = 3.83908

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

    has_angle_cut = True

    # do only MC for now.
    beam_dx = (events.recoParticles.beam_startPos.x -
               beam_startX_mc) / beam_startX_rms_mc
    beam_dy = (events.recoParticles.beam_startPos.y -
               beam_startY_mc) / beam_startY_rms_mc
    beam_dz = (events.recoParticles.beam_startPos.z -
               beam_startZ_mc) / beam_startZ_rms_mc
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5

    beam_dir = vector.normalize(vector.sub(
        events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))

    beam_dir_mc = vector.vector(
        np.cos(beam_angleX_mc * np.pi / 180),
        np.cos(beam_angleY_mc * np.pi / 180),
        np.cos(beam_angleZ_mc * np.pi / 180),
    )
    beam_dir_mc = vector.normalize(beam_dir_mc)

    beam_costh = vector.dot(beam_dir, beam_dir_mc)

    beam_quality_mask = events.eventNum > 0  # mask which is all trues

    def cut(x, xmin, xmax):
        return ((x > xmin) & (x < xmax))

    if dx_min < dx_max:
        # * should be the same as the logic below
        beam_quality_mask = beam_quality_mask & cut(beam_dx, dx_min, dx_max)

    if dy_min < dy_max:
        # * should be the same as the logic below
        beam_quality_mask = beam_quality_mask & cut(beam_dy, dy_min, dy_max)

    if dz_min < dz_max:
        # * should be the same as the logic below
        beam_quality_mask = beam_quality_mask & cut(beam_dz, dz_min, dz_max)

    if dxy_min < dxy_max:
        # * should be the same as the logic below
        beam_quality_mask = beam_quality_mask & cut(beam_dxy, dxy_min, dxy_max)

    if has_angle_cut and (costh_min < costh_max):
        # * should be the same as the logic below
        beam_quality_mask = beam_quality_mask & cut(
            beam_costh, costh_min, costh_max)
    return beam_quality_mask


@CountsWrapper
def APA3Cut(events: Data) -> ak.Array:
    """ Cuts on beam end z position to select beam particles which end in APA3.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    # APA3 cut
    return events.recoParticles.beam_endPos.z < 220  # cm


@CountsWrapper
def MichelScoreCut(events: Data) -> ak.Array:
    """ Cut on michel score to remove muon like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    score = ak.where(events.recoParticles.beam_nHits != 0, events.recoParticles.beam_michelScore / events.recoParticles.beam_nHits, -999)
    return score < 0.55


@CountsWrapper
def MedianDEdXCut(events: Data) -> ak.Array:
    """ cut on median dEdX to exlude proton background.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    beam_dEdX = events.recoParticles.beam_dEdX

    # awkward has no median function and numpy median won't work on jagged arrays, so we do it ourselves
    beam_dEdX_sorted = ak.sort(beam_dEdX, -1)  # first sort in ascending order
    count = ak.num(beam_dEdX, 1)  # get the number of entries per beam dEdX

    # calculate the median assuming the arrray length is odd
    med_odd = count // 2  # median is middle entry
    select = ak.local_index(beam_dEdX_sorted) == med_odd
    median_odd = beam_dEdX_sorted[select]

    # calculate the median assuming the arrray length is even
    med_even = (med_odd - 1) * (count > 1)  # need the middle - 1 value
    select_even = ak.local_index(beam_dEdX_sorted) == med_even
    # median is average of middle value and middle - 1 value
    median_even = (beam_dEdX_sorted[select] +
                   beam_dEdX_sorted[select_even]) / 2

    median = ak.flatten(ak.fill_none(ak.pad_none(ak.where(
        count % 2, median_odd, median_even), 1), -999))  # pick which median is the correct one

    return median < 2.4


def CreateDefaultSelection(events: Data, verbose: bool = True, return_table: bool = True) -> ak.Array:
    """ Create boolean mask for default MC beam particle selection
        (includes pi+ beam selection for now as well).

    Args:
        events (Data): events to study

    Returns:
        ak.Array: boolean mask
    """
    selection = [
        PiBeamSelection,  # * pi+ beam selection
        PandoraTagCut,
        CaloSizeCut,
        BeamQualityCut,
        APA3Cut,
        MichelScoreCut,
        MedianDEdXCut
    ]
    return CombineSelections(events, selection, 0, None, verbose, return_table)
