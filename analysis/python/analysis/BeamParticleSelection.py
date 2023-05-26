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
from python.analysis.PFOSelection import Median
from python.analysis.SelectionTools import *

@CountsWrapper
def PiBeamSelection(events: Data, use_beam_inst : bool = False) -> ak.Array:
    """ Pi+ beam particle selection. For MC only.

    Args:
        events (Data): events to study.
        use_beam_inst (bool): use beam instrumentation for PID (data)

    Returns:
        ak.Array: boolean mask.
    """
    def compare_beam_pdg(pdg : int):
        return ak.fill_none(ak.pad_none(events.recoParticles.beam_inst_PDG_candidates, 1, -1), -1, -1) == pdg


    if use_beam_inst:
        mask = events.recoParticles.beam_inst_valid
        mask = mask & (events.recoParticles.beam_inst_trigger != 8)
        mask = mask & (events.recoParticles.beam_inst_nTracks == 1) & (events.recoParticles.beam_inst_nMomenta == 1)
        # 13 is used for the mu+ pdg in the beam instrumentation.
        mask = mask & ak.any(compare_beam_pdg(211) | compare_beam_pdg(13), axis = -1)
        mask = mask & events.recoParticles.reco_reconstructable_beam_event
    else:
        # return both 211 and -13 as you can't distinguish between pi+ and mu+ in data
        beam_pdg = ak.flatten(events.trueParticles.pdg[events.trueParticles.number == 1])
        # mask = (events.trueParticlesBT.beam_pdg == 211) | (events.trueParticlesBT.beam_pdg == -13)
        mask = (beam_pdg == 211) | (beam_pdg == 13) | (beam_pdg == -13)
    return mask


@CountsWrapper
def PandoraTagCut(events: Data) -> ak.Array:
    """ Cut on Pandora slice tag, selects track like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    return events.recoParticles.beam_pandora_tag == 13


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
def BeamQualityCut(events: Data, fit_values : dict = None) -> ak.Array:
    """ Cut on beam particle start position and trajectory, 
        Selects beam particles with values consistent to the beam plug.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    # beam quality cut
    # beam_startX_data = -28.3483
    # beam_startY_data = 424.553
    # beam_startZ_data = 3.19841

    # beam_startX_rms_data = 4.63594
    # beam_startY_rms_data = 5.21649
    # beam_startZ_rms_data = 1.2887

    # beam_startX_mc = -30.7834
    # beam_startY_mc = 422.422
    # beam_startZ_mc = 0.113008

    # beam_startX_rms_mc = 4.97391
    # beam_startY_rms_mc = 4.47824
    # beam_startZ_rms_mc = 0.214533

    # beam_angleX_data = 100.464
    # beam_angleY_data = 103.442
    # beam_angleZ_data = 17.6633

    # beam_angleX_mc = 101.579
    # beam_angleY_mc = 101.212
    # beam_angleZ_mc = 16.5822

    # # beam XY parameters
    # meanX_data = -31.3139
    # meanY_data = 422.116

    # rmsX_data = 3.79366
    # rmsY_data = 3.48005

    # meanX_mc = -29.1637
    # meanY_mc = 421.76

    # rmsX_mc = 4.50311
    # rmsY_mc = 3.83908

    if fit_values == None: # use fit values from 1GeV MC by default
        fits = {
            "mu_x" : -30.7834,
            "mu_y" : 422.422,
            "mu_z" : 0.113008,
            "sigma_x" : 4.97391,
            "sigma_y" : 4.47824,
            "sigma_z" : 0.214533,
            "mu_dir_x" : np.cos(101.579 * np.pi / 180),
            "mu_dir_y" : np.cos(101.212 * np.pi / 180),
            "mu_dir_z" : np.cos(16.5822 * np.pi / 180)
        }
    else:
        fits = fit_values

    # range of acceptable deltas
    dz_min = -3
    dz_max = 3
    dxy_min = -1
    dxy_max = 3
    costh_min = 0.95
    costh_max = 2

    has_angle_cut = True

    # do only MC for now.
    beam_dx = (events.recoParticles.beam_startPos.x - fits["mu_x"]) / fits["sigma_x"]
    beam_dy = (events.recoParticles.beam_startPos.y - fits["mu_y"]) / fits["sigma_y"]
    beam_dz = (events.recoParticles.beam_startPos.z - fits["mu_z"]) / fits["sigma_z"]
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5

    beam_dir = vector.normalize(vector.sub(
        events.recoParticles.beam_endPos, events.recoParticles.beam_startPos))

    beam_dir_mc = vector.vector(
        fits["mu_dir_x"],
        fits["mu_dir_y"],
        fits["mu_dir_z"]
    )
    beam_dir_mc = vector.normalize(beam_dir_mc)

    beam_costh = vector.dot(beam_dir, beam_dir_mc)

    beam_quality_mask = events.eventNum > 0  # mask which is all trues

    def cut(x, xmin, xmax):
        return ((x > xmin) & (x < xmax))

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
    median = Median(events.recoParticles.beam_dEdX)
    return median < 2.4


def CreateDefaultSelection(events: Data, use_beam_inst : bool = False, beam_quality_fits : dict = None, verbose: bool = True, return_table: bool = True) -> ak.Array:
    """ Create boolean mask for default MC beam particle selection
        (includes pi+ beam selection for now as well).

    Args:
        events (Data): events to study
        use_beam_inst (bool) : use beam instrumentation for particle_id
        beam_quality_fits (dict) : fit parameters for beam quality cuts
        verbose (bool) : verbose printout
        return_tables (bool) : return tables with efficiencies

    Returns:
        ak.Array: boolean mask
    """
    selection = [
        PiBeamSelection,  # * pi+ beam selection
        PandoraTagCut,
        CaloSizeCut,
        MichelScoreCut,
        BeamQualityCut,
        APA3Cut,
        MedianDEdXCut
    ]
    arguments = [
        {"use_beam_inst" : use_beam_inst},
        {},
        {},
        {},
        {"fit_values" : beam_quality_fits},
        {},
        {}
    ]
    return CombineSelections(events, selection, 0, arguments, verbose, return_table)
