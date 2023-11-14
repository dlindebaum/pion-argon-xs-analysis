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
from particle import Particle

from python.analysis import vector
from python.analysis.Master import Data
from python.analysis.PFOSelection import Median, GoodShowerSelection
from python.analysis.SelectionTools import *


def PiBeamSelection(events: Data, use_beam_inst : bool = False, return_property : bool = False) -> ak.Array:
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
        beam_pdg = None # return None output if property is specified for data ntuples
    else:
        # return both 211 and -13 as you can't distinguish between pi+ and mu+ in data
        beam_pdg = ak.flatten(events.trueParticles.pdg[events.trueParticles.number == 1])
        mask = (beam_pdg == 211) | (beam_pdg == 13) | (beam_pdg == -13)
    if return_property is True:
        return mask, beam_pdg
    else:
        return mask


def PandoraTagCut(events: Data, cut : int = 13, return_property : bool = False) -> ak.Array:
    """ Cut on Pandora slice tag, selects track like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    return CreateMask(cut, "==", events.recoParticles.beam_pandora_tag, return_property)


def CaloSizeCut(events: Data, return_property : bool = False) -> ak.Array:
    """ Cut which checks the beam particle has a calorimetry object (required for median dEdX cut).

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    calo_wire = events.recoParticles.beam_caloWire
    # Analyser fills the empty entry with a -999
    calo_wire = calo_wire[calo_wire != -999]

    return CreateMask(0, ">", ak.num(calo_wire, 1), return_property)


def BeamQualityCut(events: Data, fits : dict, dxy_cut : list = [-3, 3], dz_cut : list = [-3, 3], costh_cut : list = [0.95, 2]) -> ak.Array:
    """ Cut on beam particle start position and trajectory, 
        Selects beam particles with values consistent to the beam plug.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    beam_dx = (events.recoParticles.beam_startPos_SCE.x - fits["mu_x"]) / fits["sigma_x"]
    beam_dy = (events.recoParticles.beam_startPos_SCE.y - fits["mu_y"]) / fits["sigma_y"]
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5
    beam_dz = (events.recoParticles.beam_startPos_SCE.z - fits["mu_z"]) / fits["sigma_z"]

    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos_SCE, events.recoParticles.beam_startPos_SCE))

    beam_dir_mc = vector.vector(
        fits["mu_dir_x"],
        fits["mu_dir_y"],
        fits["mu_dir_z"]
    )
    beam_dir_mc = vector.normalize(beam_dir_mc)

    beam_costh = vector.dot(beam_dir, beam_dir_mc)

    def cut(x, xmin, xmax):
        return ((x > xmin) & (x < xmax))

    beam_quality_mask = cut(beam_dz, min(dz_cut), max(dz_cut))
    beam_quality_mask = beam_quality_mask & cut(beam_dxy, min(dxy_cut), max(dxy_cut))
    beam_quality_mask = beam_quality_mask & cut(beam_costh, min(costh_cut), max(costh_cut))
    return beam_quality_mask


def DxyCut(events: Data, fits : dict, cut, return_property : bool = False):
    beam_dx = (events.recoParticles.beam_startPos_SCE.x - fits["mu_x"]) / fits["sigma_x"]
    beam_dy = (events.recoParticles.beam_startPos_SCE.y - fits["mu_y"]) / fits["sigma_y"]
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5
    return CreateMask(cut, "<", beam_dxy, return_property)


def DzCut(events: Data, fits : dict, cut, return_property : bool = False):
    beam_dz = (events.recoParticles.beam_startPos_SCE.z - fits["mu_z"]) / fits["sigma_z"]
    return CreateMask(cut, [">", "<"], beam_dz, return_property)


def CosThetaCut(events: Data, fits : dict, cut, return_property : bool = False):
    beam_dir = vector.normalize(vector.sub(events.recoParticles.beam_endPos_SCE, events.recoParticles.beam_startPos_SCE))

    beam_dir_mc = vector.vector(
        fits["mu_dir_x"],
        fits["mu_dir_y"],
        fits["mu_dir_z"]
    )
    beam_dir_mc = vector.normalize(beam_dir_mc)

    beam_costh = vector.dot(beam_dir, beam_dir_mc)
    return CreateMask(cut, ">", beam_costh, return_property)


def APA3Cut(events: Data, cut : float = 220, return_property : bool = False) -> ak.Array:
    """ Cuts on beam end z position to select beam particles which end in APA3.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    # APA3 cut
    return CreateMask(cut, "<", events.recoParticles.beam_endPos_SCE.z, return_property)


def MichelScoreCut(events: Data, cut : float = 0.55, return_property : bool = False) -> ak.Array:
    """ Cut on michel score to remove muon like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    score = ak.where(events.recoParticles.beam_nHits != 0, events.recoParticles.beam_michelScore / events.recoParticles.beam_nHits, -999)
    return CreateMask(cut, "<", score, return_property)


def MedianDEdXCut(events: Data, cut : float = 2.4, return_property : bool = False) -> ak.Array:
    """ cut on median dEdX to exlude proton background.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    median = Median(events.recoParticles.beam_dEdX)
    return CreateMask(cut, "<", median, return_property)


def BeamScraperCut(events : Data, KE_range : int, fits : dict, cut : float = 1.5, return_property : bool = False) -> ak.Array:
    """ Beam scraper cut. Required to exclude events with poor consistency between
        the beam insturmention KE and front facing KE
        (front facing means the first calorimetry point in the TPC).

    Args:
        events (Data): events to study
        fit_values (dict): beam scraper fit values
        KE_range (int): index of kinetic energy range in fit_values.
        pdg_hyp (int, optional): the particle species we assume our sample to be composed of. Defaults to 211.
        cut (float, optional): user specified cut value (normalised). Defaults to 1.5.

    Returns:
        ak.Array: boolean mask
    """

    key = str(KE_range)
    mu_x = fits[key]["mu_x_inst"]
    mu_y = fits[key]["mu_y_inst"]
    sigma_x = fits[key]["sigma_x_inst"]
    sigma_y = fits[key]["sigma_y_inst"]

    nx = (events.recoParticles.beam_inst_pos.x - mu_x)/sigma_x
    ny = (events.recoParticles.beam_inst_pos.y - mu_y)/sigma_y

    r = np.sqrt(nx**2 + ny**2)
    return CreateMask(cut, "<", r, return_property)


def HasFinalStatePFOsCut(events: Data, return_property : bool = False) -> ak.Array:
    """ Selects events which have final state PFOs that are well reconstructed.

    Args:
        events (Data): events to look at
        return_property (bool, optional): returns quantity cut on. Defaults to False.

    Returns:
        ak.Array: _description_
    """
    pfo_mask = GoodShowerSelection(events) # create this mask internally so that the PFO and event selections remain separate
    nPFO = ak.num(events.recoParticles.number[pfo_mask])
    return CreateMask(0, ">", nPFO, return_property)


def CreateDefaultSelection(events: Data,
    use_beam_inst : bool = False,
    beam_quality_fits : dict = None,
    pdg_hyp : int = 211,
    scraper : bool = False,
    scraper_fits : dict = None,
    scraper_KE_range : int = None,
    scraper_cut : float = None,
    verbose: bool = True, return_table: bool = True) -> ak.Array:
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
        MedianDEdXCut,
    ]
    arguments = [
        {"use_beam_inst" : use_beam_inst},
        {},
        {},
        {},
        {"fits" : beam_quality_fits},
        {},
        {}
    ]
    if scraper is True:
        selection.append(BeamScraperCut)
        arguments.append({"KE_range" : scraper_KE_range, "fits" : scraper_fits, "cut" : scraper_cut})
    return CombineSelections(events, selection, 0, arguments, verbose, return_table)
