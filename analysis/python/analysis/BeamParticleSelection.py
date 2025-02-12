"""
Created on: 03/03/2023 14:30

Author: Shyam Bhuller

Description: Contains cuts for Beam Particle Selection.
TODO depreciate BeamQualityCut
"""
import awkward as ak
import numpy as np

from python.analysis import vector
from python.analysis.Master import Data
from python.analysis.PFOSelection import Median, GoodShowerSelection
from python.analysis.SelectionTools import *


def GetTruncatedDEdX(sample : Data, truncate : float):
    truncated_start = sample.recoParticles.beam_calo_pos.z >= truncate
    return sample.recoParticles.beam_dEdX[truncated_start]


def GetTruncatedPos(sample : Data, truncate : float) -> tuple[ak.Array, ak.Array]:
    if truncate is not None:
        truncated_start = ak.argmax(sample.recoParticles.beam_calo_pos.z >= truncate, -1, keepdims = True)
        start_pos = sample.recoParticles.beam_calo_pos[truncated_start]
        start_pos = ak.flatten(start_pos)

        end_pos = sample.recoParticles.beam_calo_pos[:, -1:]
        end_pos = ak.pad_none(end_pos, 1, -1)
        end_pos = ak.flatten(end_pos)
    else:
        start_pos = sample.recoParticles.beam_startPos_SCE
        end_pos = sample.recoParticles.beam_endPos_SCE

    return start_pos, end_pos


def BeamTriggerSelection(events: Data, pdgs : list[int] = [211, 13, -13], use_beam_inst : bool = False, return_property : bool = False):
    """ Beam particle selection using beam instrumentation information for Data and truth information if MC.

    Args:
        events (Master.Data): events to study
        pdgs (list[int], optional): list of particle pdgs. Defaults to [211, 13, -13].
        use_beam_inst (bool, optional): use beam instrumentation (enable if using data). Defaults to False.
        return_property (bool, optional): return beam pdg. Defaults to False.
    """
    def compare_beam_pdg(pdg : int):
        return ak.fill_none(ak.pad_none(events.recoParticles.beam_inst_PDG_candidates, 1, -1), -1, -1) == pdg

    if use_beam_inst:
        mask = events.recoParticles.beam_inst_valid
        mask = mask & (events.recoParticles.beam_inst_trigger != 8)
        mask = mask & (events.recoParticles.beam_inst_nTracks == 1) & (events.recoParticles.beam_inst_nMomenta == 1)

        beam_pdg = None
        for i in pdgs:
            tmp = compare_beam_pdg(i)
            if beam_pdg is None:
                beam_pdg = tmp
            else:
                beam_pdg = beam_pdg | tmp
        mask = mask & ak.any(beam_pdg, axis = -1)
        mask = mask & events.recoParticles.reco_reconstructable_beam_event
    else:
        beam_pdg = ak.flatten(events.trueParticles.pdg[events.trueParticles.number == 1])
        mask = ak.any([beam_pdg == i for i in pdgs], axis = 0)
    if return_property is True:
        return mask, beam_pdg
    else:
        return mask


def TrueFiducialCut(events, is_mc : bool, cut : int = [30, 220], op = [">", "<"], return_property : bool = False):
    if is_mc:
        return CreateMask(cut, op, events.trueParticles.endPos.z[:, 0], return_property)
    else:
        if return_property == True:
            return ak.ones_like(events.eventNum, dtype = bool), ak.zeros_like(events.eventNum, dtype = bool)
        else:
            return ak.ones_like(events.eventNum, dtype = bool)

def FiducialStart(events, cut : int = 0, op = ">", return_property : bool = False):
    return CreateMask(cut, op, events.recoParticles.beam_endPos_SCE.z, return_property)

def PiBeamSelection(events: Data, use_beam_inst : bool = False, return_property : bool = False) -> ak.Array:
    """
    Legacy Pi+ beam particle selection.
    Used during the normalisation calculation (apps.cex_normalisation)

    Args:
        events (Data): events to study.
        use_beam_inst (bool): use beam instrumentation for PID (data)

    Returns:
        ak.Array: boolean mask.
    """

    return BeamTriggerSelection(events, [211, 13, -13], use_beam_inst, return_property)


def PandoraTagCut(events: Data, cut : int = 13, op = "==", return_property : bool = False) -> ak.Array:
    """ Cut on Pandora slice tag, selects track like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    return CreateMask(cut, op, events.recoParticles.beam_pandora_tag, return_property)


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
    start_pos, end_pos = GetTruncatedPos(events, fits["truncate"])

    beam_dx = (start_pos.x - fits["mu_x"]) / fits["sigma_x"]
    beam_dy = (start_pos.y - fits["mu_y"]) / fits["sigma_y"]
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5
    beam_dz = (start_pos.z - fits["mu_z"]) / fits["sigma_z"]

    beam_dir = vector.normalize(vector.sub(end_pos, start_pos))

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


def DxyCut(events: Data, fits : dict, cut, op = "<", return_property : bool = False):
    """ Cut on start position transverse to the z direction.

    Args:
        events (Data): events to study
        fits (dict): fit parameter values
        cut (_type_): cut value
        op (str, optional): operation for cut. Defaults to "<".
        return_property (bool, optional): return transverse start position. Defaults to False.

    Returns:
        mask and or property cut on.
    """
    start_pos = GetTruncatedPos(events, fits["truncate"])[0]

    beam_dx = (start_pos.x - fits["mu_x"]) / fits["sigma_x"]
    beam_dy = (start_pos.y - fits["mu_y"]) / fits["sigma_y"]
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5
    return CreateMask(cut, op, beam_dxy, return_property)


def DzCut(events: Data, fits : dict, cut, op = [">", "<"], return_property : bool = False):
    """ Cut on start z position.

    Args:
        events (Data): events to study
        fits (dict): fit parameter values
        cut: cut value
        op (str, optional): operation for cut. Defaults to [">", "<"].
        return_property (bool, optional): return transverse start position. Defaults to False.

    Returns:
        mask and or property cut on.
    """
    start_pos = GetTruncatedPos(events, fits["truncate"])[0]

    beam_dz = (start_pos.z - fits["mu_z"]) / fits["sigma_z"]
    return CreateMask(cut, op, beam_dz, return_property)


def CosThetaCut(events: Data, fits : dict, cut, op = ">", return_property : bool = False):
    """ Cut on direction of beam particle wrt to the average direction.

    Args:
        events (Data): events to study
        fits (dict): fit parameter values
        cut: cut value
        op (str, optional): operation for cut. Defaults to ">".
        return_property (bool, optional): return transverse start position. Defaults to False.

    Returns:
        mask and or property cut on.
    """
    start_pos, end_pos = GetTruncatedPos(events, fits["truncate"])

    beam_dir = vector.normalize(vector.sub(end_pos, start_pos))

    beam_dir_mc = vector.vector(
        fits["mu_dir_x"],
        fits["mu_dir_y"],
        fits["mu_dir_z"]
    )
    beam_dir_mc = vector.normalize(beam_dir_mc)

    beam_costh = vector.dot(beam_dir, beam_dir_mc)
    return CreateMask(cut, op, beam_costh, return_property)


def APA3Cut(events: Data, cut : float = 220, op = "<", return_property : bool = False) -> ak.Array:
    """ Cuts on beam end z position to select beam particles which end in APA3.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    return CreateMask(cut, op, events.recoParticles.beam_endPos_SCE.z, return_property)


def MichelScoreCut(events: Data, cut : float = 0.55, op = "<", return_property : bool = False) -> ak.Array:
    """ Cut on michel score to remove muon like beam particles.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    score = ak.where(events.recoParticles.beam_nHits != 0, events.recoParticles.beam_michelScore / events.recoParticles.beam_nHits, -999)
    return CreateMask(cut, op, score, return_property)

def MichelScoreCutChargeWeight(
        events: Data, cut : float = 0.55, op = "<",
        return_property : bool = False) -> ak.Array:
    """ Cut on michel score to remove muon like beam particles.
    Uses the charge weighted version.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    score = ak.where(
        events.recoParticles.beam_nHits != 0,
        (events.recoParticles.beam_michelScore_by_charge
         / events.recoParticles.beam_nHits),
        -999)
    return CreateMask(cut, op, score, return_property)

def MedianDEdXCut(events: Data, cut : float = 2.4, op = "<", truncate : float = None, return_property : bool = False) -> ak.Array:
    """ cut on median dEdX to exlude proton background.

    Args:
        events (Data): events to study.

    Returns:
        ak.Array: boolean mask.
    """
    if truncate is None:
        dEdX = events.recoParticles.beam_dEdX
    else:
        dEdX = GetTruncatedDEdX(events, truncate)
    median = Median(dEdX)
    return CreateMask(cut, op, median, return_property)


def BeamScraperCut(events : Data, KE_range : int, fits : dict, cut : float = 1.5, op = "<", return_property : bool = False) -> ak.Array:
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
    return CreateMask(cut, op, r, return_property)


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
        HasFinalStatePFOsCut,
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
        {},
        {"fits" : beam_quality_fits},
        {},
        {}
    ]
    if scraper is True:
        selection.append(BeamScraperCut)
        arguments.append({"KE_range" : scraper_KE_range, "fits" : scraper_fits, "cut" : scraper_cut})
    return CombineSelections(events, selection, 0, arguments, verbose, return_table)
