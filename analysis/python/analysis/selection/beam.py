"""
Created on: 03/03/2023 14:30

Author: Shyam Bhuller

Description: Contains cuts for Beam Particle Selection.
TODO depreciate BeamQualityCut.
"""
import awkward as ak
import numpy as np

from .common import *
from .pfo import GoodShowerSelection

from python.analysis import vector
from python.analysis.ntuple import Data

def get_truncated_dEdX(sample : Data, truncate : float):
    """ Return beam calorimetry dE/dx values at or downstream of a z threshold.

        Args:
            sample (Data): Event sample containing beam calorimetry positions and dE/dx.
            truncate (float): Minimum calorimetry z position used to select dE/dx values.

        Returns:
            Any: Beam dE/dx values whose corresponding calorimetry points satisfy the z threshold.
    """
    truncated_start = sample.recoParticles.beam_calo_pos.z >= truncate
    return sample.recoParticles.beam_dEdX[truncated_start]


def get_truncated_pos(sample : Data, truncate : float) -> tuple[ak.Array, ak.Array]:
    """ Get beam trajectory endpoints, optionally starting at the first calorimetry point past a z threshold.

        Args:
            sample (Data): Event sample containing beam positions and calorimetry points.
            truncate (float): Optional minimum calorimetry z position for the truncated start point.

        Returns:
            tuple[ak.Array, ak.Array]: Start and end position arrays used to calculate the beam trajectory.
    """
    if truncate is not None:
        truncated_start = ak.argmax(sample.recoParticles.beam_calo_pos.z >= truncate, -1, keepdims = True)
        start_pos = sample.recoParticles.beam_calo_pos[truncated_start]
        start_pos = ak.fill_none(start_pos, vector.null_vector[0])
        start_pos = ak.flatten(start_pos)

        end_pos = sample.recoParticles.beam_calo_pos[:, -1:]
        end_pos = ak.pad_none(end_pos, 1, -1)
        end_pos = ak.fill_none(end_pos, vector.null_vector[0])
        end_pos = ak.flatten(end_pos)
    else:
        start_pos = sample.recoParticles.beam_start_pos_sce
        end_pos = sample.recoParticles.beam_end_pos_sce # these should already be populaed with null vectors.

    return start_pos, end_pos


def BeamTriggerSelection(events: Data, pdgs : list[int] = [211, 13, -13], use_beam_inst : bool = False, return_property : bool = False):
    """ Beam particle selection using beam instrumentation information for Data and truth information if MC.

        Args:
            events (Data): Event sample containing beam information.
            pdgs (list[int]): Accepted beam-particle PDG codes; defaults to charged pions and muons.
            use_beam_inst (bool): Whether to identify the beam using instrumentation rather than truth particles.
            return_property (bool): If true, return the beam PDG values alongside the event mask.

        Returns:
            Any: Event-level beam-selection mask, optionally paired with beam PDG values.
    """
    def compare_beam_pdg(pdg : int):
        """ Compare beam pdg.

            Args:
                pdg (int): Beam-particle PDG code to match against the instrumentation candidates.

            Returns:
                    Any: Per-event mask identifying instrumentation candidates with this PDG code.
        """
        return ak.fill_none(ak.pad_none(events.recoParticles.beam_inst_pdg_candidates, 1, -1), -1, -1) == pdg

    if use_beam_inst:
        mask = events.recoParticles.beam_inst_valid
        mask = mask & (events.recoParticles.beam_inst_trigger != 8)
        mask = mask & (events.recoParticles.beam_inst_n_tracks == 1) & (events.recoParticles.beam_inst_n_momenta == 1)

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
        beam_pdg = ak.flatten(events.true_particles.pdg[events.true_particles.number == 1])
        mask = ak.any([beam_pdg == i for i in pdgs], axis = 0)
    if return_property is True:
        return mask, beam_pdg
    else:
        return mask


def TrueFiducialCut(events, is_mc : bool, cut : int = [30, 220], op = [">", "<"], return_property : bool = False):
    """ Select MC events by the truth beam particle's end z coordinate; data events pass unchanged.

        Args:
            events (Any): Event sample with truth beam-particle end positions.
            is_mc (bool): Whether the sample is MC; non-MC samples receive an all-true mask.
            cut (int): Threshold or bounds applied to the truth beam end z coordinate.
            op (Any): Comparison operator, or operators corresponding to the bounds.
            return_property (bool): If true, also return the cut-on property (zero-valued for data).

        Returns:
            Any: Event-level mask, optionally paired with the truth end-z values.
    """
    if is_mc:
        return create_mask(cut, op, events.true_particles.end_pos.z[:, 0], return_property)
    else:
        if return_property == True:
            return ak.ones_like(events.event_num, dtype = bool), ak.zeros_like(events.event_num, dtype = bool)
        else:
            return ak.ones_like(events.event_num, dtype = bool)


def PiBeamSelection(events: Data, use_beam_inst : bool = False, return_property : bool = False) -> ak.Array:
    """ Legacy Pi+ beam particle selection.

        Args:
            events (Data): Event sample containing beam identification information.
            use_beam_inst (bool): Whether to use beam instrumentation for particle identification.
            return_property (bool): If true, also return the identified beam PDG values.

        Returns:
            ak.Array: Boolean mask.
    """

    return BeamTriggerSelection(events, [211, 13, -13], use_beam_inst, return_property)


def PandoraTagCut(events: Data, cut : int = 13, op = "==", return_property : bool = False) -> ak.Array:
    """ Cut on Pandora slice tag, selects track like beam particles.

        Args:
            events (Data): Events to study.
            cut (int): Pandora slice-tag value or values to compare.
            op (Any): Comparison operator applied to the slice tag.
            return_property (bool): If true, return slice-tag values alongside the mask.

        Returns:
            ak.Array: Boolean mask.
    """
    return create_mask(cut, op, events.recoParticles.beam_pandora_tag, return_property)


def CaloSizeCut(events: Data, return_property : bool = False) -> ak.Array:
    """ Cut which checks the beam particle has a calorimetry object (required for median dEdX cut).

        Args:
            events (Data): Events to study.
            return_property (bool): If true, return the number of valid calorimetry points alongside the mask.

        Returns:
            ak.Array: Boolean mask.
    """
    calo_wire = events.recoParticles.beam_calo_wire
    # Analyser fills the empty entry with a -999
    calo_wire = calo_wire[calo_wire != -999]

    return create_mask(0, ">", ak.num(calo_wire, 1), return_property)


def BeamQualityCut(events: Data, fits : dict, dxy_cut : list = [-3, 3], dz_cut : list = [-3, 3], costh_cut : list = [0.95, 2]) -> ak.Array:
    """ Cut on beam particle start position and trajectory,
                     Selects beam particles with values consistent to the beam plug.

        Args:
            events (Data): Events to study.
            fits (dict): Beam-fit means, widths, reference direction, and calorimetry truncation position.
            dxy_cut (list): Lower and upper bounds on normalized transverse displacement from the fitted beam position.
            dz_cut (list): Lower and upper bounds on normalized longitudinal displacement from the fitted beam position.
            costh_cut (list): Lower and upper bounds on the cosine between the measured and fitted beam directions.

        Returns:
            ak.Array: Boolean mask.
    """
    start_pos, end_pos = get_truncated_pos(events, fits["truncate"])

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
        """ Keep values strictly between the supplied lower and upper bounds.

            Args:
                x (Any): Normalized beam-position or direction value being checked against the fit bounds.
                xmin (Any): Exclusive lower bound for the measured beam-quality value.
                xmax (Any): Exclusive upper bound for the measured beam-quality value.

            Returns:
                Any: Boolean mask for values strictly between the two bounds.
        """
        return ((x > xmin) & (x < xmax))

    beam_quality_mask = cut(beam_dz, min(dz_cut), max(dz_cut))
    beam_quality_mask = beam_quality_mask & cut(beam_dxy, min(dxy_cut), max(dxy_cut))
    beam_quality_mask = beam_quality_mask & cut(beam_costh, min(costh_cut), max(costh_cut))
    return beam_quality_mask


def DxyCut(events: Data, fits : dict, cut, op = "<", return_property : bool = False):
    """ Cut on start position transverse to the z direction.

        Args:
            events (Data): Events to study.
            fits (dict): Fit parameter values.
            cut (Any): Threshold or thresholds applied to the normalized transverse displacement.
            op (str, optional): Comparison operator(s) used for the displacement cut. Defaults to "<".
            return_property (bool): Whether to return the displacement values with the mask. Defaults to False.

        Returns:
            Any: Event-level mask, optionally paired with normalized transverse displacement values.
    """
    start_pos = get_truncated_pos(events, fits["truncate"])[0]

    beam_dx = (start_pos.x - fits["mu_x"]) / fits["sigma_x"]
    beam_dy = (start_pos.y - fits["mu_y"]) / fits["sigma_y"]
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5
    return create_mask(cut, op, beam_dxy, return_property)


def DzCut(events: Data, fits : dict, cut, op = [">", "<"], return_property : bool = False):
    """ Cut on start z position.

        Args:
            events (Data): Events to study.
            fits (dict): Fit parameter values.
            cut (Any): Threshold or bounds applied to normalized longitudinal displacement.
            op (str, optional): Comparison operator(s) used for the displacement cut. Defaults to [">", "<"].
            return_property (bool): Whether to return normalized displacement values with the mask. Defaults to False.

        Returns:
            Any: Event-level mask, optionally paired with normalized longitudinal displacement values.
    """
    start_pos = get_truncated_pos(events, fits["truncate"])[0]

    beam_dz = (start_pos.z - fits["mu_z"]) / fits["sigma_z"]
    return create_mask(cut, op, beam_dz, return_property)


def CosThetaCut(events: Data, fits : dict, cut, op = ">", return_property : bool = False):
    """ Cut on direction of beam particle wrt to the average direction.

        Args:
            events (Data): Events to study.
            fits (dict): Fit parameter values.
            cut (Any): Threshold or bounds applied to the cosine with the fitted beam direction.
            op (str, optional): Comparison operator(s) for the cosine cut. Defaults to ">".
            return_property (bool): Whether to return cosine values with the mask. Defaults to False.

        Returns:
            Any: Event-level mask, optionally paired with direction-cosine values.
    """
    start_pos, end_pos = get_truncated_pos(events, fits["truncate"])

    beam_dir = vector.normalize(vector.sub(end_pos, start_pos))

    beam_dir_mc = vector.vector(
        fits["mu_dir_x"],
        fits["mu_dir_y"],
        fits["mu_dir_z"]
    )
    beam_dir_mc = vector.normalize(beam_dir_mc)

    beam_costh = vector.dot(beam_dir, beam_dir_mc)
    return create_mask(cut, op, beam_costh, return_property)


def APA3Cut(events: Data, cut : float = 220, op = "<", return_property : bool = False) -> ak.Array:
    """ Cuts on beam end z position to select beam particles which end in APA3.

        Args:
            events (Data): Events to study.
            cut (float): Threshold or thresholds applied to the beam end z coordinate.
            op (Any): Comparison operator, or operators for a range cut.
            return_property (bool): If true, return beam end z values alongside the mask.

        Returns:
            ak.Array: Boolean mask.
    """
    return create_mask(cut, op, events.recoParticles.beam_end_pos_sce.z, return_property)


def MichelScoreCut(events: Data, cut : float = 0.55, op = "<", return_property : bool = False) -> ak.Array:
    """ Cut on michel score to remove muon like beam particles.

        Args:
            events (Data): Events to study.
            cut (float): Threshold or thresholds applied to Michel score per beam hit.
            op (Any): Comparison operator, or operators for a range cut.
            return_property (bool): If true, return the normalized Michel score alongside the mask.

        Returns:
            ak.Array: Boolean mask.
    """
    score = ak.where(events.recoParticles.beam_n_hits != 0, events.recoParticles.beam_michel_score / events.recoParticles.beam_n_hits, -999)
    return create_mask(cut, op, score, return_property)


def MedianDEdXCut(events: Data, cut : float = 2.4, op = "<", truncate : float = None, return_property : bool = False) -> ak.Array:
    """ Cut on median dEdX to exlude proton background.

        Args:
            events (Data): Events to study.
            cut (float): Threshold or thresholds applied to the median beam dE/dx.
            op (Any): Comparison operator, or operators for a range cut.
            truncate (float): Optional z position at which to begin the calorimetry dE/dx calculation.
            return_property (bool): If true, return median dE/dx values alongside the mask.

        Returns:
            ak.Array: Boolean mask.
    """
    if truncate is None:
        dEdX = events.recoParticles.beam_dEdX
    else:
        dEdX = get_truncated_dEdX(events, truncate)
    med = median(dEdX)
    return create_mask(cut, op, med, return_property)


def BeamScraperCut(events : Data, KE_range : int, fits : dict, cut : float = 1.5, op = "<", return_property : bool = False) -> ak.Array:
    """ Beam scraper cut. Required to exclude events with poor consistency between
                     the beam insturmention KE and front facing KE
                     (front facing means the first calorimetry point in the TPC).

        Args:
            events (Data): Events to study.
            KE_range (int): Index of kinetic energy range in fit_values.
            fits (dict): Beam-instrumentation fit parameters for the selected kinetic-energy bin.
            cut (float): Threshold on the normalized radial position discrepancy; defaults to 1.5.
            op (Any): Comparison operator, or operators for a range cut.
            return_property (bool): If true, return the normalized discrepancy alongside the mask.

        Returns:
            ak.Array: Event-level mask, optionally paired with normalized scraper-position discrepancy values.
    """

    key = str(KE_range - 1)
    mu_x = fits[key]["mu_x_inst"]
    mu_y = fits[key]["mu_y_inst"]
    sigma_x = fits[key]["sigma_x_inst"]
    sigma_y = fits[key]["sigma_y_inst"]

    nx = (events.recoParticles.beam_inst_pos.x - mu_x)/sigma_x
    ny = (events.recoParticles.beam_inst_pos.y - mu_y)/sigma_y

    r = np.sqrt(nx**2 + ny**2)
    return create_mask(cut, op, r, return_property)


def HasFinalStatePFOsCut(events: Data, return_property : bool = False) -> ak.Array:
    """ Selects events which have final state PFOs that are well reconstructed.

        Args:
            events (Data): Events to look at.
            return_property (bool): If true, return the number of quality-selected PFOs with the mask.

        Returns:
            ak.Array: Event-level mask identifying samples with at least one quality-selected PFO.
    """
    pfo_mask = GoodShowerSelection(events) # create this mask internally so that the PFO and event selections remain separate
    nPFO = ak.num(events.recoParticles.number[pfo_mask])
    return create_mask(0, ">", nPFO, return_property)


def create_default_selection(events: Data,
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
            events (Data): Events to study.
            use_beam_inst (bool): Use beam instrumentation for particle_id.
            beam_quality_fits (dict): Fit parameters for beam quality cuts.
            scraper (bool): Whether to include the beam-instrumentation scraper cut.
            scraper_fits (dict): Fit parameters indexed by kinetic-energy bin for the scraper cut.
            scraper_KE_range (int): Kinetic-energy range index selecting the scraper-fit parameters.
            scraper_cut (float): Threshold on the normalized scraper-position discrepancy.
            verbose (bool): Verbose printout.
            return_table (bool): Whether to return a cut-performance table with the combined mask.

        Returns:
            Any: Combined event mask, optionally paired with the cut-performance table when requested.
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
    return combine_selections(events, selection, 0, arguments, verbose, return_table)
