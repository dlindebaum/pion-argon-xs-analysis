"""
Created on: 07/03/2023 14:43

Author: Shyam Bhuller

Description: Lists cuts which selects PFOs in an event or events.
TODO Documentation.
"""
import awkward as ak
import numpy as np

from python.analysis import vector, Master
from python.analysis.SelectionTools import *


#######################################################################
#######################################################################
##########                   PFO SELECTION                   ##########
#######################################################################
#######################################################################


def Median(x : ak.Array):
    s = ak.sort(x, -1)
    count = ak.num(s, -1)

    med_odd = count // 2  # median is middle entry
    select = ak.local_index(s) == med_odd
    median_odd = s[select]

    # calculate the median assuming the arrray length is even
    med_even = (med_odd - 1) * (count > 1)  # need the middle - 1 value
    select_even = ak.local_index(s) == med_even
    # median is average of middle value and middle - 1 value
    median_even = (s[select] + s[select_even]) / 2

    median = ak.flatten(ak.fill_none(ak.pad_none(ak.where(count % 2, median_odd, median_even), 1, -1), -999), -1)  # pick which median is the correct one
    return median


@CountsWrapper
def ValidRecoEnergyCut(events : Master.Data) -> ak.Array:
    return events.recoParticles.energy != -999


@CountsWrapper
def ValidRecoPositionCut(events : Master.Data) -> ak.Array:
    return np.logical_and(
        np.logical_and(
            events.recoParticles.startPos.x != -999,
            events.recoParticles.startPos.y != -999
        ),
        events.recoParticles.startPos.z != 999
    )


@CountsWrapper
def ValidRecoMomentumCut(events : Master.Data) -> ak.Array:
    return np.logical_and(
        np.logical_and(
            events.recoParticles.momentum.x != -999,
            events.recoParticles.momentum.y != -999
        ),
        events.recoParticles.momentum.z != -999
    )


@CountsWrapper
def ValidCNNScoreCut(events : Master.Data) -> ak.Array:
    return events.recoParticles.cnnScore != -999


def GoodShowerSelection(events : Master.Data, return_table = False):
    selections = [
        ValidRecoPositionCut,
        ValidRecoMomentumCut,
        ValidRecoEnergyCut,
    ]
    return CombineSelections(events, selections, 1, return_table = return_table)


@CountsWrapper
def EMScoreCut(events : Master.Data, score = 0.5) -> ak.Array:
    return events.recoParticles.emScore > score


@CountsWrapper
def NHitsCut(events : Master.Data, hits = 80) -> ak.Array:
    return events.recoParticles.nHits > hits


@CountsWrapper
def BeamParticleDistanceCut(events : Master.Data, lims = (3., 90.)) -> ak.Array:
    # distance to beam end position in cm
    dist = find_beam_separations(events)
    return (dist > lims[0]) & (dist < lims[1])


@CountsWrapper
def BeamParticleIPCut(events : Master.Data, impact = 20.) -> ak.Array:
    ip = find_beam_impact_parameters(events)
    return ip < impact


@CountsWrapper
def TrackScoreCut(events : Master.Data, score = 0.5):
    return events.recoParticles.trackScore > score


@CountsWrapper
def BeamDaughterCut(events : Master.Data):
    beam_number = ak.where(events.recoParticles.beam_number == -999, -1, events.recoParticles.beam_number) # null beam number (no beam particle) and null mother (pf has no mother) are the same value, so change one of them to something else
    return events.recoParticles.mother == beam_number


@CountsWrapper
def VetoBeamParticle(events : Master.Data):
    return events.recoParticles.beam_number != events.recoParticles.number


@CountsWrapper
def PiPlusSelection(events : Master.Data, min_dEdX = 0.5, max_dEdX = 2.8):
    median_dEdX = Median(events.recoParticles.track_dEdX)
    return (median_dEdX < max_dEdX) & (median_dEdX > min_dEdX)


@CountsWrapper
def ProtonSelection(events : Master.Data, dEdX):
    median_dEdX = Median(events.recoParticles.track_dEdX)
    return median_dEdX > dEdX


def InitialPi0PhotonSelection(
        events : Master.Data,
        em_cut : float = 0.5,
        n_hits_cut : int = 80,
        distance_bounds_cm : tuple = (3., 90.),
        max_impact_cm : float = 20.,
        veto_beam_particle : bool = True,
        verbose : bool = False,
        return_table : bool = False):
    selections = [
        EMScoreCut,
        NHitsCut,
        BeamParticleDistanceCut,
        BeamParticleIPCut
    ]
    arguments = [
        {"score": em_cut},
        {"hits": n_hits_cut},
        {"lims": distance_bounds_cm},
        {"impact": max_impact_cm}
    ]
    if veto_beam_particle:
        selections.append(VetoBeamParticle)
        arguments.append({})
    return CombineSelections(events, selections, 1, arguments, verbose, return_table)


def DaughterPiPlusSelection(
        events : Master.Data,
        track_cut : float = 0.5,
        n_hits_cut : int = 20,
        min_dEdX : float = 0.5,
        max_dEdX : float = 2.8,
        verbose : bool = False,
        return_table : bool = False):
    selections = [
        TrackScoreCut,
        NHitsCut,
        PiPlusSelection
    ]
    arguments = [
        {"score" : track_cut},
        {"hits" : n_hits_cut},
        {"min_dEdX" : min_dEdX, "max_dEdX" : max_dEdX}
    ]
    if events.nTuple_type == Master.Ntuple_Type.SHOWER_MERGING:
        print("Apply beam daughter cuts")
        selections.insert(0, BeamDaughterCut)
        arguments.insert(0, {}) # the cut changes slightly depending on the ntuple type we use.
    return CombineSelections(events, selections, 1, arguments, verbose, return_table)


def DaughterProtonSelection(
        events : Master.Data,
        track_cut : float = 0.5,
        n_hits_cut : int = 20,
        dEdX : float = 2.8,
        verbose : bool = False,
        return_table : bool = False):
    selections = [
        TrackScoreCut,
        NHitsCut,
        ProtonSelection
    ]
    arguments = [
        {"score" : track_cut},
        {"hits" : n_hits_cut},
        {"dEdX" : dEdX}
    ]
    if events.nTuple_type == Master.Ntuple_Type.SHOWER_MERGING:
        selections.insert(0, BeamDaughterCut)
        arguments.insert(0, {}) # the cut changes slightly depending on the ntuple type we use.
    return CombineSelections(events, selections, 1, arguments, verbose, return_table)

#######################################################################
#######################################################################
##########                  PFO PROPERTIES                   ##########
#######################################################################
#######################################################################


def get_impact_parameter(direction, start_pos, beam_vertex):
    """
    Finds the impact parameter between a PFO and beam vertex.

    Parameters
    ----------
    direction : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Direction of PFO.
    start_pos : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of the PFO.
    beam_vertex : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Beam interaction point.

    Returns
    -------
    m : ak.Array
        Impact parameter between the PFO and beam vertex.
    """
    rel_pos = vector.sub(beam_vertex, start_pos)
    cross = vector.cross(rel_pos, direction)
    return vector.magnitude(cross)


def find_beam_impact_parameters(events):
    """
    Finds the impact parameter of each reconstructed PFO in `events`
    with the reconstructed beam vertex.

    Parameters
    ----------
    events : Data
        Events containing PFOs for which to calculate the impact
        parameters with the beam.

    Returns
    -------
    impacts : ak.Array
        Impact parameters between the PFOs and beam vertex.
    """
    starts = events.recoParticles.startPos
    beam_vertex = events.recoParticles.beam_endPos
    directions = events.recoParticles.direction
    return get_impact_parameter(
        directions, starts, beam_vertex)


def find_beam_separations(events):
    """
    Finds the separations of each start position of each reconstructed
    PFO in `events` and the reconstructed beam vertex.

    Parameters
    ----------
    events : Data
        Events containing PFOs for which to calcualte the beam
        separations.

    Returns
    -------
    separations : ak.Array
        Separations between the PFOs starts and beam vertex.
    """
    starts = events.recoParticles.startPos
    beam_vertex = events.recoParticles.beam_endPos
    return vector.dist(starts, beam_vertex)


def get_mother_pdgs(events):
    """
    DEPRECATED: use events.trueParticles(BT).motherPdg

    Loops through each event (warning: slow) and creates a reco-type
    array of PDG codes of mother particle. E.g. a photon from a pi0
    will be assigned a PDG code 111.

    Mother particle is defined as the mother of the truth particle
    which the PFO was backtracked to.

    Beam particle is hard-coded to have a PDG code of 211.

    Anything which cannot have a mother PDG code assigned is given 0.

    Future warning: Mother PDG codes are planned to be added into the
    ntuple data making this function redundant.

    Parameters
    ----------
    events : Data
        Set of events to produce the mother PDG codes for.

    Returns
    -------
    mother_pdgs : ak.Array
        PDG code of the mother of the truth particle that backtracked
        to the PFO.
    """
    # This loops through every event and assigned the pdg code of the
    # mother. Assigns 211 (pi+) to daughters of the beam, and 0 to
    # everything which is not found
    mother_ids = events.trueParticles.mother
    truth_ids = events.trueParticles.number
    truth_pdgs = events.trueParticles.pdg
    mother_pdgs = mother_ids.to_list()
    # ts = time.time()
    for i in range(ak.num(mother_ids, axis=0)):
        true_pdg_lookup = {
            truth_ids[i][d]:
            truth_pdgs[i][d] for d in range(ak.count(truth_ids[i]))}
        true_pdg_lookup.update({0: 0, 1: 211})
        # I have no idea why the hell I did this
        # # Presumably the beam particle gets a strange pdg code??
        for j in range(ak.count(mother_ids[i])):
            try:
                mother_pdgs[i][j] = true_pdg_lookup[mother_ids[i][j]]
            except:
                mother_pdgs[i][j] = 0
    # print(f"Mother PDG codes found in {time.time()  - ts}s")
    mother_pdgs = ak.Array(mother_pdgs)
    return mother_pdgs
