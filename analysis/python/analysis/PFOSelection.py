"""
Created on: 07/03/2023 14:43

Author: Shyam Bhuller

Description: Lists cuts which selects PFOs in an event or events.
TODO Documentation.
"""
import awkward as ak
import numpy as np

from python.analysis import vector
from python.analysis.SelectionTools import *


#######################################################################
#######################################################################
##########                   PFO SELECTION                   ##########
#######################################################################
#######################################################################


@CountsWrapper
def ValidRecoEnergyCut(events) -> ak.Array:
    return events.recoParticles.energy != -999


@CountsWrapper
def ValidRecoPositionCut(events) -> ak.Array:
    return np.logical_and(
        np.logical_and(
            events.recoParticles.startPos.x != -999,
            events.recoParticles.startPos.y != -999
        ),
        events.recoParticles.startPos.z != 999
    )


@CountsWrapper
def ValidRecoMomentumCut(events) -> ak.Array:
    return np.logical_and(
        np.logical_and(
            events.recoParticles.momentum.x != -999,
            events.recoParticles.momentum.y != -999
        ),
        events.recoParticles.momentum.z != -999
    )


@CountsWrapper
def ValidCNNScoreCut(events) -> ak.Array:
    return events.recoParticles.cnnScore != -999


def GoodShowerSelection(events, return_table=False):
    selections = [
        ValidRecoPositionCut,
        ValidRecoMomentumCut,
        ValidRecoEnergyCut,
        ValidCNNScoreCut,
    ]
    return CombineSelections(events, selections, 1, return_table=return_table)


@CountsWrapper
def EMScoreCut(events, score=0.5) -> ak.Array:
    return events.recoParticles.emScore > score


@CountsWrapper
def NHitsCut(events, hits=80) -> ak.Array:
    return events.recoParticles.nHits > hits


@CountsWrapper
def BeamParticleDistanceCut(events, lims=(3., 90.)) -> ak.Array:
    # distance to beam end position in cm
    dist = find_beam_separations(events)
    return (dist > lims[0]) & (dist < lims[1])


@CountsWrapper
def BeamParticleIPCut(events, impact=20.) -> ak.Array:
    ip = find_beam_impact_parameters(events)
    return ip < impact


def InitialPi0PhotonSelection(
        events,
        em_cut: float = 0.5,
        n_hits_cut: int = 80,
        distance_bounds_cm: tuple = (3., 90.),
        max_impact_cm: float = 20.,
        verbose: bool = False,
        return_table: bool = False):
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
    Finds the separations of eachstart position of each reconstructed
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
