"""
Created on: 07/03/2023 14:43

Author: Shyam Bhuller

Description: Lists cuts which selects PFOs in an event or events.
TODO Documentation.
"""
import awkward as ak
import numpy as np

from python.analysis import vector
from python.analysis.Master import Data
from python.analysis.SelectionTools import *


@CountsWrapper
def ValidRecoEnergyCut(events : Data) -> ak.Array:
    return events.recoParticles.energy != -999

@CountsWrapper
def ValidRecoPositionCut(events : Data) -> ak.Array:
    return np.logical_and(
                np.logical_and(
                    events.recoParticles.startPos.x != -999,
                    events.recoParticles.startPos.y != -999
                ),
                events.recoParticles.startPos.z != 999
            )

@CountsWrapper
def ValidRecoMomentumCut(events : Data) -> ak.Array:
    return np.logical_and(
                np.logical_and(
                    events.recoParticles.momentum.x != -999,
                    events.recoParticles.momentum.y != -999
                ),
                events.recoParticles.momentum.z != -999
            )

@CountsWrapper
def ValidCNNScoreCut(events : Data) -> ak.Array:
    return events.recoParticles.cnnScore != -999


def GoodShowerSelection(events : Data, return_table : bool = False):
    selections = [
        ValidRecoPositionCut,
        ValidRecoMomentumCut,
        ValidRecoEnergyCut,
        ValidCNNScoreCut,
    ]
    return CombineSelections(events, selections, 1, return_table = return_table)

@CountsWrapper
def EMScoreCut(events : Data) -> ak.Array:
    return events.recoParticles.emScore > 0.5

@CountsWrapper
def NHitsCut(events : Data) -> ak.Array:
    return events.recoParticles.nHits > 80

@CountsWrapper
def BeamParticleDistanceCut(events : Data) -> ak.Array:
    dist = vector.magnitude(vector.sub(events.recoParticles.startPos, events.recoParticles.beam_endPos)) # distance to beam end position in cm
    return (dist > 3) & (dist < 90)

@CountsWrapper
def BeamParticleIPCut(events : Data) -> ak.Array:
    ip = vector.magnitude(
            vector.cross(
                vector.sub(
                        events.recoParticles.beam_endPos,
                        events.recoParticles.startPos
                ),
                events.recoParticles.direction
            )
        )    
    return ip < 20


def InitialPi0PhotonSelection(events : Data, verbose : bool = False, return_table : bool = False):
    selections = [
        EMScoreCut,
        NHitsCut,
        BeamParticleDistanceCut,
        BeamParticleIPCut
    ]
    return CombineSelections(events, selections, 1, verbose, return_table)