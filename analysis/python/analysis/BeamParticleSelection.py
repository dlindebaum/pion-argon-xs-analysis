"""
Created on: 03/03/2023 14:30

Author: Shyam Bhuller

Description: Contains cuts for Beam Particle Selection
TODO Cleanup beam quality cut code
TODO Add property methods for the new Ntuples in Master.Data
TODO Add documentation
? Should the cuts be configurable?
? should this be kept in a class?
"""
import awkward as ak
import numpy as np

import python.analysis.vector as vector
from python.analysis.Master import Data


def CountEvents(f):
    def wrapper(*args, **kwargs):
        m = f(*args, **kwargs)
        print(m)
        print(f"number of events before|after {f.__name__}: {ak.count(m)}|{ak.count(m[m])}")
        return m
    return wrapper


#! This is a pi+ beam selection, should this be here?
@CountEvents
def PiBeamSelection(events : Data):
    # beam particle PDG cut
    pdg = events.io.Get("reco_beam_PFP_true_byHits_pdg")
    return (pdg == 211) | (pdg == -13)


@CountEvents
def PandoraTagCut(events : Data) -> ak.Array:
    # pandora tag cut
    tag = events.recoParticles.pandoraTag[events.recoParticles.beam_number == events.recoParticles.number]
    tag = ak.flatten(ak.fill_none(ak.pad_none(tag, 1), -999))
    return tag == 13

@CountEvents
def CaloSizeCut(events : Data) -> ak.Array:
    # calo size cut
    calo_wire = events.io.Get("reco_beam_calo_wire")
    calo_wire = calo_wire[calo_wire != -999] # Analyser fills the empty entry with a -999
    return ak.num(calo_wire, 1) > 0 

@CountEvents
def BeamQualityCut(events : Data) -> ak.Array:
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

    #! these are set like this so that the comparisions are not done for dx and dy
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

    # do only MC for now.
    beam_start_pos = vector.vector(
        events.io.Get("reco_beam_startX"),
        events.io.Get("reco_beam_startY"),
        events.io.Get("reco_beam_startZ")
    )

    beam_dx = (beam_start_pos.x - beam_startX_mc) / beam_startX_rms_mc
    beam_dy = (beam_start_pos.y - beam_startY_mc) / beam_startY_rms_mc
    beam_dz = (beam_start_pos.z - beam_startZ_mc) / beam_startZ_rms_mc
    beam_dxy = (beam_dx**2 + beam_dy**2)**0.5

    beam_dir = vector.normalize(vector.sub(beam_start_pos, events.recoParticles.beamVertex))

    beam_dir_mc = vector.vector(
        np.cos(beam_angleX_mc * np.pi / 180),
        np.cos(beam_angleY_mc * np.pi / 180),
        np.cos(beam_angleZ_mc * np.pi / 180),
        )
    beam_dir_mc = vector.normalize(beam_dir_mc)

    beam_costh = vector.dot(beam_dir, beam_dir_mc)

    has_angle_cut = True;

    beam_quality_mask = events.eventNum > 0 # mask which is all trues

    def beam_quality_cut(x, xmin, xmax):
        # print(f"{x=}")
        # print(f"{xmin=}")
        # print(f"{xmax=}")
        return ((x > xmin) & (x < xmax))

    if dx_min < dx_max:
        beam_quality_mask = beam_quality_mask & beam_quality_cut(beam_dx, dx_min, dx_max) #* should be the same as the logic below
        # if (beam_dx < dx_min) return false;
        # if (beam_dx > dx_max) return false;

    if dy_min < dy_max:
        beam_quality_mask = beam_quality_mask & beam_quality_cut(beam_dy, dy_min, dy_max) #* should be the same as the logic below

    if dz_min < dz_max:
        beam_quality_mask = beam_quality_mask & beam_quality_cut(beam_dz, dz_min, dz_max) #* should be the same as the logic below

    if dxy_min < dxy_max:
        beam_quality_mask = beam_quality_mask & beam_quality_cut(beam_dxy, dxy_min, dxy_max) #* should be the same as the logic below

    if has_angle_cut and (costh_min < costh_max):
        beam_quality_mask = beam_quality_mask & beam_quality_cut(beam_costh, costh_min, costh_max) #* should be the same as the logic below
    return beam_quality_mask

@CountEvents
def APA3Cut(events : Data) -> ak.Array:
    # APA3 cut
    return events.recoParticles.beamVertex.z < 220 # cm

@CountEvents
def MichelScoreCut(events : Data) -> ak.Array:
    # michel score cut
    score = events.io.Get("reco_beam_vertex_michel_score") / events.io.Get("reco_beam_vertex_nHits")
    return score < 0.55

@CountEvents
def medianDEdXCut(events : Data) -> ak.Array:
    # proton background cut
    beam_dEdX = events.io.Get("reco_beam_calibrated_dEdX_SCE")

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

    median = ak.flatten(ak.where(count % 2, median_odd, median_even)) # pick which median is the correct one

    return median < 2.4