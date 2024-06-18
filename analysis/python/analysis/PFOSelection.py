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


def ValidRecoEnergyCut(events : Master.Data) -> ak.Array:
    return events.recoParticles.shower_energy != -999

def ValidRecoTrackPositionCut(events : Master.Data) -> ak.Array:
    return np.logical_and(
        np.logical_and(
            events.recoParticles.track_start_pos.x != -999,
            events.recoParticles.track_start_pos.y != -999
        ),
        events.recoParticles.track_start_pos.z != 999
    )

def ValidRecoShowerPositionCut(events : Master.Data) -> ak.Array:
    return np.logical_and(
        np.logical_and(
            events.recoParticles.shower_start_pos.x != -999,
            events.recoParticles.shower_start_pos.y != -999
        ),
        events.recoParticles.shower_start_pos.z != 999
    )


def ValidRecoMomentumCut(events : Master.Data) -> ak.Array:
    return np.logical_and(
        np.logical_and(
            events.recoParticles.shower_momentum.x != -999,
            events.recoParticles.shower_momentum.y != -999
        ),
        events.recoParticles.shower_momentum.z != -999
    )


def ValidCNNScoreCut(events : Master.Data, return_property : bool = False) -> ak.Array:
    return CreateMask(-999, "!=", events.recoParticles.cnn_score, return_property)


def GoodShowerSelection(events : Master.Data, return_table = False):
    selections = [
        ValidRecoShowerPositionCut,
        ValidRecoMomentumCut,
        ValidRecoEnergyCut,
    ]
    return CombineSelections(events, selections, 1, return_table = return_table)


def EMScoreCut(events : Master.Data, cut = 0.5, return_property : bool = False) -> ak.Array:
    return CreateMask(cut, ">", events.recoParticles.em_score, return_property)


def NHitsCut(events : Master.Data, cut = 80, return_property : bool = False) -> ak.Array:
    return CreateMask(cut, ">", events.recoParticles.n_hits, return_property)


def BeamParticleDistanceCut(events : Master.Data, cut = [3, 90], op = [">", "<"], return_property : bool = False) -> ak.Array:
    # distance to beam end position in cm
    dist = find_beam_separations(events)
    return CreateMask(cut, op, dist, return_property)


def BeamParticleIPCut(events : Master.Data, cut = 20, op = "<", return_property : bool = False) -> ak.Array:
    ip = find_beam_impact_parameters(events)
    return CreateMask(cut, op, ip, return_property)


def TrackScoreCut(events : Master.Data, cut = 0.5, return_property : bool = False):
    return CreateMask(cut, ">", events.recoParticles.track_score, return_property)


def BeamDaughterCut(events : Master.Data):
    beam_number = ak.where(events.recoParticles.beam_number == -999, -1, events.recoParticles.beam_number) # null beam number (no beam particle) and null mother (pf has no mother) are the same value, so change one of them to something else
    return events.recoParticles.mother == beam_number


def VetoBeamParticle(events : Master.Data):
    return events.recoParticles.beam_number != events.recoParticles.number


def PiPlusSelection(events : Master.Data, cut = [0.5, 2.8], op = [">", "<"], return_property : bool = False):
    median_dEdX = Median(events.recoParticles.track_dEdX)
    return CreateMask(cut, op, median_dEdX, return_property)


def ProtonSelection(events : Master.Data, cut = 2.8, op = ">", return_property : bool = False):
    median_dEdX = Median(events.recoParticles.track_dEdX)
    return CreateMask(cut, op, median_dEdX, return_property)


def Chi2ProtonSelection(events : Master.Data, cut = 80, op = "<", return_property : bool = False):
    chi2_ndf = events.recoParticles.track_chi2_proton / events.recoParticles.track_chi2_proton_ndof
    return CreateMask(cut, op, chi2_ndf, return_property)

def TrackLengthSelection(events : Master.Data, cut = 27.1, op = ">", return_property : bool = False):
    return CreateMask(cut, op, events.recoParticles.track_len, return_property)

def InitialPi0PhotonSelection(
        events : Master.Data,
        em_cut : float = 0.5,
        n_hits_cut : int = 80,
        distance_bounds_cm : tuple = [3, 90],
        max_impact_cm : float = 20,
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
        {"cut": em_cut},
        {"cut": n_hits_cut},
        {"cut": distance_bounds_cm},
        {"cut": max_impact_cm}
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
        {"cut" : track_cut},
        {"cut" : n_hits_cut},
        {"cut" : [min_dEdX, max_dEdX]}
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
        {"cut" : track_cut},
        {"cut" : n_hits_cut},
        {"cut" : dEdX}
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

def add_event_offset(vals):
    """
    Generates event uniqueness by bitshifting integer values to leave
    room for, and set the first n bits to reference the event number.

    If insufficient bits are present in the data type, the code will
    attempt to increase the size of the integer to hold the new bits.

    Parameters
    ----------
    vals : ak.Array
        Awkward array of some integer values.
    
    Returns
    -------
    ak.Array
        Awkward array of same shape of vals, where the index of the
        outer layer is refernce as the lowest n bits of the new values.
    """
    n_events = ak.num(vals, axis=0)
    n_extra_bits = len(bin(n_events)) - 2
    n_mother_bits = len(bin(ak.max(vals))) - 2
    offset_vals = vals
    if (n_extra_bits + n_mother_bits) > 31:
        offset_vals = ak.values_astype(offset_vals, np.int64)
    if (n_extra_bits + n_mother_bits) > 63:
        offset_vals = ak.values_astype(offset_vals, np.int128)
    if (n_extra_bits + n_mother_bits) > 127:
        raise ValueError("Overflow - numbers too large")
    return (offset_vals << n_extra_bits) + np.arange(n_events)

def get_ak_intersection(vals, tests):
    """
    Find the full intersection of values in `vals` which appear in
    `tests` as a mask of the same size as `vals`.
    
    Note the search is event exclusive. Couple to add_event_offset to
    perform an event-exclusive search.

    Parameters
    ----------
    vals : ak.Array
        Values to be tested.
    tests : ak.Array
        Values for which `vals` returns a True result in the final
        mask.

    Returns
    -------
    ak.Array
        Boolean array as a mask containing True where `vals` is in
        `tests`.
    """
    n_vals_per_event = ak.num(vals, axis=1)
    vals_offset = ak.flatten(add_event_offset(vals)).to_numpy()
    tests_offset = ak.flatten(add_event_offset(tests)).to_numpy()
    mask = np.isin(vals_offset, tests_offset, kind="sort")
    return ak.unflatten(mask, n_vals_per_event)

def get_pi0_photons(events, mc=False, beam_only=True):
    beam_pi0s = events.trueParticles.mother == 1 if beam_only else True
    true_pi0s = events.trueParticles.number[np.logical_and(
        beam_pi0s, events.trueParticles.pdg == 111)]
    particles = events.trueParticles if mc else events.trueParticlesBT
    photon_mothers = particles.mother[events.trueParticlesBT.pdg == 22]
    return get_ak_intersection(photon_mothers, true_pi0s)

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


def find_beam_impact_parameters(events : Master.Data):
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
    starts = events.recoParticles.shower_start_pos
    beam_vertex = events.recoParticles.beam_endPos
    directions = events.recoParticles.shower_direction
    return get_impact_parameter(
        directions, starts, beam_vertex)


def find_beam_separations(events: Master.Data):
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
    starts = events.recoParticles.shower_start_pos
    beam_vertex = events.recoParticles.beam_endPos # Note: this is uncorrected for space charge effects beacuase no PFO position were corrected in the reconstruction.
    return vector.dist(starts, beam_vertex)


def get_mother_pdgs(events, bt=False):
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
    if bt:
        particles = events.trueParticlesBT
    else:
        particles = events.trueParticles
    mother_ids = particles.mother
    truth_ids = particles.number
    truth_pdgs = particles.pdg
    mother_pdgs = mother_ids.to_list()
    # ts = time.time()
    for i in range(ak.num(mother_ids, axis=0)):
        true_pdg_lookup = {
            truth_ids[i][d]:
            truth_pdgs[i][d] for d in range(ak.count(truth_ids[i]))}
        true_pdg_lookup.update({0: 0, 1: 211})
        # I have no idea why the hell I did this
        #   Presumably the beam particle gets a strange pdg code??
        for j in range(ak.count(mother_ids[i])):
            try:
                mother_pdgs[i][j] = true_pdg_lookup[mother_ids[i][j]]
            except:
                mother_pdgs[i][j] = 0
    # print(f"Mother PDG codes found in {time.time()  - ts}s")
    mother_pdgs = ak.Array(mother_pdgs)
    return mother_pdgs
