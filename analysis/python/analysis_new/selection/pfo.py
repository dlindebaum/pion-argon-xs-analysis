"""
Created on: 07/03/2023 14:43

Author: Shyam Bhuller

Description: Lists cuts which selects PFOs in an event or events.
TODO Documentation.
"""

def ValidRecoEnergyCut(events : Master.Data) -> ak.Array:
    return events.recoParticles.shower_energy != -999


def ValidRecoPositionCut(events : Master.Data) -> ak.Array:
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
        ValidRecoPositionCut,
        ValidRecoMomentumCut,
        ValidRecoEnergyCut,
    ]
    return CombineSelections(events, selections, 1, return_table = return_table)


def EMScoreCut(events : Master.Data, cut = 0.5, return_property : bool = False) -> ak.Array: #TODO Remove
    return CreateMask(cut, "<", events.recoParticles.track_score, return_property)


def NHitsCut(events : Master.Data, cut = 80, op = ">", return_property : bool = False) -> ak.Array:
    return CreateMask(cut, op, events.recoParticles.n_hits, return_property)


def BeamParticleDistanceCut(events : Master.Data, cut = [3, 90], op = [">", "<"], return_property : bool = False) -> ak.Array:
    # distance to beam end position in cm
    dist = find_beam_separations(events)
    return CreateMask(cut, op, dist, return_property)


def BeamParticleIPCut(events : Master.Data, cut = 20, op = "<", return_property : bool = False) -> ak.Array:
    ip = find_beam_impact_parameters(events)
    return CreateMask(cut, op, ip, return_property)


def TrackScoreCut(events : Master.Data, cut = 0.5, op = ">", return_property : bool = False):
    return CreateMask(cut, op, events.recoParticles.track_score, return_property)


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

