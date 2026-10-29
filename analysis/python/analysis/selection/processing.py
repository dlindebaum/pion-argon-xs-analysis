"""Selection processing pipelines."""
import os

import awkward as ak
import numpy as np
import pandas as pd

from rich import print as rprint

from . import beam, event

from python.analysis import cuts, io, ntuple, photon_energy_correction, tags
from python.analysis.beam import scraper


def MergeSelectionMasks(output):
    files = output["name"]

    masks ={}
    for k, v in output.items():
        if (k not in ["name", "regions"]) and (len(v) > 0):
            masks[k] = v["masks"]

    unique_files = np.unique(files)

    merged_masks = {}

    for i, file in enumerate(unique_files):
        merged_masks[file] = {} 
        for k, v in masks.items():
            merged_v = []
            for j, f in zip(v, files):
                if f == file:
                    merged_v.append(j)
            merged_masks[file][k] = merged_v

    for f in merged_masks:
        selection_masks = {}
        for t in merged_masks[f]:
            m = {}
            for i in merged_masks[f][t]:
                for k, v in i.items():
                    if k not in m:
                        m[k] = v
                    else:
                        m[k] = ak.concatenate([m[k], v])
            selection_masks[t] = m
        merged_masks[f] = selection_masks

    sorted_masks = {}
    for f in merged_masks:
        for k, v in merged_masks[f].items():
            if k not in sorted_masks:
                sorted_masks[k] = [v]
            else:
                sorted_masks[k].append(v)

    for k in output:
        if (k not in ["name", "regions"]) and (k in sorted_masks):
            output[k]["masks"] = sorted_masks[k]
    output["name"] = list(merged_masks.keys())
    return output


def merge_multiproccess_output(outputs : list) -> dict:
    """ Merge multiprocessing output into a single dictionary.

    Args:
        outputs (list): outputs from multiprocessing job.

    Returns:
        dict: merged output
    """
    merged_output = {}

    for output in outputs:
        for selection in output:
            if selection == "name":
                if selection not in merged_output:
                    merged_output[selection] = [output[selection]]
                else:
                    merged_output[selection].append(output[selection])
                continue
            if selection not in merged_output:
                merged_output[selection] = {}
            if output[selection] is None:
                continue
            for o in output[selection]:
                if o not in merged_output[selection]:
                    if o == "masks":
                        merged_output[selection][o] = [output[selection][o]] # need to treat mask merging more carefully
                    else:
                        merged_output[selection][o] = output[selection][o]
                    continue                    
                if output[selection][o] is None: continue
                if selection == "regions":
                    for m in output[selection][o]:
                        merged_output[selection][o][m] = ak.concatenate([merged_output[selection][o][m], output[selection][o][m]])
                else:
                    if o == "table":
                        tmp = merged_output[selection][o] + output[selection][o]
                        tmp.Name = merged_output[selection][o].Name
                        merged_output[selection][o] = tmp
                    elif o == "masks":
                        merged_output[selection][o].append(output[selection][o]) # need to treat mask merging more carefully
                    else:
                        for c in output[selection][o]:
                            if output[selection][o][c]["value"] is not None:
                                if c in ["PiBeamSelection"]:
                                    for i in merged_output[selection][o][c]["value"]:
                                        merged_output[selection][o][c]["value"][i] = merged_output[selection][o][c]["value"][i] + output[selection][o][c]["value"][i] 
                                else:
                                    merged_output[selection][o][c]["value"] = ak.concatenate([merged_output[selection][o][c]["value"], output[selection][o][c]["value"]]) 

                            if output[selection][o][c]["tags"] is not None:
                                for t in merged_output[selection][o][c]["tags"]:
                                    merged_output[selection][o][c]["tags"][t].mask = ak.concatenate([merged_output[selection][o][c]["tags"][t].mask, output[selection][o][c]["tags"][t].mask])

                            if output[selection][o][c]["fs_tags"] is not None:
                                for t in merged_output[selection][o][c]["fs_tags"]:
                                    merged_output[selection][o][c]["fs_tags"][t].mask = ak.concatenate([merged_output[selection][o][c]["fs_tags"][t].mask, output[selection][o][c]["fs_tags"][t].mask])

    rprint("merged_output")
    rprint(merged_output)
    return merged_output

def SaveMasks(output : dict, out : str):
    """ Save selection masks to dill file

    Args:
        output (dict): masks
        out (str): output file directory
    """
    os.makedirs(out, exist_ok = True)

    files = output["name"]

    masks = {}
    for k, v in output.items():
        if (k not in ["name", "regions"]) and (len(v) > 0):
            masks[k] = {}
            for f, i in zip(files, v["masks"]):
                masks[k][f] = i
    
    for k in masks:
        io.write_dill(out + f"{k}_selection_masks.dill", masks[k])


def MakeOutput(value : ak.Array, event_tags : tags.Tags, cuts : list = [], op : list = [], fs_tags : tags.Tags = None) -> dict:
    """ Output dictionary for multiprocessing class, if we want to be fancy this can be a dataclass instead.
        Generally the data stored in this dictionary should be before any cuts are applied, for plotting purposes.

    Args:
        value (ak.Array): value which is being studied
        tags (tags.Tags): tags which categorise the value
        cuts (list, optional): cut values used on this data. Defaults to [].
        cuts (list, optional): operation of cut. Defaults to [].
        fs_tags (tags.Tags, optional): final state tags to the data. Defaults to None.

    Returns:
        dict: dictionary of data.
    """
    return {"value" : value, "tags" : event_tags, "cuts" : cuts, "op" : op, "fs_tags" : fs_tags}


def AnalyseBeamFiducialCut(events : ntuple.Data, beam_instrumentation : bool, functions : dict, args : dict):
    output = {}
    masks = {}
    cut_table = cuts.CutHandler(events, particle_tags = tags.GenerateTrueBeamParticleTags(events) if beam_instrumentation is False else None)
    output["no_selection"] = MakeOutput(None, tags.GenerateTrueBeamParticleTags(events), None, None, event.GenerateTrueFinalStateTags(events))
    if "TrueFiducialCut" in args:
        masks = {k : functions[k](events, **v) for k, v in args.items() if k in ["APA3Cut", "TrueFiducialCut"]}
        for a in ["APA3Cut", "TrueFiducialCut"]:
            mask, property = functions[a](events, **args[a], return_property = True)
            cut_table.add_mask(mask, a)
            cut_values = args[a]["cut"] if "cut" in args[a] else None
            operations = args[a]["op"] if "op" in args[a] else None
            output[a] = MakeOutput(property, tags.GenerateTrueBeamParticleTags(events), cut_values, operations)

            events.Filter([mask], [mask])
            output[a]["fs_tags"] = event.GenerateTrueFinalStateTags(events)

    #* true particle population
    output["final_tags"] = MakeOutput(None, tags.GenerateTrueBeamParticleTags(events), None, None, event.GenerateTrueFinalStateTags(events))

    df = cut_table.get_table(init_data_name = "no selection", pfos = False, percent_remain = False, relative_percent = False, ave_per_event = False)

    return output, df, masks


def AnalyseBeamSelection(events : ntuple.Data, beam_instrumentation : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame, dict]:
    """ Manually applies the beam selection while storing the value being cut on, cut values and truth tags in order to do plotting
        and produce performance tables.

    Args:
        events (ntuple.Data): events to look at
        beam_instrumentation (bool): use beam instrumentation for beam particle selection
        beam_quality_fits (dict): fit values for the beam quality selection

    Returns:
        dict: output data.
    """
    output = {}
    cut_table = cuts.CutHandler(events, particle_tags = tags.GenerateTrueBeamParticleTags(events) if beam_instrumentation is False else None)

    output["no_selection"] = MakeOutput(None, tags.GenerateTrueBeamParticleTags(events), None, None, event.GenerateTrueFinalStateTags(events))
    output["no_selection_endz"] = MakeOutput(events.recoParticles.beam_endPos_SCE.z, tags.GenerateTrueBeamParticleTags(events), None, None)

    masks = {k : functions[k](events, **v) for k, v in args.items()}

    #* pi+ beam selection
    mask = beam.PiBeamSelection(events, **args["PiBeamSelection"])
    cut_table.add_mask(mask, "PiBeamSelection")
    counts = tags.GenerateTrueBeamParticleTags(events)
    for i in counts:
        counts[i] = ak.sum(counts[i].mask)
    output["PiBeamSelection"] = MakeOutput(counts, tags.GenerateTrueBeamParticleTags(events), None, None)
    events.Filter([mask], [mask])
    output["PiBeamSelection"]["fs_tags"] = event.GenerateTrueFinalStateTags(events)

    output["PiBeamSelection_endz"] = MakeOutput(events.recoParticles.beam_endPos_SCE.z, tags.GenerateTrueBeamParticleTags(events), None, None)

    for a in args:
        if a in ["PiBeamSelection", "TrueFiducialCut"]: continue
        if ("TrueFiducialCut" in args) and (a == "APA3Cut"): continue # should have applied fiducial cuts first
        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = MakeOutput(property, tags.GenerateTrueBeamParticleTags(events), cut_values, operations)

        if (a == "BeamScraperCut") and (beam_instrumentation == False):
            if beam_instrumentation == False:
                scraper_id = scraper.IsScraper(events, args[a]["fits"])
                output[a + "_scraper_tag"] = MakeOutput(None, tags.BeamScraperTag(scraper_id), None, None)

        events.Filter([mask], [mask])
        output[a]["fs_tags"] = event.GenerateTrueFinalStateTags(events)

    output["selection_endz"] = MakeOutput(events.recoParticles.beam_endPos_SCE.z, tags.GenerateTrueBeamParticleTags(events), None, None)

    #* true particle population
    output["final_tags"] = MakeOutput(None, tags.GenerateTrueBeamParticleTags(events), None, None, event.GenerateTrueFinalStateTags(events))

    df = cut_table.get_table(init_data_name = "no selection", pfos = False, percent_remain = False, relative_percent = False, ave_per_event = False)
    return output, df, masks


def AnalysePFOSelection(events : ntuple.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse PFO selection.

    Args:
        events (ntuple.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = cuts.CutHandler(events, particle_tags = tags.GenerateTrueParticleTagsInterestingPFOs(events))

    for a in args:
        if a in ["NHitsCut", "BeamParticleIPCut"]:
            output[a + "_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = MakeOutput(property, tags.GenerateTrueParticleTagsInterestingPFOs(events), cut_values, operations)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = event.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = MakeOutput(None, tags.GenerateTrueParticleTagsInterestingPFOs(events), None, None)
    
    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePiPlusSelection(events : ntuple.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse the daughter pi+ selection.

    Args:
        events (ntuple.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = cuts.CutHandler(events, particle_tags = tags.GenerateTrueParticleTagsPiPlus(events))

    for a in args:
        if a == "NHitsCut":
            output[a + "_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = MakeOutput(property, tags.GenerateTrueParticleTagsPiPlus(events), cut_values, operations)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = event.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = MakeOutput(None, tags.GenerateTrueParticleTagsPiPlus(events), None, None)
    
    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePhotonCandidateSelection(events : ntuple.Data, data : bool, functions : dict, args : dict) -> tuple[dict, pd.DataFrame]:
    """ Analyse the photon candidate selection.

    Args:
        events (ntuple.Data): events to look at

    Returns:
        dict: output data
    """
    output = {}
    cut_table = cuts.CutHandler(events, particle_tags = tags.GenerateTrueParticleTagsPi0Shower(events) if data is False else None)

    for a in args:
        if a in ["NHitsCut", "BeamParticleIPCut"]:
            output[a + "_completeness"] = MakeOutput(events.trueParticlesBT.completeness, [])

        mask, property = functions[a](events, **args[a], return_property = True)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        output[a] = MakeOutput(property, tags.GenerateTrueParticleTagsPi0Shower(events), cut_values, operations)
        events.Filter([mask], [mask])
        output[a]["fs_tags"] = event.GenerateTrueFinalStateTags(events)    

    #* true particle population
    output["final_tags"] = MakeOutput(None, tags.GenerateTrueParticleTagsPi0Shower(events), None, None)

    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, events = False)
    print(df)
    return output, df


def AnalysePi0Selection(events : ntuple.Data, data : bool, functions : dict, args : dict, photon_mask : ak.Array) -> tuple[dict, pd.DataFrame]:
    """ Analyse the pi0 selection.

    Args:
        events (ntuple.Data): events to look at
        data (bool): is the ntuple file data or MC
        correction (callable, optional): shower energy correction. Defaults to None.
        correction_params (dict, optional): shower energy correction parameters. Defaults to None.

    Returns:
        dict: output data
    """    
    def null_tag():
        tag = tags.Tags()
        tag["null"] = tags.Tag(mask = (events.eventNum < -1))
        return tag

    output = {}
    photonCandidates = ak.Array(photon_mask)
    cut_table = cuts.CutHandler(events, particle_tags = tags.GeneratePi0Tags(events, photon_mask) if data is False else None)

    for a in args:
        if a == "Pi0MassSelection":
            mask, property = functions[a](events, **args[a], return_property = True, photon_mask = photonCandidates)
            output["mass_event_tag"] = MakeOutput(None, event.GenerateTrueFinalStateTags(events), None, None)
        else:
            mask, property = functions[a](events, **args[a], return_property = True, photon_mask = photonCandidates)
        if a != "NPhotonCandidateSelection":
            mask = ak.flatten(mask)
        cut_table.add_mask(mask, a)
        cut_values = args[a]["cut"] if "cut" in args[a] else None
        operations = args[a]["op"] if "op" in args[a] else None
        pi0_tags = null_tag() if data else tags.GeneratePi0Tags(events, photonCandidates)
        output[a] = MakeOutput(property, pi0_tags, cut_values, operations)
        events.Filter([mask], [mask])
        photonCandidates = photonCandidates[mask]
        output[a]["fs_tags"] = event.GenerateTrueFinalStateTags(events)

    #* final counts
    output["event_tag"] = MakeOutput(None, event.GenerateTrueFinalStateTags(events), None, None)
    pi0_tags = null_tag() if data else tags.GeneratePi0Tags(events, photonCandidates)
    output["final_tags"] = MakeOutput(None, pi0_tags, None, None)

    df = cut_table.get_table(init_data_name = "Beam particle selection", percent_remain = False, relative_percent = False, ave_per_event = False, pfos = False)
    print(df)
    return output, df


def AnalyseRegions(events : ntuple.Data, photon_mask : ak.Array, is_data : bool, correction : callable = None, correction_params : dict = None) -> tuple[dict, dict]:
    """ Create masks which desribe the truth and reco regions for various exlusive cross sections.

    Args:
        events (ntuple.Data): events to look at
        photon_mask (ak.Array): mask for pi0 photon shower candidates
        is_data (bool): is this a data ntuple?
        correction (callable, optional): shower energy correction. Defaults to None.
        correction_params (str, optional): shower energy correction parameters file. Defaults to None.

    Returns:
        tuple[dict, dict]: truth and reco regions
    """
    truth_regions = event.create_regions(events.trueParticles.nPi0, events.trueParticles.nPiPlus + events.trueParticles.nPiMinus) if is_data == False else None

    if correction_params is None:
        params = None
    else:
        params = io.read_json(correction_params)

    reco_pi0_counts = event.count_pi0_candidates(events, exactly_two_photons = True, photon_mask = photon_mask, correction = photon_energy_correction.shower_energy_correction[correction], correction_params = params)
    reco_pi_plus_counts_mom_cut = event.count_charged_pi_candidates(events, energy_cut = None)
    reco_regions = event.create_regions(reco_pi0_counts, reco_pi_plus_counts_mom_cut)
    return truth_regions, reco_regions


def CreatePFOMasks(sample : ntuple.Data, selections : dict, args_type : str, extra_args : dict = None) -> dict[np.array]:
    """ Create PFO masks to save to file.

    Args:
        mc (ntuple.Data): sample.
        selections (dict): PFO selections.
        args_type (str): use Data or MC arguments.
        extra_args (dict, optional): any additional arguments to add. Defaults to None.

    Returns:
        masks: dictionary of masks.
    """
    masks = {}
    for n, c, v in zip(selections["selections"].keys(), selections["selections"].values(), selections[args_type].values()):
        if extra_args is not None:
            v = {**v, **extra_args}
        masks[n] = c(sample, **v)
    return masks
