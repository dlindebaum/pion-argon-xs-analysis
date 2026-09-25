"""
Created on: 23/09/2026 16:43

Author: Shyam Bhuller

Description: Helper functions for multiprocesssing Data/MC Ntuples.
"""
import argparse
import os

from enum import Enum

import awkward as ak

from python.analysis import Processing, Tags
from python.analysis.Master import IO, FileDescriptor, SaveObject, LoadObject

class Sample(str, Enum):
    MC = "mc"
    DATA = "data"


def file_len(file : str) -> int:
    """ Get number of events of an PDSP Analyser Ntuple file.

    Args:
        file (str): path of Ntuple.

    Returns:
        int: Number of events
    """
    return len(IO(file).Get(["EventID", "event"]))


def CalculateBatches(args : argparse.Namespace) -> dict[str, any]:
    """ Calculate the number of events to process in eacn batch.

    Args:
        args (argparse.Namespace): Application Arguments.

    Returns:
        dict[str, any]: Multiprocessing arguments.
    """
    if "data" in args.ntuple_files:
        n_data = [file_len(file_desc.file) for file_desc in args.ntuple_files["data"]]
    else:
        n_data = []

    if len(n_data) == 0:
        print("no data file was specified, 'normalisation', 'beam_reweight', 'toy_parameters' and 'analyse' will not run")

    n_mc = [file_len(file_desc.file) for file_desc in args.ntuple_files["mc"]] # must have MC

    processing_args = {"events" : None, "batches" : None, "threads" : args.cpus}

    # pass multiprocessing args
    # if max([*n_data, *n_mc]) >= 7E5:
    #     processing_args["events"] = None
    #     processing_args["batches"] = int(2 * max([*n_data, *n_mc]) // 7E5)
    #     processing_args["threads"] = args.cpus

    return processing_args


def MergeOutputs(outputs : list[dict]) -> dict:
    """ Merge a collection of dictionaries with identical structures into a single dictionary.

    Args:
        outputs (list[dict]): List of dictionaries to merge.

    Returns:
        dict: Merged dictionary.
    """
    def search(collection : dict, output : dict):
        for k, v in collection.items():
            if type(v) is dict:
                if k not in output:
                    output[k] = {}
                search(v, output[k])
            else:
                if k not in output:
                    output[k] = v
                else:
                    if type(v) == ak.Array:
                        output[k] = ak.concatenate([output[k], v])
                    elif type(v) == Tags.Tags:
                        output[k] = Tags.MergeTags([output[k], v])
                    elif type(v) == list:
                        output[k].extend(v)
                    else:
                        if type(output[k]) != list:
                            output[k] = [output[k], v]
                        else:
                            output[k].append(v)

    merged_output = {}
    for o in outputs:
        search(o, merged_output)
    return merged_output


def RunProcess(ntuple_files : list[FileDescriptor], is_data : bool, args : argparse.Namespace, func : callable, merge : bool = True) -> list[dict] | dict:
    """ Run a process on Ntuple files.

    Args:
        ntuple_files (list[FileDescriptor]): Ntuple files.
        is_data (bool): Whether the Ntuple files are Data or MC.
        args (argparse.Namespace): Application arguments.
        func (callable): Process to run.
        merge (bool, optional): Whether to merge the outputs of the processing. Defaults to True.

    Returns:
        list[dict] | dict: Output of the processing.
    """
    func_args = vars(args)
    func_args["data"] = is_data
    output = Processing.mutliprocess(func, ntuple_files, args.batches, args.events, func_args, args.threads)
    if merge:
        output = MergeOutputs(output)
    return output


def ApplicationProcessing(samples : list[Sample], outdir : str, args : argparse.Namespace, func : callable, merge : bool, outname : str = "output") -> list[dict] | dict:
    """ Processing specifically for Applications where there is an option to reload processed data rather than fully reprocessing Ntuples.

    Args:
        samples (list[Sample]): What sample types to process, can be Data, MC or both.
        outdir (str): Output directory for stored processing outputs.
        args (argparse.Namespace): Application arguments.
        func (callable): Process to run.
        merge (bool): whether to merge the processing outputs.
        outname (str, optional): Output file name. Defaults to "output".

    Returns:
        list[dict] | dict: Processing outputs, either loaded from file or processed at runtime.
    """
    if (args.regen is True) or (os.path.isfile(f"{outdir}{outname}.dill") is False):
        print("Processing Ntuples")
        outputs = {s : RunProcess(args.ntuple_files[s], s == Sample.DATA, args, func, merge) for s in samples}
        SaveObject(f"{outdir}{outname}.dill", outputs)
    else:
        print("Loading existing outputs")
        outputs = LoadObject(f"{outdir}{outname}.dill")
    return outputs
