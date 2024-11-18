"""
Created on: 02/04/2023 19:40

Author: Shyam Bhuller

Description: Tools for parallel processing a task which takes an input file and can be divided into batches i.e. analysing ntuples.
"""
import os
import argparse
from pathos.multiprocessing import ProcessPool
from enum import Enum

import itertools
import numpy as np
import awkward as ak
from rich import print

import warnings

from python.analysis import Master, Tags
from contextlib import redirect_stdout, redirect_stderr

class Sample(str, Enum):
    MC = "mc"
    DATA = "data"

def MergeOutputs(outputs : list[dict]) -> dict:
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

def ApplicationProcessing(
        samples : list[Sample],
        outdir : str,
        args : argparse.Namespace,
        func : callable,
        merge : bool,
        outname : str = "output",
        batchless: bool = False):

    if (args.regen is True) or (os.path.isfile(f"{outdir}{outname}.dill") is False):
        print("Processing Ntuples")
        outputs = {
            s : RunProcess(args.ntuple_files[s], s == Sample.DATA,
                           args, func, merge, batchless=batchless)
            for s in samples}
        Master.SaveObject(f"{outdir}{outname}.dill", outputs)
    else:
        print("Loading existing outputs")
        outputs = Master.LoadObject(f"{outdir}{outname}.dill")
    return outputs

def log_process(func):
    """ Function wrapper to redirect printout from a process to a log file.

    Args:
        func (_type_): function.
    """
    def wrap(*args, **kwargs):
        process_num = args[0]
        print(f"starting process : {process_num}")
        with open(f"out_{args[0]}.log", "w") as f:
            with redirect_stdout(f):
                with redirect_stderr(f):
                    return func(*args, **kwargs)
    return wrap


def CalculateBatches(file : str, n_batches : int = None, n_events : int = None) -> list[int]:
    """ Calculate the number of events to run per job (batch).

    Args:
        file (str): input Ntuple file
        n_batches (int, optional): Number of batches, if None number of batches created is decided based on the number of events. Defaults to None.
        n_events (int, optional): number of events to run per batch, if None, the number of events per batch is automatically calculated. Defaults to None.

    Returns:
        list[int]: list of batches
    """
    total_events = ak.count(Master.Data(file, nTuple_type = Master.Ntuple_Type.SHOWER_MERGING).eventNum) # get number of events, use shower merging ntuple type to supress false warnings
    print(f"{total_events=}")

    if n_batches is None and n_events is None:
        batches = [total_events]
    else:
        if n_batches is not None and total_events < n_batches:
            # make only as many batches as you need man!
            warnings.warn(f"number of batches specified ({n_batches}) exceeds the number of events ({total_events}), setting number of batches to number of events.")
            batches = [1] * total_events
            batch_size = 1
        elif n_events is not None and total_events < n_events:
            warnings.warn(f"number of events per batch ({n_events}) exceeds the number of events ({total_events}), setting number of events per batch to number of events, and batches to 1.")
            batches = [total_events]
        else:
            if n_events is not None and n_batches is not None:
                if (n_events * n_batches) > total_events:
                    warnings.warn(f"number of events specified (n_events * n_batches) exceeds the number of events ({total_events}), adjusting the number of batches to fit the total events")
                    n_batches = None #? will this mutate the input parameter?
            # set batch size to be equal, and the remainder is put into a new batch
            batch_size = (total_events // n_batches if n_events is None else n_events)
            
            n = total_events // n_events if n_batches is None else n_batches

            batches = [batch_size] * n

            if n_events is None:
                remainder = total_events % n
            elif n_batches is None:
                remainder = total_events - (batch_size * n)
            else:
                # assume we don't want to process the total number of events
                remainder = 0

            while remainder > 0: # evenly distribute events across batches
                batches[remainder % n] += 1
                remainder -= 1

    print(f"{batches=}")
    return batches


def file_len(file : str):
    return len(Master.IO(file).Get(["EventID", "event"]))

def CalculateEventBatches(args):
    if "data" in args.ntuple_files:
        n_data = [file_len(file["file"]) for file in args.ntuple_files["data"]]
    else:
        n_data = []

    # Printed in main of run_analysis if relevant
    # if len(n_data) == 0:
    #     print("no data file was specified, 'normalisation', 'beam_reweight', 'toy_parameters' and 'analyse' will not run")

    n_mc = [file_len(file["file"]) for file in args.ntuple_files["mc"]] # must have MC

    processing_args = {"events" : None, "batches" : None, "threads" : None}

    # pass multiprocessing args
    if max([*n_data, *n_mc]) >= 7E5:
        processing_args["events"] = None
        processing_args["batches"] = int(2 * max([*n_data, *n_mc]) // 7E5)
        processing_args["threads"] = args.cpus
    else:
        processing_args["events"] = None
        processing_args["batches"] = None
        processing_args["threads"] = args.cpus
    return processing_args

def BatchlessFormatArgs(file, args):
    """
    Dummy arguments which would be accepted by a function passed to
    Proccessing.multiprocess, but with no batching (i.e. start at 0,
    use all events)

    Expected use:
    ```
    output = func(*BatchlessFormatArgs(file, func_args))
    ```
    Replacing:
    ```
    output = mutliprocess(func, [file], args.batches, args.events, func_args, args.threads)
    ```

    Parameters
    ----------
    file : str
        File path of the ntuple to be run on.
    args : dict
        Dictionary of arguments accessible by the function
    
    Returns
    -------
    i, file, n_events, start, selected_events, args
    i : int
        Dummy. 0.
    file : str
        Input file
    n_events : int
        Dummy. Numer of events in file.
    start : int
        Dummy. 0.
    selected_events : NoneType
        Dummy. None.
    args : dict
        Input args.
    """
    # get number of events, use shower merging ntuple type to supress false warnings
    # Same method as CalculateBatches
    total_events = ak.count(
        Master.Data(file, nTuple_type = Master.Ntuple_Type.SHOWER_MERGING).eventNum)
    print(f"{total_events=}")
    print(f"Running batchless, setting batches to {total_events}")
    return 0, file, total_events, 0, None, args

def GenerateFunctionArguments(files : list, nBatches : int, nEvents : int, args : dict, event_indices : list = None) -> list:
    """ Create a list of function arguments to supply to each job. It will automatically calculate the number of events and stride for each job.

    Args:
        files (list): Input file lists
        nBatches (int): Number of batches run (i.e. number of jobs)
        nEvents (int): number of events to run per job
        args (dict): additional function arguments

    Returns:
        list: _description_
    """
    batches = []
    start = []
    for f in files: # calcualte event numbers and stride for each file
        b = CalculateBatches(f, nBatches, nEvents)
        start.append([0] + list(np.cumsum(b[:-1])))
        batches.append(b)

    # [file, n_events, start, selected_events]
    inputs = [[], [], [], []]

    for i in range(len(files)): # create the function argument lists
        inp = list(itertools.product([files[i]], batches[i])) # first join the file and batch numbers

        for j, s in zip(inp, start[i]): # zip the start point since we just want to concantenate this list
            print(*j, s)
            inputs[0].append(j[0])
            inputs[1].append(j[1])
            inputs[2].append(s)
            if event_indices is not None:
                inputs[3].append(event_indices[i])
            else:
                inputs[3].append(None)

    inputs += [[args]*len(inputs[0])] # each job will have the same additional args, so just copy this.
    return inputs


@Master.timer
def mutliprocess(func, files : list, nBatches : int, nEvents : int, args : dict, nodes : int = None, event_indices : list = None):
    """ Run a function with parallel processing, requires that it analyses ntuples.

    Args:
        func (_type_): function to run
        files (list): input root files
        nBatches (list): number of batches
        nEvents (list): number of events
        args (dict): args for function
        nodes (int) : number of processors to use, if None all are used.

    Returns:
        Any: function output
    """
    pool = ProcessPool(nodes = nodes)
    pool.restart(True)

    inputs = GenerateFunctionArguments(files, nBatches, nEvents, args = args, event_indices = event_indices)
    batch_numbers = list(range(len(inputs[0])))
    return pool.map(func, batch_numbers, *inputs)

def RunProcess(
        ntuple_files : list[dict],
        is_data : bool,
        args : argparse.Namespace,
        func : callable,
        merge : bool = True,
        batchless : bool = False) -> list:
    output = []
    for i in ntuple_files:
        func_args = vars(args)
        func_args["data"] = is_data
        func_args["nTuple_type"] = i["type"]
        func_args["pmom"] = i["pmom"]
        if "graph" in i.keys():
            func_args["graph"] = i["graph"]
        func_args["graph_norm"] = (
            i["graph_norm"] if "graph_norm" in i.keys() else None)
        func_args["train_sample"] = (
            i["train_sample"] if "train_sample" in i.keys() else False)
        if not batchless:
            output.extend(mutliprocess(func, [i["file"]], args.batches, args.events, func_args, args.threads))
        else:
            output.append(Master.timer(func)(
                *BatchlessFormatArgs(i["file"], func_args)))
    if merge:
        output = MergeOutputs(output)
    return output