"""
Created on: 02/04/2023 19:40

Author: Shyam Bhuller

Description: Tools for parallel processing a task which takes an input file and can be divided into batches i.e. analysing ntuples.
"""
from pathos.multiprocessing import ProcessPool

import itertools
import numpy as np
import awkward as ak
from rich import print

import warnings

from python.analysis import Master
from contextlib import redirect_stdout, redirect_stderr

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
    pool = ProcessPool(nodes = nodes) # use all my threads, damn it!
    pool.restart(True)

    inputs = GenerateFunctionArguments(files, nBatches, nEvents, args = args, event_indices = event_indices)
    batch_numbers = list(range(len(inputs[0])))
    return pool.map(func, batch_numbers, *inputs)