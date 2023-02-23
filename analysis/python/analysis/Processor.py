"""
Created on: 01/02/2023 21:32

Author: Shyam Bhuller

Description: Abstract class to help process anaylsis scripts in batches, using multiprocessing. Designed mannily to use with ROOT files sorted by events
"""
import argparse
import os
import subprocess
import warnings

import awkward as ak

from abc import ABC, abstractmethod
from math import ceil
from rich import print

from python.analysis.Master import Data, timer

class Processor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def Create_CLI(self, description : str = "Process description", eplilog : str = None):
        """ Creates a default arguement parser with the option to expand the command line arguements.

        Args:
            description (str, optional): description of what this Processor will do. Defaults to "Process description".
            eplilog (str, optional): epilog for the command line, can be used for things like "example usage". Defaults to None.
        """
        self.parser = argparse.ArgumentParser(description = description, formatter_class = argparse.RawDescriptionHelpFormatter, epilog = None)
        self.parser.add_argument(dest = "file", type = str, help = "ROOT file to open.") # every process should take an input file
        self.parser.add_argument(dest = "nBatches", type = int, help = "Number of batches to run in") # every process must have a number of batches to do the task in
        self.parser.add_argument("-c", "--cpu", dest = "cpu", type = int, default = 1, help = f"Number of threads to use (max is {os.cpu_count()})") # this is optional but Processors will natively support multiprocessing so always leave this in as an option.
        self.parser.add_argument("-l", "--log", dest = "log", action = "store_true", help = "Create log file for each process to be made")
        self.parser.add_argument("-d", "--debug", dest = "debug", action = "store_true", help = "Debug Processor, will not actually run the process.")

        #* an example command line arguement the user could add.
        # self.parser.add_argument("-o", "--out-csv", dest = "csv", type = str, default = "out", help = "output csv filename")
        pass

    @abstractmethod
    def ParseCLI(self):
        """ Parses the CLI, allowing for the user to configure how the arguement is parsed.
        """
        self.args = self.parser.parse_args()

        #* example of what parsing should be done here
        if self.args.log:
            os.makedirs("log/", exist_ok=True)
        # self.args.csv += ".csv"
        # print(vars(self.args))
        pass

    @abstractmethod
    def Process(self, file : str, nEvents : int, start : int, pid : int) -> subprocess.Popen:
        """ Process is the subprocess which will run a particular script or method, and must return a
            subrproccess.Popen in order for Run() to work (see below).
            The supplied arguements are must haves for any Processor. For an example see DummyProcess()

        Args:
            file (str): file to process
            nEvents (int): number of events
            start (int): which event to start at

        Returns:
            subprocess.Popen: Popen object to keep track of
        """
        pass

    @staticmethod
    def DummyProcess(self, file : str, nEvents : int, start : int, pid : int) -> subprocess.Popen:
        """ Example Process for testing purposes.

        Returns:
            subprocess.Popen: subprocees to interface with
        """
        output = subprocess.DEVNULL if not self.args.log else open("log/" + f"dummy-process-{pid}.log", "w")
        return subprocess.Popen(["sleep", "1"], stdout = output, stderr = subprocess.STDOUT)
    

    def CalculateBatches(self):
        """ Calculates the number of batches and how many events each Batch will process.
            This is pretty universal for any process so no need to make this abstract.
        """
        self.nEvents = ak.count(Data(self.args.file).eventNum) # get number of events

        if self.nEvents < self.args.nBatches:
            # make only as many batches as you need man!
            warnings.warn(f"number of batches specified ({self.args.nBatches}) exceeds the number of events ({self.nEvents}), setting number of batches to number of events.")
            self.batches = [1] * self.nEvents
            self.batch_size = 1
        else:
            # set batch size to be equal, and the remainder is put into a new batch
            self.batch_size = self.nEvents // self.args.nBatches
            self.batches = [self.batch_size] * self.args.nBatches
            remainder = self.nEvents % self.args.nBatches
            
            while remainder > 0:
                self.batches[remainder % self.args.nBatches] += 1
                remainder -= 1

        print(f"{self.batches=}")

    @timer
    def Run(self):
        """ Runs the Process.
        """
        self.CalculateBatches() # get batches
    
        chunks = ceil(len(self.batches) / self.args.cpu) # chunks is the number of batches to run per CPU
        print(f"{chunks=}")

        if self.args.debug == False:
            print(f"{self.args.debug=}")
            counter = 0
            start = 0
            for i in range(chunks):            
                # start each process per cpu simultaneously
                processes = [] 
                for c in range(self.args.cpu):
                    if start >= len(self.batches): break # if we don't need to use all our CPUs, stop
                    print(f"started process {start}")
                    processes.append(self.Process(self.args.file, self.batches[start], self.batch_size * start, start))
                    start += 1
                # check the status of each job iteratively
                for p in processes:
                    p.communicate()
                    print(f"finished process: {p}")
                    counter += 1
                print(f"chunk {i + 1} done")
            print(f"done {counter} batches")
