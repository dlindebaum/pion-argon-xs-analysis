import argparse
import os
import subprocess
from enum import Enum

import awkward as ak
from rich import print

from python.analysis.Master import Data
from python.analysis.Processor import Processor


class MERGE_METHOD(Enum):
    UNMERGED = "unmerged"
    BACKTRACKED_PFO = "backtracked_PFO"
    RECO = "reco"

    def __str__(self):
        return self.value

class CalculateMergeQuantities(Processor):
    def __init__(self, directory, file, nBatches):
        #* inject extra paramaters for the Processor here
        self.args = argparse.Namespace()
        self.args.file = file
        self.args.directory = directory
        self.args.nBatches = nBatches

    def Create_CLI(self, description: str = "Process description", eplilog: str = None):
        return super().Create_CLI(description, eplilog)

    def ParseCLI(self):
        return super().ParseCLI()

    def Process(self, file: str, nEvents: int, start: int, pid : int) -> subprocess.Popen:
        log_file = open(self.args.directory + "logs/" + f"merge-quantities-{pid}-{nEvents}-{start}.log", "w")
        return subprocess.Popen([
            "prod4a_merge_study.py",
            file,
            "-s",
            "-d", self.args.directory + "quantities/",
            "-o", f"merge-quantities-{pid}-{nEvents}-{start}",
            "-e", f"{nEvents}", f"{start}"
            ], stdout = log_file, stderr = subprocess.STDOUT)


class CalculateShowerPairs(Processor):
    def __init__(self, directory, file, nBatches, cut_type : str = None, cuts : str = None):
        self.args = argparse.Namespace()
        self.args.file = file
        self.args.directory = directory
        self.args.nBatches = nBatches
        self.args.cut_type = cut_type
        self.args.cuts = cuts
        self.args.debug = False
        super().__init__()

    def Create_CLI(self, description: str = "Process description", eplilog: str = None):
        return super().Create_CLI(description, eplilog)

    def ParseCLI(self):
        return super().ParseCLI()

    def ProcessBacktrackedPFOs(self, file: str, nEvents: int, start: int, pid : int) -> subprocess.Popen:
        log_file = open(self.args.directory + "logs/" + f"shower-pairs-btPFO-{pid}-{nEvents}-{start}.log", "w")
        return subprocess.Popen([
            "prod4a_merge_study.py",
            file,
            "-m", "cheat",
            "-s",
            "-d", self.args.directory + "shower-pairs/",
            "-o", f"shower-pairs-cheat-{pid}-{nEvents}-{start}",
            "-e", f"{nEvents}", f"{start}"
        ], stdout = log_file, stderr = subprocess.STDOUT)

    def ProcessUnmerged(self, file: str, nEvents: int, start: int, pid : int) -> subprocess.Popen:
        log_file = open(self.args.directory + "logs/" + f"shower-pairs-unmerged-{pid}-{nEvents}-{start}.log", "w")
        return subprocess.Popen([
            "prod4a_merge_study.py",
            file,
            "-m", "unmerged",
            "-s",
            "-d", self.args.directory + "shower-pairs/",
            "-o", f"shower-pairs-unmerged-{pid}-{nEvents}-{start}",
            "-e", f"{nEvents}", f"{start}"
        ], stdout = log_file, stderr = subprocess.STDOUT)

    def ProcessReco(self, file: str, nEvents: int, start: int, pid : int) -> subprocess.Popen:
        log_file = open(self.args.directory + "logs/" + f"shower-pairs-{self.args.cut_type}-{pid}-{nEvents}-{start}.log", "w")
        return subprocess.Popen([
            "prod4a_merge_study.py",
            file,
            "--cuts", self.args.cuts,
            "--cut-type", self.args.cut_type,
            "-a",
            "-m", "reco",
            "-s",
            "-d", self.args.directory + "shower-pairs/",
            "-o", f"shower-pairs-{self.args.cut_type}-{pid}-{nEvents}-{start}",
            "-e", f"{nEvents}", f"{start}"
        ], stdout = log_file, stderr = subprocess.STDOUT)
    
    def Process(self, file: str, nEvents: int, start: int, pid: int) -> subprocess.Popen:
        return super().Process(file, nEvents, start, pid)

# def CalculateShowerPairs(batchID : int, directory : str, file : str, nEvents : int, nSkip : int, cuts : str, cut_type : str):
#     # no merging
#     with open(directory + "logs/" + f"shower-pairs-unmerged-{batchID}.log", "w") as log:
#         subprocess.run(["prod4a_merge_study.py", file, "-m", "unmerged", "-s", "-d", directory + "shower-pairs/", "-o", f"shower-pairs-unmerged-{batchID}", "-e", f"{nEvents}", f"{nSkip}"], stdout = log, stderr = subprocess.STDOUT)
#     # merging using backtracked MC
#     with open(directory + "logs/" + f"shower-pairs-cheat-{batchID}.log", "w") as log:
#         subprocess.run(["prod4a_merge_study.py", file, "-m", "cheat", "-s", "-d", directory + "shower-pairs/", "-o", f"shower-pairs-cheat-{batchID}", "-e", f"{nEvents}", f"{nSkip}"], stdout = log, stderr = subprocess.STDOUT)
#     # merging using cuts
#     with open(directory + "logs/" + f"shower-pairs-{cut_type}-{batchID}.log", "w") as log:
#         subprocess.run(["prod4a_merge_study.py", file, "--cuts", cuts, "--cut-type", cut_type, "-a", "-m", "reco", "-s", "-d", directory + "shower-pairs/", "-o", f"shower-pairs-{cut_type}-{batchID}", "-e", f"{nEvents}", f"{nSkip}"], stdout = log, stderr = subprocess.STDOUT)
#     return

def main(args : argparse.Namespace):
    totalEvents = 0
    totalBatches = 0
    eventsProcessed = 0
    nEvents = []
    for file in args.files:
        n = ak.count(Data(file).eventNum)
        print(f"file: {file} number of events: {n}")

        remainder = n % args.batchSize

        if args.nBatches:
            nBatches = args.nBatches    
        else:
            nBatches = (n // args.batchSize) + 1 # + 1 to account for the remainder
        print(f"remainder: {remainder}")
        print(f"number of batches: {nBatches}")

        if args.analysis == "quantities":
            p = CalculateMergeQuantities(args.directory, file, nBatches)
        if args.analysis == "shower-pairs":
            p = CalculateShowerPairs(args.directory, file, nBatches, cut_type = args.cut_type, cuts = args.cuts)
            match args.merge_method:
                case MERGE_METHOD.UNMERGED:
                    p.Process = p.ProcessUnmerged
                case MERGE_METHOD.BACKTRACKED_PFO:
                    p.Process = p.ProcessBacktrackedPFOs
                case MERGE_METHOD.RECO:
                    p.Process = p.ProcessReco
                case other:
                    raise Exception(f"{args.merge_method} is not a valid merge method")
        p.args.cpu = 12
        p.CalculateBatches()
        p.Run()

    #     i = 0
    #     while i < nBatches:
    #         if i == (n // args.batchSize):
    #             eventsToProcess = remainder
    #         else:
    #             eventsToProcess = args.batchSize
    #         if args.analysis == "quantities":
    #             CalculateMergeQuantities(totalBatches + i, args.directory, file, eventsToProcess, args.batchSize * i)
    #         else:
    #             CalculateShowerPairs(totalBatches + i, args.directory, file, eventsToProcess, args.batchSize * i, args.cuts, args.cut_type)
    #         i += 1
    #         eventsProcessed += eventsToProcess

    #     totalEvents += n
    #     totalBatches += nBatches
    #     nEvents.append(n)
    # print(f"total number of events: {totalEvents}")
    # print(f"events analyzed: {eventsProcessed}")


if __name__ == "__main__":
    example_usage = """Example Uasge:
        Produce geometric quantities:
        python run-analysis-batches.py <root file/s> -a quantities

        Produce shower pairs:
        python run-analysis-batches.py <root file/s> -a shower-pairs -c <file of cuts>
        
    """
    parser = argparse.ArgumentParser(description = "Run shower merging analysis in batches", formatter_class = argparse.RawDescriptionHelpFormatter, epilog = example_usage)
    parser.add_argument(dest = "files", type = str, nargs = "+", help = "ROOT file to open.")
    parser.add_argument("-a", "--analysis", dest = "analysis", type = str, choices = ["quantities", "shower-pairs"], help = "analysis type to perform")
    parser.add_argument("-d", "--directory", dest = "directory", type = str, default = "job_output/", help = "directory for output files")
    parser.add_argument("-s", "--batchSize", dest = "batchSize", type = int, default = 1000, help = "number of events to analyze per batch")
    parser.add_argument("-n", "--nBatches", dest = "nBatches", type = int, default = None, help = "total number of batches to process per file")
    parser.add_argument("-m", "--merge_method", dest ="merge_method", type = MERGE_METHOD, choices=list(MERGE_METHOD), default = None, help = "merge method if calculating shower pairs.")
    parser.add_argument("-c", "--cuts", dest = "cuts", type = str, default = None, help = "cuts to look at when merging")
    parser.add_argument("-t", "--cut_type", dest="cut_type", type=str, choices=["purity", "efficiency"], default="purity", help="type of cut to pick from cut scan.")
    args = parser.parse_args()
    print(vars(args))

    if not args.analysis:
        raise Exception("Need to specify which analysis to run with -a/--analysis")
    if args.analysis == "shower-pairs":
        if args.merge_method is None:
            raise Exception("Need to pick a method for merging showers with -m/--merge_method")
        if args.merge_method is MERGE_METHOD.RECO and args.cuts is None:
            raise Exception("Need to specify the list of cuts to choose from with -c/--cuts")

    os.makedirs(args.directory, exist_ok = True)
    os.makedirs(args.directory + "logs/", exist_ok = True)
    os.makedirs(args.directory + f"{args.analysis}/", exist_ok = True)

    main(args)


# use batches to get merge quantities (no plots)
#* prod4a_merge_study.py <ROOT file> -s -d <out directory> -e <nEvents><skip>

# merge the merge quantities batches into a single csv
#* csv-merger.py [-h] [-o OUTPUT] csvs [csvs ...]

# use the merge quantities csv to find the cuts (no batches)
#* prod4a_merge_study.py <merge quantity csv> -p <plot type> -s -d <out directory> -c

# merge the PFOs in batches with the single cut
#* prod4a_merge_study.py <ROOT file> -m cheat -s -o <output filename> -d <out directory> -e <nEvents><skip>
#* prod4a_merge_study.py <ROOT file> --cuts <cuts csv> --cut-type <cut type> -m reco -s -o <output filename> -d <out directory> -a
