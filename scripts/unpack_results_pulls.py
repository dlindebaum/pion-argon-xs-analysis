#!/usr/bin/env python3

import os
import argparse
import tarfile
from rich import print

import pandas as pd

from python.analysis.Utils import ls_recursive
from python.analysis.Master import ReadHDF5, DictToHDF5

def main(args : argparse.Namespace):

    # uncompress files
    out_names = []
    for f in os.listdir(args.directory):
        if "tar.gz" in f:
            out_name = args.directory + f"{f.split('.')[0]}"

            try:
                os.makedirs(out_name, exist_ok = False)
                with tarfile.open(args.directory + f) as t:
                    t.extractall(path=out_name)
            except FileExistsError:
                print("file already unpacked.")

            out_names.append(out_name)

    # load dill files
    results = []
    for f in out_names:
        for sf in ls_recursive(f):
            if sf.split("/")[-1] == "pull_results.hdf5":
                results.append(ReadHDF5(sf))
                break

    # merge results
    merged = {}
    for r in results:
        for k, v in r.items():
            if k not in merged:
                merged[k] = [v]
            else:
                merged[k].append(v) 
    merged = {k : pd.concat(objs = v, axis = "index").reset_index(drop = True) for k, v in merged.items()}
    print(merged)
    DictToHDF5(merged, args.directory + "pull_results.hdf5")
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory", type = str, help = "job output directory.", required = True)
    args = parser.parse_args()
    if args.directory[-1] != "/":
        args.directory = args.directory + "/"
    main(args)