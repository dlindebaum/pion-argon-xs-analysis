#!/usr/bin/env python3

import os
import argparse
import tarfile
from rich import print

from python.analysis.Utils import ls_recursive
from python.analysis.Master import LoadObject, SaveObject

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
            if sf.split("/")[-1] == "results.dill":
                results.append(LoadObject(sf))
                break
    
    # merge results
    merged = {}
    for i in results[0]:
        proc = {}
        for j in results[0][i]:
            norms = {}
            for k in results[0][i][j]:
                norms[k] = []
            proc[j] = norms
        merged[i] = proc

    for r in results:
        for i in r:
            for j in r[i]:
                for k in r[i][j]:
                    merged[i][j][k].extend(r[i][j][k])

    merged["true_cv"] = results[0]["true_cv"] # shpuld be identical for all outputs

    SaveObject(args.directory + "results.dill", merged)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory", type = str, help = "job output directory.")
    args = parser.parse_args()
    if args.directory[-1] != "/":
        args.directory = args.directory + "/"
    main(args)