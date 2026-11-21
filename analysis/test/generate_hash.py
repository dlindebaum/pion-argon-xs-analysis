#!/usr/bin/env python3
"""
Created on: 17/11/2026 14:57

Author: Shyam Bhuller

Description: Generates Hashes of an analysis work area. Only to be used if the standard hashig is not working.
"""
import argparse
import os

from rich import print

from python.analysis import testing


def main(args : argparse.Namespace):
    if args.workdir is not None:
        if not os.path.isdir(os.path.abspath(args.workdir)):
            raise NotADirectoryError(f"{args.workdir} does not exist.")
        return testing.create_hash_file(args.workdir)
    elif args.files is not None:
        for f in args.files:
            hash = testing.create_hash_from_file(f)
            print(f"{f} | {hash}")
    else:
        raise ValueError("Specify -w/--workdir or -f/--files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workdir", type = str, help = "Analysis working directory.", required = False)
    parser.add_argument("-f", "--files" , nargs="+", help = "single files to generate hashes for, will not write to file")


    args = parser.parse_args()
    print(args)
    main(args)