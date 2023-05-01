#!/usr/bin/env python3
"""
Created on: 10/10/2022 10:21

Author: Shyam Bhuller

Description: Merges pandas dataframes which are stored in hdf5 format.
"""
from rich import print
import pandas as pd
import argparse

def main(hdf5s : str, output : str):
    df = pd.concat(map(pd.read_hdf, hdf5s), ignore_index=True)
    print(df)
    if output is not None:
        df.to_hdf(output, "df")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merges pandas dataframes which are stored in hdf5 format.")
    parser.add_argument(dest="hdf5s", nargs="+", help="hdf5 file/s to open.")
    parser.add_argument("-o", dest="output", default=None, type=str, help="output file name")
    args = parser.parse_args()
    if len(args.hdf5s) < 2: raise Exception("at least 2 files are needed.")
    main(*list(vars(args).values()))