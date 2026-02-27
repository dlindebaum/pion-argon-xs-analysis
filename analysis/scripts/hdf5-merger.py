#!/usr/bin/env python3
"""
Created on: 10/10/2022 10:21

Author: Shyam Bhuller

Description: Merges pandas dataframes which are stored in hdf5 format.
"""
from rich import print
from python.analysis.Master import ReadHDF5, DictToHDF5

import pandas as pd

import argparse

def main(hdf5s : str, output : str):
    dfs = [ReadHDF5(hdf5) for hdf5 in hdf5s]
    types = list(set([type(df) for df in dfs]))
    if len(types) == 1:
        if types[0] == dict:
            df_dicts = {}
            for df in dfs:
                for k, v in df.items():
                    if k not in df_dicts:
                        df_dicts[k] = []
                    df_dicts[k].append(v)

            df = {k : pd.concat(v, ignore_index = True) for k, v in df_dicts.items()}
            print(df)
            if output is not None:
                DictToHDF5(df, output)
        else:
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