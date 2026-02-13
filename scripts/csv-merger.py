#!/usr/bin/env python3
"""
Created on: 10/10/2022 10:21

Author: Shyam Bhuller

Description: Merges pandas dataframes which are stored in csv format.
"""
from rich import print
import pandas as pd
import argparse

def main(csvs : str, output : str):
    df = pd.concat(map(pd.read_csv, csvs), ignore_index=True)
    print(df)
    if output is not None:
        df.to_csv(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merges pandas dataframes which are stored in csv format.")
    parser.add_argument(dest="csvs", nargs="+", help="csv file/s to open.")
    parser.add_argument("-o", dest="output", default=None, type=str, help="output file name")
    args = parser.parse_args()
    main(*list(vars(args).values()))