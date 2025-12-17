#!/usr/bin/env python3
"""
Created on: 12/12/2025 13:43

Author: Shyam Bhuller

Description: Run configuration for regression testing.
"""
import argparse
from glob import glob
import os
from rich import print, rule
import subprocess
import time

from python.analysis.cross_section import ApplicationArguments
from python.analysis.Master import SaveConfiguration, LoadConfiguration, timer

def file_search(path : str, data_cfg : dict) -> None:
    """ Search for files based on the names provided in the configuration, then update the paths in the configuration data.

    Args:
        path (str): Path to search for. Search is done recursively.
        data_cfg (dict): Configuration data.
    """
    for f in data_cfg:
        name = f["file"].split("/")[-1]

        found_files = glob(f"{path}/**/{name}", recursive=True)

        if len(found_files) == 0:
            raise FileNotFoundError(f"could not find input file {name} in root file path provided: {path}")
        f.update({"file" : found_files[0]})

@timer
def run_regression_test(args : argparse.Namespace):
    work_dir = f"{args.work_dir}/pdune_analysis_test_{time.strftime('%Y%m%d-%H%M%S')}/"

    # check for configuration
    if os.path.isfile(args.config):
        cfg = LoadConfiguration(args.config)
    else:
        raise FileNotFoundError(f"reference configuration file could not be found at {args.config}")

    # try find the paths to the Data and MC files.
    if os.path.isdir(args.root_file_path):
        file_search(args.root_file_path, cfg["NTUPLE_FILES"]["mc"])
        print(rule.Rule("MC files"))
        print(cfg["NTUPLE_FILES"]["mc"])

        file_search(args.root_file_path, cfg["NTUPLE_FILES"]["data"])
        print(rule.Rule("Data files"))
        print(cfg["NTUPLE_FILES"]["data"])
    else:
        raise NotADirectoryError(f"{args.root_file_path} is not a directory.")


    print(f"Test area to be created: {work_dir}") 
    if not args.debug:
        # create test area
        os.mkdir(work_dir)
        SaveConfiguration(cfg, f"{work_dir}/config.json")

        # run the script
        subprocess.run(["run_analysis.py", "-o", work_dir, "-c", f"{work_dir}/config.json"])
        subprocess.run(["run_analysis.py", "-o", work_dir, "-c", f"{work_dir}/config.json", "--run", "analyse"]) # need to specify to run the xs anaysis.
        print("Test completed.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ApplicationArguments.Config(parser, required = False, default = f"{os.environ['CONFIG_PATH']}/cex_analysis_2GeV_config.json")
    parser.add_argument("-d", "--directory", dest = "work_dir", default="/tmp", help="Directory where the test is ran.")
    parser.add_argument("--debug", action = "store_true", help = "dont actually run the test, just verify the file path for the data is found.")
    parser.add_argument("-f", "--files", dest = "root_file_path", required = True, help = "File path the Data and MC root files are stored in (will do a recusive search).")
    args = parser.parse_args()
    print(args)
    run_regression_test(args)