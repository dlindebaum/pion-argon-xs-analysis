"""
Created on: 12/12/2025 13:43

Author: Shyam Bhuller

Description: Run configuration for regression testing.
"""
from glob import glob
from rich import print
import subprocess
import time

from python.analysis.cross_section import ApplicationArguments, argparse, os, CalculateBatches, file_len
from python.analysis.Master import SaveConfiguration, LoadConfiguration
from scripts import run_analysis

def file_search(path : str, data_cfg : list[dict]) -> None:
    updated_data_cfg = []
    for f in data_cfg:
        name = f["file"].split("/")[-1]

        found_files = glob(f"{path}/**/{name}", recursive=True)

        if len(found_files) == 0:
            raise FileNotFoundError(f"could not find input file {name} in root file path provided: {path}")
        f.update({"file" : found_files[0]})

def main(args : argparse.Namespace):
    """
    * Load in 2 GeV config.
    * Add requirement to pass in file directory for location with the files.
    * Do a file search in the directory to get the correct MC/Data file.
    * Update local config and run (in tmp area?)
    """

    cfg_path = f"{os.environ['CONFIG_PATH']}/cex_analysis_2GeV_config.json"

    work_dir = f"/tmp/pdune_analysis_test_{time.strftime('%Y%m%d-%H%M%S')}/"

    root_file_path = "/data/dune/common/"

    if os.path.isfile(cfg_path):
        cfg = LoadConfiguration(cfg_path)
    else:
        raise FileNotFoundError(f"reference configuration file could not be found at {cfg_path}")

    # update files to ones found required to run this test.
    file_search(root_file_path, cfg["NTUPLE_FILES"]["mc"])
    print(cfg["NTUPLE_FILES"]["mc"])

    file_search(root_file_path, cfg["NTUPLE_FILES"]["data"])
    print(cfg["NTUPLE_FILES"]["data"])

    # create test area
    os.mkdir(work_dir)
    SaveConfiguration(cfg, f"{work_dir}/config.json")

    # run the script
    subprocess.run(["run_analysis.py", "-o", work_dir, "-c", f"{work_dir}/config.json"])

    # test_args = {"config" : f"{work_dir}/config.json", "out" : work_dir}
    # test_args = argparse.Namespace(**test_args)
    # print(test_args)
    # run_analysis.main(test_args)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print(args)
    main(args)