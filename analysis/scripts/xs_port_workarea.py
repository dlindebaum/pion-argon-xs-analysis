"""
Created on: 30/01/2026 12:43

Author: Shyam Bhuller

Description: Port an analysis workarea to another file path.
"""

import argparse
import json
import os
import shutil

from python.analysis.io import read_json, write_json
from python.analysis import application

from rich import print

def update_cfg_paths(cfg : str, old : str, new : str) -> tuple[dict, str]:
    cfg_str = json.dumps(read_json(cfg))
    new_cfg = json.loads(cfg_str.replace(old, new))
    path = cfg.replace(old, new)
    return new_cfg, path


def main(args : argparse.Namespace):
    if not os.path.isdir(os.path.abspath(args.workdir)):
        raise NotADirectoryError(f"{args.workdir} does not exist.")

    # These have file paths to the data, not the work area #! a dedicated script to allow switching the data path would then be needed (or stop using file paths as keys!!!)
    # selection_masks = {k : f"{args.workdir}/masks_{k}/*.dill" for k in ["data", "mc"]}

    # Only place where work area paths are used is in the analysis configuration files, including toy config objects.

    cfgs = {}
    for i in [args.config] + args.toy_configs:
        cfg, path = update_cfg_paths(i, args.workdir, args.newdir)
        cfgs[path] = cfg
        print(cfg, path)

    # if directory already exists, just try to update the configs rather than doing complete copy
    if os.path.isdir(args.newdir):
        print("new work directory already exsits, will not copy files over and just update configs.")
    else:
        print(f"copying workarea from {args.workdir} to {args.newdir}")
        shutil.copytree(args.workdir, args.newdir) # copy files, never move

    for i in [args.config] + args.toy_configs:
        cfg, path = update_cfg_paths(i, args.workdir, args.newdir)
        write_json(cfg, path)
    print("updated paths in configuration.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    application.ApplicationArguments.config(parser, True)
    parser.add_argument("-t", "--toy-configs", type = str, nargs="+", default = [], help = "Any toy config files that need to be updated in addition to the analysis config", required = False)
    parser.add_argument("-w", "--workdir", type = str, help = "Current analysis working directory.", required = True)
    parser.add_argument("-n", "--newdir", type = str, help = "New analysis working directory.", required = True)

    args = application.ApplicationArguments.resolve_args(parser.parse_args())    
    main(args)