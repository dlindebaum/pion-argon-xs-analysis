"""
Created on: 13/11/2026 16:51

Author: Shyam Bhuller

Description: Tools for testing code.
"""
import hashlib
import os

from _hashlib import HASH

from . import io, common

REFERENCE_HASH_FILE = os.environ["PYTHONPATH"] + "/data/hash_dict.json"

def create_hash_from_file(file : str) -> str:
    """ Opens any file in binary format and convert to a SHA256 hex string.

    Args:
        file (str): File path.

    Returns:
        str: SHA256 hex string.
    """
    with open (file, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def create_hash_file(analysis_path : str, skip = ["pdf", "tex"], to_file : bool = True) -> dict[HASH]:
    """ For an analysis workarea, Create a hash dictionary of every file (recursively), and optionally save to file.

    Args:
        analysis_path (str): Analysis work area path.
        skip (list, optional): File extensions to exclude. Defaults to ["pdf", "tex"] (plots and tables).
        to_file (bool, optional): Write the hash dictionary to a json file. Defaults to True.

    Returns:
        dict[HASH]: Hash dictionary.
    """
    hash_dict = {}
    for f in common.ls_recursive(analysis_path):
        file = f.split(analysis_path)[-1]
        print(file.split("."))
        file, extension = file.split(".")[-2:]

        app_outdir = file.split("/")[0]
        fname = file.split("/")[-1]
        if extension not in skip:
            hash = create_hash_from_file(f)

            key = f"{fname}.{extension}"
            if app_outdir == file:
                hash_dict[file] = hash
            else:
                if  app_outdir not in hash_dict:
                    hash_dict[app_outdir] = {key : hash}
                else:
                    hash_dict[app_outdir][key] = hash

    if to_file:
        out = f"{analysis_path}/.hash_dict.json"
        print(f"Hash file written to {out}")
        io.write_json(hash_dict, out)

    return hash_dict


def compare_hashes(hash_dict_1 : dict, hash_dict_2 : dict) -> dict:
    """ Compare two hash dictionaries to see if hashes are equal.

    Args:
        hash_dict_1 (dict): first hash dictionary.
        hash_dict_2 (dict): second hash dictionary.

    Returns:
        dict: Result of comparing hash dictionaries. 
    """
    out = {}
    for k, v1 in hash_dict_1.items():
        if k not in hash_dict_2:
            print(f"key {k} is not in the hash dictionary being compared.")
            return
        v2 = hash_dict_2[k]

        if (type(v1) == dict) and (type(v2) == dict):
            res = compare_hashes(v1, v2)
            out[k] = res
        else:
            out[k] = v1 == v2
    return out


def application_main(func : callable):
    def wrapper(args):
        result = func(args)
        compare_hashes(io.read_json(REFERENCE_HASH_FILE), create_hash_file(args.out))
        return result
    return wrapper
