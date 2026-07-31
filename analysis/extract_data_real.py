"""
Created on: 21/07/2026 15:00.  (position-Record fix added 22/07/2026)
 
Author: Yasmine Tazi.
 
Description: Turns the REAL DATA ntuple into a PFO pkl, using the precomputed DATA masks
(masks_data/) that already exist in Shyam's analysis_demo folder.
 
This is Will's MC extraction (utils.select_mc_events + utils.extract_pfo_data)
adapted for data. The ONLY real difference is how the beam-selection mask is
obtained:
 
  * MC:   utils.get_mc_masks() loads fiducial + beam masks and combines them.
  * DATA: the fiducial cut (TrueFiducialCut) is a no-op that passes every event,
          so there is no separate data fiducial mask. We just load the
          precomputed DATA beam-selection mask from masks_data/ and use it.
 
WHY THE POSITION FIX (22/07/2026)
---------------------------------
In the LCG_110 environment, extract_pfo_data returns the position fields
(beam_end_pos, shower_start_pos, shower_direction) as awkward-array Record
objects. Each Record silently carries its parent array's backing buffers, so
pickling the raw list -- and especially re-saving it after a load round-trip --
balloons the file ~10x (a 1.2 GB extract became ~9 GB downstream). Will's MC
notebook doesn't hit this because his environment returns plain values and he
wraps the result in pd.DataFrame before saving. Here we do both: convert the
Record fields to plain {x, y, z} float dicts, then save as a DataFrame so the
output matches extracted_mc.pkl and stays compact.
 
NB: Run this on the cluster, from inside the analysis tree, in the same
environment where Will's extract-data.ipynb works.
"""
 
import os
import pickle
 
import awkward as ak
import pandas as pd
 
# Will's extraction helper (unchanged) and Shyam's framework:
from utils import extract_pfo_data
from python.analysis import Master, SelectionTools
 
# ---------------------------------------------------------------------------
# EDIT THESE THREE PATHS
# ---------------------------------------------------------------------------
DATA_FILE = "/data/dune/common/PDSPAnalyzer_Ntuples/PDSPProd4_data_2GeV_reco2_ntuple_v09_42_03_01.root"
 
# folder that contains the DATA masks
MASKS_DATA_DIR = "/home/pemb7173/xs_analysis/work/analysis_demo/masks_data"
 
# where to write the extracted data pkl (note: underscore folder name)
OUTPUT_PATH = "/home/pemb7173/PDSP-pion-classification/extracted_data/extracted_data.pkl"
 
# ---------------------------------------------------------------------------
 
# Position fields that come back as awkward Records and must be flattened.
POSITION_FIELDS = ("beam_end_pos", "shower_start_pos", "shower_direction")
 
 
def _match_key(mask_dict, data_file):
    """The mask file is keyed by ntuple filename. Find our file's entry
    robustly (exact match, then basename match, then single-key fallback)."""
    if data_file in mask_dict:
        return data_file
    base = os.path.basename(data_file)
    for k in mask_dict:
        if os.path.basename(str(k)) == base:
            return k
    keys = list(mask_dict.keys())
    if len(keys) == 1:
        return keys[0]
    raise KeyError(
        f"Could not find '{data_file}' in the mask file.\n"
        f"Available keys: {keys}\n"
        f"Set DATA_FILE to match one of these."
    )
 
 
def plainify_positions(pfo_data):
    """Convert awkward-Record position fields to plain {x, y, z} float dicts.
 
    These Records otherwise drag their backing buffers into the pickle and blow
    the file up ~10x on save/re-save. Idempotent: if a field is already a plain
    mapping it is left untouched. Returns the number of Records converted."""
    n_converted = 0
    for pfo in pfo_data:
        for name in POSITION_FIELDS:
            v = pfo.get(name)
            if isinstance(v, ak.Record):
                pfo[name] = {f: float(v[f]) for f in v.fields}
                n_converted += 1
    return n_converted
 
 
def _check_no_awkward(pfo_data):
    """Safety net: warn if any field is still an awkward object after flattening,
    so a future schema change doesn't silently re-introduce the bloat."""
    if not pfo_data:
        return
    leftover = sorted({
        k for k, v in pfo_data[0].items()
        if isinstance(v, (ak.Record, ak.Array))
    })
    if leftover:
        print(f"  WARNING: fields still awkward after flatten: {leftover}")
        print("  These will bloat the pickle -- add them to POSITION_FIELDS / flatten them.")
 
 
def select_data_events(data_file, masks_dir, verbose=True):
    """Load the data ntuple and apply the precomputed DATA beam selection."""
    # Master.Data auto-detects data from the '_data_' in the filename.
    mc = Master.Data(Master.FileDescriptor(data_file, Master.Ntuple_Type.PDSP, 2), verbose=verbose)
 
    beam_masks = Master.LoadObject(os.path.join(masks_dir, "beam_selection_masks.dill"))
    key = _match_key(beam_masks, data_file)
 
    # Combine the individual cut masks into one event-level beam-selection mask.
    prior_mask = SelectionTools.CombineMasks(beam_masks[key])
 
    if verbose:
        n_total = len(mc.eventNum)
        n_pass = int(ak.sum(prior_mask))
        print(f"{data_file}")
        print(f"  events total:            {n_total:,}")
        print(f"  events pass beam select: {n_pass:,}")
 
    # Filter to selected events (same call Will uses for MC).
    mc_selected = mc.Filter(reco_filters=[prior_mask], true_filters=[prior_mask], returnCopy=True)
    return [(data_file, mc_selected)]
 
 
def main():
    print("Selecting data events...")
    selected = select_data_events(DATA_FILE, MASKS_DATA_DIR, verbose=True)
 
    print("Extracting per-PFO data...")
    # extract_pfo_data is Will's, unchanged. It already handles no-truth data:
    # every PFO's 'particle' comes back as 'other' rather than a true label.
    pfo_data, pfo_stats = extract_pfo_data(selected, max_sequence_length=222, verbose=True)
 
    # Flatten the awkward Record position fields BEFORE saving (see module docstring).
    n = plainify_positions(pfo_data)
    print(f"Flattened {n:,} awkward position Records to plain floats")
    _check_no_awkward(pfo_data)
 
    # Save as a DataFrame to match Will's MC pipeline (extracted_mc.pkl).
    df = pd.DataFrame(pfo_data)
 
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(df, f)
 
    size_gb = os.path.getsize(OUTPUT_PATH) / 1e9
    print(f"\nSaved DataFrame with {len(df):,} PFOs to {OUTPUT_PATH} ({size_gb:.2f} GB)")
    # sanity check: positions should now be plain dicts, not awkward Records
    sample = df["beam_end_pos"].iloc[0]
    print(f"  beam_end_pos[0] type: {type(sample).__name__} -> {sample}")
    print("Next: run add_derived_fields.py on this pkl to add PFO_ID / dEdX_median / b / d / E_c.")
 
 
if __name__ == "__main__":
    main()
 