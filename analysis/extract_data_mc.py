"""
Created on: 30/07/2026.
 
Author: Yasmine Tazi.
 
Description: Turns the MC ntuples into (1) a per-PFO pkl and (2) a per-EVENT pkl,
using Will's MC selection + extraction (utils.select_mc_events + utils.extract_pfo_data).
 
This mirrors the format of extract_data_real.py (same position-Record fix, same
DataFrame save), but for MC -- so it ALSO writes an event-level table with the true
FSI topology label (absorption / charge exchange / single-pi / pion production) and
beam kinematics, which only exist for MC (truth).
 
THE JOIN
--------
extract_pfo_data and extract_event_data both walk `selected_data` in the same order
and assign the SAME global running `event_number` (a counter incremented once per
event). So each PFO row's `event_number` maps 1:1 onto a row in the event table.
They MUST be called on the same `selected_data` object in the same run.
 
POSITION FIX (see extract_data_real.py)
---------------------------------------
extract_pfo_data returns beam_end_pos / shower_start_pos / shower_direction as
awkward Record objects that balloon the pickle ~10x. I flatten them to plain
{x,y,z} float dicts and save as a DataFrame, exactly as in the data pipeline.
 
FSI TOPOLOGY (Bhuller Table 4.1)
--------------------------------
Built directly from the MC truth counts on mc.trueParticles (nPiPlus, nPiMinus,
nPi0). Charged pions of either sign are merged (ProtoDUNE has no B-field). Only
events whose beam ends in "pi+Inelastic" get a signal topology; the rest are
labelled non_signal (-1). NB: this count-based label does NOT apply the 100 MeV
KE threshold that Shyam's ProcessDefinitions uses -- swap _fsi_topology for the
framework call to match his official selection exactly.
 
NB: This is meant to run on the xs-analysis cluster, from inside the analysis tree, in the same
environment where the MC extraction notebook works.
"""
 
import os
import pickle
from collections import Counter
 
import awkward as ak
import numpy as np
import pandas as pd
 
# Will's extraction + mask helpers (unchanged) and Shyam's framework.
# NB: I do NOT import Will's select_mc_events 
# -- its Master.Data(...) call uses the OLD constructor signature (nTuple_type=...) which the current framework rejects. 
# I reimplement it below with the current FileDescriptor API, reusing Will's mask helpers untouched.
from utils import get_mc_masks, combine_fiducial_and_beam_masks, extract_pfo_data
from python.analysis import Master
 
# ---------------------------------------------------------------------------
# EDIT THESE PATHS
# ---------------------------------------------------------------------------
MC_FILES = [
    "/data/dune/common/PDSPAnalyzer_Ntuples/PDSPProd4a_MC_2GeV_sce_datadriven_ntuple_v09_81_00d01_set0.root",
    "/data/dune/common/PDSPAnalyzer_Ntuples/PDSPProd4a_MC_2GeV_sce_datadriven_ntuple_v09_81_00d01_set1.root",
    "/data/dune/common/PDSPAnalyzer_Ntuples/PDSPProd4a_MC_2GeV_sce_datadriven_ntuple_v09_81_00d01_set2.root",
    "/data/dune/common/PDSPAnalyzer_Ntuples/PDSPProd4a_MC_2GeV_sce_datadriven_ntuple_v09_81_00d01_set3.root",
    "/data/dune/common/PDSPAnalyzer_Ntuples/PDSPProd4a_MC_2GeV_reco1_sce_datadriven_v1_ntuple_v09_41_00_03.root",
]

MASKS_MC_DIR = "/home/pemb7173/xs_analysis/work/analysis_demo/masks"

# per-PFO output (matches existing MC pkl)
PFO_OUTPUT_PATH = "/home/pemb7173/PDSP-pion-classification/extracted_mc/all_data_new.pkl"

# per-EVENT output (new -- the topology + beam-KE table)
EVENTS_OUTPUT_PATH = "/home/pemb7173/PDSP-pion-classification/extracted_mc/mc_events.pkl"
 
# ---------------------------------------------------------------------------
 
# Position fields that come back as awkward Records and must be flattened.
POSITION_FIELDS = ("beam_end_pos", "shower_start_pos", "shower_direction")
 
 
def select_mc_events(mc_files, verbose=False):
    """Local copy of Will's select_mc_events, using the CURRENT Master.Data
    constructor (FileDescriptor) instead of the old nTuple_type= signature.
    Reuses Will's get_mc_masks / combine_fiducial_and_beam_masks unchanged.
    Returns a list of (filename, Master.Data) tuples, same as the original."""
    if isinstance(mc_files, str):
        mc_files = [mc_files]
 
    selected_data = []
    for mc_file in mc_files:
        mc = Master.Data(Master.FileDescriptor(mc_file, Master.Ntuple_Type.PDSP, 2), verbose=verbose)
        fiducial_mask, beam_mask = get_mc_masks(mc_file)
        prior_mask_list = combine_fiducial_and_beam_masks(fiducial_mask, beam_mask)
        prior_mask = ak.Array(prior_mask_list)
        mc_selected = mc.Filter(reco_filters=[prior_mask], true_filters=[prior_mask], returnCopy=True)
        selected_data.append((mc_file, mc_selected))
        if verbose:
            print(f"  {os.path.basename(mc_file)}: "
                  f"{int(np.sum(fiducial_mask)):,} after fiducial, "
                  f"{int(np.sum(prior_mask_list)):,} after beam")
    return selected_data
 
 
def plainify_positions(pfo_data):
    """Convert awkward-Record position fields to plain {x, y, z} float dicts.
 
    These Records otherwise drag their backing buffers into the pickle and blow
    the file up ~10x on save/re-save. Idempotent. Returns count converted."""
    n_converted = 0
    for pfo in pfo_data:
        for name in POSITION_FIELDS:
            v = pfo.get(name)
            if isinstance(v, ak.Record):
                pfo[name] = {f: float(v[f]) for f in v.fields}
                n_converted += 1
    return n_converted
 
 
def _check_no_awkward(pfo_data):
    """Warn if any field is still an awkward object after flattening."""
    if not pfo_data:
        return
    leftover = sorted({
        k for k, v in pfo_data[0].items()
        if isinstance(v, (ak.Record, ak.Array))
    })
    if leftover:
        print(f"  WARNING: fields still awkward after flatten: {leftover}")
 
 
# ---------------------------------------------------------------------------
# Event-level extraction (truth topology + beam kinematics)
# ---------------------------------------------------------------------------
 
def _fsi_topology(n_pipm, n_pi0):
    """4 FSI topologies (Bhuller Table 4.1), charged pions merged (no B-field)."""
    if n_pipm == 0 and n_pi0 == 0:
        return 0   # absorption
    if n_pipm == 0 and n_pi0 == 1:
        return 1   # charge exchange
    if n_pipm == 1 and n_pi0 == 0:
        return 2   # single-pion production
    return 3       # pion production (multi)
 
 
TOPO_NAMES = {0: "absorption", 1: "charge_exchange", 2: "single_pi",
              3: "pion_production", -1: "non_signal"}
 
 
def extract_event_data(selected_data, verbose=False):
    """One row per event, keyed by the SAME global event_num extract_pfo_data uses,
    so it joins 1:1 onto the per-PFO table."""
    if isinstance(selected_data, Master.Data):
        data_list = [selected_data]
    elif isinstance(selected_data, list) and selected_data and isinstance(selected_data[0], tuple):
        data_list = [d for (_, d) in selected_data]
    else:
        data_list = selected_data
 
    events = []
    event_num = -1
    for mc in data_list:
        has_truth = ak.count(mc.trueParticlesBT.pdg) > 0
        tp = mc.trueParticles
        for event in range(len(mc.recoParticles.track_chi2_proton)):
            event_num += 1
            row = {
                "event_number": event_num,
                "n_pfos": int(len(mc.recoParticles.track_chi2_proton[event])),
            }
            if has_truth:
                n_pipm = int(tp.nPiPlus[event]) + int(tp.nPiMinus[event])
                n_pi0 = int(tp.nPi0[event])
                end = str(tp.true_beam_endProcess[event])
                sig = (end == "pi+Inelastic")
                topo = _fsi_topology(n_pipm, n_pi0) if sig else -1
                row.update({
                    "beam_KE_front_face": float(tp.beam_KE_front_face[event]),
                    "true_beam_endProcess": end,
                    "nPiPlus": int(tp.nPiPlus[event]),
                    "nPiMinus": int(tp.nPiMinus[event]),
                    "nPi0": n_pi0,
                    "nProton": int(tp.nProton[event]),
                    "true_topology": topo,
                    "true_topology_name": TOPO_NAMES[topo],
                })
            events.append(row)
 
    if verbose:
        print(f"  events extracted: {len(events):,}")
        if events and "true_topology_name" in events[0]:
            print("  topology breakdown:", dict(Counter(e["true_topology_name"] for e in events)))
    return events
 
 
def main():
    print("Selecting MC events (fiducial + beam)...")
    selected_data = select_mc_events(MC_FILES, verbose=True)
 
    # ---- per-PFO table ------------------------------------------------------
    print("\nExtracting per-PFO data...")
    pfo_data, pfo_stats = extract_pfo_data(selected_data, max_sequence_length=222, verbose=True)
 
    n = plainify_positions(pfo_data)
    print(f"Flattened {n:,} awkward position Records to plain floats")
    _check_no_awkward(pfo_data)
 
    pfo_df = pd.DataFrame(pfo_data)
    os.makedirs(os.path.dirname(PFO_OUTPUT_PATH), exist_ok=True)
    with open(PFO_OUTPUT_PATH, "wb") as f:
        pickle.dump(pfo_df, f)
    size_gb = os.path.getsize(PFO_OUTPUT_PATH) / 1e9
    print(f"Saved {len(pfo_df):,} PFOs to {PFO_OUTPUT_PATH} ({size_gb:.2f} GB)")
 
    # ---- per-EVENT table (same selected_data -> event_number joins) ---------
    print("\nExtracting per-EVENT data...")
    event_data = extract_event_data(selected_data, verbose=True)
    event_df = pd.DataFrame(event_data)
    os.makedirs(os.path.dirname(EVENTS_OUTPUT_PATH), exist_ok=True)
    with open(EVENTS_OUTPUT_PATH, "wb") as f:
        pickle.dump(event_df, f)
    print(f"Saved {len(event_df):,} event rows to {EVENTS_OUTPUT_PATH}")
 
    # ---- join sanity check --------------------------------------------------
    n_pfo_events = pfo_df["event_number"].nunique()
    n_evt_rows = len(event_df)
    print(f"\nJoin check: PFO table spans {n_pfo_events:,} unique events; "
          f"event table has {n_evt_rows:,} rows.")
    print("  (event table >= PFO unique events is expected: some events have 0 surviving PFOs.)")
 
 
if __name__ == "__main__":
    main()