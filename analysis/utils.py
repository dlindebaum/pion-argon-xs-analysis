"""Utilities for MC event selection and PFO data extraction."""

import os

import numpy as np
import awkward as ak

from python.analysis import Master, SelectionTools
from python.analysis.Tags import GenerateTrueParticleTagsPiPlus

MASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks")


def find_particle_from_tags(tags, event, track):
    """Find particle type from tags for a given event and track."""
    for k, v in tags.items():
        if v.mask[event][track]:
            return k


def combine_fiducial_and_beam_masks(fiducial_mask, beam_mask):
    """Combine fiducial and beam masks."""
    i = -1
    combined_mask = []

    for event in range(len(fiducial_mask)):
        if fiducial_mask[event] is True:
            i += 1
            if i == len(beam_mask):
                return combined_mask
            if beam_mask[i] is True:
                combined_mask.append(True)
            else:
                combined_mask.append(False)
        else:
            combined_mask.append(False)
    return combined_mask


def get_mc_masks(mc_file):
    """Load precomputed fiducial and beam selection masks for an MC ntuple file."""
    fiducial_masks = Master.LoadObject(f"{MASKS_DIR}/fiducial_selection_masks.dill")
    beam_masks = Master.LoadObject(f"{MASKS_DIR}/beam_selection_masks.dill")

    combined_fiducial_mask = SelectionTools.CombineMasks(fiducial_masks[mc_file])
    combined_beam_mask = SelectionTools.CombineMasks(beam_masks[mc_file])

    return combined_fiducial_mask, combined_beam_mask


def select_mc_events(mc_files, verbose=False):
    """
    Load MC ntuple files, apply fiducial and beam selections using the
    precomputed masks, and return filtered Data objects with event counts.
    """
    if isinstance(mc_files, str):
        mc_files = [mc_files]

    selected_data = []
    per_file_stats = []

    total_original = 0
    total_after_fiducial = 0
    total_after_beam = 0

    for mc_file in mc_files:
        mc = Master.Data(mc_file, nTuple_type=Master.Ntuple_Type.PDSP, target_momentum=2)
        fiducial_mask, beam_mask = get_mc_masks(mc_file)

        n_original = len(mc.eventNum)
        n_after_fiducial = int(np.sum(fiducial_mask))

        prior_mask_list = combine_fiducial_and_beam_masks(fiducial_mask, beam_mask)
        n_after_beam = int(sum(prior_mask_list))

        if verbose:
            print("-" * 85)
            print(f"{mc_file}")
            print(f"  Events before fiducial cut: {n_original}")
            print(f"  Events after  fiducial cut: {n_after_fiducial}")
            print(f"  Events after  beam selection (fiducial + beam): {n_after_beam}")

        total_original += n_original
        total_after_fiducial += n_after_fiducial
        total_after_beam += n_after_beam

        prior_mask = ak.Array(prior_mask_list)
        mc_selected = mc.Filter(reco_filters=[prior_mask], true_filters=[prior_mask], returnCopy=True)

        selected_data.append((mc_file, mc_selected))
        per_file_stats.append({
            "file": mc_file,
            "n_events_original": n_original,
            "n_events_after_fiducial": n_after_fiducial,
            "n_events_after_beam": n_after_beam,
        })

    selection_stats = {
        "per_file": per_file_stats,
        "totals": {
            "n_events_original": total_original,
            "n_events_after_fiducial": total_after_fiducial,
            "n_events_after_beam": total_after_beam,
        },
    }

    if verbose:
        print("-" * 85)
        print("Combined totals across all files:")
        print(f"  Events before fiducial cut: {total_original}")
        print(f"  Events after  fiducial cut: {total_after_fiducial}")
        print(f"  Events after  beam selection (fiducial + beam): {total_after_beam}")

    return selected_data, selection_stats


def _extract_summary_statistics(mc, event, track, tags):
    """Extract summary statistics for a PFO."""
    return {
        "track_chi2/ndof_proton": (
            mc.recoParticles.track_chi2_proton[event][track]
            / mc.recoParticles.track_chi2_proton_ndof[event][track]
        ),
        "track_length": mc.recoParticles.track_len[event][track],
        "track_score": mc.recoParticles.track_score[event][track],
        "beam_end_pos": mc.recoParticles.beam_endPos[event],
        "shower_start_pos": mc.recoParticles.shower_start_pos[event][track],
        "shower_direction": mc.recoParticles.shower_direction[event][track],
        "shower_energy": mc.recoParticles.shower_energy[event][track],
        "n_hits": mc.recoParticles.n_hits[event][track],
        "n_hits_collection": mc.recoParticles.n_hits_collection[event][track],
        "particle": find_particle_from_tags(tags, event, track),
    }


def _extract_sequences(mc, event, track, tags):
    """Extract dEdX and residual-range sequences for a PFO."""
    dEdX_seq = np.array(mc.recoParticles.track_dEdX[event][track], dtype=np.float32)
    residual_range_seq = np.array(mc.recoParticles.residual_range[event][track], dtype=np.float32)

    return {
        "dEdX_sequence": dEdX_seq[::-1],
        "residual_range_sequence": residual_range_seq[::-1],
        "sequence_length": len(dEdX_seq),
        "particle": find_particle_from_tags(tags, event, track),
    }


def _pad_single_sequence(sequence, max_length, pad_value=0.0):
    """Pad or truncate a sequence to a fixed length."""
    seq_len = len(sequence)
    if seq_len < max_length:
        padded = np.pad(sequence, (0, max_length - seq_len), mode="constant", constant_values=pad_value)
    else:
        padded = sequence[:max_length]
    return padded.astype(sequence.dtype)


def extract_pfo_data(selected_data, max_sequence_length=222, verbose=False):
    """
    Extract per-PFO observables from already-selected MC samples and
    report PFO counts before and after cleaning anomalous hits.
    """
    if isinstance(selected_data, Master.Data):
        data_list = [selected_data]
    elif isinstance(selected_data, list):
        if len(selected_data) == 0:
            return [], {
                "n_pfos_before_errors": 0,
                "n_pfos_skipped_error": 0,
                "n_pfos_final": 0,
            }
        if isinstance(selected_data[0], tuple):
            data_list = [d for (_, d) in selected_data]
        else:
            data_list = selected_data
    else:
        raise TypeError(
            "selected_data must be a Master.Data instance, a list of Master.Data, "
            "or a list of (filename, Master.Data) tuples."
        )

    all_pfos = []
    dEdX_outlier_threshold = 813.9
    n_pfos_before_errors = 0
    n_pfos_skipped_error = 0
    event_num = -1

    for mc in data_list:
        tags = GenerateTrueParticleTagsPiPlus(mc)
        has_truth_info = ak.count(mc.trueParticlesBT.pdg) > 0

        for event in range(len(mc.recoParticles.track_chi2_proton)):
            n_pfos_event = len(mc.recoParticles.track_chi2_proton[event])
            n_pfos_before_errors += n_pfos_event
            event_num += 1

            for pfo_index in range(n_pfos_event):
                try:
                    sequence_info = _extract_sequences(mc, event, pfo_index, tags)

                    dEdX_seq = sequence_info["dEdX_sequence"].astype(np.float32, copy=True)
                    rr_seq = sequence_info["residual_range_sequence"].astype(np.float32, copy=True)
                    if len(dEdX_seq) != len(rr_seq):
                        raise ValueError("dEdX and residual range sequences have different lengths")

                    if len(dEdX_seq) > 0:
                        keep_mask = dEdX_seq <= dEdX_outlier_threshold
                        dEdX_seq = dEdX_seq[keep_mask]
                        rr_seq = rr_seq[keep_mask]

                        sequence_info["dEdX_sequence"] = dEdX_seq
                        sequence_info["residual_range_sequence"] = rr_seq
                        sequence_info["sequence_length"] = len(dEdX_seq)

                    summary_info = _extract_summary_statistics(mc, event, pfo_index, tags)

                    is_gamma_from_beam_pi0 = False
                    pi0_mother_id = -1
                    if has_truth_info:
                        pdg = mc.trueParticlesBT.pdg[event][pfo_index]
                        is_beam_pi0 = mc.trueParticlesBT.is_beam_pi0[event][pfo_index]
                        is_gamma = pdg == 22
                        is_gamma_from_beam_pi0 = bool(is_gamma and is_beam_pi0)
                        if is_gamma_from_beam_pi0:
                            pi0_mother_id = int(mc.trueParticlesBT.mother[event][pfo_index])

                    all_pfos.append({
                        "dEdX_sequence": _pad_single_sequence(
                            sequence_info["dEdX_sequence"], max_sequence_length
                        ),
                        "residual_range_sequence": _pad_single_sequence(
                            sequence_info["residual_range_sequence"], max_sequence_length
                        ),
                        "sequence_length": sequence_info["sequence_length"],
                        "particle": summary_info["particle"],
                        "is_gamma_from_beam_pi0": is_gamma_from_beam_pi0,
                        "pi0_mother_id": pi0_mother_id,
                        "track_chi2/ndof_proton": summary_info["track_chi2/ndof_proton"],
                        "track_length": summary_info["track_length"],
                        "track_score": summary_info["track_score"],
                        "beam_end_pos": summary_info["beam_end_pos"],
                        "shower_start_pos": summary_info["shower_start_pos"],
                        "shower_direction": summary_info["shower_direction"],
                        "shower_energy": summary_info["shower_energy"],
                        "n_hits": summary_info["n_hits"],
                        "n_hits_collection": summary_info["n_hits_collection"],
                        "event_number": event_num,
                    })

                except Exception:
                    n_pfos_skipped_error += 1
                    continue

    pfo_stats = {
        "n_pfos_before_errors": n_pfos_before_errors,
        "n_pfos_skipped_error": n_pfos_skipped_error,
        "n_pfos_final": len(all_pfos),
    }

    if verbose:
        print(f"PFOs before outlier rejection: {n_pfos_before_errors}")
        print(f"PFOs skipped due to errors:    {n_pfos_skipped_error}")
        print(f"PFOs remaining:     {pfo_stats['n_pfos_final']}")

    return all_pfos, pfo_stats
