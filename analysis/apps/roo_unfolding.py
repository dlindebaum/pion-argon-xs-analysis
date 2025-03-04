"""
Created on: 12/02/2025

Author: Dennis Lindebaum

Description: runs unfolding on a pickled input dictionary.
"""
import ROOT
import RooUnfold
import numpy as np
import pickle
import argparse

def sys_weights_to_TH1D(up_down, inds, name, bins):
    hist_up = ROOT.TH1D(name+"_up", name+"_up", bins.size-1, bins.astype(float))
    hist_down = ROOT.TH1D(name+"_down", name+"_down", bins.size-1, bins.astype(float))
    for ind, w_up, w_down in zip(inds, up_down[0], up_down[1]):
        hist_up.Fill(ind, w_up)
        hist_down.Fill(ind, w_down)
    return (hist_up, hist_down)

def sys_weights_to_TH2D(up_down, inds, name, bins_reco, bins_true):
    resp_hist_up = ROOT.TH2D(name+"_up", name+"_up",
                             bins_reco.size-1, bins_reco.astype(float),
                             bins_true.size-1, bins_true.astype(float))
    resp_hist_down = ROOT.TH2D(name+"_down", name+"_down",
                               bins_reco.size-1, bins_reco.astype(float),
                               bins_true.size-1, bins_true.astype(float))
    # print(name)
    for ind_x, ind_y, w_up, w_down in zip(
            inds[0], inds[1], up_down[0], up_down[1]):
        resp_hist_up.Fill(ind_x, ind_y, w_up)
        resp_hist_down.Fill(ind_x, ind_y, w_down)
    return (resp_hist_up, resp_hist_down)

def sys_tuple_to_TH1D(up_down, weights, name, bins):
    hist_up = ROOT.TH1D(name+"_up", name+"_up", bins.size-1, bins.astype(float))
    hist_down = ROOT.TH1D(name+"_down", name+"_down", bins.size-1, bins.astype(float))
    for up, down, w in zip(up_down[0], up_down[1], weights):
        hist_up.Fill(up, w)
        hist_down.Fill(down, w)
    return (hist_up, hist_down)

def sys_meas_tuple_to_TH1D_and_resp(up_down, weights, name, bins, truth, bins_true):
    hist_up = ROOT.TH1D(name+"_up", name+"_up", bins.size-1, bins.astype(float))
    hist_down = ROOT.TH1D(name+"_down", name+"_down", bins.size-1, bins.astype(float))
    resp_err_up = ROOT.TH2D(name+"_resp_up", name+"_resp_up",
                         bins.size-1, bins.astype(float),
                         bins_true.size-1, bins_true.astype(float))
    resp_err_down = ROOT.TH2D(name+"_resp_down", name+"_resp_down",
                         bins.size-1, bins.astype(float),
                         bins_true.size-1, bins_true.astype(float))
    for up, down, true, w in zip(up_down[0], up_down[1], truth, weights):
        hist_up.Fill(up, w)
        hist_down.Fill(down, w)
        resp_err_up.Fill(up, true, w)
        resp_err_down.Fill(down, true, w)
    return (hist_up, hist_down), (resp_err_up, resp_err_down)

def sys_true_tuple_to_TH1D_and_resp(up_down, weights, name, bins, reco, bins_reco):
    hist_up = ROOT.TH1D(name+"_up", name+"_up", bins.size-1, bins.astype(float))
    hist_down = ROOT.TH1D(name+"_down", name+"_down", bins.size-1, bins.astype(float))
    resp_err_up = ROOT.TH2D(name+"_resp_up", name+"_resp_up",
                         bins_reco.size-1, bins_reco.astype(float),
                         bins_true.size-1, bins_true.astype(float))
    resp_err_down = ROOT.TH2D(name+"_resp_down", name+"_resp_down",
                         bins_reco.size-1, bins_reco.astype(float),
                         bins.size-1, bins.astype(float))
    for up, down, true, w in zip(up_down[0], up_down[1], reco, weights):
        hist_up.Fill(up, w)
        hist_down.Fill(down, w)
        resp_err_up.Fill(reco, up, w)
        resp_err_down.Fill(reco, down, w)
    return (hist_up, hist_down), (resp_err_up, resp_err_down)

def measure_with_resp(
        uf_info, key, meas_sys_list, resp_sys_list,
        reco_bins, true_bins, true_by_meas, weights_by_meas):
    stem = key[4:]
    resp_err_up = ROOT.TH2D(
        stem+"_resp_up", stem+"_resp_up",
        reco_bins.size-1, reco_bins.astype(float),
        true_bins.size-1, true_bins.astype(float))
    resp_err_down = ROOT.TH2D(
        stem+"_resp_down", stem+"_resp_down",
        reco_bins.size-1, reco_bins.astype(float),
        true_bins.size-1, true_bins.astype(float))
    for i, info in enumerate(uf_info[key]):
        if len(info[0]) > 0:
            name = stem + f"_{i}"
            ud_tup = sys_tuple_to_TH1D(
                info, weights_by_meas[i], name, reco_bins)
            meas_sys_list.append((name,) + ud_tup)
            for up, down, true, w in zip(
                    info[0], info[1], true_by_meas[i], weights_by_meas[i]):
                resp_err_up.Fill(up, true, w)
                resp_err_down.Fill(down, true, w)
    resp_sys_list.append((stem+"_resp_all", resp_err_up, resp_err_down))
    return meas_sys_list, resp_sys_list

def truth_with_resp(
        uf_info, key, true_sys_list, resp_sys_list,
        reco_bins, true_bins, meas_by_true, weights_by_true):
    stem = key[4:]
    resp_err_up = ROOT.TH2D(
        stem+"_resp_up", stem+"_resp_up",
        reco_bins.size-1, reco_bins.astype(float),
        true_bins.size-1, true_bins.astype(float))
    resp_err_down = ROOT.TH2D(
        stem+"_resp_down", stem+"_resp_down",
        reco_bins.size-1, reco_bins.astype(float),
        true_bins.size-1, true_bins.astype(float))
    for i, info in enumerate(uf_info[key]):
        if len(info[0]) > 0:
            name = stem + f"_{i}"
            ud_tup = sys_tuple_to_TH1D(
                info, weights_by_true[i], name, true_bins)
            meas_sys_list.append((name,) + ud_tup)
            for up, down, meas, w in zip(
                    info[0], info[1], meas_by_true[i], weights_by_true[i]):
                resp_err_up.Fill(meas, up, w)
                resp_err_down.Fill(meas, down, w)
    resp_sys_list.append((stem+"_resp_all", resp_err_up, resp_err_down))
    return meas_sys_list, resp_sys_list

def generate_vectors_from_spec(
        spec, bias_calc_truth,
        which_method=ROOT.RooUnfolding.kBayes,
        e_vec_method=None,
        use_sys=ROOT.RooUnfolding.kAll,
        n_iterations=4, n_toys=10000,
        prints=False):
    uf_func = spec.makeFunc(which_method,n_iterations)
    uf = uf_func.unfolding()
    if use_sys is not None:
        uf.IncludeSystematics(use_sys)
    uf_vec = uf.Vunfold()
    uf_mat = uf.UnfoldingMatrix()
    uf.SetNToys(n_toys)
    """Returns vector of unfolding errors computed according to the withError flag:
    0: Errors are the square root of the bin content
    1: Errors from the diagonals of the covariance matrix given by the unfolding
    2: Errors from the covariance matrix given by the unfolding
    3: Errors from the covariance matrix from the variation of the results in toy MC tests"""
    if e_vec_method is not None:
        errors_vec = uf.EunfoldV(e_vec_method)
    else:
        errors_vec = uf.EunfoldV()
    uf.CalculateBias(n_toys, spec.makeHistogram(bias_calc_truth))
    bias = uf.Vbias()
    bias_err = uf.Ebias()
    cov = uf.CoverageProbV()
    if prints:
        uf.Print()
        bias.Print()
        cov.Print()
    print(f"Used systematics: {uf.SystematicsIncluded()}")
    return uf_vec, errors_vec, uf_mat, bias, bias_err, cov

def get_TH1Ds_from_info(info_dict):
    reco_bins = info_dict["reco_bin_edges"]
    true_bins = info_dict["true_bin_edges"]

    recoHistTruth = ROOT.TH1D("reco_truth", "reco_truth", reco_bins.size-1, reco_bins.astype(float))
    trueHistTruth = ROOT.TH1D("true_truth", "true_truth", true_bins.size-1, true_bins.astype(float))
    responseCombHist_train = ROOT.TH2D("response_truth", "response_truth",
                                    reco_bins.size-1, reco_bins.astype(float),
                                    true_bins.size-1, true_bins.astype(float))

    recoHistData = ROOT.TH1D("reco_data", "reco_data", reco_bins.size-1, reco_bins.astype(float))


    for reco_ind, true_ind, w in zip(
            info_dict["truth_multi_dim_reco"],
            info_dict["truth_multi_dim_true"],
            info_dict["truth_weights"]):
        recoHistTruth.Fill(reco_ind, w)
        trueHistTruth.Fill(true_ind, w)
        responseCombHist_train.Fill(reco_ind, true_ind, w)

    for ind, w in zip(info_dict["data_multi_dim"],
                   info_dict["reco_weights"]):
        recoHistData.Fill(ind, w)
    return recoHistData, recoHistTruth, trueHistTruth, responseCombHist_train

def get_sys_hists_from_info(info_dict):
    sys_keys = [k for k in info_dict.keys() if k[:4] == "sys_"]
    data_sys = []
    meas_sys = []
    true_sys = []
    resp_sys = []
    reco_bins = info_dict["reco_bin_edges"]
    true_bins = info_dict["true_bin_edges"]
    reco_inds = info_dict["reco_weight_inds"]
    true_inds = info_dict["true_weight_inds"]
    resp_inds = info_dict["resp_weight_inds"]
    # truth_w = info_dict["truth_weights"]
    # data_w = info_dict["reco_weights"]
    # true_by_meas = info_dict["true_resp_by_meas"]
    # meas_by_true = info_dict["meas_resp_by_true"]
    # data_weights_by_reco = info_dict["truth_weights_split_meas"]
    # weights_by_meas = info_dict["truth_weights_split_meas"]
    # weights_by_true = info_dict["truth_weights_split_true"]
    for key in sys_keys:
        if key[4:9] == "data_":
            stem = key[4:]
            for i, info in enumerate(info_dict[key]):
                name = stem + f"_{i}"
                ud_tup = sys_weights_to_TH1D(info, reco_inds, name, reco_bins)
                # ud_tup = sys_tuple_to_TH1D(info, data_weights_by_reco[i], name, reco_bins)
                data_sys.append((name,) + ud_tup)
        elif key[4:9] == "meas_":
            for i, info in enumerate(info_dict[key]):
                name = stem + f"_{i}"
                ud_tup = sys_weights_to_TH1D(info, reco_inds, name, reco_bins)
                meas_sys.append((name,) + ud_tup)
            # meas_sys, resp_sys = measure_with_resp(
            #     info_dict, key, meas_sys, resp_sys,
            #     reco_bins, true_bins, true_by_meas, weights_by_meas)
            # stem = key[4:]
            # for i, info in enumerate(info_dict[key]):
            #     name = stem + f"_{i}"
            #     ud_tup, resp_tup = sys_meas_tuple_to_TH1D_and_resp(
            #         info, weights_by_meas[i], name, reco_bins,
            #         true_by_meas[i], true_bins)
            #     meas_sys.append((name,) + ud_tup)
                # resp_sys.append((name+"_resp",) + resp_tup)
        elif key[4:9] == "true_":
            for i, info in enumerate(info_dict[key]):
                name = stem + f"_{i}"
                ud_tup = sys_weights_to_TH1D(info, true_inds, name, true_bins)
                true_sys.append((name,) + ud_tup)
            # true_sys, resp_sys = truth_with_resp(
            #     info_dict, key, true_sys, resp_sys,
            #     reco_bins, true_bins, weights_by_true)
            # stem = key[4:]
            # for i, info in enumerate(info_dict[key]):
            #     name = stem + f"_{i}"
            #     ud_tup, resp_tup = sys_true_tuple_to_TH1D_and_resp(
            #         info, weights_by_true[i], name, true_bins,
            #         meas_by_true[i], reco_bins)
            #     true_sys.append((name,) + ud_tup)
                # resp_sys.append((name+"_resp",) + resp_tup)
        elif key[4:9] == "resp_":
            for i, info in enumerate(info_dict[key]):
                name = stem + f"_{i}"
                ud_tup = sys_weights_to_TH2D(
                    info, resp_inds, name, reco_bins, true_bins)
                resp_sys.append((name,) + ud_tup)
        else:
            raise ValueError("Unknown systematic type: " + key)
    return data_sys, meas_sys, true_sys, resp_sys

def convert_uf_mat_to_arr(uf_mat):
    n_rows = uf_mat.GetNrows()
    n_cols = uf_mat.GetNcols()
    mat_arr = np.array(uf_mat.GetMatrixArray()[:n_rows*n_cols])
    return np.reshape(mat_arr, (n_rows, n_cols))

def main(info_file, out_file):
    info_dict = pickle.load(info_file)
    info_file.close()
    
    recoData, recoTruth, trueTruth, responseTruth = get_TH1Ds_from_info(info_dict)
    data_sys, meas_sys, true_sys, resp_sys = get_sys_hists_from_info(info_dict)
    print(f"Found systematics: {len(data_sys)} on data, {len(meas_sys)} on reco. truth, "
          f"{len(true_sys)} on true truth, and {len(resp_sys)} on the truth response matrix.")
    uf_spec = ROOT.RooUnfoldSpec(
        "unfold_spec","Unfolding",
        trueTruth,"mc_truth",
        recoTruth,"mc_reco",
        responseTruth,
        recoData,
        False)
    systs = [data_sys, meas_sys, true_sys, resp_sys]
    sys_types = [ROOT.RooUnfoldSpec.kData, ROOT.RooUnfoldSpec.kMeasured,
                 ROOT.RooUnfoldSpec.kTruth, ROOT.RooUnfoldSpec.kResponse]
    for sy, ty in zip(systs, sys_types):
        if len(sy) > 0:
            for name, up, down in sy:
                uf_spec.registerSystematic(ty, name, up, down) 
    unfolded, uf_errors, uf_mat, bias, bias_errs, coverages = generate_vectors_from_spec(
        uf_spec, trueTruth, use_sys=ROOT.RooUnfolding.kAll,
        which_method=ROOT.RooUnfolding.kBayes,
        e_vec_method=ROOT.RooUnfolding.kErrorsToys,
        n_iterations=9, n_toys=100, prints=False)
    unfolded_vec = np.array(unfolded)
    unfolded_errs = np.array(uf_errors)
    unfolding_mat = convert_uf_mat_to_arr(uf_mat)

    unfolded_stat, uf_errors_stat, _, _, _, _ = generate_vectors_from_spec(
        uf_spec, trueTruth, use_sys=ROOT.RooUnfolding.kNoSystematics,
        which_method=ROOT.RooUnfolding.kBayes,
        e_vec_method=ROOT.RooUnfolding.kErrorsToys,
        n_iterations=4, n_toys=100, prints=False)
    uf_stat_vec = np.array(unfolded_stat)
    # print(uf_stat_vec)
    # print(unfolded_vec)
    # assert np.all(uf_stat_vec == unfolded_vec), "Failed to match systematic and statistic results"
    uf_stat_errs = np.array(uf_errors_stat)

    outputs={
        "unfolded_hist": unfolded_vec,
        "unfolded_errs": unfolded_errs,
        "unfolded_stat_errs": uf_stat_errs,
        "unfolding_matrix": unfolding_mat}
    pickle.dump(outputs, out_file)
    out_file.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Uses RooUnfold to unfold some multi-dimensional reco. energy bins.",
        formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-f", "--info-file", dest = "info_file", type = argparse.FileType("rb"), help = "Pickle file containing the unfolding information as a dictionary.")
    parser.add_argument("-o", "--out-file", dest = "out_file", type = argparse.FileType("wb"), help = "File as which to save the unfolding output dictionary.")

    args = parser.parse_args()
    main(args.info_file, args.out_file)