# Class which stores a BDT classifier
# Dennis Lindebaum
# Created 14.06.24

import numpy as np
import awkward as ak
from sklearn.ensemble import GradientBoostingClassifier
from python.analysis import PFOSelection
import dill
from os import path


class BDTPropertyGenerator():
    def _gen_prop_defs(self):
        self.prop_defs = {
            "track": self.get_property("track_score"),
            "nhits": self.get_property("n_hits"),
            "dEdx_med": self.get_dEdX_med(),
            "dEdx_sum": self.get_dEdX_sum(),
            "dEdx_var": self.get_dEdX_var(),
            "d3Edx3_var": self.get_d3EdX3_var(),
            "dEdx_end25%": self.get_dEdX_end(end_frac=0.25),
            "dEdx_start75%": self.get_dEdX_start(end_frac=0.25),
            "dEdx_start10%": self.get_dEdX_start(end_frac=0.9),
            "chi2_pion_reduced": self.get_reduced_chi2("pion"),
            "chi2_muon_reduced": self.get_reduced_chi2("muon"),
            "chi2_proton_reduced": self.get_reduced_chi2("proton"),
            "length": self.get_property("track_len")}
        return

    def _gen_truth_defs(self):
        self.truth_defs = {
            "pion": self.select_pdgs([211, -211]),
            "photon": self.select_pdgs([22]),
            "proton": self.select_pdgs([2212]),
            "muon": self.select_pdgs([13, -13]),
            "electron": self.select_pdgs([11, -11])}
        return
    
    def __init__(self, which_props, which_truth):
        self._trained = False
        self.test_data = False
        self._gen_prop_defs()
        self._gen_truth_defs()
        self.prop_defs = {
            prop: self.prop_defs[prop] for prop in which_props}
        self.truth_defs = {
            truth: self.truth_defs[truth] for truth in which_truth}
        return
    
    def _check_trained(self):
        if not self._trained:
            raise AttributeError("BDT has not been trained.")
        return

    def select_pdgs(self, pdgs):
        if not isinstance(pdgs, list):
            pdgs = [pdgs]
        def calc(events):
            pdg_vals = events.trueParticlesBT.pdg
            mask = pdg_vals == pdgs[0]
            for pdg in pdgs[1:]:
                mask = np.logical_or(mask, pdg_vals == pdg)
            return mask
        return calc
    
    def get_property(self, name):
        def get_prop(events):
            return getattr(events.recoParticles, name)
        return get_prop

    def get_dEdX_end(self, end_frac=0.1):#, abs_split=None):
        def calc(events):
            # if abs_split is None:
            abs_split = end_frac*events.recoParticles.track_len
            end_mask = events.recoParticles.residual_range <= abs_split
            return ak.sum(events.recoParticles.track_dEdX[end_mask],
                          axis=-1)
        return calc
    
    def get_dEdX_start(self, end_frac=0.1):#, abs_split=None):
        def calc(events):
            # if abs_split is None:
            abs_split = end_frac*events.recoParticles.track_len
            start_mask = events.recoParticles.residual_range > abs_split
            return ak.sum(events.recoParticles.track_dEdX[start_mask],
                          axis=-1)
        return calc

    def get_reduced_chi2(self, particle):
        available_parts = ["pion", "muon", "proton"]
        if particle not in available_parts:
            raise ValueError(
                f"{particle} must be in the available particles: "
                + f"{available_parts}")
        def calc(events):
            chi2_score = getattr(events.recoParticles,
                                 f"track_chi2_{particle}")
            chi2_ndof = getattr(events.recoParticles,
                                f"track_chi2_{particle}_ndof")
            return chi2_score/chi2_ndof
        return calc

    def get_dEdX_med(self):
        def calc(events):
            return PFOSelection.Median(events.recoParticles.track_dEdX)
        return calc

    def get_dEdX_sum(self):
        def calc(events):
            return np.sum(events.recoParticles.track_dEdX, axis=-1)
        return calc

    def get_dEdX_var(self):
        def calc(events):
            return ak.fill_none(
                ak.var(events.recoParticles.track_dEdX, axis=-1), 0., axis=-1)
        return calc

    def get_d3EdX3_var(self):
        def calc(events):
            x = events.recoParticles.residual_range
            e = events.recoParticles.track_dEdX
            dxs = x[...,1:] - x[...,:-1]
            good_dxs = dxs != 0.
            dxs = dxs[good_dxs]
            dEdx = (e[...,1:] - e[...,:-1])[good_dxs]/dxs
            d2Edx2 = (dEdx[...,1:] - dEdx[...,:-1])/dxs[...,1:]
            return ak.fill_none(ak.var(d2Edx2, axis=-1), 0., axis=-1)
        return calc

    def _iter_over_dict(self, f_dict, events):
        for i, func in enumerate(f_dict.values()):
            val = ak.ravel(func(events))
            if i == 0:
                results = np.zeros((ak.count(val), len(f_dict)))
            results[:, i] = val
        return results

    def gen_features(self, events):
        # Sometimes dEdX has obscene results (single spike 1e22),
        # leading to numebrs > float 32. Clip to ensure we stay in
        # float32 range (not that much of a problem for BDT, which
        # does cuts)
        results_f64 = self._iter_over_dict(self.prop_defs, events)
        results_f32 = np.clip(results_f64,
                              np.finfo(np.float32).min,
                              np.finfo(np.float32).max,
                              dtype=np.float32)
        return results_f32

    def gen_truths(self, events):
        return self._iter_over_dict(self.truth_defs, events)
    
    def convert_truth_masks_to_flat_labels(self, truths):
        """
        Converts a set of PFO masks to a flat array of integers
        indicating the index of the mask the PFO belongs to. An
        additional index will be added at the end to catch PFOs which
        belong to no classes.
        """
        n_truths = truths.shape[-1]
        flat_truth = np.full(truths.shape[0], n_truths, dtype=int)
        for i in range(n_truths):
            flat_truth[truths[:, i] == 1.] = i
        return flat_truth

    def train(self, events, test_frac=None):
        self.bdt = GradientBoostingClassifier()
        features = self.gen_features(events)
        truth = self.gen_truths(events)
        flat_labels = self.convert_truth_masks_to_flat_labels(truth)
        if test_frac is not None:
            num_train = int(len(flat_labels) *(1-test_frac))
            self.test_data = True
            self.test_props = features[num_train:, :]
            self.test_labels = flat_labels[num_train:]
            features = features[:num_train, :]
            flat_labels = flat_labels[num_train:]
        self.bdt.fit(features, flat_labels)
        self._trained = True
        return

    def save(self, filepath):
        self._check_trained()
        if filepath[-4:] != ".dll":
            filepath = path.join(filepath, "bdt.dll")
        with open(filepath, "wb") as f:
            dill.dump(self, f)
        return filepath

    def predict(self, events):
        self._check_trained()
        features = self.gen_features(events)
        return self.bdt.predict(features)
    
    def predict_proba(self, events):
        self._check_trained()
        features = self.gen_features(events)
        return self.bdt.predict_proba(features)


def loadBDT(path):
    with open(path, "rb") as f:
        return dill.load(f)
