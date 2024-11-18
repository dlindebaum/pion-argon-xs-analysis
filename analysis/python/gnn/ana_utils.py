# Created 17/10/24
# Dennis Lindebaum
# Compatibility functions for integrating the GNN into analysis

import numpy as np
from python.gnn import DataPreparation, Models
from python.analysis import Plots, Master, cross_section
import apps.cex_analysis_input as cai

# @dataclass(init=False)
# class GNNAnalysisInput(cross_section.AnalysisInput):
#     # gnn info
#     scores : np.ndarray
#     gnn_ids : np.ndarray

#     # # masks
#     # regions : dict[np.ndarray]
#     # inclusive_process : dict[np.ndarray]
#     # exclusive_process : dict[np.ndarray]
#     # outside_tpc_reco : np.ndarray
#     # outside_tpc_true : np.ndarray
#     # # observables
#     # track_length_reco : np.ndarray
#     # KE_int_reco : np.ndarray
#     # KE_init_reco : np.ndarray
#     # KE_ff_reco : np.ndarray
#     # mean_track_score : np.ndarray
#     # track_length_true : np.ndarray
#     # KE_int_true : np.ndarray
#     # KE_init_true : np.ndarray
#     # KE_ff_true : np.ndarray
#     # # extras
#     # weights : np.ndarray = None

#     def __init__(
#             self,
#             scores,
#             gnn_ids,
#             regions,
#             inclusive_process,
#             exclusive_process,
#             outside_tpc_reco,
#             outside_tpc_true,
#             track_length_reco,
#             KE_int_reco,
#             KE_init_reco,
#             KE_ff_reco,
#             mean_track_score,
#             track_length_true,
#             KE_int_true,
#             KE_init_true,
#             KE_ff_true,
#             weights=None
#         ):
#         super().__init__(
#             regions,
#             inclusive_process,
#             exclusive_process,
#             outside_tpc_reco,
#             outside_tpc_true,
#             track_length_reco,
#             KE_int_reco,
#             KE_init_reco,
#             KE_ff_reco,
#             mean_track_score,
#             track_length_true,
#             KE_int_true,
#             KE_init_true,
#             KE_ff_true,
#             weights=None)
        

#     @staticmethod
#     def FromFile(file : str) -> "AnalysisInput": #* seems a bit extra but why not
#         """ Load analysis input from dill file.

#         Args:
#             file (str): file path.

#         Returns:
#             AnalysisInput: analysis input.
#         """
#         obj = LoadObject(file)
#         if type(obj) == AnalysisInput:
#             return obj
#         else:
#             raise Exception("not an analysis input file")


#     def NInteract(self, energy_slice : Slices, process: np.ndarray, mask : np.ndarray = None, reco : bool = True, weights : np.ndarray = None) -> np.ndarray:
#         """ Calculate exclusive interaction histogram using the energy slice method.

#         Args:
#             energy_slice (Slices): energy slices
#             process (np.ndarray): exclusive process mask
#             mask (np.ndarray, optional): additional mask. Defaults to None.
#             reco (bool, optional): use reco KE?. Defaults to True.
#             weights (np.ndarray, optional): event weights. Defaults to None.

#         Returns:
#             np.ndarray: exclusive interaction histogram.
#         """
#         if mask is None: mask = np.ones(len(self.KE_int_reco), dtype = bool)
#         if reco is True:
#             KE_int = self.KE_int_reco
#             KE_init = self.KE_init_reco
#             outside_tpc = self.outside_tpc_reco
#         else:
#             KE_int = self.KE_int_true
#             KE_init = self.KE_init_true
#             outside_tpc = self.outside_tpc_true
#         n_interact = EnergySlice.CountingExperiment(KE_int[mask], KE_init[mask], outside_tpc[mask], process[mask], energy_slice, interact_only = True, weights = weights[mask] if weights is not None else weights)
#         return n_interact

#     @staticmethod
#     def CreateAnalysisInputToy(toy : Toy) -> "AnalysisInput":
#         """ Create analysis input from a toy sample.

#         Args:
#             toy (Toy): toy sample

#         Returns:
#             AnalysisInput: analysis input object.
#         """
#         inclusive_events = np.array((toy.df.inclusive_process != "decay").values)

#         regions = {k : np.array(v.values) for k, v in toy.reco_regions.items()}
#         process = {k : np.array(v.values) for k, v in toy.truth_regions.items()}

#         return AnalysisInput(
#             regions,
#             inclusive_events,
#             process,
#             np.array(toy.outside_tpc_smeared.values),
#             np.array(toy.outside_tpc.values),
#             np.array(toy.df.z_int_smeared.values),
#             np.array(toy.df.KE_int_smeared.values),
#             np.array(toy.df.KE_init_smeared.values),
#             np.array(toy.df.KE_init_smeared.values),
#             np.array(toy.df.mean_track_score.values),
#             np.array(toy.df.z_int.values),
#             np.array(toy.df.KE_int.values),
#             np.array(toy.df.KE_init.values),
#             np.array(toy.df.KE_init.values),
#             None
#             )

#     @staticmethod
#     def CreateAnalysisInputNtuple(events : Data, upstream_energy_loss_params : dict, reco_regions : dict[np.ndarray], true_regions : dict[np.ndarray] = None, mc_reweight_params : dict = None, mc_reweight_stength : float = 3, fiducial_volume : list[float] = [0, 700], upstream_loss_func : callable = Fitting.poly2d) -> "AnalysisInput":
#         """ Create analysis input from an ntuple sample.

#         Args:
#             events (Data): ntuple sample
#             upstream_energy_loss_params (dict): upstream energy loss correction
#             reco_regions (dict[np.ndarray]): reco region masks
#             true_regions (dict[np.ndarray], optional): true process masks. Defaults to None.
#             mc_reweight_params (dict, optional): mc reweight parameters. Defaults to None.

#         Returns:
#             AnalysisInput: analysis input.
#         """
#         if mc_reweight_params is not None:
#             weights = RatioWeights(events.recoParticles.beam_inst_P, "gaussian", mc_reweight_params, mc_reweight_stength)
#         else:
#             weights = None

#         reco_KE_inst = EnergyTools.KE(events.recoParticles.beam_inst_P, Particle.from_pdgid(211).mass)
#         reco_upstream_loss = UpstreamEnergyLoss(reco_KE_inst, upstream_energy_loss_params, upstream_loss_func)
#         reco_KE_ff = reco_KE_inst - reco_upstream_loss

#         if min(fiducial_volume) > 0:
#             reco_KE_init = EnergyTools.BetheBloch.InteractingKE(reco_KE_ff, min(fiducial_volume) * np.ones_like(reco_KE_ff), 25) # initial kinetic energy in the fiducial volume
#         else:
#             reco_KE_init = reco_KE_ff

#         reco_KE_int = reco_KE_ff - EnergyTools.RecoDepositedEnergy(events, reco_KE_ff, "bb") # interacting kinetic energy
#         reco_track_length = events.recoParticles.beam_track_length
#         outside_tpc_reco = (events.recoParticles.beam_endPos_SCE.z < min(fiducial_volume)) | (events.recoParticles.beam_endPos_SCE.z > max(fiducial_volume))


#         if true_regions is not None:
#             true_KE_ff = events.trueParticles.beam_KE_front_face

#             if min(fiducial_volume) > 0:
#                 true_KE_init = EnergyTools.BetheBloch.InteractingKE(true_KE_ff, min(fiducial_volume) * np.ones_like(true_KE_ff), 25) # initial kinetic energy in the fiducial volume
#             else:
#                 true_KE_init = true_KE_ff


#             true_KE_int = events.trueParticles.beam_traj_KE[:, -2]
#             true_track_length = events.trueParticles.beam_track_length
#             outside_tpc_true = (events.trueParticles.beam_traj_pos.z[:, -1] < min(fiducial_volume)) | (events.trueParticles.beam_traj_pos.z[:, -1] > max(fiducial_volume))
#             inelastic = events.trueParticles.true_beam_endProcess == "pi+Inelastic"

#         else:
#             true_KE_int = None
#             true_KE_init = None
#             true_KE_ff = None
#             true_track_length = None
#             outside_tpc_true = None
#             inelastic = None

#         mean_track_score = ak.fill_none(ak.mean(events.recoParticles.track_score, axis = -1), -0.05) # fill null values in case empty events are supplied

#         return AnalysisInput(
#             reco_regions,
#             inelastic,
#             true_regions,
#             outside_tpc_reco,
#             outside_tpc_true,
#             reco_track_length,
#             reco_KE_int,
#             reco_KE_init,
#             reco_KE_ff,
#             mean_track_score,
#             true_track_length,
#             true_KE_int,
#             true_KE_init,
#             true_KE_ff,
#             weights,
#             )


#     @staticmethod
#     def Concatenate(ais : list["AnalysisInput"]):
#         fields = MergeOutputs([vars(a) for a in ais])
#         return AnalysisInput(**fields)


#     def CreateTrainTestSamples(self, seed : int, train_fraction : float = None) -> dict:
#         """ Split analysis input into two samples

#         Args:
#             seed (int): seed for random permutation
#             train_fraction (float, optional): fraction of events to assign to train, if None, sample is split 50/50. Defaults to None.

#         Returns:
#             dict: train and test samples.
#         """
#         rng = np.random.default_rng(seed)
#         sample = rng.permutation(len(self.KE_init_reco))

#         if train_fraction is None:
#             fraction = len(sample) // 2
#         else:
#             fraction = round(train_fraction * len(sample))

#         train_indices = sample[:fraction]
#         test_indices = sample[fraction:]

#         train = {}
#         test = {}
#         for attr in vars(self):
#             value = getattr(self, attr)
#             if hasattr(value, "__iter__"):
#                 if type(value) is dict:
#                     tmp_dict_train = {}
#                     tmp_dict_test = {}
#                     for k, v in value.items():
#                         tmp_dict_train[k] = v[train_indices]
#                         tmp_dict_test[k] = v[test_indices]
#                     train[attr] = tmp_dict_train
#                     test[attr] = tmp_dict_test
#                 else:
#                     train[attr] = value[train_indices]
#                     test[attr] = value[test_indices]

#         return {"train" : AnalysisInput(**train), "test" : AnalysisInput(**test)}


#     def CreateHistograms(self, energy_slice : Slices, exclusive_process : str, reco : bool, mask : np.ndarray = None) -> dict[np.ndarray]:
#         """ Calculate Histogrames required for the cross section measurement using energy slicing. Note exclusive interaction histogram is without background subtraction.

#         Args:
#             energy_slice (Slices): energy slices
#             exclusive_process (str): exclusive process
#             reco (bool): use reco information?
#             mask (np.ndarray, optional): additional mask. Defaults to None.

#         Returns:
#             dict[np.ndarray]: histograms
#         """
#         KE_int = self.KE_int_true if reco is False else self.KE_int_reco
#         KE_init = self.KE_init_true if reco is False else self.KE_init_reco

#         if mask is None: mask = np.zeros_like(self.outside_tpc_reco, dtype = bool)

#         if self.outside_tpc_true is None:
#             outside_tpc = self.outside_tpc_reco | mask
#         else:
#             outside_tpc = self.outside_tpc_true | mask

#         if self.exclusive_process is not None:
#             channel_mask = self.exclusive_process[exclusive_process]
#         else:
#             channel_mask = self.regions[exclusive_process]

#         #! keep just in case
#         # if efficiency is True:
#         #     KE_int = KE_int[toy.df.beam_selection_mask]
#         #     KE_init = KE_init[toy.df.beam_selection_mask]
#         #     outside_tpc = outside_tpc[toy.df.beam_selection_mask]
#         #     channel_mask = channel_mask[toy.df.beam_selection_mask]

#         n_initial, n_interact_inelastic, n_interact_exclusive, n_incident = EnergySlice.CountingExperiment(KE_int, KE_init, outside_tpc, channel_mask, energy_slice, weights = self.weights)

#         output = {"init" : n_initial, "int" : n_interact_inelastic, "int_ex" : n_interact_exclusive, "inc" : n_incident}
#         return output


def load_events_from_config():
    pass

def generate_network_predictions():
    pass

def confirm_config_valid_gnn():
    pass