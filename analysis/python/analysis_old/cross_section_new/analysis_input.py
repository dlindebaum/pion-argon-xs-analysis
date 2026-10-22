import os

import numpy as np
import awkward as ak

from particle import Particle

import BetheBloch
import EnergySlice
import Fitting
import NtupleProcessing

from DetectorGeometry import ProtoDUNESPGeometry
from Master import SaveObject, IO, LoadObject, Data
from Slices import Slices
from Utils import deprecation_warning

from dataclasses import dataclass, field
from toy import Toy

from utils import RatioWeights, KE, UpstreamEnergyLoss, TruncateTrack, TrackLength, RecoEndEnergy


@dataclass
class AnalysisInput:
    # masks
    regions : dict[np.ndarray]
    inclusive_process : dict[np.ndarray]
    exclusive_process : dict[np.ndarray]
    process_id : np.ndarray = field(init=False)
    region_id : np.ndarray = field(init=False)
    outside_fv_reco : np.ndarray
    outside_fv_true : np.ndarray
    outside_tpc_reco : np.ndarray
    outside_tpc_true : np.ndarray
    # observables
    track_length_reco : np.ndarray
    track_length_truncated_reco : np.ndarray
    end_x_reco : np.ndarray
    end_y_reco : np.ndarray
    end_z_reco : np.ndarray
    start_x_reco : np.ndarray
    start_y_reco : np.ndarray
    start_z_reco : np.ndarray
    P_inst_reco : np.ndarray
    KE_inst_reco : np.ndarray
    KE_int_reco : np.ndarray
    KE_init_reco : np.ndarray
    KE_ff_reco : np.ndarray
    KE_end_reco : np.ndarray
    mean_track_score : np.ndarray
    track_length_true : np.ndarray
    track_length_truncated_true : np.ndarray
    end_x_true : np.ndarray
    end_y_true : np.ndarray
    end_z_true : np.ndarray
    start_x_true : np.ndarray
    start_y_true : np.ndarray
    start_z_true : np.ndarray
    KE_int_true : np.ndarray
    KE_init_true : np.ndarray
    KE_ff_true : np.ndarray
    KE_end_true : np.ndarray
    # extras
    weights : np.ndarray = None
    event_num : np.ndarray = None
    run : np.ndarray = None
    sub_run : np.ndarray = None

    @property
    def has_regions(self):
        return type(self.regions) == dict

    @property
    def has_exclusive_process(self):
        return type(self.exclusive_process) == dict 

    @property
    def region_labels(self):
        if not self.has_regions:
            raise Exception("Analysis input does not have well defined regions.")
        return list(self.regions.keys())

    @property
    def process_labels(self):
        if not self.has_exclusive_process:
            raise Exception("Analysis input does not have well defined exclusive processes.")
        return list(self.exclusive_process.keys())


    def __post_init__(self):
        # need to set defaults for fields otherwise they are undefined (and python doesnt complain...)
        self.region_id = None
        self.process_id = None

        if self.regions is not None:
            self.region_id = self.IDFromSamples(self.regions)
        if self.exclusive_process is not None:
            self.process_id = self.IDFromSamples(self.exclusive_process)


    def __len__(self):
        for o in ["outside_tpc_reco", "outside_tpc_true", "track_length_reco", "KE_int_reco", "KE_init_reco", "mean_track_score", "track_length_true", "KE_int_true", "KE_init_true", "weights"]:
            if getattr(self, o) is not None:
                return len(getattr(self, o))
        return


    def ToFile(self, file : str):
        """ Save to dill file.

        Args:
            file (str): file path.
        """
        SaveObject(file, self)
        return


    def ToROOTFile(self, file : str):
        """ Save to a ROOT File as a flat TTree.

        Args:
            file (str): file path.
        """
        file_writer = IO(file)
        file_writer.WriteData(vars(self), None, True)
        return


    def ToSplitROOTFiles(self, directory : str, name : str):
        """ Save to regions and samples to multiple ROOT Files, each as a flat TTree.
            Compatible with MaCh3 ROOT input files.

        Args:
            file (str): file path.
        """
        dir_name = f"{directory}/root_analysis_input_{name}"
        os.makedirs(dir_name, exist_ok = True)

        #* create different files for a combination of regions and processes, could add support for more splits.
        if self.has_regions: # Data/MC.
            for r in self.region_labels:
                new_sample = self.SelectSample(self.regions[r])
                new_sample.ToROOTFile(f"{dir_name}/{name}_R{r}")
        else: # cheated MC
            for t in self.process_labels:
                new_sample = self.SelectSample(self.exclusive_process[t])
                new_sample.ToROOTFile(f"{dir_name}/{name}_T{t}")
        return

    @staticmethod
    def FromFile(file : str) -> "AnalysisInput": #* seems a bit extra but why not
        """ Load analysis input from dill file.

        Args:
            file (str): file path.

        Returns:
            AnalysisInput: analysis input.
        """
        obj = LoadObject(file)
        if type(obj) == AnalysisInput:
            return obj
        else:
            raise Exception("not an analysis input file")


    def NInteract(self, energy_slice : Slices, process: np.ndarray, mask : np.ndarray = None, reco : bool = True, weights : np.ndarray = None) -> np.ndarray:
        """ Calculate exclusive interaction histogram using the energy slice method.

        Args:
            energy_slice (Slices): energy slices
            process (np.ndarray): exclusive process mask
            mask (np.ndarray, optional): additional mask. Defaults to None.
            reco (bool, optional): use reco KE?. Defaults to True.
            weights (np.ndarray, optional): event weights. Defaults to None.

        Returns:
            np.ndarray: exclusive interaction histogram.
        """
        deprecation_warning()
        if mask is None: mask = np.ones(len(self.KE_int_reco), dtype = bool)
        if reco is True:
            KE_int = self.KE_int_reco
            KE_init = self.KE_init_reco
            outside_fv = self.outside_fv_reco
        else:
            KE_int = self.KE_int_true
            KE_init = self.KE_init_true
            outside_fv = self.outside_fv_true
        n_interact = EnergySlice.CountingExperiment(KE_int[mask], KE_init[mask], outside_fv[mask], process[mask], energy_slice, interact_only = True, weights = weights[mask] if weights is not None else weights)
        return n_interact


    def IDFromSamples(self, sample_masks : dict[np.ndarray], uncategorised : int = -1) -> ak.Array:
        """ Create ID from sample masks. Counts from 0 and includes override for uncategorised events.

        Args:
            sample_masks (dict[np.ndarray]): sample masks, such as reco regions.
            uncategorised (int, optional): id for uncategorised objects. Defaults to -1.

        Returns:
            ak.Array: _description_
        """
        id = ak.zeros_like(list(sample_masks.values())[0]) + uncategorised
        for i, v in enumerate(sample_masks.values()):
            id = ak.where(v, i, id)
        return id


    @staticmethod
    def CreateAnalysisInputToy(toy : Toy) -> "AnalysisInput":
        """ Create analysis input from a toy sample.

        Args:
            toy (Toy): toy sample

        Returns:
            AnalysisInput: analysis input object.
        """
        deprecation_warning() #! function wont be supported further.
        inclusive_events = np.array((toy.df.inclusive_process != "decay").values)

        regions = {k : np.array(v.values) for k, v in toy.reco_regions.items()}
        process = {k : np.array(v.values) for k, v in toy.truth_regions.items()}

        return AnalysisInput(
            regions = regions,
            inclusive_process = inclusive_events,
            exclusive_process = process,
            outside_fv_reco = np.array(toy.outside_tpc_smeared.values),
            outside_fv_true = np.array(toy.outside_tpc.values),
            outside_tpc_reco = np.array(toy.outside_tpc_smeared.values),
            outside_tpc_true = np.array(toy.outside_tpc.values),
            track_length_reco = np.array(toy.df.z_int_smeared.values),
            track_length_truncated_reco = None,
            end_x_reco = None,
            end_y_reco = None,
            end_z_reco = np.array(toy.df.z_int_smeared.values),
            start_x_reco = None,
            start_y_reco = None,
            start_z_reco = None,
            P_inst_reco = None,
            KE_inst_reco = None,
            KE_int_reco = np.array(toy.df.KE_int_smeared.values),
            KE_init_reco = None,
            KE_ff_reco = np.array(toy.df.KE_init_smeared.values),
            KE_end_reco = np.array(toy.df.KE_init_smeared.values),
            mean_track_score = np.array(toy.df.mean_track_score.values),
            track_length_true = np.array(toy.df.z_int.values),
            track_length_truncated_true = None,
            end_x_true = None,
            end_y_true = None,
            end_z_true = np.array(toy.df.z_int.values),
            start_x_true = None,
            start_y_true = None,
            start_z_true = None,
            KE_int_true = np.array(toy.df.KE_int.values),
            KE_init_true = None,
            KE_ff_true = np.array(toy.df.KE_init.values),
            KE_end_true = np.array(toy.df.KE_init.values),
            weights = None,
            event_num = None,
            run = None,
            sub_run = None,
            )

    @staticmethod
    def CreateAnalysisInputNtuple(events : Data, upstream_energy_loss_params : dict, reco_regions : dict[np.ndarray] = None, true_regions : dict[np.ndarray] = None, mc_reweight_params : dict = None, mc_reweight_stength : float = 3, fiducial_volume : list[float] = [0, 700], upstream_loss_func : callable = Fitting.poly2d, energy_method : str = "track") -> "AnalysisInput":
        """ Create analysis input from an ntuple sample.

        Args:
            events (Data): ntuple sample
            upstream_energy_loss_params (dict): upstream energy loss correction
            reco_regions (dict[np.ndarray]): reco region masks
            true_regions (dict[np.ndarray], optional): true process masks. Defaults to None.
            mc_reweight_params (dict, optional): mc reweight parameters. Defaults to None.

        Returns:
            AnalysisInput: analysis input.
        """
        if mc_reweight_params is not None:
            weights = RatioWeights(events.recoParticles.beam_inst_P, "gaussian", mc_reweight_params, mc_reweight_stength)
        else:
            weights = None

        KE_inst_reco = KE(events.recoParticles.beam_inst_P, Particle.from_pdgid(211).mass)
        upstream_loss_reco = UpstreamEnergyLoss(KE_inst_reco, upstream_energy_loss_params, upstream_loss_func)
        KE_ff_reco = KE_inst_reco - upstream_loss_reco

        track_length_reco = events.recoParticles.beam_track_length

        truncated_track_reco = TruncateTrack(events.recoParticles.beam_calo_pos, z_trunc = max(fiducial_volume))
        track_length_truncated_reco = TrackLength(tracks=truncated_track_reco)

        if min(fiducial_volume) > 0:
            track_length_min = np.where(events.recoParticles.beam_endPos_SCE.z > min(fiducial_volume), min(fiducial_volume), track_length_reco)

            KE_init_reco = BetheBloch.KE_end(KE_ff_reco, track_length_min, 50) # initial kinetic energy in the fiducial volume

        else:
            KE_init_reco = KE_ff_reco

        KE_int_reco = RecoEndEnergy(events.recoParticles.beam_calo_pos, KE_ff_reco, events.recoParticles.beam_dEdX, energy_method)

        KE_end_reco = RecoEndEnergy(truncated_track_reco, KE_ff_reco, events.recoParticles.beam_dEdX, energy_method)

        outside_tpc_reco = ProtoDUNESPGeometry().outside_tpc(events.recoParticles.beam_endPos_SCE.x, events.recoParticles.beam_endPos_SCE.y, events.recoParticles.beam_endPos_SCE.z)

        outside_fv_reco = (events.recoParticles.beam_endPos_SCE.z < min(fiducial_volume)) | (events.recoParticles.beam_endPos_SCE.z > max(fiducial_volume))
        start_pos_reco = events.recoParticles.beam_startPos_SCE
        end_pos_reco = events.recoParticles.beam_endPos_SCE

        if true_regions is not None:
            KE_ff_true = events.trueParticles.beam_KE_front_face
            KE_int_true = events.trueParticles.beam_traj_KE[:, -2]
            track_length_true = events.trueParticles.beam_track_length
            start_pos_true = events.trueParticles.beam_traj_pos[:, 0]
            end_pos_true = events.trueParticles.beam_traj_pos[:, -1]

            if min(fiducial_volume) > 0:
                track_length_min = np.where(events.trueParticles.endPos.z[:, 0] > min(fiducial_volume), min(fiducial_volume), track_length_true)

                KE_init_true = BetheBloch.KE_end(KE_ff_true, track_length_min, 50) # initial kinetic energy in the fiducial volume
            else:
                KE_init_true = KE_ff_true

            outside_tpc_true = ProtoDUNESPGeometry().outside_tpc(events.trueParticles.endPos.x[:, 0], events.trueParticles.endPos.y[:, 0], events.trueParticles.endPos.z[:, 0])

            outside_fv_true = (events.trueParticles.beam_traj_pos.z[:, -1] < min(fiducial_volume)) | (events.trueParticles.beam_traj_pos.z[:, -1] > max(fiducial_volume))
            inelastic = events.trueParticles.true_beam_endProcess == "pi+Inelastic"

            truncated_tracks_true = TruncateTrack(events.trueParticles.beam_traj_pos[events.trueParticles.in_tpc_z], z_trunc = max(fiducial_volume))
            track_length_truncated_true = TrackLength(tracks=truncated_tracks_true)

            traj_KE = events.trueParticles.beam_traj_KE[events.trueParticles.in_tpc_z]
            KE_end_true = traj_KE[ak.local_index(traj_KE) == (ak.num(truncated_tracks_true)-2)]
            KE_end_true = ak.ravel(ak.fill_none(ak.pad_none(KE_end_true, 1, -1), -999, None)) # current null value for invalid true tracks is -999
            KE_init_true = np.where(np.isnan(KE_init_true), KE_end_true, KE_init_true) # for nan, set to KE_end, as KE_end should always be smaller than or equal to KE_int. can occur when KE_ff is 0.

        else:
            KE_int_true = None
            KE_init_true = None
            KE_ff_true = None
            track_length_true = None
            outside_fv_true = None
            outside_tpc_true = None
            inelastic = None
            start_pos_true = vector.vector([None], [None], [None])
            end_pos_true = vector.vector([None], [None], [None])
            KE_end_true = None
            track_length_truncated_true = None


        mean_track_score = ak.fill_none(ak.mean(events.recoParticles.track_score, axis = -1), -0.05) # fill null values in case empty events are supplied

        return AnalysisInput(
            regions = reco_regions,
            inclusive_process = inelastic,
            exclusive_process = true_regions,
            outside_fv_reco = outside_fv_reco,
            outside_fv_true = outside_fv_true,
            outside_tpc_reco = outside_tpc_reco,
            outside_tpc_true = outside_tpc_true,
            track_length_reco = track_length_reco,
            track_length_truncated_reco = track_length_truncated_reco,
            end_x_reco = end_pos_reco.x,
            end_y_reco = end_pos_reco.y,
            end_z_reco = end_pos_reco.z,
            start_x_reco = start_pos_reco.x,
            start_y_reco = start_pos_reco.y,
            start_z_reco = start_pos_reco.z,
            P_inst_reco = events.recoParticles.beam_inst_P,
            KE_inst_reco = KE_inst_reco,
            KE_int_reco = KE_int_reco,
            KE_init_reco = KE_init_reco,
            KE_ff_reco = KE_ff_reco,
            KE_end_reco = KE_end_reco,
            mean_track_score = mean_track_score,
            track_length_true = track_length_true,
            track_length_truncated_true = track_length_truncated_true,
            end_x_true = end_pos_true.x,
            end_y_true = end_pos_true.y,
            end_z_true = end_pos_true.z,
            start_x_true = start_pos_true.x,
            start_y_true = start_pos_true.y,
            start_z_true = start_pos_true.z,
            KE_int_true = KE_int_true,
            KE_init_true = KE_init_true,
            KE_ff_true = KE_ff_true,
            KE_end_true = KE_end_true,
            weights = weights,
            event_num = events.eventNum,
            run = events.run,
            sub_run = events.subRun,
            )

    @staticmethod
    def req_fields():
        return [k for k, v in AnalysisInput.__dataclass_fields__.items() if v.init is True]

    @staticmethod
    def Concatenate(ais : list["AnalysisInput"]):
        fields = NtupleProcessing.MergeOutputs([{f : getattr(a, f) for f in AnalysisInput.req_fields()} for a in ais])

        # check for null entries after merging outputs (null entries are list of Nones)
        for k in fields:
            if (type(fields[k]) == list) and (all(ak.is_none(fields[k]))):
                fields[k] = None

        return AnalysisInput(**fields)


    def SelectSample(self, mask : ak.Array) -> "AnalysisInput":
        """ Select a subset of the sample, creating a new AnalysisInput.

        Args:
            mask (ak.Array): Sample to select, can be a boolean mask or list of indices.

        Returns:
            AnalysisInput: Selected sample.
        """
        selection = {}
        for field in AnalysisInput.req_fields():
            value = getattr(self, field)
            if value is None:
                selection[field] = value # we allow Non types when defining data samples.
            if hasattr(value, "__iter__"):
                if type(value) is dict:
                    tmp_dict = {}
                    for k, v in value.items():
                        tmp_dict[k] = v[mask]
                    selection[field] = tmp_dict
                elif type(value) is list:
                    print(f"found {field} is a list type, cannot slice using an array, skipping.") #! might want to add specific logic to check if regions and processes are not null.
                    selection[field] = None
                else:
                    selection[field] = value[mask]
        return AnalysisInput(**selection)


    def CreateTrainTestSamples(self, seed : int, train_fraction : float = None) -> dict:
        """ Split analysis input into two samples

        Args:
            seed (int): seed for random permutation
            train_fraction (float, optional): fraction of events to assign to train, if None, sample is split 50/50. Defaults to None.

        Returns:
            dict: train and test samples.
        """
        rng = np.random.default_rng(seed)
        sample = rng.permutation(len(self.KE_init_reco))

        if train_fraction is None:
            fraction = len(sample) // 2
        else:
            fraction = round(train_fraction * len(sample))

        return {
            "train" : self.SelectSample(sample[:fraction]), "test" : self.SelectSample(sample[fraction:])}


    def CreateHistograms(self, energy_slice : Slices, exclusive_process : str, reco : bool, mask : np.ndarray = None) -> dict[np.ndarray]:
        """ Calculate Histogrames required for the cross section measurement using energy slicing. Note exclusive interaction histogram is without background subtraction.

        Args:
            energy_slice (Slices): energy slices
            exclusive_process (str): exclusive process
            reco (bool): use reco information?
            mask (np.ndarray, optional): additional mask. Defaults to None.

        Returns:
            dict[np.ndarray]: histograms
        """
        deprecation_warning()
        KE_int = self.KE_int_true if reco is False else self.KE_int_reco
        KE_init = self.KE_init_true if reco is False else self.KE_init_reco

        if mask is None: mask = np.zeros_like(self.outside_fv_reco, dtype = bool)

        if self.outside_fv_true is None:
            outside_tpc = self.outside_fv_reco | mask
        else:
            outside_tpc = self.outside_fv_true | mask

        if self.exclusive_process is not None:
            channel_mask = self.exclusive_process[exclusive_process]
        else:
            channel_mask = self.regions[exclusive_process]

        #! keep just in case
        # if efficiency is True:
        #     KE_int = KE_int[toy.df.beam_selection_mask]
        #     KE_init = KE_init[toy.df.beam_selection_mask]
        #     outside_tpc = outside_tpc[toy.df.beam_selection_mask]
        #     channel_mask = channel_mask[toy.df.beam_selection_mask]

        n_initial, n_interact_inelastic, n_interact_exclusive, n_incident = EnergySlice.CountingExperiment(KE_int, KE_init, outside_tpc, channel_mask, energy_slice, weights = self.weights)

        output = {"init" : n_initial, "int" : n_interact_inelastic, "int_ex" : n_interact_exclusive, "inc" : n_incident}
        return output


    def get_outside_mask(self, fiducial_volume : tuple[float]) -> ak.Array:
        """ Get the mask for events not in the ROI for the analysis. Considers FV and TPC boundary masks.

        Args:
            fiducial_volume (tuple[float]): Fiducial volume, if None, ignores the fiducial volume mask.

        Returns:
            ak.Array: outside ROI mask.
        """
        if fiducial_volume is not None:
            init_mask = self.end_z_true < min(fiducial_volume)
            mask = self.outside_tpc_true | init_mask
        else:
            mask = self.outside_tpc_true
        return mask
