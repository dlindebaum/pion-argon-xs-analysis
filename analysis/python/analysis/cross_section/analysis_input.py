import argparse
import os

from dataclasses import dataclass, field

import numpy as np
import awkward as ak

from particle import Particle

from .toy import Toy

from python.analysis import bethe_bloch, beam, common, fitting, geometry, io, kinematics, ntuple, processing, regions, selection, slices, vector

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
            self.region_id = self.id_from_samples(self.regions)
        if self.exclusive_process is not None:
            self.process_id = self.id_from_samples(self.exclusive_process)


    def __len__(self):
        for o in ["outside_tpc_reco", "outside_tpc_true", "track_length_reco", "KE_int_reco", "KE_init_reco", "mean_track_score", "track_length_true", "KE_int_true", "KE_init_true", "weights"]:
            if getattr(self, o) is not None:
                return len(getattr(self, o))
        return


    def to_file(self, file : str):
        """ Save to dill file.

        Args:
            file (str): file path.
        """
        io.write_dill(file, self)
        return


    def to_root_file(self, file : str):
        """ Save to a ROOT File as a flat TTree.

        Args:
            file (str): file path.
        """
        file_writer = io.RootIO(file)
        file_writer.write_data(vars(self), None, True)
        return


    def to_split_root_files(self, directory : str, name : str):
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
                new_sample = self.select_sample(self.regions[r])
                new_sample.to_root_file(f"{dir_name}/{name}_R{r}")
        else: # cheated MC
            for t in self.process_labels:
                new_sample = self.select_sample(self.exclusive_process[t])
                new_sample.to_root_file(f"{dir_name}/{name}_T{t}")
        return

    @staticmethod
    def from_file(file : str) -> "AnalysisInput": #* seems a bit extra but why not
        """ Load analysis input from dill file.

        Args:
            file (str): file path.

        Returns:
            AnalysisInput: analysis input.
        """
        obj = io.read_dill(file)
        if type(obj) == AnalysisInput:
            return obj
        else:
            raise Exception("not an analysis input file")


    def n_interact(self, energy_slice : slices.Slices, process: np.ndarray, mask : np.ndarray = None, reco : bool = True, weights : np.ndarray = None) -> np.ndarray:
        """ Calculate exclusive interaction histogram using the energy slice method.

        Args:
            energy_slice (slices.Slices): energy slices
            process (np.ndarray): exclusive process mask
            mask (np.ndarray, optional): additional mask. Defaults to None.
            reco (bool, optional): use reco KE?. Defaults to True.
            weights (np.ndarray, optional): event weights. Defaults to None.

        Returns:
            np.ndarray: exclusive interaction histogram.
        """
        common.deprecation_warning()
        if mask is None: mask = np.ones(len(self.KE_int_reco), dtype = bool)
        if reco is True:
            KE_int = self.KE_int_reco
            KE_init = self.KE_init_reco
            outside_fv = self.outside_fv_reco
        else:
            KE_int = self.KE_int_true
            KE_init = self.KE_init_true
            outside_fv = self.outside_fv_true
        n_interact = slices.energy_slice.counting_experiment_old(KE_int[mask], KE_init[mask], outside_fv[mask], process[mask], energy_slice, interact_only = True, weights = weights[mask] if weights is not None else weights)
        return n_interact


    def id_from_samples(self, sample_masks : dict[np.ndarray], uncategorised : int = -1) -> ak.Array:
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
    def create_analysis_input_toy(toy : Toy) -> "AnalysisInput":
        """ Create analysis input from a toy sample.

        Args:
            toy (Toy): toy sample

        Returns:
            AnalysisInput: analysis input object.
        """
        common.deprecation_warning() #! function wont be supported further.
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
    def create_analysis_input_ntuple(events : ntuple.Data, upstream_energy_loss_params : dict, reco_regions : dict[np.ndarray] = None, true_regions : dict[np.ndarray] = None, mc_reweight_params : dict = None, mc_reweight_stength : float = 3, fiducial_volume : list[float] = [0, 700], upstream_loss_func : callable = fitting.poly2d, energy_method : str = "track") -> "AnalysisInput":
        """ Create analysis input from an ntuple sample.

        Args:
            events (ntuple.Data): ntuple sample
            upstream_energy_loss_params (dict): upstream energy loss correction
            reco_regions (dict[np.ndarray]): reco region masks
            true_regions (dict[np.ndarray], optional): true process masks. Defaults to None.
            mc_reweight_params (dict, optional): mc reweight parameters. Defaults to None.

        Returns:
            AnalysisInput: analysis input.
        """
        if mc_reweight_params is not None:
            weights = beam.reweight.calculate_weights(events.recoParticles.beam_inst_momentum, "gaussian", mc_reweight_params, mc_reweight_stength)
        else:
            weights = None

        KE_inst_reco = kinematics.KE(events.recoParticles.beam_inst_momentum, Particle.from_pdgid(211).mass)
        upstream_loss_reco = beam.upstream.upstream_energy_loss(KE_inst_reco, upstream_energy_loss_params, upstream_loss_func)
        KE_ff_reco = KE_inst_reco - upstream_loss_reco

        track_length_reco = events.recoParticles.beam_track_length

        truncated_track_reco = kinematics.truncate_track(events.recoParticles.beam_calo_pos, z_trunc = max(fiducial_volume))
        track_length_truncated_reco = kinematics.track_length(tracks=truncated_track_reco)

        if min(fiducial_volume) > 0:
            track_length_min = np.where(events.recoParticles.beam_end_pos_sce.z > min(fiducial_volume), min(fiducial_volume), track_length_reco)

            KE_init_reco = bethe_bloch.KE_end(KE_ff_reco, track_length_min, 50) # initial kinetic energy in the fiducial volume

        else:
            KE_init_reco = KE_ff_reco

        KE_int_reco = kinematics.reco_end_energy(events.recoParticles.beam_calo_pos, KE_ff_reco, events.recoParticles.beam_dEdX, energy_method)

        KE_end_reco = kinematics.reco_end_energy(truncated_track_reco, KE_ff_reco, events.recoParticles.beam_dEdX, energy_method)

        outside_tpc_reco = geometry.ProtoDUNESPGeometry().outside_tpc(events.recoParticles.beam_end_pos_sce.x, events.recoParticles.beam_end_pos_sce.y, events.recoParticles.beam_end_pos_sce.z)

        outside_fv_reco = (events.recoParticles.beam_end_pos_sce.z < min(fiducial_volume)) | (events.recoParticles.beam_end_pos_sce.z > max(fiducial_volume))
        start_pos_reco = events.recoParticles.beam_start_pos_sce
        end_pos_reco = events.recoParticles.beam_end_pos_sce

        if true_regions is not None:
            KE_ff_true = events.true_particles.beam_KE_front_face
            KE_int_true = events.true_particles.beam_traj_KE[:, -2]
            track_length_true = events.true_particles.beam_track_length
            start_pos_true = events.true_particles.beam_traj_pos[:, 0]
            end_pos_true = events.true_particles.beam_traj_pos[:, -1]

            if min(fiducial_volume) > 0:
                track_length_min = np.where(events.true_particles.end_pos.z[:, 0] > min(fiducial_volume), min(fiducial_volume), track_length_true)

                KE_init_true = bethe_bloch.KE_end(KE_ff_true, track_length_min, 50) # initial kinetic energy in the fiducial volume
            else:
                KE_init_true = KE_ff_true

            outside_tpc_true = geometry.ProtoDUNESPGeometry().outside_tpc(events.true_particles.end_pos.x[:, 0], events.true_particles.end_pos.y[:, 0], events.true_particles.end_pos.z[:, 0])

            outside_fv_true = (events.true_particles.beam_traj_pos.z[:, -1] < min(fiducial_volume)) | (events.true_particles.beam_traj_pos.z[:, -1] > max(fiducial_volume))
            inelastic = events.true_particles.true_beam_end_process == "pi+Inelastic"

            truncated_tracks_true = kinematics.truncate_track(events.true_particles.beam_traj_pos[events.true_particles.in_tpc_z], z_trunc = max(fiducial_volume))
            track_length_truncated_true = kinematics.track_length(tracks=truncated_tracks_true)

            traj_KE = events.true_particles.beam_traj_KE[events.true_particles.in_tpc_z]
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
            P_inst_reco = events.recoParticles.beam_inst_momentum,
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
            event_num = events.event_num,
            run = events.run,
            sub_run = events.sub_run,
            )

    @staticmethod
    def req_fields():
        return [k for k, v in AnalysisInput.__dataclass_fields__.items() if v.init is True]

    @staticmethod
    def concatenate(ais : list["AnalysisInput"]):
        fields = processing.merge_outputs([{f : getattr(a, f) for f in AnalysisInput.req_fields()} for a in ais])

        # check for null entries after merging outputs (null entries are list of Nones)
        for k in fields:
            if (type(fields[k]) == list) and (all(ak.is_none(fields[k]))):
                fields[k] = None

        return AnalysisInput(**fields)


    def select_sample(self, mask : ak.Array) -> "AnalysisInput":
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


    def create_train_test_samples(self, seed : int, train_fraction : float = None) -> dict:
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
            "train" : self.select_sample(sample[:fraction]), "test" : self.select_sample(sample[fraction:])}


    def create_histograms(self, energy_slice : slices.Slices, exclusive_process : str, reco : bool, mask : np.ndarray = None) -> dict[np.ndarray]:
        """ Calculate Histogrames required for the cross section measurement using energy slicing. Note exclusive interaction histogram is without background subtraction.

        Args:
            energy_slice (slices.Slices): energy slices
            exclusive_process (str): exclusive process
            reco (bool): use reco information?
            mask (np.ndarray, optional): additional mask. Defaults to None.

        Returns:
            dict[np.ndarray]: histograms
        """
        common.deprecation_warning()
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

        n_initial, n_interact_inelastic, n_interact_exclusive, n_incident = slices.energy_slice.counting_experiment_old(KE_int, KE_init, outside_tpc, channel_mask, energy_slice, weights = self.weights)

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


@common.timer
def beam_pion_selection(events : ntuple.Data, args : argparse.Namespace | dict, is_mc : bool) -> ntuple.Data:
    """ Apply beam pion selection to ntuples.
        #! This needs to be somewhere else. It really is a helper to apply any beam selection from configuration to a sample (perhaps in ntuples or selection/common?)

    Args:
        events (ntuple.Data): analysis ntuple
        args (cross_section.argparse.Namespace): analysis configuration
        is_mc (bool): is the ntuple mc or data?

    Returns:
        ntuple.Data: selected events.
    """

    args_c = common.args_to_dict(args)

    events_copy = events.filter(return_copy = True)
    if is_mc:
        selection_args = "mc_arguments"
        sample = "mc"
    else:
        selection_args = "data_arguments"
        sample = "data"

    if "selection_masks" in args:
        masks = args_c["selection_masks"][sample]
        if ("fiducial" in masks) and (len(masks["fiducial"]) > 0):
            mask = selection.combine_masks(masks["fiducial"][events.filename])
            events_copy.filter([mask], [mask])
        mask = selection.combine_masks(masks["beam"][events.filename])
        events_copy.filter([mask], [mask])
    else:
        for s in args_c["beam_selection"]["selections"]:
            mask = args_c["beam_selection"]["selections"][s](events_copy, **args_c["beam_selection"][selection_args][s])
            events_copy.filter([mask], [mask])

    if "valid_pfo_selection" in args_c:
        if args_c["valid_pfo_selection"] is True:
            if "selection_masks" in args:
                events_copy.filter([args_c["selection_masks"][sample]['null_pfo'][events.filename]['ValidPFOSelection']]) # apply PFO preselection here
            else:
                events_copy.filter(selection.pfo.GoodShowerSelection(events))
    return events_copy


def create_analysis_input(sample : ntuple.Data, args : argparse.Namespace | dict, is_mc : bool) -> AnalysisInput:
    """ Create analysis input from ntuple sample.

    Args:
        sample (ntuple.Data): sample
        args (cross_section.argparse.Namespace): analysis configurations
        is_mc (bool): is the sample mc?

    Returns:
        cross_section.AnalysisInput: analysis input.
    """
    args_c = common.args_to_dict(args)

    if type(sample) == Toy:
        ai = AnalysisInput.create_analysis_input_toy(sample)
    elif type(sample) == ntuple.Data:
        sample_selected = beam_pion_selection(sample, args_c, is_mc)
        if is_mc:
            reco_regions, true_regions = regions.region_selection(sample, args_c, True, removed = True)
            reweight_params = [args_c["beam_reweight"]["params"][k]["value"] for k in args_c["beam_reweight"]["params"]]
        else:
            reco_regions = regions.region_selection(sample, args_c, False, removed = True)
            true_regions = None
            reweight_params = None
        ai = AnalysisInput.create_analysis_input_ntuple(sample_selected, args_c["upstream_loss_correction_params"]["value"], reco_regions, true_regions, reweight_params, args_c["beam_reweight"]["strength"], args_c["fiducial_volume"], args_c["upstream_loss_response"], args_c["energy_method"])
    else:
        raise Exception(f"object type {type(sample)} not a valid sample")
    return ai


def create_analysis_input_mc_true_beam(mc : ntuple.Data, args : argparse.Namespace | dict, uncategorised : bool = False):
    args_c = common.args_to_dict(args)

    masks = [mc.true_particles.pdg[:, 0] == 211]
    #! mc true beam does not encorperate fiducial cuts in truth, as this loss in efficiency needs to be corrected for the final cross section measurement
    #! if a particle interacted outside the fiducial region, it was still incident on slices within the fiducial region
    mc_true_beam = mc.filter(masks, masks, True)

    process_def = args_c["process_definitions"]
    true_regions = process_def.CreateDefinitions(process_def.criteria_list.get_criteria_values(mc_true_beam, **args["process_args"]), uncategorised = uncategorised)

    return AnalysisInput.create_analysis_input_ntuple(mc_true_beam, args_c["upstream_loss_correction_params"]["value"], None, true_regions, [args["beam_reweight"]["params"][k]["value"] for k in args_c["beam_reweight"]["params"]], args_c["beam_reweight"]["strength"], args_c["fiducial_volume"], upstream_loss_func = args_c["upstream_loss_response"], energy_method = args_c["energy_method"])
