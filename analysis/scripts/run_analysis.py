#!/usr/bin/env python3
"""
Created on: 23/01/2024 10:34

Author: Shyam Bhuller

Description: 
"""
from rich import print

from apps import (
    cex_normalisation,
    cex_beam_quality_fits, 
    cex_beam_scraper_fits,
    cex_photon_selection,
    cex_selection_studies, 
    cex_beam_reweight, 
    cex_upstream_loss,
    cex_toy_parameters,
    cex_analysis_input,
    cex_analyse
    )

from python.analysis.cross_section import ApplicationArguments, argparse, os
from python.analysis.Master import SaveConfiguration, LoadConfiguration, IO


def template_config():
    template = {
        "NTUPLE_FILE":{
            "mc" : "MC ntuple file ENSURE ALL FILE PATHS ARE ABSOLUTE",
            "data" : "Data ntuple file",
            "type" : "type of ntuple files, this is either PDSPAnalyser or shower_merging"
        },
        "norm" : "normalisation to apply to MC when making Data/MC comparisons, usually defined as the ratio of pion-like triggers from the beam instrumentation", #! this should be inferred from one of the apps!
        "pmom" : "momentum byte of the beam i.e. central value of beam momentum in GeV, required if ntuple does not have the correct scale for the P_inst distribution",
        "fiducial_volume" : [0, 700],
        "REGION_IDENTIFICATION":{
            "type" : "default"
        },
        "BEAM_QUALITY_FITS": {
            "truncate" : None,
        },
        "BEAM_SCRAPER_FITS":{
            "energy_range" : None,
            "energy_bins" : None
        },
        "ENERGY_CORRECTION":{
            "correction_params" : None,
            "energy_range" : None,
            "correction" : "response"
        },
        "BEAM_REWEIGHT": {
            "strength" : 3,
            "params": None
        },
        "UPSTREAM_ENERGY_LOSS":{
            "cv_function" : "gaussian",
            "response" : "poly2d",
            "bins" : None,
        },
        "TOY_PARAMETERS":{
            "beam_profile" : "crystal_ball",
            "smearing_residual_ranges" : {
                "KE_init" : None,
                "KE_int" : None,
                "z_int" : None
            },
            "plot_ranges": {
                "KE_init" : None,
                "KE_int" : None,
                "z_int" : None
            }
        },
        "FIT":{
            "mc_stat_unc" : True,
            "mean_track_score" : None,
            "single_bin" : True,
            "regions": True
        },
        "bkg_sub_err" : False,
        "ESLICE":{
            "width" : None,
            "min" : None,
            "max" : None
        },
        "UNFOLDING":{
            "method" : 1,
            "ts_stop" : 0.0001,
            "max_iter" : 6,
            "ts" : "ks",
            "covariance" : "poisson"
        },
        "signal_process" : "charge_exchange",
        "BEAM_PARTICLE_SELECTION":{
            "PiBeamSelection":{
                "enable" : True,
                "use_beam_inst" : False
            },
            "PandoraTagCut":{
                "enable" : True,
                "cut" : 13,
                "op" : "=="
            },
            "CaloSizeCut":{
                "enable" : True
            },
            "HasFinalStatePFOsCut":{
                "enable" : True
            },
            "DxyCut":{
                "enable" : True,
                "cut" : 3,
                "op" : "<"
            },
            "DzCut":{
                "enable" : True,
                "cut" : [-3, 3],
                "op" : [">", "<"]
            },
            "CosThetaCut":{
                "enable" : True,
                "cut" : 0.95,
                "op" : ">"
            },
            "APA3Cut":{
                "enable" : True,
                "cut" : 220,
                "op" : "<"
            },
            "MichelScoreCut":{
                "enable" : True,
                "cut" : 0.55,
                "op" : "<"
            },
            "MedianDEdXCut":{
                "enable" : True,
                "cut" : 2.4,
                "op" : "<",
                "truncate": None
            },
            "BeamScraperCut":{
                "enable" : True,
                "KE_range" : 1,
                "cut" : 1.5,
                "op" : "<"
            }
        },
        "VALID_PFO_SELECTION":{
            "enable" : True
        },
        "FINAL_STATE_PIPLUS_SELECTION": {
            "Chi2ProtonSelection": {
            "enable": True,
            "cut": 61.2,
            "op": ">"
            },
            "TrackScoreCut": {
            "enable": True,
            "cut": 0.5,
            "op": ">"
            },
            "NHitsCut": {
            "enable": True,
            "cut": 20,
            "op": ">"
            },
            "PiPlusSelection": {
            "enable": True,
            "cut": [
                0.5,
                2.8
            ],
            "op": [
                ">",
                "<"
            ]
            }
        },
        "FINAL_STATE_PHOTON_SELECTION": {
            "Chi2ProtonSelection": {
            "enable": True,
            "cut": 61.2,
            "op": ">"
            },
            "TrackScoreCut": {
            "enable": True,
            "cut": 0.45,
            "op": "<"
            },
            "NHitsCut": {
            "enable": True,
            "cut": 80,
            "op": ">"
            },
            "BeamParticleDistanceCut": {
            "enable": True,
            "cut": [
                3,
                90
            ],
            "op": [
                ">",
                "<"
            ]
            },
            "BeamParticleIPCut": {
            "enable": True,
            "cut": 20,
            "op": "<"
            }
        },
        "FINAL_STATE_PI0_SELECTION": {
            "NPhotonCandidateSelection": {
            "enable": True,
            "cut": 2,
            "op": "=="
            },
            "Pi0MassSelection": {
            "enable": True,
            "cut": [
                50,
                250
            ],
            "op": [
                ">",
                "<"
            ]
            },
            "Pi0OpeningAngleSelection": {
            "enable": True,
            "cut": [
                10,
                80
            ],
            "op": [
                ">",
                "<"
            ]
            }
        },
        "FINAL_STATE_LOOSE_PHOTON_SELECTION": {
            "Chi2ProtonSelection": {
            "enable": True,
            "cut": 61.2,
            "op": ">"
            },
            "TrackScoreCut": {
            "enable": True,
            "cut": 0.45,
            "op": "<"
            },
            "NHitsCut": {
            "enable": True,
            "cut": 31,
            "op": ">"
            },
            "BeamParticleDistanceCut": {
            "enable": True,
            "cut": 114,
            "op": "<"
            },
            "BeamParticleIPCut": {
            "enable": True,
            "cut": 80,
            "op": "<"
            }
        },
        "FINAL_STATE_LOOSE_PION_SELECTION": {
            "Chi2ProtonSelection": {
            "enable": True,
            "cut": 61.2,
            "op": ">"
            },
            "TrackScoreCut": {
            "enable": True,
            "cut": 0.39,
            "op": ">"
            },
            "PiPlusSelection": {
            "enable": True,
            "cut": 6.3,
            "op": "<"
            }
        },
        "beam_momentum" : "nominal beam momentum in MeV", #! should be deprciated
        "P_inst_range" : "plot range",
        "KE_inst_range" : "plot range",
        "KE_init_range" : "plot range",
        "KE_int_range" : "plot range"
    }
    return template


def template_toy_config(toy_parameters_dir : str, nEvents : int, seed : int, max_cpus : int, step : float, p_init : float, region_selection : str):
    template = {
        "events" : nEvents,
        "step" : step,
        "p_init" : p_init,
        "beam_profile" : f"{toy_parameters_dir}/beam_profile/beam_profile.json",
        "beam_width" : 60,

        "smearing_params" : {
            "KE_init" : f"{toy_parameters_dir}/smearing/KE_init/double_crystal_ball.json",
            "KE_int" : f"{toy_parameters_dir}/smearing/KE_int/double_crystal_ball.json",
            "z_int" : f"{toy_parameters_dir}/smearing/z_int/double_crystal_ball.json"
        },
        "reco_region_fractions" : f"{toy_parameters_dir}/reco_regions/{region_selection}_reco_region_fractions.hdf5",
        "beam_selection_efficiencies" : f"{toy_parameters_dir}/pi_beam_efficiency/beam_selection_efficiencies_true.hdf5",
        "mean_track_score_kde" : f"{toy_parameters_dir}/meanTrackScoreKDE/kdes.dill",
        "pdf_scale_factors" : None,
        "df_format" : "f",
        "modified_PDFs" : None,
        "verbose" : True,
        "seed" : seed,
        "max_cpus" : max_cpus
    }
    return template


def update_config(config, update : dict):
    json_config = LoadConfiguration(config)
    json_config.update(update)
    SaveConfiguration(json_config, config)
    print(f"{config} has been updated")
    return


def update_args():
    return ApplicationArguments.ResolveArgs(original_args)


def file_len(file : str):
    return len(IO(file).Get(["EventID", "event"]))


def check_run(args : argparse.Namespace, step : str):
    return ((step in args.run) or (args.force is True)) and (step not in args.skip)


def main(args):
    os.makedirs(args.out, exist_ok = True)
    if args.create_config:
        SaveConfiguration(template_config(), args.out + args.create_config)
        print(f"template configuration saved as {args.out + args.create_config}")
        exit()
    else:
        print("run analysis, checking what steps have already been run")

        # keep these to figure out if batch processing is required
        if args.data_file is None:
            print("no data file was specified, 'beam_reweight', 'toy_parameters' and 'analyse' will not run")
            n_data = 0
        else:
            n_data = file_len(args.data_file)
        n_mc = file_len(args.mc_file)

        #* normalisation 
        can_run_norm = (args.norm is None) or ("beam_norm" not in os.listdir(args.out))
        if can_run_norm or check_run(args, "normalisation"):
            print("calculate beam normalisation")
            cex_normalisation.main(args)
            output_path = args.out + "beam_norm/"
            print("outputs: " + output_path)
            norm = LoadConfiguration(os.path.abspath(output_path + "norm.json"))
            update_config(args.config, {"norm" : norm["norm"]})
            args = update_args() # reload config to continue
        if args.stop == "normalisation": return

        #* beam quality
        can_run_bq = (not hasattr(args, "mc_beam_quality_fit")) or ((args.data_file is not None) and (not hasattr(args, "data_beam_quality_fit")))
        if can_run_bq or check_run(args, "beam_quality"):
            print("run beam quality fit")
            cex_beam_quality_fits.main(args)
            output_path = args.out + "beam_quality/"
            print("outputs: " + output_path)
            target_files = {
            "mc" : "mc_beam_quality_fit_values.json",
            "data" : "data_beam_quality_fit_values.json"
            }
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = os.path.abspath(output_path + v)
            new_config_entry["truncate"] = args.beam_quality_truncate
            update_config(args.config, {"BEAM_QUALITY_FITS" : new_config_entry})
            args = update_args() # reload config to continue
        if args.stop == "beam_quality": return

        #* beam scraper
        can_run_bs = not hasattr(args, "mc_beam_scraper_fit")
        if can_run_bs or check_run(args, "beam_scraper"):
            print("run beam scraper fit")
            cex_beam_scraper_fits.main(args)
            output_path = args.out + "beam_scraper/"
            print("outputs: " + output_path)
            target_files = {
            "mc" : "mc_beam_scraper_fit_values.json",
            }
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = os.path.abspath(output_path + v)
            new_config_entry["energy_range"] = args.beam_scraper_energy_range
            new_config_entry["energy_bins"] = args.beam_scraper_energy_bins
            update_config(args.config, {"BEAM_SCRAPER_FITS" : new_config_entry})
            args = update_args() # reload config to continue
        if args.stop == "beam_scraper": return

        #* photon energy correction
        can_run_pec = hasattr(args, "shower_correction") and (args.shower_correction["correction_params"] is None)
        if can_run_pec or check_run(args, "photon_correction"):
            print("run shower correction")
            args.events = None
            args.batches = None
            args.threads = 1
            cex_photon_selection.main(args)
            output_path = args.out + "shower_energy_correction/"
            print("outputs: " + output_path)
            target_files = {
            "correction_params" : "gaussian.json"
            }
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = os.path.abspath(output_path + v)
            new_config_entry["energy_range"] = args.shower_correction["energy_range"]
            new_config_entry["correction"] = "response"
            update_config(args.config, {"ENERGY_CORRECTION" : new_config_entry})
            args = update_args() # reload config to continue
        if args.stop == "photon_correction": return

        #* selection studies
        can_run_ss = not hasattr(args, "selection_masks")
        if can_run_ss or check_run(args, "selection"):
            print("run selection")
            args.mc_only = args.data_file is None
            args.nbins = 50
            # pass multiprocessing args
            if (n_data >= 7E5) or (n_mc >= 7E5):
                args.events = None
                args.batches = int(2 * max(n_data, n_mc) // 7E5)
                args.threads = 1
            else:
                args.events = None
                args.batches = None
                args.threads = 1
            cex_selection_studies.main(args)

            output_path = args.out
            print("outputs: " + output_path)
            target_files = {
            "mc" : "masks_mc",
            "data" : "masks_data"
            }
            mask_map = {
                "beam" : 'beam_selection_masks.dill',
                "null_pfo" : 'null_pfo_selection_masks.dill',
                "photon" : 'photon_selection_masks.dill',
                "pi0" : 'pi0_selection_masks.dill',
                "pi" : 'pi_selection_masks.dill',
                "loose_pi"  : "loose_pi_selection_masks.dill",
                "loose_photon" : "loose_photon_selection_masks.dill",
                "fiducial" : "fiducial_selection_masks.dill"
            }
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = {i : os.path.abspath(output_path + v + "/" + j) for i, j in mask_map.items() if os.path.isfile(output_path + v + "/" + j)}
            update_config(args.config, {"SELECTION_MASKS" : new_config_entry})
            args = update_args() # reload config to continue
        if args.stop == "selection": return

        #* beam reweight
        can_run_rw = ("params" not in args.beam_reweight) and (args.data_file is not None)
        if can_run_rw or check_run(args, "reweight"):
            print("run beam reweight")
            cex_beam_reweight.main(args)
            output_path = args.out + "beam_reweight/"
            print("outputs: " + output_path)
            target_files = {
            "params" : "gaussian.json", # default choice, rework reweight to include a choice in the config
            }
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = os.path.abspath(output_path + v)
            new_config_entry["strength"] = args.beam_reweight["strength"]
            update_config(args.config, {"BEAM_REWEIGHT" : new_config_entry})
            args = update_args() # reload config to continue
        if args.stop == "reweight": return

        #* upstream correction
        can_run_uc = not hasattr(args, "upstream_loss_correction_params")
        if can_run_uc or check_run(args, "upstream_correction"):
            print("run upstream correction")
            args.no_reweight = (not hasattr(args, "beam_reweight")) or ("params" in args.beam_reweight) 
            cex_upstream_loss.main(args)

            output_path = args.out + "upstream_loss/"
            print("outputs: " + output_path)
            target_files = {
            "correction_params" : "fit_parameters.json",
            }
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = os.path.abspath(output_path + v)
            new_config_entry["cv_function"] = args.upstream_loss_cv_function
            new_config_entry["response"] = args.upstream_loss_response.__name__
            new_config_entry["bins"] = args.upstream_loss_bins
            update_config(args.config, {"UPSTREAM_ENERGY_LOSS" : new_config_entry})
            args = update_args() # reload config to continue
        if args.stop == "upstream_correction": return

        #* toy parameters
        can_run_tp = hasattr(args, "toy_parameters") and hasattr(args, "beam_reweight") and ("toy_parameters" not in os.listdir(args.out))
        if can_run_tp or check_run(args, "toy_parameters"):
            print("run toy parameters")
            cex_toy_parameters.main(args)
            # special case where the main config is not updated, rather the results from this would be used in the toy configurations
            selection_type = LoadConfiguration(args.config)["REGION_IDENTIFICATION"]["type"]
            toy_template_config = template_toy_config(os.path.abspath(args.out + "toy_parameters"), int(1E7), 1337, os.cpu_count() - 1, 2, args.beam_momentum, selection_type)
            data_config = template_toy_config(os.path.abspath(args.out + "toy_parameters"), int(1E6), 1, os.cpu_count() - 1, 2, args.beam_momentum, selection_type)
            SaveConfiguration(toy_template_config, args.out + "toy_template_config.json")
            SaveConfiguration(data_config, args.out + "toy_data_config.json")
        if args.stop == "toy_parameters": return

        #* analysis input
        can_run_ai = (not hasattr(args, "analysis_input")) and (args.data_file is not None)
        if can_run_ai or check_run(args, "analysis_input"):
            print("run analysis input")
            cex_analysis_input.main(args)

            output_path = args.out + "analysis_input/"
            print("outputs: " + output_path)
            target_files = {
            "mc_cheated" : "analysis_input_mc_cheated.dill",
            "mc" : "analysis_input_mc_selected.dill",
            "data" : "analysis_input_data_selected.dill"
            }
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = os.path.abspath(output_path + v)
            update_config(args.config, {"ANALYSIS_INPUTS" : new_config_entry})
            args = update_args() # reload config to continue
        if args.stop == "analysis_input": return

        # if all other prerequisites were met, this should run
        if check_run(args, "analyse"):
            print("analyse")
            args.toy_template = None
            args.all = False
            args.pdsp = True # run with PDSP samples (no toys yet)
            cex_analyse.main(args)
        if args.stop == "analyse": return

    return


if __name__ == "__main__":

    analysis_options = ["normalisation", "beam_quality", "beam_scraper", "selection", "photon_correction", "reweight", "upstream_correction", "toy_parameters", "analysis_input", "analyse"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--create_config", type = str, help = "Create a template configuration with the default selection")
    ApplicationArguments.Config(parser)
    ApplicationArguments.Output(parser, "analysis/")
    parser.add_argument("--skip", type = str, nargs = "+", default = [], choices = analysis_options)
    parser.add_argument("--run", type = str, nargs = "+", default = [], choices = analysis_options)
    parser.add_argument("--force", action = "store_true")
    parser.add_argument("--stop", type = str, default = None, choices = analysis_options)

    original_args = parser.parse_args()
    
    if (original_args.create_config is None) and (original_args.config is None):
        raise Exception("either supply a configuration file with -c or request a template configuration with -C")
    elif (original_args.create_config is not None) and (original_args.config is not None):
        raise Exception("both -c and -C can't be used")
    elif (original_args.create_config is not None) and (original_args.config is None):
        args = original_args
    else:
        args = update_args()

    print(vars(args))
    main(args)