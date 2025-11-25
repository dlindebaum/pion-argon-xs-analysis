#!/usr/bin/env python3
"""
Created on: 23/01/2024 10:34

Author: Shyam Bhuller

Description: 
"""
from rich import print
import argparse
import os

from apps import (
    cex_normalisation,
    cex_beam_quality_fits, 
    cex_beam_scraper_fits,
    cex_photon_selection,
    cex_beam_selection_studies,
    cex_region_selection_studies,
    # cex_selection_studies, 
    cex_gnn_predictions,
    cex_beam_reweight, 
    cex_upstream_loss,
    cex_toy_parameters,
    cex_analysis_input,
    cex_analyse,
    cex_gnn_analyse
    )

from python.analysis.cross_section import ApplicationArguments
from python.analysis.Processing import CalculateEventBatches, file_len
from python.analysis.Master import SaveConfiguration, LoadConfiguration


def template_config():
    template = {
        "NTUPLE_FILES":{
            "mc" : [
                {
                    "file": "ABSOLUTE file path",
                    "type": "PDSPAnalyser or shower_merging",
                    "pmom": "momentum byte of the beam, may need a value different to 1 if MC was not generated properly",
                    "train_sample": "Boolean, was this used to train the GNN, or may be used for MC statistics?",
                    "graph": "Path to folder containing the (beam selected) GNN graphs",
                    "graph_norm": "Path to JSON containing the graph feature normalisations."
                }
            ],
            "data" : [
                {
                    "file": "ABSOLUTE file path",
                    "type": "PDSPAnalyser or shower_merging",
                    "pmom": 1,
                    "graph": "Path to folder containing the (beam selected) GNN graphs",
                    "graph_norm": "Path to JSON containing the graph feature normalisations."
                }
            ]
        },
        "norm" : "normalisation to apply to MC when making Data/MC comparisons, usually defined as the ratio of pion-like triggers from the beam instrumentation", #! this should be inferred from one of the apps!
        "pi_KE_lim": -1,
        "fiducial_volume" : [0, 700],
        "REGION_IDENTIFICATION":{
            "type" : "gnn"
        },
        "SYSTEMATICS": {
            "track_length": None,
            "beam_momentum": None,
            "GNN_model": None,
            "upstream_energy": None,
            "purity": None
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
            "strength" : 2,
            "params": None
        },
        "UPSTREAM_ENERGY_LOSS":{
            "cv_function" : "gaussian",
            "response" : "poly2d",
            "bins" : None,
        },
        "ESLICE":{
            "edges": "List. If present, manually define bin edges, else use below.",
            "width" : None,
            "min" : None,
            "max" : None
        },
        "UNFOLDING":{
            "purity_bin": True,
            "method" : 1,
            "ts_stop" : 0.0001,
            "max_iter" : 6,
            "ts" : "ks",
            "covariance" : "poisson"
        },
        "BEAM_PARTICLE_SELECTION":{
            "PiBeamSelection":{
                "enable" : True,
                "use_beam_inst_mc" : False,
                "use_beam_inst_data" : True
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
        "beam_momentum" : "nominal beam momentum in MeV", #! should be deprciated
        "P_inst_range" : ["plot range low", "plot range high"],
        "KE_inst_range" : ["plot range low", "plot range high"],
        "KE_init_range" : ["plot range low", "plot range high"],
        "KE_int_range" : ["plot range low", "plot range high"],
        "GNN_MODEL": {
            "model_path": "Path to GNN model.",
            "region_labels": [
                "Abs.",
                "CEx.",
                "Pion"
            ]
        }
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


def update_args(processing_args : dict = {}):
    new_args = ApplicationArguments.ResolveArgs(original_args)
    for k, v in processing_args.items():
        setattr(new_args, k, v)
    return new_args


def check_run(args : argparse.Namespace, step : str, can_run : bool):
    run_req = step in args.run if len(args.run) > 0 else True
    return (run_req
            and (can_run or (args.force is True))
            and (step not in args.skip))

def check_run_dict(args, no_data):
    can_run = {
        "normalisation": (
            (not no_data)
            and ((args.norm is None)
                    or ("beam_norm" not in os.listdir(args.out)))),
        "beam_quality":
            (not hasattr(args, "mc_beam_quality_fit"))
            or ((not no_data)
                and (not hasattr(args, "data_beam_quality_fit"))),
        "beam_scraper": not hasattr(args, "mc_beam_scraper_fit"),
        "photon_correction": 
            (not args.sample_only)
            and hasattr(args, "shower_correction")
            and (not args.gnn_do_predict)
            and (args.shower_correction["correction_params"] is None),
        "beam_selection": not hasattr(args, "beam_selection_masks"),
        "region_selection":
            (not args.sample_only)
            and (not args.gnn_do_predict)
            and (not hasattr(args, "region_selection_masks")),
        # "selection": not hasattr(args, "selection_masks"),
        "gnn_prediction":
            (not args.sample_only)
            and args.gnn_do_predict
            and (not hasattr(args, "gnn_results")),
        "reweight": 
            ("params" not in args.beam_reweight) and (not no_data),
        "upstream_correction":
            not hasattr(args, "upstream_loss_correction_params"),
        "toy_parameters":
            hasattr(args, "toy_parameters")
            and hasattr(args, "beam_reweight")
            and ("toy_parameters" not in os.listdir(args.out)),
        "analysis_input":
            (not args.sample_only)
            and (not hasattr(args, "analysis_input"))
            and (not no_data),
        "analyse": not args.sample_only}
    return {k: check_run(args, k, v) for k, v in can_run.items()}

def main(args):
    os.makedirs(args.out, exist_ok = True)
    if args.create_config:
        SaveConfiguration(template_config(), os.path.join(args.out, args.create_config))
        print(f"template configuration saved as {args.out + args.create_config}")
        exit()
    else:
        print("run analysis, checking what steps have already been run")

        if "data" in args.ntuple_files:
            n_data = [file_len(file["file"]) for file in args.ntuple_files["data"]]
        else:
            n_data = []
        no_data = len(n_data) == 0
        if no_data:
            print("no data file was specified, 'normalisation', 'beam_reweight', 'toy_parameters' and 'analyse' will not run")
        if args.sample_only:
            print("Set to 'sample_only', therefore 'region_selection', 'gnn_prediction', 'analysis_input', and 'analyse' will not run")

        processing_args = CalculateEventBatches(args)
        args = update_args(processing_args)

        do_runs = check_run_dict(args, no_data)
        # Not yet reached analysis stage yet
        do_runs["analysis_input"] = False
        do_runs["analyse"] = False
        stop_text = (f" (stopping after running {args.stop}):"
                     if args.stop is not None else ":")
        print("Apps to run" + stop_text)
        print(do_runs)

        #* normalisation 
        # can_run_norm = (not no_data) and ((args.norm is None) or ("beam_norm" not in os.listdir(args.out)))
        if do_runs["normalisation"]:
            print("calculate beam normalisation")
            cex_normalisation.main(args)
            output_path = args.out + "beam_norm/"
            print("outputs: " + output_path)
            norm = LoadConfiguration(os.path.abspath(output_path + "norm.json"))
            update_config(args.config, {"norm" : norm["norm"]})
            args = update_args(processing_args) # reload config to continue
        if args.stop == "normalisation": return

        #* beam quality
        # can_run_bq = (not hasattr(args, "mc_beam_quality_fit")) or ((len(n_data) > 0) and (not hasattr(args, "data_beam_quality_fit")))
        if do_runs["beam_quality"]:
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
            args = update_args(processing_args) # reload config to continue
        if args.stop == "beam_quality": return

        #* beam scraper
        # can_run_bs = not hasattr(args, "mc_beam_scraper_fit")
        if do_runs["beam_scraper"]:
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
            args = update_args(processing_args) # reload config to continue
        if args.stop == "beam_scraper": return

        # * photon energy correction
        # Separate to prevent force
        if do_runs["photon_correction"] and (not args.gnn_do_predict):
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
            args = update_args(processing_args) # reload config to continue
        if args.stop == "photon_correction": return

        #* beam selection studies
        if do_runs["beam_selection"]:
            print("run beam selection")
            args.mc_only = len(n_data) == 0
            args.nbins = 50
            cex_beam_selection_studies.main(args)

            output_path = args.out
            print("outputs: " + output_path)
            target_files = {
                "mc" : "masks_mc",
                "data" : "masks_data"
            }
            mask_map = {
                "beam" : 'beam_selection_masks.dill',
                "fiducial" : "fiducial_selection_masks.dill",
                "null_pfo" : 'null_pfo_selection_masks.dill'
            }
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = {i : os.path.abspath(output_path + v + "/" + j) for i, j in mask_map.items() if os.path.isfile(output_path + v + "/" + j)}
            update_config(args.config, {"BEAM_SELECTION_MASKS" : new_config_entry})
            mc_eff_map = {
                "reco" : "reco_eff_from_selection.dill",
                "truth" : "truth_eff_from_selection.dill",
                "process" : "process_eff_from_selection.dill"
            }
            mc_eff_paths = {name : os.path.abspath(output_path + "efficiency_mc/" + path)
                            for name, path in mc_eff_map.items()
                            if os.path.isfile(output_path + "efficiency_mc/" + path)}
            update_config(args.config, {"MC_EFFICIENCIES" : mc_eff_paths})
            args = update_args(processing_args) # reload config to continue
        if args.stop == "beam_selection": return

        if args.gnn_do_predict:
            #* GNN predictions, separate to prevent force changing it
            if do_runs["gnn_prediction"]:
                print("run GNN prediction")
                args.mc_only = len(n_data) == 0
                args.nbins = 50
                cex_gnn_predictions.main(args)

                output_path = args.out
                print("outputs: " + output_path)
                target_files = {
                    "mc" : "predictions_mc",
                    "data" : "predictions_data"
                }
                mask_map = {
                    "predictions" : 'gnn_predictions.dill',
                    "ids" : 'gnn_ids.dill',
                    "truth_regions" : 'gnn_truth_regions.dill'
                }
                new_config_entry = {}
                files = os.listdir(output_path)
                for k, v in target_files.items():
                    if v in files:
                        new_config_entry[k] = {i : os.path.abspath(output_path + v + "/" + j) for i, j in mask_map.items() if os.path.isfile(output_path + v + "/" + j)}
                update_config(args.config, {"GNN_PREDICTIONS" : new_config_entry})
                args = update_args(processing_args) # reload config to continue
        else:
            #* region selection studies
            if do_runs["region_selection"]:
                print("run region selection")
                args.mc_only = len(n_data) == 0
                args.nbins = 50
                cex_region_selection_studies.main(args)

                output_path = args.out
                print("outputs: " + output_path)
                target_files = {
                "mc" : "masks_mc",
                "data" : "masks_data"
                }
                mask_map = {
                    "photon" : 'photon_selection_masks.dill',
                    "pi0" : 'pi0_selection_masks.dill',
                    "pi" : 'pi_selection_masks.dill',
                    "loose_pi"  : "loose_pi_selection_masks.dill",
                    "loose_photon" : "loose_photon_selection_masks.dill"
                }
                new_config_entry = {}
                files = os.listdir(output_path)
                for k, v in target_files.items():
                    if v in files:
                        new_config_entry[k] = {i : os.path.abspath(output_path + v + "/" + j) for i, j in mask_map.items() if os.path.isfile(output_path + v + "/" + j)}
                update_config(args.config, {"REGION_SELECTION_MASKS" : new_config_entry})
                args = update_args(processing_args) # reload config to continue
        if args.stop == "gnn_prediction": return
        if args.stop == "region_selection": return

        # #* selection studies
        # # can_run_ss = not hasattr(args, "selection_masks")
        # if do_runs["selection"]:
        #     print("run selection")
        #     args.mc_only = len(n_data) == 0
        #     args.nbins = 50
        #     cex_selection_studies.main(args)

        #     output_path = args.out
        #     print("outputs: " + output_path)
        #     target_files = {
        #     "mc" : "masks_mc",
        #     "data" : "masks_data"
        #     }
        #     mask_map = {
        #         "beam" : 'beam_selection_masks.dill',
        #         "null_pfo" : 'null_pfo_selection_masks.dill',
        #         "photon" : 'photon_selection_masks.dill',
        #         "pi0" : 'pi0_selection_masks.dill',
        #         "pi" : 'pi_selection_masks.dill',
        #         "loose_pi"  : "loose_pi_selection_masks.dill",
        #         "loose_photon" : "loose_photon_selection_masks.dill",
        #         "fiducial" : "fiducial_selection_masks.dill"
        #     }
        #     new_config_entry = {}
        #     files = os.listdir(output_path)
        #     for k, v in target_files.items():
        #         if v in files:
        #             new_config_entry[k] = {i : os.path.abspath(output_path + v + "/" + j) for i, j in mask_map.items() if os.path.isfile(output_path + v + "/" + j)}
        #     update_config(args.config, {"SELECTION_MASKS" : new_config_entry})
        #     args = update_args(processing_args) # reload config to continue
        # if args.stop == "selection": return

        #* beam reweight
        # can_run_rw = ("params" not in args.beam_reweight) and (not no_data)
        if do_runs["reweight"]:
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
        # can_run_uc = not hasattr(args, "upstream_loss_correction_params")
        if do_runs["upstream_correction"]:
            print("run upstream correction")
            args.no_reweight = (not hasattr(args, "beam_reweight")) or ("params" not in args.beam_reweight) 
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
        # can_run_tp = hasattr(args, "toy_parameters") and hasattr(args, "beam_reweight") and ("toy_parameters" not in os.listdir(args.out))
        if do_runs["toy_parameters"]:
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
        # can_run_ai = (not hasattr(args, "analysis_input")) and (len(n_data) > 0)
        if do_runs["analysis_input"]:
            print("run analysis input")
            cex_analysis_input.main(args)

            output_path = args.out + "analysis_input/"
            print("outputs: " + output_path)
            target_files = {
                "mc_cheated" : "analysis_input_mc_cheated.dill",
                "mc" : "analysis_input_mc_selected.dill",
                "data" : "analysis_input_data_selected.dill",
                "mc_with_train" : "analysis_input_mc_with_train.dill"}
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = os.path.abspath(output_path + v)
            update_config(args.config, {"ANALYSIS_INPUTS" : new_config_entry})
            args = update_args() # reload config to continue
        if args.stop == "analysis_input": return

        # if all other prerequisites were met, this should run
        if do_runs["analyse"]:
            print("analyse")
            args.toy_template = None
            args.all = False
            args.pdsp = True # run with PDSP samples (no toys yet)
            if args.gnn_do_predict:
                cex_gnn_analyse.main(args)
            else:
                cex_analyse.main(args)
        if args.stop == "analyse": return

    return


if __name__ == "__main__":

    analysis_options = [
        "normalisation",
        "beam_quality",
        "beam_scraper",
        "photon_correction",
        "beam_selection",
        "region_selection",
        "gnn_prediction",
        "reweight",
        "upstream_correction",
        "toy_parameters",
        "analysis_input",
        "analyse"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--create_config", type = str, help = "Create a template configuration with the default selection")
    ApplicationArguments.Config(parser)
    ApplicationArguments.Output(parser, "analysis/")
    ApplicationArguments.Regen(parser)
    parser.add_argument("--skip", type = str, nargs = "+", default = [], choices = analysis_options)
    parser.add_argument("--run", type = str, nargs = "+", default = [], choices = analysis_options)
    parser.add_argument("--force", action = "store_true")
    parser.add_argument("--stop", type = str, default = None, choices = analysis_options)
    parser.add_argument("--cpus", type = int, default = 1)

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