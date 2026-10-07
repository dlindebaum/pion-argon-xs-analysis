#!/usr/bin/env python3
"""
Created on: 23/01/2024 10:34

Author: Shyam Bhuller

Description: 
"""
import os

from rich import print

from apps import (
    cex_normalisation,
    cex_beam_quality_fits,
    cex_beam_scraper_fits,
    cex_photon_selection,
    cex_selection_studies,
    cex_beam_reweight,
    cex_upstream_loss,
    cex_regions,
    cex_analysis_input,
    cex_mach3_input,
    cex_cross_section
    )

from python.analysis.NtupleProcessing import CalculateBatches, file_len
from python.analysis.Application import ApplicationArguments, argparse, template_config
from python.analysis.Master import SaveConfiguration, LoadConfiguration


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


def check_run(args : argparse.Namespace, step : str):
    return ((step in args.run) or (args.force is True)) and (step not in args.skip)


def step_plan(args : argparse.Namespace, no_data : bool, n_data : list):
    existing = os.listdir(args.out) if os.path.isdir(args.out) else []
    steps = [
        ("normalisation", (not no_data) and ((args.norm is None) or ("beam_norm" not in existing))),
        ("beam_quality", (not hasattr(args, "mc_beam_quality_fit")) or ((len(n_data) > 0) and (not hasattr(args, "data_beam_quality_fit")))),
        ("beam_scraper", not hasattr(args, "mc_beam_scraper_fit")),
        ("photon_correction", hasattr(args, "shower_correction") and (args.shower_correction["correction_params"] is None)),
        ("selection", not hasattr(args, "selection_masks")),
        ("reweight", ("params" not in args.beam_reweight) and (not no_data)),
        ("upstream_correction", not hasattr(args, "upstream_loss_correction_params")),
        ("region_plots", hasattr(args, "region_plots") and ("region_plots" not in existing)),
        ("analysis_input", (not hasattr(args, "analysis_input")) and (len(n_data) > 0)),
        ("mach3_input", ("mach3_input" not in existing) and (len(n_data) > 0)),
        ("cross_section", ("xs_measurement" not in existing)),
    ]

    planned = []
    for step_name, should_run in steps:
        if should_run or check_run(args, step_name):
            planned.append(step_name)

    if args.stop is not None:
        planned = [
            step_name for step_name in planned
            if analysis_options.index(step_name) <= analysis_options.index(args.stop)
        ]

    return planned


def main(args):
    os.makedirs(args.out, exist_ok = True)
    if args.create_config:
        SaveConfiguration(template_config, os.path.join(args.out, args.create_config))
        print(f"template configuration saved as {args.out + args.create_config}")
        exit()
    else:
        print("Checking what steps have already been run")

        if "data" in args.ntuple_files:
            n_data = [file_len(file.file) for file in args.ntuple_files["data"]]
        else:
            n_data = []
        no_data = len(n_data) == 0
        if no_data:
            print("no data file was specified, 'normalisation', 'beam_reweight', 'toy_parameters' and 'analyse' will not run")

        processing_args = CalculateBatches(args)
        args = update_args(processing_args)

        planned_steps = step_plan(args, no_data, n_data)
        print("steps to be run before execution:")
        if not planned_steps:
            print(f"  - None")
            print("All required analysis steps have been run, specify which step you would like to run with the --run option, likewise --force to run them all and --stop to indicate at which step to stop at.")
        else:
            for step_name in planned_steps:
                print(f"  - {step_name}")

        if args.debug:
            print("Running in debug mode, will not execute any analysis steps")
            return

        #* normalisation 
        if "normalisation" in planned_steps:
            print("calculate beam normalisation")
            cex_normalisation.main(args)
            output_path = args.out + "beam_norm/"
            print("outputs: " + output_path)
            norm = LoadConfiguration(os.path.abspath(output_path + "norm.json"))
            update_config(args.config, {"norm" : norm["norm"]})
            args = update_args(processing_args) # reload config to continue
        if args.stop == "normalisation": return

        #* beam quality
        if "beam_quality" in planned_steps:
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
        if "beam_scraper" in planned_steps:
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

        #* photon energy correction
        if "photon_correction" in planned_steps:
            print("run shower correction")
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

        #* selection studies
        if "selection" in planned_steps:
            print("run selection")
            args.mc_only = len(n_data) == 0
            args.nbins = 50
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
            args = update_args(processing_args) # reload config to continue
        if args.stop == "selection": return

        #* beam reweight
        if "reweight" in planned_steps:
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
            args = update_args(processing_args) # reload config to continue
        if args.stop == "reweight": return

        #* upstream correction
        if "upstream_correction" in planned_steps:
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
            args = update_args(processing_args) # reload config to continue
        if args.stop == "upstream_correction": return

        #* region plots
        if "region_plots" in planned_steps:
            print("run region plots")
            cex_regions.main(args)
            # special case where the main config is not updated, rather the results from this would be used in the toy configurations
        if args.stop == "region_plots": return

        #* analysis input
        if "analysis_input" in planned_steps:
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
            args = update_args(processing_args) # reload config to continue
        if args.stop == "analysis_input": return

        #* mach3_input
        if "mach3_input" in planned_steps:
            print("run mach3_input")
            cex_mach3_input.main(args)
        if args.stop == "mach3_input": return

        #* cross section extraction
        if "cross_section" in planned_steps:
            print("run cross section extraction")
            cex_cross_section.main(args)
        if args.stop == "cross_section": return

    return


if __name__ == "__main__":

    analysis_options = ["normalisation", "beam_quality", "beam_scraper", "photon_correction", "selection", "reweight", "upstream_correction", "region_plots", "analysis_input", "mach3_input", "cross_section"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--create_config", type = str, help = "Create a template configuration with the default selection")
    ApplicationArguments.Config(parser)
    ApplicationArguments.Output(parser, "analysis/")
    ApplicationArguments.Regen(parser)
    parser.add_argument("--skip", type = str, nargs = "+", default = [], choices = analysis_options)
    parser.add_argument("--run", type = str, nargs = "+", default = [], choices = analysis_options)
    parser.add_argument("--force", action = "store_true")
    parser.add_argument("--stop", type = str, default = None, choices = analysis_options)
    parser.add_argument("--debug", action = "store_true", help = "Print the list of apps to execute without running them.")
    parser.add_argument("--cpus", type = int, default = 1)
    parser.add_argument("-R", "--ROOT", dest = "root", action="store_true", help = "Saves the output to ROOT files in addition to the dill files.")

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