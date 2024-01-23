#!/usr/bin/env python3
"""
Created on: 23/01/2024 10:34

Author: Shyam Bhuller

Description: 
"""
from rich import print

from apps import (
    cex_beam_quality_fits, 
    cex_beam_scraper_fits,
    cex_photon_selection, #! notebook stuff should be moved to the application!
    cex_selection_studies, 
    cex_beam_reweight, 
    cex_upstream_loss, 
    cex_toy_parameters,
    cex_analysis_input
    )

from python.analysis.cross_section import ApplicationArguments, argparse, os
from python.analysis.Master import SaveConfiguration, LoadConfiguration

def template_config():
    template = {
        "NTUPLE_FILE":{
            "mc" : "MC ntuple file ENSURE ALL FILE PATHS ARE ABSOLUTE",
            "data" : "Data ntuple file",
            "type" : "type of ntuple files, this is either PDSPAnalyser or shower_merging"
        },
        "norm" : "normalisation to apply to MC when making Data/MC comparisons, usually defined as the ratio of pion-like triggers from the beam instrumentation", #! this should be inferred from one of the apps!
        "pmom" : "momentum byte of the beam i.e. central value of beam momentum in GeV, required if ntuple does not have the correct scale for the P_inst distribution",
        "BEAM_SCRAPER_FITS":{
            "energy_range" : None,
            "energy_bins" : None
        },
        "UPSTREAM_ENERGY_LOSS":{
            "cv_function" : "gaussian",
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
                "op" : "<"
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
        "FINAL_STATE_PIPLUS_SELECTION":{
            "TrackScoreCut":{
                "enable" : True,
                "cut" : 0.5,
                "op" : ">"
            },
            "NHitsCut":{
                "enable" : True,
                "cut" : 20,
                "op" : ">"
            },
            "PiPlusSelection":{
                "enable" : True,
                "cut" : [0.5, 2.8],
                "op" : [">", "<"]
            }
        },
        "FINAL_STATE_PHOTON_SELECTION":{    
            "TrackScoreCut":{
                "enable" : True,
                "cut" : 0.5,
                "op" : "<"
            },
            "NHitsCut":{
                "enable" : True,
                "cut" : 80,
                "op" : ">"
            },
            "BeamParticleDistanceCut":{
                "enable" : True,
                "cut" : [3, 90],
                "op" : [">", "<"]
            },
            "BeamParticleIPCut":{
                "enable" : True,
                "cut" : 20,
                "op" : "<"
            }
        },
        "FINAL_STATE_PI0_SELECTION":{
            "NPhotonCandidateSelection":{
                "enable" : True,
                "cut" : 2,
                "op" : "=="
            },
            "Pi0MassSelection":{
                "enable" : True,
                "cut" : [50, 250],
                "op" : [">", "<"]
            },
            "Pi0OpeningAngleSelection":{
                "enable" : True,
                "cut" : [10, 80],
                "op" : [">", "<"]
            }
        },
        "beam_momentum" : "nominal beam momentum in MeV", #! should be deprciated
        "P_inst_range" : "plot range",
        "KE_inst_range" : "plot range",
        "KE_init_range" : "plot range",
        "KE_int_range" : "plot range"
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


def main(args):
    if args.create_config:
        SaveConfiguration(template_config(), args.create_config)
        print(f"template configuration saved as {args.create_config}")
        exit()
    else:
        os.makedirs(args.out, exist_ok = True)
        print("run analysis, checking what steps have already been run")

        #* beam quality
        if (not hasattr(args, "mc_beam_quality_fit")) or ((args.data_file is not None) and (not hasattr(args, "data_beam_quality_fit"))):
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
            update_config(args.config, {"BEAM_QUALITY_FITS" : new_config_entry})
            args = update_args() # reload config to continue
        
        #* beam scraper
        if not hasattr(args, "mc_beam_scraper_fit"):
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

        #* photon energy correction
        if args.shower_energy_correction is True:
            if (args.correction is None) and (args.correction_params is None):
                cex_photon_selection.main(args) #? how to handle multiprocessing?
        
        #* selection studies
        if not hasattr(args, "selection_masks"):
            args.mc_only = args.data_file is None
            args.nbins = 50
            # pass multiprocessing args (using defaults)
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
                "pi" : 'pip_selection_masks.dill'
            }
            new_config_entry = {}
            files = os.listdir(output_path)
            for k, v in target_files.items():
                if v in files:
                    new_config_entry[k] = {i : os.path.abspath(output_path + v + "/" + j) for i, j in mask_map.items()}
            update_config(args.config, {"SELECTION_MASKS" : new_config_entry})
            args = update_args() # reload config to continue

        #* beam reweight
        if args.data_file is None: # data is required for this app
            print("beam reweight")

        #* upstream correction
        if not hasattr(args, "upstream_loss_correction_params"):
            print("upstream correction")
            args.no_reweight = not hasattr(args, "beam_reweight_params")
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
            new_config_entry["bins"] = args.upstream_loss_bins
            update_config(args.config, {"UPSTREAM_ENERGY_LOSS" : new_config_entry})
            args = update_args() # reload config to continue
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--create_config", type = str, help = "Create a template configuration with the default selection")
    parser.add_argument("--shower_energy_correction", action = "store_true", help = "whether to run the shower energy correction")
    ApplicationArguments.Config(parser)
    ApplicationArguments.Output(parser, "analysis/")
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