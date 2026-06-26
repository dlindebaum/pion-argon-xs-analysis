"""
Created on: 10/06/2026 08:40

Author: Shyam Bhuller

Description: Module for classes and functions which compose applications.  
"""
import argparse
import copy
import numbers

from python.analysis.Master import LoadConfiguration, LoadObject, FileDescriptor
from python.analysis.cross_section import EnergyCorrection, Slices
from python.analysis import BeamParticleSelection, PFOSelection, EventSelection, Fitting, RegionDefinitions, ProcessDefinitions, Processing


class ApplicationArguments:
    @staticmethod
    def Processing(parser : argparse.ArgumentParser):
        parser.add_argument("-b", "--batches", dest = "batches", type = int, default = None, help = "Number of batches to split n tuple files into when parallel processing processing data.")
        parser.add_argument("-e", "--events", dest = "events", type = int, default = None, help = "Number of events to process when parallel processing data.")
        parser.add_argument("-t", "--threads", dest = "threads", type = int, default = 1, help = "Number of threads to use when processsing.")

    @staticmethod
    def Regen(parser : argparse.ArgumentParser):
        parser.add_argument("-r", "--regen", dest = "regen", action = "store_true", help = "Regenerate any stored data.")

    @staticmethod
    def Output(parser : argparse.ArgumentParser, default : str = None, file : bool = False):
        if file is True:
            help = "output file name."
        else:
            help = "Directory to save output files."
        parser.add_argument("-o", "--out", dest = "out", type = str, default = default, help = help)
        return

    @staticmethod
    def Plots(parser : argparse.ArgumentParser):
        parser.add_argument("--nbins", dest = "nbins", type = int, default = 50, help = "Number of bins to make for histogram plots.")
        parser.add_argument("-a", "--annotation", dest = "annotation", type = str, default = None, help = "Annotation to add to plots.")
        return

    @staticmethod
    def Config(parser : argparse.ArgumentParser, required : bool = False, default : any = None):
        parser.add_argument("-c", "--config", dest = "config", type = str, default = default, required = required, help = "Analysis configuration file, if supplied will override command line arguments.")

    @staticmethod
    def ResolveArgs(args : argparse.Namespace, override_out : bool = True) -> argparse.Namespace:
        """ Parses command line arguements.

        Args:
            args (argparse.Namespace): arguements to parse

        Returns:
            argparse.Namespace: parsed arguements
        """
        if hasattr(args, "config"):
            args_copy = argparse.Namespace()
            for a, v in args._get_kwargs():
                setattr(args_copy, a, v)
            args = ApplicationArguments.ResolveConfig(LoadConfiguration(args.config))
            for a, v in args_copy._get_kwargs():
                if a not in args:
                    setattr(args, a, v)
        else:
            if hasattr(args, "data_file") and hasattr(args, "data_beam_quality_fit"):
                if args.data_file is not None and args.data_beam_quality_fit is None:
                    raise Exception("beam quality fit values for data are required")

        if hasattr(args, "out") and (override_out is True):
            if args.out is None:
                filename = None
                if hasattr(args, "mc_file"):
                    filename = args.mc_file
                elif hasattr(args, "data_file"):
                    filename = args.data_file
                elif hasattr(args, "file"):
                    filename = args.file
                else:
                    filename = ""

                if type(filename) == list:
                    if len(filename) == 1:
                        args.out = filename[0].split("/")[-1].split(".")[0] + "/"
                    else:
                        args.out = "output/" #? how to make a better name for multiple input files?
                else:
                    args.out = filename.split("/")[-1].split(".")[0] + "/"
            if args.out[-1] != "/": args.out += "/"

        return args

    @staticmethod
    def __CreateSelection(value : dict, module) -> dict:
        """ Creates a dictionary of selection functions and their argumenets as specified in the analysise configuration file. 

        Args:
            value (dict): dicionary describing a particular selection, key value is the function name, value is a list of function arguements.
            module (module): which python module does this selection belong to.

        Returns:
            _type_: _description_
        """
        selection = {"selections" : {}, "arguments" : {}}
        for func, opt in value.items():
            if opt["enable"] is True:
                selection["selections"][func] = getattr(module, func)
                copy = opt.copy()
                copy.pop("enable")
                selection["arguments"][func] = copy
        return selection

    @staticmethod
    def ResolveConfig(config : dict) -> argparse.Namespace:
        """ Reads analysis configuration file and unpacks/serializes relavent objects.

        Args:
            config (dict): file path

        Returns:
            argparse.Namespace: unpacked configuration
        """
        args = argparse.Namespace()
        for head, value in config.items():
            if head == "NTUPLE_FILES":
                ntuple_files = value
                for k in ntuple_files:
                    ntuple_files[k] = [FileDescriptor(**i) for i in ntuple_files[k]]
                args.ntuple_files = ntuple_files
            elif head == "SAMPLE_DEFINITIONS":
                args.region_definitions = RegionDefinitions.regions[value["region"]]
                args.process_definitions = ProcessDefinitions.processes[value["process"]]
                args.process_args = value["process_args"]
                args.region_args = value["region_args"]
            elif head == "BEAM_PARTICLE_SELECTION":
                args.beam_selection = ApplicationArguments.__CreateSelection(value, BeamParticleSelection)
            elif head == "HAS_FINAL_STATE_PFO_SELECTION":
                args.has_final_state_pfo_selection = value["enable"]
            elif head == "VALID_PFO_SELECTION":
                args.valid_pfo_selection = value["enable"]
            elif head == "FINAL_STATE_PIPLUS_SELECTION":
                args.piplus_selection = ApplicationArguments.__CreateSelection(value, PFOSelection)
            elif head == "FINAL_STATE_PHOTON_SELECTION":
                args.photon_selection = ApplicationArguments.__CreateSelection(value, PFOSelection)
            elif head == "FINAL_STATE_LOOSE_PION_SELECTION":
                args.loose_pion_selection = ApplicationArguments.__CreateSelection(value, PFOSelection)
            elif head == "FINAL_STATE_LOOSE_PHOTON_SELECTION":
                args.loose_photon_selection = ApplicationArguments.__CreateSelection(value, PFOSelection)
            elif head == "FINAL_STATE_PI0_SELECTION":
                args.pi0_selection = ApplicationArguments.__CreateSelection(value, EventSelection)
            elif head == "BEAM_QUALITY_FITS":
                if "mc" in value:
                    args.mc_beam_quality_fit = LoadConfiguration(value["mc"]) # generally expected to have MC at a minimum
                if "data" in value:
                    args.data_beam_quality_fit = LoadConfiguration(value["data"])
                args.beam_quality_truncate = value["truncate"]
            elif head == "BEAM_SCRAPER_FITS":
                args.beam_scraper_energy_range = value["energy_range"]
                args.beam_scraper_energy_bins = value["energy_bins"]
                if not hasattr(args.beam_scraper_energy_bins, "__iter__"):
                    raise Exception("energy_bins must be a list of beam energy bin ranges in MeV")
                if "mc" in value:
                    args.mc_beam_scraper_fit = LoadConfiguration(value["mc"])
            elif head == "ENERGY_CORRECTION":
                args.shower_correction = {}
                for k, v in value.items():
                    args.shower_correction[k] = v
            elif head == "UPSTREAM_ENERGY_LOSS":
                args.upstream_loss_cv_function = value["cv_function"]
                args.upstream_loss_response = getattr(Fitting, value["response"])
                if value["bins"] is None:
                    raise Exception("Upstream energy loss KE bins need to be provided (in MeV)")
                args.upstream_loss_bins = value["bins"]
                if "correction_params" in value:
                    args.upstream_loss_correction_params = LoadConfiguration(value["correction_params"])
            elif head == "BEAM_REWEIGHT":
                args.beam_reweight = {}
                if value["params"] is not None:
                    args.beam_reweight["params"] = LoadConfiguration(value["params"])
                args.beam_reweight["strength"] = value["strength"]

            elif head == "SELECTION_MASKS":
                args.selection_masks = {}
                for k, v in value.items():
                    args.selection_masks[k] = {i : LoadObject(j) for i, j in v.items()}
            elif head == "TOY_PARAMETERS":
                args.toy_parameters = {}
                for k, v in value.items():
                    if k == "beam_profile": 
                        args.toy_parameters[k] = getattr(Fitting, v)
                    else:
                        for k1, v1 in v.items():
                            if not ((hasattr(v1, "__iter__")) and (len(v1) == 2)):
                                raise Exception(f"{k1} must be a list of two numbers.")
                        args.toy_parameters[k] = v
            elif head == "FIT":
                args.fit = {}
                for k, v in value.items():
                    args.fit[k] = v
            elif head == "ESLICE":
                for k, v in value.items():
                    if not isinstance(v, numbers.Number):
                        raise Exception(f"All SLICE paramters must be a number ({k}:{v}).")
                # if value["width"] is not None:
                args.energy_slices = Slices(value["width"], value["min"] - value["width"], value["max"], reversed = True) # min - width to allocate an underflow bin (not used in the measurement)
            elif head == "ANALYSIS_INPUTS":
                args.analysis_input = {k : v for k, v in value.items()}
            elif head == "UNFOLDING":
                args.unfolding = {k : v for k, v in value.items()}
            elif head == "MACH3_INPUT":
                args.mach3_input = value
            elif head == "KINEMATIC_RANGES":
                for k, v in value.items():
                    if k == "beam_momentum":
                        msg = "must be the nominal central value of the beam momentum distribution in MeV."
                        cond = (type(v) == float) or (type(v) == int)
                    else:
                        msg = "must be a list of two elements"
                        cond = (hasattr(v, "__iter__")) and (len(v) == 2)

                    if not cond:
                        raise Exception(f"{k} {msg}")
                    else:
                        setattr(args, k, v)
            else:
                setattr(args, head, value) # allow for generic configurations in the json file
        if hasattr(args, "beam_selection"):
            ApplicationArguments.DataMCSelectionArgs(args)
        if hasattr(args, "pi0_selection"):
            ApplicationArguments.AddEnergyCorrection(args)
        if hasattr(args, "beam_selection"):
            if "PiBeamSelection" in args.beam_selection["mc_arguments"]:
                args.beam_selection["data_arguments"]["PiBeamSelection"]["use_beam_inst"] = True # make sure to set the correct settings for data.
                args.beam_selection["mc_arguments"]["PiBeamSelection"]["use_beam_inst"] = False # make sure to set the correct settings for data.
            if "TrueFiducialCut" in args.beam_selection["mc_arguments"]:            
                args.beam_selection["data_arguments"]["TrueFiducialCut"]["is_mc"] = False
                args.beam_selection["mc_arguments"]["TrueFiducialCut"]["is_mc"] = True

        return args


    @staticmethod
    def AddEnergyCorrection(args):
        if hasattr(args, "shower_correction") and (args.shower_correction["correction_params"] != None):
            method = EnergyCorrection.shower_energy_correction[args.shower_correction["correction"]]
            params = LoadConfiguration(args.shower_correction["correction_params"])
        else:
            method = None
            params = None
            args.shower_correction["correction"] = None
            args.shower_correction["correction_params"] = None
        args.pi0_selection["mc_arguments"]["Pi0MassSelection"]["correction"] = method
        args.pi0_selection["mc_arguments"]["Pi0MassSelection"]["correction_params"] = params
        args.pi0_selection["data_arguments"]["Pi0MassSelection"]["correction"] = method
        args.pi0_selection["data_arguments"]["Pi0MassSelection"]["correction_params"] = params

    @staticmethod
    def DataMCSelectionArgs(args : argparse.Namespace):
        for a in vars(args):
            if ("selection" in a) and (type(getattr(args, a)) == dict):
                if "arguments" in getattr(args, a): 
                    getattr(args, a)["mc_arguments"] = copy.deepcopy(getattr(args, a)["arguments"])
                    getattr(args, a)["data_arguments"] = copy.deepcopy(getattr(args, a)["arguments"])
                    getattr(args, a).pop("arguments")

        for i, s in args.beam_selection["selections"].items():
            if s in [BeamParticleSelection.BeamQualityCut, BeamParticleSelection.DxyCut, BeamParticleSelection.DzCut, BeamParticleSelection.CosThetaCut]:
                if hasattr(args, "mc_beam_quality_fit"): 
                    args.beam_selection["mc_arguments"][i]["fits"] = args.mc_beam_quality_fit
                if hasattr(args, "data_beam_quality_fit"): 
                    args.beam_selection["data_arguments"][i]["fits"] = args.data_beam_quality_fit
            elif s == BeamParticleSelection.BeamScraperCut:
                if hasattr(args, "mc_beam_scraper_fit"): 
                    args.beam_selection["mc_arguments"][i]["fits"] = args.mc_beam_scraper_fit
                    args.beam_selection["data_arguments"][i]["fits"] = args.mc_beam_scraper_fit
            else:
                continue
        return args
