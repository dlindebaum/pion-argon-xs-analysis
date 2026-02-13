# Analysis
Python code for ProtoDUNE analysis, requires a python 3.13 environment or greater.

If you want to install the repo in an existing environment, run the following:
``` bash
pip install -r requirements.txt
```

If you want to install a new environment, run the following:
```bash
conda env create -f environment.yml
```

To activate the new enrionment, run the following or include this in your bashrc.

```bash
conda activate pdune_analysis
```

Each time you load the python environment:
``` bash
source env.sh
```
---
To generate the `requirements.txt` file, run the following:

```bash
pip freeze > requirements.txt
```

And to generate the `environment.yml` file:

```bash
conda env export | sed -n '/prefix:/q;p' > environment.yml
```

---

Code runs on ntuples produced by the Pi0 Analyser module (to be added), a list of produced ntuples are here:

[https://cernbox.cern.ch/index.php/s/8UqObev6XPNhRXn](https://cernbox.cern.ch/index.php/s/8UqObev6XPNhRXn)

if you are working on DICE, i.e. `sc01.dice.priv` files are located in hdfs
```bash
/hdfs/DUNE/physics/cex/
```

---

Core modules are Master.py, vector.py and Plots.py (optional). A simple example of how to look at true data is shown in `cex_beam_quality.py`, and the other scripts are more complicated examples of how to analyse nTuples. For further detail on each module you can read the docstrings.
## Run Cross section analysis

This is run using configuration files, located in `config/`. All applications and notebooks which run the with the prefix `cex`. To run the entire analysis chain with Data and MC (excluding toy studies and systematics) is done through `run_analysis.py`.

First, make a work area in this directory:

```bash
mkdir work
cd work
mkdir analysis_demo
cd analysis_demo
```

To create a template configuration called `analysis_config.json`, run the following in your work area:

```bash
run_analysis.py -C analysis_config.json -o .
```

This configuration requires entry of basic information such as data file location and some configurations settings some apps cannot run without. To work off a minimal application with the basic information (except MC file location) settings check `config/cex_analysis_2GeV_config_minimal_MC.json`.

For now, copy the minimal config file to your area:

```bash
cp ../../config/cex_analysis_2GeV_config_minimal_MC.json analysis_config.json
```

open the file and note the first three entries in the json file:

```json
  "NTUPLE_FILE":{
    "mc" : "MC ntuple file ENSURE ALL FILE PATHS ARE ABSOLUTE",
    "data" : null,
    "type" : "type of ntuple files, this is either PDSPAnalyser or shower_merging"
  },
  "norm" : "normalisation to apply to MC when making Data/MC comparisons, usually defined as the ratio of pion-like triggers from the beam instrumentation",
  "pmom" : "momentum byte of the beam i.e. central value of beam momentum in GeV, required if ntuple does not have the correct scale for the P_inst distribution",

```
`null` entries refer to an empty entry in the config, the others have descriptions describing what the entry refers to an possible values. For now populate the information as follows:

```json
  "NTUPLE_FILES": {
    "mc": [
      {
        "file": "<MC file path>",
        "type": "PDSPAnalyser",
        "pmom": 2
      }
    ],
    "data": [
      {
        "file": "<Data file path>",
        "type": "PDSPAnalyser",
        "pmom": 1
      }
    ]
  },
  "norm" : 1,
```

"mc" should be set to the file path of the 2GeV MC file called `PDSPProd4a_MC_2GeV_reco1_sce_datadriven_v1_ntuple_v09_41_00_03.root` on the machine you are working on. for this ntuple file the "type" is PDSPAnalyser, no data is used, so "norm" is arbitrarily set to 1. For this specific MC file, "pmom" must be 2. If this must be set, it should be the expected beam energy in GeV.

To run without data files, the `"data"` entry should be completely excluded.

Save and close the file, now run the analysis (or most of it)

```bash
run_analysis.py -c analysis_config.json -o .
```

where `-o` sets the output path of the various results.

when running you will see this message:

```bash
no data file was specified, 'normalisation', 'beam_reweight', 'toy_parameters' and 'analyse' will not run
```

This is because without a data file, the full analysis can't be run. When the script finishes you should see multiple folders which are the outputs of the various applications:

```bash
ls *
```

```bash
analysis_config.json

analysis_input:

beam_quality:
beam_quality_fits.pdf  mc_beam_quality_fit_values.json

beam_scraper:
beam_scraper_fits.pdf  mc_beam_scraper_fit_values.json

masks_mc:
beam_selection_masks.dill  null_pfo_selection_masks.dill  photon_selection_masks.dill  pi0_selection_masks.dill  pip_selection_masks.dill

plots:
beam.pdf  photon.pdf  pi0.pdf  piplus.pdf  regions.pdf

shower_energy_correction:
gaussian.json  mean.json  photon_energies.hdf5  plots.pdf  student_t.json

tables_mc:
beam  null_pfo  photon  pi0  pip

toy_parameters:
beam_profile  meanTrackScoreKDE  pi_beam_efficiency  reco_regions  smearing

upstream_loss:
cex_upstream_loss_plots.pdf  fit_parameters.json
```

outputs will be of five types, `pdf`, `json`, `tex`, `hdf5`, and `dill`.

 * `pdf` are plots produced by the various apps
 * `json` are values computed by the apps which are important for other apps to function. This could be something like fitted parameters or numerical constants or whole configuration settings
 * `tex` are tables saved in LaTeX format i.e. for results where plots are not appropriate.
 * `hdf5` is data which can be stored as a pandas dataframe. This is usally data which is useful for further studies, dut does not require computing them again using the Ntuple file.
 * `dill`, similar to `hdf5`, this is data which is useful for further study but does not require computing them again. The difference is this data is stored as serialisable python objects i.e. can only be correcty opened using python 

This example ran with MC, to run with Data, you can add the corresponding Data ntuple file path, and run the analysis again, this time forcing all prior steps to be re-ran:

```bash
run_analysis.py -c analysis_config.json -o . --force
```

Now, if you run the analysis again, you will notice the application finishes very quickly. This is because the analysis will *NOT* run any steps again it doesn't need to. This can be overriden with the `--force` option but this can be more fine tuned.

If you want to run a specific part of the analysis you can use the `run` option:

```bash
run_analysis.py -c analysis_config.json -o . --run <list of steps to run>
```

and you can skip certain steps with

```bash
run_analysis.py -c analysis_config.json -o . --skip <list of steps to skip>
```

Note `--skip` will do nothing if --force is not specified or the analysis has not been run for the first time.

Note that these options can be combined e.g.

```bash
run_analysis.py -c analysis_config.json -o . --skip <selection, photon_correction> --run <beam_scraper_fit>
```

```bash
run_analysis.py -c analysis_config.json -o . --skip <selection, photon_correction> --force
```

check `--help` for the names of all the apps which can be skipped or forced to run.

## Toy generator
To generate toys, you need to have run `cex_toy_parameters.py`. Then create a new json file to create your toy sample. An example template for the toy configuration is

```[json]
{
  "events": 1000000,
  "step": 2,
  "p_init": 2000,
  "beam_profile": "<path_to_your_analysis_directory>/toy_parameters/beam_profile/beam_profile.json",
  "beam_width": 60,
  "smearing_params": {
    "KE_init": "<path_to_your_analysis_directory>/toy_parameters/smearing/KE_init/double_crystal_ball.json",
    "KE_int": "<path_to_your_analysis_directory>/toy_parameters/smearing/KE_int/double_crystal_ball.json",
    "z_int": "<path_to_your_analysis_directory>/toy_parameters/smearing/z_int/double_crystal_ball.json"
  },
  "reco_region_fractions": "<path_to_your_analysis_directory>/toy_parameters/reco_regions/moderate_efficiency_reco_region_fractions.hdf5",
  "beam_selection_efficiencies": "<path_to_your_analysis_directory>/toy_parameters/pi_beam_efficiency/beam_selection_efficiencies_true.hdf5",
  "mean_track_score_kde": "<path_to_your_analysis_directory>/toy_parameters/meanTrackScoreKDE/kdes.dill",
  "pdf_scale_factors": null,
  "df_format": "f",
  "modified_PDFs": null,
  "verbose": true,
  "seed": 1337,
  "max_cpus": 21
}
```
Note that the beam profile takes a file in the example, but this can also be replaced with either `uniform` or `gaussian` to generate a generic beam profile with those distribution shapes.

to generate the toy run

```
cex_toy_generator.py -c <your_toy_config_file>
```

which will produce an HDF5 file with the generated toy sample. Note the toy sample is used for systematic studies, but can also be used to do the fit, background estimation and cross section measurement.

## Running systematics

Make sure to run all the steps in `run_analysis.py` and have a configuration for a toy template file and toy data sample (the difference being reduced stats). Then run the following 

`cex_systematics.py -c <analysis configuration file> -o <analysis directory> --cv <dill file of your central value measurement>`

where, similar to `run_analysis.py` you can provide the argument `run`, `skip` and `regen`, then give a list of all the systematics you wish to evaluate.

An example would be (to evaluate the mc stat uncertainty.):

`cex_systematics.py -c <analysis configuration file> -o <analysis directory> --cv <dill file of your central value measurement> --run mc_stat`

**WARNING THIS WILL TAKE A LONG TIME IF YOU DO `--run all` SO BE CAUTIOUS**

To make a plot of the central value + any systematics you did generate, run

`cex_systematics.py -c <analysis configuration file> -o <analysis directory> --cv <dill file of your central value measurement> --plot`.