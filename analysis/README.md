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
## Extracting cross section from premade analysis inputs

**NOTE: This method uses the existing analysis inputs, so plots and results before the fit cannot be made without fully running the analysis stack procedure.**

### Setup

Make sure your environment is loaded:
```
source <path to python repo>/pion-argon-xs-analysis/analysis/env.sh
```

Within the directory `pdsp_2GeV_analysis_inputs/`, you should find two directories:

```
analysis_input:
analysis_input_data_selected.dill  analysis_input_mc_cheated.dill  analysis_input_mc_selected.dill

mach3_input:
pdsp_Rabsorption.root  pdsp_Rcharge_exchange.root  pdsp_Rpion_production.root  pdsp_Runcategorised.root
```

The `dill` files are used within the Python framework, while the `root` files are the same samples converted to a readable format for MaCh3.

### Create working directory and configuration

Make a directory, and in the directory copy the pre-made 2 GeV configuration:

```
mkdir analysis_simplified/
cd analysis_simplified/
cp <path to python repo>/pion-argon-xs-analysis/analysis/config/cex_analysis_2GeV_config.json .
```

In the following add the configuration information for the analysis inputs:

```
"ANALYSIS_INPUTS": {
    "mc_cheated": "/data/dune/common/pdsp_2GeV_analysis_inputs/analysis_input/analysis_input_mc_cheated.dill",
    "mc": "/data/dune/common/pdsp_2GeV_analysis_inputs/analysis_input/analysis_input_mc_selected.dill",
    "data": "/data/dune/common/pdsp_2GeV_analysis_inputs/analysis_input/analysis_input_data_selected.dill"
}
```

### Creating fit model

The fit model can be recreated for convenience (e.g. different binning), and to do so, run the following:

```
cex_generate_fit_model.py -c cex_analysis_2GeV_config.json -o .
```

You will find the fit model created in the workspace: `fit_model/PDSPFitModel.yaml`

Now, you can update the path for the fit models in the MaCh3 configuration:

from
```
XsecCovFile: "Configs/CovObjs/PDSPFitModel.yaml"
```

to
```
XsecCovFile: "<path to analysis workspace>/fit_model/PDSPFitModel.yaml"
```

### Performing the fit

To perform the fit, follow the instructions in the MaCh3 documentation. Before doing so, ensure the paths in `SampleHandler_PDSP.yaml` are replaced as follows:

```
mtupleprefix: "<path to analysis inputs>/pdsp_"
```

Once the fit is performed, you should get an output root file `Test.root`. This needs to be included in the analysis configuration `json` under the MACH3 section:

``` 
"MACH3": {
    "KE_int_binning": {
      "range": [
        0,
        2400
      ],
      "bin_width": 50
    },
    "fit_output": <path to fit output>/Test.root
  },
```

### Extract cross section

Run the following:

```
cex_cross_section.py -c cex_analysis_2GeV_config.json -o .
```


Core modules are Master.py, vector.py and Plots.py (optional). A simple example of how to look at true data is shown in `cex_beam_quality.py`, and the other scripts are more complicated examples of how to analyse nTuples. For further detail on each module you can read the docstrings.
## Run cross section analysis

### Simple
---
The cross section analysis can be run using the regression test script, with the assumption the root files can be located. Run the following:

```
run_regression_test.py -d . -f <path to ntuples>
```

Note that the file path is searched recusively, so you can just share the single file path. It will match the ntuple file name in the reference 2GeV configuration to the files it searches. Due to this, the regression test currently only runs with the 2 GeV Dataset.

Optionally, you can include the option `--cpus` to use more cpus for multiprocessing. The output analysis area will be timestamped and contain a complete configuration, and if ran sucessfully full plots to study. Note that no fit is preformed so the only measurements are prefit cross sections based on a cheated MC sample.

### Advanced
---
This is run using configuration files, located in `config/`. All applications and notebooks which run the with the prefix `cex`. To run the entire analysis chain except the Fit, `run_analysis.py`. Note that if the fit is not run, the cross section extraction will only calculate the prefit cross section from MC.

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

This configuration requires entry of basic information such as data file location and some configurations settings some apps cannot run without. To work off a minimal application with the basic information (except MC file location) settings check `config/cex_analysis_2GeV_config.json`.

For now, copy the minimal config file to your area:

```bash
cp ../../config/cex_analysis_2GeV_config.json analysis_config.json
```

open the file and note the first three entries in the json file:

```json
  "NTUPLE_FILES": {
    "mc": [
      {
        "file": "PDSPProd4a_MC_2GeV_sce_datadriven_ntuple_v09_81_00d01_set0.root",
        "type": "PDSPAnalyser",
        "momentum_scale": 1
      },
      {
        "file": "PDSPProd4a_MC_2GeV_sce_datadriven_ntuple_v09_81_00d01_set1.root",
        "type": "PDSPAnalyser",
        "momentum_scale": 1
      },
      {
        "file": "PDSPProd4a_MC_2GeV_sce_datadriven_ntuple_v09_81_00d01_set2.root",
        "type": "PDSPAnalyser",
        "momentum_scale": 1
      },
      {
        "file": "PDSPProd4a_MC_2GeV_sce_datadriven_ntuple_v09_81_00d01_set3.root",
        "type": "PDSPAnalyser",
        "momentum_scale": 1
      },
      {
        "file": "PDSPProd4a_MC_2GeV_reco1_sce_datadriven_v1_ntuple_v09_41_00_03.root",
        "type": "PDSPAnalyser",
        "momentum_scale": 2
      }
    ],
    "data": [
      {
        "file": "PDSPProd4_data_2GeV_reco2_ntuple_v09_42_03_01.root",
        "type": "PDSPAnalyser",
        "momentum_scale": 1
      }
    ]
  }

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


**To run without data files, the `"data"` entry should be completely excluded.**

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
config.json

analysis_input:
analysis_input_data_selected.dill  analysis_input_mc_cheated.dill  analysis_input_mc_selected.dill

beam_norm:
norm.json  output.dill  plots.pdf

beam_quality:
beam_quality_fits.pdf  data_beam_quality_fit_values.json  mc_beam_quality_fit_values.json  output.dill

beam_reweight:
chi2_reweight.hdf5  chi2_reweight.tex  crystal_ball.json  double_crystal_ball.json  double_gaussian.json  gaussian.json  output.dill  plots  poly2d.json  selection_data.hdf5  selection_mc.hdf5  student_t.json

beam_scraper:
beam_scraper_fits.pdf  mc_beam_scraper_fit_values.json  output.dill

mach3_input:
pdsp_Rabsorption.root  pdsp_Rcharge_exchange.root  pdsp_Rescaping.root  pdsp_Rpion_production.root  pdsp_Runcategorised.root

masks_data:
beam_selection_masks.dill  loose_photon_selection_masks.dill  loose_pi_selection_masks.dill  null_pfo_selection_masks.dill  photon_selection_masks.dill  pi0_selection_masks.dill  pi_selection_masks.dill

masks_mc:
beam_selection_masks.dill  loose_photon_selection_masks.dill  loose_pi_selection_masks.dill  null_pfo_selection_masks.dill  photon_selection_masks.dill  pi0_selection_masks.dill  pi_selection_masks.dill

selection:
output_data.dill  output_mc.dill  plots

shower_energy_correction:
gaussian.json  gaussian.tex  mean.json  mean.tex  photon_energies.hdf5  plots.pdf  student_t.json  student_t.tex  table.tex

tables_data:
beam  loose_photon  loose_pi  null_pfo  photon  pi  pi0

tables_mc:
beam  loose_photon  loose_pi  null_pfo  photon  pi  pi0

upstream_loss:
cex_upstream_loss_plots.pdf  fit_parameters.json  output.dill

xs_measurement:
cross_section_plots.pdf
```

outputs will be of seven types, `pdf`, `json`, `tex`, `hdf5`,  `dill`, `yaml` and `root`.

 * `pdf` are plots produced by the various apps
 * `json` are values computed by the apps which are important for other apps to function. This could be something like fitted parameters or numerical constants or whole configuration settings
 * `tex` are tables saved in LaTeX format i.e. for results where plots are not appropriate.
 * `hdf5` is data which can be stored as a pandas dataframe. This is usally data which is useful for further studies, dut does not require computing them again using the Ntuple file.
 * `dill`, similar to `hdf5`, this is data which is useful for further study but does not require computing them again. The difference is this data is stored as serialisable python objects i.e. can only be correcty opened using python 
 * `yaml`, similatr to json, a data file format used for configuraiton definition in MaCh3.
 * `root`, output file format compatible with MaCh3.

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

## Toy generator (Deprecated)
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

## Running systematics (Deprecated)

Make sure to run all the steps in `run_analysis.py` and have a configuration for a toy template file and toy data sample (the difference being reduced stats). Then run the following 

`cex_systematics.py -c <analysis configuration file> -o <analysis directory> --cv <dill file of your central value measurement>`

where, similar to `run_analysis.py` you can provide the argument `run`, `skip` and `regen`, then give a list of all the systematics you wish to evaluate.

An example would be (to evaluate the mc stat uncertainty.):

`cex_systematics.py -c <analysis configuration file> -o <analysis directory> --cv <dill file of your central value measurement> --run mc_stat`

**WARNING THIS WILL TAKE A LONG TIME IF YOU DO `--run all` SO BE CAUTIOUS**

To make a plot of the central value + any systematics you did generate, run

`cex_systematics.py -c <analysis configuration file> -o <analysis directory> --cv <dill file of your central value measurement> --plot`.