# pi0-analysis

LArSoft module which retrieves and calculates some useful quantities to study reconstructed data/MC.

## Helpful links:
[DUNE doxygen page](https://internal.dunescience.org/doxygen/index.html)

[DUNE computing page](https://wiki.dunescience.org/wiki/DUNE_Computing)

[ProtoDUNE SP data/MC list](https://wiki.dunescience.org/wiki/Look_at_ProtoDUNE_SP_data)

## Installation
Repository should be cloned in either `src/dunetpc` or `src/protoduneana`. The CMakeLists file that exists in the same directory as the cloned repo should be modified to include this line:
```
add_subdirectory(pi0-analysis)
```

Then, then rebuild the full environment by running `mrb i -j 16` in the build directory. Ensure you are doiing this on either `dunebuild01.fnal.gov` or `dunebuild02.fnal.gov` or else you will be shouted at.

## Usage
if the environment successfully builds you can run the LArsoft analyser by running one of these fcl files:
```
runDiphoton.fcl
runPi0_BeamSim.fcl
runPi0.fcl
runPi0_noFilter.fcl (same as runPi0_noFilter.fcl should be removed)
runPi0Test.fcl
```
e.g.
```
lar -c runPi0_BeamSim.fcl <file list or root file>
```

## Running grid jobs
to run grid jobs first run the following command on a dunegpvm:
```bash
setup_fnal_security
```
Which creates a new voms-proxy to let you submit jobs (note this will expire in 7 days so you should run this at least weekly). Now to run the analyser you need to create a tarball of the local products directory, as well as a bash script to setu the environment on the remote machine and a list of files you wish to process.

two bash scripts are needed to setup the environment, one called `setup-grid` and the other `setup-jobenv.sh` copy `setup-grid` into the localProducts directory and `setup-jobenv.sh` in the top directory of the environment.

to create a file list you can collect a list of prestaged data using `get_staged.py` for a given samweb definition i.e.:

```bash
python get_staged.py PDSPProd4a_MC_6GeV_reco1_sce_datadriven_v1_00
```

note before doing this check how many files are prestaged using cached_state.py:
```bash
cached_state.py -d <samweb definition>
```

if you want to prestage a dataset then run the same command with the -p flag (also nohup as prestaging takes as while?)

## Making changes
Note if you need to create new fcl files or c++ code, you need to do fully rebuild the environment i.e. `mrb i -j 16`, if you are modifying an exisitng file then you can just rebuild the changed files with
```
mrbsetenv # once per session
make install -j 16
```
in your build directory.
