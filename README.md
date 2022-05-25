# pi0-analysis

LArSoft module which retrieves and calculates some useful quantities to study reconstructed data/MC.

## Helpful links:
[DUNE doxygen page](https://internal.dunescience.org/doxygen/index.html)

[DUNE computing page](https://wiki.dunescience.org/wiki/DUNE_Computing)

[Setup a dunesw environment](https://github.com/DUNE/dunesw/wiki)

[ProtoDUNE SP data/MC list](https://wiki.dunescience.org/wiki/Look_at_ProtoDUNE_SP_data)

---
## Installation
Folder should be cloned in `src/protoduneana/`. The CMakeLists file that exists in the same directory as the cloned repo should be modified to include this line:
```
add_subdirectory(pi0-analysis)
```

Then, then rebuild the full environment by running `mrb i -j 16` in the build directory. Ensure you are doiing this on either `dunebuild01.fnal.gov` or `dunebuild02.fnal.gov` or else you will be shouted at.

---
## Usage
If the environment successfully builds you can run the LArsoft analyser by running one of these fcl files:
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

---
## Running grid jobs
To run grid jobs first run the following command on a dunegpvm:
```bash
setup_fnal_security
```
Which creates a new voms-proxy to let you submit jobs (note this will expire in 7 days so you should run this at least weekly). Now to run the analyser you need to create a tarball of the local products directory, as well as a bash script to setu the environment on the remote machine and a list of files you wish to process.

Two bash scripts are needed to setup the environment, one called `setup-grid` and the other `setup-jobenv.sh`. Copy `setup-grid` into the localProducts directory and `setup-jobenv.sh` in the top directory of the environment.

To create a file list you can collect a list of prestaged data using `get_staged.py` for a given samweb definition i.e.:

```bash
python get_staged.py PDSPProd4a_MC_6GeV_reco1_sce_datadriven_v1_00
```
note before doing this check how many files are prestaged using cached_state.py:
```bash
cached_state.py -d <samweb definition>
```
If you need to prestage a dataset then run the same command with the -p flag (also nohup as prestaging takes as while?)

To create a tarball which can run on the remote machines:
```bash
tar -czvf dunesw.tar.gz localProducts* setup-jobenv.sh <file-list>
```
Ensure these files are on the top level directory.

Now to run jobs:
```bash
python submit_job.py <bash-file-to-run> -t <tarball>
```
and more options can be found with --help.

To check the status of jobs you can use:
```bash
jobsub_q --user <fermilab-username>
```
example output:
```bash
bash-4.2$ jobsub_q --user sbhuller
JOBSUBJOBID                           OWNER           SUBMITTED     RUN_TIME   ST PRI SIZE CMD
56541241.0@jobsub01.fnal.gov          sbhuller        05/25 07:33   0+00:00:00 I   0   0.0 1_20220525_073349_292366_0_1_wrap.sh 

1 jobs; 0 completed, 0 removed, 1 idle, 0 running, 0 held, 0 suspended
```
Things to note, `I` indictes the job is idle and is waiting to be run on a worker node. `R` inidcates the job is running `H` means the job has been held. This usually occurs if the job exceeds the resource usage. However, in `submit_job.py` an option to restart jobs with increases resources is used, so if a job is held it will attempt to re-run the job one more time.

For a better monitoring experience, you can use [grafana](https://fifemon.fnal.gov/monitor/d/000000116/user-batch-details?orgId=1&var-cluster=fifebatch&var-user=sbhuller), make sure to switch to your home page using the buttons on the top left of the page.

---
## Making changes
Note if you need to create new fcl files or c++ code, you need to do fully rebuild the environment i.e. `mrb i -j 16`, if you are modifying an exisitng file then you can just rebuild the changed files with
```
mrbsetenv # once per session
make install -j 16
```
in your build directory.
