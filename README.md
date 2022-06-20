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

First you need to create a list of root files to run (1 per job). Data/MC is stored on tape and prestaged to disk when people need to run an analysis. Typically most recent data/MC remains on disk but just in case, check the status of a dataset using cached_state.py

```bash
cache_state.py -d <samweb definition>
```
If a significant portion is on tape, you need to prestage a dataset so run the same command with the -p flag (also nohup as prestaging takes as while). To get the samweb definition of a dataset you can usually locate this on the [ProtoDUNE SP data/MC list](https://wiki.dunescience.org/wiki/Look_at_ProtoDUNE_SP_data).

Now, create a file list of prestaged data using `get_staged.py` for a given samweb definition i.e.:

```bash
python get_staged.py PDSPProd4a_MC_6GeV_reco1_sce_datadriven_v1_00
```

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

## Retrieving grid job outputs
output files produced by the job script should be in the specified out directory and should be moved to persistent storage (i.e. `/pnfs/dune/persistent/users/$USER/`) as scratch directories are wiped every month.

To merge multiple ROOT files, first get a file list, one way to do so is:
```bash
ls <path to root files> > out.list
```

then you can run the command `merge-ana.sh`:
```bash
merge-ana.sh <root file name> <file list>
```
or `hadd` (used in `merge-ana.sh`)
```bash
hadd <foor file name> <ROOT files to merge>
```

Note, that exceeding ~2000 root files will cause the process to crash, so merge files in batches and progressively merge the files.

***(Moving to hdfs storage?)***

---
## Making changes
Note if you need to create new fcl files or c++ code, you need to do fully rebuild the environment i.e. `mrb i -j 16`, if you are modifying an exisitng file then you can just rebuild the changed files with
```
mrbsetenv # once per session
make install -j 16
```
in your build directory.
