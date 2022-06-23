#!/cvmfs/larsoft.opensciencegrid.org/products/python/v3_9_2/Linux64bit+3.10-2.17/bin/python3
import subprocess
import argparse
import configparser
#bashCommand = "jobsub_submit -G dune -M -N 10 --memory=2800MB --disk=3GB --expected-lifetime=3h --cpu=1 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC,OFFSITE --tar_file_name=dropbox:///dune/app/users/sbhuller/dunesw/dunesw-test.tar.gz -l '+SingularityImage=\"/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl7:latest\"' --lines '+FERMIHTC_AutoRelease=True' --lines '+FERMIHTC_GraceMemory=1024' --lines '+FERMIHTC_GraceLifetime=1800' --append_condor_requirements='(TARGET.HAS_Singularity==true&&TARGET.HAS_CVMFS_dune_opensciencegrid_org==true&&TARGET.HAS_CVMFS_larsoft_opensciencegrid_org==true&&TARGET.CVMFS_dune_opensciencegrid_org_REVISION>=1105&&TARGET.HAS_CVMFS_fifeuser1_opensciencegrid_org==true&&TARGET.HAS_CVMFS_fifeuser2_opensciencegrid_org==true&&TARGET.HAS_CVMFS_fifeuser3_opensciencegrid_org==true&&TARGET.HAS_CVMFS_fifeuser4_opensciencegrid_org==true)' file:///dune/app/users/sbhuller/dunesw/job/run_test_multi_sbhuller.sh"

class Options:
    def __init__(self):
        self.numberOfJobs = 1
        self.memory = "2800MB"
        self.disk = "10GB"
        self.lifetime = "8h"
        self.cpu = 1
        self.tarball = ""
        self.fhicl = ""
        self.outputDirectory = ""
        self.fileList = ""
    def ReadConfig(self, config):
        for var in vars(self):
            if var in config["SETTINGS"]:
                print(f"found {var}")
                setattr(self, var, config["SETTINGS"][var])
    def ReadArgs(self, args):
        for var in vars(self):
            setattr(self, var, getattr(args, var))
        

def ConfigureScript(fcl : str, outdir : str, file_list : str):
    bash_script="""\
#!/bin/bash

FILE_LIST="/pnfs/dune/resilient/users/sbhuller/missing.txt"
FHICL="runPi0_BeamSim.fcl"
OUTDIR_NAME="PDSPProd4a_MC_6GeV_reco1_sce_datadriven_v1_02/missing"

echo "Running on $(hostname) at ${GLIDEIN_Site}. GLIDEIN_DUNESite = ${GLIDEIN_DUNESite}"

# set the output location for copyback
OUTDIR=/pnfs/dune/scratch/users/${GRID_USER}/${OUTDIR_NAME}/
echo "Output directoty is ${OUTDIR}"

#Let's rename the output file so it's unique in case we send multiple jobs.
OUTFILE=pi0Test_output_${CLUSTER}_${PROCESS}_$(date -u +%Y%m%dT%H%M%SZ).root
STDOUT=out_${CLUSTER}_${PROCESS}.log

#make sure we see what we expect
pwd

ls -l $CONDOR_DIR_INPUT

if [ -e ${INPUT_TAR_DIR_LOCAL}/setup-jobenv.sh ]; then
    . ${INPUT_TAR_DIR_LOCAL}/setup-jobenv.sh
else
echo "Error, setup script not found. Exiting."
exit 1
fi

# cd back to the top-level directory since we know that's writable
cd ${_CONDOR_SETTINGS_IWD}

#symlink the desired fcl to the current directory
ln -s ${INPUT_TAR_DIR_LOCAL}/localProducts*/protoduneana/*/job/${FHICL} .

# set some other very useful environment variables for xrootd and IFDH
export IFDH_CP_MAXRETRIES=2
export XRD_CONNECTIONRETRY=32
export XRD_REQUESTTIMEOUT=14400
export XRD_REDIRECTLIMIT=255
export XRD_LOADBALANCERTTL=7200
export XRD_STREAMTIMEOUT=14400 # many vary for your job/file type


# make sure the output directory exists
ifdh ls $OUTDIR 0 # set recursion depth to 0 since we are only checking for the directory; we don't care about the full listing.

if [ $? -ne 0 ]; then
    # if ifdh ls failed, try to make the directory
    ifdh mkdir_p $OUTDIR || { echo "Error creating or checking $OUTDIR"; exit 2; }
    ifdh mkdir_p $OUTDIR/out/  || { echo "Error creating or checking $OUTDIR/out/"; exit 2; }
fi

LIST_NAME=`basename ${FILE_LIST}`
ifdh cp ${FILE_LIST} ${LIST_NAME} || { echo "Error copying ${FILE_LIST}"; exit 3; }

#now we should be in the work dir if setupMay2021Tutorial-grid.sh worked
FILE_NAME=`sed -n "$((PROCESS+1))p" < ${LIST_NAME}`
echo "ROOT file name:"
echo $FILE_NAME

mkfifo pipe
tee $STDOUT < pipe &
lar -c ${FHICL} ${FILE_NAME} > pipe
LAR_RESULT=$?   # ALWAYS keep track of the exit status or your main command!!!

if [ -f $STDOUT ]; then
    ifdh cp -D $STDOUT $OUTDIR/out/

    #check the exit status to see if the copyback actually worked. Print a message if it did not.
    IFDH_RESULT=$?
    if [ $IFDH_RESULT -ne 0 ]; then
    echo "Error during output copyback. See output logs."
    exit $IFDH_RESULT
    fi
fi

if [ $LAR_RESULT -ne 0 ]; then
    echo "lar exited with abnormal status $LAR_RESULT. See error outputs."
    exit $LAR_RESULT
fi


if [ -f pi0Test_output.root ]; then

    mv pi0Test_output.root $OUTFILE
    
    #and copy our output file back
    ifdh cp -D $OUTFILE $OUTDIR

    #check the exit status to see if the copyback actually worked. Print a message if it did not.
    IFDH_RESULT=$?
    if [ $IFDH_RESULT -ne 0 ]; then
    echo "Error during output copyback. See output logs."
    exit $IFDH_RESULT
    fi
fi

#If we got this far, we succeeded.
echo "Completed successfully."
exit 0
"""
    lines = bash_script.splitlines()
    for i in range(len(lines)):
        if "FHICL=" in lines[i] and fcl != "":
            print("fhicl file variable found")
            lines[i] = f"FHICL=\"{fcl}\""
            continue
        if "OUTDIR_NAME=" in lines[i] and outdir != "":
            print("nEvents variable found")
            lines[i] = f"OUTDIR_NAME=\"{outdir}\""
            continue
        if "FILE_LIST=" in lines[i] and file_list != "":
            print("file list name variable found")
            lines[i] = f"FILE_LIST=\"{file_list}\""
            continue

    f = open("/tmp/job_script.sh", "w")
    f.write("\n".join(lines))
    f.close()
    return


def main(options, debug=False):
    # nJobs = 10
    # memory = "2800MB"
    # disk = "3GB"
    # lifetime = "3h"
    # cpu = 1
    # tar_file = "/dune/app/users/sbhuller/dunesw/dunesw-test.tar.gz"
    # script = "/dune/app/users/sbhuller/dunesw/job/run_test_multi_sbhuller.sh"

    ConfigureScript(options.fhicl, options.outputDirectory, options.fileList)

    bashCommand = "jobsub_submit --mail_never -G dune --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC,OFFSITE -l '+SingularityImage=\"/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl7:latest\"' --lines '+FERMIHTC_AutoRelease=True' --lines '+FERMIHTC_GraceMemory=1024' --lines '+FERMIHTC_GraceLifetime=1800' --append_condor_requirements='(TARGET.HAS_Singularity==true&&TARGET.HAS_CVMFS_dune_opensciencegrid_org==true&&TARGET.HAS_CVMFS_larsoft_opensciencegrid_org==true&&TARGET.CVMFS_dune_opensciencegrid_org_REVISION>=1105&&TARGET.HAS_CVMFS_fifeuser1_opensciencegrid_org==true&&TARGET.HAS_CVMFS_fifeuser2_opensciencegrid_org==true&&TARGET.HAS_CVMFS_fifeuser3_opensciencegrid_org==true&&TARGET.HAS_CVMFS_fifeuser4_opensciencegrid_org==true)' "

    bashCommand += f"-N {options.numberOfJobs} " # 1
    bashCommand += f"--memory={options.memory} " # 2800MB
    bashCommand += f"--disk={options.disk} " # 10GB
    bashCommand += f"--expected-lifetime={options.lifetime} " # 3h
    bashCommand += f"--cpu={options.cpu} " # 1
    bashCommand += f"--tar_file_name=dropbox://{options.tarball} "
    bashCommand += f"file:///tmp/job_script.sh"

    print(bashCommand.split())
    if debug is False:
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if output: print(output.decode("utf-8"))
        if error: print(error.decode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit job using jobsub_submit (but nicer)")
    parser.add_argument("-s", "--settings", dest="settings", type=str, default=None, help="settings file so you don't need to pass arguements through the command line")
    parser.add_argument("-n", "--number-of-jobs", dest="numberOfJobs", type=int, default=1, help="number of jobs to submit")
    parser.add_argument("-m", "--memory-usage", dest="memory", type=str, default="2800MB", help="memory used per job")
    parser.add_argument("-d", "--disk-uasge", dest="disk", type=str, default="10GB", help="disk space used per job")
    parser.add_argument("-l", "--lifetime", dest="lifetime", type=str, default="3h", help="job lifetime")
    parser.add_argument("-c", "--cpu-uasge", dest="cpu", type=int, default=1, help="number of threads used")
    parser.add_argument("-t", "--tarball", dest="tarball", type=str, default="", help="custom tarball to use")
    parser.add_argument("-f", "--fhicl", dest="fhicl", type=str, default="", help="fhicl file to run")
    parser.add_argument("-o", "--outdir-name", dest="outputDirectory", type=str, default="output", help="name of output directory (will automatically create in scratch directory)")
    parser.add_argument("-i", "--input-file-list", dest="fileList", type=str, default="file_list.txt", help="file list name (should be in the tarfile!)")
    parser.add_argument("--debug", dest="debug", action="store_true", help="debug code without sumbitting a job")
    args = parser.parse_args()
    options = Options()
    options.ReadArgs(args)
    if args.settings:
        print("we have a configuation file!")
        config = configparser.ConfigParser()
        config.read(args.settings)
        options.ReadConfig(config)
    main(options, args.debug)