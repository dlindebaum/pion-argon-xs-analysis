#!/bin/bash

# we cannot rely on "whoami" in a grid job. We have no idea what the local username will be. 
# Use the GRID_USER environment variable instead (set automatically by jobsub).
USERNAME=${GRID_USER}    
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
export WORKDIR=${_CONDOR_JOB_IWD} # if we use the RCDS the our tarball will be placed in $INPUT_TAR_DIR_LOCAL.
if [ ! -d "$WORKDIR" ]; then
  export WORKDIR=`echo .`
fi

# if you are using an older release you MAY get an error of this form:

# ERROR:
# ERROR: this mrb area expects mrb < v5_00_00 (found some version > v5)!
# ERROR:

# If you do, the workaround is the following (comment these lines out if not needed)
#unsetup mrb
#setup mrb -o


source ${INPUT_TAR_DIR_LOCAL}/localProducts*/setup-grid

mrbslp
