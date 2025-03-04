# $1 path of pickle to load
# $2 path to dump pickle

# Change to your conda environment (or similar for pip)
# echo Source miniconda
source /software/wx21978/miniconda/bin/activate
# Source the setup from the RooUnfold build
# echo Source RooUnfold
source /users/wx21978/projects/pion-phys/RooUnfold/build/setup.sh
# Activate the ROOT environment
# echo Activate conda
conda activate root-only

# echo Run python script
# Your path to the unfolding app
python /users/wx21978/projects/pion-phys/pi0-analysis/analysis/apps/roo_unfolding.py -f $1 -o $2
