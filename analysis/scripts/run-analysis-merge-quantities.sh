#!/bin/bash
echo $MY_DIR
# set -x # debugging
set -e # exit if any command exits with non-zero status
POSITIONAL_ARGS=()

FILE=-1
NEVENTS=-1
NJOBS=-1
# CUT_TYPE=-1

# TYPES=("purity" "efficiency" "balanced")

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--file)
      FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--number-of-events)
      NEVENTS="$2"
      shift # past argument
      shift # past value
      ;;
    -j|--number-of-jobs)
      NJOBS="$2"
      shift # past argument
      shift # past value
      ;;
    # -c|--cut-type)
    #   CUT_TYPE="$2"
    #   shift
    #   shift
    #   ;;
    # --default)
    #   DEFAULT=YES
    #   shift # past argument
    #   ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ "$FILE" == "-1" ]
then
    echo "ROOT file must be specified"
    exit 1
fi

if [ $NEVENTS -lt 0 ]
then
    echo "Number of events must be specified"
    exit 1
fi

if [ $NJOBS -lt 0 ]
then
    echo "Number of jobs must be specified"
    exit 1
fi

# COUNTER=0
# for item in $TYPES
# do
#     if [ "$CUT_TYPE" == "$item" ]
#     then
#         ((COUNTER+=1))
#     fi
# done
# if [ $COUNTER -eq 0 ]
# then
#     echo "invalid cut type, must be one of: ${TYPES[*]}"
#     exit 1
# else
#     echo "cut type is $CUT_TYPE"
# fi

function run_job ()
{
    # make directory for job
    # DIR="job_$1"
    # mkdir $DIR
    #*Open a ROOT file and save merge quantities to file:
    python $MY_DIR/apps/prod4a_merge_study.py $FILE -s -d `pwd`/ -e $2 $3 -o merge-quantities-$1
    ls $DIR
    # #*Open a csv file with merge quantities and scan for cut values:
    # python $MY_DIR/apps/prod4a_merge_study.py "$DIR/merge-quantities.csv" -s -d "$DIR/" -c
    # ls $DIR
    # #*Open a ROOT file and csv with list of cuts and merge PFOs based on the cut type:
    # python $MY_DIR/apps/prod4a_merge_study.py $FILE --cuts "$DIR/analysedCuts.csv" --cut-type $CUT_TYPE -m reco -s -o pair_quantities_$1 -d "$DIR/" -a -e $2 $3
    # ls $DIR
    echo "job $1 done!"
}

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "FILE    = ${FILE}"
echo "NEVENTS = ${NEVENTS}"
echo "NJOBS   = ${NJOBS}"

echo "number of events per job: $((NEVENTS/NJOBS))"
echo "remainder: $((NEVENTS%NJOBS))"

for i in $(seq 0 $((NJOBS-1)))
do
    if [ $i -eq $((NJOBS-1)) ]
    then 
        N=$(( (NEVENTS/NJOBS)+(NEVENTS%NJOBS) ))
    else
        N=$((NEVENTS/NJOBS))
    fi
    run_job $i $N $((i*NEVENTS/NJOBS)) &> "out_$i.log" &
    echo $N
done

# #*Open a ROOT file and save merge quantities to file:
# python prod4a_merge_study.py <ROOT file> -s -d <out directory>

# #*Open a csv file with merge quantities and plot them:
# python prod4a_merge_study.py <merge quantity csv> -p <plot type> -s -d <out directory> -n

# #*Open a csv file with merge quantities and scan for cut values:
# python prod4a_merge_study.py <merge quantity csv> -p <plot type> -s -d <out directory> -c

# #*Open a ROOT file and csv with list of cuts and merge PFOs based on the cut type:
# python prod4a_merge_study.py <ROOT file> --cuts <cuts csv> --cut-type <cut type> -m reco -s -o <output filename> -d <out directory> -a

# #*Open a ROOT file, merge PFOs based on truth information and save shower pair quantities to file:
# python prod4a_merge_study.py <ROOT file> -m cheat -s -o <output filename> -d <out directory>