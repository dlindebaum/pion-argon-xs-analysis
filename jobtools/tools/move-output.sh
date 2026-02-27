#!/bin/bash
#* Moves files from one directory to another using xrootd, intended to move large root files around.
#* ensure you have enough disk space, and move only what is essential!
#* run as such: ./move-output.sh <file list> <target directory>

if [ -z "$1" ]
  then
    echo "file list not specified"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "target directory not specified"
    exit 1
fi

echo "MAKE SURE YOU KNOW WHAT YOU'RE DOING, STARTING IN 5s"
sleep 5

input="$1"
counter=0
while IFS= read -r line
do
  echo "------------------------------------------"
  FILEPATH=`pnfs2xrootd "$line"`
  echo $FILEPATH
  xrdcp $FILEPATH `pnfs2xrootd $2`
  let counter++
  echo $counter
done < "$input"