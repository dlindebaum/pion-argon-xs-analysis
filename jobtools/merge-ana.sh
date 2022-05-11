#!/bin/bash
#* merges root files in a list produced from an analyser gridjob into a single file
#* usage: ./merge-ana.sh <merged root file name> <file list>

if [ -z "$1" ]
  then
    echo "output root file not specified"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "root files list not specified"
    exit 1
fi

echo "WARNING, CAN TAKE AS WHILE, SIT BACK, RELAX OR JUST NOHUP"
echo "running in 5s"
sleep 5

files=$(pnfs2xrootd `cat $2`)
hadd $1 $files