#!/bin/bash
#* split text file by number of lines and then rename them add txt extension.

split $1 -l $2

for i in $(ls xa*)
do
    mv $i $i.txt
done