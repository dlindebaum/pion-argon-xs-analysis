#!/cvmfs/larsoft.opensciencegrid.org/products/python/v3_9_2/Linux64bit+3.10-2.17/bin/python3
"""
Created on: 20/04/2022 11:34

Author: Shyam Bhuller

Description: Checks which files have been prestaged, and then stores them into a file so you can run an analysis on them.
Usage: python get_staged.py <samweb definition>
Note: to actually prestage root files use cache_state.py.
"""
import sys
import subprocess
args = sys.argv

#sam_def = "PDSPProd4a_MC_1GeV_reco1_sce_datadriven_v1_02"
sam_def = args[1]

print("running process cache_state.py...")
process = subprocess.Popen(['cache_state.py', '-v', '-d', sam_def], stdout=subprocess.PIPE)

out = open("file_list.txt", 'w')
i = 0
while process.poll() is None:
    line = process.stdout.readline().decode("utf-8")
    if " ONLINE_AND_NEARLINE" in line:
        end = line.find(" ONLINE_AND_NEARLINE")
        start = line.find("/pnfs/")
        xrootDir = subprocess.Popen(["pnfs2xrootd", line[start : end]], stdout=subprocess.PIPE).stdout.readline().decode("utf-8")
        out.writelines(xrootDir)
        i += 1
        print("files added: " + str(i), end='\r')

print("files added: " + str(i))
print("files written to file_list.txt")
