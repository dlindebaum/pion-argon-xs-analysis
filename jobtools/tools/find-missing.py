#!/cvmfs/larsoft.opensciencegrid.org/products/python/v3_9_2/Linux64bit+3.10-2.17/bin/python3
"""
Created on: 13/06/2022 12:47

Author: Shyam Bhuller

Description: Retrieves a list of ROOT files which processed with errors in a grid job.
"""
import argparse

def main(args):
    out_file = open(args.out_file) # open output file list

    file_list = open(args.file_list) # open input file list
    l = file_list.readlines()
    file_list.close()

    # read output file, get the process number i.e job number
    num = []
    for line in out_file:
        line = line.split("_")[-2]
        num.append(int(line))
    out_file.close()

    # check for missing process numbers
    num.sort()
    missing = list(set(range(len(l))).difference(num))
    print(missing)

    missing_file_list = open(args.output_name, "w") # create missing file list
    for m in missing:
        missing_file_list.write(l[m]) # get the input files which were not sucessfully processed
    missing_file_list.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a list of root files not processed by grid job cluster")
    parser.add_argument(dest="out_file", type=str, help="list of root files produced in grid job output")
    parser.add_argument(dest="file_list", type=str, help="input file list used in grid job cluster")
    parser.add_argument("-o", "--output-name", dest="output_name", type=str, default="missing.txt", help="name of output file produced")
    args = parser.parse_args()
    main(args)