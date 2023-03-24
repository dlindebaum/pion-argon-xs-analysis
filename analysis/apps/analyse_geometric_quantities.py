#!/usr/bin/env python3
"""
Created on: 14/03/2023 15:17

Author: Shyam Bhuller

Description: Looks at a geometric_quantities dataframe, produce plots of geometric quantities and/or do the shower selection cut optimisation.
"""
import argparse
import itertools
import os

import awkward as ak
import numpy as np
import pandas as pd
from rich import print

from python.analysis import CutOptimization, shower_merging


def ShowerMergingCriteria(q : shower_merging.ShowerMergeQuantities, outDir : str):
    """ Performs a cut based scan on various criteria that can be used for shower merging

    Args:
        q (ShowerMergeQuantities): quantities to perform cut based scan on
    """
    def Spinner(counter : int, spinner="lines") -> str:
        """ Janky spinner, cause why not?

        Args:
            counter (int): iteration
            spinner (str, optional): type of spinner. Defaults to "lines".

        Returns:
            str: string to print at this interval
        """
        spinners = {
            "lines" : "-\|/",
            "box"   : "⠦⠆⠖⠒⠲⠰⠴⠤",
        }
        return spinners[spinner][counter % len(spinners[spinner])]

    min_val = [] # min range of each variable
    max_val = [] # max range
    for i in range(len(q.selectionVariables)):
        min_val.append(0)
        max_val.append(ak.max(getattr(q, q.selectionVariables[i])))

    min_val = np.array(min_val)
    max_val = np.array(max_val)

    values = np.linspace(min_val+(0.1*max_val), max_val-(0.1*max_val), 3, True) # values that are used to create combinations of cuts to optimize
    metric_labels = ["s", "b", "s/b", "$s\\sqrt{b}$", "purity", "$\\epsilon_{s}$", "$\\epsilon_{b}$", "$\\epsilon$"] # performance metrics to choose cuts #? add purity*efficiency?

    #* create input data strutcure
    counter = 0

    cuts = []
    for v in q.selectionVariables:
        operator = CutOptimization.Operator.GREATER if v == "cnn" else CutOptimization.Operator.LESS
        cuts.append(CutOptimization.Cuts(v, operator, None))

    print("list of cut types:")
    print(cuts)

    output_path = f"{outDir}analysedCuts.csv"

    #* loop through all combination of values for each parameter and optmize the final cut
    for initial_cuts in itertools.product(*values.T):
        for i in range(len(cuts)):
            cuts[i].value = initial_cuts[i]
        cutOptimization = CutOptimization.OptimizeSingleCut(q, cuts, False)
        c, m = cutOptimization.Optimize(10, CutOptimization.MaxSRootBRatio) # scan over 10 bins and optimize cut by looking for max s/sqrt(b)

        o = [c[i] + m[i] for i in range(len(c))] # combine output
        o = pd.DataFrame(o, columns = q.selectionVariables + metric_labels)
        # don't store cuts which exclude all signal PFOs
        o = o[o["s"] > 0]
        o.to_csv(output_path, mode = "a", header = not os.path.exists(output_path))

        counter += 1
        end = '\n' if counter == 0 else '\r'
        print(f" {Spinner(counter, 'box')} progess: {counter/(len(values)**len(initial_cuts))*100:.3f}% | {counter} | {len(values)**len(initial_cuts)}", end=end)

def main(args):
    shower_merging.SetPlotStyle()
    q = shower_merging.ShowerMergeQuantities()
    q.LoadQuantitiesFromCSV(args.file)
    
    if args.cut:
        ShowerMergingCriteria(q, args.out)
    if args.plot:
        q.PlotQuantities(q.signal, q.background, annotate = args.dataset, save = args.save, outDir = args.out)

    return

if __name__ == "__main__":
    example_usage = "fill me in!"
    parser = argparse.ArgumentParser(description = "Calculate geometric quantities of PFOs to be used for the shower merging analysis.", formatter_class = argparse.RawDescriptionHelpFormatter, epilog = example_usage)
    parser.add_argument(dest = "file", type = str, help = "NTuple file to study.")
    parser.add_argument("-e", "--events", dest = "nEvents", type = int, nargs = 2, default = [-1, 0], help = "number of events to analyse and number to skip (-1 is all)")

    parser.add_argument("-p", "--plot", dest = "plot", action = "store_true", help = "make plots")
    parser.add_argument("-c", "--cut", dest = "cut", action = "store_true", help = "generate optimised cuts (warning! this can take a long time)")

    parser.add_argument("-s", "--save", dest = "save", action = "store_true", help = "save plots")
    parser.add_argument("-o", "--out", dest = "out", type = str, default = None, help = "directory to save files")

    args = parser.parse_args()
    args.dataset = None #! temporary, make this an arguement

    if not args.plot and not args.cut:
        raise Exception("Pick either option -p or -c.")

    if args.out is None:
        args.out = args.file.split("/")[-1].split(".")[0] + "/"

    print(vars(args))
    main(args)
