# pi0-analysis
Python code for the Pi0 analysis, requires python 3.6.8 or greater. In additional to standard libraries like numpy and matplotlib, ensure you have uproot 4.1.9 and awkward 1.7.0 or later installed in your python environment.

Code runs on ntuples produced by the Pi0 Analyser module (to be added), a list of produced ntuples are here:

[https://cernbox.cern.ch/index.php/s/8UqObev6XPNhRXn](https://cernbox.cern.ch/index.php/s/8UqObev6XPNhRXn)

Core modules are Master.py, vector.py and Plots.py (optional). A simple example of how to look at true data is shown in truth_info.py, and the other scripts are more complicated examples of how to analyse nTuples.

The example scripts provided can be run on the command line, so you can run the following with the ntuple:

``` bash
python truth_info.py pi0_0p5GeV_100K_5_7_21.root -s -d pi0_0p5GeV/ 
```
and for information on command line options:
``` bash
python truth_info.py -h
```

For further detail on each module you can read the docstrings.
