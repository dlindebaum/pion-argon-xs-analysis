# pi0-analysis
Python code for the Pi0 analysis, requires python 3.10.0 or greater.
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

## Run shower merging analysis.
Shower merging analysis workflow is as follows:

---
**selection_studies.py**
 - input:
     - <span style="color: maroon">Ntuple file (root)</span>
 - output:
     - <span style="color: red">basic plots (png)</span>
     - <span style="color: red">tables (tex) of selection efficiency</span>

**generate_geometric_quantities.py**
 - input:
     - <span style="color: maroon">Ntuple file (root)</span>
 - output:
     - <span style="color: magenta">geometric quantities file (csv)</span>

**analyse_geometric_quantities.py**
 - input:
     - <span style="color: magenta">geometric quantities file (csv)</span>
 - output:
    - <span style="color: red">plots of geometric quantities (png)</span>
    - <span style="color: green">list of cuts (csv)</span>

**shower_merging.py**
 - input:
     - <span style="color: maroon">Ntuple file (root)</span>
     - <span style="color: green">list of cuts (csv)</span>
 - output: <span style="color: blue">shower pair quantities (hdf5)</span>

**plotShowerQuantities.py**
 - input:
     - <span style="color: blue">shower pair quantities (hdf5)</span>
 - output:
     - <span style="color: red">various plots of shower quantities (png)</span>
---