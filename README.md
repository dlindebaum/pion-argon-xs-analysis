# pi0-analysis

LArSoft module which retrieves and calculates some useful quantities to study reconstructed data/MC.

## Helpful links:
[DUNE doxygen page](https://internal.dunescience.org/doxygen/index.html)

[DUNE computing page](https://wiki.dunescience.org/wiki/DUNE_Computing)

[ProtoDUNE SP data/MC list](https://wiki.dunescience.org/wiki/Look_at_ProtoDUNE_SP_data)

---
## Installation
Repository should be cloned in either `src/dunetpc` or `src/protoduneana`. The CMakeLists file that exists in the same directory as the cloned repo should be modified to include this line:
```
add_subdirectory(pi0-analysis)
```

Then, then rebuild the full environment by running `mrb i -j 16` in the build directory. Ensure you are doiing this on either `dunebuild01.fnal.gov` or `dunebuild02.fnal.gov` or else you will be shouted at.

## Usage
if the environment successfully builds you can run the LArsoft analyser by running one of these fcl files:
```
runDiphoton.fcl
runPi0_BeamSim.fcl
runPi0.fcl
runPi0_noFilter.fcl (same as runPi0_noFilter.fcl should be removed)
runPi0Test.fcl
```
e.g.
```
lar -c runPi0_BeamSim.fcl <file list or root file>
```

***running grid jobs tbd.***

## Making changes
Note if you need to create new fcl files or c++ code, you need to do fully rebuild the environment i.e. `mrb i -j 16`, if you are modifying an exisitng file then you can just rebuild the changed files with
```
mrbsetenv # once per session
make install -j 16
```
in your build directory.

