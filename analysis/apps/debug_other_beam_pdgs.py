import awkward as ak
import numpy as np
from python.analysis import Master

# Pick ONE MC file (enough to diagnose "other")
file = "/storage/0/wx21978/phys/6GeV/mc/PDSPProd4a_MC_6GeV_Set00_reco1_sce_datadriven_v1_ntuple_v09_44_00_02.root"

events = Master.Data(
    file,
    -1,              # all events
    0,               # start
    "PDSPAnalyser",  # matches config
    6                # beam momentum
)

pdg = events.trueParticlesBT.beam_pdg
cosmic = events.trueParticlesBT.beam_origin == 2

tagged = (
    (pdg == 211) |   # pi+
    (pdg == -13) |   # mu+
    (pdg == -11) |   # e+
    (pdg == 2212) |  # proton
    (pdg == 321)     # K+
)

other = (~tagged) & (~cosmic)

vals = ak.to_numpy(pdg[other])

u, c = np.unique(vals, return_counts=True)

print("\nTop contributors to 'other':\n")
for p, n in sorted(zip(u, c), key=lambda x: x[1], reverse=True)[:30]:
    print(f"PDG {int(p):>6} : {int(n)}")

