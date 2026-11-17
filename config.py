"""Paths.  The data are read from data/rohe2024 (the Dryad archive of Rohe et al. 2024, unzipped) and
data/rohe2019/BehavioralData.mat (Rohe et al. 2019); both can be overridden with DBCI_DATA and DBCI_DATA_2019."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DBCI_DATA", os.path.join(HERE, "data", "rohe2024"))
DATA_2019 = os.environ.get("DBCI_DATA_2019", os.path.join(HERE, "data", "rohe2019", "BehavioralData.mat"))

OUT = os.environ.get("DBCI_OUT", os.path.join(HERE, "output"))
DERIV = os.path.join(OUT, "derivatives")
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
for _d in (DERIV, RES, FIG):
    os.makedirs(_d, exist_ok=True)

N_JOBS = int(os.environ.get("JOBS", "8"))
