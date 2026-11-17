# Dynamic causal and numerosity priors in multisensory perception: analysis code

Code for every result in the paper and its supporting information. The paper reanalyses the public data of
Rohe, Ehlis & Noppeney (2019, Nat Commun) and Rohe, Hesse, Ehlis & Noppeney (2024, PLoS Biol): 23 controls and
17 patients with schizophrenia (the 6 participants with schizoaffective disorder are not analysed).

## Data

- `data/rohe2024/`: the Dryad archive of Rohe et al. (2024), doi:10.5061/dryad.hhmgqnkr1, unzipped
  (behavioural file, clinical file and raw EEG).
- `data/rohe2019/BehavioralData.mat`: behavioural file of Rohe et al. (2019). It is only used to restore the deleted
  trials of two control runs whose EEG marker files are incomplete; the control data of the two studies are identical.

Other locations can be given with `DBCI_DATA` and `DBCI_DATA_2019`. Results are written to `output/`
(or `DBCI_OUT`).

## Running

Python 3.9 with the packages in `requirements.txt`. Run the scripts from this folder, in order:

| Script | | Used for |
|---|---|---|
| `analysis/01_eeg_preprocess.py` | marker alignment, filtering, ICA, epochs | S1 Text |
| `analysis/02_eeg_features.py` | prestimulus band power per trial | Fig 1D, S3 Table |
| `analysis/03_fit.py` | all behavioural models | Fig 1A, S1 Table |
| `analysis/04_simulate.py` | history effect and conditional tests, data and simulations | Fig 1B-C, S2 Table |
| `analysis/05_recovery.py` | parameter recovery | S1 Fig |
| `analysis/06_pull.py` | pull towards the previous report | S2 Fig |
| `analysis/07_brain_state.py` | prestimulus alpha and the causal / numerosity prior | Fig 1D, S3 Table |
| `analysis/08_channel_maps.py` | alpha weight at each sensor | Fig 1D |
| `analysis/09_encoding.py` | prestimulus power and the latent priors | S1 Text |
| `analysis/10_stats.py` | Tables S1-S3 and the other numbers in the text | |
| `figures/fig1.py`, `figS1.py`, `figS2.py` | figures | |

`JOBS` sets the number of parallel processes (default 8; `EEG_JOBS` for the two EEG steps). The per-participant
steps write one file per participant and skip participants that are already done; they also accept participant
numbers as arguments. The model fits, simulations and brain-state fits take several hours in total on a laptop.

## Code

- `lib/bci.py`: Bayesian causal inference likelihood (numerical integration over the internal samples), static,
  delta-rule and Markov causal priors, fitting routines.
- `lib/models.py`: all models as one trial-by-trial filter (dynamic numerosity prior, response repetition and their
  combinations), and simulation.
- `lib/state.py`: the same filter with a trial-wise shift of the causal or numerosity prior by prestimulus alpha.
- `lib/history.py`: the history-effect analysis of Rohe et al. (2019).
- `lib/eeg.py`: time-frequency settings, trial selection, regression design and cluster tests.
- `lib/data.py`: data loading; restores the trials without a response that were deleted from the 2024 file.
- `lib/style.py`: figure style.
