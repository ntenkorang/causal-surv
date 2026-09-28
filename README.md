# Limited Follow-Up Overlap in Causal Survival Analysis

This repository contains the code used for our paper, **Limited Follow-Up Overlap in Causal Survival Analysis**. We use it to reproduce the simulation studies, the Rotterdam breast-cancer analysis, and the numerical checks behind the theoretical results.

Our proposed censoring-adjusted overlap (CO) weight uses both treatment overlap and the probability of remaining under follow-up through the RMST horizon. For a propensity score `e(X)` and follow-up probability `G_a(tau|X)`, we define

```text
h_c(X) = [ kappa_1c(X)/e(X) + kappa_0c(X)/(1-e(X)) ]^(-1)
kappa_ac(X) = 1 + c {1/G_a(tau|X) - 1}.
```

We use `c = 0.1` for the primary analysis and `c = 0.3` and `c = 1` to examine stronger follow-up protection.

## What's in the repository

```text
data/           downloads the Rotterdam data
sims/           runs the simulation studies
realdata/       runs the Rotterdam analysis
theory_checks/ numerical checks for the theoretical results
tables/         creates the manuscript tables
run_all.sh      runs the full analysis
```

Large replicate-level simulation and bootstrap files are not stored here. The scripts generate them when needed.

## Running the code

We recommend Python 3.11 or 3.12. Install the required packages and run

```bash
pip install -r requirements.txt
bash run_all.sh
```

This reproduces the simulations, Rotterdam analysis, theory checks, and manuscript tables. Some simulations and the bootstrap take longer to run, so they can also be run in smaller pieces. For example,

```bash
python sims/co_factorial_final.py 2000 0,1,2
```

runs selected cells of the main simulation study.

## Data

We use the `rotterdam` breast-cancer data from the R package `survival`. We do not store a separate copy of the data in this repository. Instead, `data/fetch_data.py` downloads the dataset from a fixed CRAN mirror and checks the file before the analysis begins.

## Rotterdam analysis

The analysis choices used for the Rotterdam application are recorded in `realdata/rotterdam_protocol.md`.
