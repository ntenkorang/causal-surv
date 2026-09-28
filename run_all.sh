#!/usr/bin/env bash
# Reproduce every file in results/ and the LaTeX tables in tables/.
# Run from the repository root. Single-core run times are approximate.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/sims results/realdata/bootstrap results/theory_checks

python data/fetch_data.py

# ---- main simulation study (12 cells x 2000 replicates) ----
python sims/co_factorial_final.py 2000 > results/sims/factorial_R2000_log.txt
python sims/co_factorial_summary.py 2000 > results/sims/factorial_R2000_summary.txt

# ---- auxiliary simulations ----
python sims/co_checks.py const          > results/sims/check_const.txt
python sims/co_checks.py r4             > results/sims/check_r4.txt
python sims/co_alignment_censmodel.py A > results/sims/check_alignment.txt
python sims/co_alignment_censmodel.py B > results/sims/check_censmodel.txt
python sims/co_rare_treatment.py        > results/sims/check_rare_treatment.txt

# ---- Rotterdam application ----
python realdata/rotterdam_followup.py > results/realdata/rotterdam_followup.txt
python realdata/rotterdam_design.py   > results/realdata/rotterdam_design.txt
for spec in "hormon chemo Cox" "hormon chemo KM-year" "chemo hormon Cox" "chemo hormon KM-year"; do
  python realdata/rotterdam_estimate.py $spec > /dev/null
done
python realdata/rotterdam_if_decomposition.py > /dev/null
python realdata/rotterdam_boot.py -1 0 > results/realdata/rotterdam_original_sample.txt   # treated-arm ESS
for s in 0 125 250 375 500 625 750 875; do      # chunks are independent and can run in parallel
  python realdata/rotterdam_boot.py $s $((s+125)) > results/realdata/bootstrap/log_$s.txt 2>&1
done
python realdata/rotterdam_boot_summary.py > results/realdata/rotterdam_bootstrap_summary.txt

# ---- numerical checks of the theory ----
python theory_checks/bounded_support.py    > results/theory_checks/bounded_support.txt
python theory_checks/eif_pathwise_check.py > results/theory_checks/eif_pathwise.txt
python theory_checks/symbolic_audit.py     > results/theory_checks/symbolic_audit.txt
python theory_checks/remainder_check.py    > results/theory_checks/remainder.txt
python theory_checks/if_bound_check.py     > results/theory_checks/if_bound.txt

# ---- LaTeX tables ----
python tables/make_tables.py
