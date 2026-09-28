"""Run the main simulation study.

We cross treatment overlap, follow-up overlap, and tau in {1, 3, 5}, giving 12 settings in total.
The data-generating process and estimators are defined in co_simulation.py so that the same setup is used throughout.
We give each setting its own random seed. This lets us rerun one setting without changing the others.
The script saves replicate-level estimates and the corresponding true targets; co_factorial_summary.py then summarizes them.

Usage: python co_factorial_final.py [R] [cell indices, comma-separated; default all]
"""
from pathlib import Path
import sys, numpy as np, pandas as pd
import co_simulation as M
OUT = Path(__file__).resolve().parents[1]/'results'/'sims'
TAUS = (1.0, 3.0, 5.0)
CELLS = [(tau, reg) for tau in TAUS for reg in M.REGIMES]
TRUNC = [f'OW trunc {l}' for l in ('fix0.05', 'fix0.10', 'pct1', 'pct5')]

def run_cell(ci, R):
    tau, reg = CELLS[ci]; alpha, gamma = M.REGIMES[reg]
    S = {'alpha': alpha, 'gamma': gamma, 'dgp': 'het'}; M.CAP[0] = None
    S['lc0'] = M.tune_lc0(S, 3.0, 0.35); tr, cens, q = M.truths(S, tau)
    rng = np.random.default_rng(2028 + ci); rows = []
    for r in range(R):
        o = M.estimate(*M.simulate(2000, S, rng), tau)
        for k, (p, s) in o.items():
            rows.append((ci, tau, reg, r, k, p, s))
    raw = pd.DataFrame(rows, columns=['cell', 'tau', 'regime', 'rep', 'tilt', 'psi_hat', 'se'])
    truth = pd.DataFrame([dict(cell=ci, tau=tau, regime=reg, tilt=k, psi=v['psi'], mass=v['mass'], tv=v['tv'],
                               cens=cens, G1_q05=q[1][1]) for k, v in tr.items()] +
                         [dict(cell=ci, tau=tau, regime=reg, tilt=k, psi=tr['OW']['psi'], mass=tr['OW']['mass'],
                               tv=0.0, cens=cens, G1_q05=q[1][1]) for k in TRUNC])
    return raw, truth

if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    cells = [int(c) for c in sys.argv[2].split(',')] if len(sys.argv) > 2 else range(len(CELLS))
    for ci in cells:
        raw, truth = run_cell(ci, R)
        raw.to_csv(OUT/f'factorial_R{R}_cell{ci:02d}_raw.csv', index=False)
        truth.to_csv(OUT/f'factorial_R{R}_cell{ci:02d}_truth.csv', index=False)
        print('cell', ci, CELLS[ci], 'done', flush=True)
