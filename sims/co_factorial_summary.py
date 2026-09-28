"""Summarize the main simulation study.

Here we read the replicate-level estimates and compare each estimator with its true target. We report bias,
empirical SD, RMSE, coverage, kurtosis, and a robust tail index. We also compare each tilt with OW through
a paired variance ratio and use resampling to give a 95% Monte Carlo interval for that ratio.

Usage: python co_factorial_summary.py [R]
"""
from pathlib import Path
import sys, glob, numpy as np, pandas as pd
OUT = Path(__file__).resolve().parents[1]/'results'/'sims'
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT/f'factorial_R{R}_cell*_raw.csv')))])
truth = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT/f'factorial_R{R}_cell*_truth.csv')))])
d = raw.merge(truth[['cell', 'tilt', 'psi', 'mass', 'tv', 'cens', 'G1_q05']], on=['cell', 'tilt'])
rng = np.random.default_rng(1); rows = []
for (ci, tilt), g in d.groupby(['cell', 'tilt']):
    g = g.sort_values('rep'); err = (g.psi_hat - g.psi).values; n = len(err); p = g.psi_hat.values
    sd = p.std(ddof=1); k = np.mean((p-p.mean())**4)/p.var()**2
    cover = np.abs(err) <= 1.96*g.se.values; mse = err**2
    ow = d[(d.cell == ci) & (d.tilt == 'OW')].sort_values('rep').psi_hat.values
    boots = [np.var(p[i], ddof=1)/np.var(ow[i], ddof=1) for i in (rng.integers(0, n, n) for _ in range(500))]
    rows.append(dict(cell=ci, tau=g.tau.iloc[0], regime=g.regime.iloc[0], tilt=tilt, R=n,
                     cens=g.cens.iloc[0], G1_q05=g.G1_q05.iloc[0], psi=g.psi.iloc[0], mass=g.mass.iloc[0], tv=g.tv.iloc[0],
                     bias=err.mean(), bias_mcse=err.std(ddof=1)/np.sqrt(n),
                     sd=sd, sd_mcse=sd*np.sqrt(max(k-1, 0)/(4*n)),
                     rmse=np.sqrt(mse.mean()), rmse_mcse=mse.std(ddof=1)/np.sqrt(n)/(2*np.sqrt(mse.mean())),
                     coverage=cover.mean(), coverage_mcse=np.sqrt(cover.mean()*(1-cover.mean())/n),
                     kurt4=k, tail_index=np.quantile(np.abs(err), .99)/np.quantile(np.abs(err), .5),
                     var_ratio_OW=np.var(p, ddof=1)/np.var(ow, ddof=1),
                     var_ratio_lo=np.percentile(boots, 2.5), var_ratio_hi=np.percentile(boots, 97.5)))
out = pd.DataFrame(rows).sort_values(['tau', 'regime', 'tilt'])
out.to_csv(OUT/f'factorial_R{R}_summary.csv', index=False)
keys = ['OW', 'CO c=0.1', 'CO c=0.3', 'CO c=1.0', 'naive e(1-e)G1G0', 'OW trunc pct5']
pd.set_option('display.width', 250)
for (tau, reg), g in out.groupby(['tau', 'regime']):
    g = g.set_index('tilt').loc[[k for k in keys if k in g.tilt.values]]
    print(f"\n=== tau={tau:g} | {reg} | censored {g.cens.iloc[0]:.0%} | G1 q05 {g.G1_q05.iloc[0]:.3f} | R={int(g.R.iloc[0])}")
    for k, r in g.iterrows():
        print(f"  {k:18s} bias {r.bias:+.4f}({r.bias_mcse:.4f})  SD {r.sd:.4f}({r.sd_mcse:.4f})  RMSE {r.rmse:.4f}  "
              f"cover {r.coverage:.3f}({r.coverage_mcse:.3f})  kurt {r.kurt4:6.1f}  TI {r.tail_index:5.2f}  "
              f"VR {r.var_ratio_OW:.2f} [{r.var_ratio_lo:.2f},{r.var_ratio_hi:.2f}]  TV {r.tv:.2f}")
