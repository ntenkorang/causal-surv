"""Summarize the Rotterdam bootstrap.

We combine the usable bootstrap replicates, calculate the reported uncertainty summaries, and save the
compact results used in the supplement. We also keep the replicate-level values needed for bootstrap
diagnostics.
"""
from pathlib import Path
import glob, re
import numpy as np, pandas as pd
RES = Path(__file__).resolve().parents[1]/'results'/'realdata'

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES/'bootstrap'/'boot_*.csv')))], ignore_index=True)
fails = sorted({int(m) for f in glob.glob(str(RES/'bootstrap'/'log_*.txt'))
                for m in re.findall(r'replicate (\d+) failed', open(f).read())})
df['R'] = df.se_CO**2/df.se_OW**2
df['ESSr'] = df.ess1_CO/df.ess1_OW
ok = df.dropna(subset=['R', 'ESSr'])
print(f'attempted 1000 | Cox non-convergence {len(fails)} | missing estimates {df.b.nunique()-ok.b.nunique()} | usable {ok.b.nunique()}')
rows = []
for tau, g in ok.groupby('tau'):
    q = lambda x: np.percentile(x, [2.5, 50, 97.5])
    r, e = q(g.R), q(g.ESSr)
    kurt = lambda x: np.mean((x-x.mean())**4)/x.var()**2
    rows.append(dict(tau=tau, n=len(g), R_median=r[1], R_lo=r[0], R_hi=r[2], P_R_gt_1=np.mean(g.R > 1),
                     ESSr_median=e[1], ESSr_lo=e[0], ESSr_hi=e[2],
                     boot_var_ratio=g.psi_CO.var()/g.psi_OW.var(), kurt_OW=kurt(g.psi_OW), kurt_CO=kurt(g.psi_CO),
                     P_CO_gt_OW=np.mean(g.psi_CO > g.psi_OW)))
out = pd.DataFrame(rows)
print(out.round(3).to_string(index=False))
out.to_csv(RES/'rotterdam_bootstrap_summary.csv', index=False)
df.to_csv(RES/'rotterdam_bootstrap_replicates.csv', index=False)

# We also check whether failed bootstrap samples look systematically different from usable ones.
# To make this comparison fair, we rebuild each sample from the same seed used in rotterdam_boot.py.
import pyreadr
d = pyreadr.read_r(str(RES.parents[1]/'data'/'cancer.rda'))['rotterdam']
A, late = d.hormon.values, ((d.hormon.values == 1) & (d.year.values >= 1992))
seeds = np.random.default_rng(20260923).integers(0, 2**31, size=1000)
idx = [np.random.default_rng(s).integers(0, len(d), len(d)) for s in seeds]
ntr = np.array([A[i].sum() for i in idx]); nlate = np.array([late[i].sum() for i in idx])
failed = np.isin(np.arange(1000), fails); usable = np.isin(np.arange(1000), ok.b.unique())
print(f'hormonally treated per resample, median: failed {np.median(ntr[failed]):.0f} | usable {np.median(ntr[usable]):.0f}')
print(f'treated and operated on 1992-1993, median: failed {np.median(nlate[failed]):.0f} | usable {np.median(nlate[usable]):.0f}')
