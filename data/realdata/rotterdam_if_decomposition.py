"""Break the primary Rotterdam influence function into its main variance components.

We do this for the hormonal-therapy analysis with Cox censoring. The point estimate is unchanged; the goal
is simply to see how much of the estimated variance comes from the event, treatment, covariate, and
censoring/target-estimation pieces.
"""
from pathlib import Path
import numpy as np, pandas as pd
import rotterdam_estimate as R
rows = []
for tau in (5.0, 7.0, 8.0):
    r = R.analyse(R.D0, 'hormon', 'chemo', tau, 'Cox')
    for tilt in ('OW', 'CO c=0.1'):
        c = r[tilt]['components']; n = len(c['X']); tot = sum(c.values())
        row = dict(tau=tau, tilt=tilt, psi=r[tilt]['psi'], se=r[tilt]['se'])
        for k, v in c.items(): row[f'var_{k}'] = np.var(v, ddof=1)/n
        row['sum_components'] = sum(row[f'var_{k}'] for k in c)
        row['var_total'] = np.var(tot, ddof=1)/n
        row['cross_share'] = (row['var_total'] - row['sum_components'])/row['var_total']
        row['mean_IF'] = tot.mean()
        rows.append(row)
out = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(out.round(5).to_string(index=False))
out.to_csv(R.RESULTS/'rotterdam_if_decomposition.csv', index=False)
