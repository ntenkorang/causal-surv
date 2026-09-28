"""Describe the follow-up pattern in the Rotterdam cohort.

This script does not estimate a treatment effect. We use it only to understand when follow-up ends across
surgery years. Because the dataset records the year, but not the exact date, of surgery, we approximate
calendar time as surgery year + 0.5 + observed follow-up time.
"""
from pathlib import Path
import pyreadr, numpy as np
DATA = Path(__file__).resolve().parents[1]/'data'

d = pyreadr.read_r(str(DATA/'cancer.rda'))['rotterdam']
d['cal'] = d.year + 0.5 + d.dtime/365.25
c, x = d[d.death == 0], d[d.death == 1]
q10, q50, q90 = c.cal.quantile([.1, .5, .9])
print(f'women {len(d)} | deaths {len(x)} | censored {len(c)}')
print(f'correlation of surgery year with follow-up among censored: {np.corrcoef(c.year, c.dtime)[0, 1]:.2f}')
print(f'calendar end of censored follow-up: median {q50:.1f}, 10th pct {q10:.1f}, 90th pct {q90:.1f}')
print(f'deaths after the 90th pct of censored follow-up: {int((x.cal > q90).sum())}')
for a, b in ((1978, 1982), (1983, 1987), (1988, 1993)):
    g = d[(d.year >= a) & (d.year <= b)]
    print(f'hormonal therapy among women operated on in {a}-{b}: {100*g.hormon.mean():.1f}% (n = {len(g)})')
