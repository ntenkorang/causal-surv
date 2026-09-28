"""Run the smaller simulation checks used to understand the main results.

We use `const` when we want every tilt to have the same estimand, so differences are purely about precision.
We use `r4` to see what happens when the censoring model misses weak follow-up in part of the covariate space.
The `n500` option repeats the main design with a smaller sample size.

Usage: python co_checks.py {const|r4|n500}
"""
import sys, co_simulation as M
which = sys.argv[1]
K = ['OW', 'naive e(1-e)G1G0', 'CO c=0.1', 'CO c=0.3', 'CO c=1.0',
     'OW trunc fix0.05', 'OW trunc fix0.10', 'OW trunc pct1', 'OW trunc pct5']
if which == 'const':
    for tau in (3.0, 5.0):
        for reg in ('good trt / poor FU', 'poor trt / poor FU'):
            M.show(f"CONST target | tau={tau:g} | {reg}", *M.run(2000, 500, tau, reg, dgp='const'), keys=K)
if which == 'r4':
    import numpy as np
    for cap in (None, 2.0, 1.5, 1.0, 0.5):
        mass = 0 if cap is None else float(np.mean(np.abs(np.random.default_rng(0).standard_normal(10**6)) < 3.5) and
                                           np.mean(np.random.default_rng(0).standard_normal(10**6) > cap))
        M.show(f"R4 | censoring model flat for X2>{cap} (mass {mass:.3f}) | tau=5 | good trt / poor FU",
               *M.run(2000, 500, 5.0, 'good trt / poor FU', cap=cap), keys=['OW', 'CO c=0.1', 'CO c=0.3', 'CO c=1.0'])
if which == 'n500':
    for tau in (3.0, 5.0):
        for reg in M.REGIMES:
            M.show(f"n=500 | tau={tau:g} | {reg}", *M.run(500, 500, tau, reg), keys=K)
