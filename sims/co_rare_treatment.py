"""Repeat the alignment experiment when treatment is rare.

Here about 8% of subjects are treated. We use 400 replicates to see how the weighting methods behave when
weak follow-up and limited treatment information occur in the same parts of the covariate space.
"""
import numpy as np, sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import co_alignment_censmodel as C
rng = np.random.default_rng(7); tau = 5.0
for rho in (0.5, 0.75):
    S = {'exp': 'A', 'rho': rho, 'a0': -2.8}; S['lc0'] = C.tune(S); tr, wm, g05 = C.truth_geometry(S, tau)
    est = {k: [] for k in C.KEYS}
    for _ in range(400):
        o = C.estimate(*C.simulate(2000, S, rng), tau, C.CFEATS['linear'], C.TFEATS['correct'])
        for k in C.KEYS: est[k].append(o[k][0])
    p = {k: np.array(v) for k, v in est.items()}; iq = lambda x: np.subtract(*np.percentile(x, [75, 25]))
    print(f"rare treatment (~{np.mean(C.M.expit(-2.8+np.random.default_rng(1).standard_normal(10**5))):.0%}), rho={rho}: OW weak-FU mass {wm:.2f}, G1 q05 {g05:.3f}")
    for k in C.KEYS:
        print(f"   {k:18s} Var/VarOW {p[k].var()/p['OW'].var():5.2f}  IQR^2 ratio {(iq(p[k])/iq(p['OW']))**2:5.2f}  "
              f"kurt {np.mean((p[k]-p[k].mean())**4)/p[k].var()**2:6.1f}  TV {tr[k]['tv']:.2f}")
