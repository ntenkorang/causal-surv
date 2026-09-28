"""Run the subject-level bootstrap for the primary Rotterdam analysis.

START and STOP let us split the bootstrap into smaller batches. Using `-1 0` reruns the original sample,
which is useful for checking that the bootstrap code agrees with the main analysis.

Usage: python rotterdam_boot.py START STOP
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT/'data', ROOT/'results'/'realdata'
import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
import rotterdam_estimate as R

TAUS = (5.0, 7.0, 8.0)
def pipeline(d):
    T = d.dtime.values/365.25; E = d.death.values.astype(int); C = 1-E; A = d.hormon.values.astype(int); n = len(d)
    Z = R.design_matrix(d, 'chemo')
    e = LogisticRegression(C=10, max_iter=10000).fit(Z, A).predict_proba(Z)[:, 1]
    full = np.unique(np.r_[0.0, T[T < max(TAUS)], TAUS])
    HS = {a: R.cox_curves(Z, T, E, A, a, full) for a in (0, 1)}
    HC = {a: R.cox_curves(Z, T, C, A, a, full) for a in (0, 1)}
    res = {}
    for tau in TAUS:
        k = np.searchsorted(full, tau) + 1; grid = full[:k]; K = k; dg = np.diff(grid)
        U = np.minimum(T, tau); jU = np.searchsorted(grid, U); ii = np.arange(n)
        phi, mu, phiG, G = {}, {}, {}, {}
        for a in (0, 1):
            Sa = np.exp(-HS[a][:, :k]); Hc = HC[a][:, :k]; Ga = np.maximum(np.exp(-Hc), 1e-12); G[a] = Ga[:, -1]
            dLC = np.diff(np.c_[np.zeros(n), Hc], axis=1)
            mu[a] = (Sa[:, :-1]*dg).sum(1)
            Rr = np.c_[np.cumsum((Sa[:, :-1]*dg)[:, ::-1], axis=1)[:, ::-1], np.zeros(n)]
            Sm = np.c_[np.ones(n), Sa[:, :-1]]; Gm = np.c_[np.ones(n), Ga[:, :-1]]
            m = grid[None, :] + Rr/np.maximum(Sm, 1e-12)
            atrisk = (np.arange(K)[None, :] <= jU[:, None])
            comp = (atrisk*m*dLC/Gm).sum(1); Gleft = Gm[ii, jU]
            dtau = ((T >= tau) | (E == 1)).astype(float); jumpC = ((E == 0) & (T < tau)).astype(float)
            aug = dtau*U/Gleft + jumpC*m[ii, jU]/Gleft - comp
            ea = e if a else 1-e; Ia = (A == a).astype(float)
            phi[a] = Ia/ea*aug + (1-Ia/ea)*mu[a]
            y = np.maximum(Sm*Gm, 1e-12)
            phiG[a] = -Ga[:, -1]*(jumpC/y[ii, jU] - (atrisk*dLC/y).sum(1))
        Dl = mu[1]-mu[0]
        for name, f in (('OW', lambda e, g1, g0: e*(1-e)),
                        ('CO', lambda e, g1, g0: 1/((1+0.1*(1/g1-1))/e + (1+0.1*(1/g0-1))/(1-e)))):
            G1, G0 = G[1], G[0]; h = f(e, G1, G0); eps = 1e-6
            he = (f(e+eps, G1, G0)-f(e-eps, G1, G0))/(2*eps)
            h1 = (f(e, G1*(1+eps), G0)-f(e, G1*(1-eps), G0))/(2*eps*G1)
            h0 = (f(e, G1, G0*(1+eps))-f(e, G1, G0*(1-eps)))/(2*eps*G0)
            psi = np.sum(h*(phi[1]-phi[0]))/h.sum()
            corr = (Dl-psi)*(he*(A-e) + h1*(A == 1)/e*phiG[1] + h0*(A == 0)/(1-e)*phiG[0])
            psi1 = psi + corr.sum()/h.sum()
            IF = (h*(phi[1]-phi[0]) - h*psi1 + corr)/h.mean()
            w1 = h/e*A
            res[(tau, name)] = (psi1, IF.std(ddof=1)/np.sqrt(n), w1.sum()**2/(w1**2).sum())
    return res

if __name__ == '__main__':
    start, stop = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(20260923); seeds = rng.integers(0, 2**31, size=1000)
    rows = []
    if start == -1:                        # original data, as a check against the saved estimates
        r = pipeline(R.D0)
        for (tau, name), (psi, se, ess1) in r.items():
            print(f'tau={tau:g} {name}: psi {psi:.3f}  se {se:.3f}  treated-arm ESS {ess1:.0f}')
        sys.exit()
    for b in range(start, stop):
        idx = np.random.default_rng(seeds[b]).integers(0, len(R.D0), len(R.D0))
        d = R.D0.iloc[idx].reset_index(drop=True)
        try:
            r = pipeline(d)
            for tau in TAUS:
                rows.append(dict(b=b, tau=tau, psi_OW=r[(tau, 'OW')][0], se_OW=r[(tau, 'OW')][1],
                                 psi_CO=r[(tau, 'CO')][0], se_CO=r[(tau, 'CO')][1],
                                 ess1_OW=r[(tau, 'OW')][2], ess1_CO=r[(tau, 'CO')][2]))
        except Exception as ex:
            print('replicate', b, 'failed:', ex)
    pd.DataFrame(rows).to_csv(RESULTS/'bootstrap'/f'boot_{start:04d}_{stop:04d}.csv', index=False)
