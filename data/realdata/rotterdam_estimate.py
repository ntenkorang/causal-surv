"""Estimate the RMST contrasts for the Rotterdam application.

We follow the analysis choices recorded in rotterdam_protocol.md. The last argument chooses either the
primary arm-specific Cox censoring model or the year-stratified Kaplan--Meier sensitivity model.

Usage: python rotterdam_estimate.py TREATMENT OTHER_THERAPY {Cox|KM-year}
Example: python rotterdam_estimate.py hormon chemo Cox
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT/'data', ROOT/'results'/'realdata'
import pyreadr, numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from lifelines import CoxPHFitter

D0 = pyreadr.read_r(str(DATA/'cancer.rda'))['rotterdam'].copy().reset_index(drop=True)

def design_matrix(d, other):
    sy = SplineTransformer(n_knots=4, degree=3).fit_transform(d[['year']].astype(float))
    sa = SplineTransformer(n_knots=4, degree=3).fit_transform(d[['age']].astype(float))
    X = pd.DataFrame(np.column_stack([sy, sa]), columns=[f'sy{i}' for i in range(sy.shape[1])] + [f'sa{i}' for i in range(sa.shape[1])])
    X['meno'] = d.meno.astype(float).values; X['grade3'] = (d.grade == 3).astype(float).values
    X['size2'] = (d['size'].astype(str) == '20-50').astype(float).values
    X['size3'] = (d['size'].astype(str) == '>50').astype(float).values
    for v in ('nodes', 'pgr', 'er'): X['l'+v] = np.log1p(d[v].astype(float)).values
    X[other] = d[other].astype(float).values
    Z = (X - X.mean())/X.std().replace(0, 1)
    return Z.loc[:, Z.std() > 0]

def cox_curves(Z, T, E, A, a, grid):
    i = A == a; dd = Z[i].copy(); dd['T'] = T[i]; dd['E'] = E[i]
    f = CoxPHFitter(penalizer=0.1).fit(dd, 'T', 'E')
    H = f.predict_cumulative_hazard(Z, times=grid).values.T          # n x K
    return np.maximum.accumulate(H, axis=1)

def km_year_cens(d, T, C, grid):
    """Estimate censoring within surgery-year strata and return the cumulative hazard for each subject."""
    H = np.zeros((len(d), len(grid)))
    for y, idx in d.groupby('year').groups.items():
        idx = np.asarray(idx); t, c = T[idx], C[idx]
        dl = np.array([np.sum((t == u) & (c == 1))/max(np.sum(t >= u), 1) for u in grid])
        H[idx, :] = np.cumsum(dl)[None, :]
    return H

def tilts(e, G1, G0):
    H = {'IPW': lambda e, g1, g0: np.ones_like(e), 'OW': lambda e, g1, g0: e*(1-e),
         'naive': lambda e, g1, g0: e*(1-e)*g1*g0}
    for c in (0.1, 0.3, 1.0):
        H[f'CO c={c}'] = (lambda c: lambda e, g1, g0: 1/((1+c*(1/g1-1))/e + (1+c*(1/g0-1))/(1-e)))(c)
    return H

def analyse(d, trt, other, tau, cens_model):
    T = d.dtime.values/365.25; E = d.death.values.astype(int); C = 1-E
    A = d[trt].values.astype(int); n = len(d)
    Z = design_matrix(d, other)
    e = LogisticRegression(C=10, max_iter=10000).fit(Z, A).predict_proba(Z)[:, 1]
    grid = np.unique(np.r_[0.0, T[T < tau], tau])                   # g_0 = 0 ... g_K = tau
    K = len(grid); dg = np.diff(grid)
    U = np.minimum(T, tau); jU = np.searchsorted(grid, U)             # index of U on grid
    S, G, dLC, Hc = {}, {}, {}, {}
    for a in (0, 1):
        S[a] = np.exp(-cox_curves(Z, T, E, A, a, grid))
        if cens_model == 'Cox':
            Hc[a] = cox_curves(Z, T, C, A, a, grid)
    if cens_model == 'KM-year':
        Hc[0] = Hc[1] = km_year_cens(d, T, C, grid)
    for a in (0, 1):
        G[a] = np.exp(-Hc[a]); dLC[a] = np.diff(np.c_[np.zeros(n), Hc[a]], axis=1)
    phi, mu, phiG = {}, {}, {}
    for a in (0, 1):
        Sa, Ga = S[a], np.maximum(G[a], 1e-12)
        mu[a] = (Sa[:, :-1]*dg).sum(1)
        R = np.c_[np.cumsum((Sa[:, :-1]*dg)[:, ::-1], axis=1)[:, ::-1], np.zeros(n)]   # int_{g_j}^tau S
        Sm = np.c_[np.ones(n), Sa[:, :-1]]; Gm = np.c_[np.ones(n), Ga[:, :-1]]           # left limits
        m = grid[None, :] + R/np.maximum(Sm, 1e-12)
        atrisk = (np.arange(K)[None, :] <= jU[:, None])
        comp = (atrisk*m*dLC[a]/Gm).sum(1)
        ii = np.arange(n); Gleft = Gm[ii, jU]
        dtau = ((T >= tau) | (E == 1)).astype(float)
        jumpC = ((E == 0) & (T < tau)).astype(float)
        aug = dtau*U/Gleft + jumpC*m[ii, jU]/Gleft - comp
        ea = e if a == 1 else 1-e; Ia = (A == a).astype(float)
        phi[a] = Ia/ea*aug + (1-Ia/ea)*mu[a]
        # Here we calculate the censoring part of the gradient of G_a(tau|x).
        y = np.maximum(Sm*Gm, 1e-12)
        if cens_model == 'KM-year':      # stratum-level at-risk fraction, pooled arms
            y = np.zeros((n, K))
            for yr, idx in d.groupby('year').groups.items():
                idx = np.asarray(idx); y[idx, :] = np.maximum((T[idx][:, None] >= grid[None, :]).mean(0), 1e-12)
        phiG[a] = -Ga[:, -1]*(jumpC/y[ii, jU] - (atrisk*dLC[a]/y).sum(1))
    G1, G0 = G[1][:, -1], G[0][:, -1]
    Dl = mu[1]-mu[0]; out = {}
    hOW = e*(1-e); dOW = hOW/hOW.sum()
    for name, f in tilts(e, G1, G0).items():
        h = f(e, G1, G0); eps = 1e-6
        he = (f(e+eps, G1, G0)-f(e-eps, G1, G0))/(2*eps)
        h1 = (f(e, G1*(1+eps), G0)-f(e, G1*(1-eps), G0))/(2*eps*G1)
        h0 = (f(e, G1, G0*(1+eps))-f(e, G1, G0*(1-eps)))/(2*eps*G0)
        psi = np.sum(h*(phi[1]-phi[0]))/h.sum()
        if cens_model == 'Cox':
            corrG = (Dl-psi)*(h1*(A == 1)/e*phiG[1] + h0*(A == 0)/(1-e)*phiG[0])
        else:                             # pooled year-level G: weight = stratum mean of (Dl-psi)(h1+h0)
            w = pd.Series((Dl-psi)*(h1+h0)).groupby(d.year.values).transform('mean').values
            corrG = w*phiG[1]
        corr = (Dl-psi)*he*(A-e) + corrG
        psi1 = psi + corr.sum()/h.sum()
        IF = (h*(phi[1]-phi[0]) - h*psi1 + corr)/h.mean()
        se = IF.std(ddof=1)/np.sqrt(n)
        IFk = (h*(phi[1]-phi[0]) - h*psi1)/h.mean()          # known-tilt part only
        comps = {'X': h*(Dl-psi1)/h.mean(), 'T': h*(phi[1]-phi[0]-Dl)/h.mean(),
                 'A': (Dl-psi)*he*(A-e)/h.mean(), 'C': corrG/h.mean()}
        top = np.sort(IF**2)[::-1]
        out[name] = dict(components=comps, se_known_tilt=IFk.std(ddof=1)/np.sqrt(n), kurt_IF=np.mean((IF-IF.mean())**4)/IF.var()**2,
                         top10_share=top[:10].sum()/top.sum(), max_absIF=np.abs(IF).max(),
                         psi=psi1, se=se, lo=psi1-1.96*se, hi=psi1+1.96*se,
                         mass=h.mean()/hOW.mean(), TV=0.5*np.abs(h/h.sum()-dOW).sum(),
                         ESS=h.sum()**2/(h**2).sum())
    return out

if __name__ == '__main__':
    trt, other, cens = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = []
    for tau in (5.0, 7.0, 8.0):
        r = analyse(D0, trt, other, tau, cens)
        for k, v in r.items():
            v = {kk: vv for kk, vv in v.items() if kk != 'components'}
            rows.append(dict(treatment=trt, censoring=cens, tau=tau, tilt=k, **v))
    df = pd.DataFrame(rows); print(df.round(3).to_string(index=False))
    df.to_csv(RESULTS/f'rotterdam_est_{trt}_{cens}.csv', index=False)
