"""Set up the design stage of the Rotterdam analysis without using outcome estimates.

Here we define treatment, the RMST horizons, the weighting rules, and the nuisance models before looking
at treatment-effect estimates. Hormonal therapy is the primary treatment and chemotherapy is the secondary
analysis. We use tau = 5, 7, and 8 years; tau = 10 is excluded because follow-up support becomes too weak.
The primary censoring model is arm-specific Cox regression, and a surgery-year Kaplan--Meier model is used
as a sensitivity analysis. See rotterdam_protocol.md for the full frozen analysis plan.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT/'data', ROOT/'results'/'realdata'
import pyreadr, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from lifelines import CoxPHFitter

d = pyreadr.read_r(str(DATA/'cancer.rda'))['rotterdam'].copy().reset_index(drop=True)
T = d.dtime.values/365.25; E = d.death.values.astype(int)

def design_matrix(d, other):
    sp_y = SplineTransformer(n_knots=4, degree=3).fit_transform(d[['year']].astype(float))
    sp_a = SplineTransformer(n_knots=4, degree=3).fit_transform(d[['age']].astype(float))
    X = pd.DataFrame(np.column_stack([sp_y, sp_a]), columns=[f'sy{i}' for i in range(sp_y.shape[1])] + [f'sa{i}' for i in range(sp_a.shape[1])])
    X['meno'] = d.meno.astype(float); X['grade3'] = (d.grade == 3).astype(float)
    X['size2'] = (d['size'].astype(str) == '20-50').astype(float); X['size3'] = (d['size'].astype(str) == '>50').astype(float)
    for v in ('nodes', 'pgr', 'er'): X['l'+v] = np.log1p(d[v].astype(float))
    X[other] = d[other].astype(float)
    return X

def km_censoring_by_year(tau):
    """Estimate the probability of remaining under follow-up through tau within each surgery-year stratum."""
    G = np.ones(len(d))
    for y, idx in d.groupby('year').groups.items():
        t, c = T[idx], 1-E[idx]; s = 1.0
        for u in np.sort(np.unique(t[c == 1])):
            if u > tau: break
            s *= 1 - np.sum((t == u) & (c == 1))/np.sum(t >= u)
        G[idx] = s
    return G

def cox_censoring(X, A, tau):
    G = np.zeros(len(d))
    for a in (0, 1):
        i = A == a; dd = X[i].copy(); dd['T'] = T[i]; dd['C'] = 1-E[i]
        f = CoxPHFitter(penalizer=0.1).fit(dd, 'T', 'C')
        G[i] = f.predict_survival_function(X[i], times=[tau]).values.ravel()
        # We evaluate each arm-specific follow-up probability for everyone because both arms enter the tilt.
        if a == 0: G0 = f.predict_survival_function(X, times=[tau]).values.ravel()
        else: G1 = f.predict_survival_function(X, times=[tau]).values.ravel()
    return G0, G1

def tilts(e, G1, G0):
    H = {'IPW': np.ones_like(e), 'OW': e*(1-e), 'naive': e*(1-e)*G1*G0}
    for c in (0.1, 0.3, 1.0):
        H[f'CO c={c}'] = 1/((1+c*(1/G1-1))/e + (1+c*(1/G0-1))/(1-e))
    return H

rows = []
for trt, other in (('hormon', 'chemo'), ('chemo', 'hormon')):
    A = d[trt].values.astype(int); X = design_matrix(d, other)
    Z = (X - X.mean())/X.std().replace(0, 1)
    e = LogisticRegression(C=10, max_iter=10000).fit(Z, A).predict_proba(Z)[:, 1]
    for tau in (5, 7, 8):
        for cens in ('KM-by-year', 'Cox'):
            if cens == 'KM-by-year':
                G1 = G0 = np.clip(km_censoring_by_year(tau), 1e-6, 1)
            else:
                G0, G1 = cox_censoring(Z, A, tau); G0, G1 = np.clip(G0, 1e-6, 1), np.clip(G1, 1e-6, 1)
            H = tilts(e, G1, G0); ow = H['OW']; dOW = ow/ow.sum(); mid = (e > 0.2) & (e < 0.8)
            r = dict(treatment=trt, tau=tau, censoring=cens,
                     treated=int(A.sum()), PS_in_01_09=np.mean((e > .1) & (e < .9)),
                     G1_q05=np.quantile(G1, .05), G1_q10=np.quantile(G1, .10), G1_q50=np.quantile(G1, .5),
                     G0_q05=np.quantile(G0, .05),
                     equip_G1_lt10=np.mean(G1[mid] < .10), equip_G1_lt05=np.mean(G1[mid] < .05),
                     n_equip=int(mid.sum()))
            for k in ('CO c=0.1', 'CO c=0.3', 'CO c=1.0', 'naive'):
                h = H[k]; r[f'mass {k}'] = h.mean()/ow.mean(); r[f'TV {k}'] = 0.5*np.abs(h/h.sum() - dOW).sum()
            rows.append(r)
out = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(out.round(3).to_string(index=False))
out.to_csv(RESULTS/'rotterdam_design_table.csv', index=False)

# We check tau = 10 only to document why it was excluded: follow-up support is too weak there.
A = d['hormon'].values.astype(int); Z = design_matrix(d, 'chemo'); Z = (Z - Z.mean())/Z.std().replace(0, 1)
G0, G1 = cox_censoring(Z, A, 10)
print(f"\ntau = 10 (excluded): 5th percentile of G1(10|X) = {np.quantile(G1, .05):.2e}, of G0(10|X) = {np.quantile(G0, .05):.2e}")
