"""Run two auxiliary simulations that help explain when CO weighting is useful.

Experiment A changes where weak follow-up occurs relative to treatment information. We use the same
covariate, X2, in the treatment and censoring models. Positive rho places weak follow-up where treatment
information is concentrated, while negative rho places it where treatment is rare. The outcome depends
only on X3, so every tilt has the same estimand and the comparison is purely about precision.

Experiment B checks sensitivity to the censoring model. We generate a nonlinear censoring tail and then
fit models that are correct, mildly misspecified, or deliberately miss the tail. This lets us see how much
the OW and CO estimates move when weak follow-up is not modeled well.

Usage: python co_alignment_censmodel.py {A|B}
"""
import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
import co_simulation as M

def dgp(X, S):
    t3 = np.tanh(X[:, 2])
    if S['exp'] == 'A':
        e = M.expit(S.get('a0', -1.5) + 1.0*X[:, 1])
        lc = {a: S['lc0']*np.exp(S['rho']*1.4*X[:, 1] + 0.3*a) for a in (0, 1)}
    else:
        e = np.full(len(X), 0.5)
        lc = {a: S['lc0']*np.exp(0.3*X[:, 1] + 1.5*np.maximum(X[:, 1]-0.5, 0) + 0.3*a) for a in (0, 1)}
    lam = {a: 0.2*np.exp(0.8*t3 + a*(-0.3 + 0.4*t3)) for a in (0, 1)}
    return e, lam, lc

CFEATS = {'correct': lambda X: np.column_stack([np.ones(len(X)), X[:, 1], np.maximum(X[:, 1]-0.5, 0)]),
          'linear':  lambda X: np.column_stack([np.ones(len(X)), X[:, 1]]),
          'flat-tail': lambda X: np.column_stack([np.ones(len(X)), np.minimum(X[:, 1], 0.5)])}
TFEATS = {'correct': lambda X: np.column_stack([np.ones(len(X)), np.tanh(X[:, 2])]),
          'omits X3': lambda X: np.column_stack([np.ones(len(X)), np.tanh(X[:, 0])])}
EFEAT = lambda X: np.column_stack([np.ones(len(X)), X[:, 1]])
KEYS = ['OW', 'CO c=0.1', 'CO c=0.3', 'CO c=1.0', 'naive e(1-e)G1G0']

def estimate(X, A, Tt, D, tau, cf, tf):
    e = M.expit(EFEAT(X)@M.fit_logit(EFEAT(X), A)) if len(np.unique(A)) > 1 else None
    lam, lc = {}, {}
    for a in (0, 1):
        i = A == a
        lam[a] = np.exp(tf(X)@M.fit_exp(tf(X)[i], Tt[i], D[i])); lc[a] = np.exp(cf(X)@M.fit_exp(cf(X)[i], Tt[i], 1-D[i]))
    G1, G0 = np.exp(-lc[1]*tau), np.exp(-lc[0]*tau)
    Dl = M.rmst(lam[1], tau) - M.rmst(lam[0], tau)
    phi = {a: M.aipcw(a, A, Tt, D, e if a else 1-e, lam[a], lc[a], tau) for a in (0, 1)}
    pG = {a: M.phi_G(a, A, Tt, D, lam[a], lc[a], tau) for a in (0, 1)}
    out = {}
    for k in KEYS:
        f = M.TILTS[k]; h, he, h1, h0 = M.partials(f, e, G1, G0)
        psi = np.sum(h*(phi[1]-phi[0]))/h.sum()
        corr = (Dl-psi)*(he*(A-e) + h1*(A == 1)/e*pG[1] + h0*(A == 0)/(1-e)*pG[0])
        psi1 = psi + corr.sum()/h.sum()
        IF = (h*(phi[1]-phi[0]) - h*psi1 + corr)/h.mean()
        w1, w0 = h/e*A, h/(1-e)*(1-A)
        out[k] = (psi1, IF.std(ddof=1)/np.sqrt(len(A)), w1.sum()**2/(w1**2).sum() + w0.sum()**2/(w0**2).sum())
    return out

def simulate(n, S, rng):
    X = M.gen_X(n, rng); e, lam, lc = dgp(X, S); A = rng.binomial(1, e)
    T = rng.exponential(1/np.where(A == 1, lam[1], lam[0])); C = rng.exponential(1/np.where(A == 1, lc[1], lc[0]))
    return X, A, np.minimum(T, C), (T <= C).astype(float)

def tune(S, tau=3.0, target=0.35):
    X = M.gen_X(200_000, np.random.default_rng(5)); lo, hi = 1e-4, 5.0
    for _ in range(40):
        mid = np.sqrt(lo*hi); e, _, lc = dgp(X, {**S, 'lc0': mid})
        cr = np.mean(e*(1-np.exp(-lc[1]*tau)) + (1-e)*(1-np.exp(-lc[0]*tau)))
        lo, hi = (mid, hi) if cr < target else (lo, mid)
    return mid

def truth_geometry(S, tau, N=1_000_000):
    X = M.gen_X(N, np.random.default_rng(99)); e, lam, lc = dgp(X, S)
    G1, G0 = np.exp(-lc[1]*tau), np.exp(-lc[0]*tau); Dl = M.rmst(lam[1], tau)-M.rmst(lam[0], tau)
    hOW = e*(1-e); res = {}
    for k in KEYS:
        h = M.TILTS[k](e, G1, G0)
        res[k] = dict(psi=np.sum(h*Dl)/h.sum(), tv=0.5*np.abs(h/h.sum()-hOW/hOW.sum()).sum())
    wmass = np.sum(hOW*(np.minimum(G1, G0) < 0.10))/hOW.sum()
    return res, wmass, np.quantile(G1, 0.05)

if __name__ == '__main__':
    which, n, R, tau = sys.argv[1], 2000, 500, 5.0
    rng = np.random.default_rng(2027)
    if which == 'A':
        for rho in (-1.0, -0.5, 0.0, 0.5, 1.0):
            S = {'exp': 'A', 'rho': rho}; S['lc0'] = tune(S)
            tr, wmass, g05 = truth_geometry(S, tau)
            est = {k: [] for k in KEYS}
            for _ in range(R):
                o = estimate(*simulate(n, S, rng), tau, CFEATS['linear'], TFEATS['correct'])
                for k in KEYS: est[k].append(o[k])
            v = {k: np.array(x) for k, x in est.items()}; sdOW = v['OW'][:, 0].std(); essOW = v['OW'][:, 2].mean()
            print(f"\nrho={rho:+.1f} | OW weak-follow-up mass (min G<0.10) {wmass:.2f} | G1 q05 {g05:.3f} | psi {tr['OW']['psi']:.3f}")
            print(f"  {'tilt':18s} {'Var/VarOW':>9s} {'ESS/ESS_OW':>10s} {'TV_OW':>6s} {'bias':>7s} {'cover':>6s} {'kurt':>6s}")
            for k in KEYS:
                p, s, ess = v[k][:, 0], v[k][:, 1], v[k][:, 2]; t = tr[k]['psi']
                print(f"  {k:18s} {(p.std()/sdOW)**2:9.2f} {ess.mean()/essOW:10.2f} {tr[k]['tv']:6.2f} {p.mean()-t:+7.3f} "
                      f"{np.mean(np.abs(p-t) <= 1.96*s):6.1%} {np.mean((p-p.mean())**4)/p.var()**2:6.1f}")
            sys.stdout.flush()
    if which == 'B':
        S = {'exp': 'B'}; S['lc0'] = tune(S); tr, wmass, g05 = truth_geometry(S, tau)
        print(f"B: nonlinear censoring tail | OW weak-follow-up mass {wmass:.2f} | G1 q05 {g05:.3f} | psi {tr['OW']['psi']:.3f}")
        for tname, tf in TFEATS.items():
            est = {(c, k): [] for c in CFEATS for k in KEYS}
            for _ in range(R):
                dat = simulate(n, S, rng)
                for c, cf in CFEATS.items():
                    o = estimate(*dat, tau, cf, tf)
                    for k in KEYS: est[(c, k)].append(o[k][0])
            print(f"\n event model: {tname}")
            print(f"  {'tilt':18s} " + " ".join(f"{'bias '+c:>15s}" for c in CFEATS) +
                  f" {'SD correct':>10s} {'mean|lin-cor|':>13s} {'mean|flat-cor|':>14s}")
            for k in KEYS:
                p = {c: np.array(est[(c, k)]) for c in CFEATS}; t = tr[k]['psi']
                print(f"  {k:18s} " + " ".join(f"{p[c].mean()-t:+15.3f}" for c in CFEATS) +
                      f" {p['correct'].std():10.3f} {np.mean(np.abs(p['linear']-p['correct'])):13.3f} "
                      f"{np.mean(np.abs(p['flat-tail']-p['correct'])):14.3f}")
            sys.stdout.flush()
