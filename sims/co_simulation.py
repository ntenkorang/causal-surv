"""Shared functions for the simulation studies.

We keep the data-generating process, nuisance-model fitting, weighting rules, and one-step estimator in one
place so that every simulation uses the same implementation. X1 mainly controls treatment overlap, X2
controls follow-up, and the event model allows treatment effects to vary with X2. We tune the censoring
baseline so that the overall amount of censoring is comparable across follow-up regimes.

The main estimator uses the full influence function for a tilt h(e, G1, G0). We include OW, IPW, the
product tilt, the proposed CO tilts, and several truncated versions of OW for comparison.

For a small command-line run: python co_simulation.py n R taus
"""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
NG = 101

def expit(z): return 1/(1+np.exp(-z))

def gen_X(n, rng, p=3, b=3.5):
    out = np.empty((0, p))
    while len(out) < n:
        z = rng.standard_normal((2*n, p)); out = np.vstack([out, z[np.all(np.abs(z) < b, 1)]])
    return out[:n]

def feats_T(X): return np.column_stack([np.ones(len(X)), np.tanh(X[:, 0]), np.tanh(X[:, 1]), np.tanh(X[:, 2])])
CAP = [None]
def feats_C(X):
    x2 = X[:, 1] if CAP[0] is None else np.minimum(X[:, 1], CAP[0])
    return np.column_stack([np.ones(len(X)), x2])
def feats_e(X): return np.column_stack([np.ones(len(X)), X[:, 0]])

def truth_nuis(X, S):
    e = expit(S['alpha']*X[:, 0])
    t1, t2, t3 = np.tanh(X[:, 0]), np.tanh(X[:, 1]), np.tanh(X[:, 2])
    if S.get('dgp', 'het') == 'het':
        lam = {a: 0.2*np.exp(0.5*t1 + 0.5*t2 + a*(-0.3 + 0.5*t2)) for a in (0, 1)}
    else:   # outcome depends only on X3 (independent of X1, X2) -> every tilt has the same psi
        lam = {a: 0.2*np.exp(0.8*t3 + a*(-0.3 + 0.4*t3)) for a in (0, 1)}
    lc = {a: S['lc0']*np.exp(S['gamma']*X[:, 1] + 0.3*a) for a in (0, 1)}
    return e, lam, lc

def rmst(l, t): return (1-np.exp(-l*t))/l

# We define each tilt and its derivatives here so the estimator uses the same formulas throughout.
def make_tilts(clip=0.05):
    T = {'IPW': lambda e, G1, G0: np.ones_like(e),
         'OW': lambda e, G1, G0: e*(1-e),
         'naive e(1-e)G1G0': lambda e, G1, G0: e*(1-e)*G1*G0}
    for c in (0.1, 0.3, 1.0):
        T[f'CO c={c}'] = (lambda c: lambda e, G1, G0:
                          1/((1+c*(1/G1-1))/e + (1+c*(1/G0-1))/(1-e)))(c)
    return T
TILTS = make_tilts()

def partials(f, e, G1, G0, eps=1e-6):
    h = f(e, G1, G0)
    he = (f(e+eps, G1, G0) - f(e-eps, G1, G0))/(2*eps)
    h1 = (f(e, G1*(1+eps), G0) - f(e, G1*(1-eps), G0))/(2*eps*G1)
    h0 = (f(e, G1, G0*(1+eps)) - f(e, G1, G0*(1-eps)))/(2*eps*G0)
    return h, he, h1, h0

# Here we fit the nuisance models used by the one-step estimator.
def newton(fun, b, it=60):
    for _ in range(it):
        g, H = fun(b); st = np.linalg.solve(H, g); b = b + st
        if np.max(np.abs(st)) < 1e-9: break
    return b
def fit_logit(F, A):
    return newton(lambda b: (F.T@(A-expit(F@b)),
                  (F*(expit(F@b)*(1-expit(F@b)))[:, None]).T@F), np.zeros(F.shape[1]))
def fit_exp(F, t, d):
    b0 = np.zeros(F.shape[1]); b0[0] = np.log(max(d.sum(), 1)/t.sum())
    return newton(lambda b: (F.T@(d-np.exp(F@b)*t), (F*(np.exp(F@b)*t)[:, None]).T@F), b0)

# These functions build the observation-level pieces of the influence function.
def aipcw(a, A, Tt, D, pa, lam, lc, tau, gclip=None):
    """Build the uncentered AIPCW RMST transform for one arm; G can be clipped for truncation comparisons."""
    Ia = (A == a).astype(float); R = rmst(lam, tau)
    G = (lambda t: np.maximum(np.exp(-lc*t), gclip)) if gclip else (lambda t: np.exp(-lc*t))
    up = np.minimum(Tt, tau)
    Dtau = ((Tt >= tau) | (D == 1)).astype(float)
    m = lambda t: t + (1-np.exp(-lam*(tau-t)))/lam
    jump = ((D == 0) & (Tt <= tau))*m(up)/G(up)
    u = np.linspace(0, 1, NG)[:, None]*up[None, :]
    comp = np.trapezoid(m(u)*lc[None, :]/G(u), u, axis=0)
    return Ia/pa*(Dtau*up/G(up) + jump - comp) + (1-Ia/pa)*R

def phi_G(a, A, Tt, D, lam, lc, tau):
    """Calculate the conditional gradient of G_a(tau|x); the treatment-weight factor is added later."""
    up = np.minimum(Tt, tau); r = lam+lc
    jump = ((D == 0) & (Tt <= tau))*np.exp(r*up)            # dN^C/(S G)
    comp = lc/r*(np.exp(r*up)-1)
    return -np.exp(-lc*tau)*(jump - comp)

def estimate(X, A, Tt, D, tau):
    Fe, FT, FC = feats_e(X), feats_T(X), feats_C(X)
    e = expit(Fe@fit_logit(Fe, A))
    lam, lc = {}, {}
    for a in (0, 1):
        i = A == a
        lam[a] = np.exp(FT@fit_exp(FT[i], Tt[i], D[i])); lc[a] = np.exp(FC@fit_exp(FC[i], Tt[i], 1-D[i]))
    G1, G0 = np.exp(-lc[1]*tau), np.exp(-lc[0]*tau)
    mu = {a: rmst(lam[a], tau) for a in (0, 1)}; Dl = mu[1]-mu[0]
    phi = {a: aipcw(a, A, Tt, D, e if a else 1-e, lam[a], lc[a], tau) for a in (0, 1)}
    phiC = {a: aipcw(a, A, Tt, D, e if a else 1-e, lam[a], lc[a], tau, gclip=0.05) for a in (0, 1)}
    pG = {a: phi_G(a, A, Tt, D, lam[a], lc[a], tau) for a in (0, 1)}
    Pt = {}
    for lab in ('fix0.05', 'fix0.10', 'pct1', 'pct5'):
        fl = {}
        for a in (0, 1):
            i = A == a; Gobs = np.exp(-lc[a][i]*np.minimum(Tt[i], tau))
            fl[a] = float(lab[3:]) if lab.startswith('fix') else np.quantile(Gobs, float(lab[3:])/100)
        Pt[lab] = {a: aipcw(a, A, Tt, D, e if a else 1-e, lam[a], lc[a], tau, gclip=fl[a]) for a in (0, 1)}
    out = {}
    for name, f in list(TILTS.items()) + [(f'OW trunc {k}', TILTS['OW']) for k in Pt]:
        h, he, h1, h0 = partials(f, e, G1, G0)
        P = Pt[name.split()[-1]] if name.startswith('OW trunc') else phi
        core = h*(P[1]-P[0])                                  # plug-in + AIPCW
        psi = core.sum()/h.sum()
        corr = (Dl-psi)*(he*(A-e) + h1*(A == 1)/e*pG[1] + h0*(A == 0)/(1-e)*pG[0])
        if name.startswith('OW trunc'): corr = (Dl-psi)*he*(A-e)
        psi1 = psi + corr.sum()/h.sum()
        IF = (h*(P[1]-P[0]) - h*psi1 + corr)/h.mean()
        se = IF.std(ddof=1)/np.sqrt(len(A))
        out[name] = (psi1, se)
    return out

def truths(S, tau, N=1_000_000, seed=99):
    X = gen_X(N, np.random.default_rng(seed)); e, lam, lc = truth_nuis(X, S)
    G1, G0 = np.exp(-lc[1]*tau), np.exp(-lc[0]*tau)
    Dl = rmst(lam[1], tau)-rmst(lam[0], tau)
    res = {}; hOW = e*(1-e); dOW = hOW/hOW.sum()
    for name, f in list(TILTS.items()):
        h = f(e, G1, G0)
        res[name] = dict(psi=np.sum(h*Dl)/h.sum(), mass=h.mean()/hOW.mean(),
                         tv=0.5*np.abs(h/h.sum()-dOW).sum())
    cens = np.mean(e*(1-G1) + (1-e)*(1-G0))
    q = {a: np.quantile(np.exp(-lc[a]*tau), [0.01, 0.05, 0.10]) for a in (0, 1)}
    return res, cens, q

def simulate(n, S, rng):
    X = gen_X(n, rng); e, lam, lc = truth_nuis(X, S)
    A = rng.binomial(1, e)
    T = rng.exponential(1/np.where(A == 1, lam[1], lam[0])); C = rng.exponential(1/np.where(A == 1, lc[1], lc[0]))
    return X, A, np.minimum(T, C), (T <= C).astype(float)

def tune_lc0(S, tau, target):
    X = gen_X(200_000, np.random.default_rng(5))
    lo, hi = 1e-4, 5.0
    for _ in range(40):
        mid = np.sqrt(lo*hi); e, _, lc = truth_nuis(X, {**S, 'lc0': mid})
        cr = np.mean(e*(1-np.exp(-lc[1]*tau)) + (1-e)*(1-np.exp(-lc[0]*tau)))
        lo, hi = (mid, hi) if cr < target else (lo, mid)
    return mid

REGIMES = {'good trt / good FU': (0.0, 0.3), 'good trt / poor FU': (0.0, 1.4),
           'poor trt / good FU': (2.5, 0.3), 'poor trt / poor FU': (2.5, 1.4)}

def run(n, R, tau, regime, dgp='het', cap=None, seed=2026, keys=None):
    alpha, gamma = REGIMES[regime]; S = {'alpha': alpha, 'gamma': gamma, 'dgp': dgp}
    CAP[0] = None; S['lc0'] = tune_lc0(S, 3.0, 0.35); tr, cens, q = truths(S, tau)
    CAP[0] = cap; rng = np.random.default_rng(seed)
    names = list(tr) + [f'OW trunc {l}' for l in ('fix0.05','fix0.10','pct1','pct5')]
    est = {k: [] for k in names}
    for _ in range(R):
        o = estimate(*simulate(n, S, rng), tau)
        for k in est: est[k].append(o[k])
    CAP[0] = None
    rows = {}
    for k, v in est.items():
        v = np.array(v); p, s = v[:, 0], v[:, 1]; t = tr.get(k, tr['OW'])['psi']
        rows[k] = dict(bias=p.mean()-t, sd=p.std(), rmse=np.sqrt(np.mean((p-t)**2)),
                       cov=np.mean(np.abs(p-t) <= 1.96*s), kurt=np.mean((p-p.mean())**4)/p.var()**2,
                       psi=t, tv=tr.get(k, tr['OW'])['tv'])
    return rows, cens, q

def show(title, rows, cens, q, keys=None):
    print(f"\n=== {title} | cens {cens:.0%} | G1 q05 {q[1][1]:.3f}")
    print(f"  {'tilt':18s} {'bias':>7s} {'SD':>7s} {'RMSE':>7s} {'cover':>6s} {'kurt':>6s} {'TV_OW':>6s} {'psi':>7s}")
    for k, r in rows.items():
        if keys and k not in keys: continue
        print(f"  {k:18s} {r['bias']:+7.3f} {r['sd']:7.3f} {r['rmse']:7.3f} {r['cov']:6.1%} {r['kurt']:6.1f} {r['tv']:6.2f} {r['psi']:7.3f}")
    sys.stdout.flush()

if __name__ == '__main__':
    n, R, taus = int(sys.argv[1]), int(sys.argv[2]), [float(t) for t in sys.argv[3].split(',')]
    for tau in taus:
        for regime in REGIMES:
            show(f"tau={tau:g} | {regime}", *run(n, R, tau, regime))
