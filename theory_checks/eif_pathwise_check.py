"""Numerically check the efficient influence function used for the CO target.

We build a small model where the covariate distribution, treatment model, event hazard, and censoring
hazard can each be perturbed along a known one-dimensional path. For each path, we compare the numerical
derivative of the target with E[D_CO s]. Agreement gives a direct check of the corresponding influence-
function component. We also verify that the influence function has mean zero and that its four variance
components add up correctly.
"""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sims'))
import co_simulation as M
M.NG = 201
TAU, C = 3.0, 0.3
PX = np.array([0.3, 0.4, 0.3]); XS = np.arange(3.0)
b = np.array([1.0, -0.5, 0.8]); adir = np.array([0.7, -0.4, 1.1])
f = lambda x, a: 0.5 + 0.3*x - 0.4*a
g = lambda x, a: -0.3 + 0.6*x + 0.5*a

def params(t=0.0, d=None):
    px = PX*(1 + (t*b if d == 'X' else 0)); px = px/px.sum()
    e = M.expit(-0.3 + 0.6*XS + (t*adir if d == 'A' else 0))
    lam = {a: 0.2*np.exp(0.3*XS - 0.4*a + 0.2*a*XS + (t*f(XS, a) if d == 'T' else 0)) for a in (0, 1)}
    lc = {a: 0.1*np.exp(0.8*XS + 0.4*a + (t*g(XS, a) if d == 'C' else 0)) for a in (0, 1)}
    return px, e, lam, lc

def H(e, g1, g0):
    return 1/((1 + C*(1/g1 - 1))/e + (1 + C*(1/g0 - 1))/(1 - e))

def psi_exact(t=0.0, d=None):
    px, e, lam, lc = params(t, d)
    h = H(e, np.exp(-lc[1]*TAU), np.exp(-lc[0]*TAU))
    Dl = M.rmst(lam[1], TAU) - M.rmst(lam[0], TAU)
    return np.sum(px*h*Dl)/np.sum(px*h)

px, e, lam, lc = params()
G = {a: np.exp(-lc[a]*TAU) for a in (0, 1)}
h = H(e, G[1], G[0]); Eh = np.sum(px*h); Psi = psi_exact()
Dl = M.rmst(lam[1], TAU) - M.rmst(lam[0], TAU)
# We calculate the analytic derivatives of H that appear in the influence function.
k1, k0 = 1 + C*(1/G[1] - 1), 1 + C*(1/G[0] - 1)
He = h**2*(k1/e**2 - k0/(1 - e)**2)
Hg = {1: C*h**2/(e*G[1]**2), 0: C*h**2/((1 - e)*G[0]**2)}

def D_of(Xi, A, Tt, Dd):
    ea = {1: e[Xi], 0: 1 - e[Xi]}
    phi = {a: M.aipcw(a, A, Tt, Dd, ea[a], lam[a][Xi], lc[a][Xi], TAU) for a in (0, 1)}
    gam = {a: M.phi_G(a, A, Tt, Dd, lam[a][Xi], lc[a][Xi], TAU) for a in (0, 1)}
    hh, dd = h[Xi], Dl[Xi]
    parts = dict(
        X=hh*(dd - Psi),
        A=(dd - Psi)*He[Xi]*(A - e[Xi]),
        T=hh*((phi[1] - phi[0]) - dd),                    # = h[A psi1/e - (1-A) psi0/(1-e)]
        C=(dd - Psi)*sum(Hg[a][Xi]*(A == a)/ea[a]*gam[a] for a in (0, 1)))
    return {k: v/Eh for k, v in parts.items()}

rng = np.random.default_rng(11); N, chunks = 50_000, 40
acc = {d: [] for d in ('X', 'A', 'T', 'C')}; Dall = []; comp = {k: [] for k in ('X', 'A', 'T', 'C')}
for _ in range(chunks):
    Xi = rng.choice(3, N, p=px); A = rng.binomial(1, e[Xi])
    lamA = np.where(A == 1, lam[1][Xi], lam[0][Xi]); lcA = np.where(A == 1, lc[1][Xi], lc[0][Xi])
    T = rng.exponential(1/lamA); Cc = rng.exponential(1/lcA); Tt = np.minimum(T, Cc); Dd = (T <= Cc).astype(float)
    parts = D_of(Xi, A, Tt, Dd); D = sum(parts.values()); Dall.append(D)
    for k in comp: comp[k].append(parts[k])
    s = {'X': b[Xi] - np.sum(px*b), 'A': adir[Xi]*(A - e[Xi]),
         'T': f(XS[Xi], A)*(Dd - lamA*Tt), 'C': g(XS[Xi], A)*(1 - Dd - lcA*Tt)}
    for d in acc: acc[d].append(D*s[d])
Dall = np.concatenate(Dall); n = len(Dall)
print(f"Psi = {Psi:.5f}   E[D] = {Dall.mean():+.5f} (MC SE {Dall.std()/np.sqrt(n):.5f})\n")
print(f"{'direction':12s} {'dPsi/dt exact':>14s} {'E[D s] (MC)':>12s} {'MC SE':>8s} {'z':>6s}")
eps = 1e-5
for d, name in (('X', 'X-marginal'), ('A', 'treatment'), ('T', 'event law'), ('C', 'censoring')):
    fd = (psi_exact(eps, d) - psi_exact(-eps, d))/(2*eps)
    v = np.concatenate(acc[d]); mc = v.mean(); se = v.std()/np.sqrt(n)
    print(f"{name:12s} {fd:14.5f} {mc:12.5f} {se:8.5f} {(mc - fd)/se:6.2f}")

# Here we compare the analytic variance pieces with a direct Monte Carlo calculation.
def v_arm(l, lcv):
    u = np.linspace(0, TAU, 4001)[:, None]
    _, s2 = M.rmst(l, TAU), None
    m1 = (1 - np.exp(-l*TAU))/l; m2 = 2*(1 - np.exp(-l*TAU)*(1 + l*TAU))/l**2; s2 = m2 - m1**2
    r = np.maximum(TAU - u, 1e-12); mr1 = (1 - np.exp(-l*r))/l; mr2 = 2*(1 - np.exp(-l*r)*(1 + l*r))/l**2
    om = mr2 - mr1**2
    Cint = np.trapezoid(om*np.exp(-l*u)*lcv*np.exp(lcv*u), u[:, 0], axis=0)
    return s2 + Cint
v = {a: v_arm(lam[a], lc[a]) for a in (0, 1)}
varG = {a: G[a]**2*lc[a]/(lam[a] + lc[a])*(np.exp((lam[a] + lc[a])*TAU) - 1) for a in (0, 1)}
ea = {1: e, 0: 1 - e}
pred = {'X': np.sum(px*h**2*(Dl - Psi)**2)/Eh**2,
        'A': np.sum(px*(Dl - Psi)**2*He**2*e*(1 - e))/Eh**2,
        'T': np.sum(px*h**2*(v[1]/e + v[0]/(1 - e)))/Eh**2,
        'C': np.sum(px*(Dl - Psi)**2*sum(Hg[a]**2*varG[a]/ea[a] for a in (0, 1)))/Eh**2}
print(f"\n{'component':10s} {'analytic':>10s} {'MC E[part^2]':>13s}")
parts = {k: np.concatenate(vv) for k, vv in comp.items()}
for k in pred: print(f"{k:10s} {pred[k]:10.5f} {np.mean(parts[k]**2):13.5f}")
keys = list(parts); cross = max(abs(np.mean(parts[i]*parts[j])) for i in keys for j in keys if i < j)
print(f"sum        {sum(pred.values()):10.5f} {np.mean(Dall**2):13.5f}   | largest |cross moment| = {cross:.5f}")

# For X, treatment, and censoring we can check the score pairings exactly, without Monte Carlo.
# Only the matching component of D has non-zero covariance with each score (others have conditional mean zero
# given the conditioning variables of that score), so E[D s] reduces to closed forms:
#   X     : E[h (Delta - Psi) (b - E b)] / E h
#   A     : E[(Delta - Psi) H_e a(X) e(1 - e)] / E h
#   C     : sum_a E[(Delta - Psi) H_g,a * E(gamma_a s_C | X, A=a)] / E h,  E(gamma_a s_C | X,a) = -G_a lambda^C_a tau g(X,a)
def exact_pairings():
    out = {'X': np.sum(px*h*(Dl - Psi)*(b - np.sum(px*b)))/Eh,
           'A': np.sum(px*(Dl - Psi)*He*adir*e*(1 - e))/Eh,
           'C': np.sum(px*(Dl - Psi)*sum(Hg[a]*(-G[a]*lc[a]*TAU*g(XS, a)) for a in (0, 1)))/Eh}
    return out
if __name__ == '__main__':
    print("\nexact pairings vs exact derivatives")
    for d, val in exact_pairings().items():
        fd = (psi_exact(1e-5, d) - psi_exact(-1e-5, d))/2e-5
        print(f"  {d}: E[D s] = {val:.8f}   dPsi/dt = {fd:.8f}   diff = {val - fd:+.1e}")
