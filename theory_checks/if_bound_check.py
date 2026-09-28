"""Check the uniform influence-function bound for CO.

We let follow-up become progressively weaker in one covariate stratum and evaluate the CO bound using the
corresponding nuisance functions. We compare this behavior with OW, which does not have the same
cancellation. We also perturb the nuisance functions to check that the estimated influence function moves
toward the truth as the perturbation shrinks.
"""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sims'))
import co_simulation as M
M.NG = 201
TAU = 3.0; PX = np.array([0.3, 0.4, 0.3]); XS = np.arange(3.0)

def nuis(eta, eps=0.0):
    e = M.expit(-0.3 + 0.6*XS + eps*np.array([0.7, -0.4, 1.1]))
    lam = {a: 0.2*np.exp(0.3*XS - 0.4*a + 0.2*a*XS + eps*(0.5 + 0.3*XS - 0.4*a)) for a in (0, 1)}
    lc = {a: 0.1*np.exp(eta*XS + 0.4*a + eps*(-0.3 + 0.6*XS + 0.5*a)) for a in (0, 1)}
    return e, lam, lc

def D(kind, c, nu, data, Psi=None):
    e, lam, lc = nu; Xi, A, Tt, Dd = data
    G1, G0 = np.exp(-lc[1]*TAU), np.exp(-lc[0]*TAU)
    if kind == 'OW':
        h = e*(1 - e); He = 1 - 2*e; Hg = {1: 0*e, 0: 0*e}
    else:
        k1, k0 = 1 + c*(1/G1 - 1), 1 + c*(1/G0 - 1); h = 1/(k1/e + k0/(1 - e))
        He = h**2*(k1/e**2 - k0/(1 - e)**2); Hg = {1: c*h**2/(e*G1**2), 0: c*h**2/((1 - e)*G0**2)}
    Dl = M.rmst(lam[1], TAU) - M.rmst(lam[0], TAU); Eh = np.sum(PX*h)
    if Psi is None: Psi = np.sum(PX*h*Dl)/Eh
    ea = {1: e[Xi], 0: 1 - e[Xi]}
    phi = {a: M.aipcw(a, A, Tt, Dd, ea[a], lam[a][Xi], lc[a][Xi], TAU) for a in (0, 1)}
    gam = {a: M.phi_G(a, A, Tt, Dd, lam[a][Xi], lc[a][Xi], TAU) for a in (0, 1)}
    d = (h[Xi]*(phi[1] - phi[0] - Psi) + (Dl[Xi] - Psi)*(He[Xi]*(A - e[Xi])
         + sum(Hg[a][Xi]*(A == a)/ea[a]*gam[a] for a in (0, 1))))/Eh
    s0 = min(np.exp(-lam[a]*TAU).min() for a in (0, 1))
    bound = TAU/Eh*(5 + 1/c + 2/(c*s0)) if kind == 'CO' else np.nan
    return d, bound, Psi

rng = np.random.default_rng(3)
print(f"{'eta':>4s} {'G1(tau|2)':>10s} {'max|D_OW|':>11s} {'max|D_CO|':>10s} {'CO bound':>9s} {'||D^-D||_2 CO, eps=0.2/0.1/0.05':>34s}")
for eta in (0.8, 1.6, 2.4, 3.2):
    e, lam, lc = nuis(eta); n = 100_000
    Xi = rng.choice(3, n, p=PX); A = rng.binomial(1, e[Xi])
    T = rng.exponential(1/np.where(A == 1, lam[1][Xi], lam[0][Xi])); C = rng.exponential(1/np.where(A == 1, lc[1][Xi], lc[0][Xi]))
    data = (Xi, A, np.minimum(T, C), (T <= C).astype(float))
    dOW, _, _ = D('OW', None, (e, lam, lc), data)
    dCO, bnd, _ = D('CO', 0.1, (e, lam, lc), data)
    l2 = []
    for eps in (0.2, 0.1, 0.05):
        dhat, bhat, _ = D('CO', 0.1, nuis(eta, eps), data)
        assert np.abs(dhat).max() <= bhat*1.01 + 1e-9, 'estimated-nuisance bound violated'
        l2.append(np.sqrt(np.mean((dhat - dCO)**2)))
    print(f"{eta:4.1f} {np.exp(-lc[1][2]*TAU):10.1e} {np.abs(dOW).max():11.3e} {np.abs(dCO).max():10.3f} {bnd:9.2f} "
          f"{'  '.join(f'{v:.4f}' for v in l2):>34s}")
