"""Check the second-order remainder of the one-step CO estimator.

We calculate the remainder directly under a simple discrete-X model instead of relying on the algebra used
in the proof. This lets us check the mixed-bias identity, the censoring-gradient identity, and the expected
second-order behavior as nuisance errors shrink. We also use the same setup to show the key cancellation:
when follow-up becomes very weak in one stratum, the OW remainder can grow while the CO remainder stays
controlled through its G-weighting.
"""
import numpy as np
TAU = 3.0; PX = np.array([0.3, 0.4, 0.3]); XS = np.arange(3.0)
U = np.linspace(0, TAU, 40001)
trap = lambda y: np.trapezoid(y, U)

def truth(eta=0.8):
    e = 1/(1 + np.exp(-(-0.3 + 0.6*XS)))
    lam = {a: 0.2*np.exp(0.3*XS - 0.4*a + 0.2*a*XS) for a in (0, 1)}
    lc = {a: 0.1*np.exp(eta*XS + 0.4*a) for a in (0, 1)}
    return e, lam, lc

def perturb(e, lam, lc, eA=0.0, eT=0.0, eC=0.0):
    e_ = 1/(1 + np.exp(-(np.log(e/(1 - e)) + eA*np.array([0.7, -0.4, 1.1]))))
    lam_ = {a: lam[a]*np.exp(eT*(0.5 + 0.3*XS - 0.4*a)) for a in (0, 1)}
    lc_ = {a: lc[a]*np.exp(eC*(-0.3 + 0.6*XS + 0.5*a)) for a in (0, 1)}
    return e_, lam_, lc_

def tilt(kind, c):
    if kind == 'OW':
        return (lambda e, g1, g0: e*(1 - e), lambda e, g1, g0: 1 - 2*e,
                lambda e, g1, g0: 0*e, lambda e, g1, g0: 0*e)
    H = lambda e, g1, g0: 1/((1 + c*(1/g1 - 1))/e + (1 + c*(1/g0 - 1))/(1 - e))
    He = lambda e, g1, g0: H(e, g1, g0)**2*((1 + c*(1/g1 - 1))/e**2 - (1 + c*(1/g0 - 1))/(1 - e)**2)
    H1 = lambda e, g1, g0: c*H(e, g1, g0)**2/(e*g1**2)
    H0 = lambda e, g1, g0: c*H(e, g1, g0)**2/((1 - e)*g0**2)
    return H, He, H1, H0

rmst = lambda l: (1 - np.exp(-l*TAU))/l

def E_aipcw(l, lcv, lh, lch):
    """E_P[ AIPCW transform built from (S^, G^) ] for one (x, a); true hazards l, lcv; estimated lh, lch."""
    Ghat = np.exp(-lch*U); surv_obs = np.exp(-(l + lcv)*U)            # P(T~ >= u)
    mhat = U + (1 - np.exp(-lh*(TAU - U)))/lh
    ipcw = trap(U/Ghat*l*surv_obs) + TAU/Ghat[-1]*surv_obs[-1]
    jump = trap(mhat/Ghat*lcv*surv_obs)
    comp = trap(mhat*lch/Ghat*surv_obs)
    return ipcw + jump - comp

def E_gamma(l, lcv, lh, lch):
    """E_P[gamma^ | x, a], gamma^ = -G^(tau) int_0^tau dM^C^ / (S^ G^)."""
    yhat = np.exp(-(lh + lch)*U); surv_obs = np.exp(-(l + lcv)*U)
    return -np.exp(-lch*TAU)*(trap(lcv*surv_obs/yhat) - trap(lch*surv_obs/yhat))

def psi(e, lam, lc, T):
    H = T[0]; G1, G0 = np.exp(-lc[1]*TAU), np.exp(-lc[0]*TAU); h = H(e, G1, G0)
    D = rmst(lam[1]) - rmst(lam[0]); return np.sum(PX*h*D)/np.sum(PX*h)

def R2(kind, c, truth_nuis, est_nuis):
    T = tilt(kind, c); H, He, H1, H0 = T
    e, lam, lc = truth_nuis; eh, lamh, lch = est_nuis
    G1h, G0h = np.exp(-lch[1]*TAU), np.exp(-lch[0]*TAU); hh = H(eh, G1h, G0h); Eh = np.sum(PX*hh)
    Psih = psi(eh, lamh, lch, T); Psi = psi(e, lam, lc, T)
    muh = {a: rmst(lamh[a]) for a in (0, 1)}; Dh = muh[1] - muh[0]
    b = {a: np.array([E_aipcw(lam[a][x], lc[a][x], lamh[a][x], lch[a][x]) for x in range(3)]) - muh[a] for a in (0, 1)}
    L = {a: np.array([E_gamma(lam[a][x], lc[a][x], lamh[a][x], lch[a][x]) for x in range(3)]) for a in (0, 1)}
    ea, eah = {1: e, 0: 1 - e}, {1: eh, 0: 1 - eh}
    Hg = {1: H1(eh, G1h, G0h), 0: H0(eh, G1h, G0h)}
    PD = np.sum(PX*(hh*(Dh - Psih) + hh*(e/eh*b[1] - (1 - e)/(1 - eh)*b[0])
                    + (Dh - Psih)*He(eh, G1h, G0h)*(e - eh)
                    + (Dh - Psih)*sum(Hg[a]*ea[a]/eah[a]*L[a] for a in (0, 1))))/Eh
    return Psih - Psi + PD, dict(b=b, L=L, muh=muh)

if __name__ == '__main__':
    e, lam, lc = truth()
    print("(1)-(2) mixed-bias identity for r_a and exact L_a, one (x=2, a=1) cell, joint perturbation 0.2")
    eh, lamh, lch = perturb(e, lam, lc, 0.2, 0.2, 0.2); x, a = 2, 1
    l, lcv, lh, lchh = lam[a][x], lc[a][x], lamh[a][x], lch[a][x]
    r_direct = E_aipcw(l, lcv, lh, lchh) - rmst(l)
    Sh, S = np.exp(-lh*U), np.exp(-l*U); Gh, G = np.exp(-lchh*U), np.exp(-lcv*U)
    Rh = np.array([0.0] + list(np.cumsum((Sh[1:] + Sh[:-1])/2*np.diff(U))))
    r_formula = trap((rmst(lh) - Rh)*(S/Sh)*((Gh - G)/Gh)*(l - lh))
    L_formula = -Gh[-1]*trap((S*G/(Sh*Gh))*(lcv - lchh))
    print(f"   r_a  direct {r_direct:+.8f}   formula {r_formula:+.8f}")
    print(f"   L_a  direct {E_gamma(l, lcv, lh, lchh):+.8f}   formula {L_formula:+.8f}")

    print("\n(3) second order: R2 / eps^2 as eps -> 0 (CO c=0.3)")
    for name, kw in (('treatment', dict(eA=1)), ('event', dict(eT=1)), ('censoring', dict(eC=1)),
                     ('all three', dict(eA=1, eT=1, eC=1))):
        vals = []
        for eps in (0.2, 0.1, 0.05, 0.025):
            est = perturb(e, lam, lc, **{k: v*eps for k, v in kw.items()})
            vals.append(R2('CO', 0.3, (e, lam, lc), est)[0]/eps**2)
        print(f"   {name:10s} " + "  ".join(f"{v:+.5f}" for v in vals))

    print("\n(4) cancellation: follow-up collapses in stratum x=2 (censoring slope eta); fixed errors")
    print("    event hazard x exp(0.2 f), censoring hazard x exp(0.2 g), propensity exact")
    print(f"   {'eta':>4s} {'G1(tau|x=2)':>12s} {'|R2| OW':>12s} {'|R2| CO c=0.1':>14s} {'|R2| CO c=1':>12s}")
    for eta in (0.8, 1.2, 1.6, 2.0, 2.4):
        tr = truth(eta); est = perturb(*tr, 0.0, 0.2, 0.2)
        print(f"   {eta:4.1f} {np.exp(-tr[2][1][2]*TAU):12.2e} {abs(R2('OW', None, tr, est)[0]):12.3e} "
              f"{abs(R2('CO', 0.1, tr, est)[0]):14.3e} {abs(R2('CO', 1.0, tr, est)[0]):12.3e}")
