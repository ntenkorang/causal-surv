"""Reproduce the bounded-support variance example from the supplement.

We gradually widen a truncated normal covariate distribution while keeping the event and censoring models
fixed. For each support width, we calculate the asymptotic variance of OW and CO with c = 0.1. We do this
under both randomized treatment and a logistic treatment model. The calculation is numerical, using
adaptive quadrature with a tight tolerance.
"""
import numpy as np
from scipy import integrate

TAU, LAM, LC0, C = 4.0, 0.25, 0.05, 0.1

def moments(l, t):
    m1 = (1 - np.exp(-l*t))/l; m2 = 2*(1 - np.exp(-l*t)*(1 + l*t))/l**2
    return m1, m2 - m1**2

def v(x):
    lc = LC0*np.exp(x); s2 = moments(LAM, TAU)[1]
    f = lambda u: moments(LAM, max(TAU - u, 1e-12))[1]*np.exp(-LAM*u)*lc*np.exp(lc*u)
    return s2 + integrate.quad(f, 0, TAU, limit=400, epsabs=0, epsrel=1e-10)[0]

print(f"{'design':12s} {'b':>2s} {'V(h_OW)':>11s} {'V(h_CO)':>9s} {'Mass':>6s} {'E(h_CO)':>8s}")
for alpha, name in ((0.0, 'randomized'), (0.5, 'logistic')):
    e = lambda x: 1/(1 + np.exp(-alpha*x))
    G = lambda x: np.exp(-LC0*np.exp(x)*TAU)
    h_co = lambda x: 1/((1 + C*(1/G(x) - 1))/e(x) + (1 + C*(1/G(x) - 1))/(1 - e(x)))
    h_ow = lambda x: e(x)*(1 - e(x))
    for b in (2, 3, 4, 5, 6):
        pts = [b - 1, b - 0.5, b - 0.1]
        Z = integrate.quad(lambda x: np.exp(-x*x/2), -b, b)[0]
        def V(h):
            num = integrate.quad(lambda x: h(x)**2*(v(x)/e(x) + v(x)/(1 - e(x)))*np.exp(-x*x/2), -b, b,
                                 limit=400, epsrel=1e-10, points=pts)[0]
            Eh = integrate.quad(lambda x: h(x)*np.exp(-x*x/2), -b, b, limit=400, epsrel=1e-12, points=pts)[0]/Z
            return num/Z/Eh**2, Eh
        vo, eo = V(h_ow); vc, ec = V(h_co)
        print(f"{name:12s} {b:2d} {vo:11.4g} {vc:9.4f} {ec/eo:6.3f} {ec:8.4f}", flush=True)
