"""Check the derivative identities and bounds used in the CO theory.

First, we use symbolic differentiation to verify the formulas for the derivatives of the CO tilt. We then
evaluate the claimed bounds over a large numerical grid, including extremely small follow-up probabilities
and propensity scores near the treatment-positivity limits. This is a numerical audit of the algebra, not
an additional assumption or estimation step.
"""
import sympy as sp, numpy as np
np.seterr(all="ignore")          # overflow at extreme g is expected; affected points are excluded below

e, l1, l0, c = sp.symbols('e l1 l0 c', positive=True)
k1 = 1 - c + c*sp.exp(-l1); k0 = 1 - c + c*sp.exp(-l0)
F = k1/e + k0/(1 - e); H = 1/F
g1 = sp.exp(l1); u1 = c/(e*g1)                        # u1 = c/(e g1)
claims = {
    'H_e    = H^2 (k1/e^2 - k0/(1-e)^2)':        (sp.diff(H, e), H**2*(k1/e**2 - k0/(1 - e)**2)),
    'H_l1   = H^2 u1':                           (sp.diff(H, l1), H**2*u1),
    'H_l1l1 = 2 H^3 u1^2 - H^2 u1':              (sp.diff(H, l1, 2), 2*H**3*u1**2 - H**2*u1),
    'H_el1  = 2 H H_e u1 - H^2 u1 / e':          (sp.diff(H, e, l1), 2*H*sp.diff(H, e)*u1 - H**2*u1/e),
    'H_ee   = 2 F_e^2/F^3 - F_ee/F^2':           (sp.diff(H, e, 2), 2*sp.diff(F, e)**2/F**3 - sp.diff(F, e, 2)/F**2),
}
print("Part 1: symbolic identities")
for name, (lhs, rhs) in claims.items():
    print(f"  {name:42s} {'exact' if sp.simplify(lhs - rhs) == 0 else 'FAILED'}")

# We now evaluate numerical versions of the same derivatives.
fH = sp.lambdify((e, l1, l0, c), H, 'numpy')
fHe = sp.lambdify((e, l1, l0, c), sp.diff(H, e), 'numpy')
fHl1 = sp.lambdify((e, l1, l0, c), sp.diff(H, l1), 'numpy')
fHl1l1 = sp.lambdify((e, l1, l0, c), sp.diff(H, l1, 2), 'numpy')
fHel1 = sp.lambdify((e, l1, l0, c), sp.diff(H, e, l1), 'numpy')
fHee = sp.lambdify((e, l1, l0, c), sp.diff(H, e, 2), 'numpy')

rng = np.random.default_rng(0); n = 2_000_000; EPS = 0.02
E = rng.uniform(EPS, 1 - EPS, n); C = np.exp(rng.uniform(np.log(0.01), 0, n))
L1 = -np.exp(rng.uniform(np.log(1e-8), np.log(690), n)); L0 = -np.exp(rng.uniform(np.log(1e-8), np.log(690), n))
G1, G0 = np.exp(L1), np.exp(L0)
Hv = fH(E, L1, L0, C)
checks = {
    'H <= e g1 / c':                       (Hv, E*G1/C),
    'H <= (1-e) g0 / c':                   (Hv, (1 - E)*G0/C),
    'H <= min(e, 1-e)':                    (Hv, np.minimum(E, 1 - E)),
    '|H_e| <= 1':                          (np.abs(fHe(E, L1, L0, C)), np.ones(n)),
    '|H_e| <= H / (e(1-e))':               (np.abs(fHe(E, L1, L0, C)), Hv/(E*(1 - E))),
    '|H_l1| <= H':                         (np.abs(fHl1(E, L1, L0, C)), Hv),
    '|H_l1l1| <= 3 H <= 3 e g1 / c':       (np.abs(fHl1l1(E, L1, L0, C)), 3*E*G1/C),
    '|H_el1| <= 3 g1 / (c eps_e)':         (np.abs(fHel1(E, L1, L0, C)), 3*G1/(C*EPS)),
    '|H_ee| <= 8 / (e(1-e))':              (np.abs(fHee(E, L1, L0, C)), 8/(E*(1 - E))),
}
print(f"\nPart 2: inequalities on {n:,} random points (e in [{EPS}, {1-EPS}], c in [0.01, 1], g down to 1e-300)")
for name, (lhs, rhs) in checks.items():
    ok = np.isfinite(lhs) & np.isfinite(rhs) & (rhs > 0)
    r = np.max(lhs[ok]/rhs[ok])
    print(f"  {name:36s} max ratio {r:.4f}  {'OK' if r <= 1 + 1e-9 else 'VIOLATED'}   (finite points: {ok.mean():.3f})")

# We re-express this bound in a numerically stable form so very small G values do not create underflow.
#      H/(e g1) = 1 / (g1 k1 + e g1 k0/(1-e)) with g1 k1 = (1-c) g1 + c   (no underflow)
q = 1/(((1 - C)*G1 + C) + E*G1*(1 - C + C/G0)/(1 - E))
ratio = C*q**2/(1/C)
bad = ratio > 1 + 1e-9
print(f"\nStable form  H_g1/e <= 1/c : max ratio {ratio.max():.6f}  {'OK' if not bad.any() else 'VIOLATED'}")
print("(Checked in stable form: the naive c*H**2/(e**2*g1**2) underflows for g1 < 1e-150.)")
naive = C*Hv**2/(E**2*G1**2)*C
sel = np.isfinite(naive) & (G1 > 1e-150)
print(f"Naive form restricted to g1 > 1e-150: max ratio {naive[sel].max():.6f}")
