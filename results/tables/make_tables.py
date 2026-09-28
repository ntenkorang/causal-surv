"""Create the LaTeX table rows used in the manuscript and supplement.

We read every reported number from the saved result files rather than typing results by hand. This keeps
the tables synchronized with the analysis. Each output file is a small LaTeX fragment that can be inserted
directly into the corresponding table.

Usage: python tables/make_tables.py
"""
from pathlib import Path
import re
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIM, RD, TC, OUT = ROOT/'results'/'sims', ROOT/'results'/'realdata', ROOT/'results'/'theory_checks', ROOT/'tables'


def num(v, d=1):
    """Format a number consistently for the LaTeX tables and avoid negative zero."""
    t = f"{v:.{d}f}"
    if float(t) == 0:
        t = f"{0:.{d}f}"
    return t.replace('-', '$-$')


def nz(v, d=3):
    """Use the same formatting as num(), but drop the leading zero for values between -1 and 1."""
    return num(v, d).replace('0.', '.', 1) if abs(v) < 1 else num(v, d)


def sci(v):
    """Use ordinary decimals for smaller values and scientific notation for large ones."""
    if v < 100:
        return f"{v:.2f}"
    k = int(np.floor(np.log10(v)))
    return f"${v/10**k:.2f}\\times10^{{{k}}}$"


def write(name, rows, note=''):
    text = (f"% {note}\n" if note else '') + "\n".join(rows) + "\n"
    (OUT/f"{name}.tex").write_text(text)
    print(f"tables/{name}.tex  ({len([r for r in rows if r.endswith(chr(92)*2)])} rows)")


REG = {'good trt / good FU': ('Good', 'Good'), 'good trt / poor FU': ('Good', 'Limited'),
       'poor trt / good FU': ('Limited', 'Good'), 'poor trt / poor FU': ('Limited', 'Limited')}

# First we build the tables for the main simulation study.
s = pd.read_csv(SIM/'factorial_R2000_summary.csv')
vr = lambda x: f"{x.var_ratio_OW:.2f} [{x.var_ratio_lo:.2f}, {x.var_ratio_hi:.2f}]"
rows = []
for tau in (1.0, 3.0, 5.0):
    first = True
    for r, (to, fo) in REG.items():
        g = s[(s.tau == tau) & (s.regime == r)].set_index('tilt'); o, c = g.loc['OW'], g.loc['CO c=0.1']
        lead = f"{int(tau)}" if first else ""
        cells = [lead, to, fo, f"{o.G1_q05:.3f}"]
        for x in (o, c):
            cells += [num(x.bias*100), f"{x.sd*100:.1f}", f"{x.coverage*100:.1f}", f"{x.kurt4:.1f}", f"{x.tail_index:.2f}"]
        cells += [vr(c), f"{c.tv:.2f}"]
        rows.append(" & ".join(cells) + r"\\"); first = False
    if tau < 5:
        rows.append(r"\addlinespace")
write('table1_simulation', rows,
      'tau & Treatment & Follow-up & G1_q05 & OW: Bias SD Cover Kurt TI & CO c=0.1: Bias SD Cover Kurt TI & VR & TV')

LAB = [('IPW', 'IPW'), ('OW', 'OW'), ('CO c=0.1', 'CO, $c=0.1$'), ('CO c=0.3', 'CO, $c=0.3$'), ('CO c=1.0', 'CO, $c=1$'),
       ('naive e(1-e)G1G0', 'Naive $e(1-e)G_1G_0$'), ('OW trunc fix0.05', r'OW, $\hat G\ge0.05$'),
       ('OW trunc fix0.10', r'OW, $\hat G\ge0.10$'), ('OW trunc pct1', r'OW, 1\% floor on $\hat G$'),
       ('OW trunc pct5', r'OW, 5\% floor on $\hat G$')]
TXT = {'good trt / good FU': ('good', 'good'), 'good trt / poor FU': ('good', 'limited'),
       'poor trt / good FU': ('limited', 'good'), 'poor trt / poor FU': ('limited', 'limited')}
rows = []
for tau in (1.0, 3.0, 5.0):
    for r, (to, fo) in TXT.items():
        if rows:
            rows.append(r"\addlinespace")
        rows.append(rf"\multicolumn{{9}}{{l}}{{\emph{{$\tau={int(tau)}$; treatment overlap {to}; follow-up overlap {fo}}}}}\\")
        g = s[(s.tau == tau) & (s.regime == r)].set_index('tilt')
        for k, lab in LAB:
            x = g.loc[k]
            rows.append(f"{lab} & {num(x.bias*100, 2)} ({x.bias_mcse*100:.2f}) & {x.sd*100:.2f} & {x.rmse*100:.2f} & "
                        f"{x.coverage*100:.1f} ({x.coverage_mcse*100:.1f}) & {x.kurt4:.1f} & {x.tail_index:.2f} & "
                        f"{'1' if k == 'OW' else vr(x)} & {x.tv:.2f}" + r"\\")
write('tableD1_simulation_all', rows, 'Estimator & Bias (MCSE) & SD & RMSE & Cover (MCSE) & Kurt & TI & VR & TV')


# The auxiliary scripts save compact text output, so we parse those blocks here.
def blocks(path):
    """Turn each titled output block into a small data frame indexed by tilt."""
    out = {}
    for blk in path.read_text().split('=== ')[1:]:
        lines = [l for l in blk.splitlines() if l.strip()]
        rec = {}
        for l in lines[2:]:
            p = l.split()
            rec[' '.join(p[:-7])] = dict(bias=float(p[-7]), sd=float(p[-6]), rmse=float(p[-5]),
                                         cover=float(p[-4].rstrip('%')), kurt=float(p[-3]), tv=float(p[-2]), psi=float(p[-1]))
        out[lines[0]] = pd.DataFrame(rec).T
    return out


def aux_rows(title, g, keys):
    rows = [rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}}\\"]
    sdow = f"{g.loc['OW', 'sd']:.3f}"
    for k, lab in keys:
        x = g.loc[k]
        vrat = (float(f"{x['sd']:.3f}")/float(sdow))**2          # from the reported (rounded) SDs
        rows.append(f"{lab} & {num(x['bias'], 3)} & {x['sd']:.3f} & {vrat:.2f} & {x['cover']:.1f} & {x['kurt']:.1f} & {x['tv']:.2f}" + r"\\")
    return rows


AUXLAB = [('OW', 'OW'), ('naive e(1-e)G1G0', 'Product'), ('CO c=0.1', 'CO, $c=0.1$'), ('CO c=0.3', 'CO, $c=0.3$'),
          ('CO c=1.0', 'CO, $c=1$'), ('OW trunc pct5', 'OW, floor 5th pct.')]

# Web Table D.2: constant target
b = blocks(SIM/'check_const.txt'); rows = []
for title, g in b.items():
    tau = re.search(r'tau=(\d)', title).group(1); trt = 'good' if 'good trt' in title else 'limited'
    if rows:
        rows.append(r"\addlinespace")
    rows += aux_rows(f"$\\tau={tau}$; treatment overlap {trt}", g, AUXLAB)
write('tableD2_constant_target', rows, 'Estimator & Bias & SD & VR & Cover & Kurt & TV  (VR from the reported SDs)')

# Web Table D.3: misjudged follow-up (censoring model flat above X2 = k)
b = blocks(SIM/'check_r4.txt'); rows = []
for title, g in b.items():
    k = re.search(r'X2>(\S+)', title).group(1); mass = float(re.search(r'mass ([\d.]+)', title).group(1))
    cells = ['none' if k == 'None' else k, f"{mass:.3f}"]
    for t in ('OW', 'CO c=0.1', 'CO c=1.0'):
        x = g.loc[t]; cells += [num(x['bias'], 3), f"{x['sd']:.3f}", f"{x['cover']:.1f}"]
    rows.append(" & ".join(cells) + r"\\")
write('tableD3_misjudged_followup', rows, 'k & Mass & OW: Bias SD Cover & CO c=0.1: Bias SD Cover & CO c=1: Bias SD Cover')

# Web Table D.4: alignment of treatment and follow-up information
rows = []
for blk in (SIM/'check_alignment.txt').read_text().split('\n\n'):
    m = re.search(r'rho=([+-][\d.]+) \| OW weak-follow-up mass \(min G<0.10\) ([\d.]+) \| G1 q05 ([\d.]+)', blk)
    if not m:
        continue
    rec = {}
    for l in blk.splitlines()[2:]:
        p = l.split()
        if len(p) >= 7:
            rec[' '.join(p[:-6])] = p[-6:]
    if rows:
        rows.append(r"\addlinespace")
    first = True
    for k, lab in [('OW', 'OW'), ('CO c=0.1', 'CO, $c=0.1$'), ('CO c=0.3', 'CO, $c=0.3$'), ('CO c=1.0', 'CO, $c=1$'),
                   ('naive e(1-e)G1G0', 'Product')]:
        v, ess, tv, bias, cov, kurt = rec[k]
        lead = f"{num(float(m.group(1)), 1)} & {m.group(2)} & {m.group(3)}" if first else " & & "
        rows.append(f"{lead} & {lab} & {v} & {ess} & {tv} & {num(float(bias), 3)} & {cov.rstrip('%')} & {kurt}" + r"\\")
        first = False
write('tableD4_alignment', rows, 'rho & Weak FU & G1_q05 & Estimator & VR & ESS & TV & Bias & Cover & Kurt')

rows = []
for blk in (SIM/'check_rare_treatment.txt').read_text().split('rare treatment')[1:]:
    m = re.search(r'rho=([\d.]+): OW weak-FU mass ([\d.]+), G1 q05 ([\d.]+)', blk)
    if rows:
        rows.append(r"\addlinespace")
    first = True
    for l in blk.splitlines()[1:]:
        mm = re.match(r'\s+(.+?)\s+Var/VarOW\s+([\d.]+)\s+IQR\^2 ratio\s+([\d.]+)\s+kurt\s+([\d.]+)\s+TV ([\d.]+)', l)
        if mm:
            lab = dict(AUXLAB).get(mm.group(1))
            lead = f"{float(m.group(1)):.2f} & {m.group(2)} & {m.group(3)}" if first else " & & "
            rows.append(f"{lead} & {lab} & {mm.group(2)} & {mm.group(3)} & {mm.group(4)} & {mm.group(5)}" + r"\\")
            first = False
write('tableD4b_rare_treatment', rows, 'rho & Weak FU & G1_q05 & Estimator & VR & IQR^2 ratio & Kurt & TV')

# Web Table D.5: censoring-model sensitivity (event model correctly specified)
blk = (SIM/'check_censmodel.txt').read_text().split(' event model: ')[1]; rows = []
for l in blk.splitlines()[2:]:
    p = l.split()
    if len(p) >= 7:
        k = ' '.join(p[:-6]); b1, b2, b3, sd, d1, d2 = p[-6:]
        rows.append(f"{dict(AUXLAB)[k]} & {num(float(b1), 3)} & {num(float(b2), 3)} & {num(float(b3), 3)} & {sd} & {d1} & {d2}" + r"\\")
order = ['OW', 'CO, $c=0.1$', 'CO, $c=0.3$', 'CO, $c=1$', 'Product']
rows.sort(key=lambda r: order.index(r.split(' & ')[0]))
write('tableD5_censoring_model', rows, 'Estimator & Bias: correct, linear, flat tail & SD & Mean |change|: linear, flat tail')

# ---------------- Web Table A.4: bounded-support sequence ----------------
bs = {}
for l in (TC/'bounded_support.txt').read_text().splitlines()[1:]:
    p = l.split()
    bs[(p[0], int(p[1]))] = (float(p[2]), float(p[3]), float(p[4]))
rows = []
for b in (2, 3, 4, 5, 6):
    r_, l_ = bs[('randomized', b)], bs[('logistic', b)]
    rows.append(f"{b} & {sci(r_[0])} & {r_[1]:.2f} & {r_[2]:.3f} & {sci(l_[0])} & {l_[1]:.2f} & {l_[2]:.3f}" + r"\\")
write('tableA4_bounded_support', rows, 'b & randomized: V(h_OW) V(h_c) Mass & logistic: V(h_OW) V(h_c) Mass')

# Finally, we create the tables for the Rotterdam application.
RLAB = [('OW', 'OW'), ('CO c=0.1', 'CO, $c=0.1$'), ('CO c=0.3', 'CO, $c=0.3$'), ('CO c=1.0', 'CO, $c=1$'),
        ('naive', 'Product $e(1-e)G_1G_0$')]
est = pd.read_csv(RD/'rotterdam_est_hormon_Cox.csv'); boot = pd.read_csv(RD/'rotterdam_bootstrap_summary.csv').set_index('tau')
rows = []
for tau in (5.0, 7.0, 8.0):
    g = est[est.tau == tau].set_index('tilt'); first = True
    for k, lab in RLAB:
        x = g.loc[k]
        rows.append(f"{int(tau) if first else ''} & {lab} & {num(x.psi, 3)} & {x.se:.3f} & ({num(x.lo, 3)}, {num(x.hi, 3)}) & "
                    f"{x.mass:.2f} & {x.TV:.2f}" + r"\\"); first = False
    bb = boot.loc[tau]
    rows.append(rf" & \multicolumn{{6}}{{l}}{{\emph{{Bootstrap: {bb.R_median:.2f} ({bb.R_lo:.2f}, {bb.R_hi:.2f}); "
                rf"proportion above 1: {bb.P_R_gt_1:.2f}}}}}\\")
    if tau < 8:
        rows.append(r"\addlinespace")
write('table2_rotterdam', rows, 'tau & Target & psi & SE & 95% interval & Mass & TV')

d = pd.read_csv(RD/'rotterdam_design_table.csv'); rows = []
for trt in ('hormon', 'chemo'):
    for cm in ('Cox', 'KM-by-year'):
        for _, x in d[(d.treatment == trt) & (d.censoring == cm)].iterrows():
            rows.append(f"{'Hormonal' if trt == 'hormon' else 'Chemotherapy'} & {'Cox' if cm == 'Cox' else 'KM by year'} & "
                        f"{int(x.tau)} & {nz(x.G1_q05)} & {nz(x.G0_q05)} & {x.equip_G1_lt10*100:.1f} & {x.equip_G1_lt05*100:.1f} & "
                        f"{nz(x['mass CO c=0.1'], 2)} & {nz(x['mass CO c=1.0'], 2)} & {nz(x['mass naive'], 2)} & "
                        f"{nz(x['TV CO c=0.1'], 2)}" + r"\\")
meta = d.drop_duplicates('treatment').set_index('treatment')
write('tableE2_design', rows,
      'Treatment & Censoring & tau & G1_q05 & G0_q05 & Equipoise % G1<.10 & G1<.05 & Mass CO .1, CO 1, Product & TV (CO .1)\n'
      f"% equipoise n: hormonal {int(meta.loc['hormon', 'n_equip'])}, chemotherapy {int(meta.loc['chemo', 'n_equip'])}; "
      f"PS in (0.1, 0.9): hormonal {meta.loc['hormon', 'PS_in_01_09']:.0%}, chemotherapy {meta.loc['chemo', 'PS_in_01_09']:.0%}")

SLAB = [('OW', 'OW'), ('CO c=0.1', 'CO, $c=.1$'), ('CO c=0.3', 'CO, $c=.3$'), ('CO c=1.0', 'CO, $c=1$'), ('naive', 'Product')]


def sens_rows(e, lead_fn, taus):
    rows = []
    for tau in taus:
        g = e[e.tau == tau].set_index('tilt'); first = True
        for k, lab in SLAB:
            x = g.loc[k]
            rows.append(f"{lead_fn(tau) if first else ''} & {lab} & {nz(x.psi)} & {nz(x.se)} & ({nz(x.lo)},{nz(x.hi)}) & "
                        f"{nz(x.mass, 2)} & {nz(x.TV, 2)} & {x.kurt_IF:.1f} & {x.top10_share*100:.1f}" + r"\\"); first = False
        rows.append(r"\addlinespace")
    return rows[:-1]


write('tableE3_hormonal_KM', sens_rows(pd.read_csv(RD/'rotterdam_est_hormon_KM-year.csv'), lambda t: int(t), (5.0, 7.0, 8.0)),
      'tau & Target & psi & SE & 95% interval & Mass & TV & Kurt(IF) & Top 10')
rows = []
for f, lab in (('rotterdam_est_chemo_Cox.csv', 'Cox'), ('rotterdam_est_chemo_KM-year.csv', 'KM')):
    rows += sens_rows(pd.read_csv(RD/f), lambda t, lab=lab: lab, (8.0,)) + [r"\addlinespace"]
write('tableE3_chemotherapy_tau8', rows[:-1], 'Model & Target & psi & SE & 95% interval & Mass & TV & Kurt(IF) & Top 10')

dec = pd.read_csv(RD/'rotterdam_if_decomposition.csv'); rows = []
for _, x in dec.iterrows():
    rows.append(f"{int(x.tau)} & {'OW' if x.tilt == 'OW' else 'CO, $c=.1$'} & {nz(x.var_X, 5)} & {nz(x.var_A, 5)} & "
                f"{nz(x.var_T, 5)} & {nz(x.var_C, 5)} & {nz(x.var_total, 5)} & {nz(x.cross_share*100, 2)}" + r"\\")
write('tableE4_variance_decomposition', rows, 'tau & Target & D_X & D_A & D_T & D_C & Total & Cross (%)')

rows = []
for tau, x in boot.iterrows():
    rows.append(f"{int(tau)} & {int(x.n)} & {x.R_median:.2f} ({x.R_lo:.2f},{x.R_hi:.2f}) & {nz(x.P_R_gt_1)} & "
                f"{nz(x.ESSr_median, 2)} ({nz(x.ESSr_lo, 2)},{nz(x.ESSr_hi, 2)}) & {x.boot_var_ratio:.2f} & "
                f"{x.kurt_OW:.1f} & {x.kurt_CO:.1f}" + r"\\")
write('tableE5_bootstrap', rows, 'tau & Usable & Median R_b (2.5%, 97.5%) & P(R_b > 1) & Median ESS1 ratio & Boot VR & Kurt OW & Kurt CO')
