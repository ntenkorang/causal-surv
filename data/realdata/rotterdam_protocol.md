# Rotterdam application - frozen analysis protocol

Frozen 2026-09-23, before any treatment-effect estimate was computed. Amendment 1 was
agreed after the design-stage diagnostics (rotterdam_design_table.csv) and before estimation.

## Question
How does accounting for limited follow-up overlap change the target population, precision
and stability of an RMST analysis? This is not a definitive causal study of hormonal therapy.

## Hierarchy
- Primary: hormonal therapy; CO with c = 0.1; tau = 5, 7, 8 years; censoring model =
  arm-specific Cox with surgery year (cubic spline) and the baseline covariates.
- Prespecified contrasts: OW; CO with c = 0.3 and c = 1 (follow-up-protection path).
- Comparators: IPW; naive e(1-e) G1 G0.
- Sensitivity: year-stratified Kaplan-Meier censoring model (pooled over arms); chemotherapy
  as the treatment.
- Excluded a priori: tau = 10 (structural loss of follow-up in the latest surgery years).

## Models
- Propensity: logistic; splines for surgery year and age; meno, grade, size, log(1+nodes),
  log(1+pgr), log(1+er), and the other adjuvant therapy.
- Event: Cox per arm, same covariates.
- Estimator: one-step AIPCW with the full influence function (tilt corrections for e and G);
  Wald 95% intervals from the empirical influence function. No cross-fitting (semiparametric
  nuisance models).

## Amendment 1
Primary censoring model changed from year-stratified Kaplan-Meier to the arm-specific
covariate-adjusted Cox model, because year-specific KM is unstable at tau = 8 for the latest
surgery years and forces G0 = G1. Year-stratified KM becomes the sensitivity analysis.
c = 0.1 designated primary on simulation grounds, before outcomes were seen.

## Addendum 1: bootstrap of the precision comparison (declared before running)
Purpose: assess whether Var(CO c=0.1)/Var(OW) = (0.180/0.150)^2 = 1.44 at tau = 8 is stable
under resampling. No new models, tilts, horizons or censoring specifications.
- B = 1,000 subject-level bootstrap samples (seed 20260923), primary specification only
  (hormonal therapy, arm-specific Cox censoring model).
- Each replicate refits the whole frozen pipeline: propensity, censoring and event models,
  tilts, one-step estimates and full influence-function SEs.
- Saved per replicate and tau in {5, 7, 8}: R_b = SE_b(CO 0.1)^2 / SE_b(OW)^2 and
  ESS_1(CO 0.1) / ESS_1(OW) (treated-arm effective sample size), plus both point estimates.
- Reported: median R_b, 2.5-97.5% percentile interval, and P_boot(R_b > 1) as a descriptive
  measure of how consistently the ordering appears (not a p-value).
