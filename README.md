# Prudential Monetary Policy under Downward Nominal Wage Rigidity

A numerical analysis of optimal monetary policy when downward nominal wage rigidity creates an intertemporal trade-off between current inflation stabilization and future unemployment.

This project combines analytical macroeconomic derivations, dynamic programming, numerical optimization, Monte Carlo simulation, welfare analysis, and event-study methods.

Developed by **Gabriele Casto** and **Francesco Donghi** as part of the *Advanced Macroeconomics III* course at the **University of St. Gallen**.

---

## Overview

The project studies monetary policy in an economy with **downward nominal wage rigidity (DNWR)**.

The central mechanism is inherently dynamic: inflation today affects the nominal wage benchmark inherited by the economy tomorrow. When productivity subsequently falls, a high inherited wage level can make downward wage rigidity binding, forcing the economy to adjust through inflation, unemployment, or both.

This creates a prudential motive for monetary policy.

Rather than focusing only on contemporaneous inflation and unemployment, an optimal central bank internalizes how today's policy decisions affect future labor-market constraints.

The analysis proceeds from deterministic one-time productivity shocks to a fully stochastic dynamic model with repeated productivity shocks.

---

## Economic Mechanism

Let the state of the economy be

\[
s_t = (a_t, a_{t-1}, \pi_{t-1}),
\]

where:

- \(a_t\) is current productivity,
- \(a_{t-1}\) is lagged productivity,
- \(\pi_{t-1}\) is inherited inflation.

Downward nominal wage rigidity implies that unemployment is determined by

\[
u_t =
\max\left\{
0,
\frac{\lambda \pi_{t-1}-\pi_t-a_t+a_{t-1}}{\xi}
\right\}.
\]

Under optimal policy, the inflation-unemployment trade-off can be summarized by the target rule

\[
1-e^{-u_t}
=
\xi \kappa \pi_t
+
\beta \lambda
E_t\left[1-e^{-u_{t+1}}\right].
\]

The final term is the key **prudential component**.

Higher inflation today raises the indexed wage benchmark inherited next period. If productivity subsequently deteriorates, this can tighten the wage-rigidity constraint and increase future unemployment.

The optimal central bank therefore has an incentive to restrain inflation during strong economic conditions in order to reduce future wage pressure.

---

## Methodology

The project combines analytical and numerical methods:

- analytical impulse-response derivations for one-time productivity shocks;
- Taylor-rule and optimal-policy comparisons;
- AS-AD analysis;
- constrained numerical optimization;
- dynamic programming;
- value function iteration;
- Gaussian quadrature for conditional expectations;
- multidimensional interpolation of policy functions;
- Monte Carlo simulation;
- welfare comparisons;
- event studies around episodes in which downward wage rigidity becomes binding.

The stochastic model is solved over the three-dimensional state space

\[
(a_t, a_{t-1}, \pi_{t-1}),
\]

with inflation as the policy choice and unemployment determined by the DNWR constraint.

---

## Selected Results

### Optimal inflation policy

The numerical solution shows how optimal inflation varies with current productivity and inherited economic conditions.

<p align="center">
  <img src="figures/optimal_inflation_policy.png" width="850">
</p>

In boom states, optimal inflation remains close to zero and can become slightly negative. The central bank avoids using strong productivity conditions to generate additional inflation because doing so would increase the wage benchmark inherited by the future economy.

---

### Prudential policy around binding wage-rigidity episodes

The event study compares the optimal prudential policy with a counterfactual policy that optimally trades off current inflation and unemployment but ignores the effect of current inflation on future wage pressure.

<p align="center">
  <img src="figures/prudential_policy_event_study.png" width="850">
</p>

Before episodes in which wage rigidity becomes binding, the prudential regime enters with lower inflation and lower inherited wage pressure.

The policy does not mechanically minimize unemployment in every period. Instead, it improves the intertemporal allocation of inflation and labor-market distortions.

---

### Implementing nominal interest rate

The nominal interest rate consistent with each allocation is recovered from the Euler equation.

<p align="center">
  <img src="figures/nominal_interest_rate_event_study.png" width="700">
</p>

The interest-rate path reflects the different inflation and output allocations implemented under prudential and no-prudential monetary policy.

---

## Quantitative Comparison

For the baseline stochastic simulation, the two policy regimes are exposed to the **same sequence of productivity shocks**.

| Moment | Prudential | No-prudential |
|---|---:|---:|
| Mean unemployment | 0.073366 | 0.065676 |
| Frequency of binding DNWR | 0.582910 | 0.623610 |
| Inflation volatility | 0.132557 | 0.143697 |
| Average welfare flow | -1.001383 | -1.004293 |

The results highlight an important trade-off.

Prudential monetary policy:

- **raises average welfare**;
- **reduces inflation volatility**;
- **reduces the frequency of binding wage-rigidity episodes**;
- but accepts greater labor-market adjustment in some states.

Relative to the no-prudential economy, the prudential policy recovers approximately **20.9% of the welfare gap to the first-best allocation** under the baseline calibration.

These values are model-based simulation results and should be interpreted as comparisons across policy regimes rather than empirical estimates.

---

## One-Time Productivity Shocks

The project also studies the model analytically under deterministic productivity shocks.

### Negative productivity shock

Under a Taylor rule, a negative productivity shock is stagflationary: output falls, unemployment rises, and inflation increases.

<p align="center">
  <img src="figures/negative_shock_taylor_rule.png" width="750">
</p>

Optimal policy trades off inflation stabilization against unemployment:

<p align="center">
  <img src="figures/negative_shock_optimal_policy.png" width="750">
</p>

The corresponding AS-AD representation illustrates how the productivity shock shifts aggregate supply and how different monetary-policy regimes determine aggregate demand:

<p align="center">
  <img src="figures/negative_shock_as_ad.png" width="800">
</p>

### Positive productivity shock

A temporary positive productivity shock can generate a boom-bust pattern when the economy inherits a high nominal wage level after productivity returns to normal.

<p align="center">
  <img src="figures/positive_shock_taylor_rule.png" width="750">
</p>

Optimal monetary policy behaves prudentially by restraining inflation during the boom, reducing the severity of the subsequent adjustment:

<p align="center">
  <img src="figures/positive_shock_optimal_policy.png" width="750">
</p>

---

## Repository Structure

```text
Optimal-Monetary-Policy/
│
├── prudential_monetary_policy.py
├── README.md
│
├── figures/
│   ├── negative_shock_taylor_rule.png
│   ├── negative_shock_taylor_rule.pdf
│   ├── negative_shock_optimal_policy.png
│   ├── negative_shock_optimal_policy.pdf
│   ├── negative_shock_as_ad.png
│   ├── negative_shock_as_ad.pdf
│   ├── positive_shock_taylor_rule.png
│   ├── positive_shock_taylor_rule.pdf
│   ├── positive_shock_optimal_policy.png
│   ├── positive_shock_optimal_policy.pdf
│   ├── optimal_inflation_policy.png
│   ├── optimal_inflation_policy.pdf
│   ├── prudential_policy_event_study.png
│   ├── prudential_policy_event_study.pdf
│   ├── nominal_interest_rate_event_study.png
│   └── nominal_interest_rate_event_study.pdf
│
├── results/
│   ├── unconditional_moments.csv
│   ├── unconditional_moments.tex
│   ├── prudential_policy_event_study.csv
│   └── nominal_interest_rate_event_study.csv
│
└── report/
    └── prudential_monetary_policy_report.pdf
