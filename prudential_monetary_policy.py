"""Computational analysis of prudential monetary policy under downward nominal wage rigidity.

This script accompanies the Advanced Macroeconomics III project and contains the
numerical routines used to produce the project's impulse responses, AS-AD diagram,
dynamic policy functions, long-run simulations, and event studies.

Questions 4 and 5 are intentionally left as placeholders in this version and will
be added separately.

Model variables are expressed as log-linear deviations unless explicitly converted
to levels. Figures are exported in both PDF and PNG formats.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

# =============================================================================
# QUESTION 1 — ONE-TIME NEGATIVE PRODUCTIVITY SHOCK: TAYLOR RULE
# =============================================================================

# Model calibration
a0 = -1.0         # negative productivity shock
phi_pi = 1.5      # Taylor-rule coefficient on inflation, phi_pi > 1
phi_u = 0.5       # Taylor-rule coefficient on unemployment, phi_u > 0
xi = 1.0          # slope of wage Phillips curve, xi > 0
lam = 0.5         # indexation to past inflation, 0 <= lambda < 1

# Simulation horizon
T = 8
time = np.arange(-1, T + 1)   # display periods -1, 0, 1, ..., T

# Initialize response paths
y = np.zeros(len(time))
u = np.zeros(len(time))
pi = np.zeros(len(time))
pi_w = np.zeros(len(time))

# Event-time indices
idx_0 = np.where(time == 0)[0][0]
idx_1 = np.where(time == 1)[0][0]

# Initial condition: t = -1
# Already initialized at zero:
# y_{-1} = u_{-1} = pi_{-1} = pi^W_{-1} = 0

# Impact response: t = 0
# Downward nominal wage rigidity binds on impact.
denominator = 1.0 + phi_u + phi_pi * xi

u[idx_0] = - (phi_pi - 1.0) * a0 / denominator
pi[idx_0] = - (1.0 + phi_u + xi) * a0 / denominator
y[idx_0] = (phi_u + phi_pi * (1.0 + xi)) * a0 / denominator
pi_w[idx_0] = xi * (phi_pi - 1.0) * a0 / denominator

# Recovery period: t = 1
# The wage-rigidity constraint is slack.
y[idx_1] = 0.0
u[idx_1] = 0.0
pi[idx_1] = 0.0
pi_w[idx_1] = -a0

# Return to steady state: t >= 2
# Already initialized at zero.

# Consistency checks
slack_condition_t1 = pi_w[idx_1] >= lam * pi[idx_0]

print("Slack condition in t=1 holds:", slack_condition_t1)
print("pi_w[1] =", round(pi_w[idx_1], 4))
print("lambda*pi[0] =", round(lam * pi[idx_0], 4))

# Figure style

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

green = "#006400"  # dark green, similar to lecture-slide style

fig, axs = plt.subplots(2, 2, figsize=(9, 6))

series = [
    (y, r"Output $y_t$"),
    (u, r"Unemployment $u_t$"),
    (pi, r"Inflation $\pi_t$"),
    (pi_w, r"Wage inflation $\pi_t^W$"),
]

for ax, (data, title) in zip(axs.flatten(), series):
    ax.plot(time, data, color=green, linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)

    ax.set_title(title, pad=10)
    ax.set_ylabel("linear deviation")
    ax.set_xlim(time[0], time[-1])
    ax.set_xticks(time)

    # Automatic y-limits with padding
    lower = min(data.min(), 0.0)
    upper = max(data.max(), 0.0)
    padding = 0.15 * max(upper - lower, 1.0)
    ax.set_ylim(lower - padding, upper + padding)

    ax.grid(False)

# Label only the bottom panels to keep the figure compact.
axs[1, 0].set_xlabel("Time")
axs[1, 1].set_xlabel("Time")

# Figure layout
plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.90,
    bottom=0.12,
    wspace=0.35,
    hspace=0.55
)

# Export figure
plt.savefig("irf_question1.pdf", bbox_inches="tight")
plt.savefig("irf_question1.png", dpi=300, bbox_inches="tight")

plt.show()

# Report numerical values
print("\nIRF values")
print("time:", time)
print("y:", np.round(y, 4))
print("u:", np.round(u, 4))
print("pi:", np.round(pi, 4))
print("pi_w:", np.round(pi_w, 4))


# =============================================================================
# QUESTION 2 — ONE-TIME NEGATIVE PRODUCTIVITY SHOCK: OPTIMAL POLICY
# =============================================================================

# Inflation-cost parameter
kappa = 0.5        # kappa > 0: inflation is costly under optimal policy
include_kappa_zero = True   # set False if you only want the kappa > 0 case

# Solve the impact-period optimum
# FOC: 1 - exp(-u0*) = kappa * xi * (-a0 - xi*u0*)
# Domain: 0 <= u0* <= -a0/xi
def solve_optimal_unemployment_negative_shock(a0, xi, kappa, tolerance=1e-12, max_iterations=10_000):
    """
    Solves the optimal unemployment response u0* to a one-time negative
    productivity shock under optimal monetary policy.

    For kappa > 0, the solution is interior.
    For kappa = 0, the central bank chooses full employment, u0* = 0.
    """
    if a0 >= 0:
        raise ValueError("This block is for a negative productivity shock, so a0 must be negative.")
    if xi <= 0:
        raise ValueError("xi must be strictly positive.")
    if kappa < 0:
        raise ValueError("kappa must be non-negative.")

    # Boundary case: inflation has no direct welfare cost
    if kappa == 0:
        return 0.0

    upper_bound = -a0 / xi

    def first_order_condition(u0):
        pi0 = -a0 - xi * u0
        return 1.0 - np.exp(-u0) - kappa * xi * pi0

    # Bisection on [0, -a0/xi]
    lower_bound = 0.0
    lower_value = first_order_condition(lower_bound)
    upper_value = first_order_condition(upper_bound)

    if lower_value > 0 or upper_value < 0:
        raise RuntimeError("The bisection bracket is not valid. Check parameters.")

    for _ in range(max_iterations):
        midpoint = 0.5 * (lower_bound + upper_bound)
        midpoint_value = first_order_condition(midpoint)

        if abs(midpoint_value) < tolerance:
            return midpoint

        if midpoint_value > 0:
            upper_bound = midpoint
        else:
            lower_bound = midpoint

    return 0.5 * (lower_bound + upper_bound)


# Construct the optimal-policy response
def build_optimal_irf_negative_shock(a0, xi, lam, kappa, time, idx_0, idx_1):
    """
    Builds the optimal-policy IRF for y_t, u_t, pi_t and pi_t^W.
    """
    y_opt = np.zeros(len(time))
    u_opt = np.zeros(len(time))
    pi_opt = np.zeros(len(time))
    pi_w_opt = np.zeros(len(time))

    # Period t = 0
    u_star = solve_optimal_unemployment_negative_shock(a0, xi, kappa)
    pi_star = -a0 - xi * u_star

    u_opt[idx_0] = u_star
    pi_opt[idx_0] = pi_star
    y_opt[idx_0] = a0 - u_star
    pi_w_opt[idx_0] = -xi * u_star

    # Period t = 1
    # Productivity recovers, wage constraint is slack
    y_opt[idx_1] = 0.0
    u_opt[idx_1] = 0.0
    pi_opt[idx_1] = 0.0
    pi_w_opt[idx_1] = -a0

    # Periods t >= 2 are already initialized at zero
    slack_condition_t1 = pi_w_opt[idx_1] >= lam * pi_opt[idx_0]

    return y_opt, u_opt, pi_opt, pi_w_opt, u_star, pi_star, slack_condition_t1


# Main case: kappa > 0
y_opt, u_opt, pi_opt, pi_w_opt, u_star, pi_star, slack_condition_opt_t1 = (
    build_optimal_irf_negative_shock(a0, xi, lam, kappa, time, idx_0, idx_1)
)

print("\nQuestion 2: optimal policy, kappa > 0")
print("kappa =", kappa)
print("u0* =", round(u_star, 6))
print("pi0* =", round(pi_star, 6))
print("y0* =", round(y_opt[idx_0], 6))
print("pi_w0* =", round(pi_w_opt[idx_0], 6))
print("Slack condition in t=1 holds:", slack_condition_opt_t1)
print("pi_w[1] =", round(pi_w_opt[idx_1], 6))
print("lambda*pi[0] =", round(lam * pi_opt[idx_0], 6))
print("Implementing nominal rate in t=0: i0* = rho - a0 + u0*")


# Benchmark case: kappa = 0
if include_kappa_zero:
    y_opt_k0, u_opt_k0, pi_opt_k0, pi_w_opt_k0, u_star_k0, pi_star_k0, slack_condition_k0_t1 = (
        build_optimal_irf_negative_shock(a0, xi, lam, 0.0, time, idx_0, idx_1)
    )

    print("\nQuestion 2: boundary case, kappa = 0")
    print("u0* =", round(u_star_k0, 6))
    print("pi0* =", round(pi_star_k0, 6))
    print("y0* =", round(y_opt_k0[idx_0], 6))
    print("pi_w0* =", round(pi_w_opt_k0[idx_0], 6))
    print("Slack condition in t=1 holds:", slack_condition_k0_t1)


# Figure: Taylor rule vs. optimal policy

fig, axs = plt.subplots(2, 2, figsize=(9, 6))

series_comparison = [
    (y, y_opt, r"Output $y_t$"),
    (u, u_opt, r"Unemployment $u_t$"),
    (pi, pi_opt, r"Inflation $\pi_t$"),
    (pi_w, pi_w_opt, r"Wage inflation $\pi_t^W$"),
]

for ax, (data_taylor, data_optimal, title) in zip(axs.flatten(), series_comparison):
    ax.plot(time, data_taylor, color=green, linewidth=2.5, label="Taylor rule")
    ax.plot(time, data_optimal, color="#990001", linewidth=2.5, linestyle="-",
        label=rf"Optimal policy, $\kappa={kappa}$")

    if include_kappa_zero:
        if title == r"Output $y_t$":
            data_k0 = y_opt_k0
        elif title == r"Unemployment $u_t$":
            data_k0 = u_opt_k0
        elif title == r"Inflation $\pi_t$":
            data_k0 = pi_opt_k0
        else:
            data_k0 = pi_w_opt_k0

        ax.plot(time, data_k0, color="blue", linewidth=2.0, linestyle="-",
        label=r"Optimal policy, $\kappa=0$")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)

    ax.set_title(title, pad=10)
    ax.set_ylabel("linear deviation")
    ax.set_xlim(time[0], time[-1])
    ax.set_xticks(time)

    # Automatic y-limits with padding across all displayed series
    all_data = [data_taylor, data_optimal]
    if include_kappa_zero:
        all_data.append(data_k0)

    lower = min(np.min(d) for d in all_data)
    upper = max(np.max(d) for d in all_data)
    lower = min(lower, 0.0)
    upper = max(upper, 0.0)
    padding = 0.15 * max(upper - lower, 1.0)
    ax.set_ylim(lower - padding, upper + padding)

    ax.grid(False)

# Label only the bottom panels to keep the figure compact.
axs[1, 0].set_xlabel("Time")
axs[1, 1].set_xlabel("Time")

# Single legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)

plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.90,
    bottom=0.18,
    wspace=0.35,
    hspace=0.55
)

plt.savefig("irf_question2.pdf", bbox_inches="tight")
plt.savefig("irf_question2.png", dpi=300, bbox_inches="tight")

plt.show()


# Report numerical values
print("\nOptimal-policy IRF values, kappa > 0")
print("time:", time)
print("y_opt:", np.round(y_opt, 4))
print("u_opt:", np.round(u_opt, 4))
print("pi_opt:", np.round(pi_opt, 4))
print("pi_w_opt:", np.round(pi_w_opt, 4))

if include_kappa_zero:
    print("\nOptimal-policy IRF values, kappa = 0")
    print("y_opt_k0:", np.round(y_opt_k0, 4))
    print("u_opt_k0:", np.round(u_opt_k0, 4))
    print("pi_opt_k0:", np.round(pi_opt_k0, 4))
    print("pi_w_opt_k0:", np.round(pi_w_opt_k0, 4))


# =============================================================================
# QUESTION 3 — AS–AD DIAGRAM AFTER A NEGATIVE PRODUCTIVITY SHOCK
# =============================================================================

# Convert log deviations to levels
# We interpret a_t, y_t and pi_t as log deviations.
# Therefore, levels are obtained using exp(.).
A_0 = np.exp(a0)

# Initial steady state
Y_initial = 1.0
P_initial = 1.0

# Taylor-rule equilibrium
Y_taylor = np.exp(y[idx_0])
P_taylor = np.exp(pi[idx_0])

# Optimal-policy equilibrium, kappa > 0
Y_optimal = np.exp(y_opt[idx_0])
P_optimal = np.exp(pi_opt[idx_0])

# Full-accommodation benchmark, kappa = 0
if include_kappa_zero:
    Y_optimal_k0 = np.exp(y_opt_k0[idx_0])
    P_optimal_k0 = np.exp(pi_opt_k0[idx_0])


# Aggregate-supply and aggregate-demand schedules
# Initial AS, before the shock:
# Binding branch: P = Y^xi for 0 < Y <= 1
# Vertical branch: Y = 1 for P >= 1
def initial_as_price(output_grid):
    return output_grid ** xi


# Post-shock AS, after a0 < 0:
# Binding branch:
# P_0^{AS}(Y_0) = (1 / A_0) * (Y_0 / A_0)^xi
# for 0 < Y_0 <= A_0
def post_shock_as_price(output_grid):
    return (1.0 / A_0) * (output_grid / A_0) ** xi


# AD curves calibrated to pass through the relevant equilibrium point.
# This is useful for plotting because each AD curve satisfies P * Y = constant.
def aggregate_demand_price(output_grid, equilibrium_output, equilibrium_price):
    return (equilibrium_price * equilibrium_output) / output_grid


# Plotting grids
minimum_output = 0.10
maximum_output = 1.20

output_grid = np.linspace(minimum_output, maximum_output, 500)

# Aggregate-supply branches
initial_as_grid = np.linspace(minimum_output, 1.0, 300)
post_shock_as_grid = np.linspace(minimum_output, A_0, 300)

initial_as = initial_as_price(initial_as_grid)
post_shock_as = post_shock_as_price(post_shock_as_grid)

# Aggregate-demand curves
ad_taylor = aggregate_demand_price(output_grid, Y_taylor, P_taylor)
ad_optimal = aggregate_demand_price(output_grid, Y_optimal, P_optimal)

if include_kappa_zero:
    ad_optimal_k0 = aggregate_demand_price(output_grid, Y_optimal_k0, P_optimal_k0)


# Figure
as_color = "#006400"      # dark green for AS
ad_color = "#8B0000"      # dark red / burgundy for AD
ref_color = "0.55"        # gray for reference lines

fig, ax = plt.subplots(figsize=(10, 6))

# Initial AS
initial_as_line, = ax.plot(
    initial_as_grid,
    initial_as,
    color=as_color,
    linewidth=2.0,
    linestyle="--",
    label=r"Initial AS"
)
ax.vlines(
    1.0,
    1.0,
    3.4,
    color=as_color,
    linewidth=2.0,
    linestyle="--"
)

# Post-shock AS
post_shock_as_line, = ax.plot(
    post_shock_as_grid,
    post_shock_as,
    color=as_color,
    linewidth=2.8,
    label=r"Post-shock AS"
)
ax.vlines(
    A_0,
    1.0 / A_0,
    3.4,
    color=as_color,
    linewidth=2.8
)

# Aggregate-demand curves
ad_taylor_line, = ax.plot(
    output_grid,
    ad_taylor,
    color=ad_color,
    linewidth=2.5,
    linestyle="-",
    label=r"AD, Taylor rule"
)

ad_optimal_line, = ax.plot(
    output_grid,
    ad_optimal,
    color=ad_color,
    linewidth=2.5,
    linestyle="--",
    label=rf"AD, optimal policy, $\kappa={kappa}$"
)

if include_kappa_zero:
    ad_initial_line, = ax.plot(
        output_grid,
        ad_optimal_k0,
        color=ad_color,
        linewidth=2.2,
        linestyle="-.",
        label=r"Initial AD = AD, optimal policy, $\kappa=0$"
    )

# Equilibrium markers
ax.scatter(Y_initial, P_initial, color="black", marker="o", s=50, zorder=5)
ax.scatter(Y_taylor, P_taylor, color="black", marker="o", s=50, zorder=5)
ax.scatter(Y_optimal, P_optimal, color="black", marker="o", s=50, zorder=5)

if include_kappa_zero:
    ax.scatter(Y_optimal_k0, P_optimal_k0, color="black", marker="o", s=50, zorder=5)

# Equilibrium labels
ax.annotate(
    r"$E_{-1}$",
    xy=(Y_initial, P_initial),
    xytext=(8, 8),
    textcoords="offset points"
)

ax.annotate(
    r"$E_0^T$",
    xy=(Y_taylor, P_taylor),
    xytext=(-28, -5),
    textcoords="offset points"
)

ax.annotate(
    r"$E_0^{*,\kappa=0.5}$",
    xy=(Y_optimal, P_optimal),
    xytext=(-48, -7),
    textcoords="offset points"
)

if include_kappa_zero:
    ax.annotate(
        r"$E_0^{*,\kappa=0}$",
        xy=(Y_optimal_k0, P_optimal_k0),
        xytext=(8, 8),
        textcoords="offset points"
    )

# Axes and title
ax.set_xlabel(r"Output $Y_t$")
ax.set_ylabel(r"Price level $P_t$")
ax.set_title(
    r"AS-AD diagram after a one-time negative productivity shock",
    pad=12
)

# Axis limits
ax.set_xlim(minimum_output, maximum_output)
ax.set_ylim(0.0, 3.4)

# Figure formatting
ax.grid(False)
legend_handles = [
    initial_as_line,
    post_shock_as_line,
    ad_initial_line,
    ad_taylor_line,
    ad_optimal_line
]

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
    frameon=False,
    fontsize=11.5
)

plt.tight_layout()

# Export figure
plt.savefig("as_ad_question3.pdf", bbox_inches="tight")
plt.savefig("as_ad_question3.png", dpi=300, bbox_inches="tight")

plt.show()


# Report equilibrium values
print("\nAS-AD diagram values")
print("A_0 =", round(A_0, 6))
print("Initial equilibrium: Y =", round(Y_initial, 6), ", P =", round(P_initial, 6))
print("Taylor-rule equilibrium: Y =", round(Y_taylor, 6), ", P =", round(P_taylor, 6))
print("Optimal-policy equilibrium, kappa > 0: Y =", round(Y_optimal, 6), ", P =", round(P_optimal, 6))

if include_kappa_zero:
    print(
        "Optimal-policy equilibrium, kappa = 0: Y =",
        round(Y_optimal_k0, 6),
        ", P =",
        round(P_optimal_k0, 6)
    )


# =============================================================================
# QUESTION 4 — ONE-TIME POSITIVE PRODUCTIVITY SHOCK: TAYLOR RULE
# =============================================================================
# The period-0 allocation is imposed by the assignment. From period 1 onward,
# the economy follows the Taylor-rule transition derived analytically.

# Question-specific calibration
a0 = 1.0          # positive productivity shock
phi_pi = 1.5      # Taylor-rule coefficient on inflation
phi_u = 0.5       # Taylor-rule coefficient on unemployment
xi = 1.0          # slope of wage Phillips curve
lam = 0.5         # wage indexation

T = 8             # plot periods t=0,...,8

# Ensure the figure output directory exists.
os.makedirs("figures", exist_ok=True)

# Initialize transition paths
time = np.arange(T + 1)

y = np.zeros(T + 1)
u = np.zeros(T + 1)
pi = np.zeros(T + 1)
pi_w = np.zeros(T + 1)

# Impact allocation: t = 0
y[0] = a0
u[0] = 0.0
pi[0] = 0.0
pi_w[0] = a0

# Stable root governing the bounded Taylor-rule transition
B = 1 + phi_u + lam + xi * phi_pi
disc = B**2 - 4 * (1 + xi) * (1 + phi_u) * lam
r = (B - np.sqrt(disc)) / (2 * (1 + xi))

# First adjustment period: t = 1
pi1 = ((1 + phi_u) * a0) / (
    1 + phi_u + xi * phi_pi + lam - (1 + xi) * r
)
u1 = (a0 - pi1) / xi

pi[1] = pi1
u[1] = u1
y[1] = -u1
pi_w[1] = pi1 - a0

# Remaining transition: t >= 2
for t in range(2, T + 1):
    pi[t] = (r ** (t - 1)) * pi1
    u[t] = ((lam - r) / xi) * (r ** (t - 2)) * pi1
    y[t] = -u[t]
    pi_w[t] = pi[t]

# Figure style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

green = "darkgreen"

fig, axs = plt.subplots(2, 2, figsize=(9, 6))

series = [
    (y, r"Output $y_t$", (-0.65, 1.10)),
    (u, r"Unemployment $u_t$", (-0.08, 0.55)),
    (pi, r"Inflation $\pi_t$", (-0.08, 0.55)),
    (pi_w, r"Wage inflation $\pi_t^W$", (-0.65, 1.10)),
]

for ax, (data, title, ylim) in zip(axs.flatten(), series):
    ax.plot(time, data, color=green, linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("percent")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(*ylim)
    ax.set_xticks(time)
    ax.grid(False)

axs[1, 0].set_xlabel("Time")
axs[1, 1].set_xlabel("Time")

plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.90,
    bottom=0.12,
    wspace=0.35,
    hspace=0.55
)

# Export figure
plt.savefig("figures/irf_question4_prof_style.pdf", bbox_inches="tight")
plt.savefig("figures/irf_question4_prof_style.png", dpi=300, bbox_inches="tight")

plt.show()

# Report numerical values
print("Question 4 IRF values")
print("time:", time)
print("r   =", round(r, 4))
print("pi1 =", round(pi1, 4))
print("u1  =", round(u1, 4))
print("y   =", np.round(y, 4))
print("u   =", np.round(u, 4))
print("pi  =", np.round(pi, 4))
print("piW =", np.round(pi_w, 4))


# =============================================================================
# QUESTION 5 — ONE-TIME POSITIVE PRODUCTIVITY SHOCK: OPTIMAL POLICY
# =============================================================================
# The Taylor-rule path from Question 4 is compared with the optimal allocation
# for kappa = 0.5 and with the full-employment benchmark for kappa = 0.

# Question-specific calibration
a0 = 1.0          # positive productivity shock
phi_pi = 1.5      # Taylor-rule coefficient on inflation
phi_u = 0.5       # Taylor-rule coefficient on unemployment
xi = 1.0          # slope of wage Phillips curve
lam = 0.5         # wage indexation
beta = 0.95       # discount factor for optimal policy

T = 20            # horizon used for solving optimal policy
T_plot = 9        # plot periods t=0,1,...,8

# Ensure the figure output directory exists.
os.makedirs("figures", exist_ok=True)

# Labor-market mappings
def compute_u_from_pi(pi_path, a_path, xi, lam):
    """
    Given inflation pi_t and productivity a_t, compute unemployment implied by
    the downward nominal wage rigidity.

    u_t = max{0, [lambda*pi_{t-1} - pi_t - (a_t-a_{t-1})]/xi}
    """
    n = len(pi_path)
    u = np.zeros(n)

    pi_lag = 0.0
    a_lag = 0.0

    for t in range(n):
        wage_inflation = pi_path[t] + a_path[t] - a_lag
        lower_bound = lam * pi_lag

        u[t] = max(0.0, (lower_bound - wage_inflation) / xi)

        pi_lag = pi_path[t]
        a_lag = a_path[t]

    return u


def compute_piw(pi_path, a_path):
    """
    Compute wage inflation:
        pi_t^W = pi_t + a_t - a_{t-1}
    """
    n = len(pi_path)
    pi_w = np.zeros(n)

    a_lag = 0.0

    for t in range(n):
        pi_w[t] = pi_path[t] + a_path[t] - a_lag
        a_lag = a_path[t]

    return pi_w


# Optimal policy for kappa > 0
def solve_optimal_policy(kappa):
    """
    Solve the optimal policy problem over pi_0,...,pi_T.
    Unemployment is implied by the downward wage constraint.
    """
    a_path = np.zeros(T + 1)
    a_path[0] = a0

    def objective(pi_path):
        u_path = compute_u_from_pi(pi_path, a_path, xi, lam)

        welfare = 0.0
        for t in range(T + 1):
            welfare += (beta ** t) * (
                a_path[t]
                - u_path[t]
                - np.exp(-u_path[t])
                - 0.5 * kappa * pi_path[t] ** 2
            )

        return -welfare

    x0 = np.zeros(T + 1)
    bounds = [(-2.0 * a0, 2.0 * a0) for _ in range(T + 1)]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-12}
    )

    if not result.success:
        print("Warning: optimizer did not fully converge:", result.message)

    pi = result.x
    u = compute_u_from_pi(pi, a_path, xi, lam)
    y = a_path - u
    pi_w = compute_piw(pi, a_path)

    return y, u, pi, pi_w


# Taylor-rule benchmark from Question 4
def solve_taylor_rule_q4():
    """
    Taylor-rule transition from Question 4.

    Period 0 is imposed:
        y_0 = a_0, u_0 = 0, pi_0 = 0, pi_0^W = a_0.

    From period 1 onward, the central bank follows the Taylor rule.
    """
    y = np.zeros(T + 1)
    u = np.zeros(T + 1)
    pi = np.zeros(T + 1)
    pi_w = np.zeros(T + 1)

    # Period 0 imposed allocation
    y[0] = a0
    u[0] = 0.0
    pi[0] = 0.0
    pi_w[0] = a0

    # Stable root r from Question 4
    B = 1 + phi_u + lam + xi * phi_pi
    disc = B**2 - 4 * (1 + xi) * (1 + phi_u) * lam
    r = (B - np.sqrt(disc)) / (2 * (1 + xi))

    # Period 1
    pi1 = ((1 + phi_u) * a0) / (
        1 + phi_u + xi * phi_pi + lam - (1 + xi) * r
    )
    u1 = (a0 - pi1) / xi

    pi[1] = pi1
    u[1] = u1
    y[1] = -u1
    pi_w[1] = pi1 - a0

    # Periods t >= 2
    for t in range(2, T + 1):
        pi[t] = (r ** (t - 1)) * pi1
        u[t] = ((lam - r) / xi) * (r ** (t - 2)) * pi1
        y[t] = -u[t]
        pi_w[t] = pi[t]

    return y, u, pi, pi_w


# Full-employment benchmark for kappa = 0
def solve_optimal_kappa_zero():
    """
    When kappa=0, inflation has no welfare cost.
    The optimal allocation keeps unemployment at zero.
    We plot one full-employment implementation satisfying the wage constraint
    with equality.
    """
    a_path = np.zeros(T + 1)
    a_path[0] = a0

    pi = np.zeros(T + 1)
    u = np.zeros(T + 1)
    y = np.zeros(T + 1)

    pi_lag = 0.0
    a_lag = 0.0

    for t in range(T + 1):
        # Full-employment implementation:
        # pi_t + a_t - a_{t-1} = lambda*pi_{t-1}
        pi[t] = lam * pi_lag - (a_path[t] - a_lag)

        u[t] = 0.0
        y[t] = a_path[t]

        pi_lag = pi[t]
        a_lag = a_path[t]

    pi_w = compute_piw(pi, a_path)

    return y, u, pi, pi_w


# Compute policy paths
y_tr, u_tr, pi_tr, piw_tr = solve_taylor_rule_q4()
y_opt05, u_opt05, pi_opt05, piw_opt05 = solve_optimal_policy(kappa=0.5)
y_opt0, u_opt0, pi_opt0, piw_opt0 = solve_optimal_kappa_zero()

# Restrict the displayed horizon to t = 0,...,8.
time = np.arange(T_plot)

paths = {
    "Taylor rule": {
        "y": y_tr[:T_plot],
        "u": u_tr[:T_plot],
        "pi": pi_tr[:T_plot],
        "piw": piw_tr[:T_plot],
    },
    r"Optimal policy, $\kappa=0.5$": {
        "y": y_opt05[:T_plot],
        "u": u_opt05[:T_plot],
        "pi": pi_opt05[:T_plot],
        "piw": piw_opt05[:T_plot],
    },
    r"Optimal policy, $\kappa=0$": {
        "y": y_opt0[:T_plot],
        "u": u_opt0[:T_plot],
        "pi": pi_opt0[:T_plot],
        "piw": piw_opt0[:T_plot],
    },
}

# Figure style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

colors = {
    "Taylor rule": "darkgreen",
    r"Optimal policy, $\kappa=0.5$": "firebrick",
    r"Optimal policy, $\kappa=0$": "blue",
}

fig, axs = plt.subplots(2, 2, figsize=(9, 6))

variables = [
    ("y", r"Output $y_t$"),
    ("u", r"Unemployment $u_t$"),
    ("pi", r"Inflation $\pi_t$"),
    ("piw", r"Wage inflation $\pi_t^W$"),
]

for ax, (var, title) in zip(axs.flatten(), variables):
    for label, data in paths.items():
        ax.plot(
            time,
            data[var],
            color=colors[label],
            linewidth=2.0,
            label=label
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("percent")
    ax.set_xlim(time[0], time[-1])
    ax.set_xticks(time)
    ax.grid(False)

axs[1, 0].set_xlabel("Time")
axs[1, 1].set_xlabel("Time")

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.90,
    bottom=0.18,
    wspace=0.35,
    hspace=0.55
)

# Export figure
plt.savefig("figures/irf_question5_prof_style.pdf", bbox_inches="tight")
plt.savefig("figures/irf_question5_prof_style.png", dpi=300, bbox_inches="tight")

plt.show()

# Report numerical values
print("Question 5 IRF values")
print("time:", time)

for label, data in paths.items():
    print("\n", label)
    print("y   =", np.round(data["y"], 4))
    print("u   =", np.round(data["u"], 4))
    print("pi  =", np.round(data["pi"], 4))
    print("piW =", np.round(data["piw"], 4))
# =============================================================================
# QUESTION 6 — STOCHASTIC MODEL AND NUMERICAL POLICY FUNCTIONS
# =============================================================================
# State: s_t = (a_t, a_{t-1}, pi_{t-1})
# Control: pi_t; unemployment is implied by the DNWR constraint.
# The Bellman equation is solved by value function iteration.


# Figure style

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

green = "#006400"
burgundy = "#8B0000"
blue = "#003f8c"


# Numerical calibration

beta_q6 = 0.95

# Productivity process
# a_t = rho_a * a_{t-1} + epsilon_t
# rho_a controls persistence; sigma controls shock volatility.
rho_a_q6 = 0.80
sigma_q6 = 0.30

# Labor-market and policy parameters
# We set these explicitly for the full stochastic problem.
xi_q6 = 1.0
lam_q6 = 0.85

# Baseline numerical case
kappa_q6 = 0.5


# Value-function-iteration solver

def solve_q6_policy_functions(
    beta=0.95,
    rho_a=0.80,
    sigma=0.10,
    xi=1.0,
    lam=0.5,
    kappa=10.0,
    n_a=17,
    n_pi=17,
    n_pi_choices=501,
    n_quadrature=5,
    a_grid_width=3.0,
    pi_grid_min=-0.40,
    pi_grid_max=0.40,
    tolerance=1e-5,
    max_iterations=500,
    print_every=10,
):
    """
    Solves the policy functions for Question 6.

    State variables:
        s_t = (a_t, a_{t-1}, pi_{t-1})

    Control:
        pi_t

    Unemployment is recovered from the wage-rigidity complementarity condition:
        u_t = max{0, (lambda*pi_{t-1} - pi_t - a_t + a_{t-1}) / xi}

    The Bellman equation is solved by value function iteration.
    """

    if not (0 < beta < 1):
        raise ValueError("beta must be between 0 and 1.")
    if not (0 <= rho_a < 1):
        raise ValueError("rho_a must be in [0, 1).")
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive.")
    if xi <= 0:
        raise ValueError("xi must be strictly positive.")
    if not (0 <= lam < 1):
        raise ValueError("lambda must be in [0, 1).")
    if kappa <= 0:
        raise ValueError("This numerical block assumes kappa > 0.")

    # State grids
    stationary_std_a = sigma / np.sqrt(1.0 - rho_a**2)

    a_min = -a_grid_width * stationary_std_a
    a_max = a_grid_width * stationary_std_a

    a_grid = np.linspace(a_min, a_max, n_a)
    a_lag_grid = a_grid.copy()

    pi_lag_grid = np.linspace(pi_grid_min, pi_grid_max, n_pi)
    pi_choices = np.linspace(pi_grid_min, pi_grid_max, n_pi_choices)

    # Gaussian quadrature for epsilon_{t+1}
    hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(n_quadrature)

    shock_nodes = np.sqrt(2.0) * sigma * hermite_nodes
    shock_weights = hermite_weights / np.sqrt(np.pi)

    # Flatten the state space for vectorized evaluation
    A_now, A_lag, PI_lag = np.meshgrid(
        a_grid,
        a_lag_grid,
        pi_lag_grid,
        indexing="ij"
    )

    states = np.column_stack([
        A_now.ravel(),
        A_lag.ravel(),
        PI_lag.ravel()
    ])

    current_a = states[:, 0]
    lagged_a = states[:, 1]
    lagged_pi = states[:, 2]

    number_of_states = states.shape[0]
    number_of_choices = len(pi_choices)

    pi_choice_matrix = pi_choices[None, :]

    # Downward nominal wage-rigidity block
    unemployment_by_choice = np.maximum(
        0.0,
        (
            lam * lagged_pi[:, None]
            - pi_choice_matrix
            - current_a[:, None]
            + lagged_a[:, None]
        ) / xi
    )

    flow_payoff_by_choice = (
        current_a[:, None]
        - unemployment_by_choice
        - np.exp(-unemployment_by_choice)
        - 0.5 * kappa * pi_choice_matrix**2
    )

    # Value function iteration
    value_function = np.zeros((n_a, n_a, n_pi))
    best_choice_indices = np.zeros(number_of_states, dtype=int)

    for iteration in range(max_iterations):

        interpolator = RegularGridInterpolator(
            (a_grid, a_lag_grid, pi_lag_grid),
            value_function,
            bounds_error=False,
            fill_value=None
        )

        expected_continuation_value = np.zeros(
            (number_of_states, number_of_choices)
        )

        for shock, weight in zip(shock_nodes, shock_weights):

            next_a = rho_a * current_a + shock

            # Clipping avoids extrapolation outside the numerical state grid.
            next_a_clipped = np.clip(next_a, a_grid[0], a_grid[-1])
            current_a_clipped = np.clip(current_a, a_lag_grid[0], a_lag_grid[-1])

            interpolation_points = np.empty((number_of_states * number_of_choices, 3))

            interpolation_points[:, 0] = np.repeat(next_a_clipped, number_of_choices)
            interpolation_points[:, 1] = np.repeat(current_a_clipped, number_of_choices)
            interpolation_points[:, 2] = np.tile(pi_choices, number_of_states)

            continuation_values = interpolator(interpolation_points).reshape(
                number_of_states,
                number_of_choices
            )

            expected_continuation_value += weight * continuation_values

        objective_by_choice = flow_payoff_by_choice + beta * expected_continuation_value

        best_choice_indices = np.argmax(objective_by_choice, axis=1)
        new_value_flat = objective_by_choice[
            np.arange(number_of_states),
            best_choice_indices
        ]

        max_difference = np.max(
            np.abs(new_value_flat - value_function.ravel())
        )

        value_function = new_value_flat.reshape(n_a, n_a, n_pi)

        if iteration % print_every == 0:
            print(f"Iteration {iteration:4d}: max difference = {max_difference:.6e}")

        if max_difference < tolerance:
            print(f"Converged after {iteration} iterations. Max difference = {max_difference:.6e}")
            break

    # Recover policy functions
    policy_pi_flat = pi_choices[best_choice_indices]
    policy_u_flat = unemployment_by_choice[
        np.arange(number_of_states),
        best_choice_indices
    ]

    policy_pi = policy_pi_flat.reshape(n_a, n_a, n_pi)
    policy_u = policy_u_flat.reshape(n_a, n_a, n_pi)

    # Numerical validation
    wage_constraint = (
        policy_pi
        + A_now
        - A_lag
        - lam * PI_lag
        + xi * policy_u
    )

    complementarity = policy_u * wage_constraint

    lower_boundary_share = np.mean(policy_pi_flat == pi_choices[0])
    upper_boundary_share = np.mean(policy_pi_flat == pi_choices[-1])

    print("\nNumerical checks")
    print("Minimum wage-constraint residual:", np.min(wage_constraint))
    print("Maximum complementarity residual:", np.max(np.abs(complementarity)))
    print("Share at lower pi-choice boundary:", round(lower_boundary_share, 4))
    print("Share at upper pi-choice boundary:", round(upper_boundary_share, 4))

    if lower_boundary_share > 0.01 or upper_boundary_share > 0.01:
        print("\nWarning: the policy function often hits the pi-choice boundary.")
        print("Consider widening pi_grid_min / pi_grid_max or increasing kappa.")

    return {
        "value_function": value_function,
        "policy_pi": policy_pi,
        "policy_u": policy_u,
        "a_grid": a_grid,
        "a_lag_grid": a_lag_grid,
        "pi_lag_grid": pi_lag_grid,
        "pi_choices": pi_choices,
        "parameters": {
            "beta": beta,
            "rho_a": rho_a,
            "sigma": sigma,
            "xi": xi,
            "lam": lam,
            "kappa": kappa,
        },
        "last_difference": max_difference,
        "last_iteration": iteration,
        "boundary_share": {
            "lower": lower_boundary_share,
            "upper": upper_boundary_share,
        },
    }


# Solve the dynamic program

q6_solution = solve_q6_policy_functions(
    beta=beta_q6,
    rho_a=rho_a_q6,
    sigma=sigma_q6,
    xi=xi_q6,
    lam=lam_q6,
    kappa=kappa_q6,
    n_a=21,
    n_pi=21,
    n_pi_choices=501,
    n_quadrature=5,
    a_grid_width=4.0,
    pi_grid_min=-0.80,
    pi_grid_max=0.80,
    tolerance=1e-6,
    max_iterations=700,
    print_every=10,
)


# Plot policy-function slices

def nearest_index(grid, value):
    return int(np.argmin(np.abs(grid - value)))


def plot_q6_inflation_policy_slices(q6_results):
    """
    Plots the policy-function slices requested in Question 6.

    Panel 1:
        pi(a_t) for different values of a_{t-1}, holding pi_{t-1}=0.

    Panel 2:
        pi(a_t) for different values of pi_{t-1}, holding a_{t-1}=0.
    """

    a_grid = q6_results["a_grid"]
    a_lag_grid = q6_results["a_lag_grid"]
    pi_lag_grid = q6_results["pi_lag_grid"]
    policy_pi = q6_results["policy_pi"]
    parameters = q6_results["parameters"]

    zero_a_lag_index = nearest_index(a_lag_grid, 0.0)
    zero_pi_lag_index = nearest_index(pi_lag_grid, 0.0)

    stationary_std_a = parameters["sigma"] / np.sqrt(1.0 - parameters["rho_a"]**2)

    # Slices over lagged productivity
    a_lag_values = [-stationary_std_a, 0.0, stationary_std_a]
    a_lag_indices = [nearest_index(a_lag_grid, value) for value in a_lag_values]

    # Slices over lagged inflation
    pi_lag_values = [-0.10, 0.0, 0.10]
    pi_lag_indices = [nearest_index(pi_lag_grid, value) for value in pi_lag_values]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))

    # Panel 1: varying lagged productivity
    for index, color, linestyle in zip(
        a_lag_indices,
        [green, burgundy, blue],
        ["-", "--", "-."]
    ):
        value = a_lag_grid[index]

        axs[0].plot(
            a_grid,
            policy_pi[:, index, zero_pi_lag_index],
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            label=rf"$a_{{t-1}}={value:.2f}$"
        )

    axs[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
    axs[0].set_title(r"Varying $a_{t-1}$, holding $\pi_{t-1}=0$", pad=10)
    axs[0].set_xlabel(r"Current productivity $a_t$")
    axs[0].set_ylabel(r"Optimal inflation $\pi(a_t,a_{t-1},\pi_{t-1})$")

    # Panel 2: varying lagged inflation
    for index, color, linestyle in zip(
        pi_lag_indices,
        [green, burgundy, blue],
        ["-", "--", "-."]
    ):
        value = pi_lag_grid[index]

        axs[1].plot(
            a_grid,
            policy_pi[:, zero_a_lag_index, index],
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            label=rf"$\pi_{{t-1}}={value:.2f}$"
        )

    axs[1].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
    axs[1].set_title(r"Varying $\pi_{t-1}$, holding $a_{t-1}=0$", pad=10)
    axs[1].set_xlabel(r"Current productivity $a_t$")
    axs[1].set_ylabel(r"Optimal inflation $\pi(a_t,a_{t-1},\pi_{t-1})$")

    for ax in axs:
        ax.grid(False)
        ax.tick_params(direction="in")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    axs[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False
    )

    axs[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.86,
        bottom=0.27,
        wspace=0.32
    )

    plt.savefig("policy_question6_pi_slices.pdf", bbox_inches="tight")
    plt.savefig("policy_question6_pi_slices.png", dpi=300, bbox_inches="tight")

    plt.show()


plot_q6_inflation_policy_slices(q6_solution)


# Diagnostics

print("\nQuestion 6 parameters")
for key, value in q6_solution["parameters"].items():
    print(f"{key}: {value}")

print("\nPolicy function ranges")
print("pi min:", np.min(q6_solution["policy_pi"]))
print("pi max:", np.max(q6_solution["policy_pi"]))
print("u min:", np.min(q6_solution["policy_u"]))
print("u max:", np.max(q6_solution["policy_u"]))


# =============================================================================
# QUESTION 7 — PRUDENTIAL POLICY IN BOOM STATES
# =============================================================================
# No additional numerical routine is required here. The discussion in Question 7
# is based on the policy-function slices generated in Question 6.

# =============================================================================
# QUESTION 8 — PRUDENTIAL VS. NO-PRUDENTIAL MONETARY POLICY
# =============================================================================
# Unconditional simulated moments


def solve_myopic_no_prudential_policy(
    a_now,
    a_lag,
    pi_lag,
    xi,
    lam,
    kappa,
    tolerance=1e-12,
    max_iterations=200
):
    """
    Static/myopic no-prudential policy.

    The no-prudential central bank optimally trades off current inflation
    and current unemployment, but it does not internalize the effect of
    current inflation on tomorrow's wage-rigidity constraint.

    State variables:
        a_now  = a_t
        a_lag  = a_{t-1}
        pi_lag = pi_{t-1}

    Define:
        b_t = lambda*pi_{t-1} - a_t + a_{t-1}

    If b_t <= 0, full employment is feasible with zero inflation.
    If b_t > 0, the static optimum solves:
        1 - exp(-u_t) = kappa*xi*(b_t - xi*u_t),
    with:
        pi_t = b_t - xi*u_t.
    """

    b = lam * pi_lag - a_now + a_lag

    # Full employment is feasible with zero inflation.
    if b <= 0.0:
        return 0.0, 0.0, b

    # If inflation is costless, full employment is optimal.
    # In the baseline numerical exercise kappa > 0, but this keeps the
    # function well-defined.
    if kappa == 0.0:
        return b, 0.0, b

    # Binding case: pi = b - xi*u with u in [0, b/xi].
    lower = 0.0
    upper = b / xi

    def foc(u):
        return 1.0 - np.exp(-u) - kappa * xi * (b - xi * u)

    f_lower = foc(lower)
    f_upper = foc(upper)

    if f_lower > 0.0 or f_upper < 0.0:
        raise RuntimeError("Invalid bisection bracket in the no-prudential policy.")

    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = foc(midpoint)

        if abs(f_midpoint) < tolerance:
            u_star = midpoint
            break

        if f_midpoint > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    else:
        u_star = 0.5 * (lower + upper)

    pi_star = max(0.0, b - xi * u_star)

    return pi_star, u_star, b


def simulate_question8_unconditional_moments(
    q6_results,
    simulation_T=100_000,
    burn_in=5_000,
    seed=12345,
    binding_tolerance=1e-8
):
    """
    Simulates the same productivity path under two policy regimes:

        OP: optimal prudential policy from Question 6.
        NP: no-prudential myopic policy.

    The reported unconditional moments are long-run sample moments after
    discarding the initial burn-in.
    """

    parameters = q6_results["parameters"]

    beta = parameters["beta"]
    rho_a = parameters["rho_a"]
    sigma = parameters["sigma"]
    xi = parameters["xi"]
    lam = parameters["lam"]
    kappa = parameters["kappa"]

    a_grid = q6_results["a_grid"]
    a_lag_grid = q6_results["a_lag_grid"]
    pi_lag_grid = q6_results["pi_lag_grid"]
    policy_pi = q6_results["policy_pi"]

    # Interpolate the optimal prudential inflation policy.
    op_pi_interpolator = RegularGridInterpolator(
        (a_grid, a_lag_grid, pi_lag_grid),
        policy_pi,
        bounds_error=False,
        fill_value=None
    )

    total_T = simulation_T + burn_in
    rng = np.random.default_rng(seed)

    # Simulated productivity path
    # a_path[t] is a_{t-1}; a_path[t+1] is a_t in the loop below.
    # Starting from the stationary distribution reduces the importance of burn-in.
    stationary_std_a = sigma / np.sqrt(1.0 - rho_a**2)

    a_path = np.empty(total_T + 1)
    a_path[0] = rng.normal(0.0, stationary_std_a)

    shocks = rng.normal(0.0, sigma, size=total_T)

    for t in range(total_T):
        a_path[t + 1] = rho_a * a_path[t] + shocks[t]

    # Allocate simulation arrays
    variables = [
        "a",
        "pi",
        "u",
        "y",
        "pi_w",
        "output_gap",
        "welfare",
        "wage_pressure"
    ]

    op = {name: np.empty(total_T) for name in variables}
    npol = {name: np.empty(total_T) for name in variables}

    pi_lag_op = 0.0
    pi_lag_np = 0.0

    clipping_count_op = 0

    # Simulate both policy regimes on the same shocks
    for t in range(total_T):

        a_lag = a_path[t]
        a_now = a_path[t + 1]

        # OP: optimal prudential policy
        raw_state = np.array([a_now, a_lag, pi_lag_op])

        clipped_state = np.array([
            np.clip(a_now, a_grid[0], a_grid[-1]),
            np.clip(a_lag, a_lag_grid[0], a_lag_grid[-1]),
            np.clip(pi_lag_op, pi_lag_grid[0], pi_lag_grid[-1])
        ])

        if np.any(np.abs(raw_state - clipped_state) > 1e-14):
            clipping_count_op += 1

        pi_op_t = float(op_pi_interpolator(clipped_state.reshape(1, -1))[0])

        u_op_t = max(
            0.0,
            (lam * pi_lag_op - pi_op_t - a_now + a_lag) / xi
        )

        y_op_t = a_now - u_op_t
        pi_w_op_t = pi_op_t + a_now - a_lag
        output_gap_op_t = y_op_t - a_now
        welfare_op_t = (
            a_now
            - u_op_t
            - np.exp(-u_op_t)
            - 0.5 * kappa * pi_op_t**2
        )
        b_op_t = lam * pi_lag_op - a_now + a_lag

        op["a"][t] = a_now
        op["pi"][t] = pi_op_t
        op["u"][t] = u_op_t
        op["y"][t] = y_op_t
        op["pi_w"][t] = pi_w_op_t
        op["output_gap"][t] = output_gap_op_t
        op["welfare"][t] = welfare_op_t
        op["wage_pressure"][t] = b_op_t

        pi_lag_op = pi_op_t

        # NP: no-prudential myopic policy
        pi_np_t, u_np_t, b_np_t = solve_myopic_no_prudential_policy(
            a_now=a_now,
            a_lag=a_lag,
            pi_lag=pi_lag_np,
            xi=xi,
            lam=lam,
            kappa=kappa
        )

        y_np_t = a_now - u_np_t
        pi_w_np_t = pi_np_t + a_now - a_lag
        output_gap_np_t = y_np_t - a_now
        welfare_np_t = (
            a_now
            - u_np_t
            - np.exp(-u_np_t)
            - 0.5 * kappa * pi_np_t**2
        )

        npol["a"][t] = a_now
        npol["pi"][t] = pi_np_t
        npol["u"][t] = u_np_t
        npol["y"][t] = y_np_t
        npol["pi_w"][t] = pi_w_np_t
        npol["output_gap"][t] = output_gap_np_t
        npol["welfare"][t] = welfare_np_t
        npol["wage_pressure"][t] = b_np_t

        pi_lag_np = pi_np_t

    # Drop burn-in observations
    for dictionary in [op, npol]:
        for key in dictionary:
            dictionary[key] = dictionary[key][burn_in:]

    # Compute unconditional moments
    def compute_moments(series):

        a = series["a"]
        u = series["u"]
        pi = series["pi"]
        output_gap = series["output_gap"]
        welfare = series["welfare"]

        binding = u > binding_tolerance

        return {
            "mean_productivity": np.mean(a),
            "mean_unemployment": np.mean(u),
            "binding_frequency": np.mean(binding),
            "mean_unemployment_if_binding": (
                np.mean(u[binding]) if np.any(binding) else 0.0
            ),
            "p95_unemployment": np.percentile(u, 95),
            "sd_output_gap": np.std(output_gap, ddof=0),
            "sd_inflation": np.std(pi, ddof=0),
            "average_welfare_flow": np.mean(welfare),
        }

    op_moments = compute_moments(op)
    np_moments = compute_moments(npol)

    rows = [
        (r"Average productivity $E[a_t]$", "mean_productivity"),
        (r"Mean unemployment $E[u_t]$", "mean_unemployment"),
        (r"Frequency of binding DNWR $Pr(u_t>0)$", "binding_frequency"),
        (r"Mean unemployment if binding $E[u_t\mid u_t>0]$",
         "mean_unemployment_if_binding"),
        (r"95th percentile of unemployment", "p95_unemployment"),
        (r"Volatility of output gap $sd(y_t-a_t)$", "sd_output_gap"),
        (r"Volatility of inflation $sd(\pi_t)$", "sd_inflation"),
        (r"Average welfare flow $E[w_t]$", "average_welfare_flow"),
    ]

    table = pd.DataFrame([
        {
            "Moment": label,
            "Prudential policy": op_moments[key],
            "No-prudential policy": np_moments[key],
            "Difference": op_moments[key] - np_moments[key],
        }
        for label, key in rows
    ])

    formatted_table = table.copy()

    for column in ["Prudential policy", "No-prudential policy", "Difference"]:
        formatted_table[column] = formatted_table[column].map(lambda x: f"{x:.6f}")

    latex_table = formatted_table.to_latex(
        index=False,
        escape=False,
        column_format="lrrr",
        caption=(
            "Unconditional simulated moments under optimal prudential policy "
            "and no-prudential policy."
        ),
        label="tab:q8_unconditional_moments"
    )

    welfare_gain = (
        op_moments["average_welfare_flow"]
        - np_moments["average_welfare_flow"]
    )

    clipping_share_op = clipping_count_op / total_T

    print("\nQuestion 8.4: unconditional simulated moments")
    print(formatted_table.to_string(index=False))

    print("\nAverage welfare gain, OP - NP:",
          f"{welfare_gain:.8f}")

    print("Share of OP simulation states clipped to the interpolation grid:",
          f"{clipping_share_op:.4%}")

    if clipping_share_op > 0.01:
        print("\nWarning: more than 1% of OP simulation states are clipped.")
        print("Consider widening the state grids in Question 6 and re-solving the model.")

    # Export simulation outputs.
    table.to_csv("question8_unconditional_moments.csv", index=False)

    with open("table_question8_unconditional_moments.tex", "w") as file:
        file.write(latex_table)

    return {
        "op_series": op,
        "np_series": npol,
        "op_moments": op_moments,
        "np_moments": np_moments,
        "table": table,
        "formatted_table": formatted_table,
        "latex_table": latex_table,
        "welfare_gain": welfare_gain,
        "clipping_share_op": clipping_share_op,
        "parameters": parameters,
        "seed": seed,
        "simulation_T": simulation_T,
        "burn_in": burn_in,
    }


# Run the long-run simulation

q8_4_results = simulate_question8_unconditional_moments(
    q6_results=q6_solution,
    simulation_T=100_000,
    burn_in=5_000,
    seed=12345,
    binding_tolerance=1e-8
)


# Recover the implementing nominal interest rate

def add_nominal_interest_rate_to_q8_results(
    q8_4_results,
    q6_results,
    n_quadrature=5
):
    """
    Recovers the implementing nominal interest rate for OP and NP
    using the linear Euler equation:

        -y_t = -rho + i_t - E_t(pi_{t+1} + y_{t+1})

    Therefore:

        i_t = rho - y_t + E_t(pi_{t+1} + y_{t+1})

    The function stores:
        i_nominal       = level of the nominal interest rate
        i_nominal_gap   = deviation from steady state, i_t - rho
    """

    parameters = q8_4_results["parameters"]

    beta = parameters["beta"]
    rho_a = parameters["rho_a"]
    sigma = parameters["sigma"]
    xi = parameters["xi"]
    lam = parameters["lam"]
    kappa = parameters["kappa"]

    # Steady-state nominal rate in the linearized model.
    # This is rho = -log(beta), not the productivity persistence rho_a.
    rho_steady_state = -np.log(beta)

    a_grid = q6_results["a_grid"]
    a_lag_grid = q6_results["a_lag_grid"]
    pi_lag_grid = q6_results["pi_lag_grid"]
    policy_pi = q6_results["policy_pi"]

    op_pi_interpolator = RegularGridInterpolator(
        (a_grid, a_lag_grid, pi_lag_grid),
        policy_pi,
        bounds_error=False,
        fill_value=None
    )

    # Gaussian quadrature for epsilon_{t+1}.
    hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(
        n_quadrature
    )
    shock_nodes = np.sqrt(2.0) * sigma * hermite_nodes
    shock_weights = hermite_weights / np.sqrt(np.pi)

    def expected_next_nominal_component(policy_name, a_current, pi_current):
        """
        Computes E_t(pi_{t+1} + y_{t+1}) under either OP or NP.
        """

        expected_component = 0.0

        for shock, weight in zip(shock_nodes, shock_weights):

            a_next = rho_a * a_current + shock

            if policy_name == "OP":

                next_state = np.array([
                    np.clip(a_next, a_grid[0], a_grid[-1]),
                    np.clip(a_current, a_lag_grid[0], a_lag_grid[-1]),
                    np.clip(pi_current, pi_lag_grid[0], pi_lag_grid[-1])
                ])

                pi_next = float(
                    op_pi_interpolator(next_state.reshape(1, -1))[0]
                )

                u_next = max(
                    0.0,
                    (lam * pi_current - pi_next - a_next + a_current) / xi
                )

            elif policy_name == "NP":

                pi_next, u_next, _ = solve_myopic_no_prudential_policy(
                    a_now=a_next,
                    a_lag=a_current,
                    pi_lag=pi_current,
                    xi=xi,
                    lam=lam,
                    kappa=kappa
                )

            else:
                raise ValueError("policy_name must be either 'OP' or 'NP'.")

            y_next = a_next - u_next

            expected_component += weight * (pi_next + y_next)

        return expected_component

    for policy_name, series in [
        ("OP", q8_4_results["op_series"]),
        ("NP", q8_4_results["np_series"])
    ]:

        T = len(series["a"])

        nominal_rate = np.empty(T)
        expected_nominal_component = np.empty(T)

        for t in range(T):

            a_current = series["a"][t]
            y_current = series["y"][t]
            pi_current = series["pi"][t]

            expected_component = expected_next_nominal_component(
                policy_name=policy_name,
                a_current=a_current,
                pi_current=pi_current
            )

            i_t = rho_steady_state - y_current + expected_component

            nominal_rate[t] = i_t
            expected_nominal_component[t] = expected_component

        series["i_nominal"] = nominal_rate
        series["i_nominal_gap"] = nominal_rate - rho_steady_state
        series["expected_pi_plus_y_next"] = expected_nominal_component

    q8_4_results["rho_steady_state"] = rho_steady_state

    return q8_4_results


q8_4_results = add_nominal_interest_rate_to_q8_results(
    q8_4_results=q8_4_results,
    q6_results=q6_solution,
    n_quadrature=5
)


# Event study around binding DNWR episodes


def run_question8_event_study(
    q8_4_results,
    window_before=4,
    window_after=8,
    binding_tolerance=1e-8,
    minimum_spacing=13
):
    """
    Event study around dates in which the wage rigidity starts to bind
    under the no-prudential policy.

    Events are selected under NP, not OP, because the purpose is to study
    episodes that would arise absent prudential behavior.
    """

    op = q8_4_results["op_series"]
    npol = q8_4_results["np_series"]

    u_np = npol["u"]
    T = len(u_np)

    # Event definition
    # An event occurs when NP unemployment becomes strictly positive
    # after being zero in the previous period.
    binding_now = u_np > binding_tolerance
    binding_lag = np.concatenate(([False], u_np[:-1] > binding_tolerance))

    raw_events = np.where(binding_now & (~binding_lag))[0]

    # Keep only events with a complete event window.
    events = raw_events[
        (raw_events >= window_before)
        & (raw_events <= T - window_after - 1)
    ]

    # Enforce minimum spacing between event onsets.
    # Onset-based selection already avoids counting every period within
    # the same binding spell; spacing further limits overlapping windows.
    if minimum_spacing is not None:
        filtered_events = []
        last_event = -10**9

        for event in events:
            if event - last_event >= minimum_spacing:
                filtered_events.append(event)
                last_event = event

        events = np.array(filtered_events, dtype=int)

    if len(events) == 0:
        raise RuntimeError(
            "No event-study episodes found. Try lowering binding_tolerance "
            "or increasing the simulation length."
        )

    horizons = np.arange(-window_before, window_after + 1)

    # Event-time averaging helper
    def event_average(series):
        stacked = np.vstack([
            series[event + horizons]
            for event in events
        ])
        return stacked.mean(axis=0)

    # Build event-time averages
    event_study = pd.DataFrame({
        "horizon": horizons,

        "a": event_average(op["a"]),

        "pi_OP": event_average(op["pi"]),
        "pi_NP": event_average(npol["pi"]),

        "wage_pressure_OP": event_average(op["wage_pressure"]),
        "wage_pressure_NP": event_average(npol["wage_pressure"]),

        "u_OP": event_average(op["u"]),
        "u_NP": event_average(npol["u"]),

        "output_gap_OP": event_average(op["output_gap"]),
        "output_gap_NP": event_average(npol["output_gap"]),

        "welfare_OP": event_average(op["welfare"]),
        "welfare_NP": event_average(npol["welfare"]),
    })

    event_study["welfare_difference_OP_minus_NP"] = (
        event_study["welfare_OP"] - event_study["welfare_NP"]
    )

    # Report event-study diagnostics
    pre_event = event_study["horizon"].between(-window_before, -1)
    post_event = event_study["horizon"].between(1, window_after)

    print("\nQuestion 8.6: event study around binding DNWR episodes")
    print("Number of events:", len(events))
    print("Window:", f"[-{window_before}, +{window_after}]")

    print("\nAverage inflation before the event:")
    print("OP:", round(event_study.loc[pre_event, "pi_OP"].mean(), 6))
    print("NP:", round(event_study.loc[pre_event, "pi_NP"].mean(), 6))

    print("\nAverage wage pressure at the event date:")
    print(
        "OP:",
        round(event_study.loc[event_study["horizon"] == 0,
                              "wage_pressure_OP"].iloc[0], 6)
    )
    print(
        "NP:",
        round(event_study.loc[event_study["horizon"] == 0,
                              "wage_pressure_NP"].iloc[0], 6)
    )

    print("\nAverage unemployment after the event:")
    print("OP:", round(event_study.loc[post_event, "u_OP"].mean(), 6))
    print("NP:", round(event_study.loc[post_event, "u_NP"].mean(), 6))

    print("\nAverage welfare difference OP - NP over the event window:")
    print(round(event_study["welfare_difference_OP_minus_NP"].mean(), 8))

    # Figure
    color_op = globals().get("burgundy", "#8B0000")
    color_np = globals().get("green", "#006400")
    color_a = globals().get("blue", "#003f8c")

    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))

    # Panel 1: productivity
    ax = axs[0, 0]
    ax.plot(
        event_study["horizon"],
        event_study["a"],
        color=color_a,
        linewidth=2.5
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(r"Productivity $a_t$")
    ax.set_ylabel("linear deviation")

    # Panel 2: inflation
    ax = axs[0, 1]
    ax.plot(
        event_study["horizon"],
        event_study["pi_OP"],
        color=color_op,
        linewidth=2.5,
        label="Prudential policy"
    )
    ax.plot(
        event_study["horizon"],
        event_study["pi_NP"],
        color=color_np,
        linewidth=2.5,
        linestyle="--",
        label="No-prudential policy"
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(r"Inflation $\pi_t$")
    ax.set_ylabel("linear deviation")

    # Panel 3: inherited wage pressure
    ax = axs[1, 0]
    ax.plot(
        event_study["horizon"],
        event_study["wage_pressure_OP"],
        color=color_op,
        linewidth=2.5,
        label="Prudential policy"
    )
    ax.plot(
        event_study["horizon"],
        event_study["wage_pressure_NP"],
        color=color_np,
        linewidth=2.5,
        linestyle="--",
        label="No-prudential policy"
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(r"Inherited wage pressure $b_t$")
    ax.set_xlabel("Event time")
    ax.set_ylabel("linear deviation")

    # Panel 4: unemployment
    ax = axs[1, 1]
    ax.plot(
        event_study["horizon"],
        event_study["u_OP"],
        color=color_op,
        linewidth=2.5,
        label="Prudential policy"
    )
    ax.plot(
        event_study["horizon"],
        event_study["u_NP"],
        color=color_np,
        linewidth=2.5,
        linestyle="--",
        label="No-prudential policy"
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(r"Unemployment $u_t$")
    ax.set_xlabel("Event time")
    ax.set_ylabel("linear deviation")

    # Common panel formatting
    for ax in axs.flatten():
        ax.set_xlim(-window_before, window_after)
        ax.set_xticks(horizons)
        ax.grid(False)

    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.16,
        wspace=0.30,
        hspace=0.45
    )

    plt.savefig("event_study_question8.pdf", bbox_inches="tight")
    plt.savefig("event_study_question8.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Export event-study data.
    event_study.to_csv("question8_event_study.csv", index=False)

    return {
        "event_study": event_study,
        "events": events,
        "number_of_events": len(events),
        "window_before": window_before,
        "window_after": window_after,
    }


# Run the event study

q8_6_results = run_question8_event_study(
    q8_4_results=q8_4_results,
    window_before=4,
    window_after=8,
    binding_tolerance=1e-6,
    minimum_spacing=13
)


# Nominal-interest-rate event study

def plot_question8_nominal_rate_event_study(
    q8_4_results,
    q8_6_results,
    use_gap=True
):
    """
    Plots event-time averages of the nominal interest rate.

    If use_gap=True, the figure plots i_t - rho.
    If use_gap=False, the figure plots i_t in levels.
    """

    op = q8_4_results["op_series"]
    npol = q8_4_results["np_series"]

    events = q8_6_results["events"]
    window_before = q8_6_results["window_before"]
    window_after = q8_6_results["window_after"]

    horizons = np.arange(-window_before, window_after + 1)

    def event_average(series):
        stacked = np.vstack([
            series[event + horizons]
            for event in events
        ])
        return stacked.mean(axis=0)

    if use_gap:
        nominal_key = "i_nominal_gap"
        title = r"Nominal interest rate $i_t-\rho$"
        y_label = "deviation from steady state"
        file_suffix = "gap"
    else:
        nominal_key = "i_nominal"
        title = r"Nominal interest rate $i_t$"
        y_label = "linearized nominal rate"
        file_suffix = "level"

    nominal_rate_event_study = pd.DataFrame({
        "horizon": horizons,
        "nominal_rate_OP": event_average(op[nominal_key]),
        "nominal_rate_NP": event_average(npol[nominal_key]),
    })

    color_op = globals().get("burgundy", "#8B0000")
    color_np = globals().get("green", "#006400")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))

    ax.plot(
        nominal_rate_event_study["horizon"],
        nominal_rate_event_study["nominal_rate_OP"],
        color=color_op,
        linewidth=2.5,
        label="Prudential policy"
    )

    ax.plot(
        nominal_rate_event_study["horizon"],
        nominal_rate_event_study["nominal_rate_NP"],
        color=color_np,
        linewidth=2.5,
        linestyle="--",
        label="No-prudential policy"
    )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

    ax.set_title(title, pad=10)
    ax.set_xlabel("Event time")
    ax.set_ylabel(y_label)
    ax.set_xlim(-window_before, window_after)
    ax.set_xticks(horizons)
    ax.grid(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False
    )

    plt.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.88,
        bottom=0.26
    )

    plt.savefig(
        f"event_study_question8_nominal_interest_rate_{file_suffix}.pdf",
        bbox_inches="tight"
    )
    plt.savefig(
        f"event_study_question8_nominal_interest_rate_{file_suffix}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    nominal_rate_event_study.to_csv(
        f"question8_nominal_interest_rate_event_study_{file_suffix}.csv",
        index=False
    )

    print("\nQuestion 8.6: nominal interest-rate event study")
    print("Number of events:", len(events))
    print("Steady-state nominal rate rho:", round(q8_4_results["rho_steady_state"], 6))

    return nominal_rate_event_study


q8_nominal_rate_event_study = plot_question8_nominal_rate_event_study(
    q8_4_results=q8_4_results,
    q8_6_results=q8_6_results,
    use_gap=False
)
