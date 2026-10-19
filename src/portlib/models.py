"""Portfolio construction models (all solved with Gurobi).

Common constraint set X (applied to every optimized model):
    sum(x) = 1, 0 <= x_i <= w_max                 (long-only, position limit)
    G x <= s_max                                  (sector exposure caps)
    ||x - x_prev||_1 <= tau                       (turnover limit, if x_prev given)
Transaction costs are charged in the objective as  tc_scale * c * ||x - x_prev||_1, where c is
the proportional one-way cost and tc_scale converts a one-off trading cost into the same units as
the return term (e.g. 12 when returns are annualized and we rebalance monthly).

Models
    ew            1/N
    mv            mean-variance: max mu'x - (gamma/2) x'Sigma x
    gmv           minimum variance (mu ignored)
    robust_mv     ellipsoidal mean-uncertainty (Ben-Tal & Nemirovski / Goldfarb & Iyengar flavour):
                  max mu'x - kappa * ||Sigma_mu^{1/2} x||_2 - (gamma/2) x'Sigma x, Sigma_mu = Sigma/T
    cvar          mean-CVaR on scenarios:  min -(1-c) E[r'x] + c * AVaR_alpha(r'x)  (c=1: min-CVaR)
                  AVaR_alpha(Z) = min_t { t + E[(-Z - t)^+]/alpha }   (Rockafellar-Uryasev LP)
    semidev       mean-lower-semideviation (Ogryczak & Ruszczynski):
                  min -E[Z] + c * E[(E[Z] - Z)^+],  c in (0, 1]
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import gurobipy as gp
from gurobipy import GRB

_ENV = None


def env():
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(params={"OutputFlag": 0, "Threads": 1})
    return _ENV


@dataclass
class Constraints:
    w_max: float = 1.0
    sector_G: np.ndarray | None = None      # (n_sectors x N)
    s_max: float | None = None              # cap per sector
    turnover_max: float | None = None       # cap on ||x - x_prev||_1
    cost: float = 0.0                       # proportional one-way cost (e.g. 0.001 = 10 bps)
    tc_scale: float = 1.0                   # unit conversion of the cost term in the objective


def _base_model(N: int, cons: Constraints, x_prev: np.ndarray | None):
    m = gp.Model(env=env())
    x = m.addMVar(N, lb=0.0, ub=cons.w_max, name="x")
    m.addConstr(x.sum() == 1.0, name="budget")
    if cons.sector_G is not None and cons.s_max is not None:
        m.addConstr(cons.sector_G @ x <= cons.s_max, name="sector")
    tc_expr = 0.0
    if x_prev is not None and (cons.turnover_max is not None or cons.cost > 0):
        d = m.addMVar(N, lb=0.0, name="absdev")           # d >= |x - x_prev|
        m.addConstr(d >= x - x_prev); m.addConstr(d >= x_prev - x)
        if cons.turnover_max is not None:
            m.addConstr(d.sum() <= cons.turnover_max, name="turnover")
        if cons.cost > 0:
            tc_expr = cons.tc_scale * cons.cost * d.sum()
    return m, x, tc_expr


def _solve(m, x, N):
    m.Params.NumericFocus = 1
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        # retry with a different algorithm before giving up
        m.Params.Method = 2 if m.Params.Method != 2 else 1
        m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {m.Status}")
    w = np.array(x.X).clip(0)
    return w / w.sum()


def solve_ew(N, cons: Constraints, x_prev=None, **kw):
    w = np.full(N, 1.0 / N)
    if cons.w_max < 1.0 / N:
        raise ValueError("w_max too small for 1/N")
    return w


def solve_mv(mu, Sigma, gamma, cons: Constraints, x_prev=None):
    N = len(mu)
    m, x, tc = _base_model(N, cons, x_prev)
    m.setObjective(-(mu @ x) + 0.5 * gamma * (x @ Sigma @ x) + tc, GRB.MINIMIZE)
    return _solve(m, x, N)


def solve_gmv(Sigma, gamma, cons: Constraints, x_prev=None):
    """Minimum variance = mean-variance with mu = 0. Keeping the gamma/2 scaling puts the variance
    term in certainty-equivalent return units so that the transaction-cost term is commensurable."""
    N = Sigma.shape[0]
    m, x, tc = _base_model(N, cons, x_prev)
    m.setObjective(0.5 * gamma * (x @ Sigma @ x) + tc, GRB.MINIMIZE)
    return _solve(m, x, N)


def solve_robust_mv(mu, Sigma, Sigma_mu, kappa, gamma, cons: Constraints, x_prev=None):
    """Worst-case expected return over the ellipsoid {m : (m-mu)' Sigma_mu^{-1} (m-mu) <= kappa^2}
    equals mu'x - kappa ||Sigma_mu^{1/2} x||.  Implemented as an SOCP."""
    N = len(mu)
    L = np.linalg.cholesky(Sigma_mu + 1e-12 * np.eye(N))
    m, x, tc = _base_model(N, cons, x_prev)
    z = m.addMVar(N, lb=-GRB.INFINITY, name="z")
    r = m.addVar(lb=0.0, name="r")
    m.addConstr(z == L.T @ x)
    m.addConstr(z @ z <= r * r)
    m.setObjective(-(mu @ x) + kappa * r + 0.5 * gamma * (x @ Sigma @ x) + tc, GRB.MINIMIZE)
    return _solve(m, x, N)


def solve_cvar(R, alpha, c, cons: Constraints, x_prev=None, mu=None):
    """Scenario matrix R (S x N, equally likely). Minimizes -(1-c) mu'x + c AVaR_alpha(R x)."""
    S, N = R.shape
    if mu is None:
        mu = R.mean(axis=0)
    m, x, tc = _base_model(N, cons, x_prev)
    t = m.addVar(lb=-GRB.INFINITY, name="t")
    u = m.addMVar(S, lb=0.0, name="u")
    m.addConstr(u >= -(R @ x) - t, name="tail")
    m.setObjective(-(1 - c) * (mu @ x) + c * (t + u.sum() / (alpha * S)) + tc, GRB.MINIMIZE)
    return _solve(m, x, N)


def solve_semidev(R, c, cons: Constraints, x_prev=None):
    """Mean-lower-semideviation model: min -E[Z] + c E[(E[Z]-Z)^+],  Z = R x, equally likely."""
    S, N = R.shape
    mu = R.mean(axis=0)
    m, x, tc = _base_model(N, cons, x_prev)
    d = m.addMVar(S, lb=0.0, name="d")
    m.addConstr(d >= (mu @ x) - (R @ x), name="shortfall")
    m.setObjective(-(mu @ x) + c * d.sum() / S + tc, GRB.MINIMIZE)
    return _solve(m, x, N)


# ---------------------------------------------------------------------------------------------
# Strategy objects used by the backtester: given a trailing return window, produce weights.
# ---------------------------------------------------------------------------------------------
@dataclass
class Strategy:
    name: str
    kind: str                                # ew | mv | gmv | robust_mv | cvar | semidev
    params: dict = field(default_factory=dict)
    estimator: str = "sample"               # sample | lw  (covariance); means: sample | bs
    mean_estimator: str = "sample"
    periods_per_year: int = 252

    def weights(self, R: np.ndarray, cons: Constraints, x_prev: np.ndarray | None):
        from .estimators import sample_moments, ledoit_wolf_cc, bayes_stein_mean, nearest_psd
        T, N = R.shape
        if self.kind == "ew":
            return solve_ew(N, cons, x_prev)
        mu, Sig = sample_moments(R)
        if self.estimator == "lw":
            Sig, _ = ledoit_wolf_cc(R)
        if self.mean_estimator == "bs":
            mu, _ = bayes_stein_mean(R, Sig)
        Sig = nearest_psd(Sig)
        A = self.periods_per_year
        if self.kind == "mv":
            return solve_mv(mu * A, Sig * A, self.params["gamma"], cons, x_prev)
        if self.kind == "gmv":
            return solve_gmv(Sig * A, self.params.get("gamma", 5.0), cons, x_prev)
        if self.kind == "robust_mv":
            Sig_mu = Sig * A / T                    # sampling covariance of the annualized mean
            return solve_robust_mv(mu * A, Sig * A, Sig_mu, self.params["kappa"], self.params["gamma"], cons, x_prev)
        # scenario-based models work in per-period (daily) units; the cost term is rescaled
        # so that its units match the daily return objective (tc_scale is set by the caller for
        # annual units, so divide by A here).
        cons_d = Constraints(**{**cons.__dict__, "tc_scale": cons.tc_scale / A})
        if self.kind == "cvar":
            return solve_cvar(R, self.params["alpha"], self.params["c"], cons_d, x_prev, mu=mu)
        if self.kind == "semidev":
            return solve_semidev(R, self.params["c"], cons_d, x_prev)
        raise ValueError(self.kind)
