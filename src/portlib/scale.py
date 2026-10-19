"""Layer 3 -- solving the scenario-based mean-CVaR LP at scale.

Problem (equally likely scenarios r_1..r_S, loss l_s(x) = -r_s'x):
    min_x  -(1-c) mu'x + c * CVaR_alpha(l(x))        s.t. x in X = {1'x = 1, 0 <= x <= w_max, G x <= s_max}
    CVaR_alpha(l) = min_t t + (1/(alpha S)) sum_s (l_s - t)^+              (Rockafellar-Uryasev)
                  = max_{q in Q} q'l,   Q = {0 <= q <= 1/(alpha S), 1'q = 1}  (dual / ambiguity set)

Four solution methods:
    monolithic : full LP (N + S + 1 variables, S tail rows, dense S x N block) in Gurobi
    rowgen     : tail-scenario row generation. Only scenarios in a working set W carry a tail row;
                 after each solve, scenarios with l_s(x) > t are added. Exact: when no row is
                 violated the relaxed solution is optimal for the full LP. Exploits the fact that at
                 the optimum only ~alpha*S scenarios have u_s > 0 (sparse active set).
    admm       : operator splitting on y = -R x:  min f(x) + g(y),  f = -(1-c)mu'x + indicator(X),
                 g = c * CVaR_alpha(y).  x-update is an N-dimensional QP, y-update is the prox of the
                 support function of Q, computed exactly by a projection onto the capped simplex
                 (O(S log S)).  Memory is O(N S) for R but the LP is never formed.
    ph         : bundled progressive hedging (Rockafellar & Wets 1991): scenarios are split into B
                 bundles, the first-stage vector z = (x, t) is duplicated per bundle and driven to
                 consensus with augmented-Lagrangian multipliers; bundle subproblems solve in parallel
                 processes with persistent Gurobi models (warm starts).
"""
from __future__ import annotations
import time, os
import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ----------------------------------------------------------------------------- utilities
def block_bootstrap(R_hist: np.ndarray, S: int, block: int = 5, seed: int = 0) -> np.ndarray:
    """Circular moving-block bootstrap of the rows of R_hist (T x N) -> S x N scenarios."""
    rng = np.random.default_rng(seed)
    T, N = R_hist.shape
    n_blocks = int(np.ceil(S / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
    return R_hist[idx[:S]]


def cvar_exact(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Exact discrete CVaR (and the VaR t attaining the RU minimum) for equally likely losses."""
    S = len(losses)
    k = alpha * S
    srt = np.sort(losses)[::-1]           # descending
    m = int(np.floor(k))
    cvar = (srt[:m].sum() + (k - m) * (srt[m] if m < S else 0.0)) / k
    t = srt[m] if m < S else srt[-1]
    return float(cvar), float(t)


def full_objective(R, mu, x, alpha, c):
    cv, t = cvar_exact(-(R @ x), alpha)
    return -(1 - c) * float(mu @ x) + c * cv


def project_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {q : 0 <= q <= cap, sum q = 1} by bisection on the shift theta."""
    lo, hi = v.max() - 1.0 - cap, v.max()      # sum(clip(v-lo)) >= 1 >= sum(clip(v-hi))
    lo = min(lo, v.min() - cap)
    for _ in range(100):
        th = 0.5 * (lo + hi)
        s = np.clip(v - th, 0.0, cap).sum()
        if s > 1.0:
            lo = th
        else:
            hi = th
        if hi - lo < 1e-13:
            break
    return np.clip(v - 0.5 * (lo + hi), 0.0, cap)


def _env(threads):
    return gp.Env(params={"OutputFlag": 0, "Threads": threads})


def _add_X(m, N, w_max, G, s_max):
    x = m.addMVar(N, lb=0.0, ub=w_max, name="x")
    m.addConstr(x.sum() == 1.0)
    if G is not None:
        m.addConstr(G @ x <= s_max)
    return x


# ----------------------------------------------------------------------------- 1. monolithic
def solve_monolithic(R, mu, alpha, c, w_max, G, s_max, threads=8, method=-1, time_limit=1800):
    S, N = R.shape
    t0 = time.time()
    env = _env(threads)
    m = gp.Model(env=env)
    m.Params.Method = method
    m.Params.TimeLimit = time_limit
    x = _add_X(m, N, w_max, G, s_max)
    t = m.addVar(lb=-GRB.INFINITY)
    u = m.addMVar(S, lb=0.0)
    m.addConstr(u >= -(R @ x) - t)
    m.setObjective(-(1 - c) * (mu @ x) + c * (t + u.sum() / (alpha * S)), GRB.MINIMIZE)
    t_build = time.time() - t0
    m.optimize()
    status = m.Status
    xv = np.array(x.X) if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and m.SolCount else None
    out = dict(method="monolithic", status=int(status), time=time.time() - t0, build_time=t_build,
               obj=(full_objective(R, mu, xv, alpha, c) if xv is not None else np.nan), x=xv,
               rows=m.NumConstrs, nnz=m.NumNZs, iters=int(m.IterCount), threads=threads)
    m.dispose(); env.dispose()
    return out


# ----------------------------------------------------------------------------- 2. row generation
def solve_rowgen(R, mu, alpha, c, w_max, G, s_max, threads=1, init_mult=2.0, add_frac=1.0,
                 tol=1e-9, max_iter=50, time_limit=1800, x0=None, init="ew", n_sub=2000, seed=0):
    """init='ew': seed the working set with the worst scenarios of the 1/N portfolio;
    init='subsample': first solve the LP on a random subsample of n_sub scenarios (cheap) and seed
    the working set with the worst scenarios of that solution (closer to the true active set)."""
    S, N = R.shape
    t0 = time.time()
    env = _env(threads)
    if x0 is None and init == "subsample" and S > n_sub:
        rng = np.random.default_rng(seed)
        sub = rng.choice(S, size=n_sub, replace=False)
        r0 = solve_monolithic(R[sub], R[sub].mean(0), alpha, c, w_max, G, s_max, threads=threads)
        x0 = r0["x"]; init_mult = min(init_mult, 1.2)
    m = gp.Model(env=env)
    m.Params.Method = 1                     # dual simplex: rows are appended -> warm start
    x = _add_X(m, N, w_max, G, s_max)
    t = m.addVar(lb=-GRB.INFINITY, name="t")
    x0 = np.full(N, 1.0 / N) if x0 is None else x0
    loss0 = -(R @ x0)
    k0 = int(min(S, max(10, np.ceil(init_mult * alpha * S))))
    W = np.argsort(loss0)[::-1][:k0]
    inW = np.zeros(S, bool); inW[W] = True
    u_vars = {}
    m.setObjective(-(1 - c) * (mu @ x) + c * t, GRB.MINIMIZE)
    coef = c / (alpha * S)

    def add_rows(ids):
        if len(ids) == 0:
            return
        u = m.addMVar(len(ids), lb=0.0, obj=coef)
        m.addConstr(u >= -(R[ids] @ x) - t)
        for j, s in enumerate(ids):
            u_vars[s] = u[j]
    add_rows(W)
    hist = []
    for it in range(max_iter):
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"rowgen LP status {m.Status}")
        xv, tv = np.array(x.X), t.X
        lb = m.ObjVal                                   # relaxation -> lower bound
        ub = full_objective(R, mu, xv, alpha, c)        # x feasible -> upper bound
        loss = -(R @ xv)
        viol = np.where((loss > tv + tol) & (~inW))[0]
        hist.append(dict(it=it, rows=int(inW.sum()), lb=lb, ub=ub, n_viol=len(viol), time=time.time() - t0))
        if len(viol) == 0 or (ub - lb) <= 1e-10 * max(1.0, abs(ub)):
            break
        if add_frac < 1.0:
            n_add = max(1, int(np.ceil(add_frac * len(viol))))
            viol = viol[np.argsort(loss[viol] - tv)[::-1][:n_add]]
        inW[viol] = True
        add_rows(viol)
        if time.time() - t0 > time_limit:
            break
    out = dict(method="rowgen", status=2, time=time.time() - t0, obj=ub, x=xv, rows=int(inW.sum()),
               nnz=int(inW.sum()) * N, iters=len(hist), active_frac=float(inW.sum()) / S,
               n_pos_u=int((loss > tv + 1e-9).sum()), hist=hist, threads=threads)
    m.dispose(); env.dispose()
    return out


# ----------------------------------------------------------------------------- 3. ADMM / splitting
def solve_admm(R, mu, alpha, c, w_max, G, s_max, rho=None, max_iter=2000, eps_rel=1e-5,
               eps_abs=1e-7, threads=1, time_limit=1800, ref_obj=None, gap_tol=None, verbose=False):
    """ADMM on  min f(x) + g(y)  s.t.  y = -R x  (y = scenario losses)."""
    S, N = R.shape
    t0 = time.time()
    cap = 1.0 / (alpha * S)
    M = R.T @ R                                          # N x N Gram matrix (one pass over R)
    # scale rho so that rho * ||R||^2 is comparable to the objective curvature
    lam_max = float(np.linalg.eigvalsh(M)[-1])
    rho = rho if rho is not None else c / (lam_max * cap * np.sqrt(S))
    env = _env(threads)
    m = gp.Model(env=env)
    m.Params.Method = 2; m.Params.BarConvTol = 1e-9
    x = _add_X(m, N, w_max, G, s_max)
    # objective: -(1-c) mu'x + (rho/2) ||-Rx - y + u||^2  = const + (rho/2) x'Mx + rho (y-u)'R x  - (1-c) mu'x
    y = np.zeros(S); u = np.zeros(S)
    xv = np.full(N, 1.0 / N)
    hist = []
    for k in range(max_iter):
        lin = rho * (R.T @ (y - u)) - (1 - c) * mu
        m.setObjective(0.5 * rho * (x @ M @ x) + lin @ x, GRB.MINIMIZE)
        m.optimize()
        xv = np.array(x.X)
        v = -(R @ xv) + u
        # prox of (c/rho) * sigma_Q at v:  v - (c/rho) Proj_Q(rho v / c)
        q = project_capped_simplex(rho * v / c, cap)
        y_new = v - (c / rho) * q
        r_pri = -(R @ xv) - y_new
        r_dual = rho * (R.T @ (y_new - y))
        u = u + r_pri
        y = y_new
        obj = full_objective(R, mu, xv, alpha, c)
        pri, dua = np.linalg.norm(r_pri), np.linalg.norm(r_dual)
        eps_pri = eps_abs * np.sqrt(S) + eps_rel * max(np.linalg.norm(R @ xv), np.linalg.norm(y))
        eps_dua = eps_abs * np.sqrt(N) + eps_rel * np.linalg.norm(rho * (R.T @ u))
        hist.append(dict(it=k, obj=obj, r_pri=pri, r_dual=dua, rho=rho, time=time.time() - t0))
        if verbose and k % 20 == 0:
            print(k, obj, pri, dua, rho)
        if ref_obj is not None and gap_tol is not None and (obj - ref_obj) <= gap_tol * max(abs(ref_obj), 1e-12):
            break
        if pri <= eps_pri and dua <= eps_dua:
            break
        # residual balancing
        if pri > 10 * dua:
            rho *= 2.0; u /= 2.0
        elif dua > 10 * pri:
            rho /= 2.0; u *= 2.0
        if time.time() - t0 > time_limit:
            break
    out = dict(method="admm", status=2, time=time.time() - t0, obj=obj, x=xv, iters=len(hist),
               rows=0, nnz=R.size, hist=hist, threads=threads, rho_final=rho)
    m.dispose(); env.dispose()
    return out


# ----------------------------------------------------------------------------- 4. bundled PH
_PH = {}


def _ph_init(Rb, mu, alpha, c, S, w_max, G, s_max, rho):
    """Build one bundle subproblem (persistent across iterations) in a worker process."""
    env = _env(1)
    m = gp.Model(env=env)
    m.Params.Method = 2                      # barrier for QPs; warm starts are limited but the
    m.Params.BarConvTol = 1e-10              # model build cost is paid only once
    Sb, N = Rb.shape
    x = _add_X(m, N, w_max, G, s_max)
    t = m.addVar(lb=-GRB.INFINITY)
    u = m.addMVar(Sb, lb=0.0)
    m.addConstr(u >= -(Rb @ x) - t)
    _PH.update(dict(m=m, x=x, t=t, u=u, pi=Sb / S, mu=mu, c=c, coef=c / (alpha * S), rho=rho, N=N))


def _ph_step(args):
    xbar, tbar, wx, wt = args
    d = _PH
    m, x, t, u = d["m"], d["x"], d["t"], d["u"]
    rho = d["rho"]
    obj = (d["pi"] * (-(1 - d["c"]) * (d["mu"] @ x) + d["c"] * t) + d["coef"] * u.sum()
           + wx @ x + wt * t + 0.5 * rho * ((x - xbar) @ (x - xbar)) + 0.5 * rho * (t - tbar) * (t - tbar))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    return np.array(x.X), float(t.X)


def solve_ph(R, mu, alpha, c, w_max, G, s_max, n_bundles=8, n_jobs=8, rho=None, max_iter=300,
             tol=1e-6, time_limit=1800, ref_obj=None, gap_tol=None, verbose=False):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    S, N = R.shape
    t0 = time.time()
    parts = np.array_split(np.arange(S), n_bundles)
    rho = rho if rho is not None else 0.04 * c         # tuned on a 49-asset / 4000-scenario instance
    ctx = mp.get_context("fork")                       # workers inherit the data; Gurobi env is created post-fork
    # separate executors so that each worker holds exactly one bundle model
    exes = [ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_ph_init,
                                initargs=(R[p], mu, alpha, c, S, w_max, G, s_max, rho)) for p in parts]
    pis = np.array([len(p) / S for p in parts])
    xbar, tbar = np.full(N, 1.0 / N), float(np.quantile(-(R @ np.full(N, 1.0 / N)), 1 - alpha))
    wx = [np.zeros(N) for _ in parts]; wt = [0.0 for _ in parts]
    hist = []
    try:
        for k in range(max_iter):
            futs = [ex.submit(_ph_step, (xbar, tbar, wx[b], wt[b])) for b, ex in enumerate(exes)]
            sols = [f.result() for f in futs]
            xs = np.array([s[0] for s in sols]); ts = np.array([s[1] for s in sols])
            xbar = pis @ xs; tbar = float(pis @ ts)
            for b in range(len(parts)):
                wx[b] = wx[b] + rho * (xs[b] - xbar); wt[b] = wt[b] + rho * (ts[b] - tbar)
            dev = float(np.sqrt((pis * (((xs - xbar) ** 2).sum(1) + (ts - tbar) ** 2)).sum()))
            obj = full_objective(R, mu, xbar, alpha, c)
            hist.append(dict(it=k, obj=obj, dev=dev, time=time.time() - t0))
            if verbose and k % 10 == 0:
                print(k, obj, dev)
            if ref_obj is not None and gap_tol is not None and (obj - ref_obj) <= gap_tol * max(abs(ref_obj), 1e-12):
                break
            if dev < tol or time.time() - t0 > time_limit:
                break
    finally:
        for ex in exes:
            ex.shutdown(wait=False, cancel_futures=True)
    return dict(method="ph", status=2, time=time.time() - t0, obj=obj, x=xbar, iters=len(hist), rows=S,
                nnz=R.size, hist=hist, threads=n_jobs, n_bundles=n_bundles)
