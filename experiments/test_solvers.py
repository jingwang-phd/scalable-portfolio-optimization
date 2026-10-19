"""Agreement test: the four mean-CVaR solvers must reach the same optimum on a small instance."""
import numpy as np
from config import *  # noqa
from portlib import data, scale

if __name__ == "__main__":
    R_hist, sec = data.load_ff49("2023-09-01", "2025-08-31")
    X = R_hist.values
    G, _ = data.sector_matrix(sec, R_hist.columns)
    S = 4000
    R = scale.block_bootstrap(X, S, block=5, seed=1); mu = R.mean(0)
    alpha, c, w_max, s_max = 0.05, 0.25, 0.10, 0.30
    mono = scale.solve_monolithic(R, mu, alpha, c, w_max, G, s_max, threads=2)
    print(f"mono   obj={mono['obj']:.8f} time={mono['time']:.2f}s rows={mono['rows']} iters={mono['iters']}")
    rg = scale.solve_rowgen(R, mu, alpha, c, w_max, G, s_max, threads=2)
    print(f"rowgen obj={rg['obj']:.8f} time={rg['time']:.2f}s rows={rg['rows']} ({rg['active_frac']:.3f} of S) "
          f"iters={rg['iters']} n_pos_u={rg['n_pos_u']} alphaS={alpha*S:.0f}")
    ad = scale.solve_admm(R, mu, alpha, c, w_max, G, s_max, threads=2, ref_obj=mono['obj'], gap_tol=1e-4, max_iter=3000)
    print(f"admm   obj={ad['obj']:.8f} relgap={(ad['obj']-mono['obj'])/abs(mono['obj']):.2e} time={ad['time']:.2f}s iters={ad['iters']} rho={ad['rho_final']:.3g}")
    for h in ad['hist'][::max(1, len(ad['hist']) // 6)]:
        print("   ", {k: (round(v, 7) if isinstance(v, float) else v) for k, v in h.items()})
    ph = scale.solve_ph(R, mu, alpha, c, w_max, G, s_max, n_bundles=4, n_jobs=4, ref_obj=mono['obj'], gap_tol=1e-4, max_iter=400)
    print(f"ph     obj={ph['obj']:.8f} relgap={(ph['obj']-mono['obj'])/abs(mono['obj']):.2e} time={ph['time']:.2f}s iters={ph['iters']}")
    for h in ph['hist'][::max(1, len(ph['hist']) // 6)]:
        print("   ", {k: (round(v, 7) if isinstance(v, float) else v) for k, v in h.items()})
    print("||x_mono - x_rowgen||_1 =", np.abs(mono['x'] - rg['x']).sum(),
          " ||x_mono - x_admm||_1 =", np.abs(mono['x'] - ad['x']).sum(),
          " ||x_mono - x_ph||_1 =", np.abs(mono['x'] - ph['x']).sum())
