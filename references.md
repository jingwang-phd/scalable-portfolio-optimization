# References (verified 2026-10-02)

## Benchmarks and estimation risk
- DeMiguel, Garlappi, Uppal (2009). *Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy?* Review of Financial Studies 22(5):1915–1953. — none of 14 optimizing rules consistently beats 1/N out of sample; the null hypothesis any construction method must beat.
- Kirby, Ostdiek (2012). *It's all in the timing: simple active portfolio strategies that outperform naive diversification.* JFQA 47(2):437–467. — the DGU result is driven by high-estimation-risk, high-turnover designs; low-turnover timing rules beat 1/N even with costs.
- Jagannathan, Ma (2003). *Risk reduction in large portfolios: Why imposing the wrong constraints helps.* Journal of Finance 58(4):1651–1683. — long-only / upper bounds act as shrinkage of the covariance matrix.
- Ledoit, Wolf (2004). *Honey, I shrunk the sample covariance matrix.* J. Portfolio Management 30(4):110–119. — constant-correlation linear shrinkage (the estimator used here).
- Ledoit, Wolf (2022). *The power of (non-)linear shrinking: a review and guide to covariance matrix estimation.* J. Financial Econometrics 20(1):187–218.
- Jorion (1986). *Bayes-Stein estimation for portfolio analysis.* JFQA 21(3):279–292. — shrinkage of means toward the GMV return.
- Bodnar, Parolya, Thorsén (2024). *Two is better than one: regularized shrinkage of large minimum variance portfolios.* JMLR 25(173).
- Petukhina, Klochkov, Härdle, Zhivotovskiy (2024). *Robustifying Markowitz.* J. Econometrics 239(2):105387.

## Risk measures (course material and its sources)
- Rockafellar, Uryasev (2000). *Optimization of conditional value-at-risk.* Journal of Risk 2(3):21–41. — the LP formulation used for min-/mean-CVaR.
- Rockafellar, Uryasev (2002). *Conditional value-at-risk for general loss distributions.* JBF 26(7):1443–1471.
- Ogryczak, Ruszczyński (1999). *From stochastic dominance to mean-risk models: semideviations as risk measures.* EJOR 116(1):33–50. — mean–lower-semideviation model, SSD-consistent for c ≤ 1.
- Ogryczak, Ruszczyński (2002). *Dual stochastic dominance and related mean-risk models.* SIAM J. Optim. 13(1):60–78. — CVaR/quantile measures and inverse SSD.
- Ruszczyński, Shapiro (2006). *Optimization of convex risk functions.* Math. OR 31(3):433–452. — dual representation of coherent risk measures (ambiguity-set interpretation of CVaR).
- Dentcheva, Ruszczyński (2006). *Portfolio optimization with stochastic dominance constraints.* JBF 30(2):433–451.

## Robust and distributionally robust portfolios
- Goldfarb, Iyengar (2003). *Robust portfolio selection problems.* Math. OR 28(1):1–38. — ellipsoidal uncertainty, SOCP reformulation.
- Delage, Ye (2010). *Distributionally robust optimization under moment uncertainty with application to data-driven problems.* Operations Research 58(3):595–612.
- Mohajerin Esfahani, Kuhn (2018). *Data-driven distributionally robust optimization using the Wasserstein metric.* Math. Programming 171:115–166. — OOS performance improves up to a critical radius, then converges to 1/N.
- Blanchet, Chen, Zhou (2022). *Distributionally robust mean-variance portfolio selection with Wasserstein distances.* Management Science 68(9):6382–6410. — Wasserstein DRO ≡ norm-regularized MV.
- Olivares-Nadal, DeMiguel (2018). *Technical note — A robust perspective on transaction costs in portfolio optimization.* Operations Research 66(3):733–739. — proportional cost term ≡ robust / regularized formulation.
- Ban, El Karoui, Lim (2018). *Machine learning and portfolio optimization.* Management Science 64(3):1136–1154. — performance-based regularization and cross-validation for MV and mean-CVaR.
- Georgantas, Doumpos, Zopounidis (2024). *Robust optimization approaches for portfolio selection: a comparative analysis.* Annals of OR 339(3):1205–1221.
- Fan, Ji, Lejeune (2024). *Distributionally robust portfolio optimization under marginal and copula ambiguity.* JOTA 203(3):2870–2907.
- Hautsch, Voigt (2019). *Large-scale portfolio allocation under transaction costs and model uncertainty.* J. Econometrics 212(1):221–240.

## Scalable CVaR optimization
- Künzi-Bay, Mayer (2006). *Computational aspects of minimizing conditional value-at-risk.* Computational Management Science 3(1):3–27. — two-stage recourse view, L-shaped (Benders) decomposition.
- Lim, Sherali, Uryasev (2010). *Portfolio optimization by minimizing CVaR via nondifferentiable optimization.* COAP 46(3):391–415; and Śliwiński (2011) COAP 50:591–595 (dual LP makes simplex competitive).
- Espinoza, Moreno (2014). *A primal-dual aggregation algorithm for minimizing CVaR in linear programs.* COAP 59(3):617–638. — exact scenario aggregation driven by duals.
- Ramírez-Pico, Ljubić, Moreno (2023). *Benders adaptive-cuts method for two-stage stochastic programs.* Transportation Science 57(5):1252–1275.
- Luxenberg, Pérez-Piñeiro, Diamond, Boyd (2025). *An operator splitting method for large-scale CVaR-constrained quadratic programs.* arXiv:2504.10814; Optimization and Engineering (2026). — O(m log m) projection onto the CVaR set inside ADMM; the closest template for the splitting method implemented here.
- Rockafellar, Wets (1991). *Scenarios and policy aggregation in optimization under uncertainty.* Math. OR 16(1):119–147. — progressive hedging.

## Inference
- Ledoit, Wolf (2008). *Robust performance hypothesis testing with the Sharpe ratio.* J. Empirical Finance 15(5):850–859.
- Politis, Romano (1994). *The stationary bootstrap.* JASA 89(428):1303–1313.
- Bailey, López de Prado (2014). *The deflated Sharpe ratio.* J. Portfolio Management 40(5):94–107. — selection bias when choosing among many configurations.

## Data
- Kenneth R. French Data Library — 49 industry portfolios (daily, value-weighted) and Fama-French factors / risk-free rate.
- Yahoo Finance via `yfinance` — adjusted closes for current S&P 500 constituents (Wikipedia list, GICS sectors).
