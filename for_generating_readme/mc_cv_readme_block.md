## Monte Carlo synthetic CV robustness

To test whether the synthetic CV result above was just a lucky holdout, I repeated the same synthetic validation idea across `12` independently generated random holdout datasets. Each run used the same final submission code, but a different random seed for hiding observed IV cells.

```text
number of MC runs       : 12
hidden cells per run    : 2621 to 2621
mean MSE                : 0.0002965652
std MSE                 : 0.0001446755
mean RMSE               : 0.0168143457
std RMSE                : 0.0038860506
mean MAE                : 0.0046273639
mean p95 absolute error : 0.0199111351
best seed by RMSE       : 71  (0.0121440186)
worst seed by RMSE      : 11  (0.0258483124)
```

The first Monte Carlo picture shows the distribution of the main error statistics across random holdouts. The useful thing here is not a single score, but the width of each distribution: a narrow spread means the method is not depending heavily on one particular split.

![Monte Carlo metric distributions](for_generating_readme/mc_cv_metric_distributions.png)

The seed trajectory shows the same robustness in a different way. RMSE and MAE move from seed to seed, but they stay in the same band instead of exploding for a particular random holdout.

![Monte Carlo seed trajectory](for_generating_readme/mc_cv_seed_trajectory.png)

This regime plot separates the synthetic errors into pre-27 Jan and 27 Jan, and also splits CE from PE. It is the direct robustness check for the regime behavior found in the EDA: 27 Jan is harder, but the method remains controlled on both option sides.

![Monte Carlo regime robustness](for_generating_readme/mc_cv_regime_robustness.png)

The final Monte Carlo plot looks at the tail of the error distribution. The median error remains very small, while the higher quantiles show where the remaining risk lives: a small number of difficult near-expiry / wing-like points rather than a broad failure across the surface.

![Monte Carlo error quantiles](for_generating_readme/mc_cv_error_quantiles.png)
