# Final Submission: Implied Volatility Completion

This repository is centered around one file:

```bash
final_submission.py
```

That is the file I am submitting. It fills the missing implied-volatility values and writes the final submission file:

```text
submission_try_final_pchip_interior.csv
```

Run it from the repository root:

```bash
python final_submission.py --data everything_else/cv_validation_system/dataset.csv --skip-cv
```

It produces:

```text
filled_dataset_try_final_pchip_interior.csv
submission_try_final_pchip_interior.csv
diagnostics_try_final_pchip_interior.csv
cross_section_diagnostics_try_final_pchip_interior.csv
```

On the final full dataset run, the script filled `5,460` missing IV cells and left `0` missing values in the completed dataset.

## Table Of Contents

- [1. Dataset EDA](#1-dataset-eda)
- [2. Final Idea](#2-final-idea)
- [3. Final IV Surfaces](#3-final-iv-surfaces)
- [4. Filling Logic](#4-filling-logic)
- [5. Progressive Edge Filling](#5-progressive-edge-filling)
- [6. Diagnostics](#6-diagnostics)
- [7. Synthetic CV Validation](#7-synthetic-cv-validation)
- [8. What I Tried](#8-what-i-tried)
- [9. Function Map](#9-function-map)
- [10. Rebuilding Figures](#10-rebuilding-figures)

## 1. Dataset EDA

The original dataset is an option surface observed through time. It has `975` timestamp rows, `28` option contracts, and `5,460` missing IV cells, exactly `20%` of the option grid.

The first important observation is that this is not one smooth time regime. Most days from Jan 7 to Jan 23 have low, stable IV. Jan 27 is different: it is the expiry-day regime. The daily average observed IV jumps to about `0.753`, and the cross-strike dispersion jumps to about `0.320`.

![Dataset EDA regime formation](for_generating_readme/dataset_regime_eda.png)

The missingness is also structured. Each timestamp gives a partial CE smile and a partial PE smile across strikes. The model is not filling unrelated holes; it is reconstructing incomplete cross-sections.

![Given and missing values across cross-sections](for_generating_readme/missing_given_cross_section_eda.png)

Raw smiles make the regime shift obvious. Early smiles are low and calm; Jan 27 smiles become steep, wide, and asymmetric.

![Raw smile regime snapshots](for_generating_readme/raw_smile_regime_snapshots.png)

This EDA is the reason I trusted same-timestamp smile structure more than a global time-series smoother. The lag-1 EDA in `everything_else/eda/eda_lag1_calendar_gap/` showed high adjacent-time correlation, but Jan 27 has much larger IV changes. The surface is locally structured, but the time path is not uniformly gentle.

## 2. Final Idea

The model treats each timestamp as an implied-volatility smile. For a missing option cell, the question is:

> At this timestamp, what does the CE or PE smile look like across strikes, and where should this missing strike sit on that smile?

The model works in moneyness:

$$
x = \frac{K}{S}
$$

where \(K\) is strike and \(S\) is the underlying price.

Every missing cell is routed into one of two geometries:

```text
interior gap -> interpolate inside the observed smile
edge wing    -> extrapolate outward from one observed side
```

Interior gaps are safer because there are observed strikes on both sides. Edge wings are harder because the model has to extend the smile beyond the observed support.

## 3. Final IV Surfaces

These surfaces are generated from the final filled dataset. The small bright dots are IV values that were given in the original dataset. The smooth surface is the completed IV surface after the final method inferred the missing cells.

Interactive versions are also generated:

```text
for_generating_readme/iv_surface_ce_3d.html
for_generating_readme/iv_surface_pe_3d.html
for_generating_readme/iv_surface_combined_3d.html
```

![Final CE IV surface](for_generating_readme/iv_surface_ce_3d.png)

![Final PE IV surface](for_generating_readme/iv_surface_pe_3d.png)

![Combined CE and PE IV surfaces](for_generating_readme/iv_surface_combined_3d.png)

The late surface jump is the key visual result. Near expiry, IV rises sharply and the wings become much harder to extrapolate. This is why a single global quadratic or a single temporal smoother was not enough.

## 4. Filling Logic

### Interior Cells

For an interior missing value, the model uses only the same row and same option type. A missing CE cell is predicted from CE observations at the same timestamp; a missing PE cell is predicted from PE observations at the same timestamp.

The observed smile points are:

$$
(x_i, y_i), \qquad x_i = \frac{K_i}{S}
$$

where \(y_i\) is observed IV at strike \(K_i\).

For target moneyness \(x_0\), the local quadratic model is:

$$
y_i \approx \beta_0 + \beta_1(x_i-x_0) + \beta_2(x_i-x_0)^2
$$

The prediction is \(\hat{y}(x_0)=\beta_0\), because at the target point \(x_i-x_0=0\).

Nearby strikes receive more weight:

$$
w_i = \exp\left(-\frac{(x_i-x_0)^2}{2h}\right)
$$

The weighted least-squares problem is:

$$
\hat{\beta}
= \arg\min_\beta
\sum_i w_i \left(y_i - X_i\beta\right)^2
$$

which gives the normal equation:

$$
(X^\top W X)\hat{\beta} = X^\top W y
$$

The bandwidth \(h\) controls locality. The script chooses \(h\) by leave-one-out validation over:

$$
h \in \{5\cdot10^{-5},\ 7\cdot10^{-5},\ 10^{-4},\ 1.5\cdot10^{-4},\ 2\cdot10^{-4}\}
$$

For each candidate:

$$
\operatorname{MSE}(h)
= \frac{1}{n}\sum_i
\left(\hat{y}_{-i}(x_i;h)-y_i\right)^2
$$

and:

$$
h^\star = \arg\min_h \operatorname{MSE}(h)
$$

After local WLS, the model tries shape-preserving PCHIP interpolation. PCHIP is used only when the target lies inside the observed range, never for extrapolation.

The final interior prediction is:

$$
\hat{y}_{\text{interior}}
= 0.75\,\hat{y}_{\text{WLS}}
+ 0.25\,\hat{y}_{\text{PCHIP}}
$$

In the full run, all `4,491` non-edge fills used this PCHIP blend.

### Edge Cells

Edges are different because one side of the smile is missing:

```text
observed observed observed missing_1 missing_2 missing_3
```

The final code does not jump straight to the farthest missing point. It fills progressively:

```text
missing_1 -> use observed side
missing_2 -> use observed side + missing_1
missing_3 -> use observed side + missing_1 + missing_2
```

For each edge target, the model builds three estimates:

```text
primary   : valid-side observed points plus progressive context
corrected : valid-side points plus prior predictions at actual moneyness
quadratic : local wing neighborhood with degree selected by LOO
```

For edges, the model searches both degree and bandwidth:

$$
d \in \{1,2\}, \qquad h \in \{5\cdot10^{-5}, 7\cdot10^{-5}, 10^{-4}, 1.5\cdot10^{-4}, 2\cdot10^{-4}\}
$$

and selects:

$$
(d^\star,h^\star)
= \arg\min_{d,h}\operatorname{LOO\_MSE}(d,h)
$$

The final edge ensemble is:

$$
\hat{y}_{\text{edge}}
= 0.72\,\hat{y}_{\text{primary}}
+ 0.14\,\hat{y}_{\text{corrected}}
+ 0.14\,\hat{y}_{\text{quadratic}}
$$

## 5. Progressive Edge Filling

This figure shows the progressive edge idea on a real row from the final submission diagnostics. Gold crosses are still-missing edge cells. Gold diamonds are filled values. The red diamond is the newest value added at that step.

![Progressive edge fill sequence](for_generating_readme/progressive_edge_fill_sequence.png)

The important point is that later wing values get more local context than they would have if the method tried to extrapolate the entire block in one shot.

## 6. Diagnostics

The full final run completed every missing value without a global-median fallback.

```text
total filled cells       : 5,460
interior fills           : 4,491
edge fills               : 969
degree 2 edge selections : 812
degree 1 edge selections : 157
global median fallback   : 0
```

![Fill diagnostics snapshots](for_generating_readme/fill_diagnostics_snapshots.png)

![Final model decisions](for_generating_readme/final_model_decisions.png)

![Prediction source counts](for_generating_readme/prediction_source_counts.png)

The final filled smiles below show the completed cross-sections. Yellow diamonds are cells that were originally missing and filled by the model.

![Filled smile examples](for_generating_readme/filled_smile_examples.png)

## 7. Synthetic CV Validation

The repository includes a synthetic CV system under:

```text
everything_else/cv_validation_system/
```

The validation procedure:

1. Start from observed values.
2. Hide a subset of values.
3. Run `final_submission.py` on the damaged dataset.
4. Score only the hidden cells.

I ran:

```bash
python final_submission.py \
  --data everything_else/cv_validation_system/cv_split/not_dataset.csv \
  --out-prefix readme_cv_final \
  --skip-cv
```

Then evaluated with:

```bash
python everything_else/cv_validation_system/evaluate_cv_predictions_with_heatmaps.py \
  --truth everything_else/cv_validation_system/cv_split/holdout_truth.csv \
  --pred filled_dataset_readme_cv_final.csv \
  --base everything_else/cv_validation_system/cv_split/not_dataset.csv \
  --out-dir for_generating_readme/cv_eval_final_submission \
  --top-smile-plots 12 \
  --sample-smile-plots 8
```

Overall synthetic CV results:

```text
n hidden cells scored : 2621
MSE                   : 0.0001417534
RMSE                  : 0.0119060232
MAE                   : 0.0038038407
median absolute error : 0.0005887275
bias                  : 0.0002157616
p95 absolute error    : 0.0181816275
p99 absolute error    : 0.0615417097
```

![CV metric cards](for_generating_readme/cv_metric_cards.png)

The strongest errors are concentrated near the expiry regime, which matches the EDA and the final IV surface.

![Predicted vs actual](for_generating_readme/cv_predicted_vs_actual_theme.png)

![Absolute error over time](for_generating_readme/cv_abs_error_over_time_theme.png)

![Absolute error vs moneyness](for_generating_readme/cv_abs_error_vs_moneyness_theme.png)

![MSE by regime](for_generating_readme/cv_mse_by_regime_theme.png)

![Binned absolute error heatmap](for_generating_readme/cv_abs_error_heatmap_theme.png)

![Signed error heatmap](for_generating_readme/cv_signed_error_heatmap_theme.png)

One top-error smile is shown below. It shows whether the miss came from a whole-smile shift, wing movement, or thin local context.

![Top-error smile example](for_generating_readme/cv_top_error_smile_theme.png)

## 8. What I Tried

I tried several model families before choosing the final one.

![Strategy CV comparison](for_generating_readme/strategy_cv_comparison.png)

The raw quadratic approach was a useful baseline, but global quadratic wing extrapolation was unstable.

The progressive quadratic edge versions were the first strong structural improvement: edge blocks should be filled from the observed boundary outward.

The linear-edge experiments showed that degree selection matters. Sometimes a line is safer than a quadratic at the wing, but forcing every edge to be linear throws away real curvature.

The local-polynomial edge experiments were stronger because they kept the fit local in moneyness and selected smoothing by validation.

The Jan 27 temporal experiments showed why over-specializing to expiry day is risky. Time information helps explain the market, but a global temporal correction can create large tail errors when the smile changes abruptly.

The CNN idea was conceptually attractive because an IV smile is a 1D signal, but the dataset is small and validation/leakage risk is high. The final method stayed transparent, cross-sectional, and easy to diagnose.

The final model is the practical conclusion:

```text
use same-row smile structure first
separate interiors from edges
choose smoothing by LOO
blend PCHIP only for safe interior interpolation
fill edge blocks progressively
avoid aggressive temporal corrections in the final submission
```

## 9. Function Map

| Function | Role |
|---|---|
| `parse_metadata` | Extracts strike, expiry, and option type from contract names. |
| `collect_same_row_points` | Builds same-timestamp CE or PE smile points. |
| `local_poly_wls_pred` | Core local weighted least-squares predictor. |
| `select_bandwidth_by_loo` | Chooses interior bandwidth by leave-one-out MSE. |
| `select_bandwidth_and_degree_by_loo` | Chooses edge bandwidth and degree. |
| `pchip_same_row_pred` | Computes shape-preserving interior interpolation. |
| `get_edge_blocks` | Finds missing wing blocks at each timestamp. |
| `predict_non_edge_local_poly` | Runs interior WLS plus PCHIP blend. |
| `predict_edge_ensemble` | Builds and combines the three edge estimates. |
| `predict_cell` | Routes each missing cell to the correct engine. |
| `build_missing_cell_fill_order` | Enforces progressive edge filling order. |
| `main` | Loads data, fills all cells, writes submission and diagnostics. |

## 10. Rebuilding Figures

The README-specific figure generator is:

```bash
python for_generating_readme/generate_readme_eda.py
```

It reads the final filled dataset, diagnostics, original dataset, and synthetic CV outputs. It writes the images and interactive Plotly surfaces into:

```text
for_generating_readme/
```

The CV evaluator outputs used by the README live in:

```text
for_generating_readme/cv_eval_final_submission/
```

So this README is tied to generated artifacts from the actual final submission run, not hand-written summaries.
