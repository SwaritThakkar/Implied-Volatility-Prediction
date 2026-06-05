# Implied Volatility Prediction across the Nifty50 options chain (PROJECT REPORT)

This README file is the __project report__ for IITR Finclub Open Projects 2026 PS-2

Implied volatility (IV) measures how much the market believes the price of a stock (or other underlying asset) will move in the future. In practice, IV is one of the most important quantities in options markets because it captures the market’s expectations of future uncertainty and typically varies across strikes, forming structures known as __volatility smiles__ or __volatility skews__.

The objective of this project is to reconstruct a partially observed implied-volatility surface. Rather than treating the task as generic missing-value problem, I approach the problem as a cross-sectional structure prediction problem. My final submission (final_submission.py) does not take into account any temporal dependencies, as I found that for this particular problem, cross-sectional structure across moneyness and IV generate high quality signals.

The remainder of this README is organized as follows. First, I present exploratory analysis of the dataset motivating the modeling decisions. Next, I describe the final methodology, including the interpolation and extrapolation procedures used for interior and edge regions of the smile, which was critical in reducing overall mse. I then present diagnostics, validation results, and finally, comparisons against alternative approaches (some of which were very promising and innovative), which unfortunately did not work for this dataset.

Before moving to the solution, the repository is centered around one file:

```bash
final_submission.py
```

That is the file I am submitting. It fills the missing implied-volatility values and writes the final submission file:

```text
submission_final.csv
```

To run it from the repository root, run the following in terminal or run the final_submission.py file:

```bash
python final_submission.py --data everything_else/cv_validation_system/dataset.csv 
```

It produces:

```text
filled_dataset_final.csv
submission_final.csv
diagnostics_final.csv
cross_section_diagnostics_final.csv
```

On the final full dataset run, the script filled `5,460` missing IV cells. I've stored the 4 generated files in the folder:
```text
submission_files
```

The rest of the directory contains submission files, eda files, validation system files that I used during the competition. All the files in the folder "everything_else" helped me make the final submission.

## Table Of Contents

- [1. Dataset Visualisation/EDA](#1-dataset-eda)
- [2. Problem Reduction](#2-final-idea)
- [3. Final IV Surfaces](#3-final-iv-surfaces)
- [4. Filling Logic](#4-filling-logic)
- [5. Progressive Edge Filling](#5-progressive-edge-filling)
- [6. Diagnostics](#6-diagnostics)
- [7. Synthetic CV Validation](#7-synthetic-cv-validation)
- [8. What I Tried](#8-what-i-tried)
- [9. Function Map](#9-function-map)
- [10. Rebuilding Figures](#10-rebuilding-figures)

## 1. Dataset EDA (Exploratory Data Analysis) / Visualisation

This section sets the motivation for all my methods and decisions presented in this report. This was my first step in the competition.

The original dataset is a __Nifty50__ options chain surface (obviously with missing values) expiring on __27th Jan 2026__. It has `975` timestamp rows, `28` option contracts (14 Put options and 14 Call options), and `5,460` missing IV (Implied Volatility) cells, which is exactly `20%` of the all the options' IVs. Furthermore, since the missing values for each option were distributed randomly across time and not in a set train/test dataset, it seemed implausible that any ML/DL model would work, but nonetheless, I tried a few ML/DL approaches (and even a CNN-based method) all of beared no fruit.

Moving ahead, the first important observation is that this is not one smooth time regime. All days from Jan 7 to Jan 23 have low, stable IV, but Jan 27 is different. Since it was the expiry day of all options, we expect the IVs to have a very fastly increasing surface, which is what we observed as shown in the picture(s) below. In the pictures below, we clearly observe a regime shift (pre 27th Jan and 27th Jan). After looking at more expiry day IV prediction research papers, and not finding anything useful, I decided to just use different hyperparameters for both regimes and that is what the current solution does as well. However, I feel there is much to be dug upon on this topic.

![Dataset EDA regime formation](for_generating_readme/dataset_regime_eda.png)

#
The missingness is also very uniformly, but randomly distributed. Each timestamp gives a partial CE (call option) smile and a partial PE (put option) smile across strikes. As you can see from the picture below (given and missing values across time and moneyness), there is no pattern to be found from the structure of missing data. Furthermore, looking at the below figure, it gives a clear direction to proceed it - to focus not too much on previous timestamps, but on the cross-sectional structure (IV smile and skew) per timestamp.

![Given and missing values across cross-sections](for_generating_readme/missing_given_cross_section_eda.png)

#
Further analysis of Raw smiles make the regime shift obvious by observing the scale of Y-axis in the 6 figures below. Early smiles are low and calm, while Jan 27 smiles become increasing steep and noisy (if looked at across time).

![Raw smile regime snapshots](for_generating_readme/raw_smile_regime_snapshots.png)

This EDA is the reason I trusted same-timestamp smile structure more than a global time-series smoother. Although the lag-1 EDA showed high adjacent-time correlation Jan 27 has much larger IV changes for the lag-1 signal to effectively capture (or atleast that's what I found).

Further EDA files can be found in the folder `everything_else/eda`


## 2. Problem reduction 

This section presents the reduction/translation of the problem statement into a geometry problem along with a key insight I realized (after 60 submissions!)

The model treats each timestamp as an implied-volatility smile, and thus solves for missing values using geometric fits. For a missing option cell, the problem reduces to:

> Fit a curve given the datapoints that represent the current underlying asset scenario. The curve should be a function of IV vs moneyness. Now, I just had to find the "best" curve.

We know, moneyness describes an option's value at a particular point in time. It relates to the option's strike price and the price of its underlying asset and indicates whether the option would make money if it were exercised immediately.

After trying various functions of strike and underlying price, I found that a simple ratio works the best. 

Thus, we define `moneyness (= x)` as:

```math
x = \frac{K}{S}
```

where `K` is strike and `S` is the underlying price.

Now, the key observation that was responsible for reducing the mse was __to classify missing values based on their relative positions across the respective option class (call/put).__

Thus, every missing cell is "routed" into one of two geometries:

```text
interior gap: the missing value was surrounded by non-missing values
edge wing: the missing value was not surrounded by non-missing values on both sides (ie: it was on an "edge")
```

Clearly, interior gaps are safer and easier to predict because there are observed strikes on both sides. However edge wings are harder and not accurate because the model has to extend the smile beyond the observed support, without regression on the other side.

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

```math
(x_i, y_i), \qquad x_i = \frac{K_i}{S}
```

where `y_i` is observed IV at strike `K_i`.

For target moneyness `x_0`, the local quadratic model is:

```math
y_i \approx \beta_0 + \beta_1(x_i - x_0) + \beta_2(x_i - x_0)^2
```

The prediction is:

```math
\hat{y}(x_0) = \beta_0
```

because at the target point:

```math
x_i - x_0 = 0
```

Nearby strikes receive more weight:

```math
w_i = \exp\left(-\frac{(x_i - x_0)^2}{2h}\right)
```

The weighted least-squares problem is:

```math
\hat{\beta}
=
\arg\min_{\beta}
\sum_i
w_i
\left(
y_i - X_i\beta
\right)^2
```

This gives the normal equation:

```math
(X^\top W X)\hat{\beta}
=
X^\top W y
```

The bandwidth `h` controls locality. The script chooses `h` by leave-one-out validation over:

```math
h
\in
\left\{
5\cdot10^{-5},
7\cdot10^{-5},
10^{-4},
1.5\cdot10^{-4},
2\cdot10^{-4}
\right\}
```

For each candidate bandwidth:

```math
\mathrm{MSE}(h)
=
\frac{1}{n}
\sum_i
\left(
\hat{y}_{-i}(x_i;h) - y_i
\right)^2
```

Then it selects:

```math
h^\star
=
\arg\min_h
\mathrm{MSE}(h)
```

After local WLS, the model tries shape-preserving PCHIP interpolation. PCHIP is used only when the target lies inside the observed range, never for extrapolation.

The final interior prediction is:

```math
\hat{y}_{\mathrm{interior}}
=
0.75\,\hat{y}_{\mathrm{WLS}}
+
0.25\,\hat{y}_{\mathrm{PCHIP}}
```

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

```math
d
\in
\{1,2\}
```

```math
h
\in
\left\{
5\cdot10^{-5},
7\cdot10^{-5},
10^{-4},
1.5\cdot10^{-4},
2\cdot10^{-4}
\right\}
```

For every candidate pair `(d, h)`, it computes the leave-one-out error:

```math
\mathrm{LOO\_MSE}(d,h)
=
\frac{1}{n}
\sum_{i=1}^{n}
\left(
\hat{y}_{-i}(x_i;d,h) - y_i
\right)^2
```

Then it selects:

```math
(d^\star,h^\star)
=
\arg\min_{d,h}
\mathrm{LOO\_MSE}(d,h)
```

The final edge ensemble is:

```math
\hat{y}_{\mathrm{edge}}
=
0.72\,\hat{y}_{\mathrm{primary}}
+
0.14\,\hat{y}_{\mathrm{corrected}}
+
0.14\,\hat{y}_{\mathrm{quadratic}}
```

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
