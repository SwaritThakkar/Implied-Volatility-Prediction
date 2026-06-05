# Implied Volatility Prediction

This repository contains my work for filling missing implied volatility values in a NIFTY options dataset and generating a Kaggle-style submission file.

The final submission generator is:

```bash
final_submission/try_final_pchip_interior.py
```

The other folders contain validation tooling, EDA notebooks/dashboards, archived strategy outputs, and experiments that were useful during development but were not the final selected method.

## Repository Structure

```text
.
├── final_submission/
│   └── try_final_pchip_interior.py      # Final submission generator
├── things_tried/                        # Experimental methods and diagnostics
├── was_better_than_submission_but_not_confident/
│   └── try_final_pchip_temporal_j27_pchiptime.py
├── eda/                                 # Exploratory visualizations and notebooks
├── cv_validation_system/                # Synthetic holdout creation and evaluation tools
├── strategies_and_results/              # Saved strategy variants, submissions, diagnostics, plots
├── monte_carlo_for_try.py/              # Monte Carlo error-analysis experiment
├── submission-converter.ipynb           # Notebook utility for submission conversion
├── shivank_dataset.csv                  # Dataset copy/reference file
└── LICENSE
```

The `strategies_and_results/` directory is mostly an archive of attempted model variants and their outputs. Many subdirectories include generated submissions, filled datasets, diagnostics, grouped error reports, and plots.

## Final Method

The final method is a pure cross-sectional imputation strategy. For each timestamp, it treats the option smile as a same-row curve over strike/moneyness and fills only the missing option IV cells. It does not train a global neural network or use future timestamps to predict earlier timestamps.

The script separates missing values into two cases:

1. Interior/non-edge missing cells: missing strikes that still have observed same-option-type values on both sides.
2. Edge missing cells: consecutive missing values at the left or right wing of a CE or PE strike grid, where same-row interpolation is no longer possible.

### Input Format

The input CSV is expected to contain:

- `datetime`
- `underlying_price`
- option columns named like `NIFTY27JAN2625200CE` or `NIFTY27JAN2623800PE`

The option-column parser extracts:

- underlying symbol
- expiry
- strike
- option type, either `CE` or `PE`

The final script sorts rows by parsed datetime before filling.

## How to Run

From the repository root:

```bash
python final_submission/try_final_pchip_interior.py --data cv_validation_system/dataset.csv
```

To skip the script's internal synthetic CV check:

```bash
python final_submission/try_final_pchip_interior.py --data cv_validation_system/dataset.csv --skip-cv
```

You can also set a custom output prefix:

```bash
python final_submission/try_final_pchip_interior.py \
  --data cv_validation_system/dataset.csv \
  --out-prefix my_submission
```

Required Python packages for the final script:

```text
numpy
pandas
scipy
tqdm
```

`scipy` is used for PCHIP interpolation. If SciPy is unavailable, the script falls back to the local-polynomial method and simply does not use the PCHIP blend.

## Final Outputs

The script writes outputs to the current directory, or to `/kaggle/working` when running on Kaggle:

```text
filled_dataset_<out_prefix>.csv
submission_<out_prefix>.csv
diagnostics_<out_prefix>.csv
cross_section_diagnostics_<out_prefix>.csv
```

For the default final run, these are:

```text
filled_dataset_try_final_pchip_interior.csv
submission_try_final_pchip_interior.csv
diagnostics_try_final_pchip_interior.csv
cross_section_diagnostics_try_final_pchip_interior.csv
```

The submission file contains two columns:

- `id`: built as `<datetime>||<contract_column>`
- `value`: filled IV prediction

## Interior/Non-Edge Logic

For non-edge missing values, the final model uses same-row, same-option-type local quadratic weighted least squares over moneyness:

```text
moneyness = strike / underlying_price
```

For a missing CE value, only observed CE values from the same timestamp are used. For a missing PE value, only observed PE values from the same timestamp are used.

The local polynomial model:

- uses degree 2 by default
- uses Gaussian weights around the target moneyness
- selects bandwidth by row-wise leave-one-out validation
- searches the bandwidth grid:

```text
[5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4]
```

The final method then adds a conservative PCHIP interpolation blend for true interior cells:

```text
final_interior = 0.75 * local_quadratic_wls + 0.25 * pchip_prediction
```

PCHIP is only used when:

- SciPy's `PchipInterpolator` is available
- at least 4 same-row observed points exist
- the target moneyness lies inside the observed moneyness range

PCHIP is never used for extrapolation. This was intentionally kept conservative because validation suggested PCHIP was helpful as a small interior blend, not as a full replacement for the local WLS baseline.

## Edge Logic

Edge missing cells are handled separately because there is no observed same-option-type value on one side of the missing strike. Simple interpolation cannot solve these wing gaps.

The final edge model is a progressive ensemble of three edge predictors:

```text
edge_prediction =
    0.72 * claude_style_local_poly
  + 0.14 * corrected_local_poly
  + 0.14 * progressive_quadratic
```

For edge blocks, the script fills from the observed boundary outward. For example, for a right-edge block:

```text
observed observed observed missing_1 missing_2 missing_3
```

it predicts `missing_1` first, then uses that prediction as part of the information available when predicting `missing_2`, and so on. The same idea is applied symmetrically to left-edge blocks.

For the local-polynomial edge components, the script selects both:

- polynomial degree from `{1, 2}`
- bandwidth from `[5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4]`

using leave-one-out error on the available edge-side training points. If an edge model cannot produce a finite prediction, the script falls back to the global median IV.

## Diagnostics

The diagnostics CSV records one row per filled cell and includes:

- timestamp, contract, option type, strike
- final prediction
- whether the cell was an edge
- selected model/source
- selected bandwidth and LOO MSE
- selected edge degree
- whether PCHIP was used
- base WLS and PCHIP predictions for interior cells
- edge component predictions
- training columns used for the prediction
- edge block metadata

This is useful for debugging individual predictions and comparing behavior across interior gaps, left-edge gaps, and right-edge gaps.

## Validation Approach

The final script includes an internal synthetic CV validation mode. It randomly masks a fraction of observed IV values, predicts them using the same routing logic, and reports:

- overall MSE
- edge MSE
- interior MSE
- MSE on cells where PCHIP was used

This validation is not a perfect substitute for the private leaderboard, but it helped compare local changes without relying only on intuition.

The separate `cv_validation_system/` folder contains a fuller validation setup with:

- synthetic holdout creation
- holdout truth files
- prediction evaluation
- metrics by date, row, option type, regime, contract, and moneyness bucket
- heatmap and smile-level error plots

## Things Tried

The `things_tried/` folder contains the main experiments that led to the final method.

### Baseline and Local-Polynomial Family

- `try.py`: strong baseline using same-row local quadratic WLS for non-edge gaps and a progressive edge ensemble. This became the main foundation for the final submission.
- `try_v2.py`: tested linear edge local-polynomial extrapolation with LOO bias correction. The idea was that linear extrapolation might be more stable than quadratic on sparse edge points.
- `pure_cross_section_locpoly_adaptive_v2.py`: local polynomial WLS with temporal residual correction for edge cells. It explored whether recent timestamp residuals could correct systematic wing bias.
- `auto_tuned_cross_section_imputer.py`: one-file workflow that created synthetic validation masks, tuned hyperparameters, and generated outputs automatically.

### Edge-Specific Experiments

- `try_progressive_linear_edge_v2.py`: simplified progressive linear edge extrapolation without cross-option transfer or edge ensemble complexity.
- `try_progressive_linear_edge_v3_bias_eda.py`: added local timestamp-specific edge wrongness correction for selected CE/PE wings.
- `try_mirror_slope_edge.py`: tested a mirrored OTM slope prior for edge extrapolation.
- `try_regime_edge_submit.py`: used regime-specific bandwidths, especially for Jan 27 wing behavior.

### PCHIP and Final-Method Variants

- `try_final_pchip_bucket.py`: explored bucketed PCHIP behavior.
- `try_final_pchip_interior_edge_cv_blend.py`: tested PCHIP interior blending together with edge blend validation.
- `try_final_pchip_interior_regime_adaptive.py`: made the PCHIP/interior method regime-adaptive.
- `try_final_pchip_regime_adaptive.py`: another regime-aware PCHIP variant.
- `try_final_pchip_rowsearch.py`: searched row-level PCHIP blending behavior.
- `try_final_validated_tuned.py`: grid-searched CE-only PCHIP weights and edge ensemble weights, adopting changes only if synthetic CV improved.

The final selected script, `final_submission/try_final_pchip_interior.py`, keeps the robust edge logic and uses a fixed conservative 25% PCHIP blend only for true interior interpolation.

### Temporal and Underlying-Signal Experiments

- `try_final_pchip_temporal_j27.py`: tested special temporal handling for Jan 27.
- `try_final_pchip_temporal_j27_pchiptime_per_option_default0.py`: tested per-option temporal PCHIP behavior with a default-zero temporal weight.
- `try_final_pchip_underlying_signal.py`: explored whether underlying-price movement could explain residual IV errors and improve predictions.
- `was_better_than_submission_but_not_confident/try_final_pchip_temporal_j27_pchiptime.py`: a temporal Jan 27 variant that looked promising in some validation but was not selected because it was less confidence-inspiring for final submission.

### Neural-Network Experiments

- `try_cnn.py`: walk-forward CNN imputer with strict causal training windows, synthetic SVI pretraining ideas, no-arbitrage penalties, and iterative inpainting.
- `try_cnn_fixed.py`: masked 1D convolutional autoencoder over CE/PE smile vectors, with row-median normalization and blending back to the local-polynomial baseline based on reconstruction error.
- `kaggle_cnn_smile_iv_imputer.ipynb`: notebook version of the CNN smile-imputation experiments.

These models were conceptually attractive but added training complexity and validation risk. The final selected method stayed with the more transparent cross-sectional model.

### EDA and Error Analysis

- `iv_contract_time_series_dashboard_calendar_gaps.py`: interactive dashboard for inspecting IV through real calendar time while preserving non-trading gaps.
- `claude_error.py`: Monte Carlo masking and simple interpolation error dashboard generation.
- `error_dashboard.html`: generated dashboard artifact.
- `score_based_iv_completion_kaggle.ipynb`: score-based imputation exploration notebook.

The `eda/` folder contains additional notebooks and dashboards for inspecting IV surfaces, moneyness-time behavior, lag/calendar gaps, contract time series, and underlying-price movement.

