# Synthetic Cross-Validation System for NIFTY IV Imputation

This folder gives you a robust local validation system before Kaggle submission.

## Why this exists

The real Kaggle test missing values are unknown. To compare methods locally, we create a fake validation set:

1. Start with `dataset.csv`.
2. Hide some values that are currently observed.
3. Save the hidden true values separately.
4. Run your imputer on the damaged file.
5. Score only those hidden values.

This lets you compute a local MSE before submitting.

## Files

### `create_synthetic_cv_dataset.py`

Creates:

- `cv_split/not_dataset.csv`
- `cv_split/holdout_truth.csv`
- `cv_split/holdout_mask.csv`
- `cv_split/holdout_summary.csv`
- `cv_split/holdout_contract_summary.csv`
- `cv_split/cv_config.json`

### `evaluate_cv_predictions.py`

Evaluates your filled prediction file against `holdout_truth.csv`.

Creates:

- `metrics_summary.json`
- `metrics_summary.csv`
- `error_rows.csv`
- `worst_errors.csv`
- `group_metrics_by_*.csv`
- `plots/*.png`

## Quick start

### Step 1: Create synthetic validation split

```bash
python create_synthetic_cv_dataset.py \
  --input dataset.csv \
  --out-dir cv_split \
  --seed 42 \
  --holdout-frac 0.12 \
  --min-holdout-27jan 350
```

This creates `cv_split/not_dataset.csv`.

### Step 2: Run your imputer

Change your imputer's `DATA_PATH` to:

```python
DATA_PATH = Path("cv_split/not_dataset.csv")
```

Save the result as something like:

```text
cv_split/my_filled_dataset.csv
```

### Step 3: Evaluate

```bash
python evaluate_cv_predictions.py \
  --truth cv_split/holdout_truth.csv \
  --pred cv_split/my_filled_dataset.csv \
  --out-dir cv_eval_results
```

## What to inspect

Start with:

```text
cv_eval_results/metrics_summary.json
cv_eval_results/worst_errors.csv
cv_eval_results/group_metrics_by_regime_option_type.csv
cv_eval_results/group_metrics_by_contract.csv
```

Important plots:

```text
cv_eval_results/plots/predicted_vs_actual.png
cv_eval_results/plots/error_histogram.png
cv_eval_results/plots/abs_error_vs_moneyness.png
cv_eval_results/plots/abs_error_over_time.png
cv_eval_results/plots/mse_by_regime.png
```

## Design choices

The holdout is not purely random. It is stratified by:

- pre-27 Jan vs 27 Jan
- CE vs PE
- moneyness bucket

It also keeps a minimum number of observed contracts per timestamp and option type, so your cross-sectional smile models still have enough data to fit.

Original missing values are not scored. Only synthetic holdout values are scored.
