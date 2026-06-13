# Cross-Market Regressions

Reusable, config-driven framework for training, saving, reloading, explaining, and using a simple cross-market linear regression model.  A typical use case is predicting a target index or instrument's next-session return from one or more source instrument returns, optional FX returns, and optional market features.

> **Warning**
>
> This is a statistical cross-market signal model, not a guaranteed arbitrage formula.
> Predictions depend on training data quality, target/source alignment, live quote quality,
> FX convention, and regime stability.

## What the framework does

The framework:

1. Loads a YAML/JSON config into typed Python objects.
2. Loads configured source, target, FX, and optional market series from explicit providers.
3. Builds generic source/FX return features.
4. Builds the target next-session return label.
5. Trains a one-layer TensorFlow/Keras linear regression model.
6. Saves weights, scaler state, metadata, metrics, and training history.
7. Reloads the same architecture for prediction and coefficient explanation.

The front face is the notebook `notebooks/01_cross_market_regression_frontface.ipynb`; real logic lives in `src/cross_market_regression/` modules.

## Source return

The default source feature is:

```text
x_source_t = source_signal_price_t / source_reference_price_t - 1
```

For historical training, the signal is commonly a close and the reference is commonly the regular-session open.  For live prediction, the signal can be a current or after-hours quote supplied to the prediction function.

## Target return

The supervised target label is:

```text
y_target_t = target_next_close_t / target_current_close_t - 1
```

The predicted target level is:

```text
target_predicted_level = target_current_close * (1 + y_hat)
```

## Configure EWY -> KOSPI

Use `configs/examples/ewy_kospi.yaml` as the primary example.  It defines source, reference, target, FX, feature, and model settings outside the code.  The reusable package does not hard-code market names or coefficients.

## Configure another pair

Create another file under `configs/examples/` or your own config directory and change the configured assets/features/model directory.  Examples are provided for multiple source/target pairs, including `qqq_ndx.yaml`, `ewj_nikkei.yaml`, `ewt_taiwan.yaml`, and `soxx_semis.yaml`.


## Developer setup

Create and activate a virtual environment, then install the package in editable mode with developer test tooling:

```bash
python -m pip install -e ".[dev]"
```

The editable package install installs the runtime dependencies declared by the package (`numpy`, `pandas`, `pyyaml`, and `tensorflow`), while the `dev` extra adds `pytest` for local test runs.

## Running tests

For a minimal or lightweight validation pass, run the default test suite in your current environment:

```bash
pytest
```

Tests that require unavailable optional runtime libraries are skipped by their pytest markers or import checks. For a full local test run, first install the developer environment above so `numpy`, `pandas`, `pyyaml`, `tensorflow`, and `pytest` are available, then run:

```bash
python -m pytest
```

## Schwab authentication and provider behavior

Schwab credentials must come from environment variables only.  Use `.env.example` as the template:

```text
SCHWAB_API_KEY=
SCHWAB_APP_SECRET=
SCHWAB_CALLBACK_URL=http://localhost:8182/callback
SCHWAB_TOKEN_PATH=artifacts/raw/schwab/token.json
```

Do not commit real `.env` files or token JSON files.  `create_schwab_client` reads the environment variable names from the auth config and returns a credential bundle; production code should pass a project-specific authenticated Schwab client into `SchwabPriceProvider(client=...)`.

`SchwabPriceProvider` supports daily history, intraday history, and quote snapshots when that injected client implements `get_daily_ohlcv`, `get_intraday_ohlcv`, and `get_quote_snapshot`.  SDK-style clients with `get_price_history` / `get_quotes` are also accepted.  Historical daily and intraday payloads are normalized to the same `symbol,date,open,high,low,close,volume,source` schema used by CSV providers.

Historical Schwab responses are cached as JSON under `artifacts/raw/schwab` by default.  Pass `cache_dir=...` to change the location or `use_cache=False` on history calls to bypass reading and writing cache files.  Quote snapshots are treated as live data and are not cached.


## Before using actual market data

Use this checklist before training against real data:

1. **Install the runtime stack**: `numpy`, `pandas`, `pyyaml`, and `tensorflow` are required for real CSV loading, feature construction, training, and prediction. The lightweight test environment can pass with these tests skipped, so run the full suite in an environment where those packages are installed.
2. **Prepare CSV inputs explicitly**: each configured CSV must include at least `date` and `close`. Historical source features that use `open`, `previous_close`, or another price mode must also include the matching column.
3. **Keep secrets and generated data out of git**: real `.env` files, OAuth tokens, raw/processed data, logs, and generated model weights are ignored intentionally.
4. **Run a smoke train before committing to a long run**: use a short `epochs` value and verify that the model directory contains `model.weights.h5`, `scaler.json`, `metadata.json`, `metrics.json`, and `training_history.csv`.
5. **Verify train/predict compatibility**: the training CLI now saves scaler metadata in the same shape consumed by live prediction and explanation. After training, run `cmr-predict-live` with manual feature inputs or source/reference prices before relying on outputs.
6. **Validate provider choice**: `csv`, `fx_csv`, and `target_csv` are local-file providers. The `schwab` provider requires a supplied project-specific authenticated client for live API access; without one it raises an explicit `NotImplementedError`.

### CSV data contract

Minimum daily CSV schema:

```text
date,close
2024-01-02,101.50
```

Recommended OHLCV schema:

```text
date,open,high,low,close,volume
2024-01-02,100.00,102.00,99.50,101.50,123456
```

Notes:

- `date` values are normalized to `YYYY-MM-DD`.
- Column names are lower-cased by the CSV provider.
- `price_column` can select an alternate close-like column, such as `adjusted_close`.
- Example configs refer to `data/raw/...` files, but raw market data is intentionally not committed.

## Train

```bash
python -m cross_market_regression.cli.train_model \
  --config configs/examples/ewy_kospi.yaml
```

The CLI defaults to the configured `model.model_dir`; `--output-dir` can override it for the legacy row-based flow.

## Predict

```bash
python -m cross_market_regression.cli.predict_live \
  --model-dir artifacts/models/ewy_kospi_linear \
  --target-current-close 8639.41 \
  --source-signal-price 197.00 \
  --source-reference-price 212.00 \
  --fx-signal 1533.02 \
  --fx-reference 1516.93
```

Programmatic API:

```python
from cross_market_regression.modeling.predict import predict_target_from_inputs

pred = predict_target_from_inputs(
    model_dir="artifacts/models/ewy_kospi_linear",
    target_current_close=8639.41,
    source_signal_price=197.00,
    source_reference_price=212.00,
    fx_signal=1533.02,
    fx_reference=1516.93,
)
```

## Inspect learned coefficients

```bash
python -m cross_market_regression.cli.explain_model \
  --model-dir artifacts/models/ewy_kospi_linear
```

The explanation path converts standardized TensorFlow weights back to a raw-return formula:

```text
y_hat = raw_intercept + beta_1*x_feature_1 + beta_2*x_feature_2 + ...
```

## Avoiding lookahead

The alignment module maps:

```text
source_session_date_t -> target_current_session_date_t -> target_next_session_date_t
```

Features are built from source/FX/current-session data, while `target_next_close` is used only as the label.  `validate_no_lookahead` checks duplicate mappings, invalid target date ordering, and accidental next-close leakage in raw feature columns.

## Saved files

A trained model directory contains:

```text
model.weights.h5
scaler.json
metadata.json
metrics.json
training_history.csv
```

`metadata.json` records feature order, target name, split dates/counts, model settings, threshold, package version, and git commit when available.

## What not to commit

Do not commit:

- real `.env` files;
- Schwab token JSON;
- raw market data;
- processed/generated artifacts;
- model weights unless explicitly intended.

Do commit source code, tests, config examples, `.env.example`, and documentation.
