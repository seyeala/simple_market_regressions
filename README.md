# Cross Market Regression

`cross-market-regression` is a reusable Python framework for training and using simple linear cross-market signal models. It keeps market choices outside code: source instruments, target instruments, references, FX inputs, feature names, target labels, and training settings are all supplied by YAML/JSON configuration.

> **Warning**
>
> This is a statistical cross-market signal model, not a guaranteed arbitrage formula. Predictions depend on training data quality, target/source alignment, live quote quality, FX convention, and regime stability.

## What the framework does

The framework builds a supervised regression dataset from aligned market price series, trains a one-layer linear TensorFlow/Keras model, saves the trained artifacts, and provides command-line tools to:

- train a configured source/target model;
- predict with a saved model from feature rows supplied at runtime;
- inspect learned coefficients for each configured feature.

At a high level:

1. Load asset price frames from configured providers.
2. Inner-join all configured assets by common dates.
3. Compute configured source/FX relative-return features on each aligned current date.
4. Compute the target label from the target instrument's next aligned close divided by its current aligned close.
5. Fit a standardized linear regression model.
6. Persist model weights, scaler state, metadata, metrics, and training history.

## Return definitions

### Source return

For a source signal price and a source reference price observed at the same aligned time `t`, the source return feature is:

```text
x_source_t = source_signal_price_t / source_reference_price_t - 1
```

In YAML, this is represented by a feature with `kind: source_return`, `signal_asset`, and `reference_asset`.

### Target return

For a target instrument's current close and the next aligned target close, the supervised target is:

```text
y_target_t = target_next_close_t / target_current_close_t - 1
```

The framework labels the row at current date `t` with the next aligned row's target close, so each feature row is paired with a future target return only after the current row's observable features are computed.

## Configure EWY -> KOSPI

The repository includes `configs/examples/ewy_kospi.yaml`, which configures an EWY-to-KOSPI example using CSV-backed data.

```json
{
  "name": "ewy_kospi_example",
  "auth": {"provider": "csv"},
  "assets": [
    {"name": "source_etf", "symbol": "EWY", "provider": "csv", "csv_path": "data/raw/ewy.csv"},
    {"name": "source_reference", "symbol": "SPY", "provider": "csv", "csv_path": "data/raw/spy.csv"},
    {"name": "target_index", "symbol": "KOSPI", "provider": "target_csv", "csv_path": "data/raw/kospi.csv"},
    {"name": "fx_rate", "symbol": "USDKRW", "provider": "fx_csv", "csv_path": "data/raw/usdkrw.csv"},
    {"name": "fx_reference", "symbol": "USDKRW_REF", "provider": "fx_csv", "csv_path": "data/raw/usdkrw_ref.csv"}
  ],
  "features": [
    {"name": "source_relative_return", "signal_asset": "source_etf", "reference_asset": "source_reference", "kind": "source_return"},
    {"name": "fx_relative_return", "signal_asset": "fx_rate", "reference_asset": "fx_reference", "kind": "fx_return"}
  ],
  "target": {"asset": "target_index", "label": "target_next_return"},
  "model": {"epochs": 25, "batch_size": 16, "validation_fraction": 0.2, "learning_rate": 0.01}
}
```

What this means:

- `source_etf` is EWY, the source signal instrument.
- `source_reference` is SPY, used to make EWY a relative return signal.
- `target_index` is KOSPI, the target whose next aligned close creates `target_next_return`.
- `fx_rate` and `fx_reference` create an additional FX relative-return feature.
- `model` controls linear-model training hyperparameters.

The configured CSV files are expected to contain at least:

- a `date` column;
- a close column named `close` unless the asset overrides `price_column`.

## Configure another source/target pair using YAML only

You can reuse the framework for another market pair without changing Python code. Create a new YAML file under `configs/examples/` or your own config directory and change only the config values.

Example pattern:

```yaml
name: my_source_target_example
auth:
  provider: csv
assets:
  - name: source_signal
    symbol: SOURCE
    provider: csv
    csv_path: data/raw/source.csv
  - name: source_reference
    symbol: REF
    provider: csv
    csv_path: data/raw/source_reference.csv
  - name: target_market
    symbol: TARGET
    provider: target_csv
    csv_path: data/raw/target.csv
features:
  - name: source_relative_return
    signal_asset: source_signal
    reference_asset: source_reference
    kind: source_return
target:
  asset: target_market
  label: target_next_return
model:
  epochs: 50
  batch_size: 32
  validation_fraction: 0.2
  learning_rate: 0.01
```

Guidelines:

- Every `features[*].signal_asset`, `features[*].reference_asset`, and `target.asset` must match an `assets[*].name`.
- Use `kind: source_return` for a source/reference price ratio feature.
- Use `kind: fx_return` for an FX/reference price ratio feature.
- Add more feature entries when you want multiple explanatory variables.
- Point `csv_path` values at local CSVs with the expected date and close columns.
- If a file uses a different close column name, set `price_column` on that asset.

## Schwab authentication via environment variables

Schwab support is explicit in the configuration model, but the included Schwab price provider is a stub. The framework reads Schwab credentials from environment variables named by the auth config, so secrets do not need to be hard-coded in YAML.

`.env.example` documents the expected variable names:

```bash
SCHWAB_CLIENT_ID=
SCHWAB_CLIENT_SECRET=
SCHWAB_REDIRECT_URI=http://localhost:8182/callback
SCHWAB_TOKEN_PATH=.secrets/schwab_tokens.json
```

A Schwab-style auth config can reference those variable names:

```yaml
auth:
  provider: schwab
  client_id_env: SCHWAB_CLIENT_ID
  client_secret_env: SCHWAB_CLIENT_SECRET
  redirect_uri_env: SCHWAB_REDIRECT_URI
  token_path: .secrets/schwab_tokens.json
```

At runtime, `SchwabAuth.from_config(...)` resolves `client_id`, `client_secret`, and `redirect_uri` from the process environment using `os.getenv(...)`; `token_path` comes from config. Keep the token path under an ignored secrets directory such as `.secrets/`.

## Train

Train EWY -> KOSPI with:

```bash
python -m cross_market_regression.cli.train_model --config configs/examples/ewy_kospi.yaml --output-dir artifacts/models/ewy_kospi_linear
```

The requested base command is:

```bash
python -m cross_market_regression.cli.train_model --config configs/examples/ewy_kospi.yaml
```

The CLI requires `--output-dir`; the first command above writes artifacts to `artifacts/models/ewy_kospi_linear`.

## Predict

Predict from a saved model directory with feature rows in the same order recorded in `metadata.json`:

```bash
python -m cross_market_regression.cli.predict_live --model-dir artifacts/models/ewy_kospi_linear --features '[[0.0123, -0.0045]]'
```

The requested command pattern is:

```bash
python -m cross_market_regression.cli.predict_live --model-dir artifacts/models/ewy_kospi_linear ...
```

Notes:

- `--features` is a JSON list of feature rows.
- Each row must contain numeric values in the model's saved `feature_names` order.
- `--config` is accepted for workflow symmetry, but prediction uses model metadata for feature order.

## Inspect learned coefficients

Explain a saved model with:

```bash
python -m cross_market_regression.cli.explain_model --model-dir artifacts/models/ewy_kospi_linear
```

The output is JSON containing one object per feature plus the model bias, for example:

```json
[
  {"feature": "source_relative_return", "coefficient": 0.123},
  {"feature": "fx_relative_return", "coefficient": -0.045},
  {"feature": "bias", "coefficient": 0.001}
]
```

## Calendar alignment and lookahead avoidance

The default dataset builder performs an inner join across all configured asset frames by shared date. For each aligned current row:

1. Feature values are computed only from prices present on the current aligned date.
2. The target label is computed from the target asset's next aligned row.
3. The last aligned row is dropped because it has no next target close.

This avoids lookahead in feature construction because the source/reference/FX feature values for date `t` are calculated before the framework reads the target close from the next aligned date for the label. If markets have different holidays or closing times, prepare CSVs so each row's date represents the timestamp/convention you want the model to consider observable; the framework's alignment logic will then use only common aligned rows.

The data utilities also include `map_signal_to_target_calendar(...)`, which maps already-observed signal rows onto target dates by taking the most recent signal row with `signal.date <= target.date`. That rule is designed to avoid using a signal dated after the target date.

## Saved model artifacts

A training run saves these files in `--output-dir`:

- `model.weights.h5` — TensorFlow/Keras weights for the linear model.
- `scaler.json` — fitted feature standardization parameters.
- `metadata.json` — config name, ordered feature names, and target label.
- `metrics.json` — evaluation metrics computed on the validation split.
- `training_history.csv` — per-epoch training history from Keras.

Keep the entire output directory reproducible and disposable: commit the config and source code, not the generated model files.

## Files that must not be committed

Do not commit secrets, raw data, generated model artifacts, runtime outputs, caches, or logs. The repository `.gitignore` excludes common examples, including:

- `.env` and `.env.*` except `.env.example`;
- `.secrets/`;
- token files such as `*token*.json`, `*tokens*.json`, and `schwab_tokens.json`;
- raw/processed data directories such as `data/raw/` and `data/processed/`;
- generated artifacts and run outputs such as `artifacts/`, `runs/`, and `outputs/`;
- model files such as `*.weights.h5`, `*.keras`, `*.h5`, `scaler.json`, `metadata.json`, `metrics.json`, and `training_history.csv`;
- Python caches and test/tool caches such as `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and `.ruff_cache/`;
- notebook checkpoints, logs, and `*.log` files.
