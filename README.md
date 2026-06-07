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

## Schwab authentication

Schwab credentials must come from environment variables only.  Use `.env.example` as the template:

```text
SCHWAB_API_KEY=
SCHWAB_APP_SECRET=
SCHWAB_CALLBACK_URL=http://localhost:8182/callback
SCHWAB_TOKEN_PATH=artifacts/raw/schwab/token.json
```

Do not commit real `.env` files or token JSON files.

## Train

```bash
python -m cross_market_regression.cli.train_model \
  --config configs/examples/ewy_kospi.yaml
```

The CLI defaults to the configured `model.model_dir`; `--output-dir` can override it for the canonical provider-driven training flow.

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
