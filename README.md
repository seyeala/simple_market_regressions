# Cross Market Regression

A reusable Python framework for building linear cross-market regressions. The package is named `cross_market_regression` and keeps market choices explicit in YAML/JSON configuration files rather than hard-coding countries, tickers, FX pairs, or calendars.

## Quick start

```bash
pip install -e .[dev]
cmr-train --config configs/examples/ewy_kospi.yaml --output-dir runs/ewy_kospi
cmr-predict-live --model-dir runs/ewy_kospi --config configs/examples/ewy_kospi.yaml
cmr-explain --model-dir runs/ewy_kospi
```

Example configs live in `configs/examples/`. The notebook is intentionally thin and delegates all work to package modules.
