from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from cross_market_regression.modeling.predict import predict_rows, predict_target_from_inputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict from a persisted cross-market regression model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config", help="Accepted for workflow symmetry; model metadata drives feature order")
    parser.add_argument("--features", required=False, help="JSON list of raw feature rows for batch prediction")
    parser.add_argument("--target-current-close", type=float)
    parser.add_argument("--source-signal-price", type=float)
    parser.add_argument("--source-reference-price", type=float)
    parser.add_argument("--fx-signal", type=float)
    parser.add_argument("--fx-reference", type=float)
    args = parser.parse_args(argv)
    if args.features is not None:
        print(json.dumps({"predictions": predict_rows(args.model_dir, json.loads(args.features))}))
        return
    if args.target_current_close is None:
        raise SystemExit("--target-current-close is required unless --features is supplied")
    result = predict_target_from_inputs(
        model_dir=args.model_dir,
        target_current_close=args.target_current_close,
        source_signal_price=args.source_signal_price,
        source_reference_price=args.source_reference_price,
        fx_signal=args.fx_signal,
        fx_reference=args.fx_reference,
    )
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
