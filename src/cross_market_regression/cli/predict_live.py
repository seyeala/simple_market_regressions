from __future__ import annotations

import argparse
import json

from cross_market_regression.modeling.predict import predict_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict from a persisted cross-market regression model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config", help="Accepted for workflow symmetry; model metadata drives feature order")
    parser.add_argument("--features", required=False, default="[]", help="JSON list of feature rows")
    args = parser.parse_args(argv)
    rows = json.loads(args.features)
    print(json.dumps({"predictions": predict_rows(args.model_dir, rows)}))


if __name__ == "__main__":
    main()
