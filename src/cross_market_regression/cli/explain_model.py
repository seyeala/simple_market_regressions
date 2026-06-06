from __future__ import annotations

import argparse
import json

from cross_market_regression.modeling.explain import explain_coefficients


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Explain a persisted cross-market regression model")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(explain_coefficients(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
