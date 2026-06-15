"""Model training, persistence, diagnostics, and explanations."""

from __future__ import annotations

__all__ = [
    "FixedMultiWindowPolicyConfig",
    "FixedMultiWindowUtilityPolicy",
    "OHLCVFeatureContract",
    "build_fixed_multi_window_policy",
]


def __getattr__(name: str):
    """Lazily expose TensorFlow-backed policy helpers when requested."""

    if name in __all__:
        from cross_market_regression.modeling import fixed_multi_window_policy

        return getattr(fixed_multi_window_policy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
