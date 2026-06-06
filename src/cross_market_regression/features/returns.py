"""Return math for source and FX signals."""

from __future__ import annotations


def ratio_return(signal_price: float, reference_price: float) -> float:
    if reference_price == 0:
        raise ZeroDivisionError("reference_price cannot be zero")
    return signal_price / reference_price - 1.0


def source_return(signal_price: float, reference_price: float) -> float:
    return ratio_return(signal_price, reference_price)


def fx_return(signal_price: float, reference_price: float) -> float:
    return ratio_return(signal_price, reference_price)


def target_next_return(target_next_close: float, target_current_close: float) -> float:
    return ratio_return(target_next_close, target_current_close)
