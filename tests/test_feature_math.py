import pytest

from cross_market_regression.features.returns import fx_return, source_return, target_next_return
from cross_market_regression.features.scaling import StandardScaler


def test_ratio_return_formulas_for_source_fx_and_target():
    assert source_return(110.0, 100.0) == pytest.approx(0.10)
    assert fx_return(99.0, 100.0) == pytest.approx(-0.01)
    assert target_next_return(105.0, 100.0) == pytest.approx(0.05)


def test_ratio_return_rejects_zero_reference_price():
    with pytest.raises(ZeroDivisionError):
        source_return(1.0, 0.0)


def test_standard_scaler_fits_only_synthetic_training_rows(tmp_path):
    scaler = StandardScaler.fit([[1.0, 10.0], [3.0, 10.0]])

    assert scaler.transform([[2.0, 10.0]]) == [[0.0, 0.0]]
    path = tmp_path / "scaler.json"
    scaler.save(path)
    assert StandardScaler.load(path).transform([[2.0, 10.0]]) == [[0.0, 0.0]]


def test_standard_scaler_1d_identity_can_be_saved_for_unstandardized_models(tmp_path):
    from cross_market_regression.features.scalers import StandardScaler1D

    path = tmp_path / "scaler.json"
    StandardScaler1D(["x"]).fit_identity().save(path)
    loaded = StandardScaler1D.load(path)

    assert loaded.feature_names == ["x"]
    assert loaded.means == {"x": 0.0}
    assert loaded.stds == {"x": 1.0}
