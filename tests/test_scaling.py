from cross_market_regression.features.scaling import StandardScaler


def test_train_only_standardization_and_json_roundtrip(tmp_path):
    scaler = StandardScaler.fit([[1.0, 10.0], [3.0, 10.0]])
    assert scaler.transform([[2.0, 10.0]]) == [[0.0, 0.0]]
    path = tmp_path / "scaler.json"
    scaler.save(path)
    assert StandardScaler.load(path).transform([[2.0, 10.0]]) == [[0.0, 0.0]]
