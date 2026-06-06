from pathlib import Path

from cross_market_regression.config import load_config


def test_reusable_source_code_has_no_market_specific_constants():
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/cross_market_regression").rglob("*.py"))
    forbidden = ["EWY", "KOSPI", "Korea", "USDKRW"]
    assert not any(token in source for token in forbidden)


def test_multiple_example_configs_load_with_same_schema():
    names = [load_config(str(path)).name for path in Path("configs/examples").glob("*.yaml")]
    assert {"ewy_kospi_example", "qqq_ndx_example", "ewj_nikkei_example", "ewt_taiwan_example", "soxx_semis_example"}.issubset(names)
