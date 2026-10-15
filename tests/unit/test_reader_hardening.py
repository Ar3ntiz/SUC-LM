"""Python SUC1 reader directory checks (audit observation "reader hardening"; the C++ reader's checked arithmetic is
covered by cpp/tests/test_main.cpp)."""
import pytest

from suc.io.suc1 import SucFormatError, _check_payload_records


def _dir(**arr):
    a = {"key": "weight", "dtype": "f32", "shape": [2, 3], "offset": 128, "nbytes": 24, **arr}
    return {"codebooks": [], "tensors": [{"name": "w", "arrays": [a]}]}


def test_valid_record_passes():
    _check_payload_records("f", _dir(), 128, 192)


@pytest.mark.parametrize("bad", [
    {"nbytes": 28}, {"dtype": "f99"}, {"offset": 130}, {"offset": 64}, {"offset": 192},
    {"shape": [1676976733973595602, 11]},          # wraps to 6 elements modulo 2^64 in unchecked C arithmetic
    {"shape": [-2, -3]}, {"shape": [2.0, 3]}])
def test_malformed_records_fail(bad):
    with pytest.raises(SucFormatError):
        _check_payload_records("f", _dir(**bad), 128, 192)


def test_duplicate_names_and_missing_fields_fail():
    d = _dir()
    d["tensors"].append(dict(d["tensors"][0]))
    with pytest.raises(SucFormatError, match="duplicate"):
        _check_payload_records("f", d, 128, 192)
    d = _dir()
    del d["tensors"][0]["arrays"][0]["nbytes"]
    with pytest.raises(SucFormatError, match="malformed"):
        _check_payload_records("f", d, 128, 192)
