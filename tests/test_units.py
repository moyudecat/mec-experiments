"""工程单位到SI单位转换测试。"""

import math

import pytest

from mec_dualbp.config.units import (
    bit_from_byte,
    bit_from_kibibyte_binary,
    bit_from_kilobyte_decimal,
    bit_from_mebibyte_binary,
    bit_from_megabyte_decimal,
    cycle_per_s_from_ghz,
    hz_from_khz,
    hz_from_mhz,
    linear_gain_from_db,
    watt_from_dbm,
)


def test_decimal_and_binary_data_units_are_distinct() -> None:
    assert bit_from_byte(1) == 8.0
    assert bit_from_kilobyte_decimal(1) == 8_000.0
    assert bit_from_kibibyte_binary(1) == 8_192.0
    assert bit_from_megabyte_decimal(1) == 8_000_000.0
    assert bit_from_mebibyte_binary(1) == 8_388_608.0


def test_frequency_conversions() -> None:
    assert hz_from_khz(2.5) == 2_500.0
    assert hz_from_mhz(2.5) == 2_500_000.0
    assert cycle_per_s_from_ghz(2.5) == 2_500_000_000.0


def test_dbm_and_db_conversions() -> None:
    assert watt_from_dbm(0.0) == pytest.approx(1e-3)
    assert watt_from_dbm(30.0) == pytest.approx(1.0)
    assert linear_gain_from_db(-10.0) == pytest.approx(0.1)
    assert linear_gain_from_db(3.0) == pytest.approx(10 ** 0.3)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_unit_conversions_reject_nonfinite_inputs(bad: float) -> None:
    with pytest.raises(ValueError):
        hz_from_mhz(bad)
    with pytest.raises(ValueError):
        watt_from_dbm(bad)


def test_nonnegative_engineering_quantities_reject_negative_values() -> None:
    with pytest.raises(ValueError):
        bit_from_byte(-1.0)
    with pytest.raises(ValueError):
        hz_from_khz(-1.0)


def test_unit_conversions_reject_boolean_values() -> None:
    with pytest.raises(TypeError):
        bit_from_byte(True)

