"""工程单位到内部SI单位的无歧义纯函数。

数据量统一为bit，带宽为Hz，速率为bit/s，计算量为cycle，
计算频率为cycle/s，功率为W，时延为s，能耗为J。
"""

from __future__ import annotations

import math
from numbers import Real

from mec_dualbp._validation import finite_float, finite_nonnegative_output, nonnegative_float


def _scaled_nonnegative(name: str, value: Real, factor: float) -> float:
    """按给定正比例因子换算有限非负工程量，并验证换算结果。"""

    source = nonnegative_float(name, value)
    return finite_nonnegative_output(f"converted {name}", source * factor)


def bit_from_byte(value: Real) -> float:
    """将非负byte转换为bit（×8）。"""
    return _scaled_nonnegative("byte", value, 8.0)


def bit_from_kilobit_decimal(value: Real) -> float:
    """将十进制kbit转换为bit（×10^3）。"""
    return _scaled_nonnegative("kilobit_decimal", value, 1e3)


def bit_from_megabit_decimal(value: Real) -> float:
    """将十进制Mbit转换为bit（×10^6）。"""
    return _scaled_nonnegative("megabit_decimal", value, 1e6)


def bit_from_gigabit_decimal(value: Real) -> float:
    """将十进制Gbit转换为bit（×10^9）。"""
    return _scaled_nonnegative("gigabit_decimal", value, 1e9)


def bit_from_kilobyte_decimal(value: Real) -> float:
    """将十进制kB转换为bit（×8×10^3）。"""
    return _scaled_nonnegative("kilobyte_decimal", value, 8e3)


def bit_from_megabyte_decimal(value: Real) -> float:
    """将十进制MB转换为bit（×8×10^6）。"""
    return _scaled_nonnegative("megabyte_decimal", value, 8e6)


def bit_from_gigabyte_decimal(value: Real) -> float:
    """将十进制GB转换为bit（×8×10^9）。"""
    return _scaled_nonnegative("gigabyte_decimal", value, 8e9)


def bit_from_kibibyte_binary(value: Real) -> float:
    """将二进制KiB转换为bit（×8×2^10）。"""
    return _scaled_nonnegative("kibibyte_binary", value, 8.0 * 2**10)


def bit_from_mebibyte_binary(value: Real) -> float:
    """将二进制MiB转换为bit（×8×2^20）。"""
    return _scaled_nonnegative("mebibyte_binary", value, 8.0 * 2**20)


def bit_from_gibibyte_binary(value: Real) -> float:
    """将二进制GiB转换为bit（×8×2^30）。"""
    return _scaled_nonnegative("gibibyte_binary", value, 8.0 * 2**30)


def hz_from_khz(value: Real) -> float:
    """将非负kHz转换为Hz。"""
    return _scaled_nonnegative("khz", value, 1e3)


def hz_from_mhz(value: Real) -> float:
    """将非负MHz转换为Hz。"""
    return _scaled_nonnegative("mhz", value, 1e6)


def hz_from_ghz(value: Real) -> float:
    """将非负GHz转换为Hz。"""
    return _scaled_nonnegative("ghz", value, 1e9)


def bit_per_s_from_kilobit_per_s(value: Real) -> float:
    """将非负kbit/s转换为bit/s。"""
    return _scaled_nonnegative("kilobit_per_s", value, 1e3)


def bit_per_s_from_megabit_per_s(value: Real) -> float:
    """将非负Mbit/s转换为bit/s。"""
    return _scaled_nonnegative("megabit_per_s", value, 1e6)


def bit_per_s_from_gigabit_per_s(value: Real) -> float:
    """将非负Gbit/s转换为bit/s。"""
    return _scaled_nonnegative("gigabit_per_s", value, 1e9)


def cycle_from_kilocycle(value: Real) -> float:
    """将非负kcycle转换为cycle。"""
    return _scaled_nonnegative("kilocycle", value, 1e3)


def cycle_from_megacycle(value: Real) -> float:
    """将非负Mcycle转换为cycle。"""
    return _scaled_nonnegative("megacycle", value, 1e6)


def cycle_from_gigacycle(value: Real) -> float:
    """将非负Gcycle转换为cycle。"""
    return _scaled_nonnegative("gigacycle", value, 1e9)


def cycle_per_s_from_khz(value: Real) -> float:
    """按一Hz对应一cycle/s的冻结口径，将kHz转换为cycle/s。"""
    return _scaled_nonnegative("cpu_khz", value, 1e3)


def cycle_per_s_from_mhz(value: Real) -> float:
    """按一Hz对应一cycle/s的冻结口径，将MHz转换为cycle/s。"""
    return _scaled_nonnegative("cpu_mhz", value, 1e6)


def cycle_per_s_from_ghz(value: Real) -> float:
    """按一Hz对应一cycle/s的冻结口径，将GHz转换为cycle/s。"""
    return _scaled_nonnegative("cpu_ghz", value, 1e9)


def watt_from_milliwatt(value: Real) -> float:
    """将非负mW转换为W。"""
    return _scaled_nonnegative("milliwatt", value, 1e-3)


def watt_from_microwatt(value: Real) -> float:
    """将非负µW转换为W。"""
    return _scaled_nonnegative("microwatt", value, 1e-6)


def watt_from_nanowatt(value: Real) -> float:
    """将非负nW转换为W。"""
    return _scaled_nonnegative("nanowatt", value, 1e-9)


def watt_from_dbm(value: Real) -> float:
    """将有限dBm转换为W：10**((dBm-30)/10)。"""
    dbm = finite_float("dbm", value)
    result = 10.0 ** ((dbm - 30.0) / 10.0)
    if not math.isfinite(result) or result <= 0.0:
        raise ArithmeticError("converted dBm power must be finite and positive")
    return result


def linear_gain_from_db(value: Real) -> float:
    """将有限功率增益dB转换为线性值：10**(dB/10)。"""
    gain_db = finite_float("power_gain_db", value)
    result = 10.0 ** (gain_db / 10.0)
    if not math.isfinite(result) or result <= 0.0:
        raise ArithmeticError("converted linear power gain must be finite and positive")
    return result


def second_from_millisecond(value: Real) -> float:
    """将非负ms转换为s。"""
    return _scaled_nonnegative("millisecond", value, 1e-3)


def second_from_microsecond(value: Real) -> float:
    """将非负µs转换为s。"""
    return _scaled_nonnegative("microsecond", value, 1e-6)


def second_from_nanosecond(value: Real) -> float:
    """将非负ns转换为s。"""
    return _scaled_nonnegative("nanosecond", value, 1e-9)


def joule_from_millijoule(value: Real) -> float:
    """将非负mJ转换为J。"""
    return _scaled_nonnegative("millijoule", value, 1e-3)


def joule_from_microjoule(value: Real) -> float:
    """将非负µJ转换为J。"""
    return _scaled_nonnegative("microjoule", value, 1e-6)


def joule_from_nanojoule(value: Real) -> float:
    """将非负nJ转换为J。"""
    return _scaled_nonnegative("nanojoule", value, 1e-9)
