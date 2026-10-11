"""上、下行Shannon速率点响应，所有输入均使用内部SI单位。"""

from __future__ import annotations

import math
from numbers import Real

from mec_dualbp._validation import nonnegative_float, positive_float


def _shannon_rate_bit_per_s(
    *,
    bandwidth_hz: Real,
    transmit_power_w: Real,
    channel_gain_linear: Real,
    noise_power_w: Real,
    interference_w: Real,
) -> float:
    """以 SI 输入计算通用 Shannon 速率；非法定义域和非正结果均拒绝。"""

    bandwidth = positive_float("bandwidth_hz", bandwidth_hz)
    power = nonnegative_float("transmit_power_w", transmit_power_w)
    gain = nonnegative_float("channel_gain_linear", channel_gain_linear)
    noise = nonnegative_float("noise_power_w", noise_power_w)
    interference = nonnegative_float("interference_w", interference_w)
    denominator = noise + interference
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("noise_power_w + interference_w must be finite and positive")
    signal_power = power * gain
    if not math.isfinite(signal_power):
        raise ArithmeticError("signal power must be finite")
    snr = signal_power / denominator
    rate = bandwidth * (math.log1p(snr) / math.log(2.0))
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("Shannon rate must be finite and strictly positive")
    return rate


def uplink_rate_bit_per_s(
    bandwidth_hz: Real,
    uplink_tx_power_w: Real,
    uplink_channel_gain_linear: Real,
    noise_power_w: Real,
    uplink_interference_w: Real,
) -> float:
    """计算上行Shannon速率 ``B*log2(1+P*h/(noise+I))``。

    输入单位依次为Hz、W、线性功率增益、W、W；输出为bit/s。
    带宽及最终速率必须严格为正，功率、增益、噪声和干扰必须非负。
    """

    return _shannon_rate_bit_per_s(
        bandwidth_hz=bandwidth_hz,
        transmit_power_w=uplink_tx_power_w,
        channel_gain_linear=uplink_channel_gain_linear,
        noise_power_w=noise_power_w,
        interference_w=uplink_interference_w,
    )


def downlink_rate_bit_per_s(
    bandwidth_hz: Real,
    downlink_bs_tx_power_w: Real,
    downlink_channel_gain_linear: Real,
    noise_power_w: Real,
    downlink_interference_w: Real,
) -> float:
    """计算下行Shannon速率，发射功率必须是基站 ``P_d``。

    输入单位依次为Hz、W、线性功率增益、W、W；输出为bit/s。
    本函数不接受或推断终端接收功率 ``P_r``。
    """

    return _shannon_rate_bit_per_s(
        bandwidth_hz=bandwidth_hz,
        transmit_power_w=downlink_bs_tx_power_w,
        channel_gain_linear=downlink_channel_gain_linear,
        noise_power_w=noise_power_w,
        interference_w=downlink_interference_w,
    )
