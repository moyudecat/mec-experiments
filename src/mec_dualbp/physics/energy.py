"""移动终端侧通信能耗点响应，不含基站、边缘或回传能耗。"""

from __future__ import annotations

from numbers import Real

from mec_dualbp._validation import finite_nonnegative_output, nonnegative_float, positive_float


def edge_comm_energy_point_j(
    uplink_terminal_tx_power_w: Real,
    input_size_bit: Real,
    uplink_rate_bit_per_s: Real,
    terminal_rx_power_w: Real,
    output_size_bit: Real,
    downlink_rate_bit_per_s: Real,
) -> float:
    """计算移动终端通信能耗，输出为J。

    ``E=P_u*D_in/R_u + P_r*D_out/R_d``。第四个参数必须是终端接收
    功率 ``P_r``；函数接口不接收基站下行发射功率 ``P_d``。
    """

    uplink_power = nonnegative_float(
        "uplink_terminal_tx_power_w", uplink_terminal_tx_power_w
    )
    input_size = nonnegative_float("input_size_bit", input_size_bit)
    uplink_rate = positive_float("uplink_rate_bit_per_s", uplink_rate_bit_per_s)
    receive_power = nonnegative_float("terminal_rx_power_w", terminal_rx_power_w)
    output_size = nonnegative_float("output_size_bit", output_size_bit)
    downlink_rate = positive_float("downlink_rate_bit_per_s", downlink_rate_bit_per_s)
    energy = uplink_power * input_size / uplink_rate + receive_power * output_size / downlink_rate
    return finite_nonnegative_output("edge_comm_energy_point_j", energy)

