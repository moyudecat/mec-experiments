"""内层 GA 测试共享的合成 SI 参数。"""

import pytest

from mec_dualbp.inner_ga.tasks import EdgeTaskParameters
from mec_dualbp.uncertainty.interval_params import Interval


@pytest.fixture
def edge_parameters() -> EdgeTaskParameters:
    """返回不依赖论文正式参数的开发级合成用户。"""

    return EdgeTaskParameters(
        input_size_bit=Interval(1e6, 3e6),
        output_size_bit=Interval(0.5e6, 1.5e6),
        cpu_cycles=Interval(1e9, 3e9),
        uplink_interference_w=Interval(1e-10, 3e-10),
        downlink_interference_w=Interval(1e-10, 4e-10),
        uplink_bandwidth_hz=1e6,
        downlink_bandwidth_hz=2e6,
        uplink_terminal_tx_power_w=0.2,
        downlink_bs_tx_power_w=1.0,
        terminal_rx_power_w=0.08,
        uplink_channel_gain_linear=1e-6,
        downlink_channel_gain_linear=2e-6,
        noise_power_w=1e-10,
        edge_cpu_frequency_cycle_per_s=5e9,
    )

