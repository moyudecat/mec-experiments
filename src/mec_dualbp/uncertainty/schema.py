"""六维用户扰动的唯一机器可读顺序与任务有效维度。"""

from __future__ import annotations

from enum import IntEnum


class DisturbanceIndex(IntEnum):
    """冻结的六维扰动索引。"""

    D_IN = 0
    D_OUT = 1
    C = 2
    I_U = 3
    I_D = 4
    F_L = 5


DISTURBANCE_NAMES: tuple[str, ...] = (
    "D_in",
    "D_out",
    "C",
    "I_u",
    "I_d",
    "F_L",
)
DISTURBANCE_DIMENSION = len(DISTURBANCE_NAMES)
EDGE_CORE_ACTIVE_MASK: tuple[bool, ...] = (True, True, True, True, True, False)
EDGE_COMM_ENERGY_ACTIVE_MASK: tuple[bool, ...] = (
    True,
    True,
    False,
    True,
    True,
    False,
)

