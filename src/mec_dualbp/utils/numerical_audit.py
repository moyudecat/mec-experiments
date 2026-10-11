"""上下文隔离的区间端点吸附统计。"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EndpointSnapAudit:
    """记录端点吸附次数与最大ULP距离。"""

    count: int = 0
    max_ulp: float = 0.0


_ENDPOINT_AUDIT: ContextVar[EndpointSnapAudit] = ContextVar(
    "mec_dualbp_endpoint_snap_audit", default=EndpointSnapAudit()
)


def reset_endpoint_snap_audit() -> None:
    """将当前执行上下文的端点吸附统计清零。"""

    _ENDPOINT_AUDIT.set(EndpointSnapAudit())


def record_endpoint_snap(distance_ulp: float) -> None:
    """记录一次不超过16 ULP的吸附，超限时拒绝。"""

    if distance_ulp < 0.0 or distance_ulp > 16.0:
        raise ArithmeticError("endpoint snap distance must lie in [0, 16] ULP")
    current = _ENDPOINT_AUDIT.get()
    _ENDPOINT_AUDIT.set(
        EndpointSnapAudit(current.count + 1, max(current.max_ulp, distance_ulp))
    )


def get_endpoint_snap_audit() -> EndpointSnapAudit:
    """返回当前上下文的不可变端点吸附统计。"""

    return _ENDPOINT_AUDIT.get()

