"""内层边界问题的不可变接口。"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mec_dualbp.uncertainty.budget_set import (
    is_user_budget_feasible,
    proportional_budget_repair,
    validate_gamma,
    validate_user_disturbance,
)
from mec_dualbp.uncertainty.schema import DISTURBANCE_DIMENSION

Direction = Literal["minimize", "maximize"]


@dataclass(frozen=True, slots=True)
class InnerGAProblem:
    """一个确定性真实物理标量边界任务。"""

    name: str
    active_mask: tuple[bool, ...]
    gamma: float
    direction: Direction
    evaluator: Callable[[NDArray[np.float64]], float]
    dimension: int = DISTURBANCE_DIMENSION

    def __post_init__(self) -> None:
        """验证名称、六维掩码、预算、方向和评价器。"""

        if not self.name.strip():
            raise ValueError("problem name must not be empty")
        if self.dimension != DISTURBANCE_DIMENSION:
            raise ValueError("inner GA problem dimension must be 6")
        if len(self.active_mask) != self.dimension or not all(
            isinstance(value, bool) for value in self.active_mask
        ):
            raise ValueError("active_mask must contain exactly six booleans")
        if not any(self.active_mask):
            raise ValueError("at least one dimension must be active")
        object.__setattr__(self, "gamma", validate_gamma(self.gamma))
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("direction must be minimize or maximize")
        if not callable(self.evaluator):
            raise TypeError("evaluator must be callable")

    def repair(self, z: ArrayLike) -> NDArray[np.float64]:
        """按本问题掩码执行无量纲比例预算修复，不修改输入。"""

        return proportional_budget_repair(z, self.gamma, self.active_mask)

    def validate(self, z: Sequence[float] | NDArray[np.float64]) -> tuple[float, ...]:
        """严格验证六维可行性及无效维度为零。"""

        values = validate_user_disturbance(z, self.gamma)
        if any((not active) and value != 0.0 for active, value in zip(self.active_mask, values)):
            raise ValueError("inactive disturbance dimensions must equal zero")
        return values

    def is_feasible(self, z: Sequence[float] | NDArray[np.float64]) -> bool:
        """返回扰动是否满足预算、分量边界和本问题有效维度规则。"""

        try:
            values = tuple(float(value) for value in z)
            return is_user_budget_feasible(values, self.gamma) and all(
                active or value == 0.0 for active, value in zip(self.active_mask, values)
            )
        except (TypeError, ValueError):
            return False

    def evaluate(self, z: Sequence[float] | NDArray[np.float64]) -> float:
        """评价一个已可行扰动并返回有限非负真实物理响应。"""

        values = np.asarray(self.validate(z), dtype=np.float64)
        values.setflags(write=False)
        response = float(self.evaluator(values))
        if not math.isfinite(response) or response < 0.0:
            raise ArithmeticError("problem evaluator must return a finite nonnegative response")
        return response

    def fitness(self, response_value: float) -> float:
        """将真实响应转为统一最大化适应度，同时不改变响应语义。"""

        if not math.isfinite(response_value) or response_value < 0.0:
            raise ValueError("response_value must be finite and nonnegative")
        return response_value if self.direction == "maximize" else -response_value

