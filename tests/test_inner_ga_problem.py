"""内层 GA 问题接口测试。"""

import numpy as np
import pytest

from mec_dualbp.inner_ga.problem import InnerGAProblem


def _problem(direction: str = "maximize") -> InnerGAProblem:
    return InnerGAProblem(
        "quadratic",
        (True, True, False, False, False, False),
        1.0,
        direction,  # type: ignore[arg-type]
        lambda z: float(z[0] ** 2 + 2.0 * z[1] ** 2),
    )


def test_problem_repair_validate_and_input_immutability() -> None:
    problem = _problem()
    source = np.array([2.0, -2.0, 9.0, 0.0, 0.0, 0.0])
    original = source.copy()
    repaired = problem.repair(source)
    assert problem.is_feasible(repaired)
    assert repaired[2] == 0.0
    assert np.array_equal(source, original)
    assert problem.validate(repaired) == tuple(repaired)


def test_response_and_fitness_semantics() -> None:
    z = np.array([0.5, -0.5, 0, 0, 0, 0], dtype=float)
    assert _problem("maximize").fitness(_problem().evaluate(z)) == pytest.approx(0.75)
    assert _problem("minimize").fitness(_problem().evaluate(z)) == pytest.approx(-0.75)


def test_evaluate_is_deterministic_and_rejects_infeasible() -> None:
    problem = _problem()
    z = np.array([0.2, 0.3, 0, 0, 0, 0], dtype=float)
    assert problem.evaluate(z) == problem.evaluate(z)
    with pytest.raises(ValueError):
        problem.evaluate([1, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        problem.evaluate([0, 0, 1, 0, 0, 0])


def test_nonfinite_or_negative_response_is_rejected() -> None:
    for evaluator in (lambda z: float("nan"), lambda z: -1.0):
        problem = InnerGAProblem("bad", (True,) * 6, 1.0, "maximize", evaluator)
        with pytest.raises(ArithmeticError):
            problem.evaluate(np.zeros(6))
