import numpy as np
import pytest

from mec_dualbp.experiments.exp1_formal_protocol import generate_formal_samples
from mec_dualbp.inner_ga.tasks import (
    build_edge_comm_energy_lower_problem, build_edge_comm_energy_upper_problem,
    build_edge_core_latency_lower_problem, build_edge_core_latency_upper_problem,
)
from mec_dualbp.physics.batch import evaluate_edge_comm_energy_batch_si, evaluate_edge_core_latency_batch_si
from mec_dualbp.uncertainty.sampling import lhs_user_disturbances


@pytest.fixture(scope="module")
def batch_case():
    sample = generate_formal_samples()[0]
    return sample, lhs_user_disturbances(1000, sample.gamma, (True,) * 6, 912345, True, True).initialization_samples


@pytest.mark.parametrize("builder,batch", [
    (build_edge_core_latency_lower_problem, evaluate_edge_core_latency_batch_si),
    (build_edge_core_latency_upper_problem, evaluate_edge_core_latency_batch_si),
    (build_edge_comm_energy_lower_problem, evaluate_edge_comm_energy_batch_si),
    (build_edge_comm_energy_upper_problem, evaluate_edge_comm_energy_batch_si),
])
def test_scalar_batch_1000_candidate_consistency(batch_case, builder, batch) -> None:
    sample, raw = batch_case
    problem = builder(sample.parameters, sample.gamma)
    z = np.asarray([problem.repair(row) for row in raw])
    scalar = np.asarray([problem.evaluate(row) for row in z])
    vector = batch(z, sample.parameters, chunk_size=137)
    absolute = np.max(np.abs(scalar - vector))
    relative = np.max(np.abs(scalar - vector) / np.maximum(np.abs(scalar), 1e-300))
    assert absolute <= 1e-12 and relative <= 1e-10
    assert np.min(scalar) == pytest.approx(np.min(vector), abs=1e-12)
    assert np.max(scalar) == pytest.approx(np.max(vector), abs=1e-12)


def test_batch_chunk_size_and_repeat_are_exact(batch_case) -> None:
    sample, z = batch_case
    first = evaluate_edge_core_latency_batch_si(z, sample.parameters, chunk_size=1)
    second = evaluate_edge_core_latency_batch_si(z, sample.parameters, chunk_size=256)
    third = evaluate_edge_core_latency_batch_si(z, sample.parameters)
    assert np.array_equal(first, second) and np.array_equal(second, third)


def test_batch_does_not_modify_input(batch_case) -> None:
    sample, z = batch_case
    before = z.copy()
    evaluate_edge_comm_energy_batch_si(z, sample.parameters, chunk_size=31)
    assert np.array_equal(z, before)


def test_energy_ignores_c_and_fl_and_uses_terminal_receive_power(batch_case) -> None:
    sample, z = batch_case
    changed = z.copy(); changed[:, 2] *= -1; changed[:, 5] *= -1
    base = evaluate_edge_comm_energy_batch_si(z, sample.parameters)
    assert np.array_equal(base, evaluate_edge_comm_energy_batch_si(changed, sample.parameters))
    from dataclasses import replace
    altered = replace(sample.parameters, terminal_rx_power_w=sample.parameters.terminal_rx_power_w * 2)
    assert np.any(evaluate_edge_comm_energy_batch_si(z, altered) != base)


def test_edge_core_ignores_fl_and_has_no_cache_argument(batch_case) -> None:
    sample, z = batch_case
    changed = z.copy(); changed[:, 5] *= -1
    assert np.array_equal(evaluate_edge_core_latency_batch_si(z, sample.parameters), evaluate_edge_core_latency_batch_si(changed, sample.parameters))


@pytest.mark.parametrize("bad", [np.ones(6), np.full((2, 6), np.nan), np.full((2, 6), 2.0)])
def test_batch_rejects_invalid_disturbances(batch_case, bad) -> None:
    with pytest.raises((ValueError, ArithmeticError)):
        evaluate_edge_core_latency_batch_si(bad, batch_case[0].parameters)

