from __future__ import annotations

import pytest


def paid_prefix(asked, remaining):
    return asked[:remaining]


@pytest.mark.parametrize(("remaining", "batch", "expected"), [
    (1, 10, 1), (2, 10, 2), (10, 10, 10), (12, 10, 10),
    (1, 50, 1), (49, 50, 49), (100, 100, 100),
])
def test_exact_budget_tail_uses_native_ask_prefix(remaining, batch, expected) -> None:
    asked = list(range(batch))
    selected = paid_prefix(asked, remaining)
    assert selected == asked[:expected]
    assert len(selected) == min(remaining, batch)


@pytest.mark.parametrize(("population", "budget"), [(10, 101), (50, 1999), (100, 20000)])
def test_repeated_prefix_accounting_ends_exactly_at_budget(population, budget) -> None:
    requests = 0
    while requests < budget:
        asked = list(range(population))
        paid = paid_prefix(asked, budget - requests)
        assert paid == asked[:len(paid)]
        requests += len(paid)
    assert requests == budget
