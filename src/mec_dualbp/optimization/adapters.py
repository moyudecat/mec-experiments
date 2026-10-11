"""Exp-7 五算法统一 ask/tell adapter。

本模块只提供算法实现与共同接口，不包含场景数据，也不会自行启动实验。
NSGA-II、MOEA/D、RVEA绑定本地 pymoo；MOPSO与Random在此实现。所有候选
必须经同一 ``canonical_outer_repair``，并且只能由 ``CanonicalOuterEvaluator``
评价。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import APDSurvival, RVEA, calc_V, calc_gamma
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.survival.rank_and_crowding.classes import RankAndCrowding
from pymoo.util.ref_dirs import get_reference_directions

from .evaluation_archive import canonical_decision_hash
from .metrics import HistoryRecord, nondominated_min
from .outer_contract import (
    CanonicalOuterEvaluator, CanonicalSolution, RepairContext, canonical_outer_repair,
)

FORMAL_ALGORITHMS = (
    "CorrectedDualBP-CNSGA-II", "MOEA/D", "RVEA", "MOPSO", "Random",
)
V2_ALGORITHM = "CorrectedDualBP-CA-CNSGA-II"
OE_RVEA_ALGORITHM = "OE-RVEA"
ALGORITHM_REGISTRY = FORMAL_ALGORITHMS + (V2_ALGORITHM, OE_RVEA_ALGORITHM)
V2_DESIGN_HASH = "68f6f6b366f1e1a1f6d7a4d629062e850cad472ad139e542fdf94d5dabdb7197"
OE_RVEA_V1_DESIGN_HASH = "6e9795b539095f841d51bb45db47be1535d66d3c530d81f96ed91ea1503428d4"
RVEA_4D_FINITE_SPAN_DESIGN_HASH = "af83d9b0cdf436c72f894e2e13809d73fb03eaa460b234fbafd8b686d616ac33"
CA_ELITE_FRACTION = 0.10
CA_EPSILON = 1e-12


@dataclass(frozen=True)
class OuterAlgorithmConfig:
    population_size: int = 100
    evaluation_budget: int = 20_000
    workers: int = 1
    objective_count: int = 2

    def __post_init__(self) -> None:
        if (self.population_size <= 1 or self.evaluation_budget < self.population_size
                or self.workers != 1 or self.objective_count < 2):
            raise ValueError("positive serial outer budget required")


def flatten_solution(solution: CanonicalSolution) -> np.ndarray:
    return np.concatenate((solution.offload, solution.cache, solution.edge_cpu)).astype(np.float64)


def canonicalize_vector(vector: Sequence[float], users: int) -> CanonicalSolution:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3 * users,) or not np.isfinite(value).all():
        raise ValueError("raw outer vector must have 3N finite values")
    return CanonicalSolution(value[:users], value[users:2 * users], value[2 * users:])


def matched_canonical_initial_population(seed: int, size: int, context: RepairContext) -> np.ndarray:
    """同一outer seed生成可供全部算法共享的规范初始池。"""
    if size <= 1:
        raise ValueError("initial population must contain at least two candidates")
    rng = np.random.default_rng(int(seed))
    n = context.cache_sizes.size
    # One analytically defined feasible anchor is shared by every algorithm and
    # occupies the same first position. It is evaluated normally and consumes
    # one request from the common budget.
    anchor = CanonicalSolution(np.zeros(n), np.zeros(n), np.zeros(n))
    rows = [flatten_solution(canonical_outer_repair(anchor, context))]
    for _ in range(size - 1):
        raw = CanonicalSolution(rng.integers(0, 2, n), rng.integers(0, 2, n),
                                rng.uniform(context.f_min, context.f_max, n))
        rows.append(flatten_solution(canonical_outer_repair(raw, context)))
    return np.asarray(rows, dtype=np.float64)


def _result_arrays(results: Sequence[Mapping[str, Any]], expected: int,
                   objective_count: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if len(results) != expected:
        raise ValueError("one result per pending candidate required")
    f = np.asarray([row["objectives"] for row in results], dtype=np.float64)
    cv = np.asarray([row["constraint_violation"] for row in results], dtype=np.float64)
    if (f.shape != (expected, objective_count) or cv.shape != (expected,)
            or not np.isfinite(f).all() or np.any(cv < 0)):
        raise ValueError("finite configured-objective results and nonnegative CV required")
    if any(row.get("prediction_mode") != "corr_pred" for row in results):
        raise ValueError("all Formal adapter results must use corr_pred")
    return f, cv


def _deb_better(f_new: np.ndarray, cv_new: float, f_old: np.ndarray, cv_old: float) -> bool:
    if cv_new <= 0 < cv_old:
        return True
    if cv_old <= 0 < cv_new:
        return False
    if cv_new > 0 or cv_old > 0:
        return cv_new < cv_old
    return bool(np.all(f_new <= f_old) and np.any(f_new < f_old))


def _crowding(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 2:
        return np.full(len(values), np.inf)
    distance = np.zeros(len(values), dtype=np.float64)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="stable")
        distance[order[[0, -1]]] = np.inf
        span = values[order[-1], column] - values[order[0], column]
        if span > 0:
            distance[order[1:-1]] += ((values[order[2:], column] - values[order[:-2], column]) / span)
    return distance


class ConvergenceAwareSurvival(RankAndCrowding):
    """Frozen bounded CA elite preservation layered over NSGA-II survival."""

    design_hash = V2_DESIGN_HASH
    elite_fraction = CA_ELITE_FRACTION
    epsilon = CA_EPSILON

    def __init__(self, population_size: int | None = None) -> None:
        super().__init__()
        self.population_size = int(population_size) if population_size is not None else None
        self.audit_records: list[dict[str, Any]] = []

    @staticmethod
    def elite_count(population_size: int) -> int:
        return max(1, int(np.floor(CA_ELITE_FRACTION * int(population_size))))

    @staticmethod
    def _hash(individual: Individual) -> str:
        vector = np.asarray(individual.get("X"), dtype=np.float64).reshape(-1)
        return canonical_decision_hash(tuple(float(value) for value in vector))

    def _do(self, problem: Problem, pop: Any, *args: Any, random_state: Any = None,
            n_survive: int | None = None, **kwargs: Any) -> Any:
        if n_survive is None:
            n_survive = len(pop)
        # This is the exact V1 RankAndCrowding result and ordering.  The base
        # Survival class has already filtered infeasible candidates, so pop is
        # precisely the current feasible subset of R=P_t union Q_t.
        standard = super()._do(
            problem, pop, *args, random_state=random_state, n_survive=n_survive, **kwargs)
        if len(pop) == 0:
            return standard

        objectives = np.asarray(pop.get("F"), dtype=np.float64)
        fronts = self.nds.do(objectives)
        if not fronts or len(fronts[0]) == 0:
            return standard
        first = np.asarray(fronts[0], dtype=int)
        f1 = pop[first]

        lower, upper = objectives.min(axis=0), objectives.max(axis=0)
        denominator = np.maximum(upper - lower, self.epsilon)
        normalized = (np.asarray(f1.get("F"), dtype=np.float64) - lower) / denominator
        distance = np.linalg.norm(normalized, axis=1)
        crowding = np.asarray(f1.get("crowding"), dtype=np.float64).reshape(-1)
        worst_component = normalized.max(axis=1)
        hashes = np.asarray([self._hash(individual) for individual in f1], dtype=object)

        order = sorted(range(len(f1)), key=lambda index: (
            float(distance[index]), -float(crowding[index]),
            float(worst_component[index]), str(hashes[index])))
        target = self.elite_count(self.population_size if self.population_size is not None else n_survive)
        convergence = order[:min(target, len(order))]

        def extreme_index(column: int) -> int:
            minimum = float(np.min(np.asarray(f1.get("F"))[:, column]))
            tied = np.flatnonzero(np.asarray(f1.get("F"))[:, column] == minimum)
            return min(tied.tolist(), key=lambda index: (
                -float(crowding[index]), float(worst_component[index]), str(hashes[index])))

        extremes = [extreme_index(0), extreme_index(1)]
        protected_local: list[int] = []
        protected_hashes: set[str] = set()
        for index in convergence + extremes:
            digest = str(hashes[index])
            if digest not in protected_hashes:
                protected_local.append(index)
                protected_hashes.add(digest)

        unique_extreme_hashes = {str(hashes[index]) for index in extremes}
        if len(convergence) > target or len(unique_extreme_hashes) > 2 or len(protected_local) > target + 2:
            raise AssertionError("frozen CA protected-set bound violated")

        standard_hashes = [self._hash(individual) for individual in standard]
        standard_hash_set = set(standard_hashes)
        selected_individuals = [f1[index] for index in protected_local]
        protected_ids = {id(individual) for individual in selected_individuals}
        for individual in standard:
            if id(individual) not in protected_ids and len(selected_individuals) < n_survive:
                selected_individuals.append(individual)
        if len(selected_individuals) != n_survive:
            raise AssertionError("CA environmental selection did not fill the population")

        selected_indices = []
        used: set[int] = set()
        for individual in selected_individuals:
            matches = [index for index, candidate in enumerate(pop)
                       if index not in used and candidate is individual]
            if not matches:
                digest = self._hash(individual)
                matches = [index for index, candidate in enumerate(pop)
                           if index not in used and self._hash(candidate) == digest]
            if not matches:
                raise AssertionError("selected candidate not found in current union population")
            selected_indices.append(matches[0])
            used.add(matches[0])
        protected_details = [{
            "decision_hash": str(hashes[index]), "distance": float(distance[index]),
            "crowding": float(crowding[index]),
            "would_be_selected_by_v1": str(hashes[index]) in standard_hash_set,
        } for index in protected_local]
        self.audit_records.append({
            "selection_index": len(self.audit_records) + 1,
            "candidate_pool_size": len(pop), "F1_size": len(f1),
            "elite_count_target": target, "elite_count_actual": len(convergence),
            "extreme_count": len(unique_extreme_hashes),
            "protected_set_size": len(protected_local),
            "protected_decision_hashes": [item["decision_hash"] for item in protected_details],
            "protected_distance": [item["distance"] for item in protected_details],
            "protected_crowding": [item["crowding"] for item in protected_details],
            "ca_only_retained_count": sum(not item["would_be_selected_by_v1"] for item in protected_details),
            "standard_survival_overlap_count": sum(item["would_be_selected_by_v1"] for item in protected_details),
            "normalization_scope": "all feasible candidates in current R=P_t union Q_t",
        })
        return pop[np.asarray(selected_indices, dtype=int)]


def _individual_decision_hash(individual: Individual) -> str:
    vector = np.asarray(individual.get("X"), dtype=np.float64).reshape(-1)
    return canonical_decision_hash(tuple(float(value) for value in vector))


def exact_objective_equivalence_compression(pop: Population) -> tuple[Population, Population, dict[str, Any]]:
    """Compress exact raw-float objective duplicates without tolerance or rounding.

    The representative is feasibility-first (minimum CV) and then the
    lexicographically smallest canonical decision hash.  Cluster discovery
    preserves the first-seen objective order so that an all-unique pool is
    byte-for-byte ordered like Standard RVEA's input pool.
    """
    if len(pop) == 0:
        return pop, Population(), {
            "candidate_pool_size": 0, "unique_objective_count": 0,
            "equivalent_cluster_count": 0, "compressed_candidate_count": 0,
            "compression_ratio": 0.0,
        }
    objectives = np.asarray(pop.get("F"))
    if objectives.ndim != 2 or objectives.shape[0] != len(pop) or not np.isfinite(objectives).all():
        raise ValueError("OE-RVEA requires a finite objective matrix")
    cv = np.asarray(pop.get("CV"), dtype=np.float64).reshape(-1)
    if cv.shape != (len(pop),) or not np.isfinite(cv).all() or np.any(cv < 0):
        raise ValueError("OE-RVEA requires finite nonnegative constraint violation")

    clusters: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(objectives):
        # Raw scalar values are used directly.  There is deliberately no
        # epsilon, quantization, rounding, string conversion, or CSV path.
        key = tuple(row.tolist())
        clusters.setdefault(key, []).append(index)

    representative_indices: list[int] = []
    removed_indices: list[int] = []
    for indices in clusters.values():
        representative = min(indices, key=lambda index: (
            float(cv[index]), _individual_decision_hash(pop[index])))
        representative_indices.append(representative)
        removed_indices.extend(index for index in indices if index != representative)

    representatives = pop[np.asarray(representative_indices, dtype=int)]
    removed = pop[np.asarray(removed_indices, dtype=int)] if removed_indices else Population()
    equivalent_clusters = sum(len(indices) > 1 for indices in clusters.values())
    compressed_count = len(pop) - len(representatives)
    audit = {
        "candidate_pool_size": len(pop),
        "unique_objective_count": len(representatives),
        "equivalent_cluster_count": equivalent_clusters,
        "compressed_candidate_count": compressed_count,
        "compression_ratio": float(compressed_count / len(pop)),
    }
    return representatives, removed, audit


class ObjectiveEquivalenceAwareRVEA(RVEA):
    """Standard RVEA with one exact-objective compression hook.

    All variation, repair, reference-vector, APD, adaptation, and budget
    behavior comes from Standard RVEA.  Only the already evaluated
    environmental-selection pool is compressed.
    """

    design_hash = OE_RVEA_V1_DESIGN_HASH

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.objective_equivalence_audit: list[dict[str, Any]] = []

    @staticmethod
    def _fallback_key(individual: Individual) -> tuple[Any, ...]:
        cv = float(np.asarray(individual.get("CV"), dtype=np.float64).reshape(-1)[0])
        apd_raw = individual.get("apd")
        niche_raw = individual.get("niche")
        apd = float(np.asarray(apd_raw).reshape(-1)[0]) if apd_raw is not None else np.inf
        niche = int(np.asarray(niche_raw).reshape(-1)[0]) if niche_raw is not None else np.iinfo(np.int64).max
        return (cv > 0.0, cv, niche, apd, _individual_decision_hash(individual))

    def _advance(self, infills: Any = None, **kwargs: Any) -> None:
        pool = self.pop if infills is None else Population.merge(self.pop, infills)
        representatives, removed, audit = exact_objective_equivalence_compression(pool)
        fallback_used = len(representatives) < self.pop_size
        fallback_added = 0

        if not fallback_used:
            # The unchanged Standard RVEA APD environmental selection receives
            # the compressed pool exactly once.
            selected = self.survival.do(
                self.problem, representatives, n_survive=self.pop_size,
                algorithm=self, random_state=self.random_state, **kwargs)
        else:
            # First obtain Standard RVEA's native survivor order and APD/niche
            # metadata from the original uncompressed pool.  All unique reps
            # remain mandatory; vacated slots use that native order, followed
            # only when necessary by its feasibility/niche/APD fitness order.
            standard = self.survival.do(
                self.problem, pool, n_survive=self.pop_size,
                algorithm=self, random_state=self.random_state, **kwargs)
            selected_individuals = list(representatives)
            selected_ids = {id(individual) for individual in selected_individuals}
            for individual in standard:
                if id(individual) not in selected_ids and len(selected_individuals) < self.pop_size:
                    selected_individuals.append(individual)
                    selected_ids.add(id(individual))
                    fallback_added += 1
            if len(selected_individuals) < self.pop_size:
                remaining = [individual for individual in removed if id(individual) not in selected_ids]
                for individual in sorted(remaining, key=self._fallback_key):
                    if len(selected_individuals) >= self.pop_size:
                        break
                    selected_individuals.append(individual)
                    selected_ids.add(id(individual))
                    fallback_added += 1
            selected = Population.create(*selected_individuals)

        self.pop = selected
        survivor_objectives = (np.asarray(self.pop.get("F")) if len(self.pop)
                               else np.empty((0, int(self.problem.n_obj))))
        survivor_unique = len({tuple(row.tolist()) for row in survivor_objectives})
        self.objective_equivalence_audit.append({
            "generation": int(self.n_gen), **audit,
            "compression_fallback_triggered": fallback_used,
            "fallback_added_count": fallback_added,
            "survivor_count": len(self.pop),
            "survivor_unique_objective_count": survivor_unique,
        })

        # Preserve Standard RVEA's frozen reference-vector adaptation rule.
        n_gen, n_max_gen = self.n_gen, self.termination.n_max_gen
        if self.adapt_freq is not None and n_gen % np.ceil(n_max_gen * self.adapt_freq) == 0:
            self.survival.adapt()


class FiniteSpanAPDSurvival(APDSurvival):
    """APD adaptation guard used only by explicit M>2 objective modes."""

    design_hash = RVEA_4D_FINITE_SPAN_DESIGN_HASH

    def __init__(self, ref_dirs: np.ndarray, alpha: float = 2.0) -> None:
        super().__init__(ref_dirs, alpha=alpha)
        self.adaptation_audit: list[dict[str, Any]] = []

    def adapt(self) -> None:
        if self.nadir is None:
            return
        span = np.asarray(self.nadir - self.ideal, dtype=np.float64)
        positive = span > 1e-15
        if not np.any(positive):
            self.adaptation_audit.append({
                "action": "skip_all_zero_span", "span": span.tolist(),
                "finite_after": bool(np.isfinite(self.V).all()),
            })
            return
        floor = max(float(span[positive].max()) * 1e-12, 1e-15)
        safe_span = np.where(positive, span, floor)
        candidate = calc_V(calc_V(self.ref_dirs) * safe_span)
        if not np.isfinite(candidate).all():
            raise FloatingPointError("nonfinite RVEA reference-vector adaptation")
        self.V = candidate
        self.gamma = calc_gamma(self.V)
        self.adaptation_audit.append({
            "action": "native_with_zero_span_floor" if np.any(~positive) else "native",
            "span": span.tolist(), "floor": floor,
            "finite_after": bool(np.isfinite(self.V).all()),
        })


class FourObjectiveRVEA(RVEA):
    """Standard RVEA with the explicit 4D finite-span APD guard."""

    design_hash = RVEA_4D_FINITE_SPAN_DESIGN_HASH

    def __init__(self, ref_dirs: np.ndarray, *args: Any, **kwargs: Any) -> None:
        super().__init__(ref_dirs, *args, **kwargs)
        self.survival = FiniteSpanAPDSurvival(ref_dirs, alpha=float(self.survival.alpha))


class FourObjectiveObjectiveEquivalenceAwareRVEA(ObjectiveEquivalenceAwareRVEA):
    """OE-RVEA V1 plus the explicit 4D finite-span APD guard."""

    design_hash = RVEA_4D_FINITE_SPAN_DESIGN_HASH

    def __init__(self, ref_dirs: np.ndarray, *args: Any, **kwargs: Any) -> None:
        super().__init__(ref_dirs, *args, **kwargs)
        self.survival = FiniteSpanAPDSurvival(ref_dirs, alpha=float(self.survival.alpha))


class _ExternalProblem(Problem):
    def __init__(self, users: int, f_max: float, objective_count: int = 2) -> None:
        xl = np.concatenate((np.zeros(2 * users), np.zeros(users)))
        xu = np.concatenate((np.ones(2 * users), np.full(users, f_max)))
        super().__init__(n_var=3 * users, n_obj=objective_count, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Exp-7 algorithms must use the shared external evaluator")


class ConstrainedMOEAD(MOEAD):
    """MOEA/D使用Deb feasibility-first替换，补足pymoo 0.6.1.6默认缺口。"""

    def _setup(self, problem: Problem, **kwargs: Any) -> None:
        # pymoo 0.6.1.6只因默认_replace未处理CV而拒绝约束问题；本子类在
        # _replace中显式实现Deb规则，因此仅在父类建立邻域时临时绕过断言。
        count = problem.n_ieq_constr
        problem.n_ieq_constr = 0
        try:
            super()._setup(problem, **kwargs)
        finally:
            problem.n_ieq_constr = count

    def _replace(self, k: int, off: Any) -> None:
        neighbors = self.neighbors[k]
        current = self.pop[neighbors]
        weights = self.ref_dirs[neighbors]
        current_scalar = _paired_decomposition_values(
            self.decomposition, current.get("F"), weights, self.ideal)
        off_scalar = _paired_decomposition_values(
            self.decomposition, np.asarray(off.F, dtype=np.float64).reshape(-1), weights, self.ideal)
        current_cv = current.get("CV")[:, 0]
        off_cv = float(np.asarray(off.CV).reshape(-1)[0])
        replace = np.flatnonzero(_moead_replacement_mask(
            current_scalar, off_scalar, current_cv, off_cv)).tolist()
        if replace:
            self.pop[neighbors[np.asarray(replace, dtype=int)]] = off


def _paired_decomposition_values(decomposition: Any, objectives: np.ndarray,
                                 weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """Scalarize one objective per matching weight without many-to-many broadcast."""
    w = np.asarray(weights, dtype=np.float64)
    f = np.asarray(objectives, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] == 0:
        raise ValueError("neighbor weights must have shape (T,M)")
    if f.ndim == 1:
        if f.shape[0] != w.shape[1]:
            raise ValueError("offspring objective dimension mismatch")
        f = np.repeat(f[None, :], w.shape[0], axis=0)
    if f.shape != w.shape:
        raise ValueError("paired objectives and weights must both have shape (T,M)")
    values = np.asarray(decomposition.do(
        f, weights=w, ideal_point=np.asarray(ideal, dtype=np.float64), _type="one_to_one"),
        dtype=np.float64).reshape(-1)
    if values.shape != (w.shape[0],) or not np.isfinite(values).all():
        raise ValueError("decomposition must return one finite scalar per neighbor")
    return values


def _moead_replacement_mask(old_scalar: np.ndarray, child_scalar: np.ndarray,
                            old_cv: np.ndarray, child_cv: float) -> np.ndarray:
    """Frozen Deb feasibility-first replacement decision for each neighbor."""
    old_s = np.asarray(old_scalar, dtype=np.float64)
    new_s = np.asarray(child_scalar, dtype=np.float64)
    cv = np.asarray(old_cv, dtype=np.float64)
    if old_s.ndim != 1 or new_s.shape != old_s.shape or cv.shape != old_s.shape:
        raise ValueError("aligned one-dimensional neighbor vectors required")
    return ((child_cv <= 0) & (cv > 0)) | ((child_cv > 0) & (cv > 0) & (child_cv < cv)) | (
        (child_cv <= 0) & (cv <= 0) & (new_s < old_s))


class BaseOuterAdapter:
    repair_function = staticmethod(canonical_outer_repair)
    evaluator_entry = CanonicalOuterEvaluator.evaluate_outer_solution_canonical

    def __init__(self, name: str, users: int, context: RepairContext,
                 config: OuterAlgorithmConfig, seed: int, initial_pool: np.ndarray) -> None:
        if name not in ALGORITHM_REGISTRY or users != context.cache_sizes.size:
            raise ValueError("invalid formal adapter binding")
        pool = np.asarray(initial_pool, dtype=np.float64)
        if pool.shape != (config.population_size, 3 * users):
            raise ValueError("matched initial pool shape mismatch")
        self.name, self.users, self.context, self.config, self.seed = name, users, context, config, int(seed)
        self.initial_pool = pool.copy()
        self.history: list[HistoryRecord] = []
        self.pending: tuple[CanonicalSolution, ...] = ()
        self.last_results: list[dict[str, Any]] = []
        self.generation = 0
        self._last_request_count = 0
        self._last_unique_count = 0

    @property
    def objective_count(self) -> int:
        """Preserve compatibility with checkpoints created before 4D mode."""
        return int(getattr(self.config, "objective_count", 2))

    def canonicalize(self, vector: Sequence[float]) -> CanonicalSolution:
        return canonicalize_vector(vector, self.users)

    def repair(self, solution: CanonicalSolution) -> CanonicalSolution:
        return canonical_outer_repair(solution, self.context)

    def evaluate(self, evaluator: CanonicalOuterEvaluator) -> list[dict[str, Any]]:
        if not isinstance(evaluator, CanonicalOuterEvaluator):
            raise TypeError("all adapters require the one CanonicalOuterEvaluator class")
        if not self.pending:
            raise RuntimeError("ask must precede evaluate")
        if evaluator.requests + len(self.pending) > self.config.evaluation_budget:
            raise RuntimeError("evaluation request budget would be exceeded")
        results = [dict(evaluator.evaluate_outer_solution_canonical(x, self.context)) for x in self.pending]
        self._last_request_count = evaluator.requests
        self._last_unique_count = evaluator.unique_evaluations
        return results

    def get_history(self) -> tuple[HistoryRecord, ...]:
        return tuple(self.history)

    def finalize(self) -> dict[str, Any]:
        return {"algorithm": self.name, "generation": self.generation,
                "history_records": len(self.history), "workers": self.config.workers}


class PymooOuterAdapter(BaseOuterAdapter):
    def __init__(self, name: str, users: int, context: RepairContext,
                 config: OuterAlgorithmConfig, seed: int, initial_pool: np.ndarray) -> None:
        super().__init__(name, users, context, config, seed, initial_pool)
        if config.objective_count == 2:
            ref_dirs = get_reference_directions(
                "uniform", 2, n_partitions=config.population_size - 1)
        else:
            # Energy reference directions provide exactly pop_size deterministic
            # simplex vectors for M>2, avoiding a combinatorial Das-Dennis count.
            ref_dirs = get_reference_directions(
                "energy", config.objective_count, config.population_size, seed=int(seed))
        if name == "CorrectedDualBP-CNSGA-II":
            algorithm = NSGA2(pop_size=config.population_size, sampling=initial_pool)
        elif name == V2_ALGORITHM:
            algorithm = NSGA2(pop_size=config.population_size, sampling=initial_pool,
                              survival=ConvergenceAwareSurvival(config.population_size))
        elif name == "MOEA/D":
            algorithm = ConstrainedMOEAD(ref_dirs=ref_dirs, n_neighbors=min(20, config.population_size), sampling=initial_pool)
        elif name == "RVEA":
            algorithm_type = FourObjectiveRVEA if config.objective_count > 2 else RVEA
            algorithm = algorithm_type(
                ref_dirs, pop_size=config.population_size, sampling=initial_pool)
        elif name == OE_RVEA_ALGORITHM:
            algorithm_type = (FourObjectiveObjectiveEquivalenceAwareRVEA
                              if config.objective_count > 2
                              else ObjectiveEquivalenceAwareRVEA)
            algorithm = algorithm_type(
                ref_dirs, pop_size=config.population_size, sampling=initial_pool)
        else:
            raise ValueError("unsupported pymoo adapter")
        self.algorithm = algorithm
        self.problem = _ExternalProblem(users, context.f_max, config.objective_count)
        self.algorithm.setup(self.problem, seed=self.seed, termination=("n_eval", config.evaluation_budget), verbose=False)
        self._infills: Any = None

    @property
    def ca_selection_audit(self) -> tuple[dict[str, Any], ...]:
        survival = getattr(self.algorithm, "survival", None)
        return tuple(getattr(survival, "audit_records", ()))

    @property
    def objective_equivalence_audit(self) -> tuple[dict[str, Any], ...]:
        return tuple(getattr(self.algorithm, "objective_equivalence_audit", ()))

    @property
    def reference_vector_adaptation_audit(self) -> tuple[dict[str, Any], ...]:
        survival = getattr(self.algorithm, "survival", None)
        return tuple(getattr(survival, "adaptation_audit", ()))

    def ask(self) -> tuple[CanonicalSolution, ...]:
        self._infills = self.algorithm.ask()
        values = np.atleast_2d(self._infills.get("X"))
        repaired = tuple(self.repair(self.canonicalize(row)) for row in values)
        repaired_values = np.asarray([flatten_solution(x) for x in repaired])
        self._infills.set("X", repaired_values[0] if isinstance(self._infills, Individual) else repaired_values)
        self.pending = repaired
        return repaired

    def tell(self, results: Sequence[Mapping[str, Any]]) -> None:
        if self._infills is None:
            raise RuntimeError("ask must precede tell")
        f, cv = _result_arrays(results, len(self.pending), self.objective_count)
        if isinstance(self._infills, Individual):
            self._infills.set("F", f[0])
            self._infills.set("G", np.asarray([cv[0]]))
        else:
            self._infills.set("F", f)
            self._infills.set("G", cv[:, None])
        self.algorithm.tell(infills=self._infills)
        self.generation += 1
        self.last_results = [dict(row) for row in results]
        feasible = cv <= 0
        nd = len(nondominated_min(f[feasible])) if np.any(feasible) else 0
        self.history.append(HistoryRecord(self._last_request_count, self._last_unique_count,
                                          self.generation, float(feasible.mean()), nd, None))
        self.pending, self._infills = (), None

    def get_population(self) -> tuple[CanonicalSolution, ...]:
        if self.algorithm.pop is None:
            return ()
        return tuple(self.repair(self.canonicalize(row)) for row in np.atleast_2d(self.algorithm.pop.get("X")))

    def get_nondominated_front(self) -> np.ndarray:
        if self.algorithm.pop is None:
            return np.empty((0, self.objective_count))
        f, cv = self.algorithm.pop.get("F"), self.algorithm.pop.get("CV")[:, 0]
        return (nondominated_min(f[cv <= 0]) if np.any(cv <= 0)
                else np.empty((0, self.objective_count)))


class MOPSOAdapter(BaseOuterAdapter):
    """标准archive-based MOPSO；二元维采用sigmoid velocity transfer。"""

    w, c1, c2, archive_max_size = 0.5, 1.5, 1.5, 100

    def __init__(self, users: int, context: RepairContext, config: OuterAlgorithmConfig,
                 seed: int, initial_pool: np.ndarray) -> None:
        super().__init__("MOPSO", users, context, config, seed, initial_pool)
        self.rng = np.random.default_rng(self.seed)
        self.positions = initial_pool.copy()
        self.velocities = self.rng.uniform(-1.0, 1.0, self.positions.shape)
        self.pbest_positions = self.positions.copy()
        self.pbest_f = np.full((config.population_size, config.objective_count), np.inf)
        self.pbest_cv = np.full(config.population_size, np.inf)
        self.archive_positions = np.empty((0, 3 * users))
        self.archive_f = np.empty((0, config.objective_count))
        self.archive_cv = np.empty(0)

    def ask(self) -> tuple[CanonicalSolution, ...]:
        repaired = tuple(self.repair(self.canonicalize(row)) for row in self.positions)
        self.positions = np.asarray([flatten_solution(x) for x in repaired])
        self.pending = repaired
        return repaired

    def _update_archive(self, positions: np.ndarray, f: np.ndarray, cv: np.ndarray) -> None:
        pos = np.vstack((self.archive_positions, positions))
        obj = np.vstack((self.archive_f, f))
        violations = np.concatenate((self.archive_cv, cv))
        feasible = violations <= 0
        if np.any(feasible):
            candidate_idx = np.flatnonzero(feasible)
            front = nondominated_min(obj[candidate_idx])
            chosen = []
            for point in front:
                chosen.append(candidate_idx[np.flatnonzero(np.all(obj[candidate_idx] == point, axis=1))[0]])
        else:
            minimum = violations.min()
            chosen = np.flatnonzero(violations == minimum).tolist()
        if len(chosen) > self.archive_max_size:
            crowding = _crowding(obj[chosen])
            chosen = [chosen[i] for i in np.argsort(-crowding, kind="stable")[:self.archive_max_size]]
        self.archive_positions, self.archive_f, self.archive_cv = pos[chosen], obj[chosen], violations[chosen]

    def tell(self, results: Sequence[Mapping[str, Any]]) -> None:
        f, cv = _result_arrays(results, len(self.pending), self.objective_count)
        for i in range(len(f)):
            if not np.isfinite(self.pbest_f[i]).all() or _deb_better(f[i], cv[i], self.pbest_f[i], self.pbest_cv[i]):
                self.pbest_positions[i], self.pbest_f[i], self.pbest_cv[i] = self.positions[i], f[i], cv[i]
        self._update_archive(self.positions, f, cv)
        crowding = _crowding(self.archive_f)
        finite = crowding[np.isfinite(crowding)]
        cap = (finite.max() if finite.size else 1.0) * 2.0
        weights = np.where(np.isfinite(crowding), crowding, cap) + 1e-12
        leader_idx = self.rng.choice(len(self.archive_positions), len(self.positions), p=weights / weights.sum())
        leaders = self.archive_positions[leader_idx]
        r1, r2 = self.rng.random(self.positions.shape), self.rng.random(self.positions.shape)
        self.velocities = (self.w * self.velocities + self.c1 * r1 * (self.pbest_positions - self.positions)
                           + self.c2 * r2 * (leaders - self.positions))
        binary = 2 * self.users
        transfer = 1.0 / (1.0 + np.exp(-np.clip(self.velocities[:, :binary], -60, 60)))
        self.positions[:, :binary] = (self.rng.random(transfer.shape) < transfer).astype(np.float64)
        self.positions[:, binary:] = np.clip(self.positions[:, binary:] + self.velocities[:, binary:], 0, self.context.f_max)
        self.generation += 1
        self.last_results = [dict(row) for row in results]
        feasible = cv <= 0
        self.history.append(HistoryRecord(self._last_request_count, self._last_unique_count,
                                          self.generation, float(feasible.mean()),
                                          len(self.archive_f) if np.any(feasible) else 0, None))
        self.pending = ()

    def get_population(self) -> tuple[CanonicalSolution, ...]:
        return tuple(self.repair(self.canonicalize(row)) for row in self.positions)

    def get_nondominated_front(self) -> np.ndarray:
        return (self.archive_f.copy() if np.any(self.archive_cv <= 0)
                else np.empty((0, self.objective_count)))


class RandomAdapter(BaseOuterAdapter):
    def __init__(self, users: int, context: RepairContext, config: OuterAlgorithmConfig,
                 seed: int, initial_pool: np.ndarray) -> None:
        super().__init__("Random", users, context, config, seed, initial_pool)
        self.rng = np.random.default_rng(self.seed)
        self.positions = initial_pool.copy()
        self.archive_f = np.empty((0, config.objective_count))

    def ask(self) -> tuple[CanonicalSolution, ...]:
        if self.generation:
            self.positions = matched_canonical_initial_population(
                int(self.rng.integers(0, 2**31 - 1)), self.config.population_size, self.context)
        self.pending = tuple(self.repair(self.canonicalize(row)) for row in self.positions)
        return self.pending

    def tell(self, results: Sequence[Mapping[str, Any]]) -> None:
        f, cv = _result_arrays(results, len(self.pending), self.objective_count)
        feasible = cv <= 0
        if np.any(feasible):
            self.archive_f = nondominated_min(np.vstack((self.archive_f, f[feasible])))
        self.generation += 1
        self.last_results = [dict(row) for row in results]
        self.history.append(HistoryRecord(self._last_request_count, self._last_unique_count,
                                          self.generation, float(feasible.mean()), len(self.archive_f), None))
        self.pending = ()

    def get_population(self) -> tuple[CanonicalSolution, ...]:
        return tuple(self.repair(self.canonicalize(row)) for row in self.positions)

    def get_nondominated_front(self) -> np.ndarray:
        return self.archive_f.copy()


def create_adapter(name: str, users: int, context: RepairContext, config: OuterAlgorithmConfig,
                   seed: int, initial_pool: np.ndarray) -> BaseOuterAdapter:
    if name in {"CorrectedDualBP-CNSGA-II", V2_ALGORITHM, "MOEA/D", "RVEA", OE_RVEA_ALGORITHM}:
        return PymooOuterAdapter(name, users, context, config, seed, initial_pool)
    if name == "MOPSO":
        return MOPSOAdapter(users, context, config, seed, initial_pool)
    if name == "Random":
        return RandomAdapter(users, context, config, seed, initial_pool)
    raise ValueError(f"algorithm not in frozen Formal set: {name}")
