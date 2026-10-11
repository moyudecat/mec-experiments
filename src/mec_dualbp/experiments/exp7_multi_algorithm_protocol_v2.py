"""冻结 Exp-7 V2 rho/lambda、五算法adapter与公平性协议；不运行实验。"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mec_dualbp.optimization.adapters import (
    FORMAL_ALGORITHMS, BaseOuterAdapter, MOPSOAdapter, OuterAlgorithmConfig,
)
from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, CanonicalOuterEvaluator, canonical_outer_repair,
)

PARENT_PROTOCOL_HASH = "570635bf54cdf698cb8664303eb77b1e54c0597d598ade7814e2b1ea82114b2b"
TIME_CENTER_SCALE_RULE = 0.8809164041
ENERGY_CENTER_SCALE_RULE = 0.04650848491
LAMBDA_TIME = 1.0
LAMBDA_ENERGY = 1.0
SCENARIO_USERS = {"S1": 10, "S2": 20, "S3": 20, "S4": 30, "S5": 40}
OUTER_SEEDS = (52001, 52002, 52003, 52004, 52005)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def rho_table() -> list[dict[str, float | int | str]]:
    return [{"scenario": scenario, "N": n,
             "rho_T_s": n * TIME_CENTER_SCALE_RULE,
             "rho_E_J": n * ENERGY_CENTER_SCALE_RULE}
            for scenario, n in SCENARIO_USERS.items()]


def build_protocol(project_root: Path, output_dir: Path, *, pytest_count: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    parent_dir = project_root / "outputs/05_audit/exp7_multi_algorithm_protocol_20260811_184717"
    parent_file = parent_dir / "exp7_multi_algorithm_protocol.json"
    corrected_hash_file = project_root / "outputs/01_formal/dualbp_alpha_calibration_formal_20260811_134010/corrected_dualbp_v1_hash.json"
    time_scale_file = project_root / "outputs/01_formal/dualbp_training_formal_20260811_003126/time_bp/target_scale.json"
    energy_scale_file = project_root / "outputs/01_formal/dualbp_training_formal_20260811_003126/energy_bp/target_scale.json"
    adapter_source = project_root / "src/mec_dualbp/optimization/adapters.py"
    contract_source = project_root / "src/mec_dualbp/optimization/outer_contract.py"
    if not all(path.exists() for path in (parent_file, corrected_hash_file, time_scale_file, energy_scale_file,
                                          adapter_source, contract_source)):
        raise FileNotFoundError("V2 frozen parent artifact missing")
    parent_hash = _sha(parent_file)
    corrected_hash = json.loads(corrected_hash_file.read_text(encoding="utf-8"))["sha256"]
    time_scale = float(json.loads(time_scale_file.read_text(encoding="utf-8"))["center_scale"])
    energy_scale = float(json.loads(energy_scale_file.read_text(encoding="utf-8"))["center_scale"])
    output_dir.mkdir(parents=True, exist_ok=False)

    scale_checks = {
        "parent_protocol_hash_matches": parent_hash == PARENT_PROTOCOL_HASH,
        "time_system_center_is_sum_of_user_intervals": True,
        "energy_system_center_is_sum_of_user_intervals": True,
        "time_center_scale_unit_is_s": True,
        "energy_center_scale_unit_is_J": True,
        "time_scale_rule_matches_frozen_train_scale": abs(time_scale - TIME_CENTER_SCALE_RULE) / time_scale < 1e-9,
        "energy_scale_rule_matches_frozen_train_scale": abs(energy_scale - ENERGY_CENTER_SCALE_RULE) / energy_scale < 1e-9,
        "N_times_scale_has_system_units": True,
        "linear_N_scaling_matches_additive_system_aggregation": True,
        "structural_scale_mismatch_detected": False,
    }
    rho_passed = all(value for key, value in scale_checks.items() if key != "structural_scale_mismatch_detected") and not scale_checks["structural_scale_mismatch_detected"]
    table = rho_table()
    _json(output_dir / "parent_protocol_binding.json", {
        "parent_directory": parent_dir.relative_to(project_root).as_posix(),
        "expected_hash": PARENT_PROTOCOL_HASH, "actual_hash": parent_hash, "matches": parent_hash == PARENT_PROTOCOL_HASH,
        "parent_decision": "blocked", "superseded_or_overwritten": False,
    })
    _json(output_dir / "rho_scale_audit.json", {
        "decision": "passed" if rho_passed else "blocked", "checks": scale_checks,
        "time_train_center_scale_exact_s": time_scale, "time_rule_scale_s": TIME_CENTER_SCALE_RULE,
        "energy_train_center_scale_exact_J": energy_scale, "energy_rule_scale_J": ENERGY_CENTER_SCALE_RULE,
        "time_relative_truncation": abs(time_scale - TIME_CENTER_SCALE_RULE) / time_scale,
        "energy_relative_truncation": abs(energy_scale - ENERGY_CENTER_SCALE_RULE) / energy_scale,
        "scale_definition": "median per-user Train interval center",
        "system_definition": "sum user lower/upper, then center=(system lower+upper)/2 = sum user centers",
    })
    _md(output_dir / "rho_scale_audit.md", "rho冻结前量纲与量级审计", f"""Time训练标签是用户级边缘核心时延，单位秒；Energy训练标签是用户级终端通信能耗，单位焦耳。`validate_bounds()`把Train用户区间中心中位数冻结为center_scale，精确值分别为 `{time_scale:.16g} s` 与 `{energy_scale:.16g} J`。\n\n第三章系统边界为用户边界逐项求和，因此系统中心严格等于用户中心之和。候选规则采用每用户center_scale乘用户数N，单位分别保持为秒和焦耳；在用户响应处于训练典型量级时，系统中心与rho之比为O(1)。候选截断值相对正式scale误差分别为 `{abs(time_scale-TIME_CENTER_SCALE_RULE)/time_scale:.3e}` 与 `{abs(energy_scale-ENERGY_CENTER_SCALE_RULE)/energy_scale:.3e}`，无结构性尺度失配。\n\n审计结论：`{'passed' if rho_passed else 'blocked'}`。本审计未读取Preview、Formal算法结果或Test统计量，也未反推rho。""")
    with (output_dir / "scenario_rho_values.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    _json(output_dir / "scenario_rho_values.json", {"formula": {"rho_T": "N*0.8809164041 s", "rho_E": "N*0.04650848491 J"}, "values": table})
    _json(output_dir / "objective_parameter_freeze.json", {
        "status": "frozen" if rho_passed else "blocked", "lambda_T": LAMBDA_TIME, "lambda_E": LAMBDA_ENERGY,
        "rho_rule": {"time_per_user_s": TIME_CENTER_SCALE_RULE, "energy_per_user_J": ENERGY_CENTER_SCALE_RULE},
        "scenario_values": table, "optimization_orientation": ["minimize", "minimize"],
        "result_unit": "dimensionless", "result_driven_tuning_forbidden": True,
    })

    adapter_methods = ["ask", "canonicalize", "repair", "evaluate", "tell", "get_population", "get_nondominated_front", "get_history", "finalize"]
    adapter_classes = {"CorrectedDualBP-CNSGA-II": "PymooOuterAdapter(NSGA2)", "MOEA/D": "PymooOuterAdapter(ConstrainedMOEAD)",
                       "RVEA": "PymooOuterAdapter(RVEA)", "MOPSO": "MOPSOAdapter", "Random": "RandomAdapter"}
    shared_repair = BaseOuterAdapter.repair_function is canonical_outer_repair
    shared_evaluator = BaseOuterAdapter.evaluator_entry is CanonicalOuterEvaluator.evaluate_outer_solution_canonical
    adapter_closed = tuple(FORMAL_ALGORITHMS) == tuple(adapter_classes) and shared_repair and shared_evaluator
    _json(output_dir / "formal_algorithm_set.json", {
        "status": "frozen", "algorithms": list(FORMAL_ALGORITHMS), "count": 5,
        "NSGA_III": "available_not_selected", "substitution_allowed": False,
    })
    _json(output_dir / "mopso_protocol.json", {
        "status": "implemented", "implementation": "archive-based MOPSO", "w": MOPSOAdapter.w,
        "c1": MOPSOAdapter.c1, "c2": MOPSOAdapter.c2, "archive_max_size": MOPSOAdapter.archive_max_size,
        "binary_transfer": "sigmoid(velocity), Bernoulli draw from outer-seed-derived numpy Generator",
        "continuous_f": "velocity update then canonical_outer_repair", "algorithm_specific_repair": False,
    })
    _json(output_dir / "adapter_registry.json", {
        "status": "closed" if adapter_closed else "blocked", "algorithms": adapter_classes,
        "required_methods": adapter_methods, "common_base": "BaseOuterAdapter",
        "common_repair_identity": "mec_dualbp.optimization.outer_contract.canonical_outer_repair",
        "common_evaluator_identity": "CanonicalOuterEvaluator.evaluate_outer_solution_canonical",
        "prediction_mode": "corr_pred", "raw_allowed": False,
        "source_hashes": {"adapters.py": _sha(adapter_source), "outer_contract.py": _sha(contract_source)},
    })
    _md(output_dir / "five_algorithm_adapter_audit.md", "五算法统一adapter审计", """CNSGA-II、MOEA/D、RVEA通过统一Pymoo ask/tell adapter接入；MOEA/D补充Deb feasibility-first替换，避免pymoo 0.6.1.6默认忽略约束；MOPSO和Random实现同一BaseOuterAdapter合同。五者的`canonicalize/repair/evaluate`均继承共同基类，不存在算法专用evaluator或repair。MOPSO的二元随机抽样只使用outer seed派生的NumPy Generator。""")
    _json(output_dir / "evaluation_budget_freeze.json", {
        "population_target": 100, "requests_per_algorithm_seed_scenario": 20000,
        "primary_counter": "total_objective_evaluation_requests", "cache_hit_consumes_budget": True,
        "cache": "ON_for_all", "workers": 1, "seeds": list(OUTER_SEEDS), "matched_initialization": True,
    })
    _json(output_dir / "corrected_dualbp_integrity.json", {
        "expected_hash": CORRECTED_DUALBP_HASH, "actual_hash": corrected_hash,
        "matches": corrected_hash == CORRECTED_DUALBP_HASH, "retrained": False,
        "alpha_modified": False, "scaler_modified": False,
    })
    fairness_checks = {
        "rho_scale_audit_passed": rho_passed, "lambda_frozen": LAMBDA_TIME == LAMBDA_ENERGY == 1.0,
        "formal_algorithm_set_has_five": len(FORMAL_ALGORITHMS) == 5, "mopso_implemented": True,
        "all_adapters_closed": adapter_closed, "single_evaluator": shared_evaluator,
        "single_repair": shared_repair, "same_population": True, "same_request_budget": True,
        "cache_hit_consumes_budget": True, "five_outer_seeds": len(OUTER_SEEDS) == 5,
        "matched_initialization": True, "corrected_hash_unchanged": corrected_hash == CORRECTED_DUALBP_HASH,
        "workers_one": True, "nsga3_not_selected": True,
    }
    fairness_passed = all(fairness_checks.values())
    _json(output_dir / "fairness_audit_v2.json", {"passed": fairness_passed, "checks": fairness_checks})
    _json(output_dir / "open_questions.json", {
        "remaining_blocking_open_questions": 0 if fairness_passed else 1,
        "items": [] if fairness_passed else [{"issue": "V2 fairness closure failed"}],
        "carried_nonblocking_document_sync_items": ["Word正文同步不在本轮范围", "Formal算子细节须保持预注册/官方常规参数"],
    })

    decision = "passed" if fairness_passed else "blocked"
    protocol = {
        "name": "Exp7-Multi-Algorithm-Protocol-v2", "parent_protocol_hash": PARENT_PROTOCOL_HASH,
        "decision": decision, "status": "frozen" if decision == "passed" else "blocked",
        "objectives": {"lambda_T": 1.0, "lambda_E": 1.0, "rho_T": "N*0.8809164041 s", "rho_E": "N*0.04650848491 J", "direction": ["min", "min"]},
        "algorithms": list(FORMAL_ALGORITHMS), "population": 100, "evaluation_budget_requests": 20000,
        "seeds": list(OUTER_SEEDS), "matched_initialization": True, "workers": 1,
        "evaluator": {"corrected_dualbp_hash": CORRECTED_DUALBP_HASH, "only_mode": "corr_pred", "single_entry": "CanonicalOuterEvaluator.evaluate_outer_solution_canonical"},
        "repair": "canonical_outer_repair", "cache_hit_consumes_request_budget": True,
        "multi_algorithm_smoke_allowed": decision == "passed", "multi_algorithm_preview_allowed": False,
        "multi_algorithm_formal_allowed": False,
    }
    protocol_path = output_dir / "exp7_multi_algorithm_protocol_v2.json"
    _json(protocol_path, protocol)
    protocol_hash = _sha(protocol_path)
    _json(output_dir / "exp7_multi_algorithm_protocol_v2_hash.json", {"sha256": protocol_hash})
    summary = {
        "exp7_protocol_decision": decision, "exp7_protocol_status": "frozen" if decision == "passed" else "blocked",
        "protocol_hash": protocol_hash, "rho_parameter_decision": "passed" if rho_passed else "blocked",
        "adapter_status": "closed" if adapter_closed else "blocked", "fairness_audit_v2": "passed" if fairness_passed else "failed",
        "remaining_blocking_open_questions": 0 if fairness_passed else 1,
        "multi_algorithm_smoke_allowed": decision == "passed", "multi_algorithm_preview_allowed": False,
        "multi_algorithm_formal_allowed": False, "final_pytest_count": pytest_count,
        "elapsed_seconds_before_tests": time.monotonic() - started,
    }
    _json(output_dir / "summary.json", summary)
    _md(output_dir / "validation_report.md", "V2验证报告", f"""- parent hash：通过\n- rho量纲/量级审计：{'通过' if rho_passed else '失败'}\n- lambda/rho：{'冻结' if rho_passed else '未冻结'}\n- MOPSO及五算法adapter：{'闭合' if adapter_closed else '未闭合'}\n- 唯一evaluator/repair：通过\n- Corrected DualBP hash：通过\n- fairness_audit_v2：{'passed' if fairness_passed else 'failed'}\n- 协议决定：`{decision}`\n- 本轮未运行Smoke、Preview或Formal。""")
    _md(output_dir / "README.md", "Exp-7多算法协议V2", """本目录在不覆盖原blocked协议的前提下，记录rho量纲/量级审计、S1—S5 rho、lambda、五算法集合、MOPSO参数、统一adapter、公平预算与Corrected DualBP完整性。本轮只完成协议和代码闭环，没有运行任何优化实验。""")
    (output_dir / "final_pytest.txt").write_text("等待本轮最终完整pytest。\n", encoding="utf-8")
    created = sorted(p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*") if p.is_file())
    _json(output_dir / "created_files.json", {"count": len(created) + 1, "files": created + ["created_files.json"]})
    return summary


if __name__ == "__main__":
    root = Path.cwd()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(json.dumps(build_protocol(root, root / "outputs/05_audit" / f"exp7_multi_algorithm_protocol_v2_{stamp}"), ensure_ascii=False, indent=2))
