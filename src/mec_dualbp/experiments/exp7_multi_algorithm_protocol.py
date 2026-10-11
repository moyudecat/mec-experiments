"""生成 Exp-7 多算法协议与公平性审计；不运行任何外层优化实验。"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, ENERGY_MODEL_HASH, TIME_MODEL_HASH,
)

EDGE_CPU_CONFIG_HASH = "6869885f71375c28c82e66a41534746a2156c6ceb91b99f5bc76ccb565dc16c3"
CHAPTER4_HASH = "c628ff511e81c7e21ef6c942115dd5874ff2efb21405c3796374bffd9e2bde48"
CHAPTER5_HASH = "8ac99141a00c0975436f4096fe9fc59ffaa7d59e30670576b1e53849d07a3749"
OUTER_SEEDS = (52001, 52002, 52003, 52004, 52005)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def _algorithm_rows() -> list[dict[str, Any]]:
    pymoo = importlib.util.find_spec("pymoo") is not None
    return [
        {"Algorithm": "CorrectedDualBP-CNSGA-II", "Implemented": False, "MixedBinaryContinuousSupport": "adapter_required", "ConstraintSupport": "Deb/native constraints", "UsesSameRepair": True, "UsesSameEvaluator": True, "NeedsExtraTraining": False, "EvaluationBudgetComparable": True, "ConvergenceHistoryAvailable": "adapter_required", "DeterministicSeedSupport": True, "ImplementationRisk": "medium", "SelectedForFormal": "pending", "Reason": "第四章主算法；pymoo NSGA2可用，但项目适配器尚未实现"},
        {"Algorithm": "CorrectedDualBP-MOEA/D", "Implemented": False, "MixedBinaryContinuousSupport": "adapter_required", "ConstraintSupport": "common CV adapter required", "UsesSameRepair": True, "UsesSameEvaluator": True, "NeedsExtraTraining": False, "EvaluationBudgetComparable": True, "ConvergenceHistoryAvailable": "adapter_required", "DeterministicSeedSupport": True, "ImplementationRisk": "medium", "SelectedForFormal": "pending", "Reason": "第五章分解式基线；pymoo MOEAD可用"},
        {"Algorithm": "CorrectedDualBP-RVEA", "Implemented": False, "MixedBinaryContinuousSupport": "adapter_required", "ConstraintSupport": "common CV adapter required", "UsesSameRepair": True, "UsesSameEvaluator": True, "NeedsExtraTraining": False, "EvaluationBudgetComparable": True, "ConvergenceHistoryAvailable": "adapter_required", "DeterministicSeedSupport": True, "ImplementationRisk": "medium", "SelectedForFormal": "pending", "Reason": "第五章参考向量基线；pymoo RVEA可用"},
        {"Algorithm": "CorrectedDualBP-MOPSO", "Implemented": False, "MixedBinaryContinuousSupport": "not_verified", "ConstraintSupport": "not_verified", "UsesSameRepair": "not_verified", "UsesSameEvaluator": "not_verified", "NeedsExtraTraining": False, "EvaluationBudgetComparable": "not_verified", "ConvergenceHistoryAvailable": "not_verified", "DeterministicSeedSupport": "not_verified", "ImplementationRisk": "high", "SelectedForFormal": "blocked", "Reason": "第五章群智能基线，但当前项目与已装pymoo均无可绑定实现"},
        {"Algorithm": "CorrectedDualBP-Random", "Implemented": False, "MixedBinaryContinuousSupport": "straightforward", "ConstraintSupport": "shared repair/evaluator", "UsesSameRepair": True, "UsesSameEvaluator": True, "NeedsExtraTraining": False, "EvaluationBudgetComparable": True, "ConvergenceHistoryAvailable": "adapter_required", "DeterministicSeedSupport": True, "ImplementationRisk": "low", "SelectedForFormal": "pending", "Reason": "第五章无智能搜索参考线"},
        {"Algorithm": "CorrectedDualBP-NSGA-III", "Implemented": False, "MixedBinaryContinuousSupport": "adapter_required", "ConstraintSupport": "common CV adapter required", "UsesSameRepair": True, "UsesSameEvaluator": True, "NeedsExtraTraining": False, "EvaluationBudgetComparable": True, "ConvergenceHistoryAvailable": "adapter_required", "DeterministicSeedSupport": True, "ImplementationRisk": "medium", "SelectedForFormal": "alternative_only", "Reason": f"pymoo可用={pymoo}；可替代MOPSO，但与第五章清单不同，需作者确认"},
    ]


def build_protocol(project_root: Path, output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    chapter4 = project_root / "reference_docs/第四章_双BP误差修正_校订三线表版.docx"
    chapter5 = project_root / "reference_docs/第五章_实验设计参考稿_误差修正_校订三线表版.docx"
    corrected = project_root / "outputs/01_formal/dualbp_alpha_calibration_formal_20260811_134010/corrected_dualbp_v1.json"
    corrected_hash = project_root / "outputs/01_formal/dualbp_alpha_calibration_formal_20260811_134010/corrected_dualbp_v1_hash.json"
    edge_cpu = project_root / "configs/edge_cpu_formal_parameters_v1.json"
    exp3 = project_root / "outputs/01_formal/exp3_ablation_test_formal_20260811_180401/summary.json"
    required = (chapter4, chapter5, corrected, corrected_hash, edge_cpu, exp3)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("frozen source artifact missing")
    observed = {
        "chapter4_sha256": _sha(chapter4), "chapter5_sha256": _sha(chapter5),
        "corrected_semantic_hash": json.loads(corrected_hash.read_text(encoding="utf-8"))["sha256"],
        "edge_cpu_config_sha256": _sha(edge_cpu),
    }
    if observed != {"chapter4_sha256": CHAPTER4_HASH, "chapter5_sha256": CHAPTER5_HASH,
                    "corrected_semantic_hash": CORRECTED_DUALBP_HASH,
                    "edge_cpu_config_sha256": EDGE_CPU_CONFIG_HASH}:
        raise RuntimeError("frozen source hash drift")
    output_dir.mkdir(parents=True, exist_ok=False)

    blockers = [
        {"id": "EXP7-OQ-01", "blocking": True, "issue": "ρ_T、ρ_E、λ_T、λ_E无正式数值或冻结方法", "source": "docs/unresolved_questions.md UQ-09；第五章表5-1仅列待填字段", "required_decision": "作者冻结四个目标参数及来源"},
        {"id": "EXP7-OQ-02", "blocking": True, "issue": "第五章要求MOPSO，但项目和pymoo 0.6.1.6均无可绑定MOPSO实现", "source": "第五章 Exp-7；outer registry现场审计", "required_decision": "实现并验证MOPSO，或批准以pymoo NSGA-III替代并同步论文"},
        {"id": "EXP7-OQ-03", "blocking": True, "issue": "统一外层评价主体、C1-C6修复器及五算法adapter尚未实现", "source": "src现场审计", "required_decision": "按本协议实现后执行接口Smoke"},
        {"id": "EXP7-OQ-04", "blocking": False, "issue": "C7/CV浮点可行容差未冻结", "source": "docs/unresolved_questions.md UQ-14", "required_decision": "Preview前预注册统一容差，不得按算法分别设置"},
        {"id": "EXP7-OQ-05", "blocking": False, "issue": "鲁棒目标与C7仍在UQ-03后续视觉复核清单中", "source": "docs/formula_visual_review_checklist.md", "required_decision": "正式实现前完成对应视觉复核"},
    ]
    word_algorithms = ["CorrectedDualBP-CNSGA-II", "CorrectedDualBP-MOEA/D", "CorrectedDualBP-RVEA", "CorrectedDualBP-MOPSO", "CorrectedDualBP-Random"]
    protocol = {
        "name": "Exp7-Multi-Algorithm-Protocol-v1-draft-blocked",
        "stage": "protocol_and_fairness_audit_only", "workers": 1,
        "decision_variables": {"offload": "m_n binary", "cache": "c_n binary", "edge_cpu": "f_n continuous cycle/s"},
        "objectives": {"direction": "minimize", "time": "atan(T_center/rho_T)+lambda_T*atan(T_radius/rho_T)", "energy": "atan(E_center/rho_E)+lambda_E*atan(E_radius/rho_E)", "numeric_parameters_frozen": False},
        "constraints": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "default_scenario": "S3", "default_eta": 0.90,
        "evaluator": {"mode": "corr_pred", "corrected_dualbp_hash": CORRECTED_DUALBP_HASH, "raw_allowed": False, "post_search_ga_certification": False},
        "algorithm_set": {"word_target": word_algorithms, "frozen": False},
        "budget": {"primary_counter": "total_objective_evaluation_requests", "population_target": 100, "formal_evaluation_budget": 20000, "difference_limit_percent": 2.0},
        "seeds": list(OUTER_SEEDS), "cache": "ON_for_all", "primary_metric": "HV↑", "secondary_metric": "IGD↓",
        "status": "blocked", "blocker_ids": [x["id"] for x in blockers if x["blocking"]],
    }
    protocol_path = output_dir / "exp7_multi_algorithm_protocol.json"
    _json(protocol_path, protocol)
    protocol_hash = _sha(protocol_path)
    _json(output_dir / "exp7_multi_algorithm_protocol_hash.json", {"sha256": protocol_hash})

    _json(output_dir / "frozen_evaluator_binding.json", {
        "status": "artifact_hashes_verified_wrapper_contract_frozen_implementation_pending",
        "time_model_hash": TIME_MODEL_HASH, "energy_model_hash": ENERGY_MODEL_HASH,
        "corrected_dualbp_hash": CORRECTED_DUALBP_HASH,
        "alpha": {"T_minus": 0.012904954451201177, "T_plus": 0.012157642355862266, "E_minus": 0.00750780303100657, "E_plus": 0.003551192376592582},
        "only_entry": "CanonicalOuterEvaluator.evaluate_outer_solution_canonical",
        "prediction_mode": "corr_pred", "raw_rejected": True, "per_algorithm_evaluator_forbidden": True,
    })
    _json(output_dir / "outer_problem_binding.json", {
        "source_hashes": observed, "edge_cpu": {"f_min": 1.5e9, "f_max": 5e9, "F0": 20e9, "default_F_max": 20e9},
        "canonical_variables": ["offload", "cache", "edge_cpu"], "objective_orientation": ["min", "min"],
        "default_eta": 0.90, "scenario_eta": {"S1": .80, "S2": .85, "S3": .90, "S4": .90, "S5": .95},
        "objective_numeric_binding_complete": False,
    })
    _md(output_dir / "outer_problem_audit.md", "外层问题审计", """只读审计使用最新版第四章、第五章、冻结 Markdown、正式 Edge CPU 配置与当前 `src`。决策变量、C1—C7公式、S1—S5 的 η 和 Corrected DualBP 哈希一致；未发现旧服务器部署算法混入。\n\n当前 `src` 原先没有 `optimization` 目录、外层算法、Pareto排序或约束支配实现。`pymoo 0.6.1.6` 可导入 NSGA-II、MOEA/D、RVEA、NSGA-III；无MOPSO。两个目标的结构公式一致，但四个尺度/半径权重仍是UQ-09，因此外层科学问题尚未完整数值化。""")
    _md(output_dir / "objective_definition_binding.md", "目标定义绑定", """两目标均最小化：`Φ_T=atan(T_Σ^c/ρ_T)+λ_T atan(T_Σ^r/ρ_T)`，`Φ_E=atan(E_Σ^c/ρ_E)+λ_E atan(E_Σ^r/ρ_E)`。用户区间逐项相加后恢复系统中心/半径。\n\n本地用户用统一解析求界；卸载用户的核心时延与终端通信能耗使用冻结 Corrected DualBP；缓存未命中时延 `(1-c)T_dep` 同加至核心时延上下界形成完整时延；本地能耗解析计算。优化器接收上述无量纲 `Φ_T,Φ_E` 原值，不为某个算法单独归一化；HV/IGD另用共享全局界归一化。\n\n**阻塞：** `ρ_T,ρ_E,λ_T,λ_E` 尚无正式数值，故当前不能实例化目标。""")
    _md(output_dir / "constraint_definition_binding.md", "C1—C7约束绑定", """- C1 `m_n∈{0,1}`、C2 `c_n∈{0,1}`、C3 `c_n≤m_n`、C4 `Σc_ns_n≤S_max`、C5 `m_nf_min≤f_n≤m_nf_max`、C6 `Σf_n≤F_max`：硬约束，由共同解析检查/修复器处理，不调用BP或GA。\n- C7 `Pos([T_lower,T_upper]≤τ_n)≥η`：可能度约束。`upper≤tau`取1，`lower<tau<upper`取`(tau-lower)/(upper-lower)`，`tau≤lower`取0；退化区间按确定性判断。`CV=Σmax(0,η-Pos_n)`。默认S3 `η=0.90`。\n\nC7使用 `corr_pred edge_total`，不得使用core或Raw。""")

    rows = _algorithm_rows()
    with (output_dir / "algorithm_selection_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _json(output_dir / "outer_algorithm_registry_audit.json", {
        "project_outer_algorithms_before_this_turn": [], "pymoo_version": "0.6.1.6",
        "available": ["NSGA-II", "MOEA/D", "RVEA", "NSGA-III"], "not_available": ["MOPSO"],
        "old_server_algorithms_detected": 0, "formal_registry_frozen": False,
    })
    _json(output_dir / "selected_algorithm_set.json", {
        "status": "not_frozen", "word_target_set": word_algorithms,
        "blocking_member": "CorrectedDualBP-MOPSO", "author_decision_required": True,
        "audited_alternative": "CorrectedDualBP-NSGA-III", "formal_algorithm_count": None,
    })
    _json(output_dir / "canonical_solution_schema.json", {
        "version": "canonical_outer_solution_v1", "fields": [
            {"name": "offload", "symbol": "m", "shape": "[N]", "repaired_domain": "{0,1}"},
            {"name": "cache", "symbol": "c", "shape": "[N]", "repaired_domain": "{0,1}, c<=m"},
            {"name": "edge_cpu", "symbol": "f", "shape": "[N]", "unit": "cycle/s", "repaired_domain": "0 if m=0 else [1.5e9,5e9]"}],
        "decode_then_common_repair": True,
    })
    _md(output_dir / "canonical_repair_contract.md", "共同修复合同", """唯一入口为 `canonical_outer_repair()`。二元量按0.5阈值确定化，随后执行`c←cm`、本地用户`f←0`；缓存超限时按解析回源收益密度保留高值项；最低算力总需求超限时按冻结标称时延改善优先级保留用户；活动用户算力裁剪至`[f_min,f_max]`，总量超限时按高于下限的余量确定性回缩。并列按用户号，正式实现须补决策哈希末级规则。所有算法只能调用同一函数。""")
    _md(output_dir / "repair_behavior_audit.md", "修复行为审计", """静态审计确认共同修复不会读取BP目标或调用GA。协议要求Smoke逐算法记录修复前后m/c Hamming距离、f的L1/L2调整、完全重复率和群体多样性。由于本轮未运行算法，尚无经验重复率；若共同初始池上的重复率或多样性显著塌缩，视为工程问题，不能给某算法更换专用repair。""")

    populations = {name: 100 for name in word_algorithms}
    budgets = {name: 20000 for name in word_algorithms}
    _json(output_dir / "evaluation_budget_protocol.json", {"status": "preregistered_pending_problem_unblock", "primary_budget": "total objective evaluation requests", "cache_does_not_refund_budget": True, "population_target": 100, "formal_budget": 20000, "actual_population_plan": populations, "actual_budget_plan": budgets, "maximum_difference_percent": 0.0, "allowed_maximum_percent": 2.0})
    _md(output_dir / "initialization_fairness_protocol.md", "初始化公平性", """同一场景与outer seed生成一次规范初始候选池；所有算法从该池开始，或将其无损转换为内部表示。初始池先统一decode/repair，哈希写入每次运行记录。若算法机制无法直接接收整池，仍须从同池确定性抽取，且不得调用BP筛选或注入历史最优。""")
    _json(output_dir / "outer_seed_protocol.json", {"namespace": "EXP7_OUTER", "seeds": list(OUTER_SEEDS), "matched_across_algorithms": True, "independent_from_bp_training_seeds": True, "immutable_after_results": True})
    _md(output_dir / "multiobjective_metric_protocol.md", "多目标指标", """Primary为二维最小化HV（越大越好），Secondary为IGD（越小越好）。支持指标固定为可行非支配解数、Feasible Rate、CV、runtime、请求/唯一评价/cache hit；Spacing仅作可选补充。所有指标只用`corr_pred`可行解。""")
    _md(output_dir / "hypervolume_protocol.md", "HV协议", """先把两目标统一为最小化方向，以全部Formal算法×全部种子的可行解池取得每维`best,worst`。共享参考点预注册为`ref_j=worst_j+0.1*(worst_j-best_j)`；若该维零极差，裕量取`0.1*max(|worst_j|,1)`。同一全局界归一化后计算二维HV，规则不得在看到排名后改变。""")
    _md(output_dir / "igd_reference_protocol.md", "IGD参考协议", """合并全部Formal算法×全部种子的`corr_pred`可行非支配解，再执行一次统一非支配筛选，形成pooled empirical reference front。候选前沿与参考前沿共同使用pooled bounds归一化；不得以主算法单独前沿作参考。""")
    _md(output_dir / "normalization_protocol.md", "归一化协议", """原始目标永久保存。优化器仍使用冻结的原始鲁棒目标；仅指标与代表解使用全算法共享的全局min-max界。零极差维归一值固定为0。不得逐算法、逐seed单独拟合指标归一化界。""")
    _md(output_dir / "representative_solution_rule.md", "代表方案规则", """主结论以Pareto集合指标为准。每次运行只从最终`corr_pred`可行Rank-1前沿选择代表解：共享min-max归一化到理想点(0,0)，最小化等权欧氏距离。并列依次选最小QoS裕量更大、最大归一化目标更小、决策哈希字典序更小者。不做五候选或GA认证。""")
    _md(output_dir / "convergence_history_protocol.md", "收敛历史协议", """每个检查点记录generation、total evaluation requests、unique evaluations、HV proxy、feasible rate、非支配解数及目标摘要。正式曲线横轴固定为total objective evaluation requests；generation只作诊断，不能用于跨算法主比较。""")
    _md(output_dir / "runtime_metric_protocol.md", "运行效率协议", """每个algorithm×scenario×seed记录wall time、CPU time（稳定时）、requests、unique evaluations、cache hits、repair calls、invalid candidates和Corrected DualBP calls。workers固定为1；在线时间不混入DualBP训练/校准开销。""")
    _md(output_dir / "cache_protocol.md", "缓存协议", """所有算法统一cache ON。规范解经修复后使用同一量化与哈希规则作为key；总请求每次ask均增加，首次key增加unique evaluation，重复key增加cache hit。公平预算以total objective evaluation requests为准，缓存命中不返还预算，同时完整报告三类计数。""")
    _md(output_dir / "statistical_test_protocol.md", "统计检验协议", """每指标报告mean、sample std、median、IQR和每seed原值。算法数≥3时先做Friedman整体检验；主算法与各baseline按matched scenario/seed做双侧Wilcoxon signed-rank，Holm校正；效应量冻结为rank-biserial correlation。Primary/Secondary均执行，不按显著性挑指标。""")
    _md(output_dir / "smoke_preview_formal_plan.md", "Smoke / Preview / Formal计划", """- Smoke：S3、seed 52001、pop 10、100 requests、全部已确认算法，目标≤5分钟；只验接口/可行解/前沿/history。当前不允许启动。\n- Preview：S3、seeds 52001/52002、pop 50、2000 requests，目标20–40分钟；只修bug、不公平预算或数值异常。\n- Formal：S3 5个冻结seed，S1/S2/S4/S5同5 seed，pop 100、每run 20000 requests、workers=1。第五章建议S3 10次，但本Prompt正式规则冻结5 seeds；该差异记录为开放说明，未经新决定不追加seed。""")
    _json(output_dir / "runtime_estimate.json", {"basis": "protocol budget only; recalibrate from Smoke/Preview measured evaluator throughput", "smoke_minutes": {"target_max": 5}, "preview_minutes": {"range": [20, 40]}, "formal": {"algorithms_planned": 5, "scenarios": 5, "seeds_per_scenario": 5, "runs": 125, "evaluations_per_run": 20000, "total_requests": 2500000, "estimated_hours_range": [4, 12], "conservative_hours": 16}, "scientific_run_executed": False})

    fairness_checks = {
        "corrected_hash_verified": observed["corrected_semantic_hash"] == CORRECTED_DUALBP_HASH,
        "raw_rejected_by_wrapper": True, "single_canonical_schema": True, "single_repair_contract": True,
        "single_evaluator_contract": True, "same_budget_plan": True, "budget_difference_within_2pct": True,
        "matched_initialization_preregistered": True, "five_seeds_frozen": True,
        "objective_numeric_parameters_frozen": False, "all_selected_algorithm_adapters_available": False,
        "no_old_server_algorithm_contamination": True, "workers_one": True,
    }
    fairness_passed = all(fairness_checks.values())
    _json(output_dir / "fairness_audit.json", {"passed": fairness_passed, "checks": fairness_checks, "decision": "failed_blocked_before_smoke"})
    _json(output_dir / "open_questions.json", {"count": len(blockers), "blocking_count": sum(x["blocking"] for x in blockers), "items": blockers})
    summary = {
        "exp7_protocol_decision": "blocked", "exp7_protocol_status": "draft_not_frozen",
        "protocol_hash": protocol_hash, "default_eta": .90, "corrected_dualbp_hash_unchanged": True,
        "C1_C7_structural_binding_complete": True, "objective_numeric_binding_complete": False,
        "formal_algorithm_set_frozen": False, "fairness_audit": "failed",
        "open_questions": len(blockers), "blocking_questions": sum(x["blocking"] for x in blockers),
        "multi_algorithm_smoke_allowed": False, "multi_algorithm_preview_allowed": False,
        "multi_algorithm_formal_allowed": False, "pytest_baseline_before_turn": 307,
        "elapsed_seconds_before_tests": time.monotonic() - started,
    }
    _json(output_dir / "summary.json", summary)
    _md(output_dir / "validation_report.md", "验证报告", f"""- 冻结Word与Corrected DualBP哈希：通过\n- 决策变量、C1—C7结构、默认η=0.90：绑定\n- 目标数值参数：失败（UQ-09）\n- 五算法可执行集合：失败（MOPSO/adapter缺失）\n- 公平性审计：失败\n- 协议决定：`blocked`\n- Smoke/Preview/Formal：均不放行\n- protocol hash：`{protocol_hash}`""")
    _md(output_dir / "README.md", "Exp-7 多算法协议与公平性审计", """本目录只记录只读审计、共同合同和预注册计划，没有运行外层算法。冻结源哈希均保持一致；但目标尺度/权重和MOPSO可执行实现未闭合，因此协议依法阻塞，不能启动Smoke。""")
    (output_dir / "final_pytest.txt").write_text("等待本轮最终完整 pytest。\n", encoding="utf-8")
    created = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file())
    _json(output_dir / "created_files.json", {"count": len(created) + 1, "files": created + ["created_files.json"]})
    return summary


if __name__ == "__main__":
    root = Path.cwd()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = build_protocol(root, root / "outputs/05_audit" / f"exp7_multi_algorithm_protocol_{stamp}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
