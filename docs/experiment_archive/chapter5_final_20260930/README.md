# 第五章正式实验最终归档

本目录用于保存硕士论文第五章正式实验的最终、可追溯、GitHub 友好的科研快照。归档基准日期为 2026-09-30，最终实验 Gate 为：

`NO_REQUIRED_EXPERIMENT_RERUN_V2`

## 归档原则

- 原始完整实验输出仍保存在项目的 `outputs/` 目录中；本目录不是完整 `outputs` 镜像。
- 本目录只保存论文最终使用的核心数据、结论边界、来源映射和代表性图片。
- 所有数据均复制自既有 Formal/Frozen 资产，本轮未运行实验、未重新评价候选、未重新计算科研指标。
- `SOURCE_MAPPING.csv` 用于追溯各张数据表的正式上游来源及其冻结哈希。
- `PAPER_CLAIM_BOUNDARIES.md` 规定论文可以、需要谨慎以及禁止作出的结论。
- `SHA256SUMS.txt` 记录归档文件哈希；`FIGURE_SOURCE_MAPPING.csv` 记录每张归档图片的原始路径。

## 目录内容

- `FINAL_EXPERIMENT_CONCLUSIONS.md`：第五章最终实验结论摘要。
- `EXPERIMENT_STATUS.md`：Exp-1 至 Exp-8 的最终状态和 Exp-7 限制。
- `PAPER_KEY_NUMBERS.md`：冻结论文关键数字，原样复制。
- `PAPER_CLAIM_BOUNDARIES.md`：冻结论文结论边界，原样复制。
- `SOURCE_MAPPING.csv`：01—11 CSV 的正式来源映射，原样复制。
- `OLD_PC_EXECUTION_NOTE.md`：旧电脑执行环境与迁移边界。
- `tables/`：01—11 CSV 的字节级原样副本。
- `figures/`：最终论文核心图片的 PNG/PDF 副本。

## 核心数据表

| 归档文件 | 论文证据 |
|---|---|
| `tables/01_exp1_label_reliability.csv` | Exp-1 标签可靠性 |
| `tables/02_exp2_dualbp_prediction.csv` | Exp-2 Improved DualBP 正式候选预测精度 |
| `tables/03_exp2_safety_correction.csv` | Exp-2 Safety Correction |
| `tables/04_exp3_original_vs_improved.csv` | Exp-3 Original vs Improved |
| `tables/05_exp3_final_ablation_accuracy.csv` | Final Physics-Residual Ablation 精度 |
| `tables/06_exp3_final_ablation_structure.csv` | Final Physics-Residual Ablation 区间结构 |
| `tables/07_exp6_four_mechanism.csv` | Exp-6 四机制综合比较 |
| `tables/08_exp5_gamma_center_radius.csv` | Exp-5 Gamma center/radius |
| `tables/09_exp5_parameter_sensitivity.csv` | Exp-5 参数敏感性 |
| `tables/10_exp7_hv_igd.csv` | Exp-7 HV/IGD 多算法比较 |
| `tables/11_exp8_scene_scale.csv` | Exp-8 场景稳定性与规模扩展 |

## 最终图片及原始来源

| 归档图片 | 原始来源 |
|---|---|
| `figures/exp3/fig_exp3_final_ablation_wmape.{png,pdf}` | `outputs/08_misc/chapter5_exp3_ablation_sync_v2_20260930_153110/figures/main/{png,pdf}/fig_ablation_four_boundary_wmape.*` |
| `figures/exp6/fig_exp6_four_mechanism_wmape.{png,pdf}` | `outputs/01_formal/exp6_four_mechanism_formal_20260929_122818/figures/{png,pdf}/fig_exp6_boundary_error_wmape.*` |
| `figures/exp6/fig_exp6_interval_safety.{png,pdf}` | `outputs/01_formal/exp6_four_mechanism_formal_20260929_122818/figures/{png,pdf}/fig_exp6_interval_safety.*` |
| `figures/exp5/fig_exp5_gamma_c_t.{png,pdf}` | `outputs/08_misc/robust_center_radius_per_gamma_final_20260929_094513/figures/{png,pdf}/fig_exp5_gamma_c_t.*` |
| `figures/exp5/fig_exp5_gamma_r_t.{png,pdf}` | `outputs/08_misc/robust_center_radius_per_gamma_final_20260929_094513/figures/{png,pdf}/fig_exp5_gamma_r_t.*` |
| `figures/exp5/fig_exp5_gamma_c_e.{png,pdf}` | `outputs/08_misc/robust_center_radius_per_gamma_final_20260929_094513/figures/{png,pdf}/fig_exp5_gamma_c_e.*` |
| `figures/exp5/fig_exp5_gamma_r_e.{png,pdf}` | `outputs/08_misc/robust_center_radius_per_gamma_final_20260929_094513/figures/{png,pdf}/fig_exp5_gamma_r_e.*` |
| `figures/exp7/fig_exp7_pareto_s3.{png,pdf}` | `outputs/01_formal/exp7_multi_algorithm_formal_20260812_154915/final_analysis/figures/fig_exp7_pareto_S3.*` |
| `figures/exp7/fig_exp7_hv.{png,pdf}` | `outputs/01_formal/exp7_multi_algorithm_formal_20260812_154915/final_analysis/figures/fig_exp7_hv_formal.*` |
| `figures/exp7/fig_exp7_igd.{png,pdf}` | `outputs/01_formal/exp7_multi_algorithm_formal_20260812_154915/final_analysis/figures/fig_exp7_igd_formal.*` |
| `figures/exp8/fig_exp8_scenario_stability.{png,pdf}` | `outputs/08_misc/exp8_image_finish_20260922_150305/final_two_figures/{png,pdf}/fig5_8_exp8_scenario_stability.*` |
| `figures/exp8/fig_exp8_scalability.{png,pdf}` | `outputs/08_misc/exp8_image_finish_20260922_150305/final_two_figures/{png,pdf}/fig5_9_exp8_scalability_analysis.*` |

上述图片仅是论文核心证据的长期副本。历史图、Smoke/Probe 图、预览图、checkpoint 和完整运行日志仍保留在 `outputs/`，不纳入本 GitHub 归档。

## 使用约束

引用本目录数字前，应同时检查 `PAPER_KEY_NUMBERS.md`、`PAPER_CLAIM_BOUNDARIES.md` 和 `SOURCE_MAPPING.csv`。运行时间只能解释为原实验硬件环境中的测量值；如果在新电脑重新执行，必须建立新的实验版本，禁止覆盖历史 Formal/Frozen 结果。
