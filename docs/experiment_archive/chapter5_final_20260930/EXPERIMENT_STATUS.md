# 第五章实验最终状态

| 实验 | 最终状态 | 说明 |
|---|---|---|
| Exp-1 | COMPLETED | 高保真近似标签可靠性证据已冻结 |
| Exp-2 | COMPLETED | Improved DualBP 正式候选预测精度及 Safety Correction 已冻结 |
| Exp-3 | COMPLETED | Original-vs-Improved 与 Final Physics-Residual Ablation 已冻结 |
| Exp-4 | COMPLETED | 鲁棒性比较正式证据已冻结 |
| Exp-5 | COMPLETED | per-Gamma center/radius 与参数敏感性正式证据已冻结 |
| Exp-6 | COMPLETED | 四种在线区间评价机制正式结果已冻结 |
| Exp-7 | COMPLETED_WITH_LIMITATION | HV、IGD、Pareto 和正式统计有效；见下述限制 |
| Exp-8 | COMPLETED | 场景稳定性与规模扩展结果已冻结 |

## Exp-7 限制

Exp-7 的预注册 performance gate“average rank strictly equals 1”未通过，因此不能声称该严格门控成立。同时，现有冻结历史没有保存按同一全局 bounds 计算累计 HV 所需的逐 checkpoint archive，故不存在可合法补制的统一收敛曲线，也不得通过插值或重算虚构该曲线。

这一限制不否定 Exp-7 已冻结的正式证据：HV、IGD、代表性 Pareto 前沿及其正式统计仍然有效。因此 Exp-7 的状态是 `COMPLETED_WITH_LIMITATION`，而不是失败。

## 最终门控

`NO_REQUIRED_EXPERIMENT_RERUN_V2`

本门控表示第五章没有必须补跑的正式实验；它不授权覆盖任何历史 Formal/Frozen 资产。
