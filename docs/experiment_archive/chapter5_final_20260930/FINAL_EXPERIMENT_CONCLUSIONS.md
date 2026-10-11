# 第五章最终实验结论

本文件仅依据本归档中的 `PAPER_KEY_NUMBERS.md`、`PAPER_CLAIM_BOUNDARIES.md` 和 `tables/01`—`11` CSV 整理，不包含新实验或科研指标重算。

## 1. Exp-1：高保真近似标签可靠性

冻结验证覆盖60个样本、240个任务和3360个评价单元。相对 miss gap 的中位数为0、Q95为0.2299%、最大值为1.2974%；有效响应相对标准差的Q95为0.2933%、最大值为0.7037%，整体升级率为0.8333%。这些结果支持在已测试范围内将该近似流程作为后续代理模型的统一标签来源，但不外推为任意场景下的精确高保真求解。

## 2. Exp-2：Improved DualBP 预测精度

冻结正式候选 `seed=31005` 的 raw test WMAPE 为：时延下界1.0328%、时延上界0.5829%、能耗下界0.5224%、能耗上界0.3043%。该结果是单个冻结正式候选的测试精度，不是多 seed 均值。

## 3. Exp-2：Safety Correction

Safety Correction 使时延 ICR 从77.78%提高至97.78%，使能耗 ICR 从75.00%提高至93.33%。这是同一冻结候选修正前后的安全性比较，不等同于网络结构消融，也不得与 raw ablation safety 混写。

## 4. Exp-3：Original vs Improved

在五个训练 seed 的冻结汇总中，Improved 相对 Original 的四个边界 WMAPE 方向均改善；完整 MAE、RMSE、WMAPE 和 R² 见 `tables/04_exp3_original_vs_improved.csv`。这一结果支持完整改进方案的总体预测效果，不单独证明某一内部组件的因果贡献。

## 5. Exp-3：Final Physics-Residual Ablation

Full 的四边界 WMAPE（mean±std）分别为：时延下界1.2592%±0.2205%、时延上界0.7147%±0.1295%、能耗下界0.6716%±0.1385%、能耗上界0.3923%±0.0795%。去除 residual normalization 后，时延误差降低，但能耗上下界 WMAPE 分别达到40.9796%±53.8781%和40.5785%±54.2442%，支持 residual normalization 缓解跨任务尺度失衡并提高稳定性的解释。

Direct-Bounds 的能耗误差低于 Full，但 raw invalid interval rate 为0.0025±0.0023，五个 seed 中三个出现非法区间。Center–Radius 的主要证据是通过结构化参数化保证 `lower≤upper`，不能写成使所有回归误差下降。

Unified 在时延均值上优于 Full、能耗略差，ICR为0.8000±0.0308；其参数量为25350，Full为25668。两者体现任务依赖的精度、训练和推理权衡，不能声称 Dual 必然优于 Unified。

Full 与 Direct-Target 的时延上界比较在五个配对 seed 上方向一致（5/0/0），exact p=0.0625、rank-biserial=1。由于统计单位是 seed 且 `n=5`，只能表述为方向一致，不能表述为统计显著。M1同时移除 physical baseline 与 residual formulation，不能用于证明物理基线的独立贡献。

## 6. Exp-6：四机制 accuracy–safety–efficiency

四种机制分别为 Corrected-DualBP、Raw-DualBP、Online-LHS 和 Online-GA。Corrected-DualBP 的四边界 WMAPE 为0.5320%、0.3055%、0.01575%和0.00221%，其 LER/UER 均为0、ICR为100%；Raw-DualBP 的 ICR为90%，Online-LHS和Online-GA的 ICR均为3%。因此，较低的逐边界误差不自动等同于更可靠的区间覆盖。

Corrected-DualBP 的中位运行时间为0.002162 s，相对 Online-GA 的冻结 wall-clock 加速比为63784.47倍。physical evaluation count 与 wall-clock 是不同口径，不能等同。总体证据支持 Corrected-DualBP 在当前冻结协议下形成更适合在线评价的精度、安全性和效率平衡，但不外推为所有机制和硬件环境下的普遍排序。

## 7. Exp-5：Gamma center/radius

各 Gamma 对应独立重新优化的正式代表方案。Gamma从1变化到5时，`c_T` 从19.1573±0.2847上升到20.8916±0.3532，`r_T` 从4.3694±0.1710扩大到7.6426±0.2230。`c_E` 和 `r_E` 在 Gamma=4 达到较高水平、Gamma=5回落，因此不能强行表述为严格单调。25/25个代表方案均可行，QoS为1，C7违反用户数为0。

## 8. Exp-5：eta、Fmax、Smax 参数敏感性

归一化响应幅度表明，Gamma 对 ΦT 和 ΦE 的响应分别为19.5392%和0.5571%；eta 分别为0.9539%和0.2464%；Fmax分别为2.6787%和0.5891%；Smax分别为0.9552%和0.2420%。在当前冻结 OFAT 范围内，ΦT 对 Gamma 的响应最强；这些结果是局部单因素敏感性证据，不代表全局因果效应。

## 9. Exp-7：HV/IGD 多算法比较

OE-RVEA 的 HV 为0.6810±0.3547、平均秩1.64；IGD为0.3933±0.2610、平均秩1.68。在冻结的六算法正式比较中，这些指标支持 OE-RVEA 具有较好的综合 Pareto 解集质量。预注册的“average rank strictly equals 1”门控未通过，且缺少构造合法统一收敛曲线所需的逐 checkpoint archive；因此 Exp-7 必须标记为 `COMPLETED_WITH_LIMITATION`。该限制不否定已冻结的 HV、IGD、Pareto 和正式统计证据。

## 10. Exp-8：场景稳定性与规模扩展

G1/G2/G3 的 ΦT 分别为0.9263、1.0643和1.1337，ΦE分别为2.8628、3.1158和3.1190；各场景的可行率和QoS率均为1。规模实验覆盖 N=10、20、30、40，平均运行时间分别约为31.37 s、33.80 s、33.92 s和34.64 s，冻结预算下 ΦT 指数从100增至102.85，ΦE指数保持在约100附近。该结果支持已测试场景和规模范围内的稳定性与规模趋势，不能据此外推算法渐近复杂度。

## 11. 最终论文结论边界

- 可以陈述 residual normalization 缓解 Time/Energy 任务尺度失衡并提高跨任务稳定性。
- 可以陈述 Center–Radius 通过结构化参数化保证区间合法性。
- 可以陈述 Full 在时延上界的五个配对 seed 上均优于 Direct-Target，但不能写成统计显著。
- Dual 与 Unified 只能解释为任务依赖的精度、训练和推理权衡。
- 不得声称 Full 所有指标最好、Dual 必然优于 Unified、Center–Radius 使所有误差下降，或物理基线的独立贡献已被单独证明。
- 不得将 Safety Correction 后的 UER/LER/ICR 与 raw ablation safety 混写。
- Exp-5 的非单调变化不表示数据错误；Exp-8 结论仅限已测试 G1/G2/G3 和 N=10/20/30/40。
