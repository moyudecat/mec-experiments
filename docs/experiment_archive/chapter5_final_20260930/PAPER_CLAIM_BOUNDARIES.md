# 论文结论边界 V2

## SAFE
- Residual normalization 支持“缓解 Time/Energy 任务尺度失衡并提升跨任务稳定性”。
- Center–Radius 支持“通过结构化参数化保证 lower≤upper”。
- Full 在 T upper 的五个配对 seed 均优于 Direct-Target。
- Dual 与 Unified 表现为任务依赖的精度、训练和推理权衡。
- Exp-8 支持已测试 G1/G2/G3 和 N=10/20/30/40 下的稳定性与规模趋势。

## CAUTION
- 消融统计单位为 seed，n=5；exact p=0.0625 不能写成统计显著。
- M1 同时移除 physical baseline 与 residual formulation，不是物理基线单因素消融。
- M3 还伴随 direct-boundary residual scale 的表示变化。
- M4 共享训练、任务掩码和异构输入嵌入是统一网络实现的固有差异。
- Exp-5 各 Gamma 是独立重新优化，非单调不表示数据错误。

## FORBIDDEN
- “Full 所有指标最好”。
- “Dual 必然优于 Unified”。
- “Center–Radius 使所有回归误差下降”。
- “physical baseline 的独立贡献已被证明”。
- “p=0.0625 具有统计显著性”。
- 将 raw UER/LER/ICR 与 Safety Correction 后指标混写。
