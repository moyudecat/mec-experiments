# 第五章论文关键数字 V2

所有数字均来自本目录 CSV 的冻结来源，不从 Prompt 复制。

## 5.2.1 标签可靠性
- 相对 miss gap：中位数 0；Q95 0.2299%；最大值 1.2974%。
- 有效响应相对标准差：Q95 0.2933%；最大值 0.7037%。
- 60 样本、240 任务、3360 评价单元；升级率 0.8333%。

## 5.2.2 Improved DualBP 与 Safety Correction
- Improved raw Test WMAPE：Time lower 1.0328%，Time upper 0.5829%，Energy lower 0.5224%，Energy upper 0.3043%（正式候选 seed=31005）。
- Raw→Corrected：Time ICR 77.78%→97.78%，Energy ICR 75.00%→93.33%；这是安全修正效果，不等同于网络结构消融。

## 5.2.3 Original vs Improved
- 5-seed 四边界 WMAPE：Improved 相对 Original 四个边界方向均改善；完整 MAE/RMSE/WMAPE/R² 在 `04_exp3_original_vs_improved.csv`。

## 5.2.4 最终结构消融
- Full WMAPE：T lower 1.2592%±0.2205%，T upper 0.7147%±0.1295%，E lower 0.6716%±0.1385%，E upper 0.3923%±0.0795%。
- w/o Norm：Time 更低，但 Energy 为 40.9796%±53.8781% 和 40.5785%±54.2442%，呈跨任务方向反转。
- Direct-Bounds：Energy WMAPE 低于 Full，但 raw invalid rate=0.0025±0.0023，3/5 seeds 出现非法区间。
- Unified：Time 均值优于 Full、Energy 略差；ICR=0.8000±0.0308；参数量 25350，Full 为 25668。
- Full vs Direct-Target：T upper 为 5/0/0，exact p=0.0625，rank-biserial=1；只能写“五个配对 seed 方向一致”，不能写统计显著。

## 5.2.5 四机制在线评价
- 取数见 `07_exp6_four_mechanism.csv`；正文强调 accuracy-safety-efficiency 权衡，不把 physical evaluation count 等同于 wall-clock。

## 5.3.2 Gamma center/radius
- Gamma=1→5：c_T 19.1573±0.2847→20.8916±0.3532；r_T 4.3694±0.1710→7.6426±0.2230。
- c_E/r_E 在 Gamma=4 达到较高水平、Gamma=5 回落；不应强行写成严格单调。
- 25/25 代表方案可行，QoS=1，C7 违反用户数为0。

## 5.3.3 参数敏感性
- 归一化响应幅度：Gamma 对 Phi_T 19.5392%、Phi_E 0.5571%；eta 分别0.9539%、0.2464%；Fmax 分别2.6787%、0.5891%；Smax分别0.9552%、0.2420%。

## 5.4 HV/IGD
- OE-RVEA：HV 0.6810±0.3547、平均秩1.64；IGD 0.3933±0.2610、平均秩1.68。

## 5.5 场景与规模
- G1/G2/G3 Phi_T：0.9263/1.0643/1.1337；Phi_E：2.8628/3.1158/3.1190。
- 规模数据覆盖 N=10/20/30/40；正文只解释冻结预算下的趋势，不外推算法复杂度。
